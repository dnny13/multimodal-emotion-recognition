# -*- coding: utf-8 -*-
"""
models/multimodal_fusion.py
===========================
Model multimodal lengkap dengan intermediate fusion.
Mendukung 3 metode fusion: concatenation, GMU, cross-attention.
Backbone dapat dipilih secara terpisah untuk audio dan visual.

🔧 FIX:
  - forward() dan extract_fused_features() SEBELUMNYA memanggil
    self.visual_backbone(visual_input) LANGSUNG ke backbone mentah,
    yang HANYA menerima tensor 4D (B, C, H, W). Padahal data visual asli
    berbentuk 5D (B, T, C, H, W) karena setiap sampel terdiri dari
    beberapa frame (num_frames, lihat configs/multimodal.yaml).
    Akibatnya, forward() error "Expected 3D/4D input to conv2d" setiap
    kali benar-benar dipanggil dengan raw visual input (mis. saat
    roofline_analysis/FLOPs profiling di evaluation/efficiency.py) —
    meskipun training tidak pernah menyentuh bug ini karena training
    selalu memakai embedding hasil extract_features() (lihat exp3/4/5).
  - Menambahkan method _extract_visual_features() yang mereplikasi
    reshape (B,T,C,H,W)->(B*T,C,H,W) + temporal mean-pooling, PERSIS
    sama seperti models/unimodal_visual.py:UnimodalVisualModel.extract_features(),
    supaya forward() bisa menerima raw visual 5D dengan benar dan hasilnya
    konsisten dengan backbone unimodal yang dipakai untuk ekstraksi
    embedding di seluruh eksperimen.
  - Audio tidak butuh reshape T-dim (Audio1DCNN menerima (B, 120, 224)
    langsung), jadi self.audio_backbone(audio_input) tetap seperti semula.
"""

import torch
import torch.nn as nn
from .backbone import create_backbone
from .fusion_modules import (
    ConcatenationFusion,
    GatedMultimodalUnit,
    CrossAttentionFusion,
)


class MultimodalFusionModel(nn.Module):
    """
    Model multimodal dengan intermediate fusion.

    Args:
        num_classes (int): Jumlah kelas emosi (default: 8)
        fusion_type (str): 'concat', 'gmu', 'attention' (default: 'concat')
        audio_backbone (str): Nama backbone untuk audio (default: 'audio_1dcnn')
        visual_backbone (str): Nama backbone untuk visual (default: 'efficientnetv2_b0')
        feature_dim (int): Dimensi fitur target (default: 1280)
        fusion_dim (int): Dimensi fusion hidden (default: 256)
        classifier_hidden (int): Dimensi hidden classifier (default: 128)
        dropout (float): Dropout rate (default: 0.5)
        pretrained (bool): Gunakan pretrained weights (default: True)
        freeze_backbone (bool): Freeze backbone (default: False)
        use_se_block (bool): Aktifkan SE-block pada kedua backbone (default: False)
    """
    def __init__(
        self,
        num_classes: int = 8,
        fusion_type: str = 'concat',
        audio_backbone: str = 'audio_1dcnn',
        visual_backbone: str = 'efficientnetv2_b0',
        feature_dim: int = 1280,
        fusion_dim: int = 256,
        classifier_hidden: int = 128,
        dropout: float = 0.5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_se_block: bool = False
    ):
        super().__init__()

        self.fusion_type = fusion_type
        self.feature_dim = feature_dim

        # ============================================================
        # 1. Dua backbone terpisah
        # ============================================================
        self.audio_backbone, audio_orig_dim = create_backbone(
            audio_backbone, pretrained=pretrained, freeze=freeze_backbone, use_se_block=use_se_block
        )
        self.visual_backbone, visual_orig_dim = create_backbone(
            visual_backbone, pretrained=pretrained, freeze=freeze_backbone, use_se_block=use_se_block
        )

        if audio_orig_dim != feature_dim:
            self.audio_projector = nn.Linear(audio_orig_dim, feature_dim)
        else:
            self.audio_projector = nn.Identity()

        if visual_orig_dim != feature_dim:
            self.visual_projector = nn.Linear(visual_orig_dim, feature_dim)
        else:
            self.visual_projector = nn.Identity()

        # ============================================================
        # 2. Fusion module
        # ============================================================
        if fusion_type == 'concat':
            self.fusion = ConcatenationFusion(feature_dim, fusion_dim)
            self.fused_dim = fusion_dim
        elif fusion_type == 'gmu':
            self.fusion = GatedMultimodalUnit(feature_dim, fusion_dim)
            self.fused_dim = fusion_dim
        elif fusion_type == 'attention':
            self.fusion = CrossAttentionFusion(feature_dim)
            self.fused_dim = feature_dim
        else:
            raise ValueError(f"fusion_type harus 'concat', 'gmu', atau 'attention'")

        # ============================================================
        # 3. Classifier head
        # ============================================================
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for proj in [self.audio_projector, self.visual_projector]:
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

        if self.fusion_type == 'gmu':
            for m in self.fusion.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _extract_visual_features(self, visual_input: torch.Tensor) -> torch.Tensor:
        """
        🆕 Menangani visual_input yang bisa berbentuk:
          - (B, C, H, W)     : 1 frame per sampel
          - (B, T, C, H, W)  : T frame per sampel (kasus normal, num_frames>1)

        Melakukan reshape (B,T,C,H,W) -> (B*T,C,H,W) sebelum masuk backbone
        (backbone mentah dari timm hanya menerima 4D), lalu temporal
        mean-pooling kembali ke (B, original_dim) setelahnya.

        Ini WAJIB ada supaya forward()/extract_fused_features() konsisten
        dengan cara UnimodalVisualModel.extract_features() memproses data
        yang sama — tanpa ini, backbone menerima tensor 5D dan conv2d error.
        """
        if visual_input.dim() == 4:
            # Sudah (B, C, H, W) — anggap 1 frame, langsung proses
            return self.visual_backbone(visual_input)

        if visual_input.dim() == 5:
            B, T, C, H, W = visual_input.shape
            visual_input = visual_input.view(B * T, C, H, W)
            features = self.visual_backbone(visual_input)          # (B*T, orig_dim)
            features = features.view(B, T, -1)                     # (B, T, orig_dim)
            features = features.mean(dim=1)                        # (B, orig_dim) — temporal pooling
            return features

        raise ValueError(
            f"visual_input harus 4D (B,C,H,W) atau 5D (B,T,C,H,W), "
            f"tapi menerima shape {tuple(visual_input.shape)}"
        )

    def forward(self, audio_input: torch.Tensor, visual_input: torch.Tensor) -> torch.Tensor:
        audio_feat = self.audio_backbone(audio_input)
        visual_feat = self._extract_visual_features(visual_input)   # 🔧 FIX: handle 5D

        audio_feat = self.audio_projector(audio_feat)
        visual_feat = self.visual_projector(visual_feat)

        fused = self.fusion(audio_feat, visual_feat)

        return self.classifier(fused)

    def extract_fused_features(self, audio_input: torch.Tensor, visual_input: torch.Tensor) -> torch.Tensor:
        audio_feat = self.audio_backbone(audio_input)
        visual_feat = self._extract_visual_features(visual_input)   # 🔧 FIX: handle 5D
        audio_feat = self.audio_projector(audio_feat)
        visual_feat = self.visual_projector(visual_feat)
        return self.fusion(audio_feat, visual_feat)

    def extract_fused_features_with_gate(self, audio_feat: torch.Tensor, visual_feat: torch.Tensor):
        assert self.fusion_type == 'gmu', "return_gate hanya berlaku untuk fusion_type='gmu'"

        audio_feat = self.audio_projector(audio_feat)
        visual_feat = self.visual_projector(visual_feat)

        return self.fusion(audio_feat, visual_feat, return_gate=True)

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def get_fused_dim(self) -> int:
        return self.fused_dim