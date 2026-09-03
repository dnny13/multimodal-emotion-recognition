# -*- coding: utf-8 -*-
"""
models/unimodal_visual.py
=========================
Unimodal visual (facial expression) emotion recognition dengan Temporal
Attention opsional, Self-Attention opsional, dan SE-block opsional.

🆕 FIX (v3):
  - Menambahkan SEBlockVector — varian SE-block untuk feature vector 1D
    (bukan feature map 1D seperti di audio), diterapkan per-frame SETELAH
    backbone+projector, SEBELUM temporal pooling/attention.
"""

import torch
import torch.nn as nn
from .backbone import create_backbone
from .fusion_modules import LightweightSelfAttention


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim: int = 1280, hidden_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)              # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        weighted = (x * weights).sum(dim=1)      # (B, D)
        return weighted


class SEBlockVector(nn.Module):
    """
    SE-block untuk feature vector (B, D) — beroperasi per-frame sebelum
    temporal pooling, tetap valid karena reweighting channel di setiap
    frame individual (bukan setelah agregasi seperti self-attn lama di audio).
    """
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        reduced = max(dim // reduction, 8)
        self.excite = nn.Sequential(
            nn.Linear(dim, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) — sudah global-pooled dari backbone per frame
        s = self.excite(x)
        return x * s


class UnimodalVisualModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        backbone_name: str = 'efficientnetv2_b0',
        dropout: float = 0.5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        classifier_hidden: int = 512,
        target_dim: int = 1280,
        use_temporal_attention: bool = False,
        use_self_attention: bool = False,
        use_se_block: bool = False   # 🆕
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.target_dim = target_dim
        self.use_temporal_attention = use_temporal_attention
        self.use_self_attention = use_self_attention
        self.use_se_block = use_se_block

        self.backbone, self.original_dim = create_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone
        )

        if self.original_dim != target_dim:
            self.projector = nn.Linear(self.original_dim, target_dim)
        else:
            self.projector = nn.Identity()

        # 🆕 SE-block per-frame, sebelum temporal pooling
        if use_se_block:
            self.se_block = SEBlockVector(dim=target_dim, reduction=16)
        else:
            self.se_block = None

        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(feature_dim=target_dim, hidden_dim=64)
        else:
            self.temporal_attention = None

        if use_self_attention:
            self.self_attn = LightweightSelfAttention(d_model=target_dim, nhead=4, dropout=0.1)
        else:
            self.self_attn = None

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(target_dim, classifier_hidden),
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
        if isinstance(self.projector, nn.Linear):
            nn.init.xavier_uniform_(self.projector.weight)
            if self.projector.bias is not None:
                nn.init.zeros_(self.projector.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) atau (B, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, C, H, W)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.backbone(x)               # (B*T, original_dim)
        features = self.projector(features)       # (B*T, target_dim)

        # 🆕 SE-block per-frame (masih dalam bentuk B*T, sebelum reshape ke sequence)
        if self.use_se_block and self.se_block is not None:
            features = self.se_block(features)

        features = features.view(B, T, -1)        # (B, T, target_dim)

        if self.use_self_attention and self.self_attn is not None:
            features = self.self_attn(features)   # (B, T, D) -> (B, T, D)

        if self.use_temporal_attention and self.temporal_attention is not None:
            features = self.temporal_attention(features)  # (B, D)
        else:
            features = features.mean(dim=1)               # (B, D)

        return features

    def get_feature_dim(self) -> int:
        return self.target_dim

    def get_original_feature_dim(self) -> int:
        return self.original_dim
