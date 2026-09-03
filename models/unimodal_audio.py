# -*- coding: utf-8 -*-
"""
models/unimodal_audio.py
========================
Unimodal audio emotion recognition.
Default backbone: audio_1dcnn (domain-matched).
"""

import torch
import torch.nn as nn
from .backbone import create_backbone, is_audio_backbone


class UnimodalAudioModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        backbone_name: str = 'audio_1dcnn',
        dropout: float = 0.5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        classifier_hidden: int = 512,
        target_dim: int = 1280,
        use_se_block: bool = False
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.target_dim = target_dim
        self.is_1d = is_audio_backbone(backbone_name)

        self.backbone, self.original_dim = create_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
            use_se_block=use_se_block
        )

        if self.original_dim != target_dim:
            self.projector = nn.Linear(self.original_dim, target_dim)
        else:
            self.projector = nn.Identity()

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
        # 🔧 FIX: Pastikan input bertipe float32
        if x.dtype != torch.float32:
            x = x.float()
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # 🔧 FIX: Pastikan input bertipe float32
        if x.dtype != torch.float32:
            x = x.float()
        features = self.backbone(x)
        features = self.projector(features)
        return features

    def get_feature_dim(self) -> int:
        return self.target_dim

    def get_original_feature_dim(self) -> int:
        return self.original_dim