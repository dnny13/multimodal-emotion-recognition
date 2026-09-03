# -*- coding: utf-8 -*-
"""
models/audio_1dcnn.py
=====================
Backbone 1D-CNN untuk audio dengan MFCC+Delta (120 channel).

🔧 FIX (v5): Tambahkan konversi tipe float32 di forward.
"""

import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        s = self.squeeze(x).view(b, c)
        s = self.excite(s).view(b, c, 1)
        return x * s


class Audio1DCNN(nn.Module):
    def __init__(self, in_channels=120, feature_dim=1280, use_se_block: bool = False, **kwargs):
        super().__init__()
        self.use_se_block = use_se_block

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        if use_se_block:
            self.se_block = SEBlock1D(channels=256, reduction=8)
        else:
            self.se_block = None

        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Linear(256, feature_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, 120, 224) — MFCC + Delta + Delta-Delta
        Returns:
            (batch, feature_dim) — feature vector 1280 dimensi
        """
        # 🔧 FIX: Pastikan input bertipe float32
        if x.dtype != torch.float32:
            x = x.float()

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        if self.use_se_block and self.se_block is not None:
            x = self.se_block(x)

        x = self.final_pool(x)
        x = x.squeeze(-1)
        x = self.projector(x)
        return x

    def extract_features(self, x):
        """Alias untuk forward()."""
        return self.forward(x)