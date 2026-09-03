# -*- coding: utf-8 -*-
"""
models/audio_2dcnn.py
======================
Backbone 2D-CNN ringan untuk audio, menerima Mel-Spectrogram sebagai citra
1-channel (grayscale). Terinspirasi dari desain ringan seperti ArabEmoNet
(2D-CNN pada Mel-spectrogram, ~1M parameter).

Tujuan:
  - Pembanding arsitektural terhadap Audio1DCNN (menjawab kritik "1D vs 2D
    tidak apple-to-apple" dibanding EfficientNetV2-B0 pada visual).
  - Tetap sangat ringan (< 1.5M parameter), aman untuk constraint lightweight.
"""

import torch
import torch.nn as nn


class SEBlock2D(nn.Module):
    """SE-block untuk feature map 2D (B, C, H, W)."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.squeeze(x).view(b, c)
        s = self.excite(s).view(b, c, 1, 1)
        return x * s


class Audio2DCNN(nn.Module):
    """
    Input : (batch, 1, n_mels, target_time) — Mel-spectrogram sebagai citra 1-channel
    Output: (batch, feature_dim) — feature vector, default 1280 (disamakan dengan visual)
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 1280, use_se_block: bool = False, **kwargs):
        super().__init__()
        self.use_se_block = use_se_block

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # /2
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # /4
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # /8
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # /16
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        if use_se_block:
            self.se_block = SEBlock2D(channels=256, reduction=8)
        else:
            self.se_block = None

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Linear(256, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, n_mels, target_time), mis. (B, 1, 128, 224)
        Returns:
            (batch, feature_dim)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)          # (B, 256, H/16, W/16)

        if self.use_se_block and self.se_block is not None:
            x = self.se_block(x)

        x = self.global_pool(x)     # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)   # (B, 256)

        x = self.projector(x)       # (B, feature_dim)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)