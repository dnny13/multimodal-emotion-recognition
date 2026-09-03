# -*- coding: utf-8 -*-
"""
models/fusion_modules.py
========================
Modul intermediate fusion untuk multimodal emotion recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatenationFusion(nn.Module):
    def __init__(self, feature_dim: int = 1280, fusion_dim: int = 256):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, audio_feat: torch.Tensor, visual_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([audio_feat, visual_feat], dim=1)
        return self.fusion(fused)


class GatedMultimodalUnit(nn.Module):
    def __init__(self, feature_dim: int = 1280, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.proj_audio = nn.Linear(feature_dim, hidden_dim)
        self.proj_visual = nn.Linear(feature_dim, hidden_dim)
        self.gate = nn.Linear(feature_dim * 2, hidden_dim)

        # FIX: Turunkan dropout agar gate lebih ekspresif
        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_audio.weight)
        nn.init.zeros_(self.proj_audio.bias)
        nn.init.xavier_uniform_(self.proj_visual.weight)
        nn.init.zeros_(self.proj_visual.bias)
        # FIX: Naikkan gain dari 0.5 ke 1.0 agar gate lebih ekspresif
        nn.init.xavier_uniform_(self.gate.weight, gain=1.0)
        nn.init.zeros_(self.gate.bias)

    def forward(self, audio_feat: torch.Tensor, visual_feat: torch.Tensor, return_gate: bool = False):
        h_audio = self.tanh(self.proj_audio(audio_feat))
        h_visual = self.tanh(self.proj_visual(visual_feat))

        concat = torch.cat([audio_feat, visual_feat], dim=1)
        z = self.sigmoid(self.gate(concat))

        h = z * h_audio + (1 - z) * h_visual
        h = self.dropout(h)

        if return_gate:
            return h, z
        return h


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim: int = 1280, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim harus divisible oleh num_heads"
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.feature_dim = feature_dim

        self.q_audio = nn.Linear(feature_dim, feature_dim)
        self.k_audio = nn.Linear(feature_dim, feature_dim)
        self.v_audio = nn.Linear(feature_dim, feature_dim)

        self.q_visual = nn.Linear(feature_dim, feature_dim)
        self.k_visual = nn.Linear(feature_dim, feature_dim)
        self.v_visual = nn.Linear(feature_dim, feature_dim)

        self.out_audio = nn.Linear(feature_dim, feature_dim)
        self.out_visual = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, audio_feat: torch.Tensor, visual_feat: torch.Tensor) -> torch.Tensor:
        B = audio_feat.shape[0]

        Q_audio = self.q_audio(audio_feat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K_visual = self.k_visual(visual_feat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V_visual = self.v_visual(visual_feat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_audio = F.scaled_dot_product_attention(Q_audio, K_visual, V_visual, dropout_p=0.0)
        attn_audio = attn_audio.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        audio_out = self.out_audio(attn_audio.squeeze(1))

        Q_visual = self.q_visual(visual_feat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K_audio = self.k_audio(audio_feat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V_audio = self.v_audio(audio_feat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_visual = F.scaled_dot_product_attention(Q_visual, K_audio, V_audio, dropout_p=0.0)
        attn_visual = attn_visual.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        visual_out = self.out_visual(attn_visual.squeeze(1))

        fused = self.layer_norm(audio_out + visual_out)
        return fused


class GeometricMeanFusion(nn.Module):
    def forward(self, probs_audio: torch.Tensor, probs_visual: torch.Tensor) -> torch.Tensor:
        probs_audio = torch.clamp(probs_audio, min=1e-8)
        probs_visual = torch.clamp(probs_visual, min=1e-8)
        fused = torch.sqrt(probs_audio * probs_visual)
        return fused / fused.sum(dim=-1, keepdim=True)


class LightweightSelfAttention(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        out = self.norm(x + attn_out)
        if out.size(1) == 1:
            out = out.squeeze(1)
        return out


def get_fusion_module(fusion_type: str, feature_dim: int = 1280, fusion_dim: int = 384, dropout: float = 0.5, num_heads: int = 4) -> nn.Module:
    if fusion_type == 'concat':
        return ConcatenationFusion(feature_dim=feature_dim, fusion_dim=fusion_dim)
    elif fusion_type == 'gmu':
        return GatedMultimodalUnit(feature_dim=feature_dim, hidden_dim=fusion_dim)
    elif fusion_type == 'attention':
        return CrossAttentionFusion(feature_dim=feature_dim, num_heads=num_heads, dropout=dropout)
    else:
        raise ValueError(f"fusion_type tidak dikenali: {fusion_type}. Pilihan: 'concat', 'gmu', 'attention'")