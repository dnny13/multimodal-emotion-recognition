# -*- coding: utf-8 -*-
"""
experiments/__init__.py
=======================
Package untuk semua skrip eksperimen.
"""

from .exp1_unimodal_audio import run_exp1_unimodal_audio
from .exp2_unimodal_visual import run_exp2_unimodal_visual
from .exp3_multimodal_concat import run_exp3_multimodal_concat
from .exp4_multimodal_gmu import run_exp4_multimodal_gmu
from .exp5_multimodal_attention import run_exp5_multimodal_attention
from .exp7_fusion_comparison import run_exp7_fusion_comparison
from .exp9_ablation_study import run_exp9_ablation_study

__all__ = [
    'run_exp1_unimodal_audio',
    'run_exp2_unimodal_visual',
    'run_exp3_multimodal_concat',
    'run_exp4_multimodal_gmu',
    'run_exp5_multimodal_attention',
    'run_exp7_fusion_comparison',
    'run_exp9_ablation_study',
]
