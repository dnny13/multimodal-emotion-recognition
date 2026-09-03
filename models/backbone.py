# -*- coding: utf-8 -*-
"""
models/backbone.py
==================
Factory untuk backbone yang digunakan dalam penelitian.
"""

import torch
import torch.nn as nn
import timm
from .audio_1dcnn import Audio1DCNN
from .audio_2dcnn import Audio2DCNN

BACKBONE_CONFIG = {
    'efficientnetv2_b0': {
        'model_name': 'tf_efficientnetv2_b0',
        'feature_dim': 1280,
        'params_m': 7.1,
        'flops_g': 1.46,
        'description': 'EfficientNetV2-B0 (Lightweight, visual)',
        'domain': 'visual',
    },
    'mobilenetv3_small_100': {
        'model_name': 'mobilenetv3_small_100',
        'feature_dim': None,  # Akan dideteksi otomatis
        'params_m': 2.54,
        'flops_g': 0.06,
        'description': 'MobileNetV3-Small (Ultra Lightweight, visual)',
        'domain': 'visual',
    },
}

AUDIO_BACKBONE_CONFIG = {
    'audio_1dcnn': {
        'in_channels': 120,
        'feature_dim': 1280,
        'params_m': 1.2,
        'flops_g': 0.11,
        'description': '1D-CNN (MFCC+Delta) — SE-block opsional sebelum pooling',
        'domain': 'audio',
    },
    'audio_2dcnn': {
        'in_channels': 1,
        'feature_dim': 1280,
        'params_m': 0.8,
        'flops_g': 0.05,
        'description': '2D-CNN (Mel-Spectrogram) — pembanding arsitektural terhadap visual 2D-CNN',
        'domain': 'audio',
    },
}


def create_backbone(backbone_name: str, pretrained: bool = True, freeze: bool = False, use_se_block: bool = False):
    """Membuat backbone dengan deteksi otomatis dimensi output untuk MobileNetV3-Small."""
    if backbone_name == 'audio_1dcnn':
        config = AUDIO_BACKBONE_CONFIG[backbone_name]
        model = Audio1DCNN(
            in_channels=config['in_channels'],
            feature_dim=config['feature_dim'],
            use_se_block=use_se_block
        )
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        return model, config['feature_dim']

    if backbone_name == 'audio_2dcnn':
        config = AUDIO_BACKBONE_CONFIG[backbone_name]
        model = Audio2DCNN(
            in_channels=config['in_channels'],
            feature_dim=config['feature_dim'],
            use_se_block=use_se_block
        )
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        return model, config['feature_dim']

    if backbone_name not in BACKBONE_CONFIG:
        raise ValueError(
            f"Backbone '{backbone_name}' tidak dikenali. "
            f"Pilih dari: {list(BACKBONE_CONFIG.keys()) + list(AUDIO_BACKBONE_CONFIG.keys())}"
        )

    config = BACKBONE_CONFIG[backbone_name]
    model = timm.create_model(
        config['model_name'],
        pretrained=pretrained,
        num_classes=0,
        global_pool='avg'
    )

    # 🔧 DETEKSI OTOMATIS DIMENSI OUTPUT UNTUK MOBILENETV3-SMALL
    if backbone_name == 'mobilenetv3_small_100':
        try:
            dummy = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = model(dummy)
            detected_dim = out.shape[-1]
            # Update config agar sesuai dengan backbone aktual
            config['feature_dim'] = detected_dim
            print(f"[INFO] MobileNetV3-Small detected output dimension: {detected_dim}")
        except Exception as e:
            print(f"[WARNING] Gagal mendeteksi dimensi output MobileNetV3-Small: {e}. Gunakan default 576.")
            config['feature_dim'] = 576
    else:
        # Untuk backbone lain, gunakan nilai dari config
        pass

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model, config['feature_dim']


def get_backbone_info(backbone_name: str):
    if backbone_name in AUDIO_BACKBONE_CONFIG:
        return AUDIO_BACKBONE_CONFIG[backbone_name]
    if backbone_name in BACKBONE_CONFIG:
        return BACKBONE_CONFIG[backbone_name]
    raise ValueError(f"Backbone '{backbone_name}' tidak dikenali.")


def is_audio_backbone(backbone_name: str) -> bool:
    return backbone_name in AUDIO_BACKBONE_CONFIG