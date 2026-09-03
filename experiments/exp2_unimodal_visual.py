# -*- coding: utf-8 -*-
"""
experiments/exp2_unimodal_visual.py
===================================
Eksperimen 2: Unimodal Visual (Face Crop) dengan MobileNetV3-Small.
Menggunakan static split: training (Actor 1-16), validation (Actor 17-20), testing (Actor 21-24).
Early stopping berdasarkan cross-actor validation (Actor 17-20).

🔧 FIX:
  - Deteksi otomatis dimensi output backbone MobileNetV3-Small (bisa 1024, bukan 576).
  - target_dim disesuaikan dengan output aktual backbone, bukan dari config hardcoded.
  - Menambahkan penyesuaian model jika mismatch terjadi.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RavdessVisual, get_visual_transforms
from models import UnimodalVisualModel
from training import train_unimodal_visual_static
from training.train_unimodal_visual import LabelSmoothingCrossEntropy
from training.utils import compute_class_weights, load_checkpoint, set_seed
from evaluation import evaluate_test_set, save_test_metrics, roofline_analysis
from configs import load_config
import timm
import torch.nn as nn


def get_actual_backbone_dim(backbone_name: str, device: torch.device) -> int:
    """
    Mendapatkan dimensi output aktual backbone dengan menjalankan dummy forward.
    """
    model = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool='avg')
    model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    actual_dim = out.shape[-1]
    print(f"[INFO] Detected actual output dimension for backbone '{backbone_name}': {actual_dim}")
    return actual_dim


def run_exp2_unimodal_visual(config_path=None, override=None):
    config = load_config(config_path or 'configs/unimodal_visual.yaml')
    if override:
        config.update(override)

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device.upper()}")

    set_seed(config['training'].get('manual_seed', 42),
             deterministic=config['training'].get('cudnn_deterministic', True))

    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"  GPU Memory awal: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    annotation_path = config['paths']['annotations']
    visual_root = config['paths']['output_visual']
    result_path = os.path.join(config['paths']['results'], config['training']['result_subdir'])
    os.makedirs(result_path, exist_ok=True)

    n_threads = min(config['training'].get('n_threads', 4), 2)
    batch_size = config['training'].get('batch_size', 16)

    # ============================================================
    # 🔧 DETEKSI DIMENSI OUTPUT BACKBONE SECARA OTOMATIS
    # ============================================================
    backbone_name = config['model']['backbone']
    actual_feature_dim = get_actual_backbone_dim(backbone_name, device)
    # Jika config memiliki feature_dim, kita timpa dengan actual (agar konsisten)
    config['model']['feature_dim'] = actual_feature_dim
    target_dim = actual_feature_dim

    print(f"[INFO] Using feature_dim: {target_dim} (detected automatically)")

    transform_train = get_visual_transforms(is_training=True, image_size=(224, 224))
    transform_val = get_visual_transforms(is_training=False, image_size=(224, 224))
    transform_test = get_visual_transforms(is_training=False, image_size=(224, 224))

    train_dataset = RavdessVisual(
        annotation_path=annotation_path,
        root_dir=visual_root,
        subset='training',
        transform=transform_train,
        num_frames=config['data']['visual']['num_frames'],
        temporal_mode=config['data'].get('temporal_mode', 'stack')
    )
    val_dataset = RavdessVisual(
        annotation_path=annotation_path,
        root_dir=visual_root,
        subset='validation',
        transform=transform_val,
        num_frames=config['data']['visual']['num_frames'],
        temporal_mode=config['data'].get('temporal_mode', 'stack')
    )
    test_dataset = RavdessVisual(
        annotation_path=annotation_path,
        root_dir=visual_root,
        subset='testing',
        transform=transform_test,
        num_frames=config['data']['visual']['num_frames'],
        temporal_mode=config['data'].get('temporal_mode', 'stack')
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        pin_memory=True,
        prefetch_factor=2
    )

    print(f"\nData Summary:")
    print(f"  Train (Actor 1-16)      : {len(train_dataset)} samples")
    print(f"  Val   (Actor 17-20)     : {len(val_dataset)} samples")
    print(f"  Test  (Actor 21-24)     : {len(test_dataset)} samples")
    print(f"  Batch size             : {batch_size}")
    print(f"  Workers                : {n_threads}")

    classifier_hidden = config['model'].get('classifier_hidden', 128)
    dropout = config['model'].get('dropout', 0.5)
    freeze_backbone = config['model'].get('freeze_backbone', False)

    # ============================================================
    # 🔧 BANGUN MODEL DENGAN TARGET_DIM YANG DETEKSI
    # ============================================================
    model = UnimodalVisualModel(
        num_classes=config['dataset']['num_classes'],
        backbone_name=backbone_name,
        dropout=dropout,
        pretrained=config['model']['pretrained'],
        freeze_backbone=freeze_backbone,
        classifier_hidden=classifier_hidden,
        target_dim=target_dim,
        use_temporal_attention=config['model'].get('use_temporal_attention', True),
        use_self_attention=config['model'].get('use_self_attention', False),
        use_se_block=config['model'].get('use_se_block', False)
    ).to(device)

    print(f"\nModel Summary:")
    print(f"  Backbone       : {backbone_name}")
    print(f"  Feature Dim    : {model.get_feature_dim()}")
    print(f"  Classifier Dim : {classifier_hidden}")
    print(f"  Dropout        : {dropout}")
    print(f"  Freeze Backbone: {freeze_backbone}")
    print(f"  Use Temporal Attention: {config['model'].get('use_temporal_attention', True)}")
    print(f"  Use SE-Block   : {config['model'].get('use_se_block', False)}")
    print(f"  Total Params   : {sum(p.numel() for p in model.parameters()):,}")

    use_label_smoothing = config['training'].get('use_label_smoothing', False)
    if use_label_smoothing:
        smoothing = config['training'].get('smoothing', 0.1)
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        print(f"Loss: Label Smoothing Cross Entropy (smoothing={smoothing})")
    else:
        class_weights = compute_class_weights(annotation_path, config['dataset']['num_classes']).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Loss: Cross Entropy (weighted)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler_type = config['training'].get('scheduler_type', 'cosine')
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training'].get('plateau_factor', 0.5),
            patience=config['training'].get('plateau_patience', 5),
            min_lr=1e-7
        )
        print(f"Scheduler: ReduceLROnPlateau (factor={config['training'].get('plateau_factor', 0.5)}, patience={config['training'].get('plateau_patience', 5)})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['cosine_t_max'],
            eta_min=config['training']['cosine_eta_min']
        )
        print(f"Scheduler: CosineAnnealingLR (T_max={config['training']['cosine_t_max']})")

    train_config = {
        'result_path': result_path,
        'store_name': 'visual',
        'n_epochs': config['training']['n_epochs'],
        'early_stop_patience': config['training']['early_stop_patience'],
        'batch_size': batch_size,
        'n_threads': n_threads,
        'manual_seed': config['training'].get('manual_seed', 42),
        'grad_clip': config['training'].get('grad_clip', 1.0),
        'use_amp': config['training'].get('use_amp', True),
        'use_label_smoothing': use_label_smoothing,
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay']
    }

    # ============================================================
    # 🚀 TRAINING
    # ============================================================
    train_result = train_unimodal_visual_static(
        model=model,
        train_dataset=train_dataset,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_config
    )

    # ============================================================
    # 📊 EVALUASI TEST SET
    # ============================================================
    best_model_path = os.path.join(result_path, 'visual_best.pth')
    if os.path.exists(best_model_path):
        # 🔧 Saat evaluasi, pastikan model_eval juga menggunakan target_dim yang sama
        model_eval = UnimodalVisualModel(
            num_classes=config['dataset']['num_classes'],
            backbone_name=backbone_name,
            dropout=dropout,
            classifier_hidden=classifier_hidden,
            target_dim=target_dim,
            use_temporal_attention=config['model'].get('use_temporal_attention', True),
            use_self_attention=config['model'].get('use_self_attention', False),
            use_se_block=config['model'].get('use_se_block', False)
        ).to(device)

        load_checkpoint(best_model_path, model_eval, device=device)

        print("\nEvaluasi Test Set (Actor 21-24):")
        result = evaluate_test_set(
            model=model_eval,
            model_type='visual',
            test_loader=test_loader,
            device=device,
            num_classes=config['dataset']['num_classes']
        )

        test_metrics = result['metrics']
        if result['embeddings'] is not None:
            np.save(os.path.join(result_path, 'test_embeddings.npy'), result['embeddings'])
        save_test_metrics(result_path, test_metrics)

        print(f"\n  Test Accuracy      : {test_metrics['accuracy']*100:.2f}%")
        print(f"  Test F1-Macro      : {test_metrics['f1_macro']:.4f}")

        try:
            roofline_analysis(model_eval, 'Unimodal Visual', device, is_multimodal=False)
        except Exception as e:
            print(f"\nRoofline analysis gagal: {e}")
            print("   (Ini tidak mempengaruhi hasil training dan evaluasi)")

    print(f"\nEksperimen 2 selesai")
    return train_result, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/unimodal_visual.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run_exp2_unimodal_visual(args.config)


if __name__ == '__main__':
    main()