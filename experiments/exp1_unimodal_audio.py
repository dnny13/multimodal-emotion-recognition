# -*- coding: utf-8 -*-
"""
experiments/exp1_unimodal_audio.py
===================================
Eksperimen 1: Unimodal Audio.
Mendukung DUA backbone via config:
  - 'audio_1dcnn' (MFCC+Delta, 1D)   → default, metode utama
  - 'audio_2dcnn' (Mel-Spectrogram, 2D) → eksperimen pembanding

Menggunakan static split: training (Actor 1-16), validation (Actor 17-20), testing (Actor 21-24).
Early stopping berdasarkan cross-actor validation (Actor 17-20).
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RavdessAudio
from data.transforms import get_audio_transforms_1dcnn, get_audio_transforms_2dcnn
from models import UnimodalAudioModel
from training import train_unimodal_audio_static
from training.train_unimodal_audio import LabelSmoothingCrossEntropy
from training.utils import compute_class_weights, load_checkpoint, set_seed
from evaluation import evaluate_test_set, save_test_metrics, roofline_analysis
from configs import load_config


def run_exp1_unimodal_audio(config_path=None, override=None):
    config = load_config(config_path or 'configs/unimodal_audio.yaml')
    if override:
        config.update(override)

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device.upper()}")

    set_seed(config['training'].get('manual_seed', 42),
             deterministic=config['training'].get('cudnn_deterministic', True))

    annotation_path = config['paths']['annotations']
    audio_root = config['paths']['output_audio']
    result_path = os.path.join(config['paths']['results'], config['training']['result_subdir'])
    os.makedirs(result_path, exist_ok=True)

    backbone_name = config['model']['backbone']
    use_se_block = config['model'].get('use_se_block', False)
    use_spec_augment = config['data']['augmentation']['spec_augment']['enabled']
    input_mode = config['data'].get('input_mode', '1d')
    classifier_hidden = config['model'].get('classifier_hidden', 512)

    if input_mode == '2d':
        transform_train = get_audio_transforms_2dcnn(is_training=True, use_spec_augment=use_spec_augment)
        transform_val = get_audio_transforms_2dcnn(is_training=False, use_spec_augment=False)
        transform_test = get_audio_transforms_2dcnn(is_training=False, use_spec_augment=False)
    else:
        transform_train = get_audio_transforms_1dcnn(is_training=True, use_spec_augment=use_spec_augment)
        transform_val = get_audio_transforms_1dcnn(is_training=False, use_spec_augment=False)
        transform_test = get_audio_transforms_1dcnn(is_training=False, use_spec_augment=False)

    train_dataset = RavdessAudio(
        annotation_path, audio_root, 'training',
        transform=transform_train,
        backbone_name=backbone_name,
        input_mode=input_mode
    )
    val_dataset = RavdessAudio(
        annotation_path, audio_root, 'validation',
        transform=transform_val,
        backbone_name=backbone_name,
        input_mode=input_mode
    )
    test_dataset = RavdessAudio(
        annotation_path, audio_root, 'testing',
        transform=transform_test,
        backbone_name=backbone_name,
        input_mode=input_mode
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['n_threads'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['n_threads'],
        pin_memory=True
    )

    print(f"\nData Summary:")
    print(f"  Backbone                : {backbone_name} (input_mode={input_mode})")
    print(f"  Train (Actor 1-16)      : {len(train_dataset)} samples")
    print(f"  Val   (Actor 17-20)     : {len(val_dataset)} samples")
    print(f"  Test  (Actor 21-24)     : {len(test_dataset)} samples")

    model = UnimodalAudioModel(
        num_classes=config['dataset']['num_classes'],
        backbone_name=backbone_name,
        dropout=config['model']['dropout'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone'],
        classifier_hidden=classifier_hidden,
        use_se_block=use_se_block
    ).to(device)

    print(f"\nModel Summary:")
    print(f"  Backbone       : {backbone_name}")
    print(f"  Use SE-Block   : {use_se_block}")
    print(f"  Feature Dim    : {model.get_feature_dim()}")
    print(f"  Classifier Dim : {classifier_hidden}")
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
            optimizer, mode='min',
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
        'store_name': 'audio',
        'n_epochs': config['training']['n_epochs'],
        'early_stop_patience': config['training']['early_stop_patience'],
        'batch_size': config['training']['batch_size'],
        'n_threads': config['training']['n_threads'],
        'manual_seed': config['training'].get('manual_seed', 42),
        'grad_clip': config['training'].get('grad_clip', 1.0),
        'use_amp': config['training'].get('use_amp', True)
    }

    train_result = train_unimodal_audio_static(
        model=model,
        train_dataset=train_dataset,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_config
    )

    best_model_path = os.path.join(result_path, 'audio_best.pth')
    if os.path.exists(best_model_path):
        model = UnimodalAudioModel(
            num_classes=config['dataset']['num_classes'],
            backbone_name=backbone_name,
            dropout=config['model']['dropout'],
            classifier_hidden=classifier_hidden,
            use_se_block=use_se_block
        ).to(device)
        load_checkpoint(best_model_path, model, device=device)

        print("\nEvaluasi Test Set (Actor 21-24):")
        result = evaluate_test_set(
            model=model,
            model_type='audio',
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
        roofline_analysis(model, f'Unimodal Audio ({backbone_name})', device, is_multimodal=False)
    except Exception as e:
        print(f"\nRoofline analysis gagal: {e}")
        print("   (Ini tidak mempengaruhi hasil training dan evaluasi)")

    print(f"\nEksperimen 1 ({backbone_name}) selesai")
    return train_result, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/unimodal_audio.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run_exp1_unimodal_audio(args.config)


if __name__ == '__main__':
    main()