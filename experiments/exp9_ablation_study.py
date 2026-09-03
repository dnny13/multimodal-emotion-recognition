# -*- coding: utf-8 -*-
"""
experiments/exp9_ablation_study.py
===================================
Ablation Study untuk model Multimodal GMU (metode utama).

🔧 FIX (v12):
  - Baca feature_dim dari config (1024) dan gunakan sebagai target_dim untuk visual_model.
  - Baca visual_model_path dari config, fallback ke results/unimodal_visual_mobilenet/visual_best.pth.
  - Gunakan feature_dim saat membuat MultimodalFusionModel (bukan hardcode 1280).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RavdessMultimodal, get_multimodal_transforms
from models import UnimodalAudioModel, UnimodalVisualModel, MultimodalFusionModel
from training.utils import load_checkpoint, compute_class_weights
from training.train_multimodal import train_multimodal_fold
from training.split_utils import create_internal_split_from_dataset
from configs import load_config


# ============================================================
# Label Smoothing Loss (Self-Contained)
# ============================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        smooth_target = torch.zeros_like(pred).scatter_(
            1, target.unsqueeze(1), 1 - self.smoothing
        )
        smooth_target += self.smoothing / n_classes
        log_pred = F.log_softmax(pred, dim=-1)
        return - (smooth_target * log_pred).sum(dim=-1).mean()


# ============================================================
# Fungsi Ekstraksi Embedding
# ============================================================
def extract_embeddings(audio_model, visual_model, loader, device):
    audio_embeds, visual_embeds, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            audio, visual, y = batch
            audio = audio.to(device)
            visual = visual.to(device)
            a_feat = audio_model.extract_features(audio).cpu().numpy()
            v_feat = visual_model.extract_features(visual).cpu().numpy()
            audio_embeds.append(a_feat)
            visual_embeds.append(v_feat)
            labels.append(y.numpy())
    return {
        'audio': np.concatenate(audio_embeds, axis=0),
        'visual': np.concatenate(visual_embeds, axis=0),
        'labels': np.concatenate(labels, axis=0),
    }


def run_exp9_ablation_study(config_path=None, override=None):
    config = load_config(config_path or 'configs/multimodal.yaml')
    if override:
        config.update(override)

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device.upper()}")

    ablation_cfg = config.get('ablation', {})
    if not ablation_cfg.get('enabled', True):
        print("[INFO] Ablation study dinonaktifkan di config (ablation.enabled=false). Keluar.")
        return None

    variants = ablation_cfg.get('variants', [])
    if not variants:
        raise ValueError("Tidak ada varian ablasi di configs/multimodal.yaml -> ablation.variants")

    annotation_path = config['paths']['annotations']
    audio_root = config['paths']['output_audio']
    visual_root = config['paths']['output_visual']
    result_base = os.path.join(config['paths']['results'], 'ablation')
    os.makedirs(result_base, exist_ok=True)

    # ============================================================
    # 🔧 Baca feature_dim dari config (1024 untuk MobileNetV3-Small)
    # ============================================================
    feature_dim = config['model'].get('feature_dim', 1024)

    # ============================================================
    # 1. Load backbone unimodal (freeze, tidak ikut ablasi)
    # ============================================================
    audio_model_path = config['paths'].get('audio_model', 'results/unimodal_audio/audio_best.pth')
    # 🔧 FIX: fallback ke MobileNetV3-Small checkpoint
    visual_model_path = config['paths'].get('visual_model', 'results/unimodal_visual_mobilenet/visual_best.pth')

    audio_cfg = load_config('configs/unimodal_audio.yaml')
    visual_cfg = load_config('configs/unimodal_visual.yaml')
    audio_cls_hidden = audio_cfg['model'].get('classifier_hidden', 512)
    visual_cls_hidden = visual_cfg['model'].get('classifier_hidden', 128)
    audio_use_se_block = audio_cfg['model'].get('use_se_block', False)
    visual_use_se_block = visual_cfg['model'].get('use_se_block', False)
    visual_use_temporal_attention = visual_cfg['model'].get('use_temporal_attention', False)
    visual_use_self_attention = visual_cfg['model'].get('use_self_attention', False)

    print(f"  [Dynamic Sync] Audio classifier_hidden = {audio_cls_hidden}, use_se_block = {audio_use_se_block}")
    print(f"  [Dynamic Sync] Visual classifier_hidden = {visual_cls_hidden}, use_se_block = {visual_use_se_block}")

    audio_model = UnimodalAudioModel(
        num_classes=config['dataset']['num_classes'],
        backbone_name=config['model'].get('audio_backbone', 'audio_1dcnn'),
        dropout=0.5,
        classifier_hidden=audio_cls_hidden,
        use_se_block=audio_use_se_block
    ).to(device)
    load_checkpoint(audio_model_path, audio_model, device=device)
    audio_model.eval()

    # 🔧 FIX: tambahkan target_dim=feature_dim pada visual_model
    visual_model = UnimodalVisualModel(
        num_classes=config['dataset']['num_classes'],
        backbone_name=config['model'].get('visual_backbone', 'mobilenetv3_small_100'),
        dropout=0.5,
        classifier_hidden=visual_cls_hidden,
        target_dim=feature_dim,
        use_temporal_attention=visual_use_temporal_attention,
        use_self_attention=visual_use_self_attention,
        use_se_block=visual_use_se_block
    ).to(device)
    load_checkpoint(visual_model_path, visual_model, device=device)
    visual_model.eval()

    # ============================================================
    # 2. Gunakan hanya Actor 1-16 untuk internal split
    # ============================================================
    audio_mode = '1d' if config['model'].get('audio_backbone') == 'audio_1dcnn' else '2d'
    transform_audio, transform_visual = get_multimodal_transforms(
        is_training=False,
        audio_size=(224, 224),
        visual_size=(224, 224),
        use_spec_augment=False,
        audio_mode=audio_mode
    )

    train_dataset = RavdessMultimodal(
        annotation_path=annotation_path,
        audio_root=audio_root,
        visual_root=visual_root,
        subset='training',  # Actor 1-16
        transform_audio=transform_audio,
        transform_visual=transform_visual,
        num_frames=config['data']['visual']['num_frames']
    )
    val_dataset = RavdessMultimodal(
        annotation_path=annotation_path,
        audio_root=audio_root,
        visual_root=visual_root,
        subset='validation',  # Actor 17-20 (cross-actor)
        transform_audio=transform_audio,
        transform_visual=transform_visual,
        num_frames=config['data']['visual']['num_frames']
    )

    # Split hanya dari train_dataset (Actor 1-16)
    train_subset, same_actor_val_subset = create_internal_split_from_dataset(
        train_dataset, val_size=0.2, random_state=42
    )

    same_actor_val_loader = DataLoader(
        same_actor_val_subset, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=config['training']['n_threads'], pin_memory=True
    )

    # Ekstraksi embedding untuk cross-actor validation (Actor 17-20)
    val_dataset_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    audio_embeds_cross, visual_embeds_cross, labels_cross = [], [], []
    with torch.no_grad():
        for batch in val_dataset_loader:
            audio, visual, labels = batch
            audio, visual = audio.to(device), visual.to(device)
            audio_embeds_cross.append(audio_model.extract_features(audio).detach().cpu().numpy())
            visual_embeds_cross.append(visual_model.extract_features(visual).detach().cpu().numpy())
            labels_cross.append(labels.numpy())
    audio_embeds_cross = np.concatenate(audio_embeds_cross, axis=0)
    visual_embeds_cross = np.concatenate(visual_embeds_cross, axis=0)
    labels_cross = np.concatenate(labels_cross, axis=0)

    # Buat DataLoader dari embedding cross-actor
    cross_dataset = TensorDataset(
        torch.FloatTensor(audio_embeds_cross),
        torch.FloatTensor(visual_embeds_cross),
        torch.LongTensor(labels_cross)
    )
    cross_actor_val_loader = DataLoader(
        cross_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['n_threads'],
        pin_memory=True
    )

    print(f"\n✅ Data Split (Aman):")
    print(f"  Train pool (Actor 1-16)        : {len(train_dataset)} sampel")
    print(f"  Train subset (internal split)  : {len(train_subset)} sampel")
    print(f"  Same-actor val (diagnostik)    : {len(same_actor_val_subset)} sampel")
    print(f"  Cross-actor val (Actor 17-20)  : {len(val_dataset)} sampel")

    # ============================================================
    # 3. Ekstraksi embedding (dari train_subset saja)
    # ============================================================
    print("\n[INFO] Ekstraksi embedding audio+visual (dari train_subset)...")
    train_loader = DataLoader(
        train_subset, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=config['training']['n_threads'], pin_memory=True
    )
    train_embeds = extract_embeddings(audio_model, visual_model, train_loader, device)
    val_embeds = extract_embeddings(audio_model, visual_model, same_actor_val_loader, device)

    print(f"  Train: {train_embeds['audio'].shape}, Val: {val_embeds['audio'].shape}")

    # ============================================================
    # 4. Jalankan tiap varian ablasi
    # ============================================================
    rows = []

    for variant in variants:
        name = variant['name']
        use_label_smoothing = variant.get('use_label_smoothing', True)
        use_modality_dropout = variant.get('use_modality_dropout', True)
        fusion_dim = variant.get('fusion_dim', config['model'].get('fusion_dim', 128))
        use_se_block = variant.get('use_se_block', False)

        print("\n" + "="*60)
        print(f"  ABLATION VARIANT: {name}")
        print(f"    label_smoothing  : {use_label_smoothing}")
        print(f"    modality_dropout : {use_modality_dropout}")
        print(f"    fusion_dim       : {fusion_dim}")
        print(f"    use_se_block     : {use_se_block}")
        print("="*60)

        # 🔧 FIX: gunakan feature_dim dari config (bukan hardcode 1280)
        fusion_model = MultimodalFusionModel(
            num_classes=config['dataset']['num_classes'],
            fusion_type=ablation_cfg.get('base_fusion_type', 'gmu'),
            feature_dim=feature_dim,   # ← dari config (1024)
            fusion_dim=fusion_dim,
            classifier_hidden=config['model'].get('classifier_hidden', 128),
            dropout=config['model'].get('dropout', 0.5),
            pretrained=False,
            freeze_backbone=False,
            use_se_block=use_se_block
        ).to(device)

        optimizer = torch.optim.AdamW(
            fusion_model.parameters(),
            lr=config['training'].get('fusion_lr', 1e-4),
            weight_decay=config['training'].get('weight_decay', 0.001)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=ablation_cfg.get('n_epochs', 25),
            eta_min=1e-6
        )

        if use_label_smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=config['training'].get('smoothing', 0.1))
        else:
            class_weights = compute_class_weights(annotation_path, config['dataset']['num_classes']).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        variant_config = {
            'fusion_type': ablation_cfg.get('base_fusion_type', 'gmu'),
            'result_path': os.path.join(result_base, name),
            'store_name': f'ablation_{name}',
            'n_epochs': ablation_cfg.get('n_epochs', 25),
            'early_stop_patience': ablation_cfg.get('early_stop_patience', 5),
            'batch_size': config['training']['batch_size'],
            'use_label_smoothing': use_label_smoothing,
            'smoothing': config['training'].get('smoothing', 0.1),
            'use_modality_dropout': use_modality_dropout,
            'modality_dropout_prob': config['training'].get('modality_dropout_prob', 0.2),
        }

        result = train_multimodal_fold(
            fusion_model=fusion_model,
            train_embeds=train_embeds,
            val_embeds=val_embeds,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            fold_idx=0,
            config=variant_config,
            cross_val_loader=cross_actor_val_loader
        )

        rows.append({
            'variant': name,
            'use_label_smoothing': use_label_smoothing,
            'use_modality_dropout': use_modality_dropout,
            'fusion_dim': fusion_dim,
            'use_se_block': use_se_block,
            'best_val_acc': result['best_prec1'],  # cross-actor
            'best_same_acc': result.get('best_same_prec1', 0.0),
            'best_epoch': result['best_epoch'],
            'f1_macro': result['val_metrics'].get('f1_macro') if result.get('val_metrics') else None,
            'f1_weighted': result['val_metrics'].get('f1_weighted') if result.get('val_metrics') else None,
            'result_path': result['result_path'],
        })

    # ============================================================
    # 5. Simpan hasil
    # ============================================================
    df = pd.DataFrame(rows)
    out_path = os.path.join(config['paths']['results'], 'ablation_study.csv')
    df.to_csv(out_path, index=False)

    print("\n" + "="*60)
    print("  ABLATION STUDY SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    print(f"\n✅ Ablation study disimpan ke {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/multimodal.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    run_exp9_ablation_study(args.config)


if __name__ == '__main__':
    main()