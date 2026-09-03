# -*- coding: utf-8 -*-
"""
experiments/exp5_multimodal_attention.py
=========================================
Eksperimen 5: Multimodal dengan Cross-Attention Fusion (Pembanding).

🔧 FIX: Pada ensemble_predict, proyeksikan embedding sebelum fusion.
"""

import os
import sys
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RavdessMultimodal, get_multimodal_transforms
from models import UnimodalAudioModel, UnimodalVisualModel, MultimodalFusionModel
from training.utils import load_checkpoint, compute_class_weights
from training.validation import validate_epoch_with_cache
from training.train_multimodal import FocalLoss, train_multimodal_fold
from training.split_utils import create_internal_split_from_dataset
from evaluation import save_test_metrics, roofline_analysis, compute_classification_metrics
from configs import load_config


def ensemble_predict(fold_models, test_embeds, device):
    all_probs = []
    for model in fold_models:
        model.eval()
        model = model.to(device)
        audio = torch.FloatTensor(test_embeds['audio']).to(device)
        visual = torch.FloatTensor(test_embeds['visual']).to(device)
        with torch.no_grad():
            # 🔧 PERBAIKAN: Proyeksi sebelum fusion
            audio_proj = model.audio_projector(audio)
            visual_proj = model.visual_projector(visual)
            fused = model.fusion(audio_proj, visual_proj)
            outputs = model.classifier(fused)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
    avg_probs = np.mean(all_probs, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    return preds, avg_probs


def _load_unimodal_arch_flags():
    audio_cfg = load_config('configs/unimodal_audio.yaml')
    visual_cfg = load_config('configs/unimodal_visual.yaml')
    audio_flags = {
        'use_se_block': audio_cfg['model'].get('use_se_block', False)
    }
    visual_flags = {
        'use_se_block': visual_cfg['model'].get('use_se_block', False),
        'use_temporal_attention': visual_cfg['model'].get('use_temporal_attention', False),
        'use_self_attention': visual_cfg['model'].get('use_self_attention', False)
    }
    return audio_flags, visual_flags


def run_exp5_multimodal_attention(config_path=None, override=None):
    config = load_config(config_path or 'configs/multimodal.yaml')
    if override:
        config.update(override)

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device.upper()}")

    annotation_path = config['paths']['annotations']
    audio_root = config['paths']['output_audio']
    visual_root = config['paths']['output_visual']
    
    result_main_path = os.path.join(config['paths']['results'], 'multimodal_attention')
    os.makedirs(result_main_path, exist_ok=True)

    cv_base_path = os.path.join(config['paths']['results'], 'robustness_cv', 'multimodal_attention')
    os.makedirs(cv_base_path, exist_ok=True)

    cache_dir = os.path.join(config['paths']['embeddings_cache'], 'attention')
    os.makedirs(cache_dir, exist_ok=True)

    n_folds = config['training'].get('n_folds', 5)
    num_frames = config['data']['visual'].get('num_frames', 15)

    # ========== Baca dimensi dari config ==========
    feature_dim = config['model'].get('feature_dim', 1024)
    fusion_dim = config['model'].get('fusion_dim', 128)
    visual_backbone = config['model'].get('visual_backbone', 'mobilenetv3_small_100')
    audio_backbone = config['model'].get('audio_backbone', 'audio_1dcnn')
    # ===============================================

    fusion_config = {
        'fusion_type': 'attention',
        'feature_dim': feature_dim,
        'fusion_dim': fusion_dim,
        'classifier_hidden': config['model'].get('classifier_hidden', 128),
        'dropout': config['model'].get('dropout', 0.5),
        'n_epochs': config['training'].get('n_epochs', 50),
        'early_stop_patience': config['training'].get('early_stop_patience', 8),
        'batch_size': config['training']['batch_size'],
        'fusion_lr': config['training'].get('fusion_lr', 0.001),
        'weight_decay': config['training'].get('weight_decay', 0.001),
        'cache_dir': cache_dir,
        'num_classes': config['dataset']['num_classes'],
        'result_path': cv_base_path,
        'store_name': 'multimodal_attention',
        'loss_type': config['training'].get('loss_type', 'focal'),
        'focal_gamma': config['training'].get('focal_gamma', 2.0),
        'focal_alpha': config['training'].get('focal_alpha', 0.25),
        'use_mixup': config['training'].get('use_mixup', True),
        'mixup_alpha': config['training'].get('mixup_alpha', 0.2),
        'use_modality_dropout': config['training'].get('use_modality_dropout', True),
        'modality_dropout_prob': config['training'].get('modality_dropout_prob', 0.25),
        'freeze_backbone': config['model'].get('freeze_backbone', False)
    }

    print(f"\n📐 Fusion Config: {fusion_config['fusion_type']} (Pembanding)")
    print(f"  N Folds     : {n_folds} (GroupKFold - Actor-Based)")
    print(f"  Internal Split : ✅ (aktor sama untuk early stopping)")

    # ---- Load Backbones ----
    audio_model_path = config['paths'].get('audio_model', 'results/unimodal_audio/audio_best.pth')
    visual_model_path = config['paths'].get('visual_model', 'results/unimodal_visual_mobilenet/visual_best.pth')

    audio_cfg = load_config('configs/unimodal_audio.yaml')
    visual_cfg = load_config('configs/unimodal_visual.yaml')
    audio_cls_hidden = audio_cfg['model'].get('classifier_hidden', 512)
    visual_cls_hidden = visual_cfg['model'].get('classifier_hidden', 128)

    audio_flags, visual_flags = _load_unimodal_arch_flags()
    print(f"  [Dynamic Sync] Audio classifier_hidden = {audio_cls_hidden}")
    print(f"  [Dynamic Sync] Visual classifier_hidden = {visual_cls_hidden}")
    print(f"  [Arch Sync] Audio  use_se_block           = {audio_flags['use_se_block']}")
    print(f"  [Arch Sync] Visual use_se_block           = {visual_flags['use_se_block']}")
    print(f"  [Arch Sync] Visual use_temporal_attention = {visual_flags['use_temporal_attention']}")
    print(f"  [Arch Sync] Visual use_self_attention      = {visual_flags['use_self_attention']}")

    audio_model = UnimodalAudioModel(
        num_classes=8,
        backbone_name=audio_backbone,
        dropout=0.5,
        classifier_hidden=audio_cls_hidden,
        use_se_block=audio_flags['use_se_block']
    ).to(device)
    load_checkpoint(audio_model_path, audio_model, device=device)
    audio_model.eval()

    visual_model = UnimodalVisualModel(
        num_classes=8,
        backbone_name=visual_backbone,
        dropout=0.5,
        classifier_hidden=visual_cls_hidden,
        target_dim=feature_dim,
        use_temporal_attention=visual_flags['use_temporal_attention'],
        use_self_attention=visual_flags['use_self_attention'],
        use_se_block=visual_flags['use_se_block']
    ).to(device)
    load_checkpoint(visual_model_path, visual_model, device=device)
    visual_model.eval()

    # ---- Data ----
    transform_audio_train, transform_visual_train = get_multimodal_transforms(
        is_training=True, audio_size=(224,224), visual_size=(224,224),
        use_spec_augment=False, audio_mode='1d'
    )
    transform_audio_val, transform_visual_val = get_multimodal_transforms(
        is_training=False, audio_size=(224,224), visual_size=(224,224),
        use_spec_augment=False, audio_mode='1d'
    )

    train_dataset = RavdessMultimodal(
        annotation_path, audio_root, visual_root, 'training',
        transform_audio_train, transform_visual_train, num_frames
    )
    val_dataset = RavdessMultimodal(
        annotation_path, audio_root, visual_root, 'validation',
        transform_audio_val, transform_visual_val, num_frames
    )
    test_dataset = RavdessMultimodal(
        annotation_path, audio_root, visual_root, 'testing',
        transform_audio_val, transform_visual_val, num_frames
    )

    # ---- Internal Split ----
    train_subset, same_actor_val_subset = create_internal_split_from_dataset(
        train_dataset, val_size=0.2, random_state=42
    )

    # ---- Ekstraksi Embedding ----
    train_loader_emb = DataLoader(train_subset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    audio_embeds_train, visual_embeds_train, labels_train = [], [], []
    with torch.no_grad():
        for batch in train_loader_emb:
            audio, visual, labels = batch
            audio, visual = audio.to(device), visual.to(device)
            audio_embeds_train.append(audio_model.extract_features(audio).detach().cpu().numpy())
            visual_embeds_train.append(visual_model.extract_features(visual).detach().cpu().numpy())
            labels_train.append(labels.numpy())
    audio_embeds_train = np.concatenate(audio_embeds_train, axis=0)
    visual_embeds_train = np.concatenate(visual_embeds_train, axis=0)
    labels_train = np.concatenate(labels_train, axis=0)

    val_loader_emb = DataLoader(same_actor_val_subset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    audio_embeds_val, visual_embeds_val, labels_val = [], [], []
    with torch.no_grad():
        for batch in val_loader_emb:
            audio, visual, labels = batch
            audio, visual = audio.to(device), visual.to(device)
            audio_embeds_val.append(audio_model.extract_features(audio).detach().cpu().numpy())
            visual_embeds_val.append(visual_model.extract_features(visual).detach().cpu().numpy())
            labels_val.append(labels.numpy())
    audio_embeds_val = np.concatenate(audio_embeds_val, axis=0)
    visual_embeds_val = np.concatenate(visual_embeds_val, axis=0)
    labels_val = np.concatenate(labels_val, axis=0)

    # Cross-actor validation embedding
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

    print(f"\nEmbedding extracted:")
    print(f"  Train (Actor 1-16, 80%): {audio_embeds_train.shape}")
    print(f"  Same-Actor Val (Actor 1-16, 20%): {audio_embeds_val.shape}")
    print(f"  Cross-Actor Val (Actor 17-20): {audio_embeds_cross.shape}")

    train_embeds_internal = {'audio': audio_embeds_train, 'visual': visual_embeds_train, 'labels': labels_train}
    val_embeds_internal = {'audio': audio_embeds_val, 'visual': visual_embeds_val, 'labels': labels_val}
    cross_embeds = {'audio': audio_embeds_cross, 'visual': visual_embeds_cross, 'labels': labels_cross}

    cross_dataset = TensorDataset(
        torch.FloatTensor(cross_embeds['audio']),
        torch.FloatTensor(cross_embeds['visual']),
        torch.LongTensor(cross_embeds['labels'])
    )
    cross_actor_val_loader = DataLoader(
        cross_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['n_threads'],
        pin_memory=True
    )

    # ---- Internal Split Training ----
    fusion_model = MultimodalFusionModel(
        num_classes=8,
        fusion_type='attention',
        audio_backbone=audio_backbone,
        visual_backbone=visual_backbone,
        feature_dim=feature_dim,
        fusion_dim=fusion_dim,
        classifier_hidden=fusion_config['classifier_hidden'],
        dropout=fusion_config['dropout']
    ).to(device)

    if fusion_config['loss_type'] == 'focal':
        criterion = FocalLoss(gamma=fusion_config['focal_gamma'], alpha=fusion_config['focal_alpha'])
    else:
        class_weights = compute_class_weights(annotation_path, config['dataset']['num_classes']).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=fusion_config['fusion_lr'], weight_decay=fusion_config['weight_decay'])

    train_loader_temp = DataLoader(
        TensorDataset(
            torch.FloatTensor(train_embeds_internal['audio']),
            torch.FloatTensor(train_embeds_internal['visual']),
            torch.LongTensor(train_embeds_internal['labels'])
        ),
        batch_size=fusion_config['batch_size'], shuffle=True, drop_last=True
    )
    steps_per_epoch = len(train_loader_temp)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=fusion_config['n_epochs'],
        steps_per_epoch=steps_per_epoch, pct_start=0.3, div_factor=10, final_div_factor=100
    )

    print("\n🚀 Training utama dengan internal split (early stopping)...")
    train_result = train_multimodal_fold(
        fusion_model, train_embeds_internal, val_embeds_internal,
        criterion, optimizer, scheduler, device, 0, fusion_config,
        cross_val_loader=cross_actor_val_loader
    )
    same_actor_val_acc = train_result['best_same_prec1']
    print(f"\n✅ Same-Actor Validation Accuracy: {same_actor_val_acc:.2f}%")

    # ---- 5-Fold CV ----
    audio_embeds_all = np.concatenate([audio_embeds_train, audio_embeds_val], axis=0)
    visual_embeds_all = np.concatenate([visual_embeds_train, visual_embeds_val], axis=0)
    labels_all = np.concatenate([labels_train, labels_val], axis=0)

    actor_ids_train = train_dataset.get_actor_ids()
    actor_ids_train_subset = [actor_ids_train[i] for i in train_subset.indices]
    actor_ids_val_subset = [actor_ids_train[i] for i in same_actor_val_subset.indices]
    actor_ids_all_cv = actor_ids_train_subset + actor_ids_val_subset

    print("\n🔁 5-Fold Actor-Based CV (GroupKFold)...")
    gkf = GroupKFold(n_splits=n_folds)
    fold_results, fold_models = [], []
    best_fold_acc, best_fold_model_path = 0.0, None

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(audio_embeds_all, labels_all, groups=actor_ids_all_cv)):
        train_actors = sorted(set(np.array(actor_ids_all_cv)[train_idx]))
        print(f"\n  Fold {fold_idx+1}/{n_folds} - Train actors: {train_actors[:5]}... (total {len(train_actors)} actors)")

        train_embeds = {
            'audio': audio_embeds_all[train_idx],
            'visual': visual_embeds_all[train_idx],
            'labels': labels_all[train_idx]
        }
        val_embeds = {
            'audio': audio_embeds_all[val_idx],
            'visual': visual_embeds_all[val_idx],
            'labels': labels_all[val_idx]
        }

        fold_model = MultimodalFusionModel(
            num_classes=8,
            fusion_type='attention',
            audio_backbone=audio_backbone,
            visual_backbone=visual_backbone,
            feature_dim=feature_dim,
            fusion_dim=fusion_dim,
            classifier_hidden=fusion_config['classifier_hidden'],
            dropout=fusion_config['dropout']
        ).to(device)

        fold_opt = torch.optim.AdamW(fold_model.parameters(), lr=fusion_config['fusion_lr'], weight_decay=fusion_config['weight_decay'])

        fold_train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(train_embeds['audio']),
                torch.FloatTensor(train_embeds['visual']),
                torch.LongTensor(train_embeds['labels'])
            ),
            batch_size=fusion_config['batch_size'], shuffle=True, drop_last=True
        )
        steps_per_epoch_fold = len(fold_train_loader)
        fold_sched = torch.optim.lr_scheduler.OneCycleLR(
            fold_opt, max_lr=0.01, epochs=fusion_config['n_epochs'] // 2,
            steps_per_epoch=steps_per_epoch_fold, pct_start=0.3, div_factor=10, final_div_factor=100
        )

        fold_result = train_multimodal_fold(
            fold_model, train_embeds, val_embeds,
            criterion, fold_opt, fold_sched, device, fold_idx, fusion_config,
            cross_val_loader=cross_actor_val_loader
        )
        fold_results.append(fold_result)
        fold_models.append(fold_model)

        if fold_result['best_prec1'] > best_fold_acc:
            best_fold_acc = fold_result['best_prec1']
            best_fold_model_path = fold_result['result_path'] + '/multimodal_attention_best.pth'

        torch.cuda.empty_cache()

    # ---- Evaluasi Test ----
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_audio_embeds, test_visual_embeds, test_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            audio, visual, labels = batch
            audio, visual = audio.to(device), visual.to(device)
            test_audio_embeds.append(audio_model.extract_features(audio).detach().cpu().numpy())
            test_visual_embeds.append(visual_model.extract_features(visual).detach().cpu().numpy())
            test_labels.append(labels.numpy())

    test_audio_embeds = np.concatenate(test_audio_embeds, axis=0)
    test_visual_embeds = np.concatenate(test_visual_embeds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_embeds = {'audio': test_audio_embeds, 'visual': test_visual_embeds, 'labels': test_labels}

    preds_ensemble, _ = ensemble_predict(fold_models, test_embeds, device)
    metrics_ensemble = compute_classification_metrics(test_labels, preds_ensemble)
    print(f"\n📊 Test Accuracy (Ensemble): {metrics_ensemble['accuracy']*100:.2f}%")

    if best_fold_model_path and os.path.exists(best_fold_model_path):
        best_model = MultimodalFusionModel(
            num_classes=8,
            fusion_type='attention',
            audio_backbone=audio_backbone,
            visual_backbone=visual_backbone,
            feature_dim=feature_dim,
            fusion_dim=fusion_dim,
            classifier_hidden=fusion_config['classifier_hidden'],
            dropout=fusion_config['dropout']
        ).to(device)
        load_checkpoint(best_fold_model_path, best_model, device=device)

        test_dataset_emb = TensorDataset(
            torch.FloatTensor(test_audio_embeds),
            torch.FloatTensor(test_visual_embeds),
            torch.LongTensor(test_labels)
        )
        test_loader_emb = DataLoader(test_dataset_emb, batch_size=16, shuffle=False)

        _, val_acc, val_metrics = validate_epoch_with_cache(
            -1, test_loader_emb, best_model, nn.CrossEntropyLoss(), device, None
        )
        print(f"📊 Test Accuracy (Best Single): {val_acc:.2f}%")

        val_metrics['ensemble_accuracy'] = metrics_ensemble['accuracy']
        val_metrics['ensemble_f1_macro'] = metrics_ensemble['f1_macro']
        save_test_metrics(result_main_path, val_metrics)

        main_model_path = os.path.join(result_main_path, 'multimodal_attention_best.pth')
        shutil.copyfile(best_fold_model_path, main_model_path)
        print(f"✅ Best model copied to {main_model_path} for exp7 comparison.")

        roofline_analysis(best_model, 'Multimodal Attention', device, is_multimodal=True)

    # ---- CV Summary ----
    accuracies = [r['best_prec1'] for r in fold_results]
    print("\n" + "="*60)
    print("  5-FOLD ACTOR-BASED CV RESULTS (Attention)")
    for i, acc in enumerate(accuracies):
        print(f"  Fold {i+1}: {acc:.2f}%")
    print(f"  Mean: {np.mean(accuracies):.2f}% (±{np.std(accuracies):.2f})")
    print("="*60)

    cv_summary_path = os.path.join(cv_base_path, 'cv_summary.csv')
    pd.DataFrame({
        'fold': list(range(1, len(accuracies) + 1)),
        'best_prec1': accuracies,
        'f1_macro': [r.get('best_f1', 0) for r in fold_results]
    }).to_csv(cv_summary_path, index=False)
    print(f"✅ CV summary saved to {cv_summary_path}")

    with open(os.path.join(cv_base_path, 'same_actor_val_metrics.json'), 'w') as f:
        json.dump({'best_same_actor_acc': same_actor_val_acc}, f)
    print(f"✅ Same-actor validation acc saved to {cv_base_path}/same_actor_val_metrics.json")

    print(f"\n✅ Eksperimen 5 (Attention) selesai!")
    return best_model, val_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/multimodal.yaml')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run_exp5_multimodal_attention(args.config)


if __name__ == '__main__':
    main()