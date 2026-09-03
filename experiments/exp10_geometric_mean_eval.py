# -*- coding: utf-8 -*-
"""
experiments/exp10_geometric_mean_eval.py
=========================================
Eksperimen 10: Geometric Mean Fusion Comparison (Post-Hoc)

🔧 FIX:
  - Perbaiki path visual checkpoint ke results/unimodal_visual_mobilenet/visual_best.pth.
  - Tambahkan target_dim=feature_dim pada UnimodalVisualModel.
  - Baca visual_backbone dari config.
  - Baca feature_dim dari config (1024 untuk MobileNetV3-Small).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RavdessMultimodal, get_multimodal_transforms
from models import UnimodalAudioModel, UnimodalVisualModel, MultimodalFusionModel
from training.utils import load_checkpoint
from evaluation import compute_classification_metrics, geometric_mean_fusion
from configs import load_config


def get_model_dimensions_from_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    possible_cls_keys = [
        'classifier.1.weight',
        'module.classifier.1.weight',
        'model.classifier.1.weight',
    ]
    
    fusion_dim = None
    classifier_hidden = None
    
    possible_proj_keys = [
        'fusion.proj_audio.weight',
        'module.fusion.proj_audio.weight',
        'model.fusion.proj_audio.weight',
        'fusion.proj_audio.0.weight',
    ]
    
    for key in possible_proj_keys:
        if key in state_dict:
            fusion_dim = state_dict[key].shape[0]
            print(f"  [Dynamic Load] fusion_dim ditemukan dari {key}: {fusion_dim}")
            break
    
    if fusion_dim is None:
        for key in possible_cls_keys:
            if key in state_dict:
                fusion_dim = state_dict[key].shape[1]
                print(f"  [Dynamic Load] fusion_dim ditemukan dari {key}: {fusion_dim}")
                break
    
    if fusion_dim is None:
        print("  [Dynamic Load] fusion_dim tidak ditemukan, gunakan default 128")
        fusion_dim = 128
    
    for key in possible_cls_keys:
        if key in state_dict:
            classifier_hidden = state_dict[key].shape[0]
            print(f"  [Dynamic Load] classifier_hidden ditemukan dari {key}: {classifier_hidden}")
            break
    
    if classifier_hidden is None:
        print("  [Dynamic Load] classifier_hidden tidak ditemukan, gunakan default 128")
        classifier_hidden = 128
    
    print(f"  [Dynamic Load] FINAL: fusion_dim={fusion_dim}, classifier_hidden={classifier_hidden}")
    return fusion_dim, classifier_hidden


def run_exp10(config_path, device='cuda'):
    config = load_config(config_path)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ============================================================
    # 🔧 Baca dimensi dan backbone dari config
    # ============================================================
    feature_dim = config['model'].get('feature_dim', 1024)
    visual_backbone = config['model'].get('visual_backbone', 'mobilenetv3_small_100')
    audio_backbone = config['model'].get('audio_backbone', 'audio_1dcnn')

    audio_cfg = load_config('configs/unimodal_audio.yaml')
    visual_cfg = load_config('configs/unimodal_visual.yaml')

    audio_cls_hidden = audio_cfg['model'].get('classifier_hidden', 512)
    audio_use_se = audio_cfg['model'].get('use_se_block', True)

    visual_cls_hidden = visual_cfg['model'].get('classifier_hidden', 128)
    visual_use_se = visual_cfg['model'].get('use_se_block', False)
    visual_use_temp = visual_cfg['model'].get('use_temporal_attention', True)
    visual_use_self = visual_cfg['model'].get('use_self_attention', False)

    print(f"  [Audio Config] classifier_hidden={audio_cls_hidden}, use_se_block={audio_use_se}")
    print(f"  [Visual Config] classifier_hidden={visual_cls_hidden}, use_se_block={visual_use_se}")

    annotation_path = config['paths']['annotations']
    audio_root = config['paths']['output_audio']
    visual_root = config['paths']['output_visual']
    num_frames = config['data']['visual'].get('num_frames', 15)
    result_path = 'results/multimodal_gmu'
    os.makedirs(result_path, exist_ok=True)

    transform_audio, transform_visual = get_multimodal_transforms(
        is_training=False,
        audio_size=(224, 224),
        visual_size=(224, 224),
        use_spec_augment=False,
        audio_mode='1d'
    )

    test_dataset = RavdessMultimodal(
        annotation_path=annotation_path,
        audio_root=audio_root,
        visual_root=visual_root,
        subset='testing',
        transform_audio=transform_audio,
        transform_visual=transform_visual,
        num_frames=num_frames
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # ---- Load Audio Model ----
    audio_ckpt = 'results/unimodal_audio/audio_best.pth'
    audio_model = UnimodalAudioModel(
        num_classes=8,
        backbone_name=audio_backbone,
        dropout=0.5,
        classifier_hidden=audio_cls_hidden,
        use_se_block=audio_use_se
    ).to(device)
    load_checkpoint(audio_ckpt, audio_model, device=device)
    audio_model.eval()
    print(f"Audio checkpoint: {audio_ckpt}")

    # ---- Load Visual Model (MobileNetV3-Small) ----
    # 🔧 PERBAIKAN: Path checkpoint yang benar
    visual_ckpt = 'results/unimodal_visual_mobilenet/visual_best.pth'
    visual_model = UnimodalVisualModel(
        num_classes=8,
        backbone_name=visual_backbone,
        dropout=0.5,
        classifier_hidden=visual_cls_hidden,
        target_dim=feature_dim,  # ← target_dim dari config
        use_se_block=visual_use_se,
        use_temporal_attention=visual_use_temp,
        use_self_attention=visual_use_self
    ).to(device)
    load_checkpoint(visual_ckpt, visual_model, device=device)
    visual_model.eval()
    print(f"Visual checkpoint: {visual_ckpt}")

    # ---- Load GMU Model ----
    gmu_ckpt = 'results/multimodal_gmu/multimodal_gmu_best.pth'
    if not os.path.exists(gmu_ckpt):
        gmu_ckpt = 'results/multimodal_gmu/fold_0/multimodal_gmu_best.pth'
        if not os.path.exists(gmu_ckpt):
            raise FileNotFoundError(f"GMU model tidak ditemukan di: {gmu_ckpt}")

    fusion_dim, classifier_hidden = get_model_dimensions_from_checkpoint(gmu_ckpt)

    # 🔧 PERBAIKAN: tambahkan audio_backbone dan visual_backbone
    gmu_model = MultimodalFusionModel(
        num_classes=8,
        fusion_type='gmu',
        audio_backbone=audio_backbone,
        visual_backbone=visual_backbone,
        feature_dim=feature_dim,
        fusion_dim=fusion_dim,
        classifier_hidden=classifier_hidden,
        dropout=0.5
    ).to(device)
    load_checkpoint(gmu_ckpt, gmu_model, device=device)
    gmu_model.eval()
    print(f"GMU checkpoint: {gmu_ckpt}")

    # ---- Ekstraksi Embedding untuk Semua Model ----
    audio_embeds, visual_embeds, labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            audio, visual, y = batch
            audio = audio.to(device)
            visual = visual.to(device)
            audio_embeds.append(audio_model.extract_features(audio).cpu().numpy())
            visual_embeds.append(visual_model.extract_features(visual).cpu().numpy())
            labels.append(y.numpy())

    audio_embeds = np.concatenate(audio_embeds, axis=0)
    visual_embeds = np.concatenate(visual_embeds, axis=0)
    labels = np.concatenate(labels, axis=0)

    test_tensor = torch.utils.data.TensorDataset(
        torch.FloatTensor(audio_embeds),
        torch.FloatTensor(visual_embeds),
        torch.LongTensor(labels)
    )
    test_loader_emb = DataLoader(test_tensor, batch_size=16, shuffle=False)

    # ---- GMU Prediction (dengan proyeksi) ----
    gmu_preds = []
    gmu_probs = []
    with torch.no_grad():
        for batch in test_loader_emb:
            a, v, _ = batch
            a = a.to(device)
            v = v.to(device)
            # 🔧 PERBAIKAN: Proyeksi sebelum fusion
            a_proj = gmu_model.audio_projector(a)
            v_proj = gmu_model.visual_projector(v)
            fused = gmu_model.fusion(a_proj, v_proj)
            outputs = gmu_model.classifier(fused)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            gmu_probs.append(probs)
            gmu_preds.append(np.argmax(probs, axis=1))
    gmu_preds = np.concatenate(gmu_preds, axis=0)
    gmu_probs = np.concatenate(gmu_probs, axis=0)

    # ---- Unimodal Predictions ----
    audio_preds = []
    visual_preds = []
    audio_probs = []
    visual_probs = []

    with torch.no_grad():
        for batch in test_loader:
            audio, visual, _ = batch
            audio = audio.to(device)
            visual = visual.to(device)

            a_out = audio_model(audio)
            a_probs = torch.softmax(a_out, dim=1).cpu().numpy()
            audio_probs.append(a_probs)
            audio_preds.append(np.argmax(a_probs, axis=1))

            v_out = visual_model(visual)
            v_probs = torch.softmax(v_out, dim=1).cpu().numpy()
            visual_probs.append(v_probs)
            visual_preds.append(np.argmax(v_probs, axis=1))

    audio_preds = np.concatenate(audio_preds, axis=0)
    visual_preds = np.concatenate(visual_preds, axis=0)
    audio_probs = np.concatenate(audio_probs, axis=0)
    visual_probs = np.concatenate(visual_probs, axis=0)

    # ---- Post-Hoc Fusion Methods ----
    gm_probs = geometric_mean_fusion(audio_probs, visual_probs)
    gm_preds = np.argmax(gm_probs, axis=1)

    am_probs = (audio_probs + visual_probs) / 2.0
    am_preds = np.argmax(am_probs, axis=1)

    max_vote_preds = []
    for i in range(len(labels)):
        if audio_preds[i] == visual_preds[i]:
            max_vote_preds.append(audio_preds[i])
        else:
            if audio_probs[i].max() >= visual_probs[i].max():
                max_vote_preds.append(audio_preds[i])
            else:
                max_vote_preds.append(visual_preds[i])
    max_vote_preds = np.array(max_vote_preds)

    # ---- Compute Metrics ----
    metrics_gmu = compute_classification_metrics(labels, gmu_preds)
    metrics_gm = compute_classification_metrics(labels, gm_preds)
    metrics_am = compute_classification_metrics(labels, am_preds)
    metrics_max_vote = compute_classification_metrics(labels, max_vote_preds)
    metrics_audio = compute_classification_metrics(labels, audio_preds)
    metrics_visual = compute_classification_metrics(labels, visual_preds)

    comparison_data = []
    for name, metrics in [
        ('Audio Only', metrics_audio),
        ('Visual Only', metrics_visual),
        ('GMU (Adaptive, Intermediate)', metrics_gmu),
        ('Geometric Mean (Statis, Post-Hoc)', metrics_gm),
        ('Arithmetic Mean (Statis, Post-Hoc)', metrics_am),
        ('Max Voting (Statis, Post-Hoc)', metrics_max_vote)
    ]:
        comparison_data.append({
            'Method': name,
            'Accuracy (%)': metrics['accuracy'] * 100,
            'Weighted Acc (%)': metrics['weighted_accuracy'] * 100,
            'Unweighted Acc (%)': metrics['unweighted_accuracy'] * 100,
            'F1-Macro': metrics['f1_macro'],
            'F1-Weighted': metrics['f1_weighted'],
            'Cohen Kappa': metrics['cohen_kappa']
        })

    df = pd.DataFrame(comparison_data)
    output_path = os.path.join(result_path, 'geometric_mean_comparison.csv')
    df.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print("  GEOMETRIC MEAN FUSION COMPARISON")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    print(f"\nResult saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/multimodal.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run_exp10(args.config, args.device)