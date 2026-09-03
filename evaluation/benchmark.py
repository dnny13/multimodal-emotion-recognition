# -*- coding: utf-8 -*-
"""
evaluation/benchmark.py
=======================
Perbandingan antar eksperimen.

FIX:
- evaluate_experiment_efficiency: membaca classifier_hidden dan input_shape dari config YAML.
- Penanganan khusus untuk audio_2dcnn dengan input shape (1,1,128,224).
- Menggunakan count_parameters dengan include_backbone=False untuk fusion head.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from glob import glob

from .metrics import compute_classification_metrics
from .efficiency import (
    count_parameters,
    get_model_size,
    compute_flops,
    measure_inference_time,
    compute_flops_multimodal,
    measure_inference_time_multimodal,
    get_checkpoint_size_from_file
)
from models import (
    UnimodalAudioModel,
    UnimodalVisualModel,
    MultimodalFusionModel
)
from training.utils import load_checkpoint
from configs import load_config

def load_experiment_metrics(result_path, metric_file='test_metrics.json'):
    metric_path = os.path.join(result_path, metric_file)
    if os.path.exists(metric_path):
        with open(metric_path, 'r') as f:
            return json.load(f)
    return None

def evaluate_experiment_efficiency(experiment_config, device='cuda'):
    """
    Mengevaluasi efisiensi untuk satu eksperimen secara dinamis.
    Membaca config YAML untuk mendapatkan classifier_hidden dan input_shape.
    """
    model_type = experiment_config.get('model_type', 'multimodal')
    model_path = experiment_config.get('best_model_path')
    num_classes = experiment_config.get('num_classes', 8)

    if not model_path or not os.path.exists(model_path):
        print(f"⚠️ Model tidak ditemukan: {model_path}")
        return None

    # Baca config untuk mendapatkan dimensi yang benar
    try:
        multimodal_cfg = load_config('configs/multimodal.yaml')
        fusion_dim_cfg = multimodal_cfg['model'].get('fusion_dim', 384)
        classifier_hidden_cfg = multimodal_cfg['model'].get('classifier_hidden', 128)
    except:
        fusion_dim_cfg = 384
        classifier_hidden_cfg = 128

    if model_type == 'audio':
        # Baca config audio untuk menentukan backbone dan classifier_hidden
        try:
            audio_cfg = load_config('configs/unimodal_audio.yaml')
            backbone_name = audio_cfg['model'].get('backbone_name', 'audio_1dcnn')
            cls_hidden = audio_cfg['model'].get('classifier_hidden', 512)
            use_se = audio_cfg['model'].get('use_se_block', False)
        except:
            backbone_name = 'audio_1dcnn'
            cls_hidden = 512
            use_se = False

        model = UnimodalAudioModel(
            num_classes=num_classes,
            backbone_name=backbone_name,
            classifier_hidden=cls_hidden,
            use_se_block=use_se
        ).to(device)
        load_checkpoint(model_path, model, device=device)

        params = count_parameters(model)
        size = get_model_size(model)
        # Tentukan input shape berdasarkan backbone
        if '2dcnn' in backbone_name:
            input_shape = (1, 1, 128, 224)
        else:
            input_shape = (1, 120, 224)
        flops = compute_flops(model, input_shape, device)
        inference_time = measure_inference_time(model, input_shape, device)

    elif model_type == 'visual':
        try:
            visual_cfg = load_config('configs/unimodal_visual.yaml')
            cls_hidden = visual_cfg['model'].get('classifier_hidden', 256)
            use_se = visual_cfg['model'].get('use_se_block', False)
            use_temp = visual_cfg['model'].get('use_temporal_attention', False)
            use_self = visual_cfg['model'].get('use_self_attention', False)
        except:
            cls_hidden = 256
            use_se = False
            use_temp = False
            use_self = False

        model = UnimodalVisualModel(
            num_classes=num_classes,
            classifier_hidden=cls_hidden,
            use_se_block=use_se,
            use_temporal_attention=use_temp,
            use_self_attention=use_self
        ).to(device)
        load_checkpoint(model_path, model, device=device)

        params = count_parameters(model)
        size = get_model_size(model)
        # Visual backbone: per-frame FLOPs, tapi kita laporkan per video (num_frames=15)
        # Untuk fair, hitung FLOPs per frame lalu kalikan 15
        flops_per_frame = compute_flops(model, (1, 3, 224, 224), device)
        flops = flops_per_frame * 15  # total per video
        inference_time = measure_inference_time(model, (1, 3, 224, 224), device)

    elif model_type == 'multimodal':
        fusion_type = experiment_config.get('fusion_type', 'gmu')
        if fusion_type == 'attention':
            fusion_dim = 1280
        else:
            fusion_dim = fusion_dim_cfg

        model = MultimodalFusionModel(
            num_classes=num_classes,
            fusion_type=fusion_type,
            fusion_dim=fusion_dim,
            feature_dim=1280,
            classifier_hidden=classifier_hidden_cfg,
            dropout=0.5
        ).to(device)
        load_checkpoint(model_path, model, device=device)

        # Parameter hanya fusion head (tanpa backbone) untuk konsistensi
        params = count_parameters(model, include_backbone=False)
        size = get_checkpoint_size_from_file(model_path)  # ukuran murni state_dict
        flops = compute_flops_multimodal(model, device)
        inference_time = measure_inference_time_multimodal(model, device)

    else:
        return None

    return {
        'params_m': params,
        'size_mb': size,
        'flops_g': flops,
        'inference_ms': inference_time
    }

def generate_comparison_table(experiments, device='cuda'):
    rows = []
    for exp in experiments:
        if 'best_model_path' not in exp:
            continue

        metrics = load_experiment_metrics(os.path.dirname(exp['best_model_path']), 'test_metrics.json')
        if metrics is None:
            # Fallback ke val.log
            log_path = os.path.join(os.path.dirname(exp['best_model_path']), 'val.log')
            if os.path.exists(log_path):
                try:
                    df_val = pd.read_csv(log_path, sep='\t')
                    best_row = df_val.loc[df_val['prec1'].idxmax()]
                    metrics = {
                        'accuracy': best_row['prec1'] / 100,
                        'weighted_accuracy': best_row['prec1'] / 100,
                        'unweighted_accuracy': best_row['prec1'] / 100,
                        'f1_weighted': best_row.get('f1_weighted', 0.0),
                        'f1_macro': best_row.get('f1_macro', 0.0),
                        'cohen_kappa': 0.0
                    }
                except Exception:
                    continue
            else:
                continue

        efficiency = evaluate_experiment_efficiency(exp, device)
        if efficiency is None:
            continue

        row = {
            'Experiment': exp['name'],
            'Accuracy (%)': metrics['accuracy'] * 100,
            'Weighted Acc (%)': metrics.get('weighted_accuracy', metrics['accuracy']) * 100,
            'Unweighted Acc (%)': metrics.get('unweighted_accuracy', metrics['accuracy']) * 100,
            'F1-Weighted': metrics.get('f1_weighted', 0.0),
            'F1-Macro': metrics.get('f1_macro', 0.0),
            'Params (M)': efficiency['params_m'],
            'FLOPs (G)': efficiency['flops_g'],
            'Model Size (MB)': efficiency['size_mb'],
            'Inference (ms)': efficiency['inference_ms'],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def compare_experiments(experiment_dirs, device='cuda'):
    experiments = []
    for exp_dir in experiment_dirs:
        best_model = glob(os.path.join(exp_dir, '*_best.pth'))
        if not best_model:
            best_model = glob(os.path.join(exp_dir, '*_last.pth'))
        if not best_model:
            print(f"⚠️ Tidak ada model (*.pth) ditemukan di: {exp_dir}")
            continue

        name = os.path.basename(exp_dir)
        if 'audio' in name.lower():
            model_type = 'audio'
            fusion_type = None
        elif 'visual' in name.lower():
            model_type = 'visual'
            fusion_type = None
        else:
            model_type = 'multimodal'
            if 'concat' in name.lower():
                fusion_type = 'concat'
            elif 'gmu' in name.lower():
                fusion_type = 'gmu'
            elif 'attention' in name.lower():
                fusion_type = 'attention'
            else:
                fusion_type = 'gmu'

        exp = {
            'name': name,
            'model_type': model_type,
            'best_model_path': best_model[0],
            'num_classes': 8,
            'fusion_type': fusion_type
        }
        experiments.append(exp)

    if not experiments:
        print("❌ Tidak ada eksperimen valid ditemukan.")
        return None

    return generate_comparison_table(experiments, device)

def save_comparison_table(df, output_path='results/comparison_table.csv'):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Tabel perbandingan disimpan ke {output_path}")
    return output_path

def print_comparison_table(df):
    print("\n" + "=" * 120)
    print("  EXPERIMENT COMPARISON TABLE")
    print("=" * 120)
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(df.to_string(index=False))
    print("=" * 120)
    if not df.empty:
        best_acc = df.loc[df['Accuracy (%)'].idxmax(), 'Experiment']
        best_f1 = df.loc[df['F1-Macro'].idxmax(), 'Experiment']
        smallest_params = df.loc[df['Params (M)'].idxmin(), 'Experiment']
        print(f"\n  🏆 Best Accuracy  : {best_acc} ({df['Accuracy (%)'].max():.2f}%)")
        print(f"  🏆 Best F1-Macro  : {best_f1} ({df['F1-Macro'].max():.4f})")
        print(f"  🏆 Smallest Model : {smallest_params} ({df['Params (M)'].min():.2f} M)")
    print("=" * 120)