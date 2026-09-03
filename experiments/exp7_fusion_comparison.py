# -*- coding: utf-8 -*-
"""
experiments/exp7_fusion_comparison.py
=====================================
Eksperimen 7: Perbandingan Semua Metode Fusion + Lightweight Analysis.
"""

import os
import sys
import json
import argparse
import pandas as pd
import torch
import numpy as np
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import load_config
from evaluation import (
    compare_experiments,
    print_comparison_table,
    save_comparison_table,
    check_lightweight_constraints,
    roofline_analysis
)
from evaluation.metrics import convert_to_serializable
from training.utils import set_seed


def run_exp7_fusion_comparison(config_path=None, override=None):
    config = load_config(config_path or 'configs/multimodal.yaml')
    if override:
        config.update(override)

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device.upper()}")

    set_seed(config['training'].get('manual_seed', 42),
             deterministic=config['training'].get('cudnn_deterministic', True))

    # 🔧 PERBAIKAN: Baca feature_dim dari config
    feature_dim = config['model'].get('feature_dim', 576)
    fusion_dim_cfg = config['model'].get('fusion_dim', 128)
    classifier_hidden_cfg = config['model'].get('classifier_hidden', 128)

    base_results = config['paths']['results']

    experiment_dirs = [
        os.path.join(base_results, 'unimodal_audio'),
        os.path.join(base_results, 'unimodal_visual'),
        os.path.join(base_results, 'multimodal_concat'),
        os.path.join(base_results, 'multimodal_gmu'),
        os.path.join(base_results, 'multimodal_attention'),
    ]

    existing_dirs = [d for d in experiment_dirs if os.path.exists(d)]

    if not existing_dirs:
        print("❌ Tidak ada direktori eksperimen ditemukan.")
        print("   Pastikan eksperimen 1-5 sudah dijalankan.")
        return None

    print("\n📂 Eksperimen yang ditemukan:")
    for d in existing_dirs:
        print(f"  - {d}")

    df = compare_experiments(existing_dirs, device)

    if df is None or df.empty:
        print("❌ Gagal membuat tabel perbandingan.")
        return None

    # Lightweight Constraint Check
    print("\n" + "="*80)
    print("  LIGHTWEIGHT CONSTRAINT CHECK — PER MODEL")
    print("="*80)

    lightweight_status = []
    for idx, row in df.iterrows():
        params = row.get('Params (M)', 0)
        flops = row.get('FLOPs (G)', 0)
        size = row.get('Model Size (MB)', 0)
        inference = row.get('Inference (ms)', 0)

        status = check_lightweight_constraints(params, flops, size, inference, verbose=False)
        lightweight_status.append({
            'Experiment': row['Experiment'],
            'Params Pass': status['params']['pass'],
            'FLOPs Pass': status['flops']['pass'],
            'Size Pass': status['size']['pass'],
            'Inference Pass': status['inference']['pass'],
            'All Pass': all(c['pass'] for c in status.values())
        })

    df_lightweight = pd.DataFrame(lightweight_status)
    print("\n📊 Lightweight Constraint Status (Threshold: <12M, <2G, <50MB, <100ms):")
    print(df_lightweight.to_string(index=False))

    # Roofline Analysis
    print("\n" + "="*80)
    print("  ROOFLINE ANALYSIS — TRADE-OFF VISUALIZATION")
    print("="*80)

    from models import UnimodalAudioModel, UnimodalVisualModel, MultimodalFusionModel
    from training.utils import load_checkpoint

    roofline_results = []
    for exp_dir in existing_dirs:
        name = os.path.basename(exp_dir)
        
        best_model = os.path.join(exp_dir, f'{name}_best.pth')
        if not os.path.exists(best_model):
            best_model_list = glob.glob(os.path.join(exp_dir, '*_best.pth'))
            if best_model_list:
                best_model = best_model_list[0]
            else:
                continue

        try:
            if 'audio' in name.lower():
                model = UnimodalAudioModel(num_classes=8).to(device)
                load_checkpoint(best_model, model, device=device)
                is_multimodal = False
            elif 'visual' in name.lower():
                model = UnimodalVisualModel(num_classes=8, classifier_hidden=256).to(device)
                load_checkpoint(best_model, model, device=device)
                is_multimodal = False
            else:
                fusion_type = 'gmu'
                if 'concat' in name.lower():
                    fusion_type = 'concat'
                elif 'attention' in name.lower():
                    fusion_type = 'attention'
                
                if fusion_type == 'attention':
                    fusion_dim = feature_dim  # Cross-Attention output dim = feature_dim
                else:
                    fusion_dim = fusion_dim_cfg

                # 🔧 PERBAIKAN: Gunakan feature_dim dari config
                model = MultimodalFusionModel(
                    num_classes=8,
                    fusion_type=fusion_type,
                    fusion_dim=fusion_dim,
                    feature_dim=feature_dim,  # ← dari config
                    classifier_hidden=classifier_hidden_cfg,
                    dropout=0.5
                ).to(device)
                load_checkpoint(best_model, model, device=device)
                is_multimodal = True

            analysis = roofline_analysis(model, name, device, is_multimodal)
            analysis['Experiment'] = name
            roofline_results.append(analysis)
        except Exception as e:
            print(f"⚠️ Gagal roofline untuk {name}: {e}")

    df_roofline = pd.DataFrame(roofline_results)
    if not df_roofline.empty:
        print("\n📊 Roofline Analysis Results (Target: Raspberry Pi 4B):")
        cols = ['Experiment', 'params_m', 'flops_g', 'size_mb', 'inference_ms',
                'arithmetic_intensity', 'bound_type', 'deployable']
        print(df_roofline[cols].to_string(index=False))

    print_comparison_table(df)

    output_path = os.path.join(base_results, 'comparison_table.csv')
    save_comparison_table(df, output_path)

    lightweight_path = os.path.join(base_results, 'lightweight_status.csv')
    df_lightweight.to_csv(lightweight_path, index=False)
    print(f"✅ Lightweight status saved to {lightweight_path}")

    roofline_path = os.path.join(base_results, 'roofline_analysis.csv')
    if not df_roofline.empty:
        df_roofline.to_csv(roofline_path, index=False)
        print(f"✅ Roofline analysis saved to {roofline_path}")

    if not df.empty and not df_lightweight.empty:
        best_acc = df.loc[df['Accuracy (%)'].idxmax()]
        best_f1 = df.loc[df['F1-Macro'].idxmax()]
        best_efficiency = df.loc[df['Params (M)'].idxmin()]

        print("\n" + "="*60)
        print("  🏆 WINNERS")
        print("="*60)
        print(f"  Best Accuracy  : {best_acc['Experiment']} ({best_acc['Accuracy (%)']:.2f}%)")
        print(f"  Best F1-Macro  : {best_f1['Experiment']} ({best_f1['F1-Macro']:.4f})")
        print(f"  Best Efficiency: {best_efficiency['Experiment']} ({best_efficiency['Params (M)']:.2f} M)")

        gmu_name = 'multimodal_gmu'
        gmu_row = df[df['Experiment'].str.contains(gmu_name, case=False)]
        if not gmu_row.empty:
            gmu = gmu_row.iloc[0]
            gmu_light = df_lightweight[df_lightweight['Experiment'].str.contains(gmu_name, case=False)]
            if not gmu_light.empty:
                overall_pass = gmu_light['All Pass'].values[0]
                status_text = '✅ Memenuhi constraint' if overall_pass else '❌ Melebihi constraint'
            else:
                status_text = '⚠️ Data tidak tersedia'
            
            print("\n  ★ GMU Performance (Best Fold):")
            print(f"    Accuracy : {gmu['Accuracy (%)']:.2f}%")
            print(f"    F1-Macro : {gmu['F1-Macro']:.4f}")
            print(f"    Params   : {gmu['Params (M)']:.2f} M")
            print(f"    FLOPs    : {gmu['FLOPs (G)']:.4f} G")
            print(f"    Size     : {gmu['Model Size (MB)']:.2f} MB")
            print(f"    Status   : {status_text}")
        print("="*60)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/multimodal.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    run_exp7_fusion_comparison(args.config)


if __name__ == '__main__':
    main()