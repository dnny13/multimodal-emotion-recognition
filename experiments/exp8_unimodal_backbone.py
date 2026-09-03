# -*- coding: utf-8 -*-
"""
experiments/exp8_unimodal_backbone.py
=====================================
Eksperimen 8: Perbandingan Backbone pada Unimodal (Audio & Visual).

Tujuan:
- Membandingkan performa dan efisiensi backbone lightweight vs non-lightweight.
- Menjawab pertanyaan dosen: "Apakah backbone ringan cukup baik dibanding heavy?"
- Membuktikan konsistensi eksperimen dengan protokol yang identik.

Backbone yang dibandingkan:
1. MobileNetV3-Small  : 2.5M params, 0.06 GFLOPs → Ultra-light
2. EfficientNetV2-B0  : 7.1M params, 1.46 GFLOPs → Lightweight (pilihan utama)
3. ResNet50           : 25.6M params, 4.1 GFLOPs → Heavyweight (kontrol)

🔧 FIX (v11):
  - Perbaiki nama backbone: 'mobilenetv3_small_100' (bukan 'mobilenet_v3_small').
  - Menggunakan get_audio_transforms() yang sudah diperbaiki di data/transforms.py.
"""

import os
import sys
import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RavdessAudio, RavdessVisual, get_audio_transforms, get_visual_transforms
from models import UnimodalAudioModel, UnimodalVisualModel
from training import train_unimodal_audio, train_unimodal_visual
from training.utils import compute_class_weights, load_checkpoint
from evaluation import evaluate_test_set, roofline_analysis, save_test_metrics, check_lightweight_constraints
from configs import load_config


def run_exp8_unimodal_backbone(config_path=None, modality='audio', backbones=None, device='cuda'):
    """
    Menjalankan perbandingan backbone untuk unimodal (audio atau visual).
    
    Args:
        config_path (str): Path ke file config YAML.
        modality (str): 'audio' atau 'visual'.
        backbones (list): Daftar backbone yang akan diuji.
        device (str): 'cuda' atau 'cpu'.
    
    Returns:
        pd.DataFrame: Tabel hasil perbandingan.
    """
    # 🔧 PERBAIKAN: Nama backbone yang benar
    if backbones is None:
        backbones = ['mobilenetv3_small_100', 'efficientnetv2_b0', 'resnet50']
    
    config = load_config(config_path or f'configs/unimodal_{modality}.yaml')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device.upper()}")
    print(f"[INFO] Modality: {modality.upper()}")
    print(f"[INFO] Backbones to test: {backbones}")
    
    results = []
    base_results_dir = config['paths']['results']
    
    for backbone_name in backbones:
        print("\n" + "="*60)
        print(f"  TESTING {modality.upper()} BACKBONE: {backbone_name}")
        print("="*60)
        
        # Update config untuk backbone tertentu
        config['model']['backbone'] = backbone_name
        
        # Path hasil eksperimen
        result_subdir = f'unimodal_{modality}_backbone_{backbone_name}'
        result_path = os.path.join(base_results_dir, result_subdir)
        os.makedirs(result_path, exist_ok=True)
        
        # -------------------- Data & Dataloader --------------------
        if modality == 'audio':
            # Untuk backbone 2D (ResNet50, MobileNet), gunakan mode='2d'
            audio_mode = '2d' if backbone_name in ['resnet50', 'mobilenetv3_small_100'] else '1d'
            
            transform_train = get_audio_transforms(
                is_training=True,
                spec_size=(224, 224),
                use_spec_augment=config['data']['augmentation']['spec_augment']['enabled'],
                mode=audio_mode
            )
            transform_val = get_audio_transforms(
                is_training=False,
                spec_size=(224, 224),
                use_spec_augment=False,
                mode=audio_mode
            )
            
            train_dataset = RavdessAudio(
                config['paths']['annotations'],
                config['paths']['output_audio'],
                'training',
                transform_train,
                backbone_name=backbone_name,
                input_mode=audio_mode
            )
            val_dataset = RavdessAudio(
                config['paths']['annotations'],
                config['paths']['output_audio'],
                'validation',
                transform_val,
                backbone_name=backbone_name,
                input_mode=audio_mode
            )
            test_dataset = RavdessAudio(
                config['paths']['annotations'],
                config['paths']['output_audio'],
                'testing',
                transform_val,
                backbone_name=backbone_name,
                input_mode=audio_mode
            )
            
            ModelClass = UnimodalAudioModel
            train_func = train_unimodal_audio
            model_type = 'audio'
            
        else:  # visual
            transform_train = get_visual_transforms(
                is_training=True,
                image_size=(224, 224)
            )
            transform_val = get_visual_transforms(
                is_training=False,
                image_size=(224, 224)
            )
            
            train_dataset = RavdessVisual(
                config['paths']['annotations'],
                config['paths']['output_visual'],
                'training',
                transform_train,
                num_frames=config['data']['visual']['num_frames']
            )
            val_dataset = RavdessVisual(
                config['paths']['annotations'],
                config['paths']['output_visual'],
                'validation',
                transform_val,
                num_frames=config['data']['visual']['num_frames']
            )
            test_dataset = RavdessVisual(
                config['paths']['annotations'],
                config['paths']['output_visual'],
                'testing',
                transform_val,
                num_frames=config['data']['visual']['num_frames']
            )
            
            ModelClass = UnimodalVisualModel
            train_func = train_unimodal_visual
            model_type = 'visual'
        
        print(f"\n📊 Data Summary ({backbone_name}):")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val  : {len(val_dataset)} samples (Actor 17-20, cross-actor)")
        print(f"  Test : {len(test_dataset)} samples (Actor 21-24)")
        
        # -------------------- Model --------------------
        model = ModelClass(
            num_classes=config['dataset']['num_classes'],
            backbone_name=backbone_name,
            dropout=config['model']['dropout'],
            pretrained=config['model']['pretrained'],
            freeze_backbone=config['model']['freeze_backbone'],
            classifier_hidden=config['model'].get('classifier_hidden', 512)
        ).to(device)
        
        print(f"\n📐 Model Summary ({backbone_name}):")
        print(f"  Backbone       : {backbone_name}")
        print(f"  Feature Dim    : {model.get_feature_dim()}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total Params   : {total_params:,} ({total_params/1e6:.2f} M)")
        
        # -------------------- Loss & Optimizer --------------------
        class_weights = compute_class_weights(
            config['paths']['annotations'],
            config['dataset']['num_classes']
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['cosine_t_max'],
            eta_min=config['training']['cosine_eta_min']
        )
        
        # -------------------- Training --------------------
        train_config = {
            'result_path': result_path,
            'store_name': f'{model_type}_{backbone_name}',
            'n_epochs': config['training']['n_epochs'],
            'early_stop_patience': config['training']['early_stop_patience'],
            'batch_size': config['training']['batch_size'],
            'n_threads': config['training']['n_threads'],
            'manual_seed': config['training'].get('manual_seed', 42),
            'grad_clip': config['training'].get('grad_clip', 1.0),
            'use_amp': config['training'].get('use_amp', True)
        }
        
        print(f"\n🚀 Training {backbone_name}...")
        start_time = time.time()
        
        if modality == 'audio':
            train_result = train_unimodal_audio(
                model=model,
                train_dataset=train_dataset,
                val_size=0.2,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                config=train_config
            )
        else:
            train_result = train_unimodal_visual(
                model=model,
                train_dataset=train_dataset,
                val_size=0.2,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                config=train_config
            )
        
        training_time = time.time() - start_time
        best_val_acc = train_result.get('best_prec1', 0)
        print(f"✅ Training selesai. Best Val Acc: {best_val_acc:.2f}% (Time: {training_time/60:.2f} min)")
        
        # -------------------- Evaluasi Test Set --------------------
        best_model_path = os.path.join(result_path, f'{model_type}_{backbone_name}_best.pth')
        test_acc = 0.0
        test_f1 = 0.0
        
        if os.path.exists(best_model_path):
            model_eval = ModelClass(
                num_classes=config['dataset']['num_classes'],
                backbone_name=backbone_name,
                dropout=config['model']['dropout'],
                classifier_hidden=config['model'].get('classifier_hidden', 512)
            ).to(device)
            load_checkpoint(best_model_path, model_eval, device=device)
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['training']['n_threads'],
                pin_memory=True
            )
            
            print(f"\n📊 Evaluasi Test Set ({backbone_name}):")
            
            eval_result = evaluate_test_set(
                model=model_eval,
                model_type=model_type,
                test_loader=test_loader,
                device=device,
                num_classes=config['dataset']['num_classes']
            )
            
            test_metrics = eval_result['metrics']
            test_acc = test_metrics['accuracy'] * 100
            test_f1 = test_metrics['f1_macro']
            
            save_test_metrics(result_path, test_metrics)
            
            print(f"  Test Accuracy      : {test_acc:.2f}%")
            print(f"  Test F1-Macro      : {test_f1:.4f}")
            
            try:
                if modality == 'audio':
                    input_shape = (1, 120, 224) if backbone_name == 'audio_1dcnn' else (1, 3, 224, 224)
                else:
                    input_shape = (1, 3, 224, 224)
                roofline_analysis(model_eval, f'Unimodal {modality} {backbone_name}', device, is_multimodal=False)
            except Exception as e:
                print(f"⚠️ Roofline analysis gagal: {e}")
        else:
            print(f"⚠️ Best model tidak ditemukan: {best_model_path}")
        
        # -------------------- Efisiensi & Constraint --------------------
        try:
            from evaluation.efficiency import compute_flops, measure_inference_time
            if modality == 'audio' and backbone_name == 'audio_1dcnn':
                input_shape = (1, 120, 224)
            else:
                input_shape = (1, 3, 224, 224)
            flops_g = compute_flops(model, input_shape, device)
            inf_time = measure_inference_time(model, input_shape, device)
        except Exception as e:
            print(f"⚠️ Gagal menghitung FLOPs/Inference: {e}")
            flops_g = 0.0
            inf_time = 0.0
        
        size_mb = total_params * 4 / (1024 * 1024)
        
        status = check_lightweight_constraints(total_params/1e6, flops_g, size_mb, inf_time, verbose=False)
        all_pass = all(c['pass'] for c in status.values())
        
        results.append({
            'Modality': modality.upper(),
            'Backbone': backbone_name,
            'Params (M)': total_params / 1e6,
            'FLOPs (G)': flops_g,
            'Size (MB)': size_mb,
            'Inference (ms)': inf_time,
            'Best Val Acc (%)': best_val_acc,
            'Test Acc (%)': test_acc,
            'Test F1-Macro': test_f1,
            'Lightweight': '✅' if all_pass else '❌',
            'Training Time (min)': training_time / 60,
            'Result Path': result_path
        })
        
        print(f"\n📊 Lightweight Status ({backbone_name}): {'✅ PASS' if all_pass else '❌ FAIL'}")
    
    # -------------------- Ringkasan Perbandingan --------------------
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print(f"  BACKBONE COMPARISON SUMMARY — {modality.upper()}")
    print("="*80)
    print(df_results[['Backbone', 'Params (M)', 'FLOPs (G)', 'Size (MB)', 
                       'Test Acc (%)', 'Test F1-Macro', 'Lightweight']].to_string(index=False))
    print("="*80)
    print("  ✅ = Memenuhi constraint lightweight (<12M, <2G, <48MB, <100ms)")
    print("  ❌ = Tidak memenuhi constraint")
    print("="*80)
    
    summary_path = os.path.join(base_results_dir, f'unimodal_{modality}_backbone_comparison.csv')
    df_results.to_csv(summary_path, index=False)
    print(f"\n✅ Hasil perbandingan disimpan ke: {summary_path}")
    
    return df_results


def main():
    parser = argparse.ArgumentParser(description='Perbandingan Backbone Unimodal (Audio/Visual)')
    parser.add_argument('--modality', type=str, default='audio', choices=['audio', 'visual'],
                        help='Pilih modalitas: audio atau visual')
    parser.add_argument('--config', type=str, default=None,
                        help='Path ke file config YAML')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda atau cpu')
    parser.add_argument('--backbones', type=str, nargs='+', 
                        default=['mobilenetv3_small_100', 'efficientnetv2_b0', 'resnet50'],
                        help='Daftar backbone yang akan diuji')
    args = parser.parse_args()
    
    run_exp8_unimodal_backbone(
        config_path=args.config,
        modality=args.modality,
        backbones=args.backbones,
        device=args.device
    )


if __name__ == '__main__':
    main()