# -*- coding: utf-8 -*-
"""
experiments/exp6_backbone_comparison.py
=======================================
Eksperimen 6: Perbandingan Backbone Lightweight vs Heavyweight (UNIMODAL)
-- VERSI HEMAT WAKTU & MEMORI --

🔧 FIX (v11):
  - Hapus blok tuple check yang tidak diperlukan.
  - Perbaiki audio heavyweight: gunakan 'audio_2dcnn' (bukan 'resnet50' yang salah).
  - Nama backbone MobileNetV3-Small: 'mobilenetv3_small_100' (konsisten dengan backbone.py).
  - Tambahkan AMP fallback.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import RavdessMultimodal, get_multimodal_transforms
from models import UnimodalVisualModel, UnimodalAudioModel
from training.utils import compute_class_weights, save_checkpoint
from configs import load_config


# ============================================================
# 1. Training Loop dengan AMP (dengan fallback)
# ============================================================
def train_unimodal(model, train_loader, val_loader, criterion, optimizer, scheduler,
                   device, n_epochs, result_path, store_name, early_stop_patience=10):
    """Training loop untuk unimodal dengan AMP dan fallback."""
    if not isinstance(device, torch.device):
        device = torch.device(device)
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    use_amp = device.type == 'cuda'
    if use_amp:
        try:
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            audio, visual, labels = batch
            audio, visual, labels = audio.to(device), visual.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                try:
                    with torch.amp.autocast('cuda'):
                        if isinstance(model, UnimodalVisualModel):
                            outputs = model(visual)
                        else:
                            outputs = model(audio)
                        loss = criterion(outputs, labels)
                except AttributeError:
                    with torch.cuda.amp.autocast():
                        if isinstance(model, UnimodalVisualModel):
                            outputs = model(visual)
                        else:
                            outputs = model(audio)
                        loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if isinstance(model, UnimodalVisualModel):
                    outputs = model(visual)
                else:
                    outputs = model(audio)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * audio.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                audio, visual, labels = batch
                audio, visual, labels = audio.to(device), visual.to(device), labels.to(device)

                if use_amp:
                    try:
                        with torch.amp.autocast('cuda'):
                            if isinstance(model, UnimodalVisualModel):
                                outputs = model(visual)
                            else:
                                outputs = model(audio)
                            loss = criterion(outputs, labels)
                    except AttributeError:
                        with torch.cuda.amp.autocast():
                            if isinstance(model, UnimodalVisualModel):
                                outputs = model(visual)
                            else:
                                outputs = model(audio)
                            loss = criterion(outputs, labels)
                else:
                    if isinstance(model, UnimodalVisualModel):
                        outputs = model(visual)
                    else:
                        outputs = model(audio)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * audio.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = correct / total

        scheduler.step()

        print(f"Epoch {epoch:3d}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_val_acc,
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, is_best, result_path, f'{store_name}.pth')

        if is_best:
            print(f"  ✅ Best model saved (Acc: {best_val_acc:.4f})")
        else:
            print(f"  → No improvement: {patience_counter}/{early_stop_patience}")

        if patience_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    return best_val_acc, best_epoch


# ============================================================
# 2. Load hasil lightweight dari Exp 1 & 2
# ============================================================
def load_lightweight_metrics():
    """Memuat metrik dari unimodal audio dan visual yang sudah dilatih."""
    results = {}

    vis_metrics_path = 'results/unimodal_visual/test_metrics.json'
    if os.path.exists(vis_metrics_path):
        with open(vis_metrics_path, 'r') as f:
            data = json.load(f)
        results['visual_light'] = {
            'modality': 'Visual',
            'backbone': 'EfficientNetV2-B0',
            'category': 'Lightweight',
            'params_m': 6.52,
            'test_acc': data.get('accuracy', 0) * 100,
            'f1_macro': data.get('f1_macro', 0),
        }
        print(f"✅ Loaded Visual Lightweight: Acc={results['visual_light']['test_acc']:.2f}%")
    else:
        print("⚠️ Visual lightweight results not found! Run Exp 1 first.")
        return None

    aud_metrics_path = 'results/unimodal_audio/test_metrics.json'
    if os.path.exists(aud_metrics_path):
        with open(aud_metrics_path, 'r') as f:
            data = json.load(f)
        results['audio_light'] = {
            'modality': 'Audio',
            'backbone': '1D-CNN',
            'category': 'Lightweight',
            'params_m': 1.09,
            'test_acc': data.get('accuracy', 0) * 100,
            'f1_macro': data.get('f1_macro', 0),
        }
        print(f"✅ Loaded Audio Lightweight: Acc={results['audio_light']['test_acc']:.2f}%")
    else:
        print("⚠️ Audio lightweight results not found! Run Exp 2 first.")
        return None

    return results


# ============================================================
# 3. Training Heavyweight (dengan batch size kecil & AMP)
# ============================================================
def train_heavyweight_models(config, device):
    """Melatih ResNet50 untuk visual dan Audio2DCNN untuk audio."""
    if not isinstance(device, torch.device):
        device = torch.device(device)

    annotation_path = config['paths']['annotations']
    audio_root = config['paths']['output_audio']
    visual_root = config['paths']['output_visual']
    result_base = os.path.join(config['paths']['results'], 'backbone_comparison_v2')
    os.makedirs(result_base, exist_ok=True)

    num_classes = config['dataset']['num_classes']
    n_epochs = 30
    n_threads = min(config['training'].get('n_threads', 2), 2)
    heavy_batch_size = 4

    heavy_results = {}

    # ============================================================
    # 3a. VISUAL HEAVYWEIGHT: ResNet50
    # ============================================================
    print("\n" + "="*60)
    print("  VISUAL HEAVYWEIGHT: ResNet50 (25.6M params)")
    print("="*60)

    result_path = os.path.join(result_base, 'visual_resnet50')
    os.makedirs(result_path, exist_ok=True)

    _, transform_visual_train = get_multimodal_transforms(
        is_training=True, audio_size=(224,224), visual_size=(224,224),
        use_spec_augment=True, audio_mode='1d'
    )
    _, transform_visual_val = get_multimodal_transforms(
        is_training=False, audio_size=(224,224), visual_size=(224,224),
        use_spec_augment=False, audio_mode='1d'
    )

    train_dataset = RavdessMultimodal(
        annotation_path=annotation_path, audio_root=audio_root, visual_root=visual_root,
        subset='training', transform_audio=None, transform_visual=transform_visual_train,
        num_frames=config['data']['visual']['num_frames']
    )
    val_dataset = RavdessMultimodal(
        annotation_path=annotation_path, audio_root=audio_root, visual_root=visual_root,
        subset='validation', transform_audio=None, transform_visual=transform_visual_val,
        num_frames=config['data']['visual']['num_frames']
    )

    train_loader = DataLoader(train_dataset, batch_size=heavy_batch_size, shuffle=True,
                              num_workers=n_threads, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=heavy_batch_size, shuffle=False,
                            num_workers=n_threads, pin_memory=True)

    # 🔧 PERBAIKAN: backbone_name = 'resnet50' (sudah benar untuk visual)
    model = UnimodalVisualModel(
        num_classes=num_classes,
        backbone_name='resnet50',
        dropout=0.5
    ).to(device)

    # 🔧 PERBAIKAN: Hapus blok tuple check (tidak diperlukan)
    print(f"  Params      : {sum(p.numel() for p in model.parameters()):,}")

    class_weights = compute_class_weights(annotation_path, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                                  weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_val_acc, _ = train_unimodal(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, n_epochs=n_epochs, result_path=result_path,
        store_name='visual_resnet50', early_stop_patience=5
    )

    heavy_results['visual_heavy'] = {
        'modality': 'Visual',
        'backbone': 'ResNet50',
        'category': 'Heavyweight',
        'params_m': sum(p.numel() for p in model.parameters()) / 1e6,
        'best_val_acc': best_val_acc,
    }

    # ============================================================
    # 3b. AUDIO HEAVYWEIGHT: Audio2DCNN (custom 2D CNN)
    # 🔧 PERBAIKAN: Gunakan 'audio_2dcnn' yang merupakan 2D CNN sesuai untuk Mel-Spectrogram
    # ============================================================
    print("\n" + "="*60)
    print("  AUDIO HEAVYWEIGHT: Audio2DCNN (custom 2D CNN, ~1.2M params)")
    print("="*60)

    result_path = os.path.join(result_base, 'audio_2dcnn')
    os.makedirs(result_path, exist_ok=True)

    transform_audio_train, _ = get_multimodal_transforms(
        is_training=True, audio_size=(224,224), visual_size=(224,224),
        use_spec_augment=True, audio_mode='2d'
    )
    transform_audio_val, _ = get_multimodal_transforms(
        is_training=False, audio_size=(224,224), visual_size=(224,224),
        use_spec_augment=False, audio_mode='2d'
    )

    train_dataset = RavdessMultimodal(
        annotation_path=annotation_path, audio_root=audio_root, visual_root=visual_root,
        subset='training', transform_audio=transform_audio_train, transform_visual=None,
        num_frames=config['data']['visual']['num_frames']
    )
    val_dataset = RavdessMultimodal(
        annotation_path=annotation_path, audio_root=audio_root, visual_root=visual_root,
        subset='validation', transform_audio=transform_audio_val, transform_visual=None,
        num_frames=config['data']['visual']['num_frames']
    )

    train_loader = DataLoader(train_dataset, batch_size=heavy_batch_size, shuffle=True,
                              num_workers=n_threads, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=heavy_batch_size, shuffle=False,
                            num_workers=n_threads, pin_memory=True)

    # 🔧 PERBAIKAN: Gunakan 'audio_2dcnn' (bukan 'resnet50')
    model = UnimodalAudioModel(
        num_classes=num_classes,
        backbone_name='audio_2dcnn',
        dropout=0.5
    ).to(device)

    print(f"  Params      : {sum(p.numel() for p in model.parameters()):,}")

    class_weights = compute_class_weights(annotation_path, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                                  weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_val_acc, _ = train_unimodal(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, n_epochs=n_epochs, result_path=result_path,
        store_name='audio_2dcnn', early_stop_patience=5
    )

    heavy_results['audio_heavy'] = {
        'modality': 'Audio',
        'backbone': 'Audio2DCNN',
        'category': 'Heavyweight',
        'params_m': sum(p.numel() for p in model.parameters()) / 1e6,
        'best_val_acc': best_val_acc,
    }

    return heavy_results


# ============================================================
# 4. Main
# ============================================================
def run_exp6_backbone_comparison(config_path=None, override=None):
    config = load_config(config_path or 'configs/multimodal.yaml')
    if override:
        config.update(override)

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if not isinstance(device, torch.device):
        device = torch.device(device)
    print(f"[INFO] Device: {device}")

    print("\n📂 Mengumpulkan metrik dari model lightweight (sudah ada)...")
    all_results = load_lightweight_metrics()
    if all_results is None:
        print("❌ Gagal memuat hasil lightweight. Pastikan Exp 1 & 2 sudah dijalankan.")
        return

    print("\n🏋️ Melatih model heavyweight (ResNet50 visual & Audio2DCNN)...")
    heavy_results = train_heavyweight_models(config, device)

    all_results.update(heavy_results)

    print("\n" + "="*60)
    print("  BACKBONE COMPARISON SUMMARY (UNIMODAL)")
    print("="*60)
    print(f"  {'Modality':<12} {'Backbone':<25} {'Params (M)':<12} {'Accuracy (%)':<12} {'Category'}")
    print("-"*60)

    df_rows = []
    for key, data in all_results.items():
        if 'light' in key:
            acc = data.get('test_acc', 0)
        else:
            acc = data.get('best_val_acc', 0) * 100
        print(f"  {data['modality']:<12} {data['backbone']:<25} {data['params_m']:<12.2f} {acc:<12.2f} {data['category']}")
        df_rows.append({
            'modality': data['modality'],
            'backbone': data['backbone'],
            'params_m': data['params_m'],
            'accuracy': acc,
            'category': data['category']
        })

    print("="*60)

    import pandas as pd
    df = pd.DataFrame(df_rows)
    result_base = os.path.join(config['paths']['results'], 'backbone_comparison_v2')
    os.makedirs(result_base, exist_ok=True)
    df.to_csv(os.path.join(result_base, 'backbone_comparison_summary.csv'), index=False)
    print(f"\n✅ Summary saved to {os.path.join(result_base, 'backbone_comparison_summary.csv')}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/multimodal.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run_exp6_backbone_comparison(args.config, {'device': args.device})

if __name__ == '__main__':
    main()