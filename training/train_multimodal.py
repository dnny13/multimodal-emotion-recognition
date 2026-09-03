# -*- coding: utf-8 -*-
"""
training/train_multimodal.py
============================
Training multimodal dengan Focal Loss, MixUp, Modality Dropout, dan OneCycleLR.

🔧 FIX: Tambahkan proyeksi audio dan visual sebelum fusion agar dimensi cocok.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .utils import Logger, AverageMeter, save_checkpoint, calculate_accuracy, set_seed
from .validation import validate_epoch_with_cache
from models import MultimodalFusionModel


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def mixup_data(audio, visual, target, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = audio.size(0)
    index = torch.randperm(batch_size).to(audio.device)
    mixed_audio = lam * audio + (1 - lam) * audio[index, :]
    mixed_visual = lam * visual + (1 - lam) * visual[index, :]
    target_a, target_b = target, target[index]
    return mixed_audio, mixed_visual, target_a, target_b, lam


def mixup_criterion(criterion, pred, target_a, target_b, lam):
    return lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)


def apply_modality_dropout(audio_emb, visual_emb, labels, dropout_prob=0.25):
    batch_size = audio_emb.size(0)
    device = audio_emb.device

    drop_mask = torch.rand(batch_size, device=device) < dropout_prob
    drop_audio_mask = drop_mask & (torch.rand(batch_size, device=device) < 0.5)
    drop_visual_mask = drop_mask & (~drop_audio_mask)

    audio_emb = audio_emb.clone()
    visual_emb = visual_emb.clone()

    if drop_audio_mask.any():
        audio_emb[drop_audio_mask] = 0.0
    if drop_visual_mask.any():
        visual_emb[drop_visual_mask] = 0.0

    return audio_emb, visual_emb, labels


def embedding_augmentation(audio_emb, visual_emb, noise_std=0.01, feature_dropout=0.05):
    if noise_std > 0:
        audio_emb = audio_emb + torch.randn_like(audio_emb) * noise_std
        visual_emb = visual_emb + torch.randn_like(visual_emb) * noise_std
    if feature_dropout > 0:
        mask = torch.rand(audio_emb.size(), device=audio_emb.device) > feature_dropout
        audio_emb = audio_emb * mask.float()
        mask = torch.rand(visual_emb.size(), device=visual_emb.device) > feature_dropout
        visual_emb = visual_emb * mask.float()
    return audio_emb, visual_emb


def train_multimodal_fold(
    fusion_model,
    train_embeds,
    val_embeds,
    criterion,
    optimizer,
    scheduler,
    device,
    fold_idx,
    config,
    cross_val_loader=None
):
    manual_seed = config.get('manual_seed', 42)
    set_seed(manual_seed + fold_idx, deterministic=config.get('cudnn_deterministic', True))

    base_result_path = config.get('result_path', 'results/multimodal')
    result_path = os.path.join(base_result_path, f'fold_{fold_idx}')
    store_name = config.get('store_name', f'multimodal_fold_{fold_idx}')
    n_epochs = config.get('n_epochs', 30)
    early_stop_patience = config.get('early_stop_patience', 5)
    batch_size = config.get('batch_size', 32)
    use_mixup = config.get('use_mixup', True)
    mixup_alpha = config.get('mixup_alpha', 0.2)
    use_modality_dropout = config.get('use_modality_dropout', True)
    modality_dropout_prob = config.get('modality_dropout_prob', 0.25)
    embed_noise_std = config.get('embed_noise_std', 0.01)
    embed_feature_dropout = config.get('embed_feature_dropout', 0.05)

    os.makedirs(result_path, exist_ok=True)

    train_logger = Logger(os.path.join(result_path, 'train.log'), ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
    val_logger = Logger(os.path.join(result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'prec5', 'f1_weighted', 'f1_macro'])

    if cross_val_loader is not None:
        cross_val_logger = Logger(os.path.join(result_path, 'val_cross.log'), 
                                  ['epoch', 'loss', 'prec1', 'prec5', 'f1_weighted', 'f1_macro'])
    else:
        cross_val_logger = None
        print("⚠️  PERINGATAN: cross_val_loader tidak diberikan. Early stopping akan menggunakan same-actor.")

    train_dataset = TensorDataset(
        torch.FloatTensor(train_embeds['audio']),
        torch.FloatTensor(train_embeds['visual']),
        torch.LongTensor(train_embeds['labels'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_embeds['audio']),
        torch.FloatTensor(val_embeds['visual']),
        torch.LongTensor(val_embeds['labels'])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        try:
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_prec1 = 0.0
    best_same_prec1 = 0.0
    best_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    val_metrics = {}

    print(f"\n{'='*60}")
    print(f"  MULTIMODAL TRAINING — FOLD {fold_idx+1}/5")
    print(f"  Fusion Type : {config.get('fusion_type', 'gmu')}")
    print(f"  MixUp       : {'✅' if use_mixup else '❌'}")
    print(f"  Focal Loss  : {'✅' if isinstance(criterion, FocalLoss) else '❌'}")
    print(f"  Modality Dropout: {'✅' if use_modality_dropout else '❌'} (p={modality_dropout_prob})")
    print(f"  Embedding Aug : {'✅' if embed_noise_std > 0 or embed_feature_dropout > 0 else '❌'}")
    print(f"  Cross-actor Val : {'✅' if cross_val_loader is not None else '❌'}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        fusion_model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        current_lr = optimizer.param_groups[0]['lr']

        for batch_idx, batch in enumerate(train_loader):
            audio_emb, visual_emb, targets = batch
            audio_emb = audio_emb.to(device)
            visual_emb = visual_emb.to(device)
            targets = targets.to(device)

            # 🔧 FIX: Embedding augmentation
            if embed_noise_std > 0 or embed_feature_dropout > 0:
                audio_emb, visual_emb = embedding_augmentation(
                    audio_emb, visual_emb,
                    noise_std=embed_noise_std,
                    feature_dropout=embed_feature_dropout
                )

            if use_modality_dropout and epoch > 2:
                audio_emb, visual_emb, targets = apply_modality_dropout(
                    audio_emb, visual_emb, targets, dropout_prob=modality_dropout_prob
                )

            if use_mixup and epoch > 2:
                audio_emb, visual_emb, targets_a, targets_b, lam = mixup_data(
                    audio_emb, visual_emb, targets, alpha=mixup_alpha
                )

            optimizer.zero_grad(set_to_none=True)

            # 🔧 PERBAIKAN: Proyeksi sebelum fusion
            audio_emb_proj = fusion_model.audio_projector(audio_emb)
            visual_emb_proj = fusion_model.visual_projector(visual_emb)

            if scaler is not None:
                try:
                    with torch.amp.autocast('cuda'):
                        fused = fusion_model.fusion(audio_emb_proj, visual_emb_proj)
                        outputs = fusion_model.classifier(fused)
                        if use_mixup and epoch > 2:
                            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                        else:
                            loss = criterion(outputs, targets)
                except AttributeError:
                    with torch.cuda.amp.autocast():
                        fused = fusion_model.fusion(audio_emb_proj, visual_emb_proj)
                        outputs = fusion_model.classifier(fused)
                        if use_mixup and epoch > 2:
                            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                        else:
                            loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                fused = fusion_model.fusion(audio_emb_proj, visual_emb_proj)
                outputs = fusion_model.classifier(fused)
                if use_mixup and epoch > 2:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
                optimizer.step()

            if use_mixup and epoch > 2:
                prec1, prec5 = calculate_accuracy(outputs.data, targets_a.data, topk=(1, 5))
            else:
                prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))

            losses.update(loss.item(), audio_emb.size(0))
            top1.update(prec1, audio_emb.size(0))
            top5.update(prec5, audio_emb.size(0))

            current_lr = optimizer.param_groups[0]['lr']
            if epoch % 5 == 0 and batch_idx % 10 == 0:
                print(f' [{epoch}][{batch_idx:>3}/{len(train_loader)}] '
                      f'lr:{current_lr:.6f} Loss:{losses.avg:.4f} Acc@1:{top1.avg:.2f}')

        train_logger.log({
            'epoch': epoch, 'loss': losses.avg, 'prec1': top1.avg,
            'prec5': top5.avg, 'lr': current_lr
        })

        # Same-Actor Validation (sudah menggunakan proyeksi di dalam validate_epoch_with_cache)
        val_loss_same, val_prec1_same, val_metrics_same = validate_epoch_with_cache(
            epoch, val_loader, fusion_model, criterion, device, val_logger
        )

        # Cross-Actor Validation
        if cross_val_loader is not None:
            val_loss_cross, val_prec1_cross, val_metrics_cross = validate_epoch_with_cache(
                epoch, cross_val_loader, fusion_model, criterion, device, cross_val_logger
            )
            current_best_candidate = val_prec1_cross
            f1_cross = val_metrics_cross.get('f1_macro', 0.0)
            val_metrics = val_metrics_cross
        else:
            print("⚠️  Tidak ada cross_val_loader, menggunakan same-actor untuk early stopping.")
            current_best_candidate = val_prec1_same
            f1_cross = val_metrics_same.get('f1_macro', 0.0)
            val_metrics = val_metrics_same

        is_best = current_best_candidate > best_prec1
        if is_best:
            best_prec1 = current_best_candidate
            best_same_prec1 = val_prec1_same
            best_f1 = f1_cross
            best_epoch = epoch
            epochs_no_improve = 0
            if cross_val_loader is not None:
                print(f"  ✅ Best (cross) — Epoch {epoch} | Cross-Acc: {val_prec1_cross:.2f}% | Same-Acc: {val_prec1_same:.2f}% | F1: {best_f1:.4f}")
            else:
                print(f"  ✅ Best (same) — Epoch {epoch} | Same-Acc: {val_prec1_same:.2f}% | F1: {best_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  → No improvement: {epochs_no_improve}/{early_stop_patience}")

        save_checkpoint({
            'epoch': epoch,
            'state_dict': fusion_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1,
            'best_same_prec1': best_same_prec1,
            'best_f1': best_f1,
            'val_metrics': val_metrics
        }, is_best, result_path, store_name)

        if scheduler is not None:
            scheduler.step()

        if epochs_no_improve >= early_stop_patience:
            print(f"\n  [!] Early Stopping — Fold {fold_idx+1}, Epoch {epoch}")
            break

    return {
        'best_prec1': best_prec1,
        'best_same_prec1': best_same_prec1,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'result_path': result_path,
        'model': fusion_model,
        'val_metrics': val_metrics
    }