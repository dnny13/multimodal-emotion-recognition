# -*- coding: utf-8 -*-
"""
training/train_unimodal_visual.py
==================================
Training loop untuk unimodal visual dengan progressive unfreeze.
Mendukung dua mode:
1. Internal split (same-actor) untuk backward compatibility.
2. Static split (cross-actor) untuk model selection berdasarkan Actor 17-20.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.validation import validate_epoch
from training.utils import Logger, AverageMeter, save_checkpoint, load_checkpoint, set_seed
from training.split_utils import create_internal_split_from_dataset


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


def apply_progressive_unfreeze(model, epoch, start_epoch, unfreeze_steps):
    if start_epoch <= 0 or unfreeze_steps <= 0:
        return

    if epoch < start_epoch:
        return

    if start_epoch >= unfreeze_steps:
        stage = min((epoch - start_epoch) // (start_epoch // unfreeze_steps), unfreeze_steps)
    else:
        stage = min(epoch - start_epoch, unfreeze_steps)

    stage = max(0, stage)

    if hasattr(model.backbone, 'blocks'):
        blocks = list(model.backbone.blocks.children())
        num_blocks = len(blocks)

        blocks_to_unfreeze = min(num_blocks, int(num_blocks * (stage / unfreeze_steps)))

        for i in range(num_blocks - blocks_to_unfreeze, num_blocks):
            if i >= 0:
                for param in blocks[i].parameters():
                    param.requires_grad = True

        print(f"  Progressive Unfreeze - Epoch {epoch}: {blocks_to_unfreeze}/{num_blocks} blocks unfrozen")
    else:
        for param in model.backbone.parameters():
            param.requires_grad = True
        print(f"  Progressive Unfreeze - Epoch {epoch}: All backbone unfrozen (fallback)")


def train_unimodal_visual(
    model,
    train_dataset,
    val_size=0.2,
    criterion=None,
    optimizer=None,
    scheduler=None,
    device='cuda',
    config=None
):
    """
    Training loop UNIMODAL VISUAL — Internal Split (Same Actors).
    Digunakan untuk backward compatibility atau eksperimen diagnostik.
    """
    if config is None:
        config = {}

    result_path = config.get('result_path', 'results/unimodal_visual')
    store_name = config.get('store_name', 'visual')
    n_epochs = config.get('n_epochs', 60)
    early_stop_patience = config.get('early_stop_patience', 12)
    batch_size = config.get('batch_size', 16)
    n_threads = min(config.get('n_threads', 4), 2)
    manual_seed = config.get('manual_seed', 42)
    grad_clip = config.get('grad_clip', 1.0)
    use_amp = config.get('use_amp', True)

    pu_config = config.get('progressive_unfreeze', {})
    pu_enabled = pu_config.get('enabled', False)
    pu_start_epoch = pu_config.get('start_epoch', 3)
    pu_unfreeze_steps = pu_config.get('unfreeze_steps', 3)

    set_seed(manual_seed, deterministic=config.get('cudnn_deterministic', True))

    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"  GPU Memory sebelum training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    train_subset, val_subset = create_internal_split_from_dataset(
        train_dataset, val_size=val_size, random_state=manual_seed
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        pin_memory=True,
        prefetch_factor=2
    )

    print("\n" + "=" * 60)
    print("  UNIMODAL VISUAL TRAINING — Internal Split (Same Actors)")
    print(f"  Train samples: {len(train_subset)}")
    print(f"  Val samples  : {len(val_subset)}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Workers      : {n_threads}")
    print(f"  Progressive Unfreeze: {'✅' if pu_enabled else '❌'}")
    if pu_enabled:
        print(f"    Start Epoch: {pu_start_epoch}, Unfreeze Steps: {pu_unfreeze_steps}")
    print(f"  Label Smoothing: {'✅' if config.get('use_label_smoothing', False) else '❌'}")
    print(f"  Mixed Precision: {'✅' if use_amp else '❌'}")
    print("=" * 60 + "\n")

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.001)
        )

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

    logger = Logger(
        os.path.join(result_path, 'val.log'),
        header=['epoch', 'loss', 'prec1', 'prec5', 'f1_weighted', 'f1_macro']
    )
    scaler = GradScaler(enabled=use_amp)

    model.to(device)
    best_prec1 = 0.0
    early_stop_counter = 0
    best_epoch = 0

    if pu_enabled and pu_start_epoch <= 0:
        print(f"WARNING: progressive_unfreeze start_epoch={pu_start_epoch} tidak valid. Set ke 1.")
        pu_start_epoch = 1

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            if batch_idx % 10 == 0:
                print(f"  [{epoch:03d}][{batch_idx:03d}/{len(train_loader):03d}] "
                      f"lr:{optimizer.param_groups[0]['lr']:.6f} "
                      f"Loss:{losses.avg:.4f} Acc@1:{top1.avg:.2f}")

        if device == 'cuda':
            torch.cuda.empty_cache()

        if pu_enabled and epoch >= pu_start_epoch:
            apply_progressive_unfreeze(model, epoch, pu_start_epoch, pu_unfreeze_steps)

        val_loss, val_prec1, val_metrics = validate_epoch(
            epoch, val_loader, model, criterion, device, logger
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            early_stop_counter = 0

        checkpoint_state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_prec1': best_prec1,
            'val_metrics': val_metrics
        }
        save_checkpoint(checkpoint_state, is_best, result_path, store_name)

        if is_best:
            print(f"  Best — Epoch {epoch} | Val Acc: {val_prec1:.2f}%")
        else:
            early_stop_counter += 1
            print(f"  No improvement: {early_stop_counter}/{early_stop_patience}")

        if early_stop_counter >= early_stop_patience:
            print(f"\n  Early Stopping — Epoch {epoch}")
            break

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Best Val Acc = {best_prec1:.2f}%")

    best_model_path = os.path.join(result_path, f'{store_name}_best.pth')
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model, device=device)

    return {
        'best_prec1': best_prec1,
        'best_epoch': best_epoch,
        'val_metrics': val_metrics if 'val_metrics' in locals() else None,
        'result_path': result_path
    }


def train_unimodal_visual_static(
    model,
    train_dataset,
    val_loader,
    criterion=None,
    optimizer=None,
    scheduler=None,
    device='cuda',
    config=None
):
    """
    Training loop UNIMODAL VISUAL — Static Split (Cross-Actor).
    Validation loader diberikan dari luar (Actor 17-20).
    Digunakan untuk model selection yang jujur (cross-actor).
    """
    if config is None:
        config = {}

    result_path = config.get('result_path', 'results/unimodal_visual')
    store_name = config.get('store_name', 'visual')
    n_epochs = config.get('n_epochs', 60)
    early_stop_patience = config.get('early_stop_patience', 15)
    batch_size = config.get('batch_size', 16)
    n_threads = min(config.get('n_threads', 4), 2)
    manual_seed = config.get('manual_seed', 42)
    grad_clip = config.get('grad_clip', 1.0)
    use_amp = config.get('use_amp', True)

    pu_config = config.get('progressive_unfreeze', {})
    pu_enabled = pu_config.get('enabled', False)
    pu_start_epoch = pu_config.get('start_epoch', 3)
    pu_unfreeze_steps = pu_config.get('unfreeze_steps', 3)

    set_seed(manual_seed, deterministic=config.get('cudnn_deterministic', True))

    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"  GPU Memory sebelum training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2
    )

    print("\n" + "=" * 60)
    print("  UNIMODAL VISUAL TRAINING — Static Split (Cross-Actor)")
    print(f"  Train samples: {len(train_dataset)} (Actor 1-16)")
    print(f"  Val samples  : {len(val_loader.dataset)} (Actor 17-20)")
    print(f"  Batch size   : {batch_size}")
    print(f"  Workers      : {n_threads}")
    print(f"  Progressive Unfreeze: {'✅' if pu_enabled else '❌'}")
    if pu_enabled:
        print(f"    Start Epoch: {pu_start_epoch}, Unfreeze Steps: {pu_unfreeze_steps}")
    print(f"  Label Smoothing: {'✅' if config.get('use_label_smoothing', False) else '❌'}")
    print(f"  Mixed Precision: {'✅' if use_amp else '❌'}")
    print("=" * 60 + "\n")

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.001)
        )

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

    logger = Logger(
        os.path.join(result_path, 'val.log'),
        header=['epoch', 'loss', 'prec1', 'prec5', 'f1_weighted', 'f1_macro']
    )
    scaler = GradScaler(enabled=use_amp)

    model.to(device)
    best_prec1 = 0.0
    early_stop_counter = 0
    best_epoch = 0

    if pu_enabled and pu_start_epoch <= 0:
        print(f"WARNING: progressive_unfreeze start_epoch={pu_start_epoch} tidak valid. Set ke 1.")
        pu_start_epoch = 1

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            if batch_idx % 10 == 0:
                print(f"  [{epoch:03d}][{batch_idx:03d}/{len(train_loader):03d}] "
                      f"lr:{optimizer.param_groups[0]['lr']:.6f} "
                      f"Loss:{losses.avg:.4f} Acc@1:{top1.avg:.2f}")

        if device == 'cuda':
            torch.cuda.empty_cache()

        if pu_enabled and epoch >= pu_start_epoch:
            apply_progressive_unfreeze(model, epoch, pu_start_epoch, pu_unfreeze_steps)

        val_loss, val_prec1, val_metrics = validate_epoch(
            epoch, val_loader, model, criterion, device, logger
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            early_stop_counter = 0

        checkpoint_state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_prec1': best_prec1,
            'val_metrics': val_metrics
        }
        save_checkpoint(checkpoint_state, is_best, result_path, store_name)

        if is_best:
            print(f"  Best — Epoch {epoch} | Val Acc: {val_prec1:.2f}%")
        else:
            early_stop_counter += 1
            print(f"  No improvement: {early_stop_counter}/{early_stop_patience}")

        if early_stop_counter >= early_stop_patience:
            print(f"\n  Early Stopping — Epoch {epoch}")
            break

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Best Val Acc = {best_prec1:.2f}%")

    best_model_path = os.path.join(result_path, f'{store_name}_best.pth')
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model, device=device)

    return {
        'best_prec1': best_prec1,
        'best_epoch': best_epoch,
        'val_metrics': val_metrics if 'val_metrics' in locals() else None,
        'result_path': result_path
    }


def calculate_accuracy(outputs, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res