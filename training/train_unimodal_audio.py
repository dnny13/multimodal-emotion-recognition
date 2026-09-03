# -*- coding: utf-8 -*-
"""
training/train_unimodal_audio.py
=================================
Training loop untuk unimodal audio dengan mixed precision & gradient clipping.
Mendukung dua mode:
1. Internal split (same-actor) untuk backward compatibility.
2. Static split (cross-actor) untuk model selection berdasarkan Actor 17-20.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from .utils import Logger, AverageMeter, save_checkpoint, calculate_accuracy, set_seed
from .validation import validate_epoch


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


def train_unimodal_audio(
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
    Training loop UNIMODAL AUDIO — Internal Split (Same Actors).
    Digunakan untuk backward compatibility atau eksperimen diagnostik.
    """
    if config is None:
        config = {}

    result_path = config.get('result_path', 'results/unimodal_audio')
    store_name = config.get('store_name', 'audio')
    n_epochs = config.get('n_epochs', 80)
    early_stop_patience = config.get('early_stop_patience', 10)
    batch_size = config.get('batch_size', 64)
    n_threads = config.get('n_threads', 2)
    manual_seed = config.get('manual_seed', 42)
    grad_clip = config.get('grad_clip', 1.0)
    use_amp = config.get('use_amp', True)

    set_seed(manual_seed, deterministic=config.get('cudnn_deterministic', True))

    os.makedirs(result_path, exist_ok=True)

    labels = train_dataset.get_labels()
    indices = np.arange(len(train_dataset))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=manual_seed)
    train_idx, val_idx = next(sss.split(indices, labels))

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        pin_memory=True
    )

    train_logger = Logger(os.path.join(result_path, 'train.log'), ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
    val_logger = Logger(os.path.join(result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'prec5', 'f1_weighted', 'f1_macro'])

    if use_amp and torch.cuda.is_available():
        try:
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_prec1 = 0.0
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"  UNIMODAL AUDIO TRAINING — Internal Split (Same Actors)")
    print(f"  Train samples: {len(train_subset)}")
    print(f"  Val samples  : {len(val_subset)}")
    print(f"  Mixed Precision: {'YES' if use_amp else 'NO'}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                try:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                except AttributeError:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            losses.update(loss.item(), inputs.size(0))
            prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

        current_lr = optimizer.param_groups[0]['lr']
        train_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg,
            'lr': current_lr
        })

        val_loss, val_prec1, val_metrics = validate_epoch(epoch, val_loader, model, criterion, device, val_logger)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        is_best = val_prec1 > best_prec1
        best_prec1 = max(best_prec1, val_prec1)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1,
            'val_metrics': val_metrics
        }, is_best, result_path, store_name)

        if is_best:
            epochs_no_improve = 0
            print(f"  Best — Epoch {epoch} | Val Acc: {val_prec1:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"  No improvement: {epochs_no_improve}/{early_stop_patience}")

        if epochs_no_improve >= early_stop_patience:
            print(f"\n  Early Stopping — Epoch {epoch}")
            break

    return {
        'best_prec1': best_prec1,
        'result_path': result_path,
        'model': model
    }


def train_unimodal_audio_static(
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
    Training loop UNIMODAL AUDIO — Static Split (Cross-Actor).
    Validation loader diberikan dari luar (Actor 17-20).
    Digunakan untuk model selection yang jujur (cross-actor).
    """
    if config is None:
        config = {}

    result_path = config.get('result_path', 'results/unimodal_audio')
    store_name = config.get('store_name', 'audio')
    n_epochs = config.get('n_epochs', 80)
    early_stop_patience = config.get('early_stop_patience', 15)
    batch_size = config.get('batch_size', 64)
    n_threads = config.get('n_threads', 2)
    manual_seed = config.get('manual_seed', 42)
    grad_clip = config.get('grad_clip', 1.0)
    use_amp = config.get('use_amp', True)

    set_seed(manual_seed, deterministic=config.get('cudnn_deterministic', True))

    os.makedirs(result_path, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads,
        pin_memory=True,
        drop_last=True
    )

    train_logger = Logger(os.path.join(result_path, 'train.log'), ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
    val_logger = Logger(os.path.join(result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'prec5', 'f1_weighted', 'f1_macro'])

    if use_amp and torch.cuda.is_available():
        try:
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_prec1 = 0.0
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"  UNIMODAL AUDIO TRAINING — Static Split (Cross-Actor)")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples  : {len(val_loader.dataset)} (Actor 17-20)")
    print(f"  Mixed Precision: {'YES' if use_amp else 'NO'}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                try:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                except AttributeError:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            losses.update(loss.item(), inputs.size(0))
            prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

        current_lr = optimizer.param_groups[0]['lr']
        train_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg,
            'lr': current_lr
        })

        val_loss, val_prec1, val_metrics = validate_epoch(epoch, val_loader, model, criterion, device, val_logger)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        is_best = val_prec1 > best_prec1
        best_prec1 = max(best_prec1, val_prec1)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1,
            'val_metrics': val_metrics
        }, is_best, result_path, store_name)

        if is_best:
            epochs_no_improve = 0
            print(f"  Best — Epoch {epoch} | Val Acc: {val_prec1:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"  No improvement: {epochs_no_improve}/{early_stop_patience}")

        if epochs_no_improve >= early_stop_patience:
            print(f"\n  Early Stopping — Epoch {epoch}")
            break

    return {
        'best_prec1': best_prec1,
        'result_path': result_path,
        'model': model
    }
