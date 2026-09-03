# -*- coding: utf-8 -*-
"""
training/validation.py
======================
Validation loop untuk unimodal dan multimodal.

🔧 FIX: Tambahkan proyeksi audio dan visual sebelum fusion.
"""

import torch
import numpy as np
from .utils import AverageMeter, calculate_accuracy
from evaluation.metrics import compute_classification_metrics


def validate_epoch(epoch, data_loader, model, criterion, device, logger=None):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(device)
            elif len(batch) == 3:
                audio_inputs, visual_inputs, targets = batch
                audio_inputs = audio_inputs.to(device)
                visual_inputs = visual_inputs.to(device)

            targets = targets.to(device)

            if len(batch) == 2:
                outputs = model(inputs)
            else:
                outputs = model(audio_inputs, visual_inputs)

            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0) if len(batch) == 2 else audio_inputs.size(0))
            prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1, outputs.size(0))
            top5.update(prec5, outputs.size(0))

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    metrics = compute_classification_metrics(all_targets, all_preds)

    if epoch == -1:
        epoch_label = "TEST SET EVALUATION"
    else:
        epoch_label = f"Val {epoch}"

    print(f"\n[{epoch_label}]")
    print(f"  Loss       : {losses.avg:.4f}")
    print(f"  Acc@1      : {top1.avg:.2f}%")
    print(f"  Acc@5      : {top5.avg:.2f}%")
    print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"  Macro F1   : {metrics['f1_macro']:.4f}")

    if logger is not None:
        logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg,
            'f1_weighted': metrics['f1_weighted'],
            'f1_macro': metrics['f1_macro']
        })

    return losses.avg, top1.avg, metrics


def validate_epoch_with_cache(epoch, data_loader, fusion_model, criterion, device, logger=None):
    fusion_model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            audio_embeds, visual_embeds, targets = batch
            audio_embeds = audio_embeds.to(device)
            visual_embeds = visual_embeds.to(device)
            targets = targets.to(device)

            # 🔧 PERBAIKAN: Proyeksi sebelum fusion
            audio_embeds_proj = fusion_model.audio_projector(audio_embeds)
            visual_embeds_proj = fusion_model.visual_projector(visual_embeds)

            fused = fusion_model.fusion(audio_embeds_proj, visual_embeds_proj)
            outputs = fusion_model.classifier(fused)

            loss = criterion(outputs, targets)

            losses.update(loss.item(), audio_embeds.size(0))
            prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1, outputs.size(0))
            top5.update(prec5, outputs.size(0))

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    metrics = compute_classification_metrics(all_targets, all_preds)

    if epoch == -1:
        epoch_label = "TEST SET EVALUATION"
    else:
        epoch_label = f"Val {epoch}"

    print(f"\n[{epoch_label}]")
    print(f"  Loss       : {losses.avg:.4f}")
    print(f"  Acc@1      : {top1.avg:.2f}%")
    print(f"  Acc@5      : {top5.avg:.2f}%")
    print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"  Macro F1   : {metrics['f1_macro']:.4f}")

    if logger is not None:
        logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg,
            'f1_weighted': metrics['f1_weighted'],
            'f1_macro': metrics['f1_macro']
        })

    return losses.avg, top1.avg, metrics