# -*- coding: utf-8 -*-
"""
training/__init__.py
====================
Package untuk training loops unimodal dan multimodal.
"""

from .train_unimodal_audio import (
    train_unimodal_audio,
    train_unimodal_audio_static,
    LabelSmoothingCrossEntropy as AudioLabelSmoothing,
)
from .train_unimodal_visual import (
    train_unimodal_visual,
    train_unimodal_visual_static,
    LabelSmoothingCrossEntropy as VisualLabelSmoothing,
)
from .train_multimodal import (
    train_multimodal_fold,
    FocalLoss,
    mixup_data,
    mixup_criterion,
    apply_modality_dropout,
    embedding_augmentation,
)
from .split_utils import create_internal_split_from_dataset
from .utils import (
    Logger,
    AverageMeter,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    compute_class_weights,
    calculate_accuracy,
)
from .validation import validate_epoch, validate_epoch_with_cache

__all__ = [
    # Unimodal Audio
    'train_unimodal_audio',
    'train_unimodal_audio_static',
    'AudioLabelSmoothing',
    # Unimodal Visual
    'train_unimodal_visual',
    'train_unimodal_visual_static',
    'VisualLabelSmoothing',
    # Multimodal
    'train_multimodal_fold',
    'FocalLoss',
    'mixup_data',
    'mixup_criterion',
    'apply_modality_dropout',
    'embedding_augmentation',
    # Split & Utils
    'create_internal_split_from_dataset',
    'Logger',
    'AverageMeter',
    'set_seed',
    'save_checkpoint',
    'load_checkpoint',
    'compute_class_weights',
    'calculate_accuracy',
    # Validation
    'validate_epoch',
    'validate_epoch_with_cache',
]