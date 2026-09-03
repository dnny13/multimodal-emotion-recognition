# -*- coding: utf-8 -*-
"""
data/transforms.py
==================
Transformasi data untuk audio dan visual.

🔧 FIX (v5):
  - Tambahkan augmentasi visual: RandomRotation, ColorJitter, RandomAffine
  - Sesuaikan RandomResizedCrop scale range

🔧 FIX (v6):
  - Hapus RandomHorizontalFlip dari audio 2D (tidak relevan untuk spektrogram)
  - SpecAugment mengembalikan numpy array (bukan Tensor) agar kompatibel dengan ToTensor

🔧 FIX (v7):
  - Tambahkan ToPILImage() di awal transformasi visual untuk mengonversi numpy array ke PIL Image
  - Ini menyelesaikan error "img should be PIL Image. Got numpy.ndarray"
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image


# =============================================================================
# AUDIO TRANSFORMS
# =============================================================================

class SpecAugment:
    """
    SpecAugment untuk spektrogram audio.
    🔧 FIX: Mengembalikan numpy array agar kompatibel dengan ToTensor.
    """
    def __init__(self, freq_mask_param=15, time_mask_param=30, n_freq_masks=2, n_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, spec):
        if isinstance(spec, torch.Tensor):
            spec = spec.numpy()
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, spec.shape[0] - f)
            spec[f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, spec.shape[1] - t)
            spec[:, t0:t0+t] = 0
        
        return spec


def get_audio_transforms_1dcnn(is_training=True, use_spec_augment=False):
    """
    Transformasi untuk audio 1D-CNN (MFCC+Delta).
    """
    transforms_list = []
    if is_training and use_spec_augment:
        transforms_list.append(SpecAugment(freq_mask_param=15, time_mask_param=30))
    return transforms.Compose(transforms_list)


def get_audio_transforms_2dcnn(is_training=True, use_spec_augment=False, spec_size=(224, 224)):
    """
    Transformasi untuk audio 2D-CNN (Mel-Spectrogram).
    """
    transforms_list = []
    
    if is_training and use_spec_augment:
        transforms_list.append(SpecAugment(freq_mask_param=15, time_mask_param=30))
    
    transforms_list.append(transforms.ToTensor())
    
    if spec_size is not None:
        transforms_list.append(transforms.Resize(spec_size))
    
    return transforms.Compose(transforms_list)


# =============================================================================
# VISUAL TRANSFORMS (DIPERBAIKI DENGAN ToPILImage)
# =============================================================================

def get_visual_transforms(is_training=True, image_size=(224, 224)):
    """
    Transformasi untuk visual (face crop).
    
    🔧 FIX: Tambahkan ToPILImage() di awal untuk konversi numpy array ke PIL Image.
    Augmentasi:
        - RandomRotation(degrees=10)
        - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
        - RandomAffine(translate=(0.05, 0.05))
        - RandomResizedCrop scale=(0.85, 1.0)
    """
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),  # 🔧 FIX: Konversi numpy array ke PIL Image
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomResizedCrop(size=image_size, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),  # 🔧 FIX: Konversi numpy array ke PIL Image
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_visual_transforms_simple(is_training=True, image_size=(224, 224)):
    """
    Versi sederhana tanpa augmentasi agresif (untuk validation/testing).
    """
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


# =============================================================================
# MULTIMODAL TRANSFORMS
# =============================================================================

def get_multimodal_transforms(is_training=True, audio_size=(224, 224), visual_size=(224, 224),
                              use_spec_augment=False, audio_mode='1d'):
    """
    Transformasi untuk data multimodal (audio + visual).
    """
    if audio_mode == '1d':
        audio_transform = get_audio_transforms_1dcnn(is_training, use_spec_augment)
    else:
        audio_transform = get_audio_transforms_2dcnn(is_training, use_spec_augment, audio_size)
    
    visual_transform = get_visual_transforms(is_training, visual_size)
    
    return audio_transform, visual_transform