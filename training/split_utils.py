# -*- coding: utf-8 -*-
"""
training/split_utils.py
=======================
Fungsi untuk membagi dataset training secara internal (train/val dari aktor yang SAMA)
guna memenuhi saran penguji: aktor pada pelatihan dan validasi sebaiknya sama.

🔧 FIX (v9):
  - MENGHAPUS create_internal_split_from_combined() karena menggabungkan Actor 1-16
    dan Actor 17-20 lalu diacak ulang, sehingga menghilangkan validasi cross-actor.
  - HANYA menggunakan create_internal_split_from_dataset() pada satu dataset (Actor 1-16).
"""

import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_split_indices(labels, val_size=0.2, random_state=42):
    """
    Stratified split berdasarkan label.

    Args:
        labels (array-like): Label dari dataset.
        val_size (float): Proporsi untuk validation.
        random_state (int): Seed untuk reproduksibilitas.

    Returns:
        train_idx, val_idx: Indeks untuk training dan validation.
    """
    labels = np.asarray(labels)
    indices = np.arange(len(labels))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(sss.split(indices, labels))
    return train_idx, val_idx


def create_internal_split_from_dataset(dataset, val_size=0.2, random_state=42):
    """
    Membagi satu dataset (hanya dari Actor 1-16) menjadi train dan same-actor validation.
    Tidak melibatkan Actor 17-20 sama sekali.

    Args:
        dataset: Dataset yang memiliki metode get_labels().
        val_size (float): Proporsi untuk validation.
        random_state (int): Seed untuk reproduksibilitas.

    Returns:
        train_subset, val_subset: Subset untuk training dan same-actor validation.
    """
    labels = dataset.get_labels()
    train_idx, val_idx = stratified_split_indices(labels, val_size, random_state)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    print(f"\n📊 Internal Split (Same-Actor, hanya dari dataset yang diberikan):")
    print(f"  Train samples: {len(train_subset)}")
    print(f"  Val samples  : {len(val_subset)} (aktor SAMA dengan train)")

    return train_subset, val_subset
