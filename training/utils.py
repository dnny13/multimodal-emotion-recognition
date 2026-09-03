# -*- coding: utf-8 -*-
"""
training/utils.py
=================
Fungsi-fungsi bantu untuk training: logger, checkpoint, metrik, dan helper.
"""

import os
import csv
import shutil
import random
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score
)


class AverageMeter:
    """Menghitung rata-rata dan nilai terkini dari suatu metrik."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        v = val.item() if torch.is_tensor(val) else float(val)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    """
    Mencatat metrik ke file CSV (tab-separated).
    """
    def __init__(self, path, header):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.f = open(path, 'w', newline='')
        self.w = csv.writer(self.f, delimiter='\t')
        self.w.writerow(header)
        self.header = header

    def log(self, vals):
        self.w.writerow([vals[c] for c in self.header])
        self.f.flush()

    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close()


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Mengatur seed untuk reproduktibilitas penuh.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(state, is_best, result_path, store_name):
    """
    Menyimpan checkpoint model. Jika is_best, salin sebagai best model.
    """
    os.makedirs(result_path, exist_ok=True)
    last_path = os.path.join(result_path, f'{store_name}_last.pth')
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(result_path, f'{store_name}_best.pth')
        shutil.copyfile(last_path, best_path)
        print(f"  [SAVE] Best model → {best_path}")
    return last_path


def load_checkpoint(path, model, optimizer=None, device='cuda'):
    """
    Memuat checkpoint model dengan penanganan robust untuk berbagai format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Format checkpoint tidak dikenali: {path}")

    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    if optimizer is not None and isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint


def compute_class_weights(annotation_path, n_classes):
    """
    Menghitung class weights dari file anotasi untuk mengatasi imbalance.
    """
    counts = np.zeros(n_classes, dtype=np.float32)
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 4 and parts[3].strip() == 'training':
                try:
                    label_idx = int(parts[2].strip()) - 1
                    if 0 <= label_idx < n_classes:
                        counts[label_idx] += 1
                except:
                    continue

    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes

    print("\n  [INFO] Class weights (inverse-frequency):")
    for i, w in enumerate(weights):
        print(f"    [{i}] weight={w:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


def calculate_accuracy(outputs, targets, topk=(1,)):
    """
    Menghitung akurasi top-1 dan top-5.
    """
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