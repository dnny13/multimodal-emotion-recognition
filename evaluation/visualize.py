# -*- coding: utf-8 -*-
"""
evaluation/visualize.py
=======================
Visualisasi hasil eksperimen:
- Learning curves (loss & accuracy dari train/val log)
- Confusion matrix
- Perbandingan akurasi antar eksperimen (bar chart)
- Perbandingan efisiensi (bar chart)
- Perbandingan F1 score
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Setting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


def plot_learning_curves(
    train_log_path,
    val_log_path,
    save_path=None,
    title=None
):
    """
    Plot learning curves (loss dan accuracy) dari train.log dan val.log.

    Args:
        train_log_path: Path ke train.log (TSV)
        val_log_path: Path ke val.log (TSV)
        save_path: Path untuk menyimpan gambar (opsional)
        title: Judul plot (opsional)
    """
    df_train = pd.read_csv(train_log_path, sep='\t')
    df_val = pd.read_csv(val_log_path, sep='\t')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(df_train['epoch'], df_train['loss'],
                 label='Train', color='#2196F3', linestyle='--', alpha=0.7, linewidth=1.8)
    axes[0].plot(df_val['epoch'], df_val['loss'],
                 label='Val', color='#FF5722', linewidth=2)
    axes[0].set_title('Loss', fontsize=11)
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Accuracy
    axes[1].plot(df_train['epoch'], df_train['prec1'],
                 label='Train', color='#2196F3', linestyle='--', alpha=0.7, linewidth=1.8)
    axes[1].plot(df_val['epoch'], df_val['prec1'],
                 label='Val', color='#FF5722', linewidth=2)
    axes[1].set_title('Accuracy (Top-1)', fontsize=11)
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('Accuracy (%)', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Learning curves disimpan ke {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm,
    class_names,
    save_path=None,
    title=None,
    normalize=False
):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix (2D array)
        class_names: List nama kelas
        save_path: Path untuk menyimpan gambar (opsional)
        title: Judul plot (opsional)
        normalize: Normalisasi ke [0,1]
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
        vmin, vmax = 0, 1
    else:
        fmt = 'd'
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='white',
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title('Confusion Matrix', fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix disimpan ke {save_path}")

    plt.show()


def plot_accuracy_comparison(
    df,
    x_col='Experiment',
    y_col='Accuracy (%)',
    save_path=None,
    title='Model Accuracy Comparison'
):
    """
    Plot perbandingan akurasi antar eksperimen (bar chart).

    Args:
        df: DataFrame dengan kolom eksperimen dan metrik
        x_col: Nama kolom untuk sumbu x
        y_col: Nama kolom untuk sumbu y
        save_path: Path untuk menyimpan gambar
        title: Judul plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(df[x_col], df[y_col], color=sns.color_palette("Set2", len(df)))

    # Tambahkan nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.5,
            f'{height:.2f}%',
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax.set_xlabel('Experiment', fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Accuracy comparison disimpan ke {save_path}")

    plt.show()


def plot_efficiency_comparison(
    df,
    metric='Params (M)',
    save_path=None,
    title='Efficiency Comparison'
):
    """
    Plot perbandingan efisiensi (parameter, FLOPs, size, atau inference).

    Args:
        df: DataFrame dengan kolom eksperimen dan metrik efisiensi
        metric: 'Params (M)', 'FLOPs (G)', 'Model Size (MB)', 'Inference (ms)'
        save_path: Path untuk menyimpan gambar
        title: Judul plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(df['Experiment'], df[metric], color=sns.color_palette("Set3", len(df)))

    # Tambahkan nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + (0.05 * height if height > 0 else 0.5),
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax.set_xlabel('Experiment', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    # Tambahkan garis target constraint jika ada
    targets = {
        'Params (M)': 12,
        'FLOPs (G)': 2,
        'Model Size (MB)': 48,
        'Inference (ms)': 100
    }
    if metric in targets:
        ax.axhline(y=targets[metric], color='red', linestyle='--', linewidth=1.5, label='Constraint')
        ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Efficiency comparison disimpan ke {save_path}")

    plt.show()


def plot_f1_comparison(
    df,
    save_path=None,
    title='F1 Score Comparison'
):
    """
    Plot perbandingan F1-Macro dan F1-Weighted antar eksperimen.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width/2, df['F1-Macro'], width, label='F1-Macro', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, df['F1-Weighted'], width, label='F1-Weighted', color='#4ECDC4')

    ax.set_xlabel('Experiment', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Experiment'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Tambahkan nilai di atas bar
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ F1 comparison disimpan ke {save_path}")

    plt.show()