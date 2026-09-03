# -*- coding: utf-8 -*-
"""
evaluation/verify_split_distribution.py
=======================================
Membaca annotations_static.txt dan menghitung distribusi 8 kelas emosi
per subset (training, validation, testing) untuk memastikan tidak ada bias
pada split aktor (terutama Actor 21-24).

Output:
  - print tabel perbandingan proporsi
  - simpan ke CSV: results/split_class_distribution.csv
"""

import os
import argparse
import pandas as pd
import numpy as np


def verify_split_distribution(annotation_path: str, output_dir: str = 'results'):
    """
    Membaca file anotasi dan menghitung distribusi kelas per split.
    
    Args:
        annotation_path: Path ke annotations_static.txt
        output_dir: Direktori untuk menyimpan CSV hasil
    
    Returns:
        DataFrame dengan proporsi kelas per split
    """
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Anotasi tidak ditemukan: {annotation_path}")

    # Baca file anotasi (format: visual_path;audio_path;label;subset)
    df = pd.read_csv(annotation_path, sep=';', header=None,
                     names=['visual_path', 'audio_path', 'label', 'subset'])
    
    # Konversi label ke int (1-8)
    df['label'] = df['label'].astype(int)
    
    # Hitung distribusi per subset
    splits = ['training', 'validation', 'testing']
    class_names = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    
    print("\n" + "="*80)
    print("  📊 VERIFIKASI DISTRIBUSI KELAS PER SPLIT (RAVDESS Actor-Based)")
    print("="*80)
    
    distribution = {}
    total_per_split = {}
    
    for split in splits:
        df_split = df[df['subset'] == split]
        total_per_split[split] = len(df_split)
        counts = df_split['label'].value_counts().sort_index()
        # Pastikan semua kelas 1-8 ada
        for i in range(1, 9):
            if i not in counts.index:
                counts[i] = 0
        counts = counts.sort_index()
        distribution[split] = counts.values
        print(f"\n{split.upper()} (total {len(df_split)} sampel):")
        for idx, (label, count) in enumerate(zip(range(1, 9), counts.values)):
            pct = count / len(df_split) * 100
            print(f"  Kelas {label:2d} ({class_names[idx]:<10}): {count:>4} sampel ({pct:>5.2f}%)")
    
    # ============================================================
    # Buat DataFrame perbandingan
    # ============================================================
    df_dist = pd.DataFrame({
        'Kelas': [f"{i} ({class_names[i-1]})" for i in range(1, 9)],
        'Training (%)': distribution['training'] / total_per_split['training'] * 100,
        'Validation (%)': distribution['validation'] / total_per_split['validation'] * 100,
        'Testing (%)': distribution['testing'] / total_per_split['testing'] * 100,
    })
    
    # Tambahkan selisih Testing vs Training (untuk deteksi bias)
    df_dist['Selisih Test-Train (%)'] = df_dist['Testing (%)'] - df_dist['Training (%)']
    
    print("\n" + "="*80)
    print("  TABEL PERBANDINGAN PROPORSI KELAS")
    print("="*80)
    print(df_dist.to_string(index=False, float_format='{:.2f}'.format))
    print("="*80)
    
    # ============================================================
    # Interpretasi cepat
    # ============================================================
    max_diff = df_dist['Selisih Test-Train (%)'].abs().max()
    print(f"\n📌 Selisih maksimum Testing vs Training: {max_diff:.2f}%")
    if max_diff < 5.0:
        print("   ✅ Distribusi kelas antar split relatif SEIMBANG (perbedaan < 5%).")
    else:
        print("   ⚠️ Terdapat perbedaan > 5% pada beberapa kelas. Perlu diperhatikan dalam analisis.")
    
    # Simpan ke CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'split_class_distribution.csv')
    df_dist.to_csv(output_path, index=False)
    print(f"\n✅ Hasil disimpan ke: {output_path}")
    
    return df_dist


def main():
    parser = argparse.ArgumentParser(description='Verifikasi distribusi kelas per split RAVDESS')
    parser.add_argument('--annotation_path', type=str, default='annotations/annotations_static.txt',
                        help='Path ke file anotasi')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Direktori output untuk CSV')
    args = parser.parse_args()
    
    verify_split_distribution(args.annotation_path, args.output_dir)


if __name__ == '__main__':
    main()