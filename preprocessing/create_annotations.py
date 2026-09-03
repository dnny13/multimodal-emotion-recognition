# -*- coding: utf-8 -*-
"""
preprocessing/create_annotations.py
===================================
Membuat file anotasi dengan split statis berbasis aktor (16/4/4).
Hanya memproses file yang sudah memiliki .npy (baik audio maupun visual).

Format anotasi:
video_path;audio_path;label;subset

Catatan:
  - video_path: path ke .npy face crop
  - audio_path: path ke .npy MFCC+Delta (ekstensi _mfcc_delta.npy)
  - label: 1..8 (sesuai dengan 8 emosi RAVDESS)
  - subset: training / validation / testing
"""

import os
import argparse
import glob
from collections import defaultdict

# =============================================================================
# KONFIGURASI SPLIT
# =============================================================================
TRAIN_ACTORS = list(range(1, 17))
VAL_ACTORS   = list(range(17, 21))
TEST_ACTORS  = list(range(21, 25))

def get_actor_id_from_path(file_path: str) -> int:
    dirname = os.path.basename(os.path.dirname(file_path))
    if dirname.startswith('Actor_'):
        return int(dirname.split('_')[1])
    return -1

def get_emotion_from_filename(filename: str) -> int:
    parts = filename.split('-')
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except:
            pass
    return -1

def get_subset(actor_id: int) -> str:
    if actor_id in TRAIN_ACTORS:
        return 'training'
    elif actor_id in VAL_ACTORS:
        return 'validation'
    elif actor_id in TEST_ACTORS:
        return 'testing'
    return 'unknown'

def create_annotations(
    audio_root: str,
    visual_root: str,
    output_path: str,
    verbose: bool = True
) -> dict:
    stats = {
        'training': 0,
        'validation': 0,
        'testing': 0,
        'missing_visual': 0,
        'missing_audio': 0,
        'invalid_emotion': 0
    }

    # Cari semua file audio .npy dengan ekstensi _mfcc_delta.npy
    audio_files = glob.glob(f'{audio_root}/**/*_mfcc_delta.npy', recursive=True)
    if verbose:
        print(f"[INFO] Ditemukan {len(audio_files)} file audio .npy")

    visual_files = set(glob.glob(f'{visual_root}/**/*_faces.npy', recursive=True))
    if verbose:
        print(f"[INFO] Ditemukan {len(visual_files)} file visual .npy")

    visual_map = {}
    for vf in visual_files:
        base = os.path.basename(vf).replace('_faces.npy', '')
        visual_map[base] = vf

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w') as f:
        for audio_path in sorted(audio_files):
            filename = os.path.basename(audio_path)
            base_name = filename.replace('_mfcc_delta.npy', '')

            if base_name not in visual_map:
                stats['missing_visual'] += 1
                continue

            visual_path = visual_map[base_name]
            actor_id = get_actor_id_from_path(audio_path)
            emotion = get_emotion_from_filename(filename)

            if actor_id == -1:
                continue

            if emotion < 1 or emotion > 8:
                stats['invalid_emotion'] += 1
                continue

            subset = get_subset(actor_id)
            if subset == 'unknown':
                continue

            f.write(f"{visual_path};{audio_path};{emotion};{subset}\n")
            stats[subset] += 1

    total = sum(stats[s] for s in ['training', 'validation', 'testing'])
    if verbose:
        print("\n" + "=" * 60)
        print("  ANOTASI BERHASIL DIBUAT")
        print("=" * 60)
        print(f"  Train  : {stats['training']:>4} sampel — Actor_01–16")
        print(f"  Val    : {stats['validation']:>4} sampel — Actor_17–20")
        print(f"  Test   : {stats['testing']:>4} sampel — Actor_21–24")
        print(f"  Total  : {total:>4} sampel")
        print(f"  Missing visual : {stats['missing_visual']}")
        print(f"  Invalid emotion: {stats['invalid_emotion']}")
        print("=" * 60)

    return stats

def main():
    parser = argparse.ArgumentParser(description='Membuat file anotasi split 16/4/4 untuk RAVDESS')
    parser.add_argument('--audio_root', type=str, default='/content/RAVDESS_MFCC_DELTA',
                        help='Root folder .npy MFCC+Delta')
    parser.add_argument('--visual_root', type=str, default='/content/RAVDESS_FACES',
                        help='Root folder .npy face crop')
    parser.add_argument('--output', type=str, default='annotations/annotations_static.txt',
                        help='Path output file anotasi')
    parser.add_argument('--quiet', action='store_true', help='Mode silent')

    args = parser.parse_args()

    print("=" * 60)
    print("  CREATE ANNOTATIONS — SPLIT STATIS 16/4/4 (MFCC+Delta)")
    print("=" * 60)
    print(f"  Audio Root  : {args.audio_root}")
    print(f"  Visual Root : {args.visual_root}")
    print(f"  Output      : {args.output}")
    print(f"  Train Actors: {TRAIN_ACTORS[0]}–{TRAIN_ACTORS[-1]} (16 actors)")
    print(f"  Val Actors  : {VAL_ACTORS[0]}–{VAL_ACTORS[-1]}   (4 actors)")
    print(f"  Test Actors : {TEST_ACTORS[0]}–{TEST_ACTORS[-1]}  (4 actors)")
    print("=" * 60)

    create_annotations(
        audio_root=args.audio_root,
        visual_root=args.visual_root,
        output_path=args.output,
        verbose=not args.quiet
    )

if __name__ == '__main__':
    main()