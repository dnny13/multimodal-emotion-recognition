# -*- coding: utf-8 -*-
"""
preprocessing/sync_audio_visual.py
==================================
Verifikasi sinkronisasi temporal audio-visual dan validasi dataset multimodal.
Fungsi:
1. Memastikan setiap video memiliki pasangan audio & visual yang lengkap
2. Memastikan shape .npy sesuai dengan yang diharapkan
3. Memberikan laporan statistik kesiapan data multimodal
"""

import os
import argparse
import numpy as np
import glob
from collections import defaultdict

# =============================================================================
# KONFIGURASI
# =============================================================================
EXPECTED_AUDIO_SHAPE = (120, 224)   # MFCC+Delta (120 channel)
EXPECTED_VISUAL_SHAPE = (15, 224, 224, 3)

def verify_multimodal_data(audio_root: str, visual_root: str) -> dict:
    stats = {
        'total_audio': 0,
        'total_visual': 0,
        'matched_pairs': 0,
        'audio_invalid_shape': 0,
        'visual_invalid_shape': 0,
        'audio_min': float('inf'),
        'audio_max': float('-inf'),
        'audio_mean': 0.0,
        'visual_min': float('inf'),
        'visual_max': float('-inf'),
        'visual_mean': 0.0,
        'by_actor': defaultdict(lambda: {'audio': 0, 'visual': 0, 'matched': 0})
    }

    audio_files = {}
    for ap in glob.glob(f'{audio_root}/**/*_mfcc_delta.npy', recursive=True):
        base = os.path.basename(ap).replace('_mfcc_delta.npy', '')
        audio_files[base] = ap
        stats['total_audio'] += 1
        actor = os.path.basename(os.path.dirname(ap))
        if actor.startswith('Actor_'):
            actor_id = int(actor.split('_')[1])
            stats['by_actor'][actor_id]['audio'] += 1

    visual_files = {}
    for vp in glob.glob(f'{visual_root}/**/*_faces.npy', recursive=True):
        base = os.path.basename(vp).replace('_faces.npy', '')
        visual_files[base] = vp
        stats['total_visual'] += 1
        actor = os.path.basename(os.path.dirname(vp))
        if actor.startswith('Actor_'):
            actor_id = int(actor.split('_')[1])
            stats['by_actor'][actor_id]['visual'] += 1

    matched_bases = set(audio_files.keys()) & set(visual_files.keys())
    stats['matched_pairs'] = len(matched_bases)

    for base in matched_bases:
        actor = os.path.basename(os.path.dirname(audio_files[base]))
        actor_id = int(actor.split('_')[1]) if actor.startswith('Actor_') else -1

        try:
            audio = np.load(audio_files[base])
            if audio.shape != EXPECTED_AUDIO_SHAPE:
                stats['audio_invalid_shape'] += 1
            else:
                stats['audio_min'] = min(stats['audio_min'], audio.min())
                stats['audio_max'] = max(stats['audio_max'], audio.max())
                stats['audio_mean'] += audio.mean()
        except Exception:
            stats['audio_invalid_shape'] += 1

        try:
            visual = np.load(visual_files[base])
            if visual.shape != EXPECTED_VISUAL_SHAPE:
                stats['visual_invalid_shape'] += 1
            else:
                stats['visual_min'] = min(stats['visual_min'], visual.min())
                stats['visual_max'] = max(stats['visual_max'], visual.max())
                stats['visual_mean'] += visual.mean()
        except Exception:
            stats['visual_invalid_shape'] += 1

        if actor_id != -1:
            stats['by_actor'][actor_id]['matched'] += 1

    if stats['matched_pairs'] > 0:
        stats['audio_mean'] /= stats['matched_pairs']
        stats['visual_mean'] /= stats['matched_pairs']

    return stats

def main():
    parser = argparse.ArgumentParser(description='Verifikasi sinkronisasi audio-visual RAVDESS')
    parser.add_argument('--audio_root', type=str, default='/content/RAVDESS_MFCC_DELTA',
                        help='Root folder .npy MFCC+Delta')
    parser.add_argument('--visual_root', type=str, default='/content/RAVDESS_FACES',
                        help='Root folder .npy face crop')

    args = parser.parse_args()

    print("=" * 60)
    print("  VERIFIKASI SINKRONISASI AUDIO-VISUAL (MFCC+Delta)")
    print("=" * 60)
    print(f"  Audio Root  : {args.audio_root}")
    print(f"  Visual Root : {args.visual_root}")
    print("=" * 60)
    print(f"  Expected Audio Shape  : {EXPECTED_AUDIO_SHAPE}")
    print(f"  Expected Visual Shape : {EXPECTED_VISUAL_SHAPE}")
    print("=" * 60)

    stats = verify_multimodal_data(args.audio_root, args.visual_root)

    print("\n📊 STATISTIK DATA")
    print(f"  Total file audio       : {stats['total_audio']}")
    print(f"  Total file visual      : {stats['total_visual']}")
    print(f"  Pasangan matched       : {stats['matched_pairs']}")
    print(f"  Audio invalid shape    : {stats['audio_invalid_shape']}")
    print(f"  Visual invalid shape   : {stats['visual_invalid_shape']}")

    print("\n📊 STATISTIK NILAI")
    print(f"  Audio - Min: {stats['audio_min']:.4f}, Max: {stats['audio_max']:.4f}, Mean: {stats['audio_mean']:.4f}")
    print(f"  Visual - Min: {stats['visual_min']:.4f}, Max: {stats['visual_max']:.4f}, Mean: {stats['visual_mean']:.4f}")

    print("\n📊 PER AKTOR")
    sorted_actors = sorted(stats['by_actor'].keys())
    for actor in sorted_actors:
        data = stats['by_actor'][actor]
        status = "✅" if data['audio'] == data['visual'] == data['matched'] > 0 else "⚠️"
        print(f"  {status} Actor_{actor:02d}: Audio={data['audio']:>3}, Visual={data['visual']:>3}, Matched={data['matched']:>3}")

    total_expected = 1440
    if stats['matched_pairs'] == total_expected:
        print(f"\n✅ SEMUA {total_expected} file modalitas 01 memiliki pasangan yang valid!")
    else:
        print(f"\n⚠️ Hanya {stats['matched_pairs']}/{total_expected} file yang memiliki pasangan valid.")

if __name__ == '__main__':
    main()