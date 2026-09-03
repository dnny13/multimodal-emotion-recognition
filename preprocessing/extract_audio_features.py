# -*- coding: utf-8 -*-
"""
preprocessing/extract_audio_features.py
=======================================
Mengekstrak fitur audio dari file MP4 RAVDESS (modalitas 01).

🔧 FIX (v10 - FINAL):
  - Memperbaiki pemanggilan compute_mfcc_delta dan compute_mel_spectrogram
    agar tidak saling bertukar parameter (n_mels vs n_mfcc).
  - TIDAK PERNAH menyimpan array nol. Jika ekstraksi gagal, lemparkan exception.
  - Memastikan file .npy yang disimpan VALID (bukan all-zero).
  - Menambahkan validasi shape dan nilai sebelum menyimpan.
"""

import os
import argparse
import glob
import subprocess
import tempfile
import numpy as np
import librosa
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DEFAULT_SR = 16000
DEFAULT_N_MFCC = 40
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_FFT = 2048
DEFAULT_TARGET_TIME = 224
DEFAULT_DURATION = 3.0
DEFAULT_TOP_DB = 30
DEFAULT_N_MELS = 128


# ============================================================
# DECODE MP4 -> WAV VIA FFMPEG
# ============================================================
def decode_mp4_to_wav(mp4_path: str, sr: int = DEFAULT_SR) -> str:
    tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_wav_path = tmp_wav.name
    tmp_wav.close()

    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', mp4_path,
        '-ar', str(sr),
        '-ac', '1',
        tmp_wav_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if result.returncode != 0 or not os.path.exists(tmp_wav_path) or os.path.getsize(tmp_wav_path) == 0:
        if os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)
        raise RuntimeError(f"ffmpeg gagal decode {mp4_path}: {result.stderr.decode(errors='ignore')[:200]}")

    return tmp_wav_path


def load_audio_safe(mp4_path: str, sr: int, duration: float):
    tmp_wav_path = decode_mp4_to_wav(mp4_path, sr=sr)
    try:
        y, _ = librosa.load(tmp_wav_path, sr=sr, duration=duration, mono=True)
    finally:
        if os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)
    return y


# ============================================================
# MFCC + DELTA (dengan validasi)
# ============================================================
def compute_mfcc_delta(
    audio_path: str,
    sr: int = DEFAULT_SR,
    n_mfcc: int = DEFAULT_N_MFCC,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_fft: int = DEFAULT_N_FFT,
    duration: float = DEFAULT_DURATION,
    target_time: int = DEFAULT_TARGET_TIME,
    top_db: int = DEFAULT_TOP_DB,
) -> np.ndarray:
    """
    🔧 FIX: Melempar exception jika gagal, TIDAK mengembalikan zeros.
    """
    y = load_audio_safe(audio_path, sr=sr, duration=duration)
    if len(y) < sr * duration:
        y = np.pad(y, (0, int(sr * duration) - len(y)), mode='constant')

    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y_trim) < 512:
        y_trim = y

    mfccs = librosa.feature.mfcc(
        y=y_trim, sr=sr, n_mfcc=n_mfcc,
        hop_length=hop_length, n_fft=n_fft
    )
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    features = np.concatenate([mfccs, delta, delta2], axis=0)

    if features.shape[1] != target_time:
        features = cv2.resize(
            features, (target_time, features.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # 🔧 FIX: Validasi keras — jika semua nol atau NaN, lemparkan exception
    if not np.isfinite(features).all():
        raise ValueError(f"MFCC menghasilkan NaN/Inf pada {audio_path}")
    if np.max(features) == np.min(features):
        raise ValueError(f"MFCC menghasilkan flat/semua nol pada {audio_path}")

    return features.astype(np.float32)


# ============================================================
# MEL-SPECTROGRAM (dengan validasi)
# ============================================================
def compute_mel_spectrogram(
    audio_path: str,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_fft: int = DEFAULT_N_FFT,
    duration: float = DEFAULT_DURATION,
    target_time: int = DEFAULT_TARGET_TIME,
    top_db: int = DEFAULT_TOP_DB,
) -> np.ndarray:
    """
    🔧 FIX: Melempar exception jika gagal, TIDAK mengembalikan zeros.
    """
    y = load_audio_safe(audio_path, sr=sr, duration=duration)
    if len(y) < sr * duration:
        y = np.pad(y, (0, int(sr * duration) - len(y)), mode='constant')

    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y_trim) < 512:
        y_trim = y

    mel = librosa.feature.melspectrogram(
        y=y_trim, sr=sr, n_mels=n_mels,
        hop_length=hop_length, n_fft=n_fft
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    if mel_norm.shape[1] != target_time:
        mel_norm = cv2.resize(
            mel_norm, (target_time, n_mels),
            interpolation=cv2.INTER_LINEAR
        )

    # 🔧 FIX: Validasi keras
    if not np.isfinite(mel_norm).all():
        raise ValueError(f"Mel-Spectrogram menghasilkan NaN/Inf pada {audio_path}")
    if np.max(mel_norm) == np.min(mel_norm):
        raise ValueError(f"Mel-Spectrogram menghasilkan flat/semua nol pada {audio_path}")

    return mel_norm.astype(np.float32)


# ============================================================
# FUNGSI EKSTRAKSI BATCH (DIPERBAIKI)
# ============================================================
def extract_audio_features(
    root_path: str,
    output_path: str,
    feature_type: str = 'mfcc',
    n_mfcc: int = DEFAULT_N_MFCC,
    n_mels: int = DEFAULT_N_MELS,
    force: bool = False,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
) -> dict:
    stats = {'total': 0, 'processed': 0, 'skipped': 0, 'failed': 0, 'corrupted': 0}

    os.makedirs(output_path, exist_ok=True)
    mp4_files = glob.glob(f'{root_path}/**/01-*.mp4', recursive=True)
    stats['total'] = len(mp4_files)

    if feature_type == 'mfcc':
        suffix = '_mfcc_delta.npy'
        expected_shape = (120, DEFAULT_TARGET_TIME)
        desc = "Ekstraksi MFCC+Delta"
    elif feature_type == 'melspec':
        suffix = '_melspec.npy'
        expected_shape = (n_mels, DEFAULT_TARGET_TIME)
        desc = "Ekstraksi Mel-Spectrogram"
    else:
        raise ValueError(f"feature_type '{feature_type}' tidak dikenali.")

    for video_path in tqdm(mp4_files, desc=desc, unit="file"):
        filename = os.path.basename(video_path)
        rel_path = os.path.relpath(video_path, root_path)
        npy_name = filename.replace('.mp4', suffix)
        npy_dir = os.path.join(output_path, os.path.dirname(rel_path))
        npy_path = os.path.join(npy_dir, npy_name)

        if not force and os.path.exists(npy_path):
            try:
                arr = np.load(npy_path)
                if arr.shape == expected_shape and np.max(arr) > 1e-8:
                    stats['skipped'] += 1
                    continue
                else:
                    print(f"⚠️ File {npy_path} ada tetapi corrupt, akan diekstrak ulang.")
                    os.remove(npy_path)
            except Exception:
                pass

        os.makedirs(npy_dir, exist_ok=True)

        try:
            # ============================================================
            # 🔧 FIX: Panggil fungsi dengan parameter yang sesuai
            # ============================================================
            if feature_type == 'mfcc':
                features = compute_mfcc_delta(
                    audio_path=video_path,
                    sr=sr,
                    n_mfcc=n_mfcc,
                    hop_length=DEFAULT_HOP_LENGTH,
                    n_fft=DEFAULT_N_FFT,
                    duration=duration,
                    target_time=DEFAULT_TARGET_TIME,
                    top_db=DEFAULT_TOP_DB
                )
            else:  # feature_type == 'melspec'
                features = compute_mel_spectrogram(
                    audio_path=video_path,
                    sr=sr,
                    n_mels=n_mels,
                    hop_length=DEFAULT_HOP_LENGTH,
                    n_fft=DEFAULT_N_FFT,
                    duration=duration,
                    target_time=DEFAULT_TARGET_TIME,
                    top_db=DEFAULT_TOP_DB
                )

        except Exception as e:
            stats['failed'] += 1
            print(f"❌ Gagal ekstraksi {filename}: {e}")
            continue

        # 🔧 FIX: Validasi double-check sebelum save
        if features.shape != expected_shape:
            stats['failed'] += 1
            print(f"⚠️ Shape tidak valid untuk {filename}: {features.shape} (expected {expected_shape})")
            continue

        if np.max(features) <= 1e-8:
            stats['corrupted'] += 1
            print(f"⚠️ File {filename} hasil ekstraksi all-zero, dianggap gagal.")
            continue

        np.save(npy_path, features)
        stats['processed'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description='Ekstraksi fitur audio dari RAVDESS (modalitas 01)')
    parser.add_argument('--root_path', type=str, default='/content/drive/MyDrive/RAVDESS')
    parser.add_argument('--output_path', type=str, default='/content/RAVDESS_MFCC_DELTA')
    parser.add_argument('--feature_type', type=str, default='mfcc', choices=['mfcc', 'melspec'])
    parser.add_argument('--n_mfcc', type=int, default=DEFAULT_N_MFCC)
    parser.add_argument('--n_mels', type=int, default=DEFAULT_N_MELS)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--sr', type=int, default=DEFAULT_SR)
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION)
    parser.add_argument('--no_augment', action='store_true',
                         help='Flag informatif — augmentasi permanen memang tidak pernah diterapkan di sini.')

    args = parser.parse_args()

    ffmpeg_check = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE)
    if ffmpeg_check.returncode != 0:
        print("❌ ffmpeg tidak ditemukan di sistem. Install dengan: !apt-get install -y ffmpeg")
        return

    print("=" * 60)
    print(f"  EKSTRAKSI AUDIO — {args.feature_type.upper()} — RAVDESS")
    print("=" * 60)
    print(f"  Sumber MP4    : {args.root_path}")
    print(f"  Output .npy   : {args.output_path}")
    print("  🔧 FIX: File corrupt/gagal TIDAK akan disimpan.")
    if args.feature_type == 'mfcc':
        print(f"  N-MFCC        : {args.n_mfcc}")
    else:
        print(f"  N-Mels        : {args.n_mels}")
    print("=" * 60)

    stats = extract_audio_features(
        root_path=args.root_path,
        output_path=args.output_path,
        feature_type=args.feature_type,
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels,
        force=args.force,
        sr=args.sr,
        duration=args.duration,
    )

    print("\n" + "=" * 60)
    print("  HASIL EKSTRAKSI")
    print("=" * 60)
    print(f"  Total file MP4        : {stats['total']}")
    print(f"  Berhasil diproses     : {stats['processed']}")
    print(f"  Di-skip (sudah valid) : {stats['skipped']}")
    print(f"  Gagal (error/exception): {stats['failed']}")
    print(f"  Corrupt (all-zero)    : {stats['corrupted']}")
    print("=" * 60)

    if stats['failed'] > 0 or stats['corrupted'] > 0:
        print("\n⚠️ PERINGATAN: Ada file yang gagal/corrupt. Periksa kembali data sumber.")


if __name__ == '__main__':
    main()