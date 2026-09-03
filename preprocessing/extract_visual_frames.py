# -*- coding: utf-8 -*-
"""
preprocessing/extract_visual_frames.py
======================================
Mengekstrak face crop dari file MP4 RAVDESS (modalitas 01) menjadi .npy.

🔧 FIX (v10 - FINAL):
  - TIDAK fallback otomatis ke Haar Cascade jika unduhan DNN-SSD gagal.
  - Jika DNN-SSD gagal unduh, LEMPAR RuntimeError dan minta pengguna memilih --detector haar.
  - Menambahkan validasi kualitas deteksi (jika crop gagal, catat sebagai failed).
  - Memastikan file .npy yang disimpan VALID (bukan background full-frame).
"""

import os
import argparse
import urllib.request
import numpy as np
import cv2
from tqdm import tqdm
import warnings
import glob

warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURASI
# =============================================================================
DEFAULT_NUM_FRAMES = 15
DEFAULT_FACE_SIZE = 224
CACHE_SUFFIX = '_faces.npy'

SSD_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
)
SSD_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
SSD_PROTOTXT_PATH = "deploy.prototxt"
SSD_MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"


# =============================================================================
# AUTO-DOWNLOAD MODEL SSD (dengan FAIL-FAST)
# =============================================================================
def ensure_ssd_model_downloaded(force_download: bool = False) -> bool:
    """Mengunduh model DNN-SSD jika belum ada. Return True jika berhasil."""
    try:
        if force_download or not os.path.exists(SSD_PROTOTXT_PATH):
            print("⬇️  Mengunduh deploy.prototxt (DNN-SSD)...")
            urllib.request.urlretrieve(SSD_PROTOTXT_URL, SSD_PROTOTXT_PATH)
        if force_download or not os.path.exists(SSD_MODEL_PATH):
            print("⬇️  Mengunduh res10_300x300_ssd_iter_140000.caffemodel (DNN-SSD, ~10MB)...")
            urllib.request.urlretrieve(SSD_MODEL_URL, SSD_MODEL_PATH)
        return os.path.exists(SSD_PROTOTXT_PATH) and os.path.exists(SSD_MODEL_PATH)
    except Exception as e:
        print(f"❌ Gagal mengunduh model DNN-SSD: {e}")
        return False


# =============================================================================
# DETEKSI WAJAH — Hanya DNN-SSD (tidak ada fallback otomatis)
# =============================================================================
_ssd_net = None
_detector_type = None  # 'dnn_ssd' atau 'haar'


def init_detector(preferred: str = 'dnn_ssd'):
    """Inisialisasi detector. Jika preferred='dnn_ssd' dan gagal, lemparkan error."""
    global _ssd_net, _detector_type

    if preferred == 'haar':
        _haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        _detector_type = 'haar'
        print("⚠️  Detector aktif: Haar Cascade (akurasi rendah, hanya untuk fallback eksplisit).")
        return

    # DNN-SSD
    if not ensure_ssd_model_downloaded():
        raise RuntimeError(
            "❌ Gagal mengunduh model DNN-SSD. Pastikan koneksi internet aktif.\n"
            "   Untuk menggunakan Haar Cascade (akurasi lebih rendah), jalankan dengan:\n"
            "   --detector haar"
        )
    
    try:
        _ssd_net = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT_PATH, SSD_MODEL_PATH)
        _detector_type = 'dnn_ssd'
        print("✅ Detector aktif: DNN-SSD (res10_300x300)")
    except Exception as e:
        raise RuntimeError(
            f"❌ Gagal load DNN-SSD: {e}\n"
            "   Untuk menggunakan Haar Cascade (akurasi lebih rendah), jalankan dengan:\n"
            "   --detector haar"
        )


def _detect_face_ssd(frame: np.ndarray, target_size: int, conf_threshold: float = 0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    _ssd_net.setInput(blob)
    detections = _ssd_net.forward()

    best_conf, best_box = 0.0, None
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_conf, best_box = conf, box.astype(int)

    if best_box is not None and best_conf > conf_threshold:
        x1, y1, x2, y2 = best_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            face = frame[y1:y2, x1:x2]
            return cv2.resize(face, (target_size, target_size))
    return None


def _detect_face_haar(frame: np.ndarray, target_size: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _haar_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
        pad = int(0.1 * min(w, h))
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)
        if w > 30 and h > 30:
            face = frame[y:y + h, x:x + w]
            return cv2.resize(face, (target_size, target_size))
    return None


def detect_and_crop_face(frame: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Deteksi wajah menggunakan detector aktif. Jika gagal, return None."""
    face = None
    try:
        if _detector_type == 'dnn_ssd' and _ssd_net is not None:
            face = _detect_face_ssd(frame, target_size)
        elif _detector_type == 'haar':
            face = _detect_face_haar(frame, target_size)
    except Exception:
        face = None

    return face  # Kembalikan None jika tidak ada wajah


# =============================================================================
# EKSTRAKSI FRAME (dengan validasi)
# =============================================================================
def extract_frames_uniform(video_path: str, num_frames: int, face_size: int = 224):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    detection_success = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        face = detect_and_crop_face(frame, face_size)
        if face is not None:
            detection_success += 1
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            frames.append(face_rgb)
        else:
            # 🔧 FIX: Jika gagal deteksi, jangan append apa-apa (tidak pakai resize full frame)
            pass

    cap.release()

    # 🔧 FIX: Jika kurang dari 50% frame terdeteksi, anggap gagal
    if len(frames) < num_frames * 0.5:
        return None

    # Jika kurang dari num_frames, padding dengan duplikasi frame terakhir
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if frames else np.zeros((face_size, face_size, 3), dtype=np.uint8))

    return frames[:num_frames]


# =============================================================================
# FUNGSI UTAMA EKSTRAKSI
# =============================================================================
def extract_visual_frames(
    root_path: str,
    output_path: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    face_size: int = DEFAULT_FACE_SIZE,
    force: bool = False,
    detector: str = 'dnn_ssd'
) -> dict:
    init_detector(preferred=detector)

    stats = {'total': 0, 'processed': 0, 'skipped': 0, 'failed': 0, 'low_quality': 0}
    os.makedirs(output_path, exist_ok=True)

    mp4_files = glob.glob(f'{root_path}/**/01-*.mp4', recursive=True)
    stats['total'] = len(mp4_files)

    desc = f"Face Crop ({_detector_type})"
    for video_path in tqdm(mp4_files, desc=desc, unit="file"):
        filename = os.path.basename(video_path)
        rel_path = os.path.relpath(video_path, root_path)
        npy_name = filename.replace('.mp4', CACHE_SUFFIX)
        npy_dir = os.path.join(output_path, os.path.dirname(rel_path))
        npy_path = os.path.join(npy_dir, npy_name)

        if not force and os.path.exists(npy_path):
            try:
                arr = np.load(npy_path)
                if arr.shape == (num_frames, face_size, face_size, 3) and arr.dtype == np.uint8:
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass

        os.makedirs(npy_dir, exist_ok=True)
        try:
            frames = extract_frames_uniform(video_path, num_frames, face_size)
            if frames is None or len(frames) == 0:
                stats['failed'] += 1
                continue

            # 🔧 FIX: Validasi kualitas — jika semua frame sama (duplikat berlebihan)
            faces_array = np.array(frames, dtype=np.uint8)
            if np.max(faces_array) == np.min(faces_array):
                stats['low_quality'] += 1
                print(f"⚠️ Low quality (all pixels same) for {filename}, skipping.")
                continue

            np.save(npy_path, faces_array)
            stats['processed'] += 1
        except Exception as e:
            stats['failed'] += 1
            print(f"❌ Error processing {filename}: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Ekstraksi Face Crop RAVDESS - DNN-SSD (default) atau Haar (eksplisit)'
    )
    parser.add_argument('--root_path', default='/content/drive/MyDrive/RAVDESS')
    parser.add_argument('--output_path', default='/content/RAVDESS_FACES')
    parser.add_argument('--num_frames', type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument('--face_size', type=int, default=DEFAULT_FACE_SIZE)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--detector', type=str, default='dnn_ssd', choices=['dnn_ssd', 'haar'],
                        help='dnn_ssd (default, akurat) atau haar (fallback, kurang akurat)')

    args = parser.parse_args()

    print("=" * 60)
    print("  📷 EKSTRAKSI FACE CROP")
    print("=" * 60)
    print(f"  Sumber MP4    : {args.root_path}")
    print(f"  Output .npy   : {args.output_path}")
    print(f"  Num Frames    : {args.num_frames}")
    print(f"  Detector      : {args.detector}")
    print("  🔧 FIX: Gagal deteksi wajah -> SKIP (tidak pakai resize full frame).")
    print("=" * 60)

    stats = extract_visual_frames(
        root_path=args.root_path,
        output_path=args.output_path,
        num_frames=args.num_frames,
        face_size=args.face_size,
        force=args.force,
        detector=args.detector
    )

    print("\n" + "=" * 60)
    print("  HASIL EKSTRAKSI")
    print("=" * 60)
    print(f"  Total file MP4        : {stats['total']}")
    print(f"  Berhasil diproses     : {stats['processed']}")
    print(f"  Di-skip (sudah ada)   : {stats['skipped']}")
    print(f"  Gagal (error)         : {stats['failed']}")
    print(f"  Low Quality (<50% deteksi): {stats['low_quality']}")
    print("=" * 60)

    if stats['failed'] > 0 or stats['low_quality'] > 0:
        print("\n⚠️ PERINGATAN: Ada file yang gagal/kualitas rendah. Periksa data.")
        print("   Pastikan koneksi internet stabil untuk download model DNN-SSD.")


if __name__ == '__main__':
    main()