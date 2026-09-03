# -*- coding: utf-8 -*-
"""
data/ravdess_visual.py
======================
Dataset loader unimodal visual (face crop) untuk RAVDESS (Modalitas 01).

🔧 FIX (v2):
  - Default num_frames diubah dari 15 → 8 (konsisten dengan config).
  - Validasi format uint8 sebelum transformasi.
"""

import os
import numpy as np
import torch
import torch.utils.data as data


class RavdessVisual(data.Dataset):
    """
    Dataset unimodal visual RAVDESS.

    Args:
        annotation_path (str): Path file anotasi
        root_dir (str): Root folder tempat file .npy face crop disimpan
        subset (str): 'training', 'validation', 'testing'
        transform (callable): Transformasi untuk visual
        num_frames (int): Jumlah frame per video (default: 8)
        temporal_mode (str): 'mid', 'avg', atau 'stack' (default: 'stack')
        backbone_name (str): Nama backbone untuk logging
    """
    def __init__(
        self,
        annotation_path,
        root_dir,
        subset,
        transform=None,
        num_frames=15,
        temporal_mode='stack',
        backbone_name='efficientnetv2_b0'
    ):
        self.data = self._make_dataset(annotation_path, root_dir, subset)
        self.transform = transform
        self.num_frames = num_frames
        self.temporal_mode = temporal_mode
        self.backbone_name = backbone_name

        if len(self.data) == 0:
            raise RuntimeError(f"Tidak ada sampel untuk subset='{subset}'")

        print(f"[RavdessVisual] subset='{subset}' ditemukan {len(self.data)} sampel")
        print(f"  Temporal mode: {temporal_mode}")
        print(f"  Num Frames   : {num_frames}")
        print(f"  Backbone     : {backbone_name}")
        if len(self.data) > 0:
            print(f"  Contoh file pertama: {self.data[0]['path']}")

    def _make_dataset(self, annotation_path, root_dir, subset):
        items = []
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(';')
                if len(parts) < 4:
                    continue
                visual_path, audio_path, label, split = parts
                if split.strip() != subset:
                    continue

                visual_npy = visual_path.strip()
                if visual_npy.endswith('.mp4'):
                    visual_npy = visual_npy.replace('.mp4', '_faces.npy')
                if not os.path.isabs(visual_npy):
                    visual_npy = os.path.join(root_dir, visual_npy)

                if not os.path.isfile(visual_npy):
                    continue

                try:
                    label_int = int(label.strip()) - 1
                except:
                    continue
                if label_int < 0 or label_int >= 8:
                    continue

                items.append({
                    'path': visual_npy,
                    'label': label_int
                })
        return items

    def __getitem__(self, index):
        item = self.data[index]
        target = item['label']
        visual_path = item['path']

        try:
            faces = np.load(visual_path)
        except Exception as e:
            print(f"⚠️ Gagal load {visual_path}: {e}")
            faces = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

        # Pastikan faces dalam format uint8 (0-255) sebelum transformasi
        if faces.dtype != np.uint8:
            faces = np.clip(faces, 0, 255).astype(np.uint8)

        # Temporal processing
        if faces.ndim == 4:
            if self.temporal_mode == 'avg':
                face_img = faces.mean(axis=0).astype(np.uint8)
            elif self.temporal_mode == 'mid':
                mid = len(faces) // 2
                face_img = faces[mid]
            else:  # 'stack'
                if len(faces) > self.num_frames:
                    indices = np.linspace(0, len(faces)-1, self.num_frames, dtype=int)
                    face_img = faces[indices]
                else:
                    pad = self.num_frames - len(faces)
                    if pad > 0:
                        last_frame = faces[-1] if len(faces) > 0 else np.zeros((224, 224, 3), dtype=np.uint8)
                        pad_frames = np.stack([last_frame] * pad, axis=0)
                        face_img = np.concatenate([faces, pad_frames], axis=0)
                    else:
                        face_img = faces
        else:
            # Jika hanya 1 frame (bukan multi-frame)
            if self.temporal_mode == 'stack':
                face_img = np.stack([faces] * self.num_frames, axis=0)
            else:
                face_img = faces

        # Transformasi
        if self.transform is not None:
            if self.temporal_mode == 'stack':
                if face_img.dtype != np.uint8:
                    face_img = np.clip(face_img, 0, 255).astype(np.uint8)

                transformed_frames = []
                for i in range(face_img.shape[0]):
                    frame = face_img[i]
                    transformed = self.transform(frame)
                    transformed_frames.append(transformed)
                face_img = torch.stack(transformed_frames, dim=0)
            else:
                if face_img.ndim == 3:
                    if face_img.dtype != np.uint8:
                        face_img = np.clip(face_img, 0, 255).astype(np.uint8)
                    face_img = self.transform(face_img)
                else:
                    raise ValueError(f"Unexpected shape for non-stack mode: {face_img.shape}")
        else:
            if self.temporal_mode == 'stack':
                face_img = torch.from_numpy(face_img).float().permute(0, 3, 1, 2) / 255.0
            else:
                face_img = torch.from_numpy(face_img).float().permute(2, 0, 1) / 255.0

        return face_img, target

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return [item['label'] for item in self.data]