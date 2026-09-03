# -*- coding: utf-8 -*-
"""
data/ravdess_audio.py
=====================
Dataset loader unimodal audio untuk RAVDESS (Modalitas 01).

🔧 FIX (v3): Perbaikan bug remap path.
🔧 FIX (v4): Tambahkan konversi float32 di __getitem__.
"""

import os
import numpy as np
import torch
import torch.utils.data as data


EXPECTED_SHAPE = {
    'audio_1dcnn': (120, 224),
    'audio_2dcnn': (128, 224),
}

FILE_SUFFIX = {
    'audio_1dcnn': '_mfcc_delta.npy',
    'audio_2dcnn': '_melspec.npy',
}


class RavdessAudio(data.Dataset):
    def __init__(
        self,
        annotation_path,
        root_dir,
        subset,
        transform=None,
        backbone_name='audio_1dcnn',
        input_mode=None
    ):
        self.backbone_name = backbone_name
        self.root_dir = root_dir
        self.input_mode = input_mode or ('2d' if backbone_name == 'audio_2dcnn' else '1d')
        self.expected_shape = EXPECTED_SHAPE.get(backbone_name, (120, 224))
        self.file_suffix = FILE_SUFFIX.get(backbone_name, '_mfcc_delta.npy')

        self.data = self._make_dataset(annotation_path, root_dir, subset)
        self.transform = transform

        if len(self.data) == 0:
            raise RuntimeError(
                f"Tidak ada sampel untuk subset='{subset}' (backbone={backbone_name}). "
                f"Cek apakah file dengan suffix '{self.file_suffix}' sudah diekstrak "
                f"ke root_dir='{root_dir}'."
            )

        print(f"[RavdessAudio] subset='{subset}' ditemukan {len(self.data)} sampel")
        print(f"  Backbone      : {backbone_name}")
        print(f"  Root Dir      : {root_dir}")
        print(f"  File Suffix   : {self.file_suffix}")
        print(f"  Expected Shape: {self.expected_shape}")
        print(f"  Contoh file pertama: {self.data[0]['path']}")

    def _rebuild_path(self, original_audio_path: str, root_dir: str) -> str:
        filename = os.path.basename(original_audio_path)
        actor_folder = os.path.basename(os.path.dirname(original_audio_path))

        for known_suffix in FILE_SUFFIX.values():
            if filename.endswith(known_suffix):
                base_name = filename[: -len(known_suffix)]
                filename = base_name + self.file_suffix
                break
        else:
            base_name = os.path.splitext(filename)[0]
            filename = base_name + self.file_suffix

        return os.path.join(root_dir, actor_folder, filename)

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
                video_path, audio_path, label, split = parts
                if split.strip() != subset:
                    continue

                original_audio_path = audio_path.strip()
                audio_npy = self._rebuild_path(original_audio_path, root_dir)

                if not os.path.isfile(audio_npy):
                    continue

                try:
                    label_int = int(label.strip()) - 1
                except Exception:
                    continue
                if label_int < 0 or label_int >= 8:
                    continue

                items.append({'path': audio_npy, 'label': label_int})
        return items

    def __getitem__(self, index):
        item = self.data[index]
        target = item['label']
        audio_path = item['path']

        try:
            # 🔧 FIX: Konversi ke float32
            spec = np.load(audio_path).astype(np.float32)
        except Exception as e:
            print(f"⚠️ Gagal load {audio_path}: {e}")
            spec = np.zeros(self.expected_shape, dtype=np.float32)

        if spec.ndim == 3:
            if spec.shape[0] == 1:
                spec = spec.squeeze(0)
            elif spec.shape[-1] == 1:
                spec = spec.squeeze(-1)

        if self.transform is not None:
            spec = self.transform(spec)
        else:
            spec = torch.from_numpy(spec).float()
            if self.input_mode == '2d' and spec.dim() == 2:
                spec = spec.unsqueeze(0)

        # 🔧 FIX: Jaga-jaga jika transform mengembalikan float64
        if isinstance(spec, torch.Tensor) and spec.dtype != torch.float32:
            spec = spec.float()

        return spec, target

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return [item['label'] for item in self.data]