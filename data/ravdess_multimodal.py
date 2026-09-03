# -*- coding: utf-8 -*-
"""
data/ravdess_multimodal.py
==========================
Dataset loader multimodal RAVDESS untuk feature caching & fusion.
"""

import os
import re
import numpy as np
import torch
import torch.utils.data as data


class RavdessMultimodal(data.Dataset):
    def __init__(
        self,
        annotation_path: str,
        audio_root: str,
        visual_root: str,
        subset: str,
        transform_audio=None,
        transform_visual=None,
        num_frames: int = 15,
        audio_backbone: str = 'audio_1dcnn',
        visual_backbone: str = 'mobilenetv3_small_100',
        audio_input_mode: str = '1d',
        cache_size: int = 64,
        temporal_offset: bool = True,
        max_offset_ms: int = 200
    ):
        self.data = self._make_dataset(annotation_path, audio_root, visual_root, subset)
        self.transform_audio = transform_audio
        self.transform_visual = transform_visual
        self.num_frames = num_frames
        self.audio_backbone = audio_backbone
        self.visual_backbone = visual_backbone
        self.audio_input_mode = audio_input_mode
        self.subset = subset

        self._audio_cache = {}
        self._visual_cache = {}
        self._cache_size = cache_size

        self.temporal_offset = temporal_offset
        self.max_offset_ms = max_offset_ms

        if len(self.data) == 0:
            raise RuntimeError(f"Tidak ada sampel untuk subset='{subset}'")

        print(f"[RavdessMultimodal] subset='{subset}' ditemukan {len(self.data)} sampel")
        print(f"  Audio Backbone : {audio_backbone} (input_mode={audio_input_mode})")
        print(f"  Visual Backbone: {visual_backbone}")
        print(f"  Num Frames     : {num_frames}")
        print(f"  Temporal Offset: {'✅' if temporal_offset and subset=='training' else '❌'}")
        if len(self.data) > 0:
            print(f"  Contoh file pertama:")
            print(f"    Audio : {self.data[0]['audio_path']}")
            print(f"    Visual: {self.data[0]['visual_path']}")

    def _make_dataset(self, annotation_path, audio_root, visual_root, subset):
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

                audio_npy = audio_path.strip()
                if not audio_npy.endswith('.npy'):
                    audio_npy = audio_npy.replace('.wav', '_mfcc_delta.npy')
                if not os.path.isabs(audio_npy):
                    audio_npy = os.path.join(audio_root, audio_npy)

                visual_npy = video_path.strip()
                if visual_npy.endswith('.mp4'):
                    visual_npy = visual_npy.replace('.mp4', '_faces.npy')
                if not os.path.isabs(visual_npy):
                    visual_npy = os.path.join(visual_root, visual_npy)

                if not os.path.isfile(audio_npy) or not os.path.isfile(visual_npy):
                    continue

                try:
                    label_int = int(label.strip()) - 1
                except:
                    continue
                if label_int < 0 or label_int >= 8:
                    continue

                items.append({
                    'audio_path': audio_npy,
                    'visual_path': visual_npy,
                    'label': label_int
                })
        return items

    def _load_audio(self, path):
        if path in self._audio_cache:
            return self._audio_cache[path].copy()

        try:
            spec = np.load(path)
        except Exception as e:
            print(f"⚠️ Gagal load audio {path}: {e}")
            spec = np.zeros((120, 224), dtype=np.float32)

        if spec.ndim == 3:
            if spec.shape[0] == 1:
                spec = spec.squeeze(0)
            elif spec.shape[-1] == 1:
                spec = spec.squeeze(-1)

        if len(self._audio_cache) < self._cache_size:
            self._audio_cache[path] = spec.copy()
        return spec

    def _load_audio_with_offset(self, path, offset_ms):
        spec = self._load_audio(path)
        if offset_ms != 0:
            total_duration_ms = 3000
            total_frames = spec.shape[1]
            ms_per_frame = total_duration_ms / total_frames
            offset_frames = int(offset_ms / ms_per_frame)
            if offset_frames != 0:
                spec = np.roll(spec, shift=offset_frames, axis=1)
        return spec

    def _load_visual(self, path):
        if path in self._visual_cache:
            return self._visual_cache[path].copy()

        try:
            faces = np.load(path)
        except Exception as e:
            print(f"⚠️ Gagal load visual {path}: {e}")
            faces = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

        if faces.ndim == 4:
            if len(faces) > self.num_frames:
                indices = np.linspace(0, len(faces) - 1, self.num_frames, dtype=int)
                faces = faces[indices]
            elif len(faces) < self.num_frames:
                pad = self.num_frames - len(faces)
                last_frame = faces[-1] if len(faces) > 0 else np.zeros((224, 224, 3), dtype=np.uint8)
                pad_frames = np.stack([last_frame] * pad, axis=0)
                faces = np.concatenate([faces, pad_frames], axis=0)
        else:
            faces = np.stack([faces] * self.num_frames, axis=0)

        if faces.dtype != np.uint8:
            faces = np.clip(faces, 0, 255).astype(np.uint8)

        if len(self._visual_cache) < self._cache_size:
            self._visual_cache[path] = faces.copy()
        return faces

    def _apply_audio_augmentation(self, spec: np.ndarray) -> np.ndarray:
        spec = spec.copy()

        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.005, spec.shape)
            spec = spec + noise

        if np.random.rand() < 0.3:
            T = spec.shape[1]
            mask_len = np.random.randint(0, int(T * 0.1) + 1)
            mask_start = np.random.randint(0, max(1, T - mask_len))
            if mask_start + mask_len < T:
                spec[:, mask_start:mask_start + mask_len] = 0.0

        if np.random.rand() < 0.3:
            F = spec.shape[0]
            mask_len = np.random.randint(0, int(F * 0.1) + 1)
            mask_start = np.random.randint(0, max(1, F - mask_len))
            if mask_start + mask_len < F:
                spec[mask_start:mask_start + mask_len, :] = 0.0

        return spec

    def __getitem__(self, index):
        item = self.data[index]
        target = item['label']

        visual_frames = self._load_visual(item['visual_path'])

        if self.temporal_offset and self.subset == 'training':
            offset_ms = np.random.randint(-self.max_offset_ms, self.max_offset_ms)
            audio_spec = self._load_audio_with_offset(item['audio_path'], offset_ms)
        else:
            audio_spec = self._load_audio(item['audio_path'])

        if self.subset == 'training':
            audio_spec = self._apply_audio_augmentation(audio_spec)

        if self.transform_audio is not None:
            audio_tensor = self.transform_audio(audio_spec)
        else:
            audio_tensor = torch.from_numpy(audio_spec).float()

        if self.transform_visual is not None:
            transformed_frames = [self.transform_visual(visual_frames[i]) for i in range(visual_frames.shape[0])]
            visual_tensor = torch.stack(transformed_frames, dim=0)
        else:
            visual_tensor = torch.from_numpy(visual_frames).float().permute(0, 3, 1, 2) / 255.0

        return audio_tensor, visual_tensor, target

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return [item['label'] for item in self.data]

    def get_audio_paths(self):
        return [item['audio_path'] for item in self.data]

    def get_visual_paths(self):
        return [item['visual_path'] for item in self.data]

    def get_actor_ids(self):
        actor_ids = []
        for item in self.data:
            path = item.get('audio_path', item.get('visual_path', ''))
            actor_id = -1

            dirname = os.path.basename(os.path.dirname(path))
            if dirname.startswith('Actor_'):
                try:
                    actor_id = int(dirname.split('_')[1])
                except:
                    pass

            if actor_id == -1:
                match = re.search(r'Actor_(\d+)', path)
                if match:
                    try:
                        actor_id = int(match.group(1))
                    except:
                        pass

            actor_ids.append(actor_id)
        return actor_ids

    def get_subset_by_actors(self, actor_list):
        indices = []
        actor_ids = self.get_actor_ids()
        for i, aid in enumerate(actor_ids):
            if aid in actor_list:
                indices.append(i)
        return torch.utils.data.Subset(self, indices)