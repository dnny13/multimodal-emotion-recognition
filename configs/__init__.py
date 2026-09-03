# -*- coding: utf-8 -*-
"""
configs/__init__.py
===================
Fungsi untuk memuat konfigurasi dari file YAML dengan dukungan inheritance dan deteksi siklus.
"""

import os
import yaml

def _deep_merge(base, override):
    """Menggabungkan dua dictionary secara rekursif."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base

def load_config(config_path, _visited=None):
    """
    Memuat konfigurasi dari file YAML. Mendukung inheritance melalui key 'inherit'.
    Mendeteksi siklus untuk menghindari infinite recursion.
    """
    if _visited is None:
        _visited = set()

    # Gunakan absolute path untuk deteksi siklus yang konsisten
    abs_path = os.path.abspath(config_path)
    if abs_path in _visited:
        raise RecursionError(f"Circular inheritance detected: {config_path}")
    _visited.add(abs_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'inherit' in config:
        base_ref = config['inherit']
        base_dir = os.path.dirname(config_path)

        # Tentukan base path
        if os.path.isabs(base_ref):
            base_path = base_ref
        else:
            # Jika base_ref hanya nama file (tanpa path)
            if os.path.sep not in base_ref:
                base_path = os.path.join(base_dir, base_ref)
            else:
                base_path = os.path.join(base_dir, base_ref)

        # Load base config dengan _visited yang sama
        base_config = load_config(base_path, _visited)

        # Deep merge
        merged = base_config.copy()
        _deep_merge(merged, config)
        config = merged

    return config

__all__ = ['load_config']