# -*- coding: utf-8 -*-
"""
configs/__init__.py
===================
Package untuk konfigurasi YAML.
Menyediakan fungsi load_config() untuk memuat file konfigurasi.
"""

import os
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'inherit' in config:
        inherit_path = config.pop('inherit')
        base_dir = os.path.dirname(config_path)
        inherit_path = os.path.join(base_dir, inherit_path)

        if os.path.exists(inherit_path):
            base_config = load_config(inherit_path)
            merged = {**base_config, **config}
            return merged

    return config

__all__ = ['load_config']