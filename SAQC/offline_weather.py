"""Read one fixed materialized weather split without invoking online weather."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from gspr_communication.runtime import sha256
from gspr_evidence import runtime as er


DEFAULT_ROOT = '/data/cjm/datasets/opv2v-physics-fixed-v1'


def make_loader(state, args, *, split, weather):
    if split not in ('train', 'validate') or weather not in (
            'clean', 'fog', 'rain', 'snow', 'mixed'):
        raise ValueError(f'Unsupported fixed weather request: {split}/{weather}')
    root = Path(args.weather_dataset_root).resolve()
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'Generate the fixed weather dataset first: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise RuntimeError(f'Fixed weather dataset is incomplete: {root}')
    if manifest.get('frontend_config_sha256') != sha256(args.frontend_config):
        raise ValueError('Fixed weather dataset frontend config differs')
    if manifest.get('weather_config') != state['hypes'].get('weather_augmentation'):
        raise ValueError('Fixed weather simulation config differs')
    if manifest.get('lidar_range') != list(
            state['hypes']['preprocess']['cav_lidar_range']):
        raise ValueError('Fixed weather LiDAR range differs')
    source_key = 'root_dir' if split == 'train' else 'validate_dir'
    source = Path(state['hypes'][source_key]).resolve()
    if manifest.get('source_roots', {}).get(split) != str(source):
        raise ValueError(f'Fixed weather {split} source root differs')

    hypes = copy.deepcopy(state['hypes'])
    if weather != 'clean':
        selected = root / weather / split
        if not selected.is_dir():
            raise FileNotFoundError(selected)
        hypes[source_key] = str(selected)
    # The PCD already contains the degradation. The existing DataLoader sees
    # it as its sole input branch; online augmentation is disabled explicitly.
    hypes.pop('weather_augmentation', None)
    return er.make_loader(
        hypes, state['options'], train=(split == 'train'),
        weather='clean', smoke=args.smoke)
