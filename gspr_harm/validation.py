"""Read-only protocol checks for expansion to the full validation split."""
import json
import math
from pathlib import Path


def validation_indices(ds):
    if len(ds) < 1:
        raise ValueError('Empty validation dataset')
    return list(range(len(ds)))


def check_scene_split(train_root, validation_root):
    train_root, validation_root = Path(train_root).resolve(), Path(validation_root).resolve()
    if train_root == validation_root:
        raise ValueError('Validation must not use the training root')
    train = {p.name for p in train_root.iterdir() if p.is_dir()}
    validation = {p.name for p in validation_root.iterdir() if p.is_dir()}
    overlap = train & validation
    if overlap:
        raise ValueError('Train/validation scene names overlap: '+str(sorted(overlap)))
    if not validation:
        raise ValueError('No validation scenes')
    return sorted(validation)


def frozen_epsilon(run, weather, checkpoint_hash, config_hash, communication, seed, weather_config):
    run = Path(run)
    audit = json.loads((run/'audit.json').read_text(encoding='utf-8'))
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'complete' or not summary.get('frozen_model_unchanged'):
        raise ValueError('Calibration run must be completed with unchanged frozen model')
    if audit.get('args', {}).get('full_validation') or not audit.get('purpose', '').startswith('TRAIN'):
        raise ValueError('Use a TRAIN diagnostic calibration, not validation/test calibration')
    for key, expected in [('checkpoint_sha256', checkpoint_hash), ('frontend_config_sha256', config_hash),
                          ('communication', communication), ('seed', seed)]:
        if audit.get(key) != expected:
            raise ValueError('Calibration protocol mismatch: '+key)
    protocol = json.loads((run/weather/'data_protocol.json').read_text(encoding='utf-8'))
    if protocol['weather'] != weather_config:
        raise ValueError('Weather settings differ from calibration run: '+weather)
    result = json.loads((run/weather/'epsilon.json').read_text(encoding='utf-8'))
    epsilon = float(result['epsilon'])
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Invalid stored epsilon')
    return epsilon, {'source_run': str(run.resolve()), 'source_calibration': result,
                     'epsilon': epsilon, 'policy': 'frozen from training; never refitted on validation'}
