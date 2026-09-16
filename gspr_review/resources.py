"""Resolve review-only weather assets before allocating a model or dataset."""
import copy
import os
from pathlib import Path
import re

DEFAULT_FOG_TABLES = 'TripleMixer-main/tools/fog_sim/integral_lookup_tables_seg_light_0.008beta/original'


def resolve_weather(config, root, environ=None):
    weather = copy.deepcopy(config)
    if weather.get('mode') not in ('physics_fog', 'mixed_physics_weather'):
        return weather
    env = os.environ if environ is None else environ
    fog = weather.setdefault('physics_fog', {})
    raw = env.get('FOG_LOOKUP_DIR') or fog.get('lookup_dir') or DEFAULT_FOG_TABLES
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(root)/path
    path = path.resolve()
    pattern = re.compile(r'alpha_([0-9]+(?:\.[0-9]+)?)\.pickle$')
    alphas = [float(m.group(1)) for f in path.glob('*.pickle') if f.is_file()
              for m in [pattern.search(f.name)] if m]
    if not alphas:
        raise FileNotFoundError(
            f'GSPR review fog tables missing: {path}. Set FOG_LOOKUP_DIR to the directory '
            'directly containing *alpha_*.pickle (include /original for TripleMixer), '
            'or set weather_augmentation.physics_fog.lookup_dir in the experiment YAML.')
    low, high = float(fog.get('alpha_low', .005)), float(fog.get('alpha_high', .03))
    if fog.get('fixed_alpha') is None and fog.get('alpha_sampling', 'lookup_discrete').lower() == 'lookup_discrete' and not any(low <= a <= high for a in alphas):
        raise ValueError(f'No fog table alpha in configured [{low}, {high}] at {path}')
    fog['lookup_dir'] = str(path)
    return weather
