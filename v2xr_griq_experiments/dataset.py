"""Versioned cache and deterministic scene partitions shared by all groups."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = Path('/data/cjm/datasets')
RUNTIME = DATA / 'v2xr-official-runtime-v1'
OUT = Path(os.environ.get('GRIQ_OUT', '/data/cjm/datasets/logs/v2xr-griq-experiments-v1'))
F0 = DATA / 'logs/v2xr-radar-ra-rb-v1/F0'
WEATHERS = ('clean', 'fog', 'snow')
SEED = 20261017
FLOORS = (.10, .05, .035, .02)
CAPS = (128, 256, 512)
TRAIN_PER_SCENE = 32
FAILED_TRAIN_KEY = ['2024_06_27_10_09_59', '000330']
LEGACY_CONTRACT = '1adc5454f743ffd923dc1362f917323186abaa6b2854426a30db3a8d735085ea'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def bind(path, obj):
    if path.exists():
        if read(path) != obj:
            raise ValueError(f'Identity changed: {path}')
    else:
        save(path, obj)


def partitions(keys):
    scenes = sorted({key[0] for key in keys})
    order = np.random.default_rng(SEED).permutation(scenes).tolist()
    if len(order) != 32:
        raise ValueError(f'Expected 32 training scenes, got {len(order)}')
    return dict(fit=sorted(order[:22]), select=sorted(order[22:27]), calibrate=sorted(order[27:]))


def accepted_frame_contracts(root, split):
    current = sha(OUT / 'CONTRACT.json')
    if root == OUT and split == 'train':
        legacy = read(OUT / 'CONTRACT.json').get('compatible_train_cache')
        if legacy:
            return {current, legacy}
    return {current}


def smoke_root():
    return OUT / ('smoke_gt_fix' if (OUT / 'CONTRACT.v1.json').exists() else 'smoke')


def paths(root, split, weather, index):
    f = root / 'shared_candidates' / split / weather / f'{index:06}'
    return f.with_suffix('.npz'), f.with_suffix('.json')


def load(row, policy):
    with np.load(row['npz'], allow_pickle=False) as data:
        result = dict(data)
    keep = pool_mask(result['prob'], result['raw_rank'], result['native_threshold'].item(),
                     policy['floor'], policy['cap'])
    for key in ('ids', 'x', 'iou', 'gt_index', 'decoded', 'corners', 'prob', 'logit', 'anchor', 'raw_rank', 'stage', 'y30', 'y50', 'y70'):
        result[key] = result[key][keep]
    result['meta'] = read(row['meta'])
    return result


def pool_mask(prob, raw_rank, native, floor, cap):
    # Every native-eligible proposal is kept, even if the low-score quota is exhausted.
    return (prob > native) | ((prob > floor) & (raw_rank < cap))


def manifest(root):
    result = read(root / 'DATA_AUDIT.json')
    if result['contract'] != sha(root / 'CONTRACT.json'):
        raise ValueError('Shared cache contract differs')
    return result


def frame_rows(index, split):
    return [r for r in index['files'] if r['split'] == split]


def import_runtime():
    import sys
    sys.path.insert(0, str(REPO / 'v2xr_radar_ra_rb'))
    sys.path.insert(0, str(REPO / 'v2xr_radar_evidence'))
    from context import setup
    return setup()
