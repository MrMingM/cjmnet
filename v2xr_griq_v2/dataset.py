"""Versioned manifests and bounded cache readers for GRIQ v2."""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from v2xr_griq_experiments.dataset import (
    DATA, F0, REPO, WEATHERS, bind, import_runtime, load, read, save, sha,
)

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('GRIQ2_OUT', '/data/cjm/datasets/logs/v2xr-griq-v2'))
V1 = DATA/'logs/v2xr-griq-experiments-v1'
WEIGHT = 'a5e12d10cde649c5a198b814277134ae06f8069a5b8e4d32505438c19b1fe8b1'
SEEDS = (20261017, 20261018, 20261019)


def row_path(root, area, weather, index):
    return root / area / weather / f'{index:06d}.npz'


def nested(all_ids, original, limit=128):
    if not set(original) <= set(all_ids) or len(original) != min(32,len(all_ids)):
        raise ValueError('Historical 32-frame sample does not match source indices')
    remaining = sorted(set(all_ids)-set(original))
    n = min(limit,len(all_ids))-len(original)
    extra = [remaining[i] for i in np.linspace(0,len(remaining)-1,n,dtype=int)] if n else []
    return sorted([*original,*extra])


@lru_cache(maxsize=16)
def frame(row_json, floor, cap, local):
    import json
    row = json.loads(row_json)
    d = load(row,dict(floor=floor,cap=cap))
    if local:
        with np.load(row['local'],allow_pickle=False) as z:
            if not np.array_equal(z['ids'],d['ids']):
                raise ValueError(f'Local evidence and anchor IDs differ: {row["meta"]}')
            d['local'] = z['evidence']
    return d


def get(row, policy, local=False):
    import json
    return frame(json.dumps(row,sort_keys=True),policy['floor'],policy['cap'],local)


def root_for(smoke):
    return OUT/'smoke' if smoke else OUT
