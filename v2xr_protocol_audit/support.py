"""Standard-library support; historical packages are read-only dependencies."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent / 'v2xr_official_repro'
sys.path.insert(1, str(REPRO))
from common import atomic_json, identity, jsonable, lock, no_test, read_json, server_only, sha256

CONDITIONS = ('clean', 'fog', 'snow')
THRESHOLDS = (.3, .5, .7)
DEFINITIONS = {
    'O': dict(name='OFFICIAL_EXACT', gt='condition-specific live official final evaluation GT',
              none='skip entire frame exactly as official inference.py; empty non-None is counted'),
    'O*': dict(name='OFFICIAL_GT_NO_SKIP', gt='identical condition-specific GT and ordering as O',
               none='zero predictions; GT contributes to denominator'),
    'F': dict(name='OFFICIAL_CLEAN_FIXED_GT',
              gt='live official Clean-selected GT fixed by scene/timestamp/object ID, 6224 for each condition',
              none='zero predictions; GT contributes to denominator'),
}


@contextmanager
def patched(obj, name, replacement):
    previous = getattr(obj, name)
    setattr(obj, name, replacement)
    try:
        yield previous
    finally:
        setattr(obj, name, previous)


def verify_sources():
    expected = read_json(HERE / 'source_snapshot.json')['files']
    actual = {p.relative_to(HERE).as_posix(): sha256(p) for p in HERE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if actual != expected:
        difference = dict(missing=sorted(expected.keys() - actual.keys()),
            unexpected=sorted(actual.keys() - expected.keys()),
            changed=sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k]))
        raise ValueError('Synchronize whole v2xr_protocol_audit, including its snapshot: ' + json.dumps(difference))
    return expected


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_jsonl(path, rows):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)


def check_identity(record, field='identity'):
    if identity({k: v for k, v in record.items() if k != field}) != record[field]:
        raise ValueError(f'Invalid {field}')


def tensor_array(tensor):
    return tensor.detach().cpu().numpy()


def corner_array(gt):
    import numpy as np
    return np.asarray([r['corners'] for r in gt], dtype=np.float32).reshape(-1, 8, 3)


def compare_gt(actual, expected, label):
    import numpy as np
    a = {str(r['object_id']): r for r in actual}
    b = {str(r['object_id']): r for r in expected}
    if len(a) != len(actual) or len(b) != len(expected) or a.keys() != b.keys():
        raise ValueError(f'GT IDs differ: {label}: missing={sorted(b.keys()-a.keys())}, added={sorted(a.keys()-b.keys())}')
    maximum = 0.0
    exact = True
    for key in a:
        x, y = np.asarray(a[key]['corners']), np.asarray(b[key]['corners'])
        if x.shape != (8, 3) or y.shape != (8, 3) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'Invalid corners: {label}/{key}')
        delta = float(np.max(np.abs(x - y)))
        maximum = max(maximum, delta)
        exact = exact and np.array_equal(x, y)
        if delta > 1e-4:
            raise ValueError(f'GT geometry differs: {label}/{key}: {delta} m')
        if 'box' in a[key] and 'box' in b[key]:
            import math
            x_box, y_box = a[key]['box'], b[key]['box']
            if (len(x_box) != 7 or len(y_box) != 7
                    or not all(math.isfinite(v) for v in x_box + y_box)
                    or max(abs(x_box[i] - y_box[i]) for i in range(6)) > 1e-4
                    or abs(math.atan2(math.sin(x_box[6]-y_box[6]), math.cos(x_box[6]-y_box[6]))) > 1e-5):
                raise ValueError(f'GT center/dimensions/yaw differ: {label}/{key}')
    return dict(max_corner_delta_m=maximum, corners_bitwise_equal=bool(exact))


def prediction_tensors(record):
    import numpy as np
    import torch
    if record['prediction_is_none']:
        if record['prediction_boxes'] is not None or record['prediction_count'] != 0 or record['prediction_empty']:
            raise ValueError('Inconsistent None prediction state')
        return None, None
    boxes = np.asarray(record['prediction_boxes'], dtype=record['box_dtype']).reshape(-1, 8, 3)
    scores = np.asarray(record['scores'], dtype=record['score_dtype']).reshape(-1)
    if (len(boxes) != len(scores) or len(boxes) != record['prediction_count']
            or record['prediction_empty'] != (len(boxes) == 0)
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()):
        raise ValueError('Invalid prediction cache')
    return torch.from_numpy(boxes), torch.from_numpy(scores)


def load_prediction(path, contract, condition, index, frame_key):
    record = read_json(path)
    check_identity(record, 'record_identity')
    if (record['contract_identity'] != contract['identity'] or record['condition'] != condition
            or record['index'] != index or record['frame_key'] != frame_key):
        raise ValueError(f'Prediction cache identity mismatch: {path}')
    return record
