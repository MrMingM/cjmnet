#!/bin/sh
# Standalone server CPU-only inspection. Does not execute training or change any artifact.
set -eu
/home/cjm/miniconda3/envs/opencood/bin/python -B - <<'PY'
from pathlib import Path
import json
import numpy as np
import torch
torch.set_num_threads(1)
r = Path('/data/cjm/datasets/logs/uecp_gspr_v2xr_v1')
p = [r/'smoke/E0/baseline/epochs/01', r/'smoke_replay/E0/epochs/01']
a, b = [torch.load(x/'resume.pth', map_location='cpu', weights_only=False) for x in p]
def equal(x, y):
    if type(x) is not type(y): return False
    if isinstance(x, torch.Tensor): return x.dtype == y.dtype and torch.equal(x, y)
    if isinstance(x, np.ndarray): return x.dtype == y.dtype and np.array_equal(x, y)
    if isinstance(x, dict): return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)): return len(x) == len(y) and all(equal(v, w) for v, w in zip(x, y))
    return bool(x == y)
fields = ['contract_identity','variant','epoch','position','global_step','order_identity','microbatch','accumulation_position']
out = {'DIAGNOSIS_ONLY': True, 'metadata_equal': {k: equal(a[k], b[k]) for k in fields},
       'sections_exact_equal': {k: equal(a[k], b[k]) for k in ['actual','rng','scheduler','gradients','optimizer']},
       'first_update_gradients_left': a['gradients'], 'first_update_gradients_right': b['gradients']}
out['sampling_equal'] = json.loads((p[0]/'sampling.json').read_text()) == json.loads((p[1]/'sampling.json').read_text())
out['initialization_equal'] = json.loads((p[0].parents[1]/'initialization.json').read_text()) == json.loads((p[1].parents[1]/'initialization.json').read_text())
tolerance = json.loads((r/'config.json').read_text())['resume_replay']
rows = []
for k, x in a['model'].items():
    y = b['model'][k]
    if not x.is_floating_point() or torch.equal(x, y): continue
    x, y = x.double().numpy(), y.double().numpy()
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        rows.append({'path': k, 'nonfinite': True}); continue
    d = np.abs(x-y)
    rows.append({'path': k, 'shape': list(x.shape), 'max_abs': float(d.max()),
        'relative_l2': float(np.linalg.norm(x-y)/max(np.linalg.norm(x), np.linalg.norm(y), 1e-30)),
        'failed_elements': int((d > tolerance['atol']+tolerance['rtol']*np.abs(y)).sum()), 'elements': int(x.size)})
out['changed_model_tensors'] = len(rows)
out['failed_model_tensors'] = sum(v.get('nonfinite', False) or v.get('failed_elements', 0) > 0 for v in rows)
out['first_failed_model_tensor'] = next((v for v in rows if v.get('nonfinite') or v.get('failed_elements', 0)), None)
out['largest_model_changes'] = sorted(rows, key=lambda v: v.get('relative_l2', float('inf')), reverse=True)[:5]
print(json.dumps(out, indent=2, allow_nan=False))
print('DIAGNOSIS_FINISHED: no experiment files changed.')
PY
