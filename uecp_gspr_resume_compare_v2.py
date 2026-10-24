"""Approved recovery comparison, pure NumPy. Training mathematics are unchanged."""
import math
import numbers

import numpy as np

from uecp_gspr_resume_policy_proposal import ATOL, RTOL, floating_array


KEYS = ('model', 'optimizer', 'scheduler', 'criterion', 'rng', 'global_step', 'position',
        'order_identity', 'microbatch', 'accumulation_position', 'contract_identity',
        'variant', 'epoch', 'totals', 'actual', 'amp_scaler')
POLICY = dict(revision=2, atol=ATOL, rtol=RTOL,
    model='Original finite elementwise abs(a-b)<=atol+rtol*abs(b)',
    moments='Only Adam exp_avg/exp_avg_sq/max_exp_avg_sq: original elementwise OR '
            'L2(a-b)<=atol+rtol*max(L2(a),L2(b)), independently per tensor',
    exact='Post-load state, shapes/dtypes, optimizer step/param_groups, RNG, LR/scheduler, '
          'input digests, metadata, counters and identities',
    comparison_start='A single identical committed checkpoint, never independent first-epoch training',
    gradient_differences='Diagnostic only; not an additional recovery gate',
    limits='Engineering recovery check; no claim of bitwise training determinism or long-run equivalence')


def compare_states(left, right, checkpoint_verified, exact_load_verified):
    failures, numeric = [], []
    if not checkpoint_verified or not exact_load_verified:
        return dict(status='FAILED', failures=[dict(path='', reason='Common checkpoint/exact load evidence missing')], numeric=[])
    missing = [k for k in KEYS if k not in left or k not in right]
    if missing:
        return dict(status='FAILED', failures=[dict(path=k, reason='Missing checkpoint field') for k in missing], numeric=[])
    def fail(path, reason):
        failures.append(dict(path=path, reason=reason))
    def walk(a, b, path):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            result = floating_array(a, b, path, same_checkpoint_verified=True, exact_load_verified=True)
            result = {k: v for k, v in result.items() if k not in ('policy_applied', 'formal_recovery_gate_passed')}
            numeric.append(result)
            if not result['accepted']:
                fail(path, result['reason'])
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                fail(path, 'Dictionary keys/type differ')
                return
            for k in a:
                walk(a[k], b[k], path+'/'+str(k))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                fail(path, 'Sequence type/length differ')
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path+'/'+str(i))
        elif isinstance(a, numbers.Real) and not isinstance(a, (bool, int, np.integer)):
            if not isinstance(b, numbers.Real) or not math.isfinite(a) or not math.isfinite(b):
                fail(path, 'Nonfinite/incompatible scalar')
            elif path in ('totals/loss', 'totals/fused', 'totals/single', 'totals/auxiliary_each'):
                if abs(a-b) > ATOL+RTOL*abs(b):
                    fail(path, 'Loss log exceeds original tolerance')
            elif type(a) is not type(b) or a != b:
                fail(path, 'Exact scalar/type differs')
        elif type(a) is not type(b) or a != b:
            fail(path, 'Exact scalar/type differs')
    for key in KEYS:
        walk(left[key], right[key], key)
    return dict(status='COMPLETE' if not failures else 'FAILED', failures=failures, numeric=numeric)
