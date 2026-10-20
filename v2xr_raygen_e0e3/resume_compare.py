"""Resume validation: exact structural state, float logs, tensor-wise Adam moment norms.

NumPy/stdlib only. Training and its hyperparameters are unaffected by this check policy.
"""
import math
import numbers

import numpy as np

from .common import identity

POLICY = dict(revision=2, rtol=1e-4, atol=1e-5,
    model='Original elementwise abs(a-b)<=atol+rtol*abs(b), finite values required',
    optimizer='exp_avg/exp_avg_sq/max_exp_avg_sq: original elementwise test OR '
              'L2(a-b)<=atol+rtol*max(L2(a),L2(b)), separately for each tensor; finite values required. '
              'All other optimizer fields including step/param_groups are exact.',
    logs='criterion_accumulators floating scalars use original elementwise rtol/atol; counters remain exact',
    exact='RNG, shapes,dtypes,keys,group,epoch,scheduler,sampler,LR,step,microbatch,config,input/source identities',
    prerequisite='Exact post-load model/aux/Adam/scheduler/criterion/RNG roundtrip before floating replay acceptance',
    limitation='Post-training replay checks two resumed runs; it does not establish bitwise training determinism.')
POLICY_ID = identity(POLICY)
POLICY_V3 = dict(POLICY, revision=3,
    optimizer='exp_avg/exp_avg_sq/max_exp_avg_sq: original elementwise test OR '
              'RMS(a-b)<=atol+rtol*max(RMS(a),RMS(b)), separately for each tensor; finite values required. '
              'RMS=L2/sqrt(number of elements), equivalently '
              'L2(a-b)<=atol*sqrt(number of elements)+rtol*max(L2(a),L2(b)). '
              'All other optimizer fields including step/param_groups are exact.',
    prerequisite='Reuse exact post-load v2 check: its exact comparisons and actual loading function are unchanged',
    interpretation='v3 keeps rtol=1e-4/atol=1e-5 and measures the aggregate absolute budget per element; '
                   'v2 used an absolute budget for the entire tensor. This is a recorded policy revision.')
POLICY_V3_ID = identity(POLICY_V3)
KEYS = ('model', 'aux', 'optimizer', 'scheduler', 'rng', 'sampler', 'global_step', 'accumulation_position',
        'criterion', 'criterion_accumulators', 'config', 'input_identity', 'microbatch', 'epoch',
        'contract_identity', 'group', 'completed_training', 'amp_scaler')


def numpy_array(value):
    return value if isinstance(value, np.ndarray) else None


def compare_states(left, right, as_array=numpy_array, exact=False, policy=POLICY):
    if policy not in (POLICY, POLICY_V3):
        raise ValueError('Unknown frozen resume comparison policy')
    failures, numeric = [], []
    arrays_checked = 0
    def fail(path, reason):
        failures.append(dict(path=path, reason=reason))
    def walk(a, b, path):
        nonlocal arrays_checked
        x, y = as_array(a), as_array(b)
        if x is not None or y is not None:
            arrays_checked += 1
            if x is None or y is None or x.shape != y.shape or x.dtype != y.dtype:
                fail(path, 'array type/shape/dtype differs')
                return
            floating = np.issubdtype(x.dtype, np.floating)
            if floating and (not np.isfinite(x).all() or not np.isfinite(y).all()):
                fail(path, 'nonfinite array')
                return
            moments = path.startswith('optimizer/state/') and path.rsplit('/', 1)[-1] in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')
            toleranced = not exact and (path.startswith(('model/', 'aux/', 'criterion/', 'criterion_accumulators/')) or moments)
            if not floating or not toleranced:
                if not np.array_equal(x, y):
                    fail(path, 'exact array differs')
                return
            u, v = x.astype(np.float64), y.astype(np.float64)
            delta = u - v
            l2_delta = float(np.linalg.norm(delta.ravel()))
            l2_scale = max(float(np.linalg.norm(u.ravel())), float(np.linalg.norm(v.ravel())))
            elementwise = bool(np.allclose(x, y, rtol=policy['rtol'], atol=policy['atol'], equal_nan=False))
            root_count = math.sqrt(max(int(x.size), 1))
            absolute_budget = policy['atol'] * (root_count if policy['revision'] == 3 else 1.)
            norm_pass = moments and l2_delta <= absolute_budget + policy['rtol'] * l2_scale
            passed = elementwise or norm_pass
            numeric.append(dict(path=path, passed=bool(passed), original_elementwise_pass=elementwise,
                optimizer_norm_pass=bool(norm_pass), maximum_absolute_delta=float(np.abs(delta).max()) if x.size else 0.,
                l2_delta=l2_delta, l2_scale=l2_scale, relative_l2=l2_delta / max(l2_scale, policy['atol']),
                elements=int(x.size), rms_delta=l2_delta / root_count, rms_scale=l2_scale / root_count,
                aggregate_l2_limit=absolute_budget + policy['rtol'] * l2_scale))
            if not passed:
                fail(path, 'floating state exceeds registered v' + str(policy['revision']) + ' policy')
        elif isinstance(a, dict):
            if not isinstance(b, dict):
                fail(path, 'dictionary type differs')
                return
            if a.keys() != b.keys():
                fail(path, 'dictionary keys differ')
            for key in sorted(a.keys() & b.keys(), key=str):
                walk(a[key], b[key], path + '/' + str(key) if path else str(key))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                fail(path, 'sequence type/length differs')
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + '/' + str(index))
        elif isinstance(a, numbers.Real) and not isinstance(a, (bool, int, np.integer)):
            if not isinstance(b, numbers.Real) or not math.isfinite(float(a)) or not math.isfinite(float(b)):
                fail(path, 'nonfinite or incompatible scalar')
            elif not exact and path.startswith('criterion_accumulators/'):
                delta = abs(float(a) - float(b))
                passed = delta <= policy['atol'] + policy['rtol'] * abs(float(b))
                numeric.append(dict(path=path, passed=passed, left=float(a), right=float(b),
                                    maximum_absolute_delta=delta, scalar_log=True))
                if not passed:
                    fail(path, 'floating log exceeds original tolerance')
            elif a != b:
                fail(path, 'exact scalar differs')
        elif type(a) is not type(b) or a != b:
            fail(path, 'exact scalar/type differs')
    walk(left, right, '')
    return dict(status='complete' if not failures else 'failed', exact=exact,
                failures=failures, numeric=numeric, arrays_checked=arrays_checked, policy_identity=identity(policy))
