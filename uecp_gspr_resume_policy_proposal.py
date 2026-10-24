"""Unapplied recovery-check proposal, pure NumPy; not imported by the pipeline.

Only optimizer moment tensors may use the extra per-tensor norm check. Model
tensors retain the original elementwise criterion. Exact loading remains zero
tolerance. Integrating this proposal requires an explicit protocol amendment.
"""
import numpy as np


ATOL, RTOL = 1e-5, 1e-4
MOMENTS = {'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}


def floating_array(left, right, path, exact=False, same_checkpoint_verified=False,
                   exact_load_verified=False):
    """Return a proposed comparison decision without passing any experiment gate."""
    base = dict(policy_applied=False, formal_recovery_gate_passed=False, path=path)
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
        return dict(base, accepted=False, reason='array type differs')
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(base, accepted=False, reason='shape/dtype differs')
    if not np.issubdtype(left.dtype, np.floating):
        return dict(base, accepted=bool(np.array_equal(left, right)), reason='exact nonfloating comparison')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return dict(base, accepted=False, reason='nonfinite state')
    identical = bool(np.array_equal(left, right))
    if exact:
        return dict(base, accepted=identical, reason='zero-tolerance post-load comparison')
    if not same_checkpoint_verified or not exact_load_verified:
        return dict(base, accepted=False, reason='common checkpoint and exact loading evidence required')
    parts = path.split('/')
    moment = (len(parts) == 4 and parts[:2] == ['optimizer', 'state'] and parts[3] in MOMENTS)
    model = path.startswith('model/')
    if not moment and not model:
        return dict(base, accepted=identical, reason='exact optimizer step/other state comparison')
    x, y = left.astype(np.float64), right.astype(np.float64)
    difference = x-y
    pointwise = bool((np.abs(difference) <= ATOL+RTOL*np.abs(y)).all())
    delta = float(np.linalg.norm(difference.ravel()))
    scale = max(float(np.linalg.norm(x.ravel())), float(np.linalg.norm(y.ravel())))
    norm_pass = moment and delta <= ATOL+RTOL*scale
    return dict(base, accepted=bool(pointwise or norm_pass), original_pointwise_pass=pointwise,
        optimizer_moment_norm_pass=bool(norm_pass), l2_delta=delta, l2_scale=scale,
        atol=ATOL, rtol=RTOL, reason='proposed moment comparison' if moment else 'original model comparison')
