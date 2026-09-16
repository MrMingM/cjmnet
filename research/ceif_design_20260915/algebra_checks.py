"""CEIF design checks: interval algebra and local correction, not detector tests."""
import json
import math
from itertools import permutations


def combine(intervals):
    lower = max((a for a, _ in intervals), default=0.0)
    upper = min((b for _, b in intervals), default=1.0)
    return lower, upper, max(upper - lower, 0), max(lower - upper, 0)


def violation(p, lower, upper):
    positive = max(lower - p, 0.0)
    negative = max(p - upper, 0.0)
    return positive**2 + negative**2, -2 * positive + 2 * negative


def scalar_correction(initial, lower, upper, strength=10.0):
    """Exact scalar minimizer of .5*(p-initial)^2 + strength*violation."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        gradient = mid - initial + strength * violation(mid, lower, upper)[1]
        if gradient < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def group_prox(vector, threshold):
    norm = math.sqrt(sum(x*x for x in vector))
    factor = max(1 - threshold / norm, 0) if norm else 0
    return tuple(factor*x for x in vector)


def main():
    close = lambda a, b: math.isclose(a, b, abs_tol=1e-10)
    intervals = [(0.8, 1.0), (0.0, 0.1), (0.2, 0.9)]
    base = combine(intervals)
    assert combine(intervals + [(0, 1)]) == base  # vacuous identity
    assert combine(intervals + intervals) == base  # duplicate idempotence
    assert all(combine(order) == base for order in permutations(intervals))
    for index in range(len(intervals)):
        reduced = combine(intervals[:index] + intervals[index+1:])
        assert reduced[0] <= base[0] and reduced[1] >= base[1]
    assert combine([]) == (0, 1, 1, 0)
    for p in (0, 0.1, 0.5, 0.9, 1):
        assert violation(p, 0, 1) == (0, 0)
    solo = combine([(0.8, 1), (0, 1)])
    assert solo == (0.8, 1, 1-0.8, 0)
    unknown = scalar_correction(0.1, 0, 1)
    completion = scalar_correction(0.1, 0.8, 1)
    refutation = scalar_correction(0.9, 0, 0.1)
    assert close(unknown, 0.1) and completion > 0.7 and refutation < 0.2
    conflict = combine([(0.9, 1), (0, 0.1)])
    assert close(conflict[3], 0.8)
    loss, gradient = violation(0.5, conflict[0], conflict[1])
    assert close(gradient, 0) and loss > 0.3  # zero net update is NOT no conflict
    # Mean reliability alone does not determine committed reliable belief.
    assert close(0.6 + 0.2/2, 0.4 + 0.6/2)
    assert 0.6 > 0.4
    # Check vector group shrinkage against its first-order optimality condition.
    vec, threshold = (3.0, 4.0), 2.0
    prox = group_prox(vec, threshold)
    norm = math.sqrt(sum(x*x for x in prox))
    assert all(close(e-r+threshold*e/norm, 0) for e, r in zip(prox, vec))
    assert group_prox((0.3, 0.4), 1.0) == (0.0, 0.0)
    print(json.dumps({
        "status": "PASS",
        "scope": "interval algebra and scalar correction only; no detector AP",
        "unknown_preserves_initial": unknown,
        "hit_drives_completion": completion,
        "free_drives_refutation": refutation,
        "symmetric_conflict_gap": conflict[3],
        "symmetric_conflict_violation_at_zero_net_gradient": loss,
        "group_prox_example": prox,
    }, indent=2))


if __name__ == '__main__':
    main()
