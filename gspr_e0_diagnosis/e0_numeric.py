"""Pure NumPy intervention and bounded-memory descriptive statistics."""
import hashlib
import json
import numpy as np


def array_hash(value):
    x = np.ascontiguousarray(value)
    h = hashlib.sha256(str(x.dtype).encode() + repr(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def replace_weights(r, valid, agents, arm, key, seed=20261021, agent_ids=None):
    r, valid, agents = np.asarray(r), np.asarray(valid), np.asarray(agents)
    if (r.ndim != 2 or valid.shape != r.shape or valid.dtype != np.bool_
            or agents.shape != (len(r),) or agents.dtype.kind not in 'iu'
            or r.dtype.kind != 'f' or not np.isfinite(r).all()
            or (r < 0).any() or (r > 1).any() or (r[~valid] != 0).any()):
        raise ValueError('Invalid point reliability/mask/CAV alignment')
    if arm not in ('R0', 'R1', 'R2', 'R3'):
        raise ValueError('Unknown reliability intervention')
    out = r.copy()
    audit = []
    for agent in np.unique(agents):
        mask = valid & (agents[:, None] == agent)
        old = r[mask]
        cav_id = int(agent) if agent_ids is None else str(agent_ids[int(agent)])
        token = json.dumps([seed, *key, cav_id], separators=(',', ':'), ensure_ascii=False)
        digest = hashlib.sha256(token.encode()).hexdigest()
        permutation_hash = None
        if len(old):
            if arm == 'R1':
                out[mask] = 1
            elif arm == 'R2':
                out[mask] = old.mean(dtype=np.float64)
            elif arm == 'R3':
                order = np.random.Generator(np.random.PCG64(int(digest[:16], 16))).permutation(len(old))
                out[mask] = old[order]
                permutation_hash = array_hash(order.astype('<i8'))
                if not np.array_equal(np.sort(old), np.sort(out[mask])):
                    raise ValueError('Permutation changed reliability multiset')
        audit.append(dict(agent_slot=int(agent), seed_sha256=digest, points=len(old),
            original_mean=float(old.mean(dtype=np.float64)) if len(old) else None,
            applied_mean=float(out[mask].mean(dtype=np.float64)) if len(old) else None,
            original_sha256=array_hash(old), applied_sha256=array_hash(out[mask]),
            permutation_sha256=permutation_hash))
    out[~valid] = 0
    if out.dtype != r.dtype or not np.isfinite(out).all():
        raise ValueError('Intervention changed dtype or produced nonfinite values')
    return out, audit


def range_group(distance):
    x = np.asarray(distance)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Invalid distance')
    return np.searchsorted([20., 40., 70.], x, side='right')


def distribution(x, upper=1., bins=1000):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite statistical observation')
    hist = np.histogram(x, bins=bins, range=(0, upper))[0]
    return dict(n=len(x), sum=float(x.sum()), sumsq=float(x @ x),
        min=float(x.min()) if len(x) else None, max=float(x.max()) if len(x) else None,
        histogram=hist.tolist(), upper=upper, outside_histogram=int(((x < 0) | (x > upper)).sum()))


def merge_distribution(items):
    items = list(items)
    if not items:
        return dict(n=0, mean=None, std=None, quantiles=None)
    n = sum(v['n'] for v in items)
    hist = np.sum([v['histogram'] for v in items], axis=0, dtype=np.int64)
    total = sum(v['sum'] for v in items)
    sq = sum(v['sumsq'] for v in items)
    upper = items[0]['upper']
    if any(v['upper'] != upper or len(v['histogram']) != len(hist) for v in items):
        raise ValueError('Incompatible histogram definitions')
    q = None
    outside = sum(v['outside_histogram'] for v in items)
    if n and not outside:
        q = {str(p): float((min(np.searchsorted(np.cumsum(hist), max(1, p*n), side='left'), len(hist)-1)+.5)*upper/len(hist))
             for p in (.01, .05, .25, .5, .75, .95, .99)}
    return dict(n=n, sum=total, sumsq=sq, mean=total/n if n else None,
        std=float(np.sqrt(max(0., sq/n-(total/n)**2))) if n else None,
        min=min((v['min'] for v in items if v['n']), default=None),
        max=max((v['max'] for v in items if v['n']), default=None),
        quantiles=q, quantile_bin_width=upper/len(hist), histogram=hist.tolist(),
        upper=upper, outside_histogram=outside)


def occupancy(xyz, cell=.4):
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] < 2 or not np.isfinite(xyz).all():
        raise ValueError('Invalid occupancy points')
    return {tuple(row) for row in np.floor(xyz[:, :2]/cell).astype(np.int64)}


def coverage(clean, weather, cell=.4):
    a, b = occupancy(clean, cell), occupancy(weather, cell)
    return dict(clean_occupied=len(a), weather_occupied=len(b), intersection=len(a & b),
        missing_clean_cells=len(a-b), additional_weather_cells=len(b-a),
        clean_coverage_fraction=len(a & b)/len(a) if a else None,
        cell_origin_xy=[0, 0], cell_m=cell)


def nearest_proxy(clean, weather, tolerances=(.05, .1, .2)):
    from scipy.spatial import cKDTree
    a, b = np.asarray(clean), np.asarray(weather)
    if a.shape != (len(a), 3) or b.shape != (len(b), 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid paired coordinates')
    to_a = cKDTree(a).query(b, workers=1)[0] if len(a) else np.full(len(b), np.inf)
    to_b = cKDTree(b).query(a, workers=1)[0] if len(b) else np.full(len(a), np.inf)
    result = {str(t): dict(clean_without_near_return=int((to_b > t).sum()),
        weather_without_near_clean=int((to_a > t).sum()), clean_points=len(a), weather_points=len(b)) for t in tolerances}
    return result, to_a


def author_mask_proxy(points, agents, original_xyz, original_agents, nonzero, tolerance=1e-4):
    """Map the author's green==0 mask without forcing colliding/unknown points."""
    from scipy.spatial import cKDTree
    x, a, raw, source, label = map(np.asarray, (points, agents, original_xyz, original_agents, nonzero))
    if (x.shape != (len(x), 3) or raw.shape != (len(raw), 3) or a.shape != (len(x),)
            or source.shape != (len(raw),) or label.shape != (len(raw),) or label.dtype != np.bool_
            or not np.isfinite(x).all() or not np.isfinite(raw).all()):
        raise ValueError('Invalid author color-mask correspondence inputs')
    result = np.full(len(x), -1, dtype=np.int8)
    for agent in np.unique(a):
        mask = a == agent
        distances = []
        for value in (False, True):
            selected = raw[(source == agent) & (label == value)]
            distances.append(cKDTree(selected).query(x[mask], workers=1)[0] if len(selected) else np.full(mask.sum(), np.inf))
        zero, other = distances[0] <= tolerance, distances[1] <= tolerance
        status = np.full(mask.sum(), -1, dtype=np.int8)
        status[zero & ~other] = 0
        status[other & ~zero] = 1
        status[zero & other] = 2
        result[mask] = status
    return result


def covariance_summary(names, n, sums, gram):
    if n < 2:
        return dict(n=n, names=names, correlation=None, partial_correlation=None)
    s = np.asarray(sums, dtype=np.float64)
    cov = np.asarray(gram, dtype=np.float64)/n-np.outer(s/n, s/n)
    sd = np.sqrt(np.maximum(np.diag(cov), 0))
    denom = np.outer(sd, sd)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-20)
    partial = {}
    controls = [names.index(k) for k in ('density', 'sensor_range', 'intensity', 'local_pillar_density') if k in names]
    # Linear residual association conditional on measured density/range/intensity.
    for i, name in enumerate(names):
        if i == 0 or i in controls:
            continue
        target = [0, i]
        c = cov[np.ix_(target, target)] - cov[np.ix_(target, controls)] @ np.linalg.pinv(cov[np.ix_(controls, controls)]) @ cov[np.ix_(controls, target)]
        d = np.sqrt(max(c[0, 0], 0)*max(c[1, 1], 0))
        partial[name] = float(np.clip(c[0, 1]/d, -1, 1)) if d > 1e-15 else None
    return dict(n=n, names=names, correlation=corr.tolist(), partial_correlation=partial,
        constant_variables=[names[i] for i, value in enumerate(sd) if value < 1e-10],
        interpretation='linear descriptive association; no calibration or causal interpretation')
