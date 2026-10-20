"""NumPy finite-ray labels. No point-row correspondence, inferred firing beams or GT solids."""
import numpy as np


def transform(points, matrix):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return p @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def nearest_returns(points, origin, angular_bin):
    d = np.asarray(points) - origin
    r = np.linalg.norm(d, axis=1)
    az = np.mod(np.arctan2(d[:, 1], d[:, 0]) + angular_bin / 2, 2 * np.pi)
    el = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
    keys = np.stack((np.floor(az / angular_bin), np.rint(el / angular_bin)), axis=1).astype(np.int64)
    order = np.argsort(r, kind='stable')
    _, first = np.unique(keys[order], axis=0, return_index=True)
    result = np.zeros(len(points), dtype=bool)
    result[order[first]] = True
    return result & (r > 1e-6)


def candidate_indices(points, origins, center, size, jitter):
    """Conservative segment AABB query includes all jittered windows, before clipping rays."""
    lo = np.asarray(center) - np.asarray(size) / 2 - [jitter, jitter, 0]
    hi = np.asarray(center) + np.asarray(size) / 2 + [jitter, jitter, 0]
    return np.flatnonzero(np.all(np.maximum(points, origins) >= lo, axis=1)
                         & np.all(np.minimum(points, origins) <= hi, axis=1))


def segment_cells(origin, endpoint, lo, size, voxel, margin, minimum):
    """Exact segment/grid intersections; only positive-length interior crossings are free."""
    delta = endpoint - origin
    distance = np.linalg.norm(delta)
    if distance <= margin:
        return
    end_t = 1 - margin / distance
    enter, leave = 0., end_t
    for k in range(3):
        if abs(delta[k]) < 1e-12:
            if not lo[k] <= origin[k] < lo[k] + size[k]:
                return
        else:
            a, b = (lo[k] - origin[k]) / delta[k], (lo[k] + size[k] - origin[k]) / delta[k]
            enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
    if leave <= enter:
        return
    cuts = [enter, leave]
    for k in range(3):
        if abs(delta[k]) > 1e-12:
            planes = lo[k] + np.arange(int(round(size[k] / voxel[k])) + 1) * voxel[k]
            t = (planes - origin[k]) / delta[k]
            cuts.extend(t[(t > enter) & (t < leave)])
    cuts = np.unique(cuts)
    shape = np.rint(size / voxel).astype(int)
    for a, b in zip(cuts[:-1], cuts[1:]):
        if (b - a) * distance < minimum:
            continue
        cell = np.floor((origin + ((a + b) / 2) * delta - lo) / voxel).astype(int)
        if np.all(cell >= 0) and np.all(cell < shape):
            yield tuple(cell[::-1])


def labels(points, origins, source, nearest, center, config, fov=None):
    w, rules = config['window'], config['rays']
    size, voxel = np.asarray(w['size']), np.asarray(w['voxel'])
    lo = np.asarray(center) - size / 2
    shape = tuple(w['shape_xyz'][::-1])
    o_raw, e_raw, ego, peer = [np.zeros(shape, dtype=bool) for _ in range(4)]
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    q = (p - lo) / voxel
    # floor owns a boundary on its positive side; no tolerance-based dilation into neighbouring cells.
    cells = np.floor(q).astype(np.int64)
    valid = np.all((p >= lo) & (p < lo + size), axis=1)
    for i in np.flatnonzero(valid):
        idx = tuple(cells[i, ::-1])
        o_raw[idx] = True
        (ego if source[i] == 0 else peer)[idx] = True
    for i in np.flatnonzero(nearest):
        for idx in segment_cells(origins[i], p[i], lo, size, voxel,
                                 rules['free_margin_m'], rules['minimum_crossing_m']):
            e_raw[idx] = True
    # Entire free cells must remain inside native ego feature range and, when supplied,
    # the ego camera frustum. Endpoint occupancy uses the actual native-filtered point.
    for idx in np.argwhere(e_raw):
        xyz = idx[::-1]
        lower = lo + xyz * voxel
        upper = lower + voxel
        if np.any(lower < [-140.8, -40, -3]) or np.any(upper > [140.8, 40, 1]):
            e_raw[tuple(idx)] = False
            continue
        if fov is not None:
            corners = np.array([[x, y, z, 1.] for x in (lower[0], upper[0])
                                for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
            uvw = corners @ fov['projection'].T
            if np.any(uvw[:, 2] <= 1e-6):
                e_raw[tuple(idx)] = False
                continue
            uv = uvw[:, :2] / uvw[:, 2:3]
            if np.any(uv < 0) or np.any(uv >= fov['image_wh']):
                e_raw[tuple(idx)] = False
    o, e = o_raw & ~e_raw, e_raw & ~o_raw
    return dict(x0=np.where(o_raw, 1., -1.).astype(np.float32), O_raw=o_raw,
                E_raw=e_raw, O=o, E=e, U=~(o | e), ego=o & ego, peer=o & peer & ~ego)


def windows(boxes, seed, epoch, uid, config):
    from .common import stream_seed
    rng = np.random.RandomState(stream_seed(seed, epoch, uid, 'windows'))
    boxes = np.asarray(boxes).reshape(-1, 7)
    ids = rng.permutation(len(boxes))[:config['window']['maximum']]
    centers = boxes[ids, :3].copy()
    centers[:, 2] = -1.0  # fixed native vertical range [-3,1], never GT-height scaled
    centers[:, :2] += rng.uniform(-config['window']['jitter'], config['window']['jitter'], (len(ids), 2))
    return ids, centers


def loss_weights(group, lab):
    """Weights sum to one per nonempty window; E3 positive groups share the positive mass."""
    if group == 'E1':
        positives, negative = [lab['O_raw']], ~lab['O_raw']
    else:
        positives = [lab['O']] if group == 'E2' else [lab['ego'], lab['peer']]
        negative = lab['E']
    positives = [m for m in positives if m.any()]
    classes = int(bool(positives)) + int(negative.any())
    weights = np.zeros_like(lab['x0'], dtype=np.float32)
    if classes:
        for m in positives:
            weights[m] = 1. / (classes * len(positives) * m.sum())
        if negative.any():
            weights[negative] = 1. / (classes * negative.sum())
    return weights
