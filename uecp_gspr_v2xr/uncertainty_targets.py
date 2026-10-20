"""NumPy density maps share native x=W/y=H, half-open spatial bounds, current-weather points."""
import numpy as np


def density(points, bounds=(-140.8, -40, -3, 140.8, 40, 1), resolution=.4, cap=10):
    p = np.asarray(points).reshape(-1, 4)
    shape = (int(round((bounds[4] - bounds[1]) / resolution)), int(round((bounds[3] - bounds[0]) / resolution)))
    counts = np.zeros(shape, np.float32)
    keep = np.isfinite(p).all(1) & np.all((p[:, :3] >= bounds[:3]) & (p[:, :3] < bounds[3:]), axis=1)
    xy = np.floor((p[keep, :2] - bounds[:2]) / resolution).astype(int)
    np.add.at(counts, (xy[:, 1], xy[:, 0]), 1)
    return counts, 1 - np.minimum(counts, cap) / cap


def pairwise(transforms, maximum=7):
    out = np.tile(np.eye(4), (maximum, maximum, 1, 1))
    for i, a in enumerate(transforms):
        for j, b in enumerate(transforms):
            out[i, j] = np.linalg.solve(b, a)  # source i -> target j; grid sampling uses [ego, source].
    return out


def split_scenes(manifest, seed, number):
    from .common import stream_seed
    names = sorted(s['scene'] for s in manifest['scenes'])
    if not 1 <= number < len(names):
        raise ValueError('Invalid scene split')
    ordered = sorted(names, key=lambda s: stream_seed(seed, 0, s, 'scene_split'))
    dev = set(ordered[:number])
    uids = [[s['scene'], t, s['ego']] for s in manifest['scenes'] for t in s['timestamps']]
    return dict(train_dev_scenes=sorted(dev), train_core_scenes=sorted(set(names) - dev),
                train_dev=[i for i, u in enumerate(uids) if u[0] in dev],
                train_core=[i for i, u in enumerate(uids) if u[0] not in dev], uids=uids)


def pool_numpy(x, scale, reduction='mean'):
    h, w = x.shape[-2:]
    h, w = h // scale * scale, w // scale * scale
    shaped = x[..., :h, :w].reshape(*x.shape[:-2], h // scale, scale, w // scale, scale)
    return shaped.sum(axis=(-1, -3)) if reduction == 'sum' else shaped.mean(axis=(-1, -3))
