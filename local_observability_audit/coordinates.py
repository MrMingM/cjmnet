"""Pure NumPy geometry; canonical box is [x,y,z,l,w,h,yaw] in ego frame."""
import numpy as np

def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])

def local_xy(points, theta):
    return (np.asarray(points, float)[..., :2] - np.asarray(theta)[:2]) @ rotation(theta[4])

def world_xy(local, theta):
    return np.asarray(local) @ rotation(theta[4]).T + np.asarray(theta)[:2]

def canonical(boxes, order):
    b = np.asarray(boxes, float).reshape(-1, 7).copy()
    if order == 'hwl':
        b = b[:, [0, 1, 2, 5, 4, 3, 6]]
    elif order != 'lwh':
        raise ValueError('Unknown box size order')
    if not np.isfinite(b).all() or np.any(b[:, 3:6] <= 0):
        raise ValueError('Invalid GT geometry')
    return b

def bev_state(box):
    return np.asarray(box)[[0, 1, 3, 4, 6]]

def crop(points, box, expansion):
    p = np.asarray(points)
    q = local_xy(p, bev_state(box))
    return ((np.abs(q) <= np.asarray(box)[3:5] * expansion / 2 + 1e-9).all(-1)
            & (np.abs(p[:, 2] - box[2]) <= box[5] * expansion / 2 + 1e-9))

def corners_to_box(corners):
    c = np.asarray(corners, float)
    center = c.mean(axis=0)
    long = c[0, :2] - c[3, :2]
    width = c[1, :2] - c[0, :2]
    return np.r_[center, np.linalg.norm(long), np.linalg.norm(width),
                 np.linalg.norm(c[4] - c[0]), np.arctan2(long[1], long[0])]

def box_corners(box):
    xy = world_xy(np.array([[1,-1],[1,1],[-1,1],[-1,-1]]) * np.asarray(box)[3:5] / 2, bev_state(box))
    return np.concatenate([np.c_[xy, np.full(4, box[2] - box[5]/2)],
                           np.c_[xy, np.full(4, box[2] + box[5]/2)]])

def angle_error(a, b, period=np.pi):
    return abs((a-b + period/2) % period - period/2)

def retained_points(processed, reliability, source_count, voxel_size, limits):
    """Exact slot provenance; coordinates already projected into ego by dataset."""
    f, c, n = (np.asarray(processed[k]) for k in ('voxel_features', 'voxel_coords', 'voxel_num_points'))
    if f.ndim != 3 or c.shape != (len(f), 4) or n.shape != (len(f),):
        raise ValueError('Invalid voxel layout')
    if not np.array_equal(c, c.astype(int)) or np.any(c < 0) or np.any(c[:, 0] >= source_count):
        raise ValueError('Invalid source/grid indices')
    if np.any(n < 1) or np.any(n > f.shape[1]) or not np.array_equal(n, n.astype(int)):
        raise ValueError('Invalid voxel point counts')
    mask = np.arange(f.shape[1])[None] < n[:, None]
    if not np.array_equal(mask, reliability['point_valid_mask']):
        raise ValueError('GSPR valid mask does not match original slots')
    for key in ('point_reliability', 'point_uncertainty'):
        if np.asarray(reliability[key]).shape != mask.shape:
            raise ValueError('GSPR point/slot mismatch')
        if not np.isfinite(reliability[key][mask]).all():
            raise ValueError('Invalid GSPR output')
    low = np.asarray(limits[:3]) + c[:, [3, 2, 1]] * np.asarray(voxel_size)
    inside = ((f[..., :3] >= low[:, None] - 2e-5) &
              (f[..., :3] <= low[:, None] + np.asarray(voxel_size) + 2e-5)).all(-1)
    if not np.isfinite(f[mask]).all() or not inside[mask].all():
        raise ValueError('Ego-frame XYZ / voxel-cell mismatch')
    output = []
    for i in range(source_count):
        select = mask & (c[:, 0, None] == i)
        output.append((f[select], reliability['point_reliability'][select], reliability['point_uncertainty'][select]))
    return output
