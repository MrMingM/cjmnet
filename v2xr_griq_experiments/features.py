"""Inference-only features in the ego frame, ordered explicitly for ablations."""
import numpy as np
from scipy.spatial import cKDTree

BASE = ('logit', 'score', 'distance', 'height', 'width', 'length', 'sin_yaw', 'cos_yaw',
        'lidar_count', 'lidar_cells', 'lidar_car_count', 'lidar_empty')
GSPR = ('gspr_points', 'r_mean', 'r_var', 'r_p10', 'r_p50', 'r_p90', 'u_mean',
        'effective_count', 'gspr_empty')
RADAR = ('radar_count', 'radar_density', 'background_density', 'log_density_ratio',
         'offset_mean', 'offset_p10', 'offset_p50', 'offset_p90',
         'spread_x', 'spread_y', 'spread_z', 'shared_cells', 'radar_only_cells',
         'radar_car_count', 'radar_empty')
FEATURES = BASE + GSPR + RADAR


def indices(fields):
    return [FEATURES.index(name) for name in fields]


class Cloud:
    def __init__(self, streams):
        self.points = np.concatenate(streams)[:, :3].astype(np.float32)
        self.cars = np.repeat(np.arange(len(streams)), [len(p) for p in streams])
        self.tree = cKDTree(self.points)

    def near(self, box, radius):
        ids = np.asarray(self.tree.query_ball_point(box[:3], radius), dtype=np.int64)
        return ids, self.points[ids]


def occupied(points):
    return set(map(tuple, np.floor(points[:, :2] / .4).astype(np.int32)))


def extract(boxes, prob, logits, context, inside):
    (ls, rs), xyz, reliability, uncertainty, agents = context
    clouds = Cloud(ls), Cloud(rs), Cloud([xyz])
    result = np.empty((len(boxes), len(FEATURES)), dtype=np.float32)
    for i, (box, p, z) in enumerate(zip(boxes, prob, logits)):
        h, w, length, yaw = box[3:]
        if not np.isfinite(box).all() or min(h, w, length) <= 0:
            raise ValueError('Invalid box reached feature extraction')
        halo = box.copy()
        halo[4:6] *= 1.5
        radius = np.linalg.norm(halo[3:6]) / 2 + 1e-4
        selected = []
        for cloud in clouds:
            ids, points = cloud.near(box, radius)
            keep = inside(points, box)
            selected.append((ids[keep], points[keep], points))
        (li, lidar, _), (ri, radar, local_radar), (gi, _, _) = selected
        lc, rc = occupied(lidar), occupied(radar)
        rel, unc = reliability[gi], uncertainty[gi]
        dist = np.linalg.norm(radar - box[:3], axis=1)
        density = len(ri) / (w * length)
        background = (inside(local_radar, halo).sum() - len(ri)) / (1.25 * w * length)
        r_stats = [rel.mean(), rel.var(), *np.quantile(rel, [.1, .5, .9]), unc.mean(),
                   np.sum(rel * (1 - unc))] if len(gi) else [np.nan]*7
        r_geometry = [dist.mean(), *np.quantile(dist, [.1, .5, .9]),
                      *radar.std(0)] if len(ri) else [np.nan]*7
        result[i] = [z, p, np.linalg.norm(box[:2]), h, w, length, np.sin(yaw), np.cos(yaw),
                     len(li), len(lc), len(np.unique(clouds[0].cars[li])), not len(li),
                     len(gi), *r_stats, not len(gi),
                     len(ri), density, background, np.log((density+.05)/(background+.05)),
                     *r_geometry, len(lc & rc), len(rc - lc),
                     len(np.unique(clouds[1].cars[ri])), not len(ri)]
    return result


def check_voxel_alignment(ego, context):
    # The official proj_first loader voxelizes the projected LiDAR array in-place.
    (lidar, radar), xyz, r, u, agents = context
    if len(lidar) != len(radar) or not (len(xyz) == len(r) == len(u) == len(agents)):
        raise ValueError('Point evidence/CAV alignment failed')
    if not np.isfinite(xyz).all() or not np.isfinite(r).all() or not np.isfinite(u).all():
        raise ValueError('Nonfinite GSPR point evidence')
    if len(agents) and (agents.min() < 0 or agents.max() >= len(lidar)):
        raise ValueError('GSPR points outside the projected CAV slots')
    for i, points in enumerate(lidar):
        g = xyz[agents == i]
        if len(g):
            if not len(points) or cKDTree(points[:, :3]).query(g, k=1)[0].max() > .001:
                raise ValueError(f'GSPR points are not in projected ego-frame LiDAR for CAV {i}')
