"""GT-free geometry and action generation. No ground truth enters this module."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Settings:
    reliable_belief: float = .6
    support_belief: float = .15
    ray_radius: float = .20
    depth_margin: float = 1.0
    min_support: int = 3
    min_conflicts: int = 2
    conflict_fraction: float = .3
    max_queries: int = 32
    boxes_per_source: int = 12
    max_actions: int = 8

    def validate(self):
        if not 0 < self.support_belief < self.reliable_belief < 1:
            raise ValueError('Invalid belief thresholds')
        if self.ray_radius <= 0 or self.depth_margin <= 0 or not 0 < self.conflict_fraction <= 1:
            raise ValueError('Invalid geometry thresholds')
        if min(self.min_support, self.min_conflicts, self.max_queries,
               self.boxes_per_source, self.max_actions) < 1:
            raise ValueError('Counts must be positive')
        return asdict(self)


def point_blocks(xyz, grid, extent):
    h, w = grid
    x = np.floor((xyz[:, 0]-extent[0])/(extent[3]-extent[0])*w).astype(int)
    y = np.floor((xyz[:, 1]-extent[1])/(extent[4]-extent[1])*h).astype(int)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return y.clip(0, h-1)*w+x.clip(0, w-1), valid


def inside_box(points, corners):
    if not len(points):
        return np.zeros(0, dtype=bool)
    polygon = corners[:4, :2]
    edges = np.roll(polygon, -1, axis=0)-polygon
    delta = points[:, None, :2]-polygon[None]
    cross = edges[None, :, 0]*delta[:, :, 1]-edges[None, :, 1]*delta[:, :, 0]
    planar = (cross >= -1e-5).all(1) | (cross <= 1e-5).all(1)
    return planar & (points[:, 2] >= corners[:, 2].min()) & (points[:, 2] <= corners[:, 2].max())


def box_patch(corners, grid, extent):
    # AABB of the predicted box; never a GT box. At least its center block.
    h, w = grid
    low, high = corners[:, :2].min(0), corners[:, :2].max(0)
    if np.any(high < np.asarray(extent[:2])) or np.any(low >= np.asarray(extent[3:5])):
        return np.zeros(grid, bool)
    scale = np.array([w/(extent[3]-extent[0]), h/(extent[4]-extent[1])])
    a = np.floor((low-np.asarray(extent[:2]))*scale).astype(int)
    b = np.floor((high-np.asarray(extent[:2]))*scale).astype(int)
    a = np.maximum(a, 0); b = np.minimum(b, [w-1, h-1])
    patch = np.zeros(grid, bool)
    patch[a[1]:b[1]+1, a[0]:b[0]+1] = True
    return patch


def angular_ids(local):
    # Only conservative first-return rejection, not a free-space certificate.
    az = np.arctan2(local[:, 1], local[:, 0])
    el = np.arctan2(local[:, 2], np.linalg.norm(local[:, :2], axis=1))
    a = np.floor((az+np.pi)/(2*np.pi)*720).astype(int).clip(0, 719)
    e = np.floor((el+np.pi/2)/np.pi*180).astype(int).clip(0, 179)
    return e*720+a


class Source:
    def __init__(self, xyz, opinions, raw_cloud, pose, received, grid, extent, settings):
        self.received = np.asarray(received, bool).reshape(grid)
        self.grid, self.extent, self.settings = grid, extent, settings
        self.origin = np.asarray(pose[:3, 3], float)
        ids, valid = point_blocks(xyz, grid, extent)
        valid &= self.received.ravel()[ids]
        xyz, opinions = np.asarray(xyz[valid], float), np.asarray(opinions[valid], float)
        if len(xyz):
            _, unique = np.unique(xyz, axis=0, return_index=True)
            xyz, opinions = xyz[unique], opinions[unique]
        self.xyz, self.opinions = xyz, opinions
        self.belief = opinions[:, 0]
        raw = np.asarray(raw_cloud[:, :3], float)
        raw = raw[np.isfinite(raw).all(1)]
        ray_ok = np.zeros(len(xyz), bool)
        if len(raw) and len(xyz):
            # Exact retained-to-raw association; never borrow another return's reliability.
            distance, _ = cKDTree(raw).query(xyz, k=1)
            raw_local = (raw-self.origin) @ pose[:3, :3]
            local = (xyz-self.origin) @ pose[:3, :3]
            depth = np.full(720*180, np.inf)
            np.minimum.at(depth, angular_ids(raw_local), np.linalg.norm(raw_local, axis=1))
            radius = np.linalg.norm(local, axis=1)
            ray_ok = ((distance <= 1e-3) & (radius <= depth[angular_ids(local)]+.05)
                      & (radius > 1e-3) & (self.belief >= settings.reliable_belief))
        endpoints = xyz[ray_ok]
        vector = endpoints-self.origin
        self.ranges = np.linalg.norm(vector, axis=1)
        self.directions = vector/np.maximum(self.ranges[:, None], 1e-12)
        self.ray_beliefs = self.belief[ray_ok]
        self.tree = cKDTree(self.directions) if len(endpoints) else None
        # xyz+b_R+b_N+u float32 plus one sender-computed valid-ray flag per point.
        self.certificate_bytes = int(len(xyz)*25 + (64 if len(xyz) else 0))

    def free_at(self, queries):
        out = np.zeros(len(queries))
        if self.tree is None or not len(queries):
            return out
        ids, available = point_blocks(queries, self.grid, self.extent)
        available &= self.received.ravel()[ids]
        vector = queries-self.origin
        radius = np.linalg.norm(vector, axis=1)
        unit = vector/np.maximum(radius[:, None], 1e-12)
        # Query all rays inside the angular radius: no nearest-ray cutoff can hide
        # a valid farther return behind a shorter candidate in another direction.
        chord = np.minimum(2., 2*np.sin(np.arcsin(np.minimum(1., self.settings.ray_radius/
                                    np.maximum(radius, 1e-12)))/2))
        neighbors = self.tree.query_ball_point(unit, chord+1e-10)
        for q, candidates in enumerate(neighbors):
            if not available[q] or radius[q] <= 1e-3 or not candidates:
                continue
            candidates = np.asarray(candidates)
            along = self.directions[candidates] @ vector[q]
            perpendicular = np.sqrt(np.maximum(radius[q]**2-along**2, 0))
            valid = ((along > 0) & (perpendicular <= self.settings.ray_radius)
                     & (self.ranges[candidates] > along+self.settings.depth_margin))
            if valid.any():
                out[q] = self.ray_beliefs[candidates[valid]].max()
        return out


def box_evidence(box, sources, settings):
    points, owners, beliefs = [], [], []
    trusted_counts = []
    for peer, source in enumerate(sources):
        inside = inside_box(source.xyz, box)
        trusted_counts.append(int((inside & (source.belief >= settings.reliable_belief)).sum()))
        indices = np.flatnonzero(inside & (source.belief >= settings.support_belief))
        if len(indices) > settings.max_queries:
            indices = indices[np.linspace(0, len(indices)-1, settings.max_queries).astype(int)]
        points.extend(source.xyz[indices]); beliefs.extend(source.belief[indices])
        owners.extend([peer]*len(indices))
    points = np.asarray(points).reshape(-1, 3)
    owners, beliefs = np.asarray(owners), np.asarray(beliefs)
    counts, strengths = [], []
    for peer, source in enumerate(sources):
        free = source.free_at(points)
        # Cross-source, directed conflict: a clearly stronger free witness.
        conflict = ((owners != peer) & (free >= settings.reliable_belief)
                    & (free-beliefs >= .2))
        denominator = int((owners != peer).sum())
        count = int(conflict.sum())
        counts.append(count)
        strengths.append(count/max(denominator, 1))
    return dict(trusted_counts=trusted_counts, conflict_counts=counts,
                conflict_fractions=strengths, sampled_endpoints=len(points))


def make_actions(base_boxes, base_scores, donors, sources, masks, extent, settings):
    """All proposals, regions, priorities and eligibility are GT-independent."""
    grid = tuple(masks.shape[-2:])
    actions, evidence_rows = [], []
    if len(sources) <= 1:
        return actions, evidence_rows
    for kind, owner, boxes, scores in [('correct', -1, base_boxes, base_scores)]+[
            ('complete', peer, b, s) for peer, b, s in donors]:
        for index in np.argsort(-scores)[:settings.boxes_per_source]:
            box = boxes[index]
            evidence = box_evidence(box, sources, settings)
            if kind == 'correct':
                eligible_peers = [p for p in range(len(sources))
                    if evidence['conflict_counts'][p] >= settings.min_conflicts
                    and evidence['conflict_fractions'][p] >= settings.conflict_fraction]
                peers = eligible_peers or list(range(len(sources)))
            else:
                peers = [owner]
            row = dict(kind=kind, owner=owner, prediction_index=int(index), score=float(scores[index]), **evidence)
            evidence_rows.append(row)
            for peer in peers:
                # Existing received payload only; don't replace missing blocks by hidden features.
                region = box_patch(box, grid, extent) & masks[peer].reshape(grid).astype(bool)
                if not region.any():
                    continue
                if kind == 'correct':
                    eligible = peer in eligible_peers
                    priority = evidence['conflict_fractions'][peer] + min(evidence['conflict_counts'][peer], 10)/10
                else:
                    eligible = (evidence['trusted_counts'][peer] >= settings.min_support
                                and evidence['trusted_counts'][0] == 0
                                and max(evidence['conflict_counts'], default=0) < settings.min_conflicts)
                    priority = min(evidence['trusted_counts'][peer], 10)/10 + float(scores[index])
                actions.append(dict(peer=peer, region=region, eligible=bool(eligible), kind=kind,
                                    priority=float(priority), origin=row))
    # Round-robin the two intervention types, then fill the remaining pool.
    actions.sort(key=lambda a: (a['eligible'], a['priority']), reverse=True)
    queues = [[a for a in actions if a['kind'] == kind] for kind in ('correct', 'complete')]
    ordered, seen = [], set()
    while any(queues) and len(ordered) < settings.max_actions:
        for queue in queues:
            while queue:
                action = queue.pop(0)
                key = (action['peer'], action['region'].tobytes())
                if key not in seen:
                    seen.add(key); ordered.append(action)
                    break
            if len(ordered) >= settings.max_actions:
                break
    return ordered, evidence_rows
