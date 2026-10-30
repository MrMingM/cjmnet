"""Independent ID-carrying CPU voxelizer, checked against actual four-feature output."""
import numpy as np

REASONS = ('no_return', 'inactive_CAV', 'ego_filter', 'range_filter',
           'voxel_boundary', 'voxel_total_limit', 'voxel_point_limit', 'PillarVFE_slot')


class Tracer:
    def __init__(self, pre):
        self.pre = pre
        args = dict(voxel_size=pre.voxel_size, point_cloud_range=pre.lidar_range,
                    max_num_points=pre.max_points_per_voxel, max_voxels=pre.max_voxels)
        if pre.spconv == 1:
            from spconv.utils import VoxelGeneratorV2
            self.generator = VoxelGeneratorV2(**args)
        else:
            from spconv.utils import Point2VoxelCPU3d
            self.generator = Point2VoxelCPU3d(vsize_xyz=pre.voxel_size, coors_range_xyz=pre.lidar_range,
                max_num_points_per_voxel=pre.max_points_per_voxel, num_point_features=5, max_num_voxels=pre.max_voxels)

    def tagged(self, pc):
        ids = np.arange(len(pc), dtype=np.float32)
        if len(pc) >= 2**24:
            raise ValueError('ID not exactly representable in float32')
        five = np.ascontiguousarray(np.column_stack((pc, ids)), dtype=np.float32)
        if self.pre.spconv == 1:
            v = self.generator.generate(five)
        else:
            from cumm import tensorview as tv
            v = self.generator.point_to_voxel(tv.from_numpy(five))
        if isinstance(v, dict):
            v = (v['voxels'], v['coordinates'], v['num_points_per_voxel'])
        return tuple((x.numpy() if self.pre.spconv == 2 else x).copy() for x in v)

    def slots(self, pc):
        official = {k: v.copy() for k, v in self.pre.preprocess(pc).items()}
        v, coords, counts = self.tagged(pc)
        if not np.array_equal(coords, official['voxel_coords']) or not np.array_equal(counts, official['voxel_num_points']):
            raise ValueError('BLOCKED_ID_VOXEL_COORDINATE_OR_COUNT_MISMATCH')
        mask = np.arange(v.shape[1])[None, :] < counts[:, None]
        if not np.array_equal(v[:, :, :4], official['voxel_features']):
            raise ValueError('BLOCKED_ID_VOXEL_FEATURE_MISMATCH')
        ids = v[:, :, 4][mask].astype(np.int64)
        if len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= len(pc)).any():
            raise ValueError('BLOCKED_AMBIGUOUS_ID_SLOTS')
        if not np.array_equal(pc[ids], official['voxel_features'][mask]):
            raise ValueError('BLOCKED_ID_SLOT_FEATURE_MISMATCH')
        # Native collation only pads batch index and concatenates; verify actual tensors.
        collated = self.pre.collate_batch([official])
        if not np.array_equal(collated['voxel_features'].numpy(), official['voxel_features']) or not np.array_equal(
                collated['voxel_num_points'].numpy(), counts) or not np.array_equal(
                collated['voxel_coords'].numpy()[:, 1:], coords):
            raise ValueError('BLOCKED_COLLATION_CHANGED_SLOTS')
        reason = np.full(len(pc), REASONS.index('voxel_boundary'), dtype=np.uint8)
        lo = np.asarray(self.pre.lidar_range[:3], dtype=np.float32)
        size = np.asarray(self.pre.voxel_size, dtype=np.float32)
        xyz = np.floor((pc[:, :3] - lo) / size).astype(np.int64)
        valid = ((xyz >= 0) & (xyz < self.pre.grid_size)).all(axis=1)
        # Compare voxel addresses, never approximate XYZ/feature nearest neighbors.
        coord_map = {tuple(x): i for i, x in enumerate(coords.tolist())}
        in_voxel = np.zeros(len(pc), dtype=bool)
        for i in np.flatnonzero(valid):
            k = coord_map.get(tuple(xyz[i, ::-1]))
            if k is None:
                if len(coords) != self.pre.max_voxels:
                    raise ValueError('Unexplained voxel discard below total capacity')
                reason[i] = REASONS.index('voxel_total_limit')
            else:
                in_voxel[i] = True
                reason[i] = REASONS.index('voxel_point_limit')
        if len(ids) and not in_voxel[ids].all():
            raise ValueError('Voxel coordinate arithmetic differs from native spconv')
        # Independently verify source-order selection, including bitwise-identical points.
        expected = [[] for _ in coords]
        for i in np.flatnonzero(in_voxel):
            k = coord_map[tuple(xyz[i, ::-1])]
            if len(expected[k]) < counts[k]:
                expected[k].append(int(i))
        if [i for group in expected for i in group] != ids.tolist():
            raise ValueError('BLOCKED_DUPLICATE_POINT_ORDER_AMBIGUITY')
        reason[ids] = REASONS.index('PillarVFE_slot')
        for i in np.flatnonzero(in_voxel & (reason != REASONS.index('PillarVFE_slot'))):
            if counts[coord_map[tuple(xyz[i, ::-1])]] != self.pre.max_points_per_voxel:
                raise ValueError('Unexplained per-voxel discard below capacity')
        return reason, in_voxel, dict(voxels=len(coords), slots=len(ids),
            exact_4D_features=True, exact_coordinates=True, exact_counts=True,
            valid_slot_original_IDs_unique=True, collation_exact=True, diagnostic_features=5, model_features=4)


def project(cloud, pose, ego_pose, pcd, limit):
    """Same float32 assignment and float64 transform as Runtime.project, no shuffle."""
    from shared_weather_benchmark.geometry import inverse, pose_matrix
    indices = np.arange(len(cloud), dtype=np.int64)
    pc = np.column_stack((cloud, indices.astype(np.float32)))
    after_ego = pcd.mask_ego_points(pc.copy()); s2 = after_ego[:, 4].astype(np.int64)
    m = np.asarray(inverse(pose_matrix(ego_pose))) @ np.asarray(pose_matrix(pose))
    after_ego[:, :3] = after_ego[:, :3] @ m[:3, :3].T + m[:3, 3]
    cropped = pcd.mask_points_by_range(after_ego, limit)
    return np.ascontiguousarray(cropped[:, :4]), s2, cropped[:, 4].astype(np.int64)


def trace(cloud, pose, ego_pose, tracer, pcd, active):
    if not active:
        return (np.full(len(cloud), REASONS.index('inactive_CAV'), dtype=np.uint8),
                np.empty((0, 4), dtype=np.float32), dict(inactive=True), np.zeros(len(cloud), dtype=bool))
    pc, s2, s3 = project(cloud, pose, ego_pose, pcd, tracer.pre.lidar_range)
    reasons = np.full(len(cloud), REASONS.index('ego_filter'), dtype=np.uint8)
    reasons[s2] = REASONS.index('range_filter')
    vr, s4, audit = tracer.slots(pc)
    reasons[s3] = vr; in_voxel = np.zeros(len(cloud), dtype=bool); in_voxel[s3] = s4
    return reasons, pc, audit, in_voxel


def counts(reasons, mask=None):
    a = reasons if mask is None else reasons[mask]
    return {name: int((a == i).sum()) for i, name in enumerate(REASONS)}
