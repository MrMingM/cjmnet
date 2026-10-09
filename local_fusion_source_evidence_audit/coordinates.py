"""Predicted-box coordinates for already ego-projected retained voxel slots."""
import numpy as np
from local_fusion_action_utility_audit.features import geometry
from .common import ROOT, digest

SOURCES = ('opencood/data_utils/datasets/intermediate_fusion_dataset.py',
           'gspr_communication/dataset_adapter.py', 'gspr_evidence/model.py',
           'attfuse_gspr/reliability.py', 'attfuse_gspr/pillar_vfe.py',
           'opencood/utils/box_utils.py')


def specification():
    return {'schema': 1, 'candidate_frame': 'ego_aligned',
            'source_hashes': {p: digest(ROOT / p) for p in SOURCES},
            'tensors': [
                {'name': 'processed_lidar[_weather].voxel_features[..., :3]',
                 'frame': 'ego_aligned', 'source': SOURCES[0],
                 'transform_used': 'source transformation_matrix applied before preprocess when proj_first=True'},
                {'name': 'voxel_coords', 'frame': 'ego_aligned_grid',
                 'source': 'preprocessor/scatter', 'layout': '[source,z,y,x]',
                 'transform_used': 'voxel_size * (index + 0.5) + lidar_range_min'},
                {'name': 'levels[0:2]', 'frame': 'ego_aligned_BEV',
                 'source': SOURCES[2], 'transform_used': 'none; downsampled common XY grid'},
                {'name': 'point/pillar_reliability_uncertainty_evidence',
                 'frame': 'same_retained_voxel_slots', 'source': SOURCES[3],
                 'transform_used': 'none; point_valid_mask and voxel_num_points required'},
                {'name': 'communication_transforms', 'frame': 'source_to_ego',
                 'source': SOURCES[1], 'transform_used': 'pose only; never reproject already aligned points'},
                {'name': 'Shared proposal corners', 'frame': 'ego_aligned',
                 'source': 'oracle._proxy_cache', 'transform_used': 'ego transformation_matrix (required identity)'}],
            'roi': 'oriented Shared predicted box, base expansion in XYZ; no GT boxes',
            'raw_clouds': 'not used; voxelization truncation is explicitly retained-slot evidence',
            'semantic_roi': 'BEV cell centers inside same expanded predicted box; empty has validity=0',
            'unsupported': 'proj_first=False, asynchronous/localization-error pipeline, nonidentity ego transform'}


def local_xyz(points, corners):
    center, size, yaw = geometry(corners)
    delta = np.asarray(points, dtype=np.float64) - center
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack((delta[..., 0]*c + delta[..., 1]*s,
                     -delta[..., 0]*s + delta[..., 1]*c, delta[..., 2]), -1), size


def inside(points, corners, expansion, bev=False):
    local, size = local_xyz(points, corners)
    axes = 2 if bev else 3
    return (np.abs(local[..., :axes]) <= size[:axes] * expansion / 2 + 1e-9).all(-1)


def grid_centers(height, width, lidar_range, z=0.):
    x0, y0, _, x1, y1, _ = lidar_range
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    return np.stack((x0+(xx+.5)*(x1-x0)/width,
                     y0+(yy+.5)*(y1-y0)/height, np.full_like(xx, z, dtype=float)), -1)


def validate(runtime, batch, processed, levels):
    args = runtime.hypes['fusion'].get('args', {})
    if not isinstance(args, dict) or not args.get('proj_first', True):
        raise ValueError('S4 requires explicitly ego-projected voxels')
    ego_transform = batch['ego']['transformation_matrix'].detach().cpu().numpy().reshape(-1, 4, 4)
    if not np.allclose(ego_transform, np.eye(4), atol=1e-6, rtol=0):
        raise ValueError('Nonidentity ego proposal transform: coordinate mapping unsupported')
    coords = processed['voxel_coords'].detach().cpu().numpy()
    points = processed['voxel_features'].detach().cpu().numpy()
    numbers = processed['voxel_num_points'].detach().cpu().numpy().astype(int)
    if numbers.shape != (len(coords),) or np.any(numbers < 0) or np.any(numbers > points.shape[1]):
        raise ValueError('Invalid retained point counts')
    n = int(batch['ego']['record_len'].sum())
    if len(levels[0]) != n or coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError('CAV/voxel layout mismatch')
    if not np.isfinite(coords).all() or not np.array_equal(coords, np.round(coords)):
        raise ValueError('Voxel coordinates must be finite integer grid indices')
    h, w = runtime.model.engine.spatial_shape
    if len(coords) and (np.any(coords < 0) or np.any(coords[:, 0] >= n)
                        or np.any(coords[:, 1] != 0) or np.any(coords[:, 2] >= h) or np.any(coords[:, 3] >= w)):
        raise ValueError('Voxel coordinate outside the verified one-height grid')
    transforms = batch['ego']['communication_transforms'].detach().cpu().numpy()
    if transforms.shape != (n, 4, 4) or not np.isfinite(transforms).all():
        raise ValueError('Source pose order/cardinality mismatch')
    for matrix in transforms:
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0) or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-5, rtol=2e-5):
            raise ValueError('Nonrigid source pose')
    voxel_size = np.asarray(runtime.hypes['preprocess']['args']['voxel_size'])
    limits = runtime.lidar_range
    if not np.allclose([w*voxel_size[0], h*voxel_size[1]], [limits[3]-limits[0], limits[4]-limits[1]], atol=2e-5, rtol=2e-5):
        raise ValueError('Voxel and BEV extents differ')
    valid = np.arange(points.shape[1])[None, :] < numbers[:, None]
    lower = np.asarray(limits[:3]) + coords[:, [3, 2, 1]]*voxel_size
    inside_cell = ((points[..., :3] >= lower[:, None]-2e-5) &
                   (points[..., :3] <= lower[:, None]+voxel_size+2e-5)).all(-1)
    if not np.isfinite(points[valid]).all() or not inside_cell[valid].all():
        raise ValueError('Retained XYZ does not match recorded ego voxel cells')
    return transforms
