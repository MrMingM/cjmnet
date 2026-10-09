"""Small descriptive evidence features; no batch, GT or labels accepted here."""
import re
import numpy as np
from local_fusion_action_utility_audit.features import schema as old_schema, geometry
from .common import KEEP
from .coordinates import inside, local_xyz, grid_centers

STATS = ('valid_count', 'mean', 'std', 'min', 'max', 'q10', 'q25', 'q50', 'q75', 'q90')
GEO = ('point_count', 'valid_point_count', 'nonzero_voxel_count', 'nonzero_pillar_count',
       'occupied_pillar_ratio', 'variance_x', 'variance_y', 'variance_z',
       'xy_eigen_min', 'xy_eigen_max', 'z_range', 'xy_spread', 'longitudinal_coverage',
       'lateral_coverage', 'front_coverage', 'back_coverage', 'left_coverage',
       'right_coverage', 'quadrant_coverage', 'boundary_support',
       'distance_to_source', 'bearing_sin', 'bearing_cos', 'view_yaw_sin', 'view_yaw_cos')
SEM_LOCAL = ('l1', 'l2', 'channel_mean', 'channel_std', 'spatial_variance', 'max_activation', 'positive_ratio')
SEM_REL = tuple(f'{ref}_{stat}' for ref in ('shared_cosine', 'ego_cosine', 'other_cosine', 'mean_distance', 'median_distance') for stat in ('mean', 'std', 'max', 'min'))
SEM = tuple(f's{scale}_{key}' for scale in (0, 1) for key in SEM_LOCAL + SEM_REL)
GSPR = tuple(f'{field}_{stat}' for field in ('point_reliability', 'point_uncertainty', 'point_evidence_reliable', 'point_evidence_unreliable', 'pillar_reliability', 'pillar_uncertainty') for stat in STATS) + ('point_coverage', 'pillar_coverage', 'observation_valid')
LOO_KEYS = ('applicable', 'full_logit', 'removed_logit', 'delta_logit', 'delta_probability',
            'roi_max_delta', 'roi_mean_delta', 'delta_x', 'delta_y', 'delta_z',
            'delta_length', 'delta_width', 'delta_height', 'delta_yaw_sin',
            'delta_yaw_cos', 'rm_l1', 'rm_l2', 'before_after_pred_iou') + tuple(f'rm_delta_{i}' for i in range(7))
LOO = tuple(f'{ref}_{key}' for ref in ('shared', 'attfuse') for key in LOO_KEYS) + ('query_self_applicable',)
SQ = tuple(f'{scope}_{key}' for scope in ('anchor', 'roi_max', 'roi_mean') for key in ('single', 'query', 'query_minus_single', 'query_minus_shared', 'single_minus_shared'))
BLOCKS = {'geometry': GEO, 'semantic': SEM, 'gspr': GSPR, 'loo': LOO, 'singlequery': SQ}
GROUP_BLOCKS = {
    'output_only': (), 'output_plus_geometry': ('geometry', 'loo'),
    'output_plus_semantic': ('semantic', 'singlequery', 'loo'),
    'output_plus_gspr': ('gspr',),
    'output_plus_geometry_semantic': ('geometry', 'semantic', 'singlequery', 'loo'),
    'all_evidence': tuple(BLOCKS),
    'all_evidence_without_loo': ('geometry', 'semantic', 'gspr', 'singlequery'),
    'all_evidence_without_gspr': ('geometry', 'semantic', 'loo', 'singlequery')}


def block_names(block):
    keys = [block + '_' + name for name in BLOCKS[block]]
    return keys + [name + '_valid' for name in keys]


def block_columns(block, group):
    if block != 'loo' or group not in ('output_plus_geometry', 'output_plus_semantic'):
        return list(range(len(block_names(block))))
    classification = ('full_logit', 'removed_logit', 'delta_logit', 'delta_probability', 'roi_max_delta', 'roi_mean_delta')
    selected = []
    for i, name in enumerate(LOO):
        is_cls = any(name.endswith('_'+key) for key in classification)
        applicability = name.endswith('_applicable')
        if applicability or (is_cls if group == 'output_plus_semantic' else not is_cls):
            selected.append(i)
    return selected + [len(LOO)+i for i in selected]


def schema(task, group):
    names = old_schema(task) + [block_names(block)[i] for block in GROUP_BLOCKS[group] for i in block_columns(block, group)]
    check_names(names, task)
    return names


def check_names(names, task):
    allowed = set(old_schema(task)) | {key for block in BLOCKS for key in block_names(block)}
    forbidden = {'gt', 'target', 'focal', 'matched', 'unmatched', 'oracle', 'recovered', 'lost', 'post', 'nms', 'cav_index', 'source_id'}
    if len(names) != len(set(names)):
        raise ValueError('Duplicate schema field')
    for name in names:
        if name not in allowed or forbidden.intersection(re.split('[^a-z0-9]+', name.lower())):
            raise ValueError('Unknown/leaking feature: ' + name)


def feature_leakage_check(names, task):
    """Unknown fields are rejected by default, including GT-derived aliases."""
    check_names(names, task)
    return True


def schema_document():
    return {'schema': 1, 'primary_group': 'all_evidence',
            'tasks': {t: {g: schema(t, g) for g in GROUP_BLOCKS} for t in ('cls', 'reg')},
            'old_output_primary': 'with_competition_features', 'inference_gt_fields': 0,
            'validity': 'Each new scalar has a corresponding *_valid column; zero placeholders never denote observations.',
            'query': 'query:i consumes all sources; local evidence belongs to its underlying query CAV. Absolute index is not a feature.',
            'keep': 'pooled retained support; learned B0 fused BEV; no underlying source pose or LOO',
            'geometry_output_subgroup': [x for x in old_schema('reg') if x not in old_schema('cls')],
            'loo': 'B0 Shared reroute and original ego-query AttFuse separate; removing ego undefined. Query-self removal undefined.',
            'group_semantics': {k: list(v) for k, v in GROUP_BLOCKS.items()}}


def vector(block, values):
    if set(values)-set(BLOCKS[block]):
        raise ValueError('Unknown evidence fields: '+str(sorted(set(values)-set(BLOCKS[block]))))
    data, valid = [], []
    for name in BLOCKS[block]:
        value = values.get(name)
        flag = value is not None
        if flag and not np.isfinite(value):
            raise ValueError('Nonfinite evidence: ' + block + '/' + name)
        data.append(float(value) if flag else 0.)
        valid.append(float(flag))
    return np.asarray(data + valid, dtype=np.float32)


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite GSPR observation')
    output = {'valid_count': len(values)}
    if len(values):
        output.update(mean=values.mean(), std=values.std(), min=values.min(), max=values.max())
        output.update(zip(('q10', 'q25', 'q50', 'q75', 'q90'), np.quantile(values, [.1, .25, .5, .75, .9])))
    return output


def geometric(points, pillar_ids, corners, expansion, possible_pillars, pose=None):
    """points are valid retained slots in ego frame, already clipped to predicted ROI."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pillars = len(np.unique(pillar_ids, axis=0)) if len(pillar_ids) else 0
    values = dict(point_count=len(points), valid_point_count=len(points),
                  nonzero_voxel_count=pillars, nonzero_pillar_count=pillars,
                  occupied_pillar_ratio=pillars / possible_pillars if possible_pillars else None)
    center, size, yaw = geometry(corners)
    if pose is not None:
        delta = center - pose[:3, 3]
        bearing = np.arctan2(delta[1], delta[0]) - np.arctan2(pose[1, 0], pose[0, 0])
        view = yaw - np.arctan2(delta[1], delta[0])
        values.update(distance_to_source=np.linalg.norm(delta), bearing_sin=np.sin(bearing),
                      bearing_cos=np.cos(bearing), view_yaw_sin=np.sin(view), view_yaw_cos=np.cos(view))
    if len(points):
        local, _ = local_xyz(points, corners)
        normalized = local[:, :2] / (size[:2] * expansion / 2).clip(1e-6)
        variance = points.var(0)
        eigen = np.linalg.eigvalsh((local[:, :2]-local[:, :2].mean(0)).T @ (local[:, :2]-local[:, :2].mean(0)) / len(points))
        bins = np.floor((normalized + 1)*2).astype(int).clip(0, 3)
        values.update(variance_x=variance[0], variance_y=variance[1], variance_z=variance[2],
                      xy_eigen_min=eigen[0], xy_eigen_max=eigen[1], z_range=np.ptp(points[:, 2]),
                      xy_spread=np.linalg.norm(np.ptp(local[:, :2], axis=0)),
                      longitudinal_coverage=len(np.unique(bins[:, 0]))/4,
                      lateral_coverage=len(np.unique(bins[:, 1]))/4,
                      front_coverage=len(np.unique(bins[normalized[:, 0] >= 0, 1]))/4,
                      back_coverage=len(np.unique(bins[normalized[:, 0] < 0, 1]))/4,
                      left_coverage=len(np.unique(bins[normalized[:, 1] >= 0, 0]))/4,
                      right_coverage=len(np.unique(bins[normalized[:, 1] < 0, 0]))/4,
                      quadrant_coverage=len(np.unique((normalized[:, 0] >= 0).astype(int)*2+(normalized[:, 1] >= 0)))/4,
                      boundary_support=np.mean(np.max(np.abs(normalized), axis=1) >= .8))
    return values


def semantic(source, shared, ego, all_sources, mask, underlying):
    """Input [C,H,W] and [S,C,H,W]; local comparisons use the identical BEV mask."""
    if not mask.any():
        return {}
    x = source[:, mask].astype(np.float64)
    local_sources = all_sources[:, :, mask].astype(np.float64)
    others = local_sources if underlying is None else np.delete(local_sources, underlying, axis=0)
    mean = local_sources.mean(0)
    median = np.median(local_sources, axis=0)
    values = dict(l1=np.abs(x).mean(), l2=np.sqrt((x*x).sum(0)).mean(),
                  channel_mean=x.mean(0).mean(), channel_std=x.std(0).mean(),
                  spatial_variance=x.var(1).mean(), max_activation=x.max(), positive_ratio=(x > 0).mean())
    comparisons = {'shared_cosine': shared[:, mask], 'ego_cosine': ego[:, mask]}
    if len(others):
        comparisons['other_cosine'] = others.mean(0)
    arrays = {'mean_distance': np.linalg.norm(x-mean, axis=0), 'median_distance': np.linalg.norm(x-median, axis=0)}
    for key, reference in comparisons.items():
        denominator = np.linalg.norm(x, axis=0)*np.linalg.norm(reference, axis=0)
        valid = denominator > 1e-12
        if valid.any():
            arrays[key] = (x*reference).sum(0)[valid]/denominator[valid]
    for key, value in arrays.items():
        for stat in ('mean', 'std', 'max', 'min'):
            values[key+'_'+stat] = getattr(value, stat)()
    return values


def combine(old, blocks, group):
    result = np.concatenate([np.asarray(old, dtype=np.float32)] + [blocks[k][..., block_columns(k, group)] for k in GROUP_BLOCKS[group]], axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite feature matrix')
    return result
