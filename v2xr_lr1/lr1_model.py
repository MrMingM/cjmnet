"""LR-G / LR1: the frozen GSPR and GGCA-A1 methods on LR0's LiDAR path.

No radar, fusion backbone, detection head, GT or post-processing changes.
"""
import json
import sys
from pathlib import Path

import torch
from torch import nn
from opencood.models.point_pillar_intermediate_lrf import PointPillarIntermediateLRF
from opencood.utils.transformation_utils import x1_to_x2

ROOT = Path(__file__).resolve().parent.parent
# The unchanged GGCA module also imports its historical 'bridge' at module load.
if str(ROOT / 'gspr_ggca') not in sys.path:
    sys.path.append(str(ROOT / 'gspr_ggca'))
from attfuse_gspr.reliability import GeometryFirstPillarReliability
from gspr_ggca.modules import GGCA

GSPR_SETTINGS = json.loads((ROOT / 'v2xr_gspr_clean/settings.json').read_text(encoding='utf-8'))
A1_SETTINGS = json.loads((ROOT / 'gspr_ggca/settings.json').read_text(encoding='utf-8'))


def sensor_origins(data, dtype, device):
    """Identical semantics to gspr_ggca/models.py: one origin for each CAV."""
    def plain(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        if isinstance(item, (list, tuple)):
            return [plain(x) for x in item]
        return item

    if 'lidar_pose' not in data or 'ego_flag' not in data:
        raise ValueError('A1 requires real per-CAV lidar_pose and ego_flag')
    poses = plain(data['lidar_pose'])
    flags = plain(data['ego_flag'])
    lengths = data['record_len'].detach().cpu().tolist()
    if len(poses) != len(lengths) or len(flags) != len(lengths):
        raise ValueError('Missing or misaligned per-scene sensor poses')
    origins = []
    for group, group_flags, count in zip(poses, flags, lengths):
        if len(group) != count or len(group_flags) != count or sum(bool(x) for x in group_flags) != 1:
            raise ValueError('Sensor poses and ego flags differ from native CAV order')
        ego = group[next(i for i, flag in enumerate(group_flags) if flag)]
        for pose in group:
            origins.append(x1_to_x2(pose, ego)[:3, 3].tolist())
    return torch.tensor(origins, device=device, dtype=dtype)


class LR1Model(PointPillarIntermediateLRF):
    def __init__(self, args, variant):
        if variant not in ('LRG', 'LR1'):
            raise ValueError('Expected LRG (GSPR) or LR1 (GSPR+A1)')
        # Construct the official dual-modal network before adding extra parameters.
        super().__init__(args)
        self.variant = variant
        self._point_outputs = None
        self._raw = None
        self._point_features = None
        self._origins = None
        self._lidar_calls = 0
        self._radar_calls = 0
        pfn_layers = self.lidar_pillar_vfe.pfn_layers
        if len(pfn_layers) != 1 or not pfn_layers[0].use_norm or pfn_layers[0].linear.out_features != 64:
            raise ValueError('Official LiDAR PFN topology differs from GGCA A1 contract')
        rng = torch.get_rng_state()
        try:
            self.gspr = GeometryFirstPillarReliability(
                GSPR_SETTINGS['model'], args['voxel_size'], args['lidar_range'])
            if variant == 'LR1':
                self.ggca = GGCA(A1_SETTINGS)
        finally:
            torch.set_rng_state(rng)
        self.lidar_pillar_vfe.register_forward_pre_hook(self._before_lidar)
        pfn_layers[0].register_forward_pre_hook(self._before_pfn)
        if variant == 'LR1':
            pfn_layers[0].norm.register_forward_hook(self._capture_norm)
            self.lidar_pillar_vfe.register_forward_hook(self._after_lidar)
        self.radar_pillar_vfe.register_forward_hook(self._after_radar)

    def _before_lidar(self, module, inputs):
        batch = inputs[0]
        self._raw = batch
        self._point_features = None
        self._point_outputs = None
        features, counts = batch['voxel_features'], batch['voxel_num_points']
        if features.ndim != 3 or features.shape[-1] != 4:
            raise ValueError('GSPR needs original LiDAR XYZI voxels')
        if features.shape[0] == 0:
            if len(counts) or len(batch['voxel_coords']):
                raise ValueError('Empty voxel feature and coordinate arrays disagree')
            return
        if (counts <= 0).any() or not torch.isfinite(features).all():
            raise ValueError('Nonfinite or invalid LiDAR pillar input')
        self._point_outputs = self.gspr(features, counts, batch['voxel_coords'])

    def _before_pfn(self, module, inputs):
        if self._point_outputs is None:
            if self._raw is None or self._raw['voxel_features'].shape[0] != 0:
                raise ValueError('LiDAR PFN was called without GSPR')
            return inputs
        r = self._point_outputs['point_reliability']
        decorated = inputs[0]
        if tuple(decorated.shape[:2]) != tuple(r.shape):
            raise ValueError('GSPR and decorated LiDAR point slots do not align')
        return (decorated * r[..., None],)

    def _capture_norm(self, module, inputs, output):
        if self._point_outputs is not None:
            # Capture the original BN output; do not execute BN a second time.
            self._point_features = output.transpose(1, 2).relu()

    def _after_lidar(self, module, inputs, result):
        if self._point_outputs is None:
            if result['pillar_features'].numel():
                raise ValueError('Empty LiDAR must not manufacture pillar features')
            return result
        if self._point_features is None or self._origins is None:
            raise ValueError('A1 missing normalized point features or per-CAV origins')
        raw, out = self._raw, self._point_outputs
        xyz = raw['voxel_features'][..., :3]
        agent = raw['voxel_coords'][:, 0].long()
        if (agent < 0).any() or int(agent.max()) >= len(self._origins):
            raise ValueError('A1 pillar CAV index outside origin list')
        base = result['pillar_features']
        ranges = ((xyz - self._origins[agent, None]).square().sum(-1) + 1e-6).sqrt()
        result['pillar_features'] = self.ggca(
            self._point_features, base.reshape(-1, 64), xyz,
            out['point_valid_mask'], out['point_reliability'],
            out['point_uncertainty'], ranges, 1).reshape_as(base)
        self._point_features = None
        return result

    def _after_radar(self, module, inputs, result):
        self._radar_calls += 1
        return result

    def forward(self, data_dict):
        self._point_outputs = None
        self._raw = None
        self._point_features = None
        self._origins = sensor_origins(
            data_dict, data_dict['processed_lidar']['voxel_features'].dtype,
            data_dict['processed_lidar']['voxel_features'].device) if self.variant == 'LR1' else None
        self._lidar_calls += 1
        return super().forward(data_dict)


class LR1Loss(nn.Module):
    """Native dual-modal detection loss + the original GSPR keep/evidence terms."""
    def __init__(self, native_loss, model):
        super().__init__()
        import weakref
        self.native = native_loss
        # Do not register the model under the loss; optimizer/checkpoint own it.
        self._model = weakref.ref(model)
        self.last_terms = {}

    def forward(self, prediction, target):
        detection = self.native(prediction, target)
        model = self._model()
        if model is None or model._point_outputs is None:
            raise ValueError('Missing real LiDAR observations for GSPR regularization')
        output = model._point_outputs
        valid = output['point_valid_mask']
        if not valid.any():
            raise ValueError('No valid GSPR point for regularization')
        settings = GSPR_SETTINGS['loss']
        keep = (output['point_reliability'][valid].mean() - settings['target_keep_rate']).square()
        evidence = output['point_evidence'][valid].square().mean()
        total = detection + settings['keep_weight'] * keep + settings['evidence_weight'] * evidence
        self.last_terms = {'detection': float(detection.detach()),
                           'keep': float(keep.detach()), 'evidence': float(evidence.detach())}
        return total
