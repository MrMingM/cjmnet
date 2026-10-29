"""Attach GGCA to the official PFN outputs; original hooks and forward stay intact."""
import time
import torch
from auditlib import METHOD, ROOT, load_frozen
from modules import GGCA

integration = load_frozen('gspr_integration', 'integration.py', ROOT / 'v2xr_gspr_clean')
OBSERVATIONS = []
OBSERVED_FILM = []  # Historical engine interface only. No FiLM is supported.


def sensor_origins(data, dtype, device):
    from opencood.utils.transformation_utils import x1_to_x2
    def plain(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        return x
    poses, flags = plain(data['lidar_pose']), plain(data['ego_flag'])
    lengths = data['record_len'].detach().cpu().tolist()
    if len(poses) != len(lengths) or len(flags) != len(lengths):
        raise ValueError('Missing per-batch sensor poses')
    origins = []
    for ps, fs, count in zip(poses, flags, lengths):
        if len(ps) != count or len(fs) != count or sum(bool(v) for v in fs) != 1:
            raise ValueError('Sensor poses/ego flags do not match official CAV order')
        ego = ps[next(i for i, v in enumerate(fs) if v)]
        origins.extend(x1_to_x2(pose, ego)[:3, 3].tolist() for pose in ps)
    return torch.tensor(origins, dtype=dtype, device=device)


class Model(integration.PointPillarGsprOfficial):
    def __init__(self, args, variant):
        if variant not in ('A0', 'A1', 'A2', 'A3'):
            raise ValueError('Unknown GGCA variant')
        super().__init__(args)
        self.variant, self.level = variant, int(variant[1])
        self.enabled = self.level > 0
        self._raw = self._point_features = self._origins = None
        self._before_update = {}
        self.weighting_checks = 0
        self.last_pfn = None
        if self.enabled:
            if len(self.pillar_vfe.pfn_layers) != 1:
                raise ValueError('This reviewed integration requires one final PFN')
            pfn = self.pillar_vfe.pfn_layers[0]
            if not pfn.use_norm or pfn.linear.out_features != 64:
                raise ValueError('Official normalized 64-channel PFN contract differs')
            rng = torch.get_rng_state()
            try:
                self.ggca = GGCA(METHOD)
            finally:
                torch.set_rng_state(rng)
            # Capture the SAME normalization output; no norm/linear re-execution.
            pfn.norm.register_forward_hook(self._capture_normalized)
            self.pillar_vfe.register_forward_hook(self._after_vfe)

    def _before_vfe(self, module, inputs):
        super()._before_vfe(module, inputs)
        self._raw = inputs[0]

    def _before_pfn(self, module, inputs):
        weighted = super()._before_pfn(module, inputs)
        if self.force_unit_weights:
            raise ValueError('Unit reliability is forbidden in GGCA runs; use explicit diagnostic only')
        # The returned tensor is exactly the original GSPR-weighted PFN input.
        if not torch.equal(weighted[0], inputs[0]*self._point_outputs['point_reliability'][..., None]):
            raise ValueError('Original GSPR feature weighting was bypassed')
        self.weighting_checks += 1
        return weighted

    def _capture_normalized(self, module, inputs, output):
        self._point_features = output.transpose(1, 2).relu()
        # No detach: detection gradients flow through the official PFN and GSPR.

    def _after_vfe(self, module, inputs, result):
        original = result['pillar_features']
        self.last_pfn = original.detach().clone() if getattr(self, 'capture_pfn', False) else None
        if not self.enabled:
            return result
        raw, out = self._raw, self._point_outputs
        agent = raw['voxel_coords'][:, 0].long()
        if self._origins is None or int(agent.max()) >= len(self._origins):
            raise ValueError('Sensor-origin mapping is missing')
        xyz = raw['voxel_features'][..., :3]
        ranges = ((xyz-self._origins[agent, None]).square().sum(-1)+1e-6).sqrt()
        changed = self.ggca(self._point_features, original.reshape(-1, 64), xyz,
            out['point_valid_mask'], out['point_reliability'], out['point_uncertainty'], ranges, self.level)
        result['pillar_features'] = changed.reshape_as(original)
        self._point_features = None
        return result

    def forward(self, data_dict):
        self._origins = sensor_origins(data_dict, data_dict['processed_lidar']['voxel_features'].dtype,
            data_dict['processed_lidar']['voxel_features'].device) if self.enabled else None
        if not self.training:
            torch.cuda.synchronize()
        start = time.monotonic()
        result = super().forward(data_dict)
        if not self.training:
            torch.cuda.synchronize()
            OBSERVATIONS.append(dict(seconds=time.monotonic()-start,
                parameters=sum(p.numel() for p in self.parameters()),
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
                reliability_histogram=torch.histc(result['point_reliability'].detach()[result['point_valid_mask']],
                                                 bins=20, min=0, max=1).tolist(),
                uncertainty_histogram=torch.histc(result['point_uncertainty'].detach()[result['point_valid_mask']],
                                                 bins=20, min=0, max=1).tolist(),
                ggca=self.ggca.last if hasattr(self, 'ggca') else {}))
        return result

    def repe_check_gradients(self):
        gradients = [p.grad for p in self.ggca.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError('Missing/nonfinite GGCA gradients')
        if not any(g.abs().sum() > 0 for g in gradients):
            raise ValueError('All GGCA gradients are zero')
        self._before_update = {k: p.detach().clone() for k, p in self.ggca.named_parameters()}

    def repe_statistics(self, loss):
        changes = {k: float((p.detach()-self._before_update[k]).square().sum().sqrt())
                   for k, p in self.ggca.named_parameters()}
        gradients = {k: float(p.grad.detach().square().sum().sqrt())
                     for k, p in self.ggca.named_parameters() if p.grad is not None}
        self._before_update = {}
        return dict(loss=loss.loss_dict.copy(), ggca=self.ggca.last,
            gradient_l2=gradients, update_l2=changes, original_weighting_checks=self.weighting_checks)


def create(hypes, variant):
    return Model(hypes['model']['args'], variant)


def criterion(hypes, variant, film=False):
    if film:
        raise ValueError('No FiLM or auxiliary loss in GGCA')
    return integration.GsprOfficialLoss(hypes['loss']['args'])


def install(variant):
    from opencood.tools import train_utils
    train_utils.create_model = lambda h: create(h, variant)
    train_utils.create_loss = lambda h: criterion(h, variant)


def frozen_backbone(*args):
    raise ValueError('GGCA uses independent full training')


strict_backbone = frozen_backbone
