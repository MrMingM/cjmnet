"""Pinned original UECP model/loss in a private namespace; official V2X-R postprocess aliases."""
import random

import numpy as np
import torch
from torch.nn import functional as F

from .common import read
from .runtime import host_config
from .upstream import module
from .quality_adapter import load_quality, aggregate, regional, shuffled, QualityAdapter, describe
from .dual_quality_fusion import DualFusion, enable_precision_gradient

original = module('models.bev_text_v1')


class Detector(original.BEVtextv1):
    def __init__(self, args, config, variant):
        super().__init__(args)
        if self.use_gt_unmap or not self.supervise_single:
            raise ValueError('Require predicted uncertainty and original single-CAV supervision')
        self.variant = variant
        self.config = config
        self.last_stats = {}
        self._precision_stats = []
        self._empty = None
        def uncertainty_stats(layer, inputs, out):
            self.last_u = describe(out)
        self.uncertainty_head.register_forward_hook(uncertainty_stats)
        def empty_hook(layer, inputs, out):
            if self._empty is not None:
                out['pillar_features'] = out['pillar_features'].masked_fill(self._empty[:, None], 0)
            return out
        self.pillar_vfe.register_forward_hook(empty_hook)
        for scale, block in zip(self.fusion_net.scales, self.fusion_net.blocks):
            blur = block._blur3
            def observed_blur(x, blur=blur, scale=scale):
                result = blur(x)
                weights = result.detach()
                weights = weights/(weights.sum(1, keepdim=True)+1e-6)
                self._precision_stats.append(dict(scale=scale, normalized_weight=describe(weights),
                    maximum_weight=float(weights.max()),
                    entropy=float(-(weights*weights.clamp_min(1e-12).log()).sum(1).mean())))
                return result
            block._blur3 = observed_blur
        if variant != 'E0':
            # Extra initialization must not advance the host model RNG stream.
            with torch.random.fork_rng():
                torch.manual_seed(config['seed'] + 91)
                self.gspr = load_quality(config)
                self.quality_adapter = QualityAdapter()
            if variant in ('E2', 'E3_no_separation'):
                enable_precision_gradient(self.fusion_net)
            elif variant.startswith('E3_'):
                self.fusion_net = DualFusion.from_original(self.fusion_net)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'gspr'):
            self.gspr.eval()
        return self

    def quality(self, data, hw):
        p = data['processed_lidar']
        with torch.no_grad():
            values = self.gspr(p['voxel_features'], p['voxel_num_points'], p['voxel_coords'])
            grid = aggregate(values, p, int(data['record_len'].sum()), data['empty_voxels'],
                             low=self.config['diagnosis']['low_reliability'])
            q, valid = regional(grid, hw)
        return q, valid

    def forward(self, data_dict):
        data = data_dict.get('uecp_input', data_dict)
        self._empty = data['empty_voxels']
        self._precision_stats = []
        try:
            if self.variant == 'E0':
                # This call executes the original upstream forward, auxiliary loss and all.
                output = super().forward(data)
                with torch.no_grad():
                    self.last_stats = dict(model='original UECP forward', gspr=False, U=self.last_u, scales=self._precision_stats)
            else:
                output = self.improved_forward(data)
        finally:
            self._empty = None
        output.update(psm=output['cls_preds'], rm=output['reg_preds'], dm=output['dir_preds'])
        return output

    def improved_forward(self, data):
        feature = self.extract_pts_feat(data)
        u = self.uncertainty_head(feature)
        out = {}
        if 'un_maps' in data:
            target = F.interpolate(self._format_uncertainty_map(data['un_maps']), size=u.shape[-2:], mode='bilinear', align_corners=False)
            out['aux_loss'] = 5*self.uncertainty_loss_func(u, target)
        out.update(cls_preds_single=self.cls_head_single(feature), reg_preds_single=self.reg_head_single(feature),
                   dir_preds_single=self.dir_head_single(feature))
        q, valid = self.quality(data, u.shape[-2:])
        coverage = (1-u)*valid
        counts = F.adaptive_avg_pool2d(data['point_counts'].float(), u.shape[-2:]) * 4
        if self.variant == 'E3_shuffled':
            q = shuffled(q, valid, counts, data['quality_seeds'])
        elif self.variant in ('E3_constant_quality', 'E3_coverage'):
            q = torch.cat((torch.full_like(u, .5), torch.full_like(u, .5), torch.full_like(u, .5), torch.zeros_like(u)), 1)*valid
        affine = original.normalize_pairwise_tfm(data['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        if self.variant in ('E2', 'E2_original_detach', 'E3_no_separation'):
            adjusted = self.quality_adapter(u, coverage, q, valid)
            adjusted = torch.where(valid, adjusted, torch.ones_like(adjusted))
            fused = self.fusion_net(feature, adjusted, data['record_len'], affine)
        else:
            if self.variant == 'E3_quality':
                coverage = valid.float()
            base = q[:, :1]
            adapted = self.quality_adapter(base, coverage if self.variant != 'E3_quality' else coverage*.5, q, valid)
            if self.variant == 'E3_coverage':
                adapted = torch.ones_like(base)  # No learned or frozen Q information in this ablation.
            fused = self.fusion_net(feature, coverage, adapted*valid, valid, data['record_len'], affine)
        self.last_stats = dict(U=describe(u), C=describe(coverage, valid),
            Q=describe(q[:, :1], valid), Q_uncertainty=describe(q[:, 2:3], valid),
            observed_fraction=float(valid.float().mean()),
            adapter=describe(adjusted if self.variant in ('E2', 'E2_original_detach', 'E3_no_separation') else adapted, valid),
            scales=getattr(self.fusion_net, 'last_stats', self._precision_stats))
        self.last_stats['quality_nearly_constant'] = self.last_stats['Q']['std'] is not None and self.last_stats['Q']['std'] < 1e-3
        self.last_stats['adapter_nearly_constant'] = self.last_stats['adapter']['std'] is not None and self.last_stats['adapter']['std'] < 1e-3
        out.update(cls_preds=self.cls_head(fused), reg_preds=self.reg_head(fused), dir_preds=self.dir_head(fused))
        return out


def make(output, variant):
    c = read(output / 'config.json')
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return Detector(host_config(output)['model']['args'], c, variant)


def criterion(output):
    cfg = host_config(output)
    return module('loss.' + cfg['loss']['core_method']).PointPillarDepthLoss(cfg['loss']['args'])


def losses(crit, output, labels, single):
    labels = {k: v.float() for k, v in labels.items()}
    single = {k: v.float() for k, v in single.items()}
    fused = crit(output, labels)
    # Preserve released UECP train.py exactly: aux_loss is added by both calls.
    one = crit(output, single, suffix='_single')
    return fused + one, dict(fused=float(fused.detach()), single=float(one.detach()),
        auxiliary_each=float(output['aux_loss'].detach()) if 'aux_loss' in output else 0.)


def shared_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith(('gspr.', 'quality_adapter.'))}
