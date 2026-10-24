"""Explicit independent adapter, retaining the frozen official attention and heads."""
import time
import torch
from torch import nn
from torch.nn import functional as F
from auditlib import ROOT, load_frozen
from modules import Aggregation, Observation, Enhancement, statistics

integration = load_frozen('gspr_integration', 'integration.py', ROOT/'v2xr_gspr_clean')
FLAGS = {'B0': (False,False,False), 'B1': (True,False,False),
         'B2': (False,False,True), 'B3': (True,True,True), 'B4': (True,True,True)}
OBSERVED_FILM = []
OBSERVATIONS = []


class Model(integration.PointPillarGsprOfficial):
    def __init__(self, args, variant):
        super().__init__(args)
        self.variant = variant
        self.flags = tuple(args.get('repe_modules', FLAGS.get(variant, (False,False,False))))
        self._elapsed = 0
        self._calls = 0
        self._aux = None
        self._obs = None
        self._stats = None
        self._raw = None
        self._decorated = None
        if len(self.pillar_vfe.pfn_layers) != 1:
            raise ValueError('Reviewed integration requires one final PFN')
        pfn = self.pillar_vfe.pfn_layers[0]
        rng = torch.get_rng_state()
        try:
            if self.flags[0]:
                self.repe_a = Aggregation(pfn.linear.in_features, 64)
            if self.flags[1]:
                self.repe_b = Observation()
            if self.flags[2]:
                self.repe_c = Enhancement()
        finally:
            torch.set_rng_state(rng)
        if self.flags[0]:
            pfn.forward = lambda features: self._masked_pfn.unsqueeze(1)
        self.pillar_vfe.register_forward_hook(self._after_vfe)
        self.scatter.register_forward_hook(self._after_scatter)

    def _before_vfe(self, module, inputs):
        if self.flags[0]:
            batch = inputs[0]
            points, counts = batch['voxel_features'], batch['voxel_num_points']
            valid = torch.arange(points.shape[1], device=points.device)[None] < counts[:,None]
            batch['voxel_features'] = points.masked_fill(~valid[...,None], 0)
        super()._before_vfe(module, inputs)
        self._raw = inputs[0]

    def _before_pfn(self, module, inputs):
        if not self.flags[0]:
            return super()._before_pfn(module, inputs)
        self._decorated = inputs[0]
        # Own PFN path: valid slots only through normalization, excluding padding from BN and max.
        valid = self._point_outputs['point_valid_mask']
        x = module.linear(inputs[0][valid])
        if module.use_norm:
            x = module.norm(x) if x.shape[0] > 1 or not module.training else F.batch_norm(
                x, module.norm.running_mean, module.norm.running_var, module.norm.weight, module.norm.bias, training=False)
        x = F.relu(x)
        dense = x.new_full((*valid.shape, x.shape[-1]), -1e4)
        dense[valid] = x
        self._masked_pfn = dense.amax(1)
        # The instance PFN forward returns this masked result; normalization executes once.
        return inputs

    def _after_vfe(self, module, inputs, result):
        if self.flags[0]:
            out = self._point_outputs
            result['pillar_features'] = self.repe_a(self._decorated, self._masked_pfn,
                self._raw['voxel_features'], out['point_valid_mask'], out['point_reliability'], out['point_uncertainty'])
        return result

    def _after_scatter(self, module, inputs, result):
        if not (self.flags[1] or self.flags[2]):
            return result
        x = result['spatial_features']
        raw, out = self._raw, self._point_outputs
        stats = statistics(raw['voxel_features'], out['point_valid_mask'], out['point_reliability'], out['point_uncertainty'])
        maps = x.new_zeros((x.shape[0], 7, *x.shape[-2:]))
        coords = raw['voxel_coords'].long()
        maps[coords[:,0], :, coords[:,2], coords[:,3]] = stats
        self._stats = maps
        occupied = maps[:, :1] > 0
        if self.flags[1]:
            obs = self.repe_b(x, maps)
            self._obs = obs
            # Detached weak quality/support proxies; never assert physical missing-object truth.
            # Normalize effective density by same-distance-bin training batch reference.
            support = torch.zeros_like(maps[:, :1])
            for low, high in ((0,30),(30,60),(60,100),(100,10000)):
                mask = occupied & (maps[:,6:7] >= low) & (maps[:,6:7] < high)
                values = maps[:,1:2][mask].detach()
                reference = values.median().clamp_min(1) if values.numel() else maps.new_tensor(1)
                support = torch.where(mask, (maps[:,1:2].detach()/reference).clamp(0,1), support)
            mask = occupied.expand_as(support)
            qloss = F.binary_cross_entropy(obs['q'][mask].clamp(1e-6,1-1e-6), maps[:,2:3].detach()[mask])
            sloss = F.binary_cross_entropy(obs['s'][mask], support[mask])
            # Paired TRAIN Clean supplies relative support only at observed reference cells.
            reference = getattr(self, '_clean_reference', None)
            if self.training and reference is not None:
                rc = reference['voxel_coords'].long()
                count = maps.new_zeros(maps[:, :1].shape)
                count[rc[:,0], 0, rc[:,2], rc[:,3]] = reference['voxel_num_points'].to(maps.dtype)
                paired_mask = (count >= 2) & occupied
                if paired_mask.any():
                    relative = (maps[:,1:2].detach()/count.clamp_min(1)).clamp(0,1)
                    sloss = F.binary_cross_entropy(obs['s'][paired_mask], relative[paired_mask])
            self._aux = (qloss, sloss)
        else:
            obs = dict(q=occupied.float(), s=occupied.float(), unknown=(~occupied).float(), candidate=occupied.float())
        if self.flags[2]:
            result['spatial_features'] = self.repe_c(x, obs, occupied,
                constrained=self.variant == 'B4', state_gate=self.variant == 'B3')
        return result

    def forward(self, data_dict):
        self._aux = None
        self._clean_reference = data_dict.get('repe_clean_lidar')
        if not self.training:
            torch.cuda.synchronize()
        start = time.monotonic()
        result = super().forward(data_dict)
        if self._aux is not None:
            result['repe_aux'] = self._aux
        if not self.training:
            torch.cuda.synchronize()
            OBSERVATIONS.append(dict(seconds=time.monotonic()-start,
                parameters=sum(p.numel() for p in self.parameters()),
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
                states={k:float(v.detach().mean()) for k,v in (self._obs or {}).items()},
                propagation={k:float(v.mean()) for k,v in self.repe_c.last.items()} if hasattr(self,'repe_c') else {}))
        self._elapsed += time.monotonic()-start
        self._calls += 1
        return result

    def repe_check_gradients(self):
        for name in ('repe_a','repe_b','repe_c'):
            if hasattr(self, name):
                grads = [p.grad for p in getattr(self,name).parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):
                    raise ValueError('Missing/nonfinite/zero gradients: '+name)

    def repe_statistics(self, loss):
        result = dict(loss=loss.loss_dict.copy(), forward_host_seconds=self._elapsed/max(self._calls,1))
        if self._obs:
            result.update({k: float(v.detach().mean()) for k,v in self._obs.items()})
        if hasattr(self,'repe_c'):
            result.update({k:float(v.mean()) for k,v in self.repe_c.last.items()})
        return result


class Loss(integration.GsprOfficialLoss):
    def forward(self, output_dict, target_dict, epoch=1, train=True):
        total = super().forward(output_dict, target_dict, epoch, train)
        if 'repe_aux' in output_dict:
            q, s = output_dict['repe_aux']
            total = total + .01*q + .01*s
            self.loss_dict.update(repe_q=float(q.detach()), repe_s=float(s.detach()), repe_q_weight=.01, repe_s_weight=.01)
        return total


def create(hypes, variant):
    return Model(hypes['model']['args'], variant)


def criterion(hypes, variant, film=False):
    return Loss(hypes['loss']['args'])


def install(variant):
    from opencood.tools import train_utils
    train_utils.create_model = lambda h: create(h, variant)
    train_utils.create_loss = lambda h: criterion(h, variant)


def frozen_backbone(model):
    raise ValueError('FiLM is outside REPE protocol')


def strict_backbone(model, base):
    raise ValueError('FiLM is outside REPE protocol')
