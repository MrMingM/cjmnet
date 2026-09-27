"""B1 local/global gates between frozen B0 Collapse and Split endpoints.

B0 already learned two task-specific source distributions.  B1 never relearns
those distributions.  It only predicts how far each BEV cell should move from
their common average (Collapse) toward the original task-specific weights
(Split).
"""
import torch
import torch.nn.functional as F
from torch import nn

from local_fusion_task_split_pilot.model import (
    _decode_task,
    _fuse_from_weights,
    _mean_weights,
    _route,
)


def _same_hw(value, hw):
    if value.shape[-2:] == tuple(hw):
        return value
    return F.interpolate(value, size=tuple(hw), mode='bilinear', align_corners=False)


class TaskConflictGate(nn.Module):
    """Tiny gate with identical parameters for local and global variants."""

    def __init__(self, scales=(0, 1), hidden=8, mode='local', initial_bias=-4.0):
        super().__init__()
        if mode not in ('local', 'global'):
            raise ValueError('mode must be local or global')
        if not scales or len(set(scales)) != len(scales):
            raise ValueError('scales must be non-empty and unique')
        if hidden < 1:
            raise ValueError('hidden must be positive')
        self.scales = tuple(int(x) for x in scales)
        self.mode = mode
        self.hidden = int(hidden)
        self.initial_bias = float(initial_bias)
        self.networks = nn.ModuleDict()
        for scale in self.scales:
            network = nn.Sequential(
                nn.Conv2d(3, self.hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(self.hidden, 1, 1),
            )
            nn.init.zeros_(network[-1].weight)
            nn.init.constant_(network[-1].bias, self.initial_bias)
            self.networks[str(scale)] = network

    def forward(self, scale, features):
        key = str(int(scale))
        if key not in self.networks:
            raise ValueError('Gate requested for an unconfigured scale')
        if features.ndim != 4 or features.shape[1] != 3:
            raise ValueError('Gate input must have shape [B,3,H,W]')
        if self.mode == 'global':
            pooled = features.mean(dim=(-2, -1), keepdim=True)
            value = torch.sigmoid(self.networks[key](pooled))
            return value.expand(-1, -1, features.shape[-2], features.shape[-1])
        return torch.sigmoid(self.networks[key](features))

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def prepare_frozen_context(base, levels, split_arm, scales=(0, 1)):
    """Compute B0 endpoints and deployable task-conflict signals once.

    All B0 routers and detector modules are frozen.  The returned tensors are
    detached from those modules; gradients in B1 can only flow through the new
    gate and the interpolation from common to Split weights.
    """
    scales = tuple(int(x) for x in scales)
    with torch.no_grad():
        _, weights_cls, _ = _route(split_arm.router_a, levels)
        _, weights_reg, _ = _route(split_arm.router_b, levels)
        common = _mean_weights(weights_cls, weights_reg)

        common_levels = _fuse_from_weights(levels, common)
        split_cls_levels = _fuse_from_weights(levels, weights_cls)
        split_reg_levels = _fuse_from_weights(levels, weights_reg)

        collapse_prediction = {
            'psm': _decode_task(base, common_levels, base.cls_head),
            'rm': _decode_task(base, common_levels, base.reg_head),
        }
        split_prediction = {
            'psm': _decode_task(base, split_cls_levels, base.cls_head),
            'rm': _decode_task(base, split_reg_levels, base.reg_head),
        }
        activity_full = torch.sigmoid(collapse_prediction['psm']).amax(
            dim=1, keepdim=True)

        signals = {}
        for scale in scales:
            if scale >= len(levels):
                raise ValueError('Configured gate scale is absent from feature levels')
            left, right = weights_cls[scale], weights_reg[scale]
            if left.shape != right.shape:
                raise ValueError('B0 task-weight shapes differ')
            if len(levels[scale]) <= 1:
                continue
            delta = (left - right).abs()
            tv = 0.5 * delta.sum(dim=0, keepdim=True)
            top_disagree = (
                left.argmax(dim=0, keepdim=True)
                != right.argmax(dim=0, keepdim=True)
            ).float()
            activity = _same_hw(activity_full, tv.shape[-2:])
            features = torch.cat((tv, top_disagree, activity), dim=1)
            signals[scale] = {
                'features': features.detach(),
                'total_variation': tv.detach(),
                'top_disagreement': top_disagree.detach(),
                'activity': activity.detach(),
            }

    return {
        'levels': levels,
        'weights_cls': [x.detach() for x in weights_cls],
        'weights_reg': [x.detach() for x in weights_reg],
        'common_weights': [x.detach() for x in common],
        'signals': signals,
        'collapse_prediction': collapse_prediction,
        'split_prediction': split_prediction,
    }


def _interpolate(common, task, gate_map):
    """Convex interpolation preserving a valid per-source distribution."""
    local_gate = gate_map.squeeze(0)
    if local_gate.ndim != 3 or local_gate.shape[0] != 1:
        raise ValueError('Gate map must reduce to [1,H,W] for source broadcast')
    if common.shape[-2:] != local_gate.shape[-2:]:
        raise ValueError('Gate/source-weight spatial shapes differ')
    return common + local_gate * (task - common)


def predict_with_gate(base, frozen, gate):
    """Apply a trainable local/global gate to frozen B0 task weights."""
    levels = frozen['levels']
    cls_weights, reg_weights = [], []
    gate_maps = {}
    for scale, (common, cls_raw, reg_raw) in enumerate(zip(
            frozen['common_weights'], frozen['weights_cls'], frozen['weights_reg'])):
        if scale in frozen['signals']:
            signal = frozen['signals'][scale]
            value = gate(scale, signal['features'])
            cls_weights.append(_interpolate(common, cls_raw, value))
            reg_weights.append(_interpolate(common, reg_raw, value))
            gate_maps[scale] = value
        else:
            cls_weights.append(common)
            reg_weights.append(common)

    cls_levels = _fuse_from_weights(levels, cls_weights)
    reg_levels = _fuse_from_weights(levels, reg_weights)
    prediction = {
        'psm': _decode_task(base, cls_levels, base.cls_head),
        'rm': _decode_task(base, reg_levels, base.reg_head),
    }

    zero = levels[0].new_zeros(())
    means = [value.mean() for value in gate_maps.values()]
    info = {
        'gate_mean': torch.stack(means).mean() if means else zero,
        'gate_maps': gate_maps,
        'signals': frozen['signals'],
    }
    for scale in gate.scales:
        value = gate_maps.get(scale)
        info[f'gate_s{scale}'] = value.mean() if value is not None else zero
    return prediction, info
