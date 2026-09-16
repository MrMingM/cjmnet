"""Point evidence summaries, local queries, and receiver-only reliability correction."""
import torch
import math
from torch import nn


def point_cells(processed, grid, z_range, z_bins, ids):
    coords = processed['voxel_coords'].long()
    xyz = processed['voxel_features'][..., :3]
    h, w = grid
    if coords.numel() and ((coords[:, 2] < 0).any() or (coords[:, 2] >= h*8).any() or (coords[:, 3] < 0).any() or (coords[:, 3] >= w*8).any()):
        raise ValueError('Voxel outside aligned grid')
    lookup = torch.full((h*w,), -1, device=coords.device, dtype=torch.long)
    lookup[ids] = torch.arange(len(ids), device=coords.device)
    block = (coords[:, 2]//8)*w + coords[:, 3]//8
    qi = lookup[block][:, None].expand(xyz.shape[:2])
    z = ((xyz[..., 2]-z_range[0])/(z_range[1]-z_range[0])*z_bins).floor().long().clamp(0, z_bins-1)
    cell = (((qi.clamp_min(0)*8 + coords[:, 2, None]%8)*8 + coords[:, 3, None]%8)*z_bins+z)
    return qi, cell


def summarize(processed, rel, ids, grid, z_range, z_bins):
    """Only this sender's point slots and the received block IDs are accepted."""
    qi, cell = point_cells(processed, grid, z_range, z_bins, ids)
    valid = rel['point_valid_mask'] & (qi >= 0)
    e = rel['point_evidence']
    strength = e.sum(-1)+2
    beliefs = torch.stack((e[..., 0]/strength, e[..., 1]/strength, 2/strength), -1)
    count = e.new_zeros(len(ids)*64*z_bins)
    sums = e.new_zeros((len(count), 3))
    if valid.any():
        index = cell[valid]
        count.index_add_(0, index, torch.ones_like(index, dtype=e.dtype))
        sums.index_add_(0, index, beliefs[valid])
    means = sums / count[:, None].clamp_min(1)
    # Saturated retained-slot support, not raw density or visibility probability.
    return torch.cat((means, (count/(count+4))[:, None]), -1).reshape(len(ids), 8, 8, z_bins, 4)


def combine(current, received):
    """Keep one strongest reliable observation per cell; never sum correlated evidence."""
    strength = received[..., 0]*received[..., 3]
    old_strength = current[..., 0]*current[..., 3]
    return torch.where((strength > old_strength)[..., None], received, current)


class PointReviewer(nn.Module):
    def __init__(self, hidden=32, max_adjustment=.25, floor=.05):
        super().__init__()
        if hidden < 1 or not 0 < max_adjustment <= 1 or not 0 <= floor < 1:
            raise ValueError('Invalid point reviewer configuration')
        self.net = nn.Sequential(nn.Linear(11, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.maximum, self.floor = max_adjustment, floor

    def weight_delta(self, inputs):
        scale = float(getattr(self, 'evaluation_scale', 1.))
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Evaluation scale must be finite and nonnegative')
        if self.training and scale != 1.:
            raise ValueError('Amplitude scanning is evaluation-only')
        delta = self.maximum*self.net(inputs).squeeze(-1).tanh()
        if scale != 1.:
            delta = (delta*scale).clamp(-self.maximum, self.maximum)
        return delta

    def forward(self, processed, rel, ids, received, grid, z_range, z_bins):
        original = rel['point_reliability']
        if not len(ids):
            return original, torch.zeros_like(original, dtype=torch.bool)
        qi, cell = point_cells(processed, grid, z_range, z_bins, ids)
        peer = received.reshape(-1, 4)[cell]
        eligible = rel['point_valid_mask'] & (qi >= 0) & (peer[..., 3] > 0) & (peer[..., 0] > 0)
        e = rel['point_evidence']
        s = e.sum(-1)+2
        p, u = (e[..., 0]+1)/s, 2/s
        xyz = processed['voxel_features'][..., :3]
        # Point position within pillar plus bounded height; no remote dense feature input.
        valid = rel['point_valid_mask']
        centroid = (xyz*valid[..., None]).sum(1, keepdim=True)/valid.sum(1)[:, None, None].clamp_min(1)
        local = (xyz-centroid).clamp(-2, 2)/2
        intensity = processed['voxel_features'][..., 3:4].clamp(0, 1)
        inputs = torch.cat((p[..., None], u[..., None], original[..., None], local, intensity, peer), -1)
        intervention = getattr(self, 'intervention', 'normal')
        if intervention == 'neutral':
            from .counterfactual import neutral_reference
            inputs = neutral_reference(inputs)
        elif intervention == 'permuted':
            from .diagnostic_utils import altered_inputs
            inputs = altered_inputs(inputs, cell, eligible, 'permuted')
        elif intervention != 'normal':
            raise ValueError('Unknown reviewer evidence intervention')
        delta = self.weight_delta(inputs)
        # No positive geometric observation means abstain, NOT evidence of noise.
        changed = (original+delta).clamp(self.floor, 1)
        return torch.where(eligible, changed, original), eligible
