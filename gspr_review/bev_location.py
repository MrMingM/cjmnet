"""Apply the SAME evidence-conditioned scalar at encoded ego BEV locations.

No neighbor feature is read here. Neighbor semantics enter only through the
existing serialized BEV packets and masked AttFuse path.
"""
import torch
import math
import torch.nn.functional as F


def correct_bev(levels, processed, proposed_delta, supported, fixed_adjustment=None):
    if fixed_adjustment is not None:
        if not math.isfinite(fixed_adjustment) or abs(fixed_adjustment) > .25:
            raise ValueError('Fixed BEV adjustment must be finite and within +/-0.25')
        proposed_delta = torch.full_like(proposed_delta, fixed_adjustment)
    h, w = levels[0].shape[-2]*2, levels[0].shape[-1]*2
    coords = processed['voxel_coords'].long()
    indices = coords[:, 2]*w+coords[:, 3]
    values = (proposed_delta*supported).sum(1)
    counts = supported.sum(1).to(proposed_delta.dtype)
    sums = proposed_delta.new_zeros(h*w).index_add(0, indices, values).reshape(1, 1, h, w)
    nums = proposed_delta.new_zeros(h*w).index_add(0, indices, counts).reshape(1, 1, h, w)
    result, maps = [], []
    for level in levels:
        # Point-count-weighted average within each encoded spatial cell.
        factor = h//level.shape[-2]
        if (h % level.shape[-2] or w % level.shape[-1] or w//level.shape[-1] != factor):
            raise ValueError('BEV level is not aligned to the pillar grid')
        total = F.avg_pool2d(sums, factor)*factor**2
        number = F.avg_pool2d(nums, factor)*factor**2
        adjustment = total/number.clamp_min(1)
        result.append(level*(1+adjustment))
        maps.append(adjustment)
    return result, maps
