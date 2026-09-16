"""Support of RETAINED voxel slots, not raw density or free-space evidence."""
import torch


def block_statistics(reliability, processed, agents, grid, block_voxels=8):
    h, w = grid
    valid = reliability["point_valid_mask"]
    evidence = reliability["point_evidence"]
    alpha = evidence + 1
    probability = alpha[..., 0] / alpha.sum(-1)
    count = valid.sum(1).to(probability.dtype)
    support = (probability * valid).sum(1)
    uncertainty = (reliability["point_uncertainty"] * valid).sum(1)
    coords = processed["voxel_coords"].long()
    if coords.numel() and (coords[:, 0].min() < 0 or coords[:, 0].max() >= agents):
        raise ValueError("voxel CAV index outside record_len")
    by, bx = coords[:, 2] // block_voxels, coords[:, 3] // block_voxels
    if coords.numel() and ((by < 0).any() or (by >= h).any() or (bx < 0).any() or (bx >= w).any()):
        raise ValueError("voxel coordinate outside communication grid")
    indices = coords[:, 0]*h*w + by*w + bx
    sums = probability.new_zeros((agents*h*w, 4))
    sums.index_add_(0, indices, torch.stack([count, support, uncertainty, (count > 0).to(count.dtype)], -1))
    n, s, u, occupied = sums.unbind(-1)
    known = n > 0
    channels = torch.stack([
        s / n.clamp_min(1), u / n.clamp_min(1),
        torch.log1p(s), torch.log1p(n), occupied / (block_voxels**2),
        known.to(n.dtype),
    ], -1)
    return channels.reshape(agents, h, w, 6).permute(0, 3, 1, 2).contiguous()
