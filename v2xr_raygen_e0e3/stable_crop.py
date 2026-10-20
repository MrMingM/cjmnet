"""Bilinear zero-padded crops using deterministic index_select backward; no custom GPU kernel."""
from .common import identity

POLICY = dict(revision=4, deterministic_algorithms=True, warn_only=False,
              cudnn_benchmark=False, cudnn_deterministic=True,
              crop='bilinear four-neighbour index_select; zeros; align_corners=False',
              scope='Same settings for E0/E1/E2/E3 and inference; no change to v3 resume tolerances',
              limitation='Determinism is validated on the accepted software/device; no cross-platform guarantee')
POLICY_ID = identity(POLICY)


def configure(torch):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    return check_backend(torch)


def check_backend(torch):
    actual = dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic)
    if any(actual[key] != POLICY[key] for key in actual):
        raise RuntimeError('Stable calculation settings changed: ' + repr(actual))
    return actual


def neighbors(grid, width, height, floor):
    """The actual interpolation plan; floor is torch.floor on server, numpy.floor in local checks.

    Coordinate conversion follows PyTorch 2.5.1 GridSampler.h, align_corners=False.
    No geometry/label interpolation: only continuous student feature values are sampled.
    """
    x = ((grid[..., 0] + 1) * width - 1) / 2
    y = ((grid[..., 1] + 1) * height - 1) / 2
    x0, y0 = floor(x), floor(y)
    x1, y1 = x0 + 1, y0 + 1
    return ((x0, y0, (x1 - x) * (y1 - y)),
            (x1, y0, (x - x0) * (y1 - y)),
            (x0, y1, (x1 - x) * (y - y0)),
            (x1, y1, (x - x0) * (y - y0)))


def grid_for(bev, centers):
    import torch
    centers = torch.as_tensor(centers, device=bev.device, dtype=bev.dtype)
    if centers.requires_grad or not torch.isfinite(centers).all():
        raise ValueError('Window centers must be finite fixed geometry')
    xy = (torch.arange(32, device=bev.device, dtype=bev.dtype) + .5) * .4 - 6.4
    yy, xx = torch.meshgrid(xy, xy, indexing='ij')
    x = centers[:, 0, None, None] + xx
    y = centers[:, 1, None, None] + yy
    return torch.stack((2 * (x + 140.8) / 281.6 - 1, 2 * (y + 40) / 80 - 1), dim=-1)


def sample(bev, owners, grid):
    import torch
    check_backend(torch)
    b, channels, height, width = bev.shape
    owners = torch.as_tensor(owners, device=bev.device)
    if (owners.dtype != torch.long or owners.ndim != 1 or grid.ndim != 4 or grid.shape[-1] != 2
            or len(grid) != len(owners) or grid.device != bev.device or grid.dtype != bev.dtype
            or not torch.isfinite(grid).all()):
        raise ValueError('Invalid feature crop registration')
    if len(owners) == 0:
        return bev[:0, :, :grid.shape[1], :grid.shape[2]]
    if owners.min() < 0 or owners.max() >= b:
        raise ValueError('Window owner is outside the batch')
    # Channel-major storage lets one index_select gather all owners without duplicating whole BEVs per window.
    flat = bev.permute(1, 0, 2, 3).reshape(channels, -1)
    output = None
    for x, y, weight in neighbors(grid, width, height, torch.floor):
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        index = owners[:, None, None] * (height * width) + y.clamp(0, height - 1).long() * width + x.clamp(0, width - 1).long()
        value = torch.index_select(flat, 1, index.reshape(-1)).reshape(channels, *grid.shape[:-1])
        term = value * (weight * valid)[None]
        output = term if output is None else output + term
    return output.permute(1, 0, 2, 3).contiguous()


def crop_condition(bev, owners, centers):
    if not len(centers):
        return bev[:0]
    return sample(bev, owners, grid_for(bev, centers))


def reference_crop(bev, owners, centers):
    """Old definition, used only by server equivalence checks with its nondeterministic backward declared."""
    import torch
    from torch.nn import functional as F
    return F.grid_sample(bev[torch.as_tensor(owners, device=bev.device)], grid_for(bev, centers),
                         mode='bilinear', padding_mode='zeros', align_corners=False)
