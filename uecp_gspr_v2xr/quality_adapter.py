"""Frozen, supervised OPV2V GSPR; slot-weighted regional statistics, never point deletion."""
from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import sha, ROOT, HERE, read


def load_quality(config):
    # Native bootstrap removes the project root from sys.path. Load this reviewed,
    # self-contained file explicitly, without restoring a competing OpenCOOD path.
    name = 'uecp_gspr_v2xr.gspr_vendor'
    if name not in sys.modules:
        source = ROOT / 'attfuse_gspr/reliability.py'
        if sha(source) != read(HERE / 'PINS.json')['files']['attfuse_gspr/reliability.py']:
            raise ValueError('GSPR implementation source changed')
        spec = importlib.util.spec_from_file_location(name, source)
        imported = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported); sys.modules[name] = imported
    GeometryFirstPillarReliability = sys.modules[name].GeometryFirstPillarReliability
    path = Path(config['gspr']['checkpoint'])
    if sha(path) != config['gspr']['checkpoint_sha256']:
        raise ValueError('Frozen supervised quality checkpoint hash differs')
    model = GeometryFirstPillarReliability(config['gspr']['model'], config['voxel'], config['range'])
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = state.get('state_dict', state)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    selected = {k[5:]: v for k, v in state.items() if k.startswith('gspr.')}
    expected = model.state_dict()
    if selected.keys() != expected.keys() or any(selected[k].shape != v.shape or selected[k].dtype != v.dtype
                                               or not torch.isfinite(selected[k]).all() for k, v in expected.items()):
        raise ValueError('Frozen GSPR keys/shape/dtype/finite tensors differ')
    model.load_state_dict(selected, strict=True)
    model.requires_grad_(False).eval()
    return model


def aggregate(points, processed, n_cavs, empty, height=200, width=704, low=0.5):
    """Point sums first, spatial sums second: every valid point has equal weight."""
    coords = processed['voxel_coords'].long()
    valid = points['point_valid_mask'] & ~empty[:, None]
    v = valid.float()
    r, u = points['point_reliability'], points['point_uncertainty']
    e = points['point_evidence'].sum(-1)
    pillar = torch.stack((v.sum(1), (r*v).sum(1), ((r < low)*v).sum(1),
                          (u*v).sum(1), (e*v).sum(1)), dim=1)
    index = coords[:, 0]*height*width + coords[:, 2]*width + coords[:, 3]
    grid = pillar.new_zeros((n_cavs*height*width, 5))
    grid.index_add_(0, index, pillar)
    grid = grid.view(n_cavs, height, width, 5).permute(0, 3, 1, 2)
    return grid


def regional(grid, size):
    sums = F.adaptive_avg_pool2d(grid, size)
    count = sums[:, :1]
    valid = count > 0
    quality = sums[:, 1:] / count.clamp_min(1e-12)
    quality = torch.where(valid, quality, torch.zeros_like(quality))
    return quality, valid


def shuffled(quality, valid, counts, seeds):
    """Permute complete Q vectors within observed count bins; preserve masks and C."""
    out = quality.clone()
    bins = torch.floor(torch.log2(counts.clamp_min(1))).long()
    for cav in range(len(quality)):
        rng = np.random.RandomState(int(seeds[cav]))
        flat = out[cav].flatten(1)
        source = quality[cav].flatten(1)
        for b in bins[cav][valid[cav]].unique().tolist():
            ids = ((bins[cav, 0] == b) & valid[cav, 0]).flatten().nonzero().flatten()
            permutation = torch.as_tensor(rng.permutation(len(ids)), device=ids.device)
            flat[:, ids] = source[:, ids[permutation]]
    return out


class QualityAdapter(nn.Module):
    """Zero last layer gives an identity logit correction on observed cells."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(5, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base, coverage, quality, valid):
        # Evidence magnitude is reported separately; use independent uncertainty here.
        x = torch.cat((coverage, quality[:, :3], valid.float()), 1)
        delta = 2 * torch.tanh(self.net(x))
        adjusted = torch.sigmoid(torch.logit(base.clamp(1e-5, 1-1e-5)) + delta)
        return adjusted


def describe(x, mask=None):
    value = x.detach() if mask is None else x.detach()[mask.expand_as(x)]
    if not value.numel():
        return dict(n=0, mean=None, std=None, min=None, max=None)
    return dict(n=value.numel(), mean=float(value.mean()), std=float(value.std(unbiased=False)),
                min=float(value.min()), max=float(value.max()))
