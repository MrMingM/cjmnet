"""Server-only conditional epsilon UNet; detector always remains the official class."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class ConditionalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(32, 128), nn.SiLU(), nn.Linear(128, 256))
        self.condition_tokens = nn.Conv2d(384, 64, 1)
        self.modulate = nn.Linear(64, 256)
        self.e1, self.e2, self.e3 = Block(8, 32), Block(32, 64), Block(64, 128)
        self.d2, self.d1 = Block(128 + 64, 64), Block(64 + 32, 32)
        self.out = nn.Conv2d(32, 8, 1)

    def forward(self, xt, timestep, condition):
        frequencies = torch.exp(torch.arange(16, device=xt.device) * (-math.log(10000) / 15))
        phase = timestep.float()[:, None] * frequencies[None]
        time = self.time(torch.cat((phase.sin(), phase.cos()), dim=1))
        tokens = self.condition_tokens(condition).flatten(2).transpose(1, 2)
        # Explicit conditional affine modulation at the bottleneck. No detach or GT/mask input.
        scale, shift = (time + self.modulate(tokens.mean(dim=1))).chunk(2, dim=1)
        a = self.e1(xt)
        b = self.e2(F.avg_pool2d(a, 2))
        x = self.e3(F.avg_pool2d(b, 2))
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        x = self.d2(torch.cat((F.interpolate(x, size=b.shape[-2:], mode='nearest'), b), dim=1))
        x = self.d1(torch.cat((F.interpolate(x, size=a.shape[-2:], mode='nearest'), a), dim=1))
        return self.out(x)


class FeatureTap:
    """Forward observer only; official state_dict and forward outputs remain unchanged."""
    def __init__(self, detector):
        self.value = None
        self.handle = detector.backbone.register_forward_hook(self.capture)

    def capture(self, module, args, result):
        self.value = result['spatial_features_2d']
        if self.value.shape[1] != 384:
            raise ValueError('Unexpected fused BEV channel count')

    def clear(self):
        self.value = None


def attach_empty_voxel_padding(detector):
    """Zero only explicitly empty sparse padding before scatter; never filter real LiDAR points."""
    def before_scatter(module, args):
        batch = args[0]
        voxels = batch['voxel_features']
        coord = batch['voxel_coords']
        padding = (voxels == 0).flatten(1).all(1) & (coord[:, 1:] == 0).all(1)
        batch['pillar_features'] = batch['pillar_features'].masked_fill(padding[:, None], 0)
    return detector.scatter.register_forward_pre_hook(before_scatter)


def crop_condition(bev, batch_index, centers):
    from .stable_crop import crop_condition as stable_crop
    return stable_crop(bev, batch_index, centers)
