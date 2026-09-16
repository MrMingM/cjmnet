"""Dense task-risk prediction and equal-size matched/concatenation responders."""
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import OBS
from .protocol import FIELDS


class RiskHead(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(64+OBS, hidden, 1), nn.GroupNorm(4, hidden), nn.SiLU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden, 4, 1))

    def forward(self, semantics, obs):
        return self.net(torch.cat([semantics, obs], 1)).sigmoid()


class GainHead(nn.Module):
    def __init__(self, hidden=64, matching=True):
        super().__init__()
        self.matching = matching
        self.net = nn.Sequential(nn.Linear(64+OBS+FIELDS+5+12, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, request, own, semantics, pose):
        # request is Qx17 from bytes, NEVER the ego BEV or GT. Own inputs are QxC.
        if self.matching:
            extra = torch.cat([own[:, 4:8]*(1-request[:, 4:8]),
                               own[:, 8:12]*(1-request[:, 8:12]),
                               own[:, :4]-request[:, :4]], -1)
        else:
            extra = torch.cat([own[:, 4:8], request[:, 4:8], request[:, :4]], -1)
        return self.net(torch.cat([request, own, semantics, pose.expand(len(request), -1), extra], -1)).squeeze(-1)


def pair_ranking_loss(prediction, target, tolerance=1e-4):
    # Loss-prediction idea adapted to ALL distinct candidate pairs; ties contribute zero.
    i, j = torch.triu_indices(len(target), len(target), 1, device=target.device)
    difference = target[i] - target[j]
    keep = difference.abs() > tolerance
    if not keep.any():
        return prediction.sum()*0
    return F.softplus(-difference[keep].sign()*(prediction[i[keep]]-prediction[j[keep]])).mean()


def risk_loss(prediction, target, foreground):
    # Foreground/background are independently normalized; masks are training-only GT.
    error = (prediction-target).square()
    fg = foreground.float()
    total = prediction.sum()*0
    for mask in (fg, 1-fg):
        total = total + (error*mask).sum()/(mask.sum().clamp_min(1)*4)
    return total
