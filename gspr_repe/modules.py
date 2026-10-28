"""Small pure-Torch modules. No new native operators."""
import torch
from torch import nn
from torch.nn import functional as F


def statistics(points, valid, reliability, uncertainty):
    """Per pillar: count, effective count, quality, uncertainty, coverage, spread, range."""
    xyz = points[..., :3].masked_fill(~valid[..., None], 0)
    r = reliability.masked_fill(~valid, 0)
    n = valid.sum(1).clamp_min(1)
    mass = r.sum(1).clamp_min(1e-6)
    center = (xyz*r[..., None]).sum(1)/mass[:, None]
    spread = ((xyz-center[:, None]).square().sum(-1)*r).sum(1)/mass
    trusted = valid & (reliability >= .5) & (uncertainty < .75)
    low = xyz.masked_fill(~trusted[..., None], float('inf')).amin(1)
    high = xyz.masked_fill(~trusted[..., None], -float('inf')).amax(1)
    has_trusted = trusted.any(1)[:, None]
    # Remove empty-reduction sentinels before subtraction/norm, including backward.
    low = torch.where(has_trusted, low, torch.zeros_like(low))
    high = torch.where(has_trusted, high, torch.zeros_like(high))
    extent = (high-low).clamp_min(0).norm(dim=-1)
    # sqrt(0) has an infinite derivative. A zero upstream gradient can still produce
    # NaN and poison GSPR through reliability, especially for singleton pillars.
    # A 1 mm smoothing scale preserves a zero value with a bounded derivative.
    spread_rms = (spread.clamp_min(0) + 1e-6).sqrt() - 1e-3
    return torch.stack((n.float(), r.sum(1), r.sum(1)/n,
        uncertainty.masked_fill(~valid, 0).sum(1)/n, extent,
        spread_rms, center[:, :2].norm(dim=-1)), 1)


class Aggregation(nn.Module):
    def __init__(self, decorated, channels):
        super().__init__()
        self.value = nn.Linear(decorated, channels)
        self.score = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 1))
        self.gate = nn.Parameter(torch.tensor(-5.))

    def forward(self, features, original, points, valid, r, u):
        xyz = points[..., :3].masked_fill(~valid[..., None], 0)
        center = xyz.sum(1)/valid.sum(1).clamp_min(1)[:, None]
        relative = xyz-center[:, None]
        geometry = torch.cat((relative, relative.norm(dim=-1, keepdim=True), r[..., None], u[..., None]), -1)
        # Reliability appears once as a prior in normalized weights. Value uses unweighted features.
        logits = self.score(geometry).squeeze(-1) + r.clamp_min(1e-6).log()
        logits = logits.masked_fill(~valid, -1e4)
        weights = logits.softmax(1)*valid
        weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-6)
        pooled = (self.value(features.masked_fill(~valid[..., None], 0))*weights[..., None]).sum(1)
        return original + self.gate.sigmoid()*(pooled-original)


class Observation(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.feature = nn.Conv2d(channels, 8, 1)
        self.context = nn.Sequential(nn.Conv2d(15, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 4, 1))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self, x, stats):
        encoded = torch.cat((torch.log1p(stats[:, :2]), stats[:, 2:4],
            torch.log1p(stats[:, 4:6]), stats[:, 6:7]/150), 1)
        q, s, evidence, unknown = self.context(torch.cat((self.feature(x), encoded), 1)).sigmoid().chunk(4, 1)
        occupied = stats[:, :1] > 0
        # Unobserved cells remain unknown and cannot become foreground by density alone.
        q = q*occupied
        unknown = torch.where(occupied, unknown, torch.ones_like(unknown))
        candidate = (1-s)*evidence*(1-unknown)*occupied
        return dict(q=q, s=s, evidence=evidence, unknown=unknown, candidate=candidate)


class Enhancement(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.key = nn.Conv2d(channels, 8, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -5)
        self.last = {}

    def forward(self, x, obs, occupied, constrained=True, state_gate=False):
        key = self.key(x)
        value = self.value(x)
        numerator, denominator = torch.zeros_like(x), torch.zeros_like(x[:, :1])
        quality = obs['q']*obs['s']*(1-obs['unknown']) if constrained else occupied.float()
        # Slice padded arrays; no wraparound and no cross-CAV propagation. Sequential shifts bound memory.
        kp, vp, qp = F.pad(key, (1,1,1,1)), F.pad(value, (1,1,1,1)), F.pad(quality, (1,1,1,1))
        h, w = x.shape[-2:]
        for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            neighbor = kp[:, :, 1+dy:1+dy+h, 1+dx:1+dx+w]
            similarity = F.cosine_similarity(key, neighbor, dim=1, eps=1e-6)[:, None]
            weight = similarity.exp()*qp[:, :, 1+dy:1+dy+h, 1+dx:1+dx+w]/(1+dx*dx+dy*dy)
            numerator = numerator + weight*vp[:, :, 1+dy:1+dy+h, 1+dx:1+dx+w]
            denominator = denominator + weight
        strength = self.gate(x).sigmoid()
        if constrained or state_gate:
            strength = strength*obs['candidate']
        strength = strength*(denominator > 1e-6)
        self.last = dict(strength=strength.detach(), source_mass=denominator.detach())
        return x + strength*numerator/denominator.clamp_min(1e-6)
