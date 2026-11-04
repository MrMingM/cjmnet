"""Pure Torch pillar-local residual aggregation. No PFN replacement or extra BN."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def masked_softmax(logits, mask, dim):
    # All-invalid rows produce zeros, never NaN or an artificial uniform vote.
    weights = logits.masked_fill(~mask, -1e4).softmax(dim) * mask.to(logits.dtype)
    return weights / weights.sum(dim, keepdim=True).clamp_min(1e-6)


class A1(nn.Module):
    def __init__(self, settings, channels=64):
        super().__init__()
        self.cfg = settings
        d = settings['relation_dim']
        self.query = nn.Linear(channels, d, bias=False)
        self.key = nn.Linear(channels, d, bias=False)
        self.geometry = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
        self.residual_norm = nn.LayerNorm(channels)
        self.project = nn.Linear(channels, channels, bias=False)
        self.strength = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.project.weight)
        self.last = {}
        self.force_zero = False

    def chunk(self, features, base, xyz, valid, r, u, ranges):
        m, t, c = features.shape
        # Mask only the NEW computations. The original PFN has already run.
        f = features.masked_fill(~valid[..., None], 0)
        x = xyz.masked_fill(~valid[..., None], 0)
        r = r.masked_fill(~valid, 0)
        u = u.masked_fill(~valid, 0)
        prior = (r * (1-u)).clamp_min(1e-6) * valid
        n = valid.sum(1).to(f.dtype)
        mass = prior.sum(1).clamp_min(1e-6)
        q, key = self.query(f), self.key(f)
        # At most 32 slots; keep the quadratic distance search within small chunks.
        delta = x[:, :, None] - x[:, None, :]
        distance2 = delta.square().sum(-1)
        allowed = valid[:, :, None] & valid[:, None, :]
        allowed = allowed & ~torch.eye(t, dtype=torch.bool, device=f.device)[None]
        k = min(self.cfg['neighbors'], t)
        indices = distance2.masked_fill(~allowed, float('inf')).topk(k, largest=False, dim=-1).indices
        batch = torch.arange(m, device=f.device)[:, None, None]
        neighbor_f, neighbor_k = f[batch, indices], key[batch, indices]
        neighbor_x = x[batch, indices]
        neighbor_r, neighbor_u = r[batch, indices], u[batch, indices]
        neighbor_prior = prior[batch, indices]
        mask = allowed.gather(-1, indices)
        rel = neighbor_x - x[:, :, None]
        d2 = rel.square().sum(-1)
        # Smoothed square root has a finite derivative at coincident points.
        distance = (d2 + 1e-6).sqrt()
        relative = torch.cat((rel / rel.new_tensor([.4, .4, 4.]), distance[..., None],
            r[:, :, None, None].expand(-1, -1, k, -1), neighbor_r[..., None],
            u[:, :, None, None].expand(-1, -1, k, -1), neighbor_u[..., None]), -1)
        learned = (q[:, :, None] * neighbor_k).sum(-1) / math.sqrt(q.shape[-1])
        learned = self.cfg['relation_logit_bound'] * (learned + self.geometry(relative).squeeze(-1)).tanh()

        attention = masked_softmax(learned + neighbor_prior.clamp_min(1e-6).log(), mask, -1)
        messages = (attention[..., None] * neighbor_f).sum(-2)
        # Isolated points retain their own already GSPR-influenced features.
        messages = torch.where(mask.any(-1)[..., None], messages, f)
        weights = masked_softmax(prior.clamp_min(1e-6).log(), valid, 1)
        pooled = (weights[..., None] * messages).sum(1)
        mean_r = r.sum(1) / n.clamp_min(1)
        mean_u = u.sum(1) / n.clamp_min(1)
        variance_r = ((r-mean_r[:, None]).square()*valid).sum(1) / n.clamp_min(1)
        low = x.masked_fill(~valid[..., None], float('inf')).amin(1)
        high = x.masked_fill(~valid[..., None], -float('inf')).amax(1)
        low = torch.where((n > 0)[:, None], low, torch.zeros_like(low))
        high = torch.where((n > 0)[:, None], high, torch.zeros_like(high))
        coverage = ((high-low).square().sum(-1) + 1e-6).sqrt() - .001
        mean_range = (ranges.masked_fill(~valid, 0)*valid).sum(1)/n.clamp_min(1)
        effective = mass.square() / prior.square().sum(1).clamp_min(1e-6)
        inputs = torch.stack((n/32, mean_r, mean_u, variance_r, coverage.clamp_max(4)/4,
                              mean_range/150, effective/32, mass/n.clamp_min(1)), -1)
        strength = self.cfg['residual_bound'] * self.strength(inputs).sigmoid().squeeze(-1)
        strength = strength * (n > 0)
        correction = self.project(self.residual_norm(pooled-base)).tanh()
        amplitude = (base.square().mean(-1) + 1e-6).sqrt()
        residual = strength[:, None] * amplitude[:, None] * correction
        result = base + residual
        if self.force_zero:
            result = base  # Exact identity, including training-mode BN state.
        return result

    def forward(self, features, base, xyz, valid, r, u, ranges):
        if features.shape[0] == 0:
            self.last = {'pillars': 0}
            return base
        outputs = []
        for start in range(0, len(base), self.cfg['pillar_chunk']):
            end = start + self.cfg['pillar_chunk']
            args = tuple(a[start:end] for a in (features, base, xyz, valid, r, u, ranges))
            if self.training and torch.is_grad_enabled():
                result = checkpoint(self.chunk, *args, use_reentrant=False)
            else:
                result = self.chunk(*args)
            outputs.append(result)
        return torch.cat(outputs)
