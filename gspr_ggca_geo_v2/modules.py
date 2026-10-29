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


class GGCA(nn.Module):
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

    def chunk(self, features, base, xyz, valid, r, u, ranges, level):
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

        # Local spacing and feature agreement, not a single global surface/center.
        spacing = (distance * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        neighbor_spacing = spacing[batch, indices]
        scale = (spacing[:, :, None] + neighbor_spacing).clamp_min(.02)
        compatibility = torch.exp(-d2 / scale.square())
        compatibility = compatibility * torch.exp(-(spacing[:, :, None]-neighbor_spacing).abs()/scale)
        similarity = F.cosine_similarity(f[:, :, None], neighbor_f, dim=-1, eps=1e-6).clamp(0, 1)
        support = (compatibility * (.25 + .75*similarity) * mask).detach()
        denominator = support.sum(-1)
        measured = (support * neighbor_prior.detach()).sum(-1) / denominator.clamp_min(1e-6)
        # No supporting neighbor means unknown; retain initial evidence neutrally.
        consensus = torch.where(denominator > 1e-6, measured, prior.detach())
        conflict = consensus - prior.detach()
        adjustment = (1 + self.cfg['geometric_adjustment_bound']*conflict).clamp(.5, 1.5)
        adjusted = prior * adjustment if level >= 2 else prior
        neighbor_adjusted = adjusted[batch, indices]
        attention = masked_softmax(learned + neighbor_adjusted.clamp_min(1e-6).log(), mask, -1)
        messages = (attention[..., None] * neighbor_f).sum(-2)
        # Isolated points retain their own already GSPR-influenced features.
        messages = torch.where(mask.any(-1)[..., None], messages, f)
        weights = masked_softmax(adjusted.clamp_min(1e-6).log(), valid, 1)
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
        mean_consensus = (consensus*valid).sum(1)/n.clamp_min(1)
        effective = mass.square() / prior.square().sum(1).clamp_min(1e-6)
        inputs = torch.stack((n/32, mean_r, mean_u, variance_r, coverage.clamp_max(4)/4,
                              mean_range/150, effective/32, mass/n.clamp_min(1)), -1)
        strength = self.cfg['residual_bound'] * self.strength(inputs).sigmoid().squeeze(-1)
        sufficiency = ((effective-2)/4).clamp(0, 1) * (1-mean_u).clamp(0, 1)
        # Far-range spacing is expected to grow; a short span is not proof of noise.
        span_evidence = coverage / (coverage + .05 + .002*mean_range)
        sufficient = sufficiency * span_evidence * mean_consensus.clamp(0, 1)
        sufficient = torch.where(n <= 2, torch.zeros_like(sufficient), sufficient)
        if level == 3:
            strength = strength * sufficient
        strength = strength * (n > 0)
        correction = self.project(self.residual_norm(pooled-base)).tanh()
        amplitude = (base.detach().square().mean(-1) + 1e-6).sqrt()
        residual = strength[:, None] * amplitude[:, None] * correction
        result = base + residual
        if self.force_zero:
            result = base  # Exact identity, including training-mode BN state.
        # One detached row per pillar. Histograms are reduced once outside chunks.
        protected = (level == 3) & (sufficient < .25) & (n > 0)
        diag = torch.stack((n, mean_r, mean_u, mean_consensus, strength, sufficient,
            protected.to(f.dtype), (residual.detach().square().mean(-1)+1e-12).sqrt(),
            ((prior >= .5) & (consensus >= .5) & valid).sum(1),
            ((prior >= .5) & (consensus < .5) & valid).sum(1),
            ((prior < .5) & (consensus >= .5) & valid).sum(1),
            ((prior < .5) & (consensus < .5) & valid).sum(1)), -1).detach()
        return result, diag

    def forward(self, features, base, xyz, valid, r, u, ranges, level):
        if features.shape[0] == 0:
            self.last = {'pillars': 0}
            return base
        outputs, diagnostics = [], []
        for start in range(0, len(base), self.cfg['pillar_chunk']):
            end = start + self.cfg['pillar_chunk']
            args = tuple(a[start:end] for a in (features, base, xyz, valid, r, u, ranges))
            def calculate(*values):
                return self.chunk(*values, level)
            if self.training and torch.is_grad_enabled() and self.cfg['checkpoint_chunks']:
                result, diag = checkpoint(calculate, *args, use_reentrant=False)
            else:
                result, diag = calculate(*args)
            outputs.append(result); diagnostics.append(diag)
        d = torch.cat(diagnostics)
        self.pillar_rows = d  # Detached, for evaluator-only target analysis.
        bins = d[:, 0].long().clamp(0, 32)
        counts = torch.bincount(bins, minlength=33)
        strengths = d.new_zeros(33).scatter_add_(0, bins, d[:, 4])
        self.last = dict(pillars=len(d), points=int(d[:, 0].sum()), mean_r=float(d[:, 1].mean()),
            mean_u=float(d[:, 2].mean()), mean_consensus=float(d[:, 3].mean()),
            mean_strength=float(d[:, 4].mean()), mean_sufficiency=float(d[:, 5].mean()),
            protected_pillars=int(d[:, 6].sum()), protected_points=int((d[:, 0]*d[:, 6]).sum()),
            residual_rms=float(d[:, 7].mean()), quadrants=d[:, 8:12].sum(0).tolist(),
            point_count_histogram=counts.tolist(), strength_sum_by_count=strengths.tolist(),
            reliability_histogram=torch.histc(r.detach()[valid], bins=20, min=0, max=1).tolist(),
            consensus_histogram=torch.histc(d[:, 3], bins=20, min=0, max=1).tolist())
        return torch.cat(outputs)


# Only the historical B1 diagnostic imports these names; never instantiated by GGCA.
from bridge import load
_old = load('old_aggregation_modules', 'gspr_repe/modules.py')
Aggregation, Observation, Enhancement, statistics = _old.Aggregation, _old.Observation, _old.Enhancement, _old.statistics


# Keep the entire revision-1 implementation above unchanged for A1. Its parameter
# construction, RNG consumption, calculations and diagnostics are inherited.
_OriginalA1 = GGCA


class GGCA(_OriginalA1):
    def chunk(self, features, base, xyz, valid, r, u, ranges, level):
        if level == 1:
            return super().chunk(features, base, xyz, valid, r, u, ranges, level)
        from geometry import chunk
        return chunk(self, features, base, xyz, valid, r, u, ranges, level)

    def forward(self, features, base, xyz, valid, r, u, ranges, level):
        result = super().forward(features, base, xyz, valid, r, u, ranges, level)
        if level >= 2 and len(base):
            self.last.update(geometry_revision=2, consensus_definition='absolute mean pair compatibility',
                mean_geometric_support=float(self.pillar_rows[:, 3].mean()),
                mean_neighbor_trust=float(self.pillar_rows[:, 12].mean()),
                mean_exchange_mass=float(self.pillar_rows[:, 13].mean()),
                mean_self_retention=1-float(self.pillar_rows[:, 13].mean()),
                mean_absolute_prior_mass=float(self.pillar_rows[:, 14].mean()),
                mean_effective_points=float(self.pillar_rows[:, 15].mean()),
                sparse_protection_revision=2,
                pair_trust_definition='conditional mean neighbor prior; unknown when no support')
        return result
