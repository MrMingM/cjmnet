"""POT-CF adds two bounded cross-vehicle corrections to native AttFusion."""
from itertools import accumulate

import torch
from torch import nn
from torch.nn import functional as F

from transport import Grid, attention, dustbin, unbalanced


MODES = ('module_off', 'matched_only', 'unmatched_only', 'matched_plus_unmatched')


def support_maps(ego, count, shape, stride):
    """Count unique nonempty raw pillars near the actual strided convolution centers."""
    height, width = shape
    maps = []
    for sensor in ('lidar', 'radar'):
        data = ego[f'processed_{sensor}']
        ids, _, y, x = data['voxel_coords'].long().unbind(1)
        y = ((y + stride // 2) // stride).clamp_max(height - 1)
        x = ((x + stride // 2) // stride).clamp_max(width - 1)
        linear = (ids * height + y) * width + x
        occupied = torch.zeros(count * height * width, dtype=torch.long, device=ids.device)
        occupied.scatter_add_(0, linear, (data['voxel_num_points'] > 0).long())
        maps.append(occupied.reshape(count, height * width))
    return torch.stack(maps, dim=-1)


class LocalCorrespondence(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg['projection_channels']
        self.query = nn.Linear(channels, dim, bias=False)
        self.key = nn.Linear(channels, dim, bias=False)

    def forward(self, q, k, ego_support, neighbor_support, grid):
        q = F.normalize(q.float(), dim=-1, eps=1e-6)
        k = F.normalize(k.float(), dim=-1, eps=1e-6)
        distance = (grid.dx.square() + grid.dy.square()).float()
        costs = []
        for start in range(0, len(grid.neighbor), self.cfg['spatial_chunk']):
            end = start + self.cfg['spatial_chunk']
            index = grid.neighbor[start:end]
            similarity = (q[:, start:end, None] * k[:, index]).sum(-1).clamp(-1., 1.)
            cost = 1. - similarity + self.cfg['distance_weight'] * distance + self.cfg['cost_bias']
            if self.cfg['modality_mismatch_cost']:
                shared_sensor = (ego_support[:, start:end, None] & neighbor_support[:, index]).any(-1)
                cost = cost + self.cfg['modality_mismatch_cost'] * (~shared_sensor)
            costs.append(cost)
        return torch.cat(costs, dim=1)


class CollaborativeFusion(nn.Module):
    def __init__(self, native, channels, cfg, solver, mode):
        super().__init__()
        if mode not in MODES or solver not in ('attention', 'unbalanced', 'dustbin'):
            raise ValueError('Unknown fusion mode or solver')
        self.native, self.cfg, self.solver, self.mode = native, dict(cfg), solver, mode
        self.sources = None
        self.observe = self.profile = False
        self.report = None
        self.elapsed = None
        self._grid = None
        dim = cfg['projection_channels']
        self.correspondence = LocalCorrespondence(channels, self.cfg)
        self.value = nn.Linear(channels, dim, bias=False)
        self.matched = nn.Linear(dim, channels, bias=False)
        nn.init.zeros_(self.matched.weight)
        self.unmatched = self.gate = None
        if mode in ('unmatched_only', 'matched_plus_unmatched'):
            self.unmatched = nn.Linear(dim, channels, bias=False)
            self.gate = nn.Linear(dim + 2, 1)
            nn.init.zeros_(self.unmatched.weight)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.bin_score = nn.Parameter(torch.tensor(float(cfg['dustbin_score']))) if solver == 'dustbin' else None

    def grid(self, height, width, device):
        key = (height, width, self.cfg['window'], device)
        if self._grid is None or self._grid[0] != key:
            self._grid = key, Grid(*key)
        return self._grid[1]

    def pair(self, q, k, value, ego, neighbor, grid):
        cfg = self.cfg
        a, b = ego.any(-1).float(), neighbor.any(-1).float()
        cost = self.correspondence(q, k, ego > 0, neighbor > 0, grid)
        if self.solver == 'attention':
            plan = attention(cost, a, b, grid, cfg['attention_temperature'])
        elif self.solver == 'unbalanced':
            plan = unbalanced(cost, a, b, grid, cfg['epsilon'], cfg['rho_ego'],
                              cfg['rho_neighbor'], cfg['iterations'])
        else:
            plan = dustbin(cost, a, b, grid, cfg['epsilon'], self.bin_score, cfg['dustbin_iterations'])

        # Pairwise mass is retained in the feature amplitude even after row averaging.
        matched = []
        for start in range(0, len(grid.neighbor), cfg['spatial_chunk']):
            end = start + cfg['spatial_chunk']
            indices = grid.neighbor[start:end]
            weights = plan.weights[:, start:end] / plan.row[:, start:end, None].clamp_min(1.)
            matched.append((weights[..., None] * value[:, indices]).sum(-2))
        matched = torch.cat(matched, dim=1)

        candidate = neighbor.sum(-1) >= cfg['unmatched_min_pillars']
        near_ego = (a[:, grid.neighbor].bool() & grid.valid).any(-1)
        unmatched_weight = plan.col_remaining * candidate
        unmatched_weight = unmatched_weight * torch.where(near_ego, cfg['unmatched_ego_support_scale'], 1.)
        if self.gate is not None:
            gate = self.gate(torch.cat((value, (neighbor.float().log1p() / 4.).clamp_max(1.)), -1))
            unmatched_weight = unmatched_weight * gate.squeeze(-1).sigmoid()
        unmatched = value * unmatched_weight[..., None]

        with torch.no_grad():
            ratio = plan.row.sum(-1) / a.sum(-1).minimum(b.sum(-1)).clamp_min(1.)
            finite = (torch.isfinite(cost).all() & torch.isfinite(plan.weights).all()
                      & torch.isfinite(plan.residual) & torch.isfinite(unmatched).all())
            health = torch.stack((plan.residual, ratio.max(), finite.float()))
            observed = None
            if self.observe:
                peak = plan.weights.detach().amax(-1) / plan.row.detach().clamp_min(1e-12)
                effective_edges = plan.row.detach().square() / plan.weights.detach().square().sum(-1).clamp_min(1e-24)
                observed = dict(ego=ego.detach(), neighbor=neighbor.detach(), row=plan.row.detach(),
                    col=plan.col.detach(), row_remaining=plan.row_remaining.detach(),
                    col_remaining=plan.col_remaining.detach(), peak_fraction=peak,
                    effective_edges=effective_edges, unmatched_weight=unmatched_weight.detach(),
                    candidate=candidate.detach(), near_ego=near_ego.detach(),
                    allowed_edges=(grid.valid & a[..., None].bool() & b[:, grid.neighbor].bool()).sum(-1))
        return matched, unmatched, health, observed

    def forward(self, x, record_len):
        base = self.native(x, record_len)
        self.report, self.elapsed = None, None
        if self.mode == 'module_off':
            return base
        lengths = record_len.tolist()
        if min(lengths) < 1 or sum(lengths) != len(x):
            raise ValueError('Invalid CAV grouping')
        if max(lengths) == 1:
            return base
        if self.sources is None or self.sources.shape != (len(x), x.shape[-2] * x.shape[-1], 2):
            raise ValueError('Missing or misaligned voxel support')
        start_event = end_event = None
        if self.profile:
            start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start_event.record()
        out = self.correction(x, base, lengths)
        if self.profile:
            end_event.record()
            self.elapsed = start_event, end_event
        return out

    def correction(self, x, base, lengths):
        _, channels, height, width = x.shape
        starts = list(accumulate(lengths, initial=0))
        ego_ids = [start for start, end in zip(starts, starts[1:]) for _ in range(start + 1, end)]
        neighbor_ids = [i for start, end in zip(starts, starts[1:]) for i in range(start + 1, end)]
        grid = self.grid(height, width, x.device)
        # Solve in FP32 even if the rest of the detector uses automatic mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            features = x.float().flatten(2).transpose(1, 2)
            q = self.correspondence.query(features[ego_ids])
            k = self.correspondence.key(features[neighbor_ids])
            value = self.value(features[neighbor_ids])
            e, n = self.sources[ego_ids], self.sources[neighbor_ids]
            matched, unmatched, health, observations = [], [], [], []
            for start in range(0, len(q), self.cfg['pair_chunk']):
                sl = slice(start, start + self.cfg['pair_chunk'])
                m, u, status, observed = self.pair(q[sl], k[sl], value[sl], e[sl], n[sl], grid)
                matched.append(m)
                unmatched.append(u)
                health.append(status)
                if self.observe:
                    observations.append(observed)
            health = torch.stack(health)
            residual = health[:, 0].amax()
            ratio = health[:, 1].amax()
            tolerance = self.cfg['dustbin_residual_limit'] if self.solver == 'dustbin' else self.cfg['residual_limit']
            # Softmax can assign the same neighbor to many rows by design.
            mass_ok = (ratio <= self.cfg['max_mass_ratio']) | (self.solver == 'attention')
            if not bool(health[:, 2].bool().all() & (residual <= tolerance) & mass_ok):
                raise FloatingPointError(f'{self.solver} failed: residual/mass_ratio/finite={health.detach().tolist()}')
            m, u = torch.cat(matched), torch.cat(unmatched)
            sizes = [count - 1 for count in lengths]

            def reduce_pairs(features):
                return torch.stack([part.sum(0) / max(count, 1)
                                    for part, count in zip(features.split(sizes), sizes)])

            delta_m = delta_u = 0.
            if self.mode in ('matched_only', 'matched_plus_unmatched'):
                delta_m = self.cfg['matched_limit'] * self.matched(reduce_pairs(m)).tanh()
            if self.mode in ('unmatched_only', 'matched_plus_unmatched'):
                if self.unmatched is None:
                    raise ValueError('Unmatched path requires weights from a model trained with that path')
                delta_u = self.cfg['unmatched_limit'] * self.unmatched(reduce_pairs(u)).tanh()
            if self.observe:
                self.report = dict(**{name: torch.cat([row[name] for row in observations])
                                      for name in observations[0]},
                    matched_rms=delta_m.detach().square().mean(-1).sqrt() if torch.is_tensor(delta_m) else None,
                    unmatched_rms=delta_u.detach().square().mean(-1).sqrt() if torch.is_tensor(delta_u) else None,
                    residual=residual.detach(), mass_ratio_max=ratio.detach(),
                    iterations={'attention': 0, 'unbalanced': self.cfg['iterations'],
                                'dustbin': self.cfg['dustbin_iterations']}[self.solver])
            delta = (delta_m + delta_u).transpose(1, 2).reshape(-1, channels, height, width)
        return base + delta.to(base.dtype)
