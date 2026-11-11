"""Sparse, per-vehicle feature operations. Coordinates are [vehicle,z,y,x]."""
import math
import torch
from torch import nn


def keys(coords, ny, nx):
    return (coords[:, 0].long() * ny + coords[:, 2]) * nx + coords[:, 3]


def lookup(reference, query):
    ordered, order = reference.sort()
    if not len(ordered):
        return torch.zeros_like(query), torch.zeros_like(query, dtype=torch.bool)
    pos = torch.searchsorted(ordered, query.contiguous()).clamp_max(len(ordered)-1)
    return order[pos], ordered[pos] == query


def point_statistics(batch):
    x = batch['voxel_features'][..., :3]
    n = batch['voxel_num_points']
    valid = torch.arange(x.shape[1], device=x.device)[None] < n[:, None]
    mean = (x * valid[..., None]).sum(1) / n[:, None].clamp_min(1)
    spread = (((x - mean[:, None]).square().sum(-1) * valid).sum(1) /
              n.clamp_min(1)).sqrt()
    return n.to(x.dtype), mean, spread


class RadarSupport(nn.Module):
    def __init__(self, voxel, grid, radius=1.2, neighbors=8, bound=.25):
        super().__init__()
        self.nx, self.ny = grid[:2]
        self.radius, self.neighbors, self.bound = radius, neighbors, bound
        self.query = nn.Linear(64, 16, bias=False)
        self.key = nn.Linear(64, 16, bias=False)
        self.geometry = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
        self.project = nn.Linear(64, 64, bias=False)
        nn.init.zeros_(self.project.weight)
        dx, dy = voxel[:2]
        offsets = [(y, x) for y in range(-math.ceil(radius/dy)-1, math.ceil(radius/dy)+2)
                   for x in range(-math.ceil(radius/dx)-1, math.ceil(radius/dx)+2)
                   if (y or x) and math.hypot(max(0, abs(y)-1)*dy, max(0, abs(x)-1)*dx) <= radius]
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long), persistent=False)

    def neighbors_of(self, coords, centers):
        yx = coords[:, None, 2:].long() + self.offsets
        y, x = yx.unbind(-1)
        query = (coords[:, :1].long()*self.ny + y)*self.nx + x
        index, present = lookup(keys(coords, self.ny, self.nx), query)
        present &= (y >= 0) & (y < self.ny) & (x >= 0) & (x < self.nx)
        delta = centers[index] - centers[:, None]
        distance = delta.norm(dim=-1)
        present &= distance <= self.radius + 1e-6
        ranking = distance.masked_fill(~present, float('inf'))
        order = ranking.topk(min(self.neighbors, len(self.offsets)), largest=False).indices
        return index.gather(1, order), present.gather(1, order)

    def forward(self, features, batch, enabled=True):
        count, center, spread = point_statistics(batch)
        if not len(features):
            return features, dict(count=count, spread=spread, support=count, neighbors=count,
                                  distance=count, residual=count)
        index, mask = self.neighbors_of(batch['voxel_coords'], center)
        delta = center[index] - center[:, None]
        distance = delta.norm(dim=-1)
        neighbors = mask.sum(1)
        average_distance = (distance * mask).sum(1) / neighbors.clamp_min(1)
        strength = (mask * torch.exp(-distance/self.radius)).sum(1) / self.neighbors
        support = .25 * (count/4).clamp(max=1) + .75 * strength
        residual = torch.zeros_like(features)
        if enabled:
            geom = torch.cat((delta/self.radius, (distance/self.radius)[..., None],
                (count[:, None].expand_as(distance)/32)[..., None], (count[index]/32)[..., None],
                spread[:, None, None].expand(*distance.shape, 1), spread[index][..., None]), dim=-1)
            relation = (self.query(features)[:, None] * self.key(features)[index]).sum(-1) / 4
            weight = (relation + self.geometry(geom).squeeze(-1)).masked_fill(~mask, -1e4).softmax(-1) * mask
            pooled = (features[index] * weight[..., None]).sum(1)
            scale = features.detach().square().mean(-1, keepdim=True).add(1e-6).sqrt()
            residual = self.bound * scale * support[:, None] * (neighbors > 0)[:, None] * self.project(pooled-features).tanh()
        return features + residual, dict(count=count, spread=spread, support=support, neighbors=neighbors,
                                         distance=average_distance, residual=residual.detach().norm(dim=-1))


class RadarCompensation(nn.Module):
    def __init__(self, ordinary=False, bound=.25):
        super().__init__()
        self.ordinary, self.bound = ordinary, bound
        # 4,481 + 4,096 parameters for the condition network and projection.
        self.gate = nn.Sequential(nn.Linear(138, 32), nn.SiLU(), nn.Linear(32, 1))
        self.project = nn.Linear(64, 64, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, lidar, radar, state, support):
        # Inputs here are indexed exclusively by original nonempty radar cells.
        # Ordinary control has exactly the same parameter count and bypasses all observation statistics.
        inputs = torch.cat((lidar, radar, radar[:, :10] if self.ordinary else state), -1)
        opportunity = self.gate(inputs).squeeze(-1).sigmoid()
        scale = radar.detach().square().mean(-1, keepdim=True).add(1e-6).sqrt().clamp_max(4.)
        evidence = torch.ones_like(support) if self.ordinary else support
        delta = self.bound * scale * (opportunity * evidence)[:, None] * self.project(radar).tanh()
        return delta, opportunity

