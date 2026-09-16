"""Point reliability estimation aligned with OpenCOOD PointPillar voxels.

The module deliberately consumes the *unencoded* points stored in
``voxel_features``.  It therefore runs before PillarVFE/BEV construction while
keeping the spconv point slots and coordinates unchanged.  This is the safe,
differentiable replacement for deleting points or moving them to (9999, ...).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PillarHaarMixer(nn.Module):
    """A light, XY-pillar-aligned substitute for TripleMixer's three planes."""

    def __init__(self, channels, grid_size, downsample=4):
        super().__init__()
        self.nx = int(grid_size[0])
        self.ny = int(grid_size[1])
        self.downsample = int(downsample)
        if self.downsample < 1:
            raise ValueError("spectral downsample must be >= 1")
        self.low_filter = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False)
        self.high_filter = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, pillar_features, coords):
        if pillar_features.numel() == 0:
            return pillar_features
        batch_size = int(coords[:, 0].max().item()) + 1
        height = (self.ny + self.downsample - 1) // self.downsample
        width = (self.nx + self.downsample - 1) // self.downsample
        channels = pillar_features.shape[-1]
        grid = pillar_features.new_zeros((batch_size, channels, height, width))
        count = pillar_features.new_zeros((batch_size, 1, height, width))

        by = torch.div(coords[:, 2].long(), self.downsample,
                       rounding_mode="floor").clamp(0, height - 1)
        bx = torch.div(coords[:, 3].long(), self.downsample,
                       rounding_mode="floor").clamp(0, width - 1)
        flat_index = coords[:, 0].long() * (height * width) + by * width + bx
        flat_grid = grid.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_count = count.permute(0, 2, 3, 1).reshape(-1, 1)
        flat_grid.index_add_(0, flat_index, pillar_features)
        flat_count.index_add_(0, flat_index,
                              torch.ones_like(flat_index, dtype=grid.dtype)[:, None])
        grid = (flat_grid / flat_count.clamp_min(1.0)).view(
            batch_size, height, width, channels).permute(0, 3, 1, 2)

        # One-level Haar-like decomposition: the low-frequency field and its
        # aligned high-frequency residual.  No FFT-sized dense OPV2V plane is
        # constructed, which keeps memory bounded.
        if height >= 2 and width >= 2:
            low_small = F.avg_pool2d(grid, kernel_size=2, stride=2)
            low = F.interpolate(low_small, size=(height, width), mode="nearest")
        else:
            low = grid
        high = grid - low
        mixed = self.mix(torch.cat((self.low_filter(low),
                                    self.high_filter(high)), dim=1))
        return mixed[coords[:, 0].long(), :, by, bx]


class GeometryFirstPillarReliability(nn.Module):
    """Predict calibrated point reliability and evidential uncertainty.

    Geometry and radiometry are encoded independently.  Per-agent sensor
    statistics modulate only the radiometric branch.  The branches meet as
    non-negative evidence for the two hypotheses ``reliable`` and ``noise``.
    """

    def __init__(self, cfg, voxel_size, point_cloud_range):
        super().__init__()
        hidden = int(cfg.get("hidden_dim", 32))
        context_dim = int(cfg.get("context_dim", 16))
        self.max_points = int(cfg.get("max_points_per_voxel", 32))
        self.intensity_index = int(cfg.get("intensity_index", 3))
        self.ring_index = cfg.get("ring_index", None)
        self.evidence_cap = float(cfg.get("evidence_cap", 30.0))
        self.return_branch_support = False
        self.reliability_floor = float(cfg.get("reliability_floor", 0.05))
        self.prior_reliability = float(cfg.get("initial_reliability", 0.95))
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.pc_range = tuple(float(v) for v in point_cloud_range)
        max_x = max(abs(self.pc_range[0]), abs(self.pc_range[3]))
        max_y = max(abs(self.pc_range[1]), abs(self.pc_range[4]))
        max_z = max(abs(self.pc_range[2]), abs(self.pc_range[5]))
        self.xyz_scale = (max(max_x, 1.0), max(max_y, 1.0), max(max_z, 1.0))
        self.max_range = math.sqrt(max_x ** 2 + max_y ** 2 + max_z ** 2)

        # xyz, range, elevation/ring, density, xyz cluster offset, xy center offset
        self.geometry_encoder = _MLP(11, hidden, hidden)
        self.geometry_to_context = nn.Linear(hidden, context_dim)
        grid_size = cfg.get("grid_size")
        if grid_size is None:
            grid_size = [
                round((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]),
                round((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]),
                round((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2]),
            ]
        self.spectral = PillarHaarMixer(
            context_dim, grid_size, int(cfg.get("spectral_downsample", 4)))
        self.geometry_evidence = _MLP(hidden + context_dim, hidden, 2)

        # robust intensity, raw bounded intensity, ring/elevation, density
        self.radiometric_encoder = _MLP(4, hidden, hidden)
        # median intensity, MAD, mean range, range std, mean density
        self.sensor_adapter = nn.Sequential(
            nn.Linear(5, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden * 2))
        self.radiometric_evidence = _MLP(hidden, hidden, 2)
        self.physics_evidence = _MLP(3, hidden // 2 or 1, 2)

        # Start as a nearly transparent module when bootstrapping AttFuse.
        # Evidence heads are zeroed so the requested initial reliability is
        # controlled by the Dirichlet prior rather than random initialization.
        for head in (self.geometry_evidence, self.radiometric_evidence,
                     self.physics_evidence):
            nn.init.zeros_(head.net[-1].weight)
            nn.init.zeros_(head.net[-1].bias)
        raw_prior = (self.prior_reliability - self.reliability_floor) / \
            max(1.0 - self.reliability_floor, 1e-6)
        raw_prior = min(max(raw_prior, 0.51), 0.98)
        reliable_evidence = raw_prior / (1.0 - raw_prior) - 1.0
        reliable_bias = math.log(math.expm1(max(reliable_evidence, 1e-3)))
        self.fusion_bias = nn.Parameter(
            torch.tensor([reliable_bias, -8.0], dtype=torch.float32))

    @staticmethod
    def _valid_mask(num_points, point_count):
        slots = torch.arange(point_count, device=num_points.device)[None, :]
        return slots < num_points.long()[:, None]

    @staticmethod
    def _masked_mean(value, mask, dim=1, keepdim=True):
        weight = mask.to(value.dtype)
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        return (value * weight).sum(dim=dim, keepdim=keepdim) / \
            weight.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)

    def _agent_statistics(self, intensity, ranges, density, valid, coords):
        agent_count = int(coords[:, 0].max().item()) + 1
        stats = intensity.new_zeros((agent_count, 5))
        for agent in range(agent_count):
            pillar_mask = coords[:, 0].long() == agent
            point_mask = valid[pillar_mask]
            agent_i = intensity[pillar_mask][point_mask]
            agent_r = ranges[pillar_mask][point_mask]
            if agent_i.numel() == 0:
                continue
            median = agent_i.median()
            mad = (agent_i - median).abs().median().clamp_min(1e-3)
            stats[agent, 0] = median
            stats[agent, 1] = mad
            stats[agent, 2] = agent_r.mean()
            stats[agent, 3] = agent_r.std(unbiased=False)
            stats[agent, 4] = density[pillar_mask].mean()
        return stats

    def forward(self, voxel_features, voxel_num_points, coords):
        if voxel_features.ndim != 3 or voxel_features.shape[-1] < 4:
            raise ValueError("voxel_features must have shape [M,T,C>=4]")
        point_count = voxel_features.shape[1]
        valid = self._valid_mask(voxel_num_points, point_count)
        xyz = voxel_features[..., :3]
        intensity = voxel_features[..., self.intensity_index]
        ranges = torch.linalg.vector_norm(xyz, dim=-1)
        density = (voxel_num_points.to(voxel_features.dtype) /
                   float(max(point_count, 1))).clamp(0.0, 1.0)
        density_points = density[:, None].expand_as(ranges)

        xyz_mean = self._masked_mean(xyz, valid)
        cluster = xyz - xyz_mean
        center_x = (coords[:, 3].to(xyz.dtype) * self.voxel_size[0] +
                    self.voxel_size[0] / 2 + self.pc_range[0])
        center_y = (coords[:, 2].to(xyz.dtype) * self.voxel_size[1] +
                    self.voxel_size[1] / 2 + self.pc_range[1])
        center_offset = torch.stack(
            (xyz[..., 0] - center_x[:, None],
             xyz[..., 1] - center_y[:, None]), dim=-1)
        elevation = torch.atan2(xyz[..., 2],
                                torch.linalg.vector_norm(xyz[..., :2], dim=-1)
                                .clamp_min(1e-3)) / (math.pi / 2)
        if self.ring_index is not None and int(self.ring_index) < voxel_features.shape[-1]:
            ring = voxel_features[..., int(self.ring_index)]
            ring = (ring - self._masked_mean(ring, valid)).tanh()
        else:
            ring = elevation
        geometry_input = torch.cat((
            xyz / xyz.new_tensor(self.xyz_scale),
            (ranges / self.max_range)[..., None], ring[..., None],
            density_points[..., None],
            cluster / xyz.new_tensor(self.xyz_scale),
            center_offset / xyz.new_tensor(self.voxel_size[:2])), dim=-1)
        geometry = self.geometry_encoder(geometry_input)

        pillar_geometry = self._masked_mean(
            self.geometry_to_context(geometry), valid, keepdim=False)
        spectral = self.spectral(pillar_geometry, coords)
        spectral_points = spectral[:, None, :].expand(-1, point_count, -1)
        geometry_evidence = self.geometry_evidence(
            torch.cat((geometry, spectral_points), dim=-1))

        agent_stats = self._agent_statistics(
            intensity.detach(), (ranges / self.max_range).detach(),
            density.detach(), valid, coords)
        point_stats = agent_stats[coords[:, 0].long()]
        median = point_stats[:, 0, None]
        mad = point_stats[:, 1, None].clamp_min(1e-3)
        robust_intensity = ((intensity - median) / mad).clamp(-8.0, 8.0) / 8.0
        bounded_intensity = torch.log1p(intensity.clamp_min(0.0))
        bounded_intensity = bounded_intensity / \
            (bounded_intensity.detach().amax().clamp_min(1.0))
        radiometric_input = torch.stack((
            robust_intensity, bounded_intensity, ring, density_points), dim=-1)
        radiometric = self.radiometric_encoder(radiometric_input)
        adapter_stats = point_stats.clone()
        adapter_stats[:, :2] = torch.log1p(
            adapter_stats[:, :2].clamp_min(0.0)) / 6.0
        gamma, beta = self.sensor_adapter(adapter_stats).chunk(2, dim=-1)
        radiometric = radiometric * (1.0 + 0.1 * torch.tanh(gamma[:, None])) + \
            0.1 * beta[:, None]
        radiometric_evidence = self.radiometric_evidence(radiometric)

        physics_input = torch.stack((
            ranges / self.max_range, robust_intensity, density_points), dim=-1)
        physics_evidence = self.physics_evidence(physics_input)

        raw_evidence = geometry_evidence + radiometric_evidence + \
            physics_evidence + self.fusion_bias
        evidence = F.softplus(raw_evidence).clamp_max(self.evidence_cap)
        alpha = evidence + 1.0
        reliability = alpha[..., 0] / alpha.sum(dim=-1)
        uncertainty = 2.0 / alpha.sum(dim=-1)
        reliability = self.reliability_floor + \
            (1.0 - self.reliability_floor) * reliability
        reliability = reliability * valid.to(reliability.dtype)
        uncertainty = uncertainty * valid.to(uncertainty.dtype)
        output = {
            "point_reliability": reliability,
            "point_uncertainty": uncertainty,
            "point_evidence": evidence,
            "point_valid_mask": valid,
            "pillar_reliability": self._masked_mean(
                reliability, valid, keepdim=False),
            "pillar_uncertainty": self._masked_mean(
                uncertainty, valid, keepdim=False),
        }
        if self.return_branch_support:
            # These are positive diagnostic transforms of the branch logits.
            # Their sum is not the fused evidence because fusion applies one
            # softplus after summing branch logits and the learned prior.
            output.update({
                "geometry_support": F.softplus(geometry_evidence),
                "radiometric_support": F.softplus(radiometric_evidence),
                "physics_support": F.softplus(physics_evidence),
            })
        return output
