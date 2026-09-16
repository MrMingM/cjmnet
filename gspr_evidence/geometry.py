"""Endpoint support and angular-depth traversal samples, NOT occupancy probabilities.

Conceptual D-Map adaptation; no ROS/octree or original code is imported/copied.
Depth observations are made in the actual sender frame. BEV stays ego-aligned.
"""
import math
import torch
from gspr_communication.observation_stats import block_statistics

OBS = 13


def angular_bins(xyz, azimuth=720, elevation=180):
    radius = xyz.norm(dim=-1)
    az = torch.atan2(xyz[:, 1], xyz[:, 0])
    el = torch.atan2(xyz[:, 2], xyz[:, :2].norm(dim=-1))
    ai = ((az + math.pi)/(2*math.pi)*azimuth).floor().long().clamp(0, azimuth-1)
    ei = ((el + math.pi/2)/math.pi*elevation).floor().long().clamp(0, elevation-1)
    return ei*azimuth + ai, radius


def traversal_samples(cloud, transform, queries, margin=1.):
    # Row-vector inverse of x_ego = R x_sensor + t; never transform BEV again.
    cloud = cloud.to(device=queries.device, dtype=queries.dtype)
    transform = transform.to(device=queries.device, dtype=queries.dtype)
    rotation, origin = transform[:3, :3], transform[:3, 3]
    local = (cloud[:, :3] - origin) @ rotation
    local = local[torch.isfinite(local).all(1)]
    depth = queries.new_full((720*180,), float('inf'))
    ids, radii = angular_bins(local)
    valid = radii > 1e-3
    depth.scatter_reduce_(0, ids[valid], radii[valid], reduce='amin', include_self=True)
    qids, qr = angular_bins((queries - origin) @ rotation)
    seen = depth[qids]
    # One sampled line at the bin resolution. This does NOT label the whole cell free.
    return (torch.isfinite(seen) & (qr > 1e-3) & (qr + margin < seen)).float()


def describe(rel, processed, confidence, transforms, clouds, grid, lidar_range):
    n, _, h, w = confidence.shape
    stats = block_statistics(rel, processed, n, grid)
    profile = stats.new_zeros((n, OBS, h, w))
    profile[:, 0] = stats[:, 0]
    profile[:, 1] = torch.where(stats[:, 5] > 0, stats[:, 1], torch.ones_like(stats[:, 1]))
    profile[:, 2] = (stats[:, 2]/8).clamp(0, 1)
    profile[:, 3] = stats[:, 5]
    profile[:, 12:13] = confidence
    valid = rel['point_valid_mask'].bool()
    evidence = rel['point_evidence'] + 1
    p = evidence[..., 0]/evidence.sum(-1)
    trust = p  # Keep p-support separate from u so a u-channel ablation is identifiable.
    coords = processed['voxel_coords'].long()
    z = processed['voxel_features'][..., 2]
    zmin, zmax = float(lidar_range[2]), float(lidar_range[5])
    zb = ((z-zmin)/(zmax-zmin)*4).floor().long().clamp(0, 3)
    # Max trustworthy endpoint per retained pillar/height cell, then area fraction.
    hh, ww = h*8, w*8
    cell = ((coords[:, 0, None]*4 + zb)*hh + coords[:, 2, None])*ww + coords[:, 3, None]
    support = p.new_zeros(n*4*hh*ww)
    support.scatter_reduce_(0, cell[valid], trust[valid], reduce='amax', include_self=True)
    support = support.reshape(n*4, 1, hh, ww)
    profile[:, 4:8] = torch.nn.functional.avg_pool2d(support, 8).reshape(n, 4, h, w)
    xmin, ymin, _, xmax, ymax, _ = [float(x) for x in lidar_range]
    yy, xx = torch.meshgrid(torch.arange(h, device=p.device), torch.arange(w, device=p.device), indexing='ij')
    x = xmin + (xx.float()+.5)*(xmax-xmin)/w
    y = ymin + (yy.float()+.5)*(ymax-ymin)/h
    queries = torch.stack([torch.stack([x, y, torch.full_like(x, zmin+(k+.5)*(zmax-zmin)/4)], -1)
                           for k in range(4)]).reshape(-1, 3)
    if len(clouds) != n or transforms.shape != (n, 4, 4):
        raise ValueError('Raw cloud/true sensor pose order mismatch')
    for agent in range(n):
        profile[agent, 8:12] = traversal_samples(clouds[agent], transforms[agent], queries).reshape(4, h, w)
    return profile
