"""
Non-learned weather reliability maps for Where2comm.

The utilities in this file build an agent-wise BEV reliability map from
PointPillars voxel statistics.  They intentionally contain no learnable state,
so enabling or disabling the feature does not affect checkpoint loading.
"""

import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F


_STATS_CACHE = {}
_WARNED_MESSAGES = set()


def _warn_once(message):
    if message not in _WARNED_MESSAGES:
        warnings.warn(message)
        _WARNED_MESSAGES.add(message)


def _cfg(config, key, default):
    if config is None:
        return default
    return config[key] if key in config else default


def _bounded_cfg_float(config, key, default, min_value=0.0, max_value=1.0):
    value = float(_cfg(config, key, default))
    return min(max(value, min_value), max_value)


def infer_grid_size(lidar_range=None, voxel_size=None, grid_size=None):
    """
    Return BEV grid size as (H, W), matching PointPillarScatter output.
    """
    if grid_size is not None:
        # OpenCOOD stores grid_size as [nx, ny, nz].
        return int(grid_size[1]), int(grid_size[0])
    if lidar_range is None or voxel_size is None:
        raise ValueError('Either grid_size or both lidar_range and voxel_size are required.')

    x_min, y_min, _, x_max, y_max, _ = lidar_range
    vx, vy = voxel_size[:2]
    width = int(np.ceil((x_max - x_min) / vx))
    height = int(np.ceil((y_max - y_min) / vy))
    return height, width


def build_count_map_from_voxels(voxel_coords,
                                voxel_num_points,
                                record_len=None,
                                lidar_range=None,
                                voxel_size=None,
                                grid_size=None):
    """
    Build per-agent BEV point count maps from sparse voxel statistics.

    Parameters
    ----------
    voxel_coords : torch.Tensor
        Shape (num_voxels, 4), formatted as [agent_batch_idx, z_idx, y_idx, x_idx].
    voxel_num_points : torch.Tensor
        Shape (num_voxels,), number of LiDAR points in each voxel/pillar.
    record_len : torch.Tensor, optional
        Number of CAVs per scene. Used to keep empty agents in the output.

    Returns
    -------
    torch.Tensor
        Count map with shape (sum(record_len), 1, H, W).
    """
    if voxel_coords is None or voxel_num_points is None:
        raise ValueError('voxel_coords and voxel_num_points are required.')
    if voxel_coords.numel() == 0:
        if record_len is None:
            raise ValueError('record_len is required when voxel_coords is empty.')
        batch_size = int(record_len.sum().item())
        height, width = infer_grid_size(lidar_range, voxel_size, grid_size)
        return torch.zeros(batch_size, 1, height, width,
                           dtype=torch.float32,
                           device=voxel_num_points.device)

    coords = voxel_coords.long()
    counts = voxel_num_points.to(dtype=torch.float32, device=coords.device)
    height, width = infer_grid_size(lidar_range, voxel_size, grid_size)

    if record_len is not None:
        batch_size = int(record_len.sum().item())
    else:
        batch_size = int(coords[:, 0].max().item()) + 1

    count_map = torch.zeros(batch_size, 1, height, width,
                            dtype=counts.dtype,
                            device=counts.device)

    batch_idx = coords[:, 0]
    y_idx = coords[:, 2]
    x_idx = coords[:, 3]
    valid = ((batch_idx >= 0) & (batch_idx < batch_size) &
             (y_idx >= 0) & (y_idx < height) &
             (x_idx >= 0) & (x_idx < width))
    if not torch.all(valid):
        _warn_once('Some voxel coordinates fell outside the configured BEV grid.')
        batch_idx = batch_idx[valid]
        y_idx = y_idx[valid]
        x_idx = x_idx[valid]
        counts = counts[valid]

    count_map.index_put_((batch_idx, torch.zeros_like(batch_idx), y_idx, x_idx),
                         counts,
                         accumulate=True)
    return count_map


def load_clean_statistics(stat_path, device, dtype):
    if not stat_path:
        return None
    stat_path = os.path.abspath(os.path.expanduser(stat_path))
    if not os.path.exists(stat_path):
        _warn_once('Weather reliability statistics file not found: %s' % stat_path)
        return None
    if stat_path not in _STATS_CACHE:
        _STATS_CACHE[stat_path] = dict(np.load(stat_path, allow_pickle=True))
    stats = _STATS_CACHE[stat_path]
    if 'bin_edges' not in stats or 'mean_count' not in stats:
        _warn_once('Weather reliability statistics missing bin_edges or mean_count: %s' %
                   stat_path)
        return None
    return {
        'bin_edges': torch.as_tensor(stats['bin_edges'], device=device, dtype=dtype),
        'mean_count': torch.as_tensor(stats['mean_count'], device=device, dtype=dtype)
    }


def make_distance_map(height, width, lidar_range, voxel_size, device, dtype):
    x_min, y_min = lidar_range[0], lidar_range[1]
    vx, vy = voxel_size[:2]
    y_ids = torch.arange(height, device=device, dtype=dtype)
    x_ids = torch.arange(width, device=device, dtype=dtype)
    try:
        yy, xx = torch.meshgrid(y_ids, x_ids, indexing='ij')
    except TypeError:
        yy, xx = torch.meshgrid(y_ids, x_ids)
    x_centers = x_min + (xx + 0.5) * vx
    y_centers = y_min + (yy + 0.5) * vy
    return torch.sqrt(x_centers * x_centers + y_centers * y_centers)


def density_from_clean_stats(count_map, stats, lidar_range, voxel_size, eps,
                             config=None):
    _, _, height, width = count_map.shape
    distance_map = make_distance_map(height, width, lidar_range, voxel_size,
                                     count_map.device, count_map.dtype)
    bin_edges = stats['bin_edges']
    mean_count = stats['mean_count']
    if mean_count.numel() + 1 != bin_edges.numel():
        _warn_once('Weather reliability statistics have incompatible bin sizes.')
        return None

    bin_ids = torch.bucketize(distance_map.reshape(-1), bin_edges[1:-1])
    expected = mean_count[bin_ids].reshape(1, 1, height, width)
    valid_expected = expected > eps
    fallback = mean_count[mean_count > eps].mean() if torch.any(mean_count > eps) \
        else torch.tensor(1.0, device=count_map.device, dtype=count_map.dtype)
    expected = torch.where(valid_expected, expected, fallback)
    density = count_map / (expected + eps)
    density = torch.clamp(density, min=0.0, max=1.0)
    empty_reliability = _bounded_cfg_float(
        config, 'empty_cell_reliability', 0.0)
    density = torch.where(count_map > 0, density,
                          torch.full_like(density, empty_reliability))
    return density


def density_from_local_mean(count_map, config):
    eps = float(_cfg(config, 'eps', 1.0e-6))
    kernel_size = int(_cfg(config, 'density_local_kernel_size', 9))
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    local_mean = F.avg_pool2d(count_map,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              count_include_pad=False)
    density = count_map / (local_mean + eps)
    density = torch.clamp(density, min=0.0, max=1.0)
    empty_reliability = _bounded_cfg_float(
        config, 'empty_cell_reliability', 0.0)
    density = torch.where(count_map > 0, density,
                          torch.full_like(density, empty_reliability))
    return density


def compute_density_reliability(count_map, config, lidar_range, voxel_size):
    if not bool(_cfg(config, 'density_enable', True)):
        return torch.ones_like(count_map)

    eps = float(_cfg(config, 'eps', 1.0e-6))
    stats = None
    if bool(_cfg(config, 'use_clean_stats', True)):
        stats = load_clean_statistics(_cfg(config, 'stat_path', None),
                                      count_map.device,
                                      count_map.dtype)
    if stats is not None and lidar_range is not None and voxel_size is not None:
        density = density_from_clean_stats(count_map, stats, lidar_range,
                                           voxel_size, eps, config=config)
        if density is not None:
            return density
    return density_from_local_mean(count_map, config)


def compute_isolation_reliability(count_map, config, lidar_range, voxel_size):
    if not bool(_cfg(config, 'isolation_enable', True)):
        return torch.ones_like(count_map)

    kernel_size = int(_cfg(config, 'isolation_kernel_size', 3))
    if kernel_size % 2 == 0:
        kernel_size += 1
    threshold = float(_cfg(config, 'isolation_neighbor_threshold', 2))
    penalty_value = float(_cfg(config, 'isolation_penalty_value', 0.5))
    penalty_value = min(max(penalty_value, 0.0), 1.0)

    occupancy = (count_map > 0).to(count_map.dtype)
    kernel = torch.ones(1, 1, kernel_size, kernel_size,
                        dtype=count_map.dtype,
                        device=count_map.device)
    neighbor_count = F.conv2d(occupancy, kernel, padding=kernel_size // 2)
    isolated = (occupancy > 0) & (neighbor_count <= threshold)

    if lidar_range is not None and voxel_size is not None:
        _, _, height, width = count_map.shape
        distance_map = make_distance_map(height, width, lidar_range, voxel_size,
                                         count_map.device, count_map.dtype)
        min_range = float(_cfg(config, 'isolation_min_range', 0.0))
        max_range = float(_cfg(config, 'isolation_max_range', 1.0e8))
        range_mask = ((distance_map >= min_range) &
                      (distance_map <= max_range)).reshape(1, 1, height, width)
        isolated = isolated & range_mask

    reliability = torch.ones_like(count_map)
    reliability = torch.where(isolated,
                              reliability * (1.0 - penalty_value),
                              reliability)
    return torch.clamp(reliability, min=0.0, max=1.0)


def resize_reliability(reliability, confidence_map, resize_mode='bilinear'):
    """
    Resize reliability to confidence map spatial shape.

    confidence_map may be (N, C, H, W) or (N, H, W). The returned tensor is
    (N, 1, H, W) for safe broadcasting over confidence channels.
    """
    if confidence_map.dim() == 3:
        target_size = confidence_map.shape[-2:]
    elif confidence_map.dim() == 4:
        target_size = confidence_map.shape[-2:]
    else:
        raise ValueError('confidence_map must have shape (N,H,W) or (N,C,H,W).')

    reliability = reliability.to(device=confidence_map.device,
                                 dtype=confidence_map.dtype)
    if reliability.shape[-2:] == target_size:
        return reliability

    kwargs = {}
    if resize_mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        kwargs['align_corners'] = False
    return F.interpolate(reliability, size=target_size, mode=resize_mode, **kwargs)


def build_weather_reliability(voxel_coords,
                              voxel_num_points,
                              record_len=None,
                              confidence_map=None,
                              config=None,
                              lidar_range=None,
                              voxel_size=None,
                              grid_size=None):
    """
    Build count, component reliability maps, and final R(x, y).
    """
    count_map = build_count_map_from_voxels(voxel_coords,
                                            voxel_num_points,
                                            record_len=record_len,
                                            lidar_range=lidar_range,
                                            voxel_size=voxel_size,
                                            grid_size=grid_size)
    r_density = compute_density_reliability(count_map, config,
                                            lidar_range, voxel_size)
    r_isolation = compute_isolation_reliability(count_map, config,
                                                lidar_range, voxel_size)
    r_intensity = torch.ones_like(count_map)

    alpha = float(_cfg(config, 'alpha_density', 0.7))
    beta = float(_cfg(config, 'beta_isolation', 0.3))
    gamma = float(_cfg(config, 'gamma_intensity', 0.0))

    r_final = torch.pow(torch.clamp(r_density, 0.0, 1.0), alpha) * \
        torch.pow(torch.clamp(r_isolation, 0.0, 1.0), beta) * \
        torch.pow(torch.clamp(r_intensity, 0.0, 1.0), gamma)
    r_final = torch.clamp(r_final, min=0.0, max=1.0)
    try:
        r_final = torch.nan_to_num(r_final, nan=0.0, posinf=1.0, neginf=0.0)
    except AttributeError:
        r_final = torch.where(torch.isfinite(r_final), r_final,
                              torch.zeros_like(r_final))

    resize_mode = _cfg(config, 'resize_mode', 'bilinear')
    r_resized = resize_reliability(r_final, confidence_map, resize_mode) \
        if confidence_map is not None else r_final

    return {
        'count_map': count_map,
        'R_density': r_density,
        'R_isolation': r_isolation,
        'R_intensity': r_intensity,
        'R_final': r_final,
        'R_final_resized': r_resized
    }


def summarize_tensor(tensor):
    tensor = tensor.detach()
    return {
        'min': tensor.min(),
        'max': tensor.max(),
        'mean': tensor.mean()
    }


def _to_numpy(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()


def save_weather_debug_npz(debug_dir, sample_index, weather_maps, comm_details):
    """
    Save one npz per agent for lightweight visual debugging.
    """
    os.makedirs(debug_dir, exist_ok=True)
    required = ['count_map', 'R_density', 'R_isolation', 'R_final']
    for key in required:
        if key not in weather_maps:
            return

    count_map = weather_maps['count_map']
    num_agents = count_map.shape[0]
    for agent_idx in range(num_agents):
        save_path = os.path.join(
            debug_dir,
            'sample_%06d_agent_%02d.npz' % (sample_index, agent_idx))
        np.savez_compressed(
            save_path,
            count_map=_to_numpy(weather_maps['count_map'][agent_idx, 0]),
            R_density=_to_numpy(weather_maps['R_density'][agent_idx, 0]),
            R_isolation=_to_numpy(weather_maps['R_isolation'][agent_idx, 0]),
            R_final=_to_numpy(weather_maps['R_final'][agent_idx, 0]),
            confidence_map=_to_numpy(comm_details.get('confidence_map', None)[agent_idx, 0])
            if comm_details.get('confidence_map', None) is not None else None,
            effective_confidence=_to_numpy(
                comm_details.get('effective_confidence', None)[agent_idx, 0])
            if comm_details.get('effective_confidence', None) is not None else None,
            communication_mask=_to_numpy(
                comm_details.get('communication_mask', None)[agent_idx, 0])
            if comm_details.get('communication_mask', None) is not None else None)
