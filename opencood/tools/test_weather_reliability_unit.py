"""
Unit checks for non-learned Weather Reliability Map.

Run:
python opencood/tools/test_weather_reliability_unit.py
"""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.utils.weather_reliability import build_weather_reliability


def main():
    lidar_range = [0.0, 0.0, -3.0, 16.0, 8.0, 1.0]
    voxel_size = [1.0, 1.0, 4.0]
    grid_size = [16, 8, 1]
    num_agents = 4
    num_voxels = 80

    batch_idx = torch.randint(0, num_agents, (num_voxels, 1))
    z_idx = torch.zeros(num_voxels, 1, dtype=torch.long)
    y_idx = torch.randint(0, grid_size[1], (num_voxels, 1))
    x_idx = torch.randint(0, grid_size[0], (num_voxels, 1))
    voxel_coords = torch.cat([batch_idx, z_idx, y_idx, x_idx], dim=1)
    voxel_num_points = torch.randint(1, 32, (num_voxels,), dtype=torch.int32)
    record_len = torch.tensor([2, 2], dtype=torch.long)
    confidence_map = torch.rand(num_agents, 2, 4, 8)

    config = {
        'enable': True,
        'density_enable': True,
        'isolation_enable': True,
        'use_clean_stats': False,
        'alpha_density': 0.7,
        'beta_isolation': 0.3,
        'eps': 1.0e-6,
        'resize_mode': 'bilinear'
    }
    weather = build_weather_reliability(voxel_coords,
                                        voxel_num_points,
                                        record_len=record_len,
                                        confidence_map=confidence_map,
                                        config=config,
                                        lidar_range=lidar_range,
                                        voxel_size=voxel_size,
                                        grid_size=grid_size)

    assert weather['count_map'].shape == (num_agents, 1, 8, 16)
    assert weather['R_final'].shape == (num_agents, 1, 8, 16)
    assert weather['R_final_resized'].shape == (num_agents, 1, 4, 8)
    assert torch.isfinite(weather['R_final']).all()
    assert torch.isfinite(weather['R_final_resized']).all()
    assert float(weather['R_final'].min()) >= 0.0
    assert float(weather['R_final'].max()) <= 1.0
    assert float(weather['R_final_resized'].min()) >= 0.0
    assert float(weather['R_final_resized'].max()) <= 1.0
    print('Weather reliability unit test passed.')


if __name__ == '__main__':
    main()
