"""
Precompute clean-weather distance statistics for Weather-Reliability Where2comm.

Example:
python opencood/tools/precompute_weather_reliability_stats.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_where2comm.yaml \
  --split train \
  --output weather_reliability_stats_opv2v_clean.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.weather_reliability import (
    build_count_map_from_voxels,
    make_distance_map
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute clean LiDAR density statistics per distance bin.')
    parser.add_argument('--hypes_yaml', required=True, type=str,
                        help='Clean dataset yaml file.')
    parser.add_argument('--split', choices=['train', 'validate'], default='train',
                        help='Dataset split used for statistics.')
    parser.add_argument('--output', default='weather_reliability_stats_opv2v_clean.npz',
                        type=str)
    parser.add_argument('--bin_size', default=5.0, type=float,
                        help='Distance bin size in meters.')
    parser.add_argument('--max_range', default=0.0, type=float,
                        help='Maximum distance in meters. Use 0 to infer from lidar range.')
    parser.add_argument('--include_empty_cells', action='store_true',
                        help='Include empty BEV cells in count statistics.')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--max_samples', default=0, type=int,
                        help='Limit samples for quick checks. Use 0 for all samples.')
    return parser.parse_args()


def resolve_bin_edges(lidar_range, bin_size, max_range):
    if max_range <= 0:
        corners = np.array([
            [lidar_range[0], lidar_range[1]],
            [lidar_range[0], lidar_range[4]],
            [lidar_range[3], lidar_range[1]],
            [lidar_range[3], lidar_range[4]]
        ])
        max_range = float(np.sqrt((corners ** 2).sum(axis=1)).max())
    num_bins = int(np.ceil(max_range / bin_size))
    return np.linspace(0.0, num_bins * bin_size, num_bins + 1, dtype=np.float32)


def fill_empty_bins(mean_count, median_count, p25, p75, num_cells):
    valid = num_cells > 0
    global_mean = float(mean_count[valid].mean()) if np.any(valid) else 1.0
    for i in range(len(mean_count)):
        if num_cells[i] > 0:
            continue
        left = i - 1
        right = i + 1
        while left >= 0 and num_cells[left] == 0:
            left -= 1
        while right < len(mean_count) and num_cells[right] == 0:
            right += 1
        if left >= 0 and right < len(mean_count):
            source_values = [left, right]
        elif left >= 0:
            source_values = [left]
        elif right < len(mean_count):
            source_values = [right]
        else:
            mean_count[i] = global_mean
            median_count[i] = global_mean
            p25[i] = global_mean
            p75[i] = global_mean
            continue
        mean_count[i] = float(np.mean(mean_count[source_values]))
        median_count[i] = float(np.mean(median_count[source_values]))
        p25[i] = float(np.mean(p25[source_values]))
        p75[i] = float(np.mean(p75[source_values]))


def main():
    opt = parse_args()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml)
    train = opt.split == 'train'
    dataset = build_dataset(hypes, visualize=False, train=train)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             num_workers=opt.num_workers,
                             collate_fn=dataset.collate_batch_test if not train
                             else dataset.collate_batch_train,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    lidar_range = hypes['preprocess']['cav_lidar_range']
    voxel_size = hypes['preprocess']['args']['voxel_size']
    grid_size = hypes['model']['args']['point_pillar_scatter']['grid_size']
    bin_edges = resolve_bin_edges(lidar_range, opt.bin_size, opt.max_range)
    bin_values = [[] for _ in range(len(bin_edges) - 1)]

    height = int(grid_size[1])
    width = int(grid_size[0])
    distance_map = make_distance_map(height, width, lidar_range, voxel_size,
                                     torch.device('cpu'), torch.float32)
    bin_ids = torch.bucketize(distance_map.reshape(-1),
                              torch.as_tensor(bin_edges[1:-1]))
    bin_ids_np = bin_ids.numpy()

    max_iter = len(data_loader) if opt.max_samples <= 0 else min(opt.max_samples, len(data_loader))
    for sample_idx, batch_data in tqdm(enumerate(data_loader), total=max_iter):
        if sample_idx >= max_iter:
            break
        ego_dict = batch_data['ego']
        processed = ego_dict['processed_lidar']
        count_map = build_count_map_from_voxels(
            processed['voxel_coords'],
            processed['voxel_num_points'],
            record_len=ego_dict.get('record_len', None),
            lidar_range=lidar_range,
            voxel_size=voxel_size,
            grid_size=grid_size)

        flat_counts = count_map[:, 0].reshape(count_map.shape[0], -1).cpu().numpy()
        for agent_counts in flat_counts:
            if opt.include_empty_cells:
                valid_mask = np.ones_like(agent_counts, dtype=bool)
            else:
                valid_mask = agent_counts > 0
            selected_bins = bin_ids_np[valid_mask]
            selected_counts = agent_counts[valid_mask]
            for bin_idx in range(len(bin_values)):
                values = selected_counts[selected_bins == bin_idx]
                if values.size > 0:
                    bin_values[bin_idx].append(values.astype(np.float32))

    mean_count = np.zeros(len(bin_values), dtype=np.float32)
    median_count = np.zeros(len(bin_values), dtype=np.float32)
    percentile_25 = np.zeros(len(bin_values), dtype=np.float32)
    percentile_75 = np.zeros(len(bin_values), dtype=np.float32)
    num_cells = np.zeros(len(bin_values), dtype=np.int64)

    for bin_idx, chunks in enumerate(bin_values):
        if not chunks:
            continue
        values = np.concatenate(chunks)
        num_cells[bin_idx] = values.size
        mean_count[bin_idx] = float(values.mean())
        median_count[bin_idx] = float(np.median(values))
        percentile_25[bin_idx] = float(np.percentile(values, 25))
        percentile_75[bin_idx] = float(np.percentile(values, 75))

    fill_empty_bins(mean_count, median_count, percentile_25, percentile_75, num_cells)

    output = os.path.abspath(os.path.expanduser(opt.output))
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    np.savez_compressed(output,
                        bin_edges=bin_edges,
                        mean_count=mean_count,
                        median_count=median_count,
                        percentile_25=percentile_25,
                        percentile_75=percentile_75,
                        num_cells=num_cells,
                        x_range=np.array([lidar_range[0], lidar_range[3]], dtype=np.float32),
                        y_range=np.array([lidar_range[1], lidar_range[4]], dtype=np.float32),
                        H=np.array(height, dtype=np.int64),
                        W=np.array(width, dtype=np.int64),
                        include_empty_cells=np.array(opt.include_empty_cells))
    print('Saved weather reliability statistics to %s' % output)


if __name__ == '__main__':
    main()
