"""Sweep a weather-augmentation dataset without running the model.

This preflight exercises every training sample and the real collate function,
catching empty-agent or clean/weather index mismatches before a long GPU run.
It does not update weights or write checkpoints.
"""

import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sweep all pre-voxel weather augmentation samples.')
    parser.add_argument('--hypes_yaml', required=True)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_samples', default=0, type=int,
                        help='0 checks the complete training dataset.')
    return parser.parse_args()


def main():
    opt = parse_args()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml)
    dataset = build_dataset(hypes, visualize=False, train=True)
    if hasattr(dataset, 'set_weather_augmentation_epoch'):
        dataset.set_weather_augmentation_epoch(0)
    sample_count = len(dataset) if opt.max_samples <= 0 else min(
        len(dataset), opt.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        collate_fn=dataset.collate_batch_train,
        shuffle=False,
        pin_memory=False,
        drop_last=False)

    checked = 0
    for batch_data in tqdm(loader, total=len(dataset)):
        ego = batch_data['ego']
        if 'processed_lidar_weather' not in ego:
            raise KeyError('processed_lidar_weather is missing')
        if int(ego['record_len'].sum().item()) <= 0:
            raise AssertionError('record_len must contain at least one agent')
        checked += 1
        if checked >= sample_count:
            break

    print('Weather augmentation dataset sweep passed for %d samples' %
          checked)


if __name__ == '__main__':
    main()
