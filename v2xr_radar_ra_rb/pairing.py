"""Training-only counterfactual LiDAR reads, on the exact native per-car path."""
from pathlib import Path
import random

import numpy as np
import torch

from inputs import TRAIN, SUFFIX
from support import patched


def coverage_at(voxels, queries, nx):
    coords = voxels['voxel_coords']
    n = voxels['voxel_num_points']
    xyz = voxels['voxel_features'][..., :3]
    valid = np.arange(xyz.shape[1])[None] < n[:, None]
    mean = (xyz * valid[..., None]).sum(1) / np.maximum(n[:, None], 1)
    spread = np.sqrt((((xyz-mean[:, None])**2).sum(-1)*valid).sum(1) / np.maximum(n, 1))
    cells = {int(y)*nx+int(x): (count, width) for (_, y, x), count, width in zip(coords, n, spread)}
    return np.asarray([cells.get(int(y)*nx+int(x), (0, 0.)) for _, y, x in queries],
                      dtype=np.float32).reshape(-1, 2)


def targets(clean, fog, is_fog):
    n, span = clean.T
    m, adverse_span = fog.T
    density_loss = np.clip((n-m)/np.maximum(n, 1), 0, 1)
    spread_loss = np.clip((span-adverse_span)/np.maximum(span, .05), 0, 1)
    damage = .75*density_loss + .25*spread_loss
    damage[(n > 0) & (m == 0)] = 1.
    known_fog = (n > 0) & is_fog
    healthy = (n > 0) & (not is_fog or ((m > 0) & (damage <= .1)))
    return np.column_stack((damage, known_fog, healthy)).astype(np.float32)


class PairedDataset:
    def __init__(self, base, nx):
        self.base, self.native, self.keys, self.nx = base, base.native, base.keys, nx
        if not self.native.train or not self.native.train_sim or self.native.async_flag:
            raise ValueError('Paired supervision is restricted to synchronous mixed training')
        self.native_collate = self.native.collate_batch_train
        self.native.collate_batch_train = self.collate

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        from opencood.utils import pcd_utils

        dataset = self.native
        original_read = pcd_utils.pcd_to_np
        retrieve, single = dataset.retrieve_base_data, dataset.get_item_single_car
        paths, pairs, audit = [], [], []
        mapped = {}

        def read(path, *args, **kwargs):
            if kwargs.get('get_weather'):
                paths.append(Path(path))
            return original_read(path, *args, **kwargs)

        def capture(*args, **kwargs):
            paths.clear()
            pairs.clear()
            audit.clear()
            mapped.clear()
            data, scene_index, stamp = retrieve(*args, **kwargs)
            if len(data) != len(paths):
                raise ValueError('Per-car weather reads do not match official retrieval')
            for car, path in zip(data, paths):
                clean = path.with_name(stamp + SUFFIX['clean'])
                fog = path.with_name(stamp + SUFFIX['fog'])
                if (not path.is_relative_to(TRAIN) or path.parent.name != str(car)
                        or path not in (clean, fog) or data[car]['time_delay'] != 0):
                    raise ValueError(f'Unpaired training read {path}')
                mapped[car] = (clean, fog, path == fog)
            return data, scene_index, stamp

        def paired(car, ego_pose, ego_params):
            before = np.random.get_state()
            base = single(car, ego_pose, ego_params)
            clean_path, fog_path, is_fog = mapped[car['id']]
            radar = base['processed_radar_features']['voxel_coords']
            if len(radar):
                other = original_read(str(clean_path if is_fog else fog_path), get_weather=True)[0]
                after, py_after = np.random.get_state(), random.getstate()
                try:
                    np.random.set_state(before)
                    counter = single(dict(car, lidar_np=other), ego_pose, ego_params)
                finally:
                    np.random.set_state(after)
                    random.setstate(py_after)
                current = coverage_at(base['processed_lidar_features'], radar, self.nx)
                alternate = coverage_at(counter['processed_lidar_features'], radar, self.nx)
                labels = targets(alternate if is_fog else current, current if is_fog else alternate, is_fog)
            else:
                labels = np.empty((0, 3), np.float32)
            pairs.append((radar.copy(), labels))
            audit.append((str(clean_path.relative_to(TRAIN)), str(fog_path.relative_to(TRAIN)), bool(is_fog), len(radar)))
            return base

        with patched(pcd_utils, 'pcd_to_np', read), patched(dataset, 'retrieve_base_data', capture), \
                patched(dataset, 'get_item_single_car', paired):
            sample = self.base[index]
        if len(pairs) != sample['ego']['cav_num']:
            raise ValueError('Paired order differs from the actual participating vehicle order')
        sample['pair'] = pairs
        sample['pair_audit'] = audit
        return sample

    def collate(self, samples):
        output = self.native_collate(samples)
        rows, labels, audit = [], [], []
        car = 0
        for sample in samples:
            for coords, value in sample['pair']:
                rows.append(np.pad(coords, ((0, 0), (1, 0)), constant_values=car))
                labels.append(value)
                car += 1
            audit.extend(sample['pair_audit'])
        coords = torch.from_numpy(np.concatenate(rows))
        if not torch.equal(coords, output['ego']['processed_radar']['voxel_coords']):
            raise ValueError('Paired labels changed native Radar vehicle/grid order')
        output['paired_weather'] = (coords, torch.from_numpy(np.concatenate(labels)))
        output['pair_audit'] = audit
        return output

