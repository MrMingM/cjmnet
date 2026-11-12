"""Audit F3 training inputs without stepping its optimizer or changing the loader."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from checkpoint import seed, rng, restore_rng
from common import read_json, sha256
from enhancement import point_statistics
from feature_probe import boxes_mask, hook
from pairing import PairedDataset
from support import patched
from v2xr_weather_l012.data import tensor_digest


def logs(group, epoch):
    totals = Counter()
    evidence = {}
    for i in range(1,epoch+1):
        directory = group/'epochs'/f'{i:02d}'
        row = read_json(directory/'complete.json')
        if row['epoch'] != i or row['train_updates'] != 1010:
            raise ValueError('F3 epoch did not complete')
        path = directory/'auxiliary.json'
        aux = read_json(path)
        totals.update(aux['paired_vehicle_reads'])
        totals.update(aux['loss_and_mask_sums'])
        evidence[str(path)] = sha256(path)
    return dict(totals=totals, evidence=evidence, epochs=epoch)


def correlation(x, y):
    if len(x)<2 or x.std()==0 or y.std()==0:
        return None
    return float(np.corrcoef(x,y)[0,1])


def inspect(experiment, model, config, samples=48):
    from opencood.tools import train_utils

    unpaired, _ = experiment.build(config, split='train')
    native, cfg = experiment.build(config, split='train')
    nx = int(cfg['model']['args']['point_pillar_scatter']['grid_size'][0])
    pair = PairedDataset(native, nx)
    source_single = pair.native.get_item_single_car
    retrieved = pair.native.retrieve_base_data
    values = defaultdict(list)
    aligned_pairs = 0
    observations = {}
    indices = np.linspace(0,len(pair)-1,samples,dtype=int).tolist()
    vehicles = Counter()

    def reset(*args, **kwargs):
        observations.clear()
        return retrieved(*args, **kwargs)

    def verify_pair(car, ego_pose, ego_params):
        result = source_single(car, ego_pose, ego_params)
        coords = result['processed_radar_features']['voxel_coords']
        # LiDAR length can change Radar shuffle order. Compare occupied cells as sets.
        ordered = sorted(map(tuple, coords.tolist()))
        geometry = tensor_digest([car['params']['transformation_matrix'], ego_pose,
                                  ego_params['c2e_transformation_matrix']])
        boxes = result['object_bbx_center'].copy()
        if car['id'] in observations:
            ref, cells, old_boxes = observations[car['id']]
            if ref!=geometry or cells!=ordered or not np.array_equal(old_boxes,boxes):
                raise ValueError('Paired projected geometry or Radar occupied set differs')
        else:
            observations[car['id']] = (geometry,ordered,boxes)
        return result

    for index in indices:
        seed(20261017 + index)
        before = rng()
        ordinary = unpaired[index]
        ordinary_rng = tensor_digest(rng())
        restore_rng(before)
        with patched(pair.native,'retrieve_base_data',reset), patched(pair.native,'get_item_single_car',verify_pair):
            sample = pair[index]
        if tensor_digest(ordinary['ego']) != tensor_digest(sample['ego']) or ordinary_rng!=tensor_digest(rng()):
            raise ValueError('Paired read changed official native sample or RNG')
        batch = pair.collate([sample])
        coords, labels = batch['paired_weather']
        if not torch.equal(coords,batch['ego']['processed_radar']['voxel_coords']):
            raise ValueError('Label occupied coordinates differ from Radar')
        # Weather-loss mask is defined only for sampled Fog with Clean coverage.
        is_fog = np.repeat([v[2] for v in batch['pair_audit']], [v[3] for v in batch['pair_audit']]).astype(bool)
        aligned_pairs += sum(v[3]>0 for v in batch['pair_audit'])
        vehicles.update('fog' if v[2] else 'clean' for v in batch['pair_audit'])
        if not len(coords):
            continue
        ego = train_utils.to_device(batch['ego'], torch.device('cuda:0'))
        with torch.no_grad(), hook(model,'D0') as captured:
            model(ego)
        trace, = captured
        if not torch.equal(coords,trace['coords'].cpu()):
            raise ValueError('Label alignment differs at R-B input')
        centers = point_statistics(ego['processed_radar'])[1].cpu().numpy()
        foreground = np.zeros(len(coords),dtype=bool)
        for car, audit in enumerate(batch['pair_audit']):
            car_id = Path(audit[0]).parent.name
            mask = coords[:,0].numpy()==car
            foreground[mask] = boxes_mask(centers[mask],observations[car_id][2]).any(0)
        for key,array in dict(label=labels[:,0].numpy(), eligible=labels[:,1].numpy().astype(bool),
            healthy=labels[:,2].numpy().astype(bool), fog=is_fog, foreground=foreground,
            gate=trace['gate'].cpu().numpy(), support=trace['support'].cpu().numpy()).items():
            values[key].append(array)
    fields = {k:np.concatenate(a) for k,a in values.items()}
    eligible, fog, fg = (fields[k] for k in ('eligible','fog','foreground'))
    if not eligible.any() or (eligible & ~fog).any():
        raise ValueError('Unexpected Fog label eligibility')
    report = {}
    for key,mask in dict(all=np.ones(len(fg),bool),foreground=fg,background=~fg).items():
        active = mask & eligible
        x,y = (fields[k][active] for k in ('label','gate'))
        weight = fields['support'][active]
        report[key] = dict(occupied=int(mask.sum()), sampled_fog=int((mask & fog).sum()),
            supervised=int(active.sum()), healthy_label=int((mask & fields['healthy']).sum()),
            positive_damage=int((x>0).sum()), fully_missing=int((x==1).sum()),
            mean_label=float(x.mean()) if len(x) else None, mean_gate=float(y.mean()) if len(y) else None,
            pearson=correlation(x,y), weighted_support=float(weight.sum()),
            mean_gate_by_label=[dict(n=int(selected.sum()), mean=float(y[selected].mean()) if selected.any() else None)
                for selected in (x==0,(x>0)&(x<1),x==1)])
    return dict(indices=indices, frames=len(indices), vehicles=vehicles, aligned_paired_reads=aligned_pairs,
                by_region=report, label_gate_bins=['0','(0,1)','1'],
                foreground='pillar centroid in the BEV footprint of pre-weather-filter same-car annotations')

