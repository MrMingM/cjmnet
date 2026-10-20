"""Identical diffusion inputs for E1/E2/E3; only position weights differ."""
import numpy as np
import torch

from .common import stream_seed
from .geometry import loss_weights
from .model import crop_condition


def diffusion_inputs(x0, seed, epoch, uid, object_index, config):
    r = np.random.RandomState(stream_seed(seed, epoch, [uid, object_index], 'diffusion'))
    d = config['diffusion']
    timestep = int(r.randint(d['steps']))
    epsilon = r.standard_normal(x0.shape).astype(np.float32)
    alpha = np.cumprod(1 - np.linspace(d['beta_start'], d['beta_end'], d['steps'], dtype=np.float64))[timestep]
    xt = np.sqrt(alpha) * x0 + np.sqrt(1 - alpha) * epsilon
    return xt.astype(np.float32), epsilon, timestep


def auxiliary_loss(group, aux, bev, batch, epoch, config):
    xts, eps, times, weights, centers, owners = [], [], [], [], [], []
    counters = dict(windows=0, valid_windows=0, O=0, E=0, U=0, ego=0, peer=0, voxels=0)
    for i, (audit, targets) in enumerate(zip(batch['audits'], batch['auxiliary'])):
        for row in targets:
            lab = row['target']
            xt, epsilon, timestep = diffusion_inputs(lab['x0'], config['seed'], epoch, audit['uid'], row['object_index'], config)
            w = loss_weights(group, lab)
            xts.append(xt); eps.append(epsilon); times.append(timestep); weights.append(w)
            centers.append(row['center']); owners.append(i)
            counters['windows'] += 1
            counters['valid_windows'] += int(w.any())
            counters['voxels'] += w.size
            for key in ('O', 'E', 'U', 'ego', 'peer'):
                counters[key] += int(lab[key].sum())
    if not xts:
        return bev.sum() * 0, counters
    device = bev.device
    condition = crop_condition(bev, torch.as_tensor(owners, device=device), np.array(centers))
    pred = aux(torch.from_numpy(np.stack(xts)).to(device), torch.tensor(times, device=device), condition)
    target = torch.from_numpy(np.stack(eps)).to(device)
    weight = torch.from_numpy(np.stack(weights)).to(device)
    # Per-window spatial mean with registered class/source normalization, then all-window mean.
    # Empty windows contribute zero and still count in the denominator; detection sample retained.
    value = ((pred - target).square() * weight).flatten(1).sum(1).mean()
    if not torch.isfinite(value):
        raise FloatingPointError('Nonfinite auxiliary loss')
    return value, counters
