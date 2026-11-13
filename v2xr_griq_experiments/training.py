"""Small CPU quality heads and training-only scene-safe comparison utilities."""
import os
import time

import numpy as np
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

from v2xr_griq_experiments.dataset import SEED, frame_rows, load
from v2xr_griq_experiments.features import BASE, GSPR, RADAR, FEATURES, indices
from v2xr_griq_experiments.metrics import binary

EPOCHS = 16


class Quality(nn.Module):
    def __init__(self, inputs, kind, radar_inputs=0):
        super().__init__()
        self.kind = kind
        if kind == 'additive':
            self.split = (inputs-radar_inputs)//2
            self.base = nn.Sequential(nn.Linear(inputs-radar_inputs,48),nn.ReLU(),nn.Linear(48,24),nn.ReLU(),nn.Linear(24,1))
            self.radar = nn.Sequential(nn.Linear(radar_inputs,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
        elif kind == 'logistic':
            self.net = nn.Linear(inputs, 1)
        else:
            self.net = nn.Sequential(nn.Linear(inputs,64), nn.ReLU(), nn.Linear(64,32),
                                     nn.ReLU(), nn.Linear(32,2 if kind == 'dual' else 1))

    def forward(self, x):
        if self.kind == 'additive':
            k = x.shape[1]//2
            return self.base(torch.cat([x[:,:self.split],x[:,k:k+self.split]],1)) + self.radar(
                torch.cat([x[:,self.split:k],x[:,k+self.split:]],1))
        return self.net(x)

    def logits(self, x):
        v = self(x)
        if self.kind == 'dual':
            p = v.sigmoid().prod(1)
            return torch.logit(p.clamp(1e-7,1-1e-7))
        return v[:,0]


def prepare(index):
    groups = {name: [] for name in index['split']}
    for row in frame_rows(index, 'train'):
        frame = load(row, index['policy'])
        for name, scenes in index['split'].items():
            if row['scene'] in scenes:
                groups[name].append(frame)
    return groups


def array(frames):
    if not frames:
        raise ValueError('No frames in training partition')
    x = np.concatenate([d['x'] for d in frames])
    y = np.concatenate([d['iou'] for d in frames])
    return x, y


def pair_indices(frames, hard, rng):
    pairs = []
    offset = 0
    for d in frames:
        x, iou = d['x'], d['iou']
        pos, neg = np.flatnonzero(iou >= .7), np.flatnonzero(iou < .7)
        if len(neg) and len(pos):
            for i in rng.choice(pos, min(32, len(pos)), replace=False):
                # Keep a fixed ordinary pair fraction in the hard sampling arm.
                if hard and rng.random() < .75:
                    costs = (np.abs(x[neg,1]-x[i,1])/.1 + np.abs(x[neg,2]-x[i,2])/10)
                    r = x[neg, FEATURES.index('r_mean')]
                    radar = x[neg, FEATURES.index('radar_count')]
                    costs -= .25 * ((r < .5) & (radar > 0))
                    xy = np.linalg.norm(d['decoded'][neg,:2]-d['decoded'][i,:2], axis=1)
                    costs -= .5 * (xy < 3)
                    j = neg[costs.argmin()]
                else:
                    j = rng.choice(neg)
                pairs.append((offset+i, offset+j))
        offset += len(iou)
    return np.asarray(pairs, dtype=np.int64).reshape(-1,2)


def normalize(x, cols, mu, scale):
    sub = x[:,cols].copy()
    missing = np.isnan(sub)
    sub = np.where(missing, mu, sub)
    return torch.from_numpy(np.concatenate([(sub-mu)/scale, missing.astype(np.float32)], axis=1).astype(np.float32))


def shuffled(x, seed):
    result = x.copy()
    r = indices(RADAR)
    # Preserve the range-dependent statistics while destroying association.
    rng = np.random.default_rng(seed)
    for k in np.unique(np.floor(x[:,2]/15).astype(int)):
        ids = np.flatnonzero(np.floor(x[:,2]/15).astype(int) == k)
        result[np.ix_(ids,r)] = x[np.ix_(rng.permutation(ids),r)]
    return result


def fit(train, selection, fields, kind='single', pair=None, weight=0., smoke=False, shuffle=False):
    train_x, train_iou = array(train)
    dev_x, dev_iou = array(selection)
    if shuffle:
        train_x, dev_x = shuffled(train_x, SEED), shuffled(dev_x, SEED+1)
    cols = indices(fields)
    sub = train_x[:,cols]
    finite = np.isfinite(sub)
    mu = np.where(finite, sub, 0).sum(0) / np.maximum(1, finite.sum(0))
    scale = np.sqrt(np.where(finite, (sub-mu)**2, 0).sum(0) / np.maximum(1, finite.sum(0))).clip(.01)
    a, b = normalize(train_x, cols, mu, scale), normalize(dev_x, cols, mu, scale)
    y = torch.from_numpy(np.stack([train_iou>=.7,train_iou>=.3], axis=1).astype(np.float32))
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    net = Quality(a.shape[1], kind, 2*len(RADAR))
    initial = [p.detach().clone() for p in net.parameters()]
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
    best, history = None, []
    for epoch in range(1, (1 if smoke else EPOCHS)+1):
        pairs = pair_indices(train, pair=='hard', rng) if pair else np.empty((0,2), dtype=np.int64)
        if pair and not len(pairs):
            raise ValueError('No within-frame positive/negative training pairs')
        net.train()
        loss_sum = np.zeros(3)
        batches = 0
        for j in np.array_split(rng.permutation(len(a)), max(1, (len(a)+1023)//1024)):
            pred = net(a[j])
            rank = torch.zeros(())
            if kind == 'dual':
                obj = F.binary_cross_entropy_with_logits(pred[:,0], y[j,1])
                valid = y[j,1].bool()
                loc = F.binary_cross_entropy_with_logits(pred[valid,1], y[j[valid.numpy()],0]) if valid.any() else torch.zeros(())
                loss = obj + weight * loc
            else:
                loss = F.binary_cross_entropy_with_logits(pred[:,0], y[j,0])
                if pair:
                    p = pairs[rng.integers(len(pairs), size=min(256,len(pairs)))]
                    rank = F.softplus(-(net.logits(a[p[:,0]])-net.logits(a[p[:,1]]))).mean()
            total = loss + (weight * rank if pair else 0)
            if not torch.isfinite(total):
                raise ValueError('Nonfinite quality loss')
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            loss_sum += [total.item(), loss.item(), rank.item()]
            batches += 1
        net.eval()
        with torch.no_grad():
            p = torch.cat([net.logits(chunk).sigmoid() for chunk in b.split(4096)]).numpy()
        metric = binary(dev_iou >= .7, p)
        history.append(dict(epoch=epoch, losses=(loss_sum/batches).tolist(), selection=metric, pairs=len(pairs)))
        if best is None or metric['logloss'] < best['criterion']:
            best = dict(state={k:v.detach().clone() for k,v in net.state_dict().items()},
                        epoch=epoch, criterion=metric['logloss'])
    if not any(not torch.equal(a, b) for a,b in zip(initial,net.parameters())):
        raise ValueError('Scorer parameters did not update')
    best.update(kind=kind, fields=list(fields), cols=cols, mean=mu.tolist(), scale=scale.tolist(),
                history=history, weight=weight, pair=pair, shuffle=shuffle,
                parameters=sum(p.numel() for p in net.parameters()))
    return best


class Scorer:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = Quality(2*len(checkpoint['cols']),checkpoint['kind'],2*len(RADAR))
        self.model.load_state_dict(checkpoint['state'])
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, x, split_seed=None):
        ck = self.checkpoint
        if ck['shuffle']:
            x = shuffled(x, SEED+2 if split_seed is None else split_seed)
        z = normalize(x, ck['cols'], np.asarray(ck['mean']), np.asarray(ck['scale']))
        if ck['kind'] == 'dual':
            heads = self.model(z).sigmoid().numpy()
            return np.asarray(torch.logit(torch.from_numpy(heads.prod(1)).clamp(1e-7,1-1e-7))), heads
        return self.model.logits(z).numpy(), None


def tune(train, dev, fields, kind='single', pair=None, weights=(0.,), smoke=False):
    trials = [fit(train,dev,fields,kind,pair,w,smoke) for w in (weights[:1] if smoke else weights)]
    best = min(trials, key=lambda x:x['criterion'])
    return best | dict(trials=[dict(weight=t['weight'], history=t['history']) for t in trials])
