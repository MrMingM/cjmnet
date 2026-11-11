"""Replay official loaders and build audited offline paired interventions."""
import numpy as np
import shapely
import torch

from diag_bootstrap import pot, read_json
from p0_failure import archived_loader, voxel_masks
from runtime import build


def samples(weather, limit):
    a,cfg = build(pot.CONFIG,condition=weather,visualize=True)
    b = a if weather=='clean' else build(pot.CONFIG,condition='clean',visualize=True)[0]
    if a.keys!=b.keys:
        raise ValueError('Weather pairing keys differ')
    first = archived_loader(a,weather)
    if weather=='clean':
        pairs=((x,x) for x in first)
    else:
        pairs=zip(first,archived_loader(b,'clean'),strict=True)
    # Iterating original positions reproduces worker seeds and point shuffling.
    from itertools import islice
    for index,(original,clean) in enumerate(islice(pairs,limit)):
        base=pot.LR0/'final_evaluation/predictions'
        fixed=read_json(base/'clean'/f'{index:06d}.json')
        live=fixed if weather=='clean' else read_json(base/weather/f'{index:06d}.json')
        if fixed['frame_key']!=list(a.keys[index]) or live['frame_key']!=fixed['frame_key']:
            raise ValueError('Archived GT frame identity differs')
        yield index,original['ego'],clean['ego'],cfg,a.native,fixed,live


def regions(fixed, known_ids):
    targets=fixed['official_gt']
    by_id={str(t['object_id']):t for t in targets}
    if not set(known_ids)<=by_id.keys():
        raise ValueError('Historical cohort absent from fixed GT')
    # Use exact historical union on cohort frames. Else protect one fixed target per range.
    selected=[by_id[k] for k in known_ids]
    if not selected:
        strata={}
        for target in targets:
            strata.setdefault(distance(target),target)
        selected=list(strata.values())
    return targets,selected


def distance(target):
    metres=np.linalg.norm(target['box'][:2])
    return '0-30' if metres<30 else '30-60' if metres<60 else '60+'


def splice(ego, clean, targets, cfg, scope, weather):
    if (not torch.equal(ego['record_len'],clean['record_len'])
            or ego['ego_flag']!=clean['ego_flag'] or not np.array_equal(ego['lidar_pose'],clean['lidar_pose'])):
        raise ValueError('Paired CAV layout differs')
    if weather=='clean' or not targets:
        return ego,dict(removed=0,added=0,protected=0,scope=scope)
    low=np.asarray(cfg['preprocess']['cav_lidar_range'][:2])
    spacing=np.asarray(cfg['preprocess']['args']['voxel_size'][:2])
    a,b=(v['processed_lidar'] for v in (ego,clean))

    def masks(data):
        coords=data['voxel_coords'].numpy()
        protect,near=voxel_masks(coords,low,spacing,targets)
        applies=np.ones(len(coords),dtype=bool)
        if scope!='both':
            applies=(coords[:,0]==0) if scope=='ego' else (coords[:,0]!=0)
        return protect,near&~protect&applies
    protected,remove=masks(a)
    _,add=masks(b)
    mixed={k:torch.cat((a[k][~remove],b[k][add])) for k in a}
    protected_after,mutable=masks(mixed)
    if not all(torch.equal(mixed[k][~mutable],a[k][~remove])
               and torch.equal(mixed[k][protected_after],a[k][protected]) for k in a):
        raise ValueError('Protected target or fixed vehicle pillars changed')
    if len(torch.unique(mixed['voxel_coords'],dim=0))!=len(mixed['voxel_coords']):
        raise ValueError('Duplicate pillar in context swap')
    return dict(ego,processed_lidar=mixed),dict(removed=int(remove.sum()),added=int(add.sum()),
                                               protected=int(protected.sum()),scope=scope)


def grid_xy(cfg, shape, stride):
    h,w=shape
    low=np.asarray(cfg['preprocess']['cav_lidar_range'][:2])
    v=np.asarray(cfg['preprocess']['args']['voxel_size'][:2])
    x,y=np.meshgrid(low[0]+(np.arange(w)*stride+.5)*v[0],
                    low[1]+(np.arange(h)*stride+.5)*v[1])
    return np.stack((x,y),axis=-1),float(np.linalg.norm(v)*stride/2)
