"""Spatially stratify perturbations and preserve cohort tensors at sampled cells."""
from pathlib import Path
import os
import numpy as np
import shapely
import torch
from diag_data import grid_xy,distance


def measure(cfg,metrics,raw,targets,known,artifact):
    groups, by_target, arrays = {}, {}, {}
    for scale,comparisons in metrics.items():
        if not comparisons:
            continue
        delta,_ = next(iter(comparisons.values()))
        stride=1 if scale==-1 else 2**(scale+1)
        xy,radius=grid_xy(cfg,delta.shape[-2:],stride)
        flat=xy.reshape(-1,2)
        occupied=np.zeros(len(flat),dtype=bool)
        ids={}
        for target in targets:
            oid=str(target['object_id'])
            polygon=np.asarray(target['corners'])[:4,:2]
            near=np.flatnonzero(((flat>=polygon.min(0)-radius)&(flat<=polygon.max(0)+radius)).all(1))
            chosen=near[shapely.distance(shapely.points(flat[near]),shapely.Polygon(polygon))<=radius+1e-5]
            occupied[chosen]=True
            ids[oid]=chosen
        group_masks={'all':np.arange(len(flat)), 'target':np.flatnonzero(occupied),
                     'background':np.flatnonzero(~occupied)}
        ranges=np.linalg.norm(flat,axis=1)
        for label,mask in (('0-30',ranges<30),('30-60',(ranges>=30)&(ranges<60)),('60+',ranges>=60)):
            group_masks['target/'+label]=np.flatnonzero(occupied&mask)
            group_masks['background/'+label]=np.flatnonzero(~occupied&mask)
        values={}
        for name,(diff,ref) in comparisons.items():
            if not len(diff):
                values[name]=np.zeros((2,len(flat)),dtype=float)
                continue
            # Mean over vehicles and channels gives comparable per-cell RMS.
            values[name]=torch.stack((diff.float().square().mean((0,1)),ref.float().square().mean((0,1)))).flatten(1).cpu().numpy()
        def reduce(chosen):
            return {name:[float(value[0,chosen].sum(dtype=np.float64)),float(value[1,chosen].sum(dtype=np.float64)),len(chosen)]
                    for name,value in values.items()}
        groups[str(scale)]={name:reduce(chosen) for name,chosen in group_masks.items()}
        for target in targets:
            oid=str(target['object_id'])
            by_target.setdefault(oid,dict(distance=distance(target),cohort=oid in known,scales={}))
            by_target[oid]['scales'][str(scale)]=reduce(ids[oid])
        if known and raw.get(scale):
            selected=np.unique(np.concatenate([ids[k] for k in known]))
            bg=group_masks['background']
            if len(bg):
                selected=np.unique(np.r_[selected,bg[np.linspace(0,len(bg)-1,min(24,len(bg)),dtype=int)]])
            arrays[f'{scale}_indices']=selected
            arrays[f'{scale}_xy_m']=flat[selected]
            arrays[f'{scale}_background']=~occupied[selected]
            for oid in known:
                arrays[f'{scale}_target_{oid}_indices']=ids[oid]
            for name,tensor in raw[scale].items():
                sampled=tensor[selected] if name.startswith('attention') else tensor.flatten(2)[:,:,selected]
                arrays[f'{scale}_{name}']=sampled.cpu().numpy()
    if arrays:
        artifact.parent.mkdir(parents=True,exist_ok=True)
        temporary=artifact.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,**arrays)
        os.replace(temporary,artifact)
    return groups,by_target
