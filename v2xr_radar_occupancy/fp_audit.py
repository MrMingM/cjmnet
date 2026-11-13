"""Identify radar-on-only false positives and inspect local LiDAR occupancy."""
import json

import numpy as np
from shapely.geometry import Polygon

from paths import F0,F3,OUT,read,write
from driver import original
from checkpoint import seed
from occupancy import Grid
from opencood.hypes_yaml import yaml_utils
from types import SimpleNamespace


def cached(mode):
    folder=F3/mode
    report=read(folder/'DIAGNOSTIC.json')
    with (folder/'per_frame_metrics.jsonl').open() as f:
        rows=[r for line in f if (r:=json.loads(line))['condition']=='fog']
    if [r['index'] for r in rows]!=list(range(120)):
        raise ValueError('Expected archived Fog prefix')
    result=[]
    for row in rows:
        idx=row['index']
        pred=read(folder/'predictions/fog'/f'{idx:06d}.json')
        if row['frame_key']!=pred['frame_key']:
            raise ValueError('Archived frame mismatch')
        pairs=row['protocols']['F'].get('matched_pairs',{}).get('0.7',[])
        hits={p['prediction_index'] for p in pairs}
        boxes=np.asarray(pred['prediction_boxes'] or []).reshape(-1,8,3)
        fps=[dict(index=i,box=b,score=pred['scores'][i]) for i,b in enumerate(boxes) if i not in hits]
        if len(fps)!=row['protocols']['F']['thresholds']['0.7']['fp']:
            raise ValueError('Archived false positive count differs')
        result.append((row['frame_key'],fps))
    return report['config']['sha256'],result


def match(on,off,threshold):
    polys=[[Polygon(p['box'][:4,:2]) for p in s] for s in (on,off)]
    edges=[]
    for i,a in enumerate(polys[0]):
        for j,b in enumerate(polys[1]):
            if a.is_valid and b.is_valid and a.intersects(b):
                union=a.union(b).area
                overlap=a.intersection(b).area/union if union else 0
                if overlap>=threshold:
                    edges.append((-overlap,i,j))
    used_on,used_off=set(),set()
    for _,i,j in sorted(edges):
        if i not in used_on and j not in used_off:
            used_on.add(i)
            used_off.add(j)
    return used_on,used_off


def run():
    h,on=cached('full')
    j,off=cached('radar_off')
    if h!=j or any(a[0]!=b[0] for a,b in zip(on,off)):
        raise ValueError('F3 cache weight or frame mismatch')
    candidates=[]
    shared=0
    for i,((_,a),(_,b)) in enumerate(zip(on,off)):
        paired,_=match(a,b,.5)
        shared+=len(paired)
        candidates.extend((i,p) for k,p in enumerate(a) if k not in paired)
    rng=np.random.default_rng(20261113)
    if len(candidates)<50:
        raise ValueError('Fewer than 50 unmatched radar-on FP')
    sample=[candidates[i] for i in sorted(rng.choice(len(candidates),50,replace=False))]
    cfg=yaml_utils.load_yaml(str(F0/'config.yaml'))
    nx,ny,_=cfg['model']['args']['point_pillar_scatter']['grid_size']
    grid=Grid(SimpleNamespace(scatter=SimpleNamespace(nx=int(nx),ny=int(ny))),cfg)
    data,_=original.build(F0/'config.yaml',condition='fog',visualize=True)
    seed(20261016)
    evidence=[]
    for i in sorted({i for i,_ in sample}):
        if on[i][0]!=list(data.keys[i]):
            raise ValueError('Archived and reloaded dataset keys differ')
        ego=data.native.collate_batch_test([data[i]])['ego']
        lidar=grid.raw(ego,'lidar').reshape(grid.ny,grid.nx)
        radar=grid.raw(ego,'radar').reshape(grid.ny,grid.nx)
        for frame,p in sample:
            if frame!=i:
                continue
            center=p['box'].mean(0)
            idx,valid=grid.cell(center[None,:2])
            cell=None
            if valid[0]:
                y,x=divmod(int(idx[0]),grid.nx)
                hood=np.s_[max(y-1,0):y+2,max(x-1,0):x+2]
                cell=dict(lidar=int(lidar[y,x]),radar=int(radar[y,x]),lidar_3x3=int(lidar[hood].sum()))
            evidence.append(dict(frame=i,key=on[i][0],index=p['index'],score=p['score'],
                                 center=center.tolist(),cell=cell))
    counts=lambda k:dict(zero=sum(r['cell'] is not None and r['cell'][k]==0 for r in evidence),
                         occupied=sum(r['cell'] is not None and r['cell'][k]>0 for r in evidence),
                         out_of_grid=sum(r['cell'] is None for r in evidence))
    total=lambda rows:sum(len(p) for _,p in rows)
    write(OUT/'D3.json',dict(checkpoint=h,fp_on=total(on),fp_off=total(off),matched_fp=shared,
        on_only=len(candidates),off_only=total(off)-shared,sample_seed=20261113,matching_bev_iou=.5,
        center=counts('lidar'),neighborhood_3x3=counts('lidar_3x3'),samples=evidence,
        note='Geometric association; occupancy groups do not identify causal mechanism.'))
    print('D3_COMPLETE',counts('lidar'),counts('lidar_3x3'),flush=True)


if __name__=='__main__':
    run()

