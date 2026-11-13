"""Run frozen F0 diagnostics with separate outputs and no training."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys

from paths import F0, LR0, OUT, WEATHERS, read, write

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v2xr_radar_ra_rb'))
import run as original
import numpy as np
import torch

from checkpoint import seed
from common import identity, lock, sha256
from occupancy import Grid, summarize, table
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from support import patched


def load():
    selected=read(F0/'selected.json')
    weight=Path(selected['checkpoint'])
    digest=sha256(weight)
    if (selected['epoch'],digest)!=(20,selected['sha256']):
        raise ValueError('F0 selected checkpoint differs')
    cfg=yaml_utils.load_yaml(str(F0/'config.yaml'))
    seed(20261016)
    model=original.make_model(cfg,read(F0/'CONTRACT.json')['settings']).eval()
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False),strict=True)
    return model,cfg,digest


def fixed(i, key):
    obj=read(LR0/'final_evaluation/predictions/clean'/f'{i:06d}.json')
    if obj['frame_key']!=list(key):
        raise ValueError('Fixed GT and dataset frame differ')
    return obj['official_gt']


def d1(model,grid,digest):
    results={}
    for w in WEATHERS:
        seed(20261016)
        dataset,_=original.build(F0/'config.yaml',condition=w,visualize=True)
        rows=[]
        gt_count=0
        for i,batch in enumerate(original.loader(dataset,evaluation=True)):
            ego=batch['ego']
            gt=fixed(i,dataset.keys[i])
            gt_count+=len(gt)
            foreground=grid.gt(gt)
            lidar,radar=(grid.raw(ego,s) for s in ('lidar','radar'))
            vox=ego['processed_lidar']
            dev=train_utils.to_device(vox,torch.device('cuda:0'))
            with torch.inference_mode():
                if len(dev['voxel_coords']):
                    values=model.gspr(dev['voxel_features'],dev['voxel_num_points'],dev['voxel_coords'])
                    mass=values['point_reliability'].sum(-1).cpu().numpy()
                else:
                    mass=np.empty(0)
            weighted=grid.voxel(ego,'lidar',mass)
            effective_radar=grid.voxel(ego,'radar')
            voxel_lidar=grid.voxel(ego,'lidar')
            rows.append(dict(frame=list(dataset.keys[i]), raw=table(lidar,radar,foreground).tolist(),
                retained=table(voxel_lidar,effective_radar,foreground).tolist(),
                weighted=table(weighted,effective_radar,foreground).tolist(),
                weighted_mass=np.bincount((weighted>0).astype(int)*2+(effective_radar<=0),
                    weights=weighted,minlength=4).tolist()))
            if (i+1)%100==0:
                print('D1',w,i+1,flush=True)
        if len(rows)!=829 or gt_count!=6235:
            raise ValueError('Validation or GT count differs')
        write(OUT/f'D1_{w}_frames.json',rows)
        groups=[r['frame'][0] for r in rows]
        results[w]={k:summarize([r[k] for r in rows],groups) for k in ('raw','retained','weighted')}
        results[w]['weighted_mass']=np.asarray([r['weighted_mass'] for r in rows]).sum(0).tolist()
        write(OUT/'D1.json',dict(checkpoint=digest, grid=[grid.nx,grid.ny],weather=results))
    fog=results['fog']['raw']
    p=fog['probability'][0]
    print('D1_COMPLETE',p,fog['scene_bootstrap_95ci'],flush=True)


@contextmanager
def oracle(model, grid, weather, stats):
    import metrics
    from opencood.tools import inference_utils

    native_build,infer=metrics.build,inference_utils.inference_intermediate_fusion
    cursor={}
    current={}
    def build(*args,**kwargs):
        dataset,cfg=native_build(*args,**kwargs)
        cursor.update(index=0,keys=dataset.keys)
        return dataset,cfg
    def observe(batch, network, dataset):
        i=cursor['index']
        cursor['index']+=1
        current['ego']=batch['ego']
        current['mask']=grid.gt(fixed(i,cursor['keys'][i]))
        try:
            return infer(batch,network,dataset)
        finally:
            current.clear()
    def inject(_, inputs):
        # Per-agent valid voxel coordinates share the same ego grid. Restrict edits
        # to the global LiDAR-empty / Radar-present GT cells from D1.
        spatial=inputs[0]['spatial_features']
        lidar=grid.voxel(current['ego'],'lidar')>0
        radar=grid.voxel(current['ego'],'radar')>0
        mask=torch.as_tensor(current['mask']&~lidar&radar,device=spatial.device)
        rf=spatial[:,64:].flatten(2)
        # Explicit per-agent occupancy, independent of whether learned features are zero.
        coords=current['ego']['processed_radar']['voxel_coords'].long()
        agents,_,ys,xs=coords.unbind(1)
        index=ys*grid.nx+xs
        take=mask[index]
        target,cell=agents[take],index[take]
        stats.append(dict(cells=int(mask.sum()),agent_cells=len(cell)))
        if len(cell):
            output=spatial.clone().flatten(2)
            output[target,:64,cell]=rf[target,:,cell]
            inputs[0]['spatial_features']=output.reshape_as(spatial)
    hook=model.backbone.register_forward_pre_hook(inject)
    try:
        with patched(metrics,'build',build), patched(inference_utils,'inference_intermediate_fusion',observe):
            yield
    finally:
        hook.remove()


def oracle_smoke(model,grid):
    seed(20261016)
    data,_=original.build(F0/'config.yaml',condition='fog',visualize=True)
    ego=train_utils.to_device(data.native.collate_batch_test([data[0]])['ego'],torch.device('cuda:0'))
    with torch.no_grad():
        first=model(ego)
        empty=dict(ego,processed_radar={k:v[:0] for k,v in ego['processed_radar'].items()})
        zero=model(empty)
        # Real path and checkpoint smoke, including empty Radar forward.
        if not all(torch.isfinite(v).all() for v in (*first.values(),*zero.values())):
            raise ValueError('Nonfinite F0 smoke')
    print('D2_SMOKE',len(ego['processed_radar']['voxel_coords']),flush=True)


def d2(model,grid,digest,w):
    oracle_smoke(model,grid)
    out=OUT/'D2'/w
    stats=[]
    seed(20261016)
    sources={p.name:sha256(p) for p in HERE.glob('*.py')}
    run_id=identity([digest,w,sources,'occupied gt direct radar copy'])
    with oracle(model,grid,w,stats):
        result=original.engine.evaluate(model,None,F0/'config.yaml',out,(w,) if w=='clean' else ('clean',w),
                                       829,run_id,digest)
    # Frozen replay requires a Clean cache for F. Non-Clean jobs build the same
    # oracle Clean condition, but primary F metrics use the frozen Clean GT.
    ref=read(LR0/'final_evaluation/gt_manifest.json')['frames']
    expected={(r['condition'],r['index']):r for r in ref}
    for row in read(out/'gt_manifest.json')['frames']:
        src=expected[(row['condition'],row['index'])]
        if any(row[k]!=src[k] for k in ('frame_key','fixed_ids','fixed_gt_identity')):
            raise ValueError('Oracle changed fixed ground truth')
    old=read(F0/'FINAL_RESULTS.json')['evaluation']['results'][w]['AP']
    new=result['results'][w]['AP']
    if stats:
        write(out/'injected.json',stats)
    stats=read(out/'injected.json')
    chosen=stats[-829:]
    write(OUT/f'D2_{w}.json',dict(checkpoint=digest,baseline=old,oracle=new,
        delta_pp={p:{k:100*(v-old[p][k]) for k,v in vals.items()} for p,vals in new.items()},
        injected_cells=sum(x['cells'] for x in chosen), frames_with_injection=sum(x['cells']>0 for x in chosen),
        counts=result['results'][w]['matches'],scope='GT-selected direct feature copy at occupied pillars'))
    print('D2_COMPLETE',w,100*(new['F']['ap70']-old['F']['ap70']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('job',choices=('D1',*WEATHERS))
    job=parser.parse_args().job
    if torch.cuda.device_count()!=1 or not torch.version.hip:
        raise RuntimeError('Exactly one visible ROCm device required')
    with lock(OUT/f'{job}.lock'):
        model,cfg,digest=load()
        grid=Grid(model,cfg)
        if job=='D1':
            d1(model,grid,digest)
        else:
            d2(model,grid,digest,job)

