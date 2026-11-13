"""Audit V2X-R's fourth ASCII field against independently estimated motion."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import open3d as o3d
import yaml
from scipy.stats import spearmanr


@lru_cache(maxsize=256)
def annotation(path):
    # Only construct motion fields. Unrelated trajectory arrays can have
    # NumPy-specific YAML tags that the safe loader deliberately rejects.
    class MotionLoader(yaml.SafeLoader):
        pass

    loader=MotionLoader(path.read_text())
    try:
        node=loader.get_single_node()
        wanted={'world','lidar_pose','radar_pose','vehicles'}
        return {key.value:loader.construct_object(value,deep=True)
                for key,value in node.value if key.value in wanted}
    finally:
        loader.dispose()


def pcd(path):
    with path.open() as stream:
        header = {}
        for line in stream:
            cells = line.strip().split()
            if cells and not cells[0].startswith('#'):
                header[cells[0]] = cells[1:]
            if cells and cells[0] == 'DATA':
                break
        if (header.get('FIELDS'), header.get('TYPE'), header.get('SIZE'), header.get('DATA')) != (
                ['x','y','z','rgb'], ['F']*4, ['4']*4, ['ascii']):
            raise ValueError(f'Unsupported PCD layout: {path}')
        data = np.loadtxt(stream, dtype=np.float32, ndmin=2)
    if data.shape != (int(header['POINTS'][0]),4) or not np.isfinite(data).all():
        raise ValueError(f'Unexpected payload: {path}')
    return data


def motion(path, data, dt, rotation):
    """Use a centered difference and unique, interior annotated object associations."""
    stem=path.name.removesuffix('_radar.pcd')
    seq=sorted(p for p in path.parent.glob('*.yaml') if re.fullmatch(r'\d+',p.stem))
    stamps={p.stem:i for i,p in enumerate(seq)}
    index=stamps[stem]
    if index==0 or index+1==len(seq):
        return []
    left,mid,right=(annotation(seq[i]) for i in (index-1,index,index+1))
    elapsed=(int(seq[index+1].stem)-int(seq[index-1].stem))*dt
    # Explicit time convention from the simulator step and numeric frame IDs.
    pose=rotation(mid['lidar_pose'])
    world=data[:,:3] @ pose[:3,:3].T + pose[:3,3]
    sensor=(np.asarray(right['radar_pose'][:3])-left['radar_pose'][:3])/elapsed
    line=world-np.asarray(mid['radar_pose'][:3])
    line/=np.linalg.norm(line,axis=1,keepdims=True).clip(1e-6)
    vehicles=mid['vehicles']
    hits=[]
    membership=np.zeros(len(world),int)
    for ident,actor in vehicles.items():
        transform=rotation([*actor['location'],*actor['angle']])
        local=(world-transform[:3,3]) @ transform[:3,:3]-actor['center']
        mask=np.all(np.abs(local)<=np.asarray(actor['extent'])+.05,axis=1)
        membership+=mask
        a,b=left['vehicles'].get(ident),right['vehicles'].get(ident)
        if mask.any() and a is not None and b is not None:
            target=(np.asarray(b['location'])-a['location'])/elapsed
            hits.append((ident,mask,target,actor.get('speed')))
    rows=[]
    for ident,mask,target,speed in hits:
        mask=mask&(membership==1)
        for scalar,vrel,vabs in zip(data[mask,3],line[mask]@(sensor-target),line[mask]@(-target)):
            rows.append((str(ident),float(scalar),float(vrel),float(vabs),float(speed) if speed is not None else None))
    return rows


def measures(rows):
    """Fit diagnostic linear scalings on train scenes, score on validation scenes."""
    train=[row for row in rows if row['split']=='train']
    val=[row for row in rows if row['split']=='validate']
    result={}
    for channel in ('relative_toward','absolute_toward'):
        for name,func in [('signed',lambda x:x),('magnitude',np.abs),
                          ('toward_only',lambda x:x.clip(0)),('away_only',lambda x:(-x).clip(0))]:
            def inputs(group):
                x=np.asarray([r[channel] for r in group])
                return func(x),np.asarray([r['scalar'] for r in group])
            x,y=inputs(train)
            vx,vy=inputs(val)
            if len(x)<2 or len(vx)<2 or not x.std() or not vx.std() or not vy.std():
                continue
            slope,offset=np.linalg.lstsq(np.column_stack((x,np.ones(len(x)))),y,rcond=None)[0]
            pred=(slope*vx+offset).clip(0,1)
            result[f'{channel}_{name}']=dict(train_n=len(y),val_n=len(vy),slope=float(slope),
                offset=float(offset),val_spearman=float(spearmanr(vx,vy).statistic),
                val_mae=float(abs(pred-vy).mean()))
    # A concrete falsifiable hypothesis. Do not fit the cutoff on validation.
    for split,group in [('train',train),('validate',val)]:
        toward=np.asarray([r['relative_toward'] for r in group])
        raw=np.asarray([r['scalar'] for r in group])
        away=-toward
        prediction=np.rint(255*away.clip(0,1))/255
        interior=(away>.05)&(away<.95)&(raw>0)&(raw<1)
        result[f'{split}_clip_away_0_1']=dict(n=len(raw),
            approaching_n=int((toward>=.2).sum()),
            approaching_zero_fraction=float(np.mean(raw[toward>=.2]==0)),
            receding_n=int((away>=1.2).sum()),
            receding_one_fraction=float(np.mean(raw[away>=1.2]==1)),
            interior_n=int(interior.sum()),
            interior_abs_mae=float(np.mean(abs(raw[interior]-away[interior]))),
            overall_mae=float(np.mean(abs(raw-prediction))),
            overall_within_0_05=float(np.mean(abs(raw-prediction)<=.05)))
    return result


def run(args):
    sys_path=args.runtime.resolve(strict=True)
    import sys
    sys.path.insert(0,str(sys_path))
    from opencood.utils.transformation_utils import x_to_world
    source=json.loads(args.audit.read_text())
    samples=[]
    rows=[]
    distribution=Counter()
    collisions=Counter()
    reader_max=0.
    grid_max=0.
    steps={}
    for r in source['samples']:
        path=Path(r['path'])
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=r['sha256']:
            raise ValueError(f'Archived sample changed: {path}')
        data=pcd(path)
        v=data[:,3]
        code=np.rint(v*255).astype(int)
        grid_max=max(grid_max,float(np.max(abs(v-code/255))))
        distribution.update(map(int,code))
        rgb=np.asarray(o3d.io.read_point_cloud(str(path)).colors)
        if rgb.shape!=(len(data),3):
            raise ValueError(f'Unexpected Open3D color array: {path}')
        red=np.rint(rgb[:,0]*255).astype(int)
        packed=v.copy().view(np.uint32)
        reader_max=max(reader_max,float(np.max(abs(rgb[:,0]-((packed>>16)&255)/255))))
        collisions.update(zip(map(int,code),map(int,red)))
        scene=(r['split'],r['scene'])
        if scene not in steps:
            p=args.raw/scene[0]/scene[1]/'data_protocol.yaml'
            if not p.is_file():
                raise FileNotFoundError(p)
            steps[scene]=float(annotation(p)['world']['fixed_delta_seconds'])
        # Move endpoint examples inward so both bracketing poses are available.
        seq=sorted(p for p in path.parent.glob('*_radar.pcd') if p.name.removesuffix('_radar.pcd').isdigit())
        if len(seq)<3:
            continue
        target=seq[min(max(seq.index(path),1),len(seq)-2)]
        sample_key=(scene,target)
        if sample_key in samples:
            continue
        if len(samples)%50==0:
            print(f'Checking motion {len(samples)}/{len(source["samples"])}',flush=True)
        samples.append(sample_key)
        for ident,scalar,relative,absolute,speed in motion(target,pcd(target),steps[scene],x_to_world):
            rows.append(dict(split=scene[0],scene=scene[1],file=str(target),target=ident,scalar=scalar,
                             relative_toward=relative,absolute_toward=absolute,annotated_speed=speed))
    lut={}
    for (value,red),n in sorted(collisions.items()):
        lut.setdefault(red,[]).append(dict(original_code=value,points=n))
    result=dict(samples=len(source['samples']),points=sum(distribution.values()),unique_raw=len(distribution),
        raw_code_counts=dict(sorted(distribution.items())),raw_code_max_error=grid_max,
        loader_red_bitcast_max_error=reader_max,red_code_count=len(lut),
        red_codes_with_collisions=sum(len(x)>1 for x in lut.values()),red_to_original_codes=lut,
        motion_samples=len(samples),motion_rows=rows,
        motion_counts=dict(Counter(r['split'] for r in rows)),
        motion_diagnostics=measures(rows),
        meaning='published normalized scalar; physical mapping not certified by exporter',
        time_basis='numeric CARLA tick differences times scene fixed_delta_seconds')
    args.output.mkdir(parents=True,exist_ok=True)
    out=args.output/'FOURTH_CHANNEL.json'
    tmp=out.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(out)
    print(json.dumps({k:v for k,v in result.items() if k not in ('motion_rows','raw_code_counts','red_to_original_codes')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

