"""Reuse frozen historical geometry; extend training frames and attach local observations."""
import os
import sys
from collections import defaultdict
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from v2xr_griq_v2.dataset import (DATA,F0,OUT,V1,WEIGHT,WEATHERS,bind,get,import_runtime,nested,read,
                                 root_for,row_path,sha)
from v2xr_griq_v2.local_evidence import DL,extract_local
from v2xr_griq_experiments.dataset import pool_mask
from v2xr_griq_experiments.features import check_voxel_alignment,extract
from v2xr_griq_experiments.native import replay
from v2xr_griq_experiments.prepare_candidates import check_predictions,check_weather_gt


class FrameSubset:
    """Point shuffling must not depend on worker scheduling or skipped cached frames."""
    def __init__(self,data,indices,weather):
        self.data,self.indices,self.weather=data,indices,weather

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,i):
        frame=self.indices[i]
        seed=20261016+frame+100000*WEATHERS.index(self.weather)
        random.seed(seed)
        np.random.seed(seed)
        return self.data[frame]


def save_npz(path, **kw):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**kw)
    tmp.replace(path)


def plan(old,smoke):
    scenes = read(DATA/'v2xr-official-train-v2/manifest.json')['scenes']
    available = defaultdict(list)
    for i,key in enumerate((r['scene'],t) for r in scenes for t in r['timestamps']):
        available[key[0]].append(i)
    s32 = defaultdict(list)
    for row in old['files']:
        if row['split']=='train' and row['weather']=='clean':
            s32[row['scene']].append(row['index'])
    if len(s32)!=32 or set(s32)!=set(available):
        raise ValueError('Historical scene set does not match train index')
    s128 = {s:nested(available[s],s32[s]) for s in s32}
    if smoke:
        selected = [old['split'][k][0] for k in ('fit','select','calibrate')]
        available = {s:sorted(set(s32[s][:2]+[next(i for i in s128[s] if i not in s32[s])])) for s in selected}
        s32 = {s:s32[s][:2] for s in selected}
        s128 = {s:available[s] for s in selected}
    return dict(s32=dict(s32),s128=s128,all=dict(available))


def row_identity(row,source):
    for field,h in [('npz','npz_sha256'),('meta','meta_sha256')]:
        if sha(row[field])!=row[h]:
            raise ValueError(f'Candidate cache hash mismatch: {row[field]}')
    m = read(row['meta'])
    if m['index']!=row['index'] or m['key'][0]!=row['scene'] or m['weather']!=row['weather'] or m['split']!=row['split']:
        raise ValueError(f'Candidate cache identity mismatch: {row["meta"]}')
    if source and m['contract'] not in source:
        raise ValueError(f'Candidate has unknown historical contract: {row["meta"]}')
    return m


def new_frame(batch, device_batch, model, pp, key, w, idx, fixed, policy, root):
    from collect import dev_gt,feature_context,inside,inspect_head,iou_matrix,native_post
    from diagnostics import input_context
    ego = device_batch['ego']
    out = model(ego)
    prob,decoded,corners = inspect_head(ego,out,pp)
    p,d,c = [v.cpu().numpy() for v in (prob,decoded,corners)]
    z = out['psm'].permute(0,2,3,1).reshape(-1).cpu().numpy()
    t = ego['transformation_matrix'].cpu().numpy()
    anchors = ego['anchor_box'].reshape(-1,7).cpu().numpy()
    if len(anchors)!=len(p) or not np.allclose(t,np.eye(4),atol=1e-6) or not np.isfinite(p).all():
        raise ValueError('Unexpected native head or candidate coordinate system')
    native_threshold = float(pp.params['target_args']['score_threshold'])
    rank = np.empty(len(p),np.int32)
    rank[np.argsort(-p,kind='stable')] = np.arange(len(p))
    ids = np.flatnonzero(pool_mask(p,rank,native_threshold,policy['floor'],policy['cap']))
    xctx = feature_context(batch['ego'],model)
    check_voxel_alignment(batch['ego'],xctx)
    x = extract(d[ids],p[ids],z[ids],xctx,inside)
    ogt = dev_gt(pp,device_batch)
    fgt = ogt if w=='clean' else fixed
    extra = check_weather_gt(fgt,ogt,'train',key)
    overlaps = iou_matrix(c[ids],[g['corners'] for g in fgt])
    best = overlaps.argmax(1) if len(fgt) else np.full(len(ids),-1)
    iou = overlaps.max(1) if len(fgt) else np.zeros(len(ids))
    best[iou==0] = -1
    a,b,_ = native_post(device_batch,out,pp,native_threshold)
    a2,b2,stage,final = replay(pp,d[ids],z[ids],t,native_threshold,trace=True)
    error = check_predictions(a2,b2,None if a is None else a.cpu().numpy(),
                              None if b is None else b.cpu().numpy(),key)
    path = row_path(root,'extra',w,idx)
    save_npz(path,ids=ids,x=x,decoded=d[ids],corners=c[ids],prob=p[ids],logit=z[ids],anchor=anchors[ids],
             stage=stage,y30=iou>=.3,y50=iou>=.5,y70=iou>=.7,raw_rank=rank[ids],iou=iou,gt_index=best,
             transform=t,native_threshold=np.float32(native_threshold),final_ids=ids[final])
    meta = path.with_suffix('.json')
    bind(meta,dict(key=key,weather=w,split='train',index=idx,contract=sha(OUT/'CONTRACT.json'),
                   fgt=fgt,ogt=ogt,observations=input_context(batch,fgt)['objects'],
                   weather_only_gt=extra,replay_error=error))
    row = dict(scene=key[0],index=idx,weather=w,split='train',npz=str(path),meta=str(meta),
               npz_sha256=sha(path),meta_sha256=sha(meta))
    bind(path.with_suffix('.done.json'),row)
    return row,xctx


def gspr_only(batch, model, device):
    # Matches LR1._before_lidar exactly on the frozen native voxels.
    from collect import feature_context
    vox = batch['ego']['processed_lidar']
    model._point_outputs = None
    if len(vox['voxel_features']):
        model._point_outputs = model.gspr(*(vox[k].to(device) for k in
                                             ('voxel_features','voxel_num_points','voxel_coords')))
    return feature_context(batch['ego'],model)


def prepare(area,smoke):
    run = import_runtime()
    from context import load
    from collect import dev_data,inside,feature_context
    from checkpoint import seed
    from opencood.tools.train_utils import to_device
    from v2xr_weather_l012.data import tensor_digest

    old = read(V1/'DATA_AUDIT.json')
    root = root_for(smoke)
    contract = sha(OUT/'CONTRACT.json')
    dest = root/('FULL_DATA.json' if area=='full' else 'DATA_AUDIT.json')
    if dest.exists():
        done = read(dest)
        if done['contract']!=contract:
            raise ValueError('Committed data stage contract differs')
        for row in done['files']:
            row_identity(row,None)
            if 'local' in row and sha(row['local'])!=row['local_sha256']:
                raise ValueError('Committed local cache changed')
        return
    source_contracts = {old['contract'],sha(V1/'CONTRACT.v1.json')}
    sampling = plan(old,smoke)
    policy = old['policy']
    fit = old['split']['fit']
    historic = {(r['split'],r['weather'],r['index']):r for r in old['files']}
    shared = read(root/'DATA_AUDIT.json') if area=='full' else None
    if shared:
        for r in shared['files']:
            historic[(r['split'],r['weather'],r['index'])] = r
    device = torch.device('cuda:0')
    if not torch.version.hip or torch.cuda.device_count()!=1 or torch.cuda.mem_get_info()[0] < 6*2**30:
        raise RuntimeError('Need one visible ROCm GPU with 6 GiB free')
    model,config,digest = load(run,'F0')
    model.eval().requires_grad_(False)
    if digest!=WEIGHT:
        raise ValueError('F0 changed')
    state = tensor_digest(model.state_dict())
    rows = []
    buckets = [('train',w) for w in ('clean','fog')]
    if area=='shared':
        buckets += [('validate',w) for w in WEATHERS]
    clean = {}
    checked = 0
    for split,w in buckets:
        seed(20261016)
        data,_ = dev_data(config,w) if split=='train' else run.build(config,condition=w,visualize=True)
        if not data.native.proj_first:
            raise ValueError('Expected projections into the ego frame')
        if split=='validate':
            ids = list(range(3 if smoke else len(data)))
        elif area=='full':
            ids = sorted(i for s in fit for i in sampling['all'].get(s,[]))
        else:
            ids = sorted(i for s in sampling['s32'] for i in
                         (sampling['s128'][s] if s in fit else sampling['s32'][s]))
        selected = {}
        for i in ids:
            row = historic.get((split,w,i))
            path = row_path(root,'extra',w,i)
            if row is None and split=='train' and path.with_suffix('.done.json').exists():
                row = read(path.with_suffix('.done.json'))
            if row is not None:
                meta = row_identity(row,source_contracts | {contract})
                if meta['key']!=data.keys[i]:
                    raise ValueError('Dataset key differs from candidate cache')
                selected[i] = row
                if split=='train' and w=='clean':
                    clean[i] = meta['fgt']
        local_required = area=='shared'
        pending = []
        for i in ids:
            path = row_path(root,'local_'+split,w,i)
            marker = path.with_suffix('.json')
            if local_required and marker.exists() and i in selected:
                m = read(marker)
                if m['source']!=selected[i]['npz_sha256'] or m['contract']!=contract or sha(path)!=m['sha256']:
                    raise ValueError('Local sidecar identity differs')
                selected[i] = selected[i] | dict(local=str(path),local_sha256=m['sha256'])
            if i not in selected or (local_required and 'local' not in selected[i]):
                pending.append(i)
        print('PREPARE',area,split,w,'total',len(ids),'reads',len(pending),
              'new_detector_frames',sum(i not in selected for i in pending),flush=True)
        loader = DataLoader(FrameSubset(data,pending,w),batch_size=1,num_workers=8,
                            collate_fn=data.native.collate_batch_test)
        with torch.inference_mode():
            for count,(i,batch) in enumerate(zip(pending,loader),1):
                key = data.keys[i]
                ctx = None
                if i not in selected:
                    selected[i],ctx = new_frame(batch,to_device(batch,device),model,data.native.post_processor,
                                                 key,w,i,clean.get(i),policy,root)
                    if w=='clean':
                        clean[i] = read(selected[i]['meta'])['fgt']
                if local_required:
                    row = selected[i]
                    d = get(row,policy)
                    if ctx is None:
                        ctx = gspr_only(batch,model,device)
                    check_voxel_alignment(batch['ego'],ctx)
                    if smoke and checked==0:
                        model(to_device(batch,device)['ego'])
                        gold = feature_context(batch['ego'],model)
                        checks = [(a.shape,b.shape,float(np.max(np.abs(a-b))) if a.shape==b.shape and a.size else None)
                                  for a,b in zip(ctx[1:],gold[1:])]
                        print('GSPR REPLAY',checks,flush=True)
                        if (not np.array_equal(ctx[1],gold[1]) or not np.array_equal(ctx[4],gold[4]) or
                                any(a.shape!=b.shape or not np.allclose(a,b,atol=1e-6,rtol=1e-6)
                                    for a,b in zip(ctx[2:4],gold[2:4]))):
                            raise ValueError(f'Isolated GSPR differs from frozen full detector: {checks}')
                        checked += 1
                        if ctx is not None:
                            ctx = gold
                    a = extract_local(d['decoded'],ctx,inside)
                    if (not np.array_equal(a[:,:,0].sum(1),d['x'][:,8]) or
                            not np.array_equal(a[:,:,DL].sum(1),d['x'][:,21])):
                        raise ValueError(f'Local point totals differ from archived global counts: {split}/{w}/{key}')
                    path = row_path(root,'local_'+split,w,i)
                    save_npz(path,ids=d['ids'],evidence=a)
                    marker = dict(source=row['npz_sha256'],contract=contract,sha256=sha(path))
                    bind(path.with_suffix('.json'),marker)
                    selected[i] = row | dict(local=str(path),local_sha256=marker['sha256'])
                if count%100==0:
                    print('FRAMES',split,w,count,len(pending),flush=True)
        if set(selected)!=set(ids):
            raise ValueError(f'Incomplete shared candidates {split}/{w}')
        rows.extend(selected[i] for i in ids)
    if tensor_digest(model.state_dict())!=state or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen detector changed')
    if area=='shared':
        if not smoke and any(sum(r['split']=='validate' and r['weather']==w for r in rows)!=829 for w in WEATHERS):
            raise ValueError('Formal frame count differs')
        bind(root/'BASELINE_REPLAY.json',dict(source=sha(V1/'BASELINE_REPLAY.json'),weight=WEIGHT,
             state=state, local_alignment='native proj_first, point totals and anchor IDs verified'))
    if shared:
        rows.extend(r for r in shared['files'] if r['split']=='train' and r['scene'] not in fit)
    counts = {s:{w:sum(r['split']==s and r['weather']==w for r in rows)
                  for w in (WEATHERS if s=='validate' else ('clean','fog'))}
              for s in ('train','validate')}
    if counts['train']['clean']!=counts['train']['fog']:
        raise ValueError('Unpaired training candidate counts')
    for s in ('fit','select','calibrate'):
        if any(set(old['split'][s]) & set(old['split'][t]) for t in old['split'] if t!=s):
            raise ValueError('Scene partition overlap')
    if not smoke and {r['scene'] for r in rows if r['split']=='validate'} & set(sampling['all']):
        raise ValueError('Validation scenes overlap with training')
    bind(dest,dict(contract=contract,policy=policy,split=old['split'],sampling=sampling,files=rows,weight=WEIGHT,
                   counts=counts,frozen_model_state=state))
    print('COMPLETE CACHE',area,len(rows),flush=True)


if __name__=='__main__':
    prepare(sys.argv[1],'--smoke' in sys.argv)
