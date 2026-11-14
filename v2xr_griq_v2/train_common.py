"""Frame-streamed feature packing and resumable GPU training."""
import os
import time
from pathlib import Path

import numpy as np
import torch

from v2xr_griq_v2.dataset import get,bind,read,save,sha
from v2xr_griq_v2.local_complement import model
from v2xr_griq_v2.local_evidence import DL,DR,intervene
from v2xr_griq_v2.multi_quality import LEVELS,D,loss,probs,size
from v2xr_griq_experiments.features import indices, GSPR, RADAR


def rows_for(audit, subset, fold=None):
    s = audit['split']
    return {k:[r for r in audit['files'] if r['split']=='train' and r['scene'] in scenes and
               r['index'] in audit['sampling'][subset if k=='fit' else 's32'][r['scene']]]
            for k,scenes in (fold or s).items()}


def pack(rows,policy,root,local):
    root.mkdir(parents=True,exist_ok=True)
    manifest = root/'PACK.json'
    identity = [[r['npz_sha256'],r.get('local_sha256') if local else None] for r in rows]
    if not manifest.exists():
        count = sum(len(get(r,policy)['ids']) for r in rows)
        if not count:
            raise ValueError('Empty candidate training partition')
        specs = {'x':(count,D), 'y':(count,3)}
        if local:
            specs['l']=(count,6,DL+DR)
        arrays = {n:np.lib.format.open_memmap(root/f'{n}.npy',mode='w+',dtype=np.float32,shape=s)
                  for n,s in specs.items()}
        cursor = 0
        for row in rows:
            d = get(row,policy,local)
            n = len(d['ids'])
            j = slice(cursor,cursor+n)
            arrays['x'][j] = d['x']
            arrays['y'][j] = np.stack([d[k] for k in ('y30','y50','y70')],1)
            if local:
                arrays['l'][j] = d['local']
            cursor += n
        for a in arrays.values():
            a.flush()
        bind(manifest,dict(identity=identity,policy=policy,files={n:sha(root/f'{n}.npy') for n in specs}))
    data = read(manifest)
    if data['identity']!=identity or data['policy']!=policy or any(
            sha(root/f'{k}.npy')!=h for k,h in data['files'].items()):
        raise ValueError('Packed training input changed')
    return {k:np.load(root/f'{k}.npy',mmap_mode='r') for k in data['files']}


def normalizer(x):
    count,total,square = np.zeros((3,D))
    for i in range(0,len(x),8192):
        a = x[i:i+8192].astype(np.float64)
        count += np.isfinite(a).sum(0)
        total += np.nansum(a,0)
        square += np.nansum(a*a,0)
    mean = total/count.clip(1)
    sd = np.sqrt(np.maximum(0,square/count.clip(1)-mean**2)).clip(.01)
    return mean.astype(np.float32),sd.astype(np.float32)


def transformed(x,loc,mean,sd,ablation,seed):
    x = x.copy()
    if ablation in ('no_gspr','points','no_radar'):
        fields = GSPR if ablation in ('no_gspr','points') else RADAR
        x[:,indices(fields)] = np.nan
        if ablation=='points':
            # Replace effective counts with unweighted physical counts, preserve the architecture.
            x[:,indices(('gspr_points','effective_count','gspr_empty'))] = x[:,[8,8,11]]
    finite = np.isfinite(x)
    x = np.concatenate([np.where(finite,(x-mean)/sd,0),~finite],1).astype(np.float32)
    if loc is not None:
        loc = intervene(loc,ablation,np.random.default_rng(seed))
        if ablation in ('no_gspr','points'):
            loc[:,:,1] = loc[:,:,0] if ablation=='points' else 0
            loc[:,:,2:4] = 0
            loc[:,:,6] = 1
        if ablation=='no_radar':
            loc[:,:,DL:] = 0
            loc[:,:,-1] = 1
        for k in (0,1,DL,DL+1,DL+4):
            loc[:,:,k] = np.log1p(loc[:,:,k])
    return x,loc


def batch(data,idx,ck,device,seed):
    loc = data['l'][idx] if 'l' in data and ck['variant'] in ('cross','independent','lidar','radar') else None
    x,l = transformed(data['x'][idx],loc,np.asarray(ck['mean'],np.float32),
                      np.asarray(ck['sd'],np.float32),ck['ablation'],seed)
    return torch.from_numpy(x).to(device), None if l is None else torch.from_numpy(l).to(device)


def predict(net,data,ck,device, ablation=None, seed=0, contribution=False):
    if ablation is not None:
        ck = ck | dict(ablation=ablation)
    predictions,residuals = [],[]
    with torch.inference_mode():
        for i in range(0,len(data['x']),2048):
            x,l = batch(data,slice(i,i+2048),ck,device,seed+i)
            z = net(x,l)
            predictions.append(probs(z,ck['mode']).cpu().numpy())
            if contribution and l is not None:
                for k in range(6):
                    held = l.clone()
                    held[:,k,:] = 0
                    held[:,k,[5,6,DL+5]] = 1
                    residuals.append((k,(z-net(x,held)).cpu().numpy()))
    n = 1 if ck['mode']=='strict' else 3
    p = np.concatenate(predictions) if predictions else np.empty((0,n),np.float32)
    return p,residuals


def score(p):
    return p.mean(1)


def steps_for(budget):
    return 4000 if budget=='fixed' else 12000


def train(fit,dev,out,spec,device,smoke=False):
    mode,variant,seed = spec['mode'],spec['variant'],spec['seed']
    torch.manual_seed(seed)
    net = model(variant,mode).to(device)
    initial = [p.detach().clone() for p in net.parameters()]
    optimizer = torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001)
    mean,sd = normalizer(fit['x'])
    ck = spec | dict(mean=mean.tolist(),sd=sd.tolist(),parameters=size(net),examples=len(fit['x']),
                     positives=fit['y'].sum(0).astype(int).tolist())
    out.mkdir(parents=True,exist_ok=True)
    best_path,progress = out/'best.pt',out/'progress.pt'
    def save_pt(path,obj):
        tmp = path.with_suffix('.tmp')
        torch.save(obj,tmp)
        tmp.replace(path)

    if (out/'COMPLETE.json').exists():
        mark = read(out/'COMPLETE.json')
        if mark['spec']!=spec or sha(best_path)!=mark['best'] or sha(out/'history.json')!=mark['history']:
            raise ValueError('Training checkpoint identity changed')
        return torch.load(best_path,map_location='cpu',weights_only=False)
    trace,best,bad,start = [],None,0,0
    if progress.exists():
        obj = torch.load(progress,map_location='cpu',weights_only=False)
        if obj['spec']!=spec:
            raise ValueError('Incomplete checkpoint has a different specification')
        net.load_state_dict(obj['weights'])
        optimizer.load_state_dict(obj['optimizer'])
        trace,best,bad,start = (obj[k] for k in ('trace','best','bad','step'))
    stride = 2 if smoke else 400
    ceiling = 2 if smoke else steps_for(spec['budget'])
    w = torch.tensor(spec['weights'],dtype=torch.float32,device=device)
    pos = np.flatnonzero(fit['y'][:,2]>.5)
    neg = np.flatnonzero(fit['y'][:,2]<.5)
    if not len(pos) or not len(neg):
        raise ValueError('Training lacks positive or negative strict candidates')
    # Balance is fixed across size arms. Sample with replacement at each update.
    ratio = spec['ratio']
    started = time.monotonic()
    if spec['budget']=='converge' and bad>=5:
        start=ceiling
    for step in range(start,ceiling,stride):
        net.train()
        losses = []
        counts = np.zeros((3,2),dtype=np.int64)
        for j in range(step,min(step+stride,ceiling)):
            rng = np.random.default_rng(seed+j)
            k = int(1024*ratio)
            ids = np.r_[rng.choice(pos,k),rng.choice(neg,1024-k)]
            x,l = batch(fit,ids,ck,device,seed+j)
            y = torch.from_numpy(fit['y'][ids]).to(device)
            z = net(x,l)
            v = loss(z,y,mode,w)
            optimizer.zero_grad()
            v.sum().backward()
            if not torch.isfinite(v).all() or not all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):
                raise ValueError('Nonfinite scoring update')
            optimizer.step()
            yy = fit['y'][ids]
            eligible = np.column_stack([np.ones(len(yy)),yy[:,0],yy[:,1]]) if mode=='ordinal' else np.ones_like(yy)
            counts[:,0] += (yy*eligible).sum(0).astype(int)
            counts[:,1] += ((1-yy)*eligible).sum(0).astype(int)
            losses.append(v.detach().cpu().numpy())
        net.eval()
        p,_ = predict(net,dev,ck,device)
        y = dev['y'][:,2:3] if mode=='strict' else dev['y']
        pc = p.clip(1e-7,1-1e-7)
        val = float(-(y*np.log(pc)+(1-y)*np.log1p(-pc)).mean())
        entry = dict(step=min(step+stride,ceiling),loss=np.mean(losses,0).tolist(),
                     masked_counts=counts.tolist(),selection_loss=val,
                     elapsed_seconds=time.monotonic()-started)
        trace.append(entry)
        if best is None or val<best['selection_loss']-1e-5:
            best = ck | dict(weights_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                             best_step=entry['step'],selection_loss=val)
            bad = 0
        else:
            bad += 1
        save_pt(progress,dict(spec=spec,weights=net.state_dict(),optimizer=optimizer.state_dict(),
                              trace=trace,best=best,bad=bad,step=entry['step']))
        print('TRAIN',out.name,entry['step'],f'{val:.6f}',flush=True)
        if spec['budget']=='converge' and bad>=5:
            break
    if start==0 and not any(not torch.equal(p,a) for p,a in zip(net.parameters(),initial)):
        raise ValueError('No quality parameter updated')
    if smoke and hasattr(net,'alpha') and (net.alpha.item()==0 or not any(
            not torch.equal(p,a) for (name,p),a in zip(net.named_parameters(),initial)
            if name.startswith(('cross.','head.','l.','r.')))):
        raise ValueError('Local residual did not learn')
    best.update(updates=trace[-1]['step'],peak_gpu_bytes=torch.cuda.max_memory_allocated(device),
                elapsed_seconds=sum(a['elapsed_seconds'] for i,a in enumerate(trace)
                                    if i==len(trace)-1 or trace[i+1]['elapsed_seconds']<a['elapsed_seconds']))
    save_pt(best_path,best)
    save(out/'history.json',trace)
    bind(out/'COMPLETE.json',dict(spec=spec,best=sha(best_path),history=sha(out/'history.json')))
    return best


def restore(ck,device):
    net = model(ck['variant'],ck['mode']).to(device)
    net.load_state_dict(ck['weights_state'])
    return net.eval()
