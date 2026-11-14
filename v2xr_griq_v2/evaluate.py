"""One ranking score, native postprocessing, and paired scene statistics."""
import numpy as np
import torch
from scipy.special import logit

from v2xr_griq_v2.dataset import get
from v2xr_griq_v2.train_common import predict, score
from v2xr_griq_experiments.metrics import binary,combine,equal_fp_frontier,target_changes
from v2xr_griq_experiments.native import replay

THRESHOLDS = (.05,.10,.15,.20,.30,.45,.60,.75)
IOU = (.3,.5,.7)


def stats(d,pp,z,threshold,full=True):
    from evaluate import official_match_trace,statistics
    from support import corner_array
    from opencood.utils.eval_utils_opv2v import caluclate_tp_fp
    pred,s,stage,final = replay(pp,d['decoded'],z,d['transform'],threshold,trace=True)
    m=d['meta']
    out=dict(scene=m['key'][0],key=m['key'],observations=m['observations'],hits={},
             none=pred is None,stage=stage,final=d['ids'][final].tolist() if pred is not None else [])
    for protocol in (('F','O','O*') if full else ('F',)):
        g=m['fgt' if protocol=='F' else 'ogt']
        gt=torch.from_numpy(corner_array(g))
        out[protocol]=statistics()
        if protocol=='O' and pred is None:
            continue
        for t in IOU:
            if full and protocol=='F' and t==.7:
                pairs,_=official_match_trace(pred,s,gt,[str(r['object_id']) for r in g],out[protocol],t)
                out['hits']={r['object_id']:dict(anchor=int(d['ids'][final[r['prediction_index']]]),
                       low=bool(d['prob'][final[r['prediction_index']]]<=d['native_threshold']),iou=r['iou']) for r in pairs}
            else:
                caluclate_tp_fp(pred,s,gt,out[protocol],t)
    return out


def prefix(frame,cut):
    result={}
    for t,row in frame.items():
        n=sum(s>np.float32(cut) for s in row['score'])
        result[t]={k:(v if k=='gt' else v[:n]) for k,v in row.items()}
    return result


def summary(rows,p='F'):
    a=combine([r[p] for r in rows])
    a['macro']=float(np.mean([r['ap'] for r in a.values()])) if all(r['ap'] is not None for r in a.values()) else None
    return a


def aps(s):
    return [s[f'{i:.1f}']['ap'] for i in IOU]


def choose(trials,base):
    target=np.asarray(aps(base))
    def order(r):
        a=np.asarray(aps(r['F']))
        d=a-target
        feasible=bool((d>=-.005).all())
        return feasible, r['F']['macro'], min(d), -abs(r['F']['0.7']['fp']-base['0.7']['fp']), r['threshold']
    return max(trials,key=order)


def cached_logits(rows,policy,net,ck,device,intervention=None):
    output=[]
    local=ck is not None and ck['variant'] in ('cross','lidar','radar','independent')
    for i,row in enumerate(rows):
        d=get(row,policy,local)
        if net is None:
            z=d['logit']
            p=None
        else:
            l=d.get('local')
            x=dict(x=d['x'],l=l) if local else dict(x=d['x'])
            p,_=predict(net,x,ck,device,ablation=intervention,seed=ck['seed']+i)
            z=logit(score(p).clip(1e-6,1-1e-6)).astype(np.float32)
        output.append((z,p))
    return output


def calibrate(rows,policy,zs,pp,smoke=False):
    native=[stats(get(r,policy),pp,get(r,policy)['logit'],.2,False) for r in rows]
    base=summary(native)
    traces=[]
    for r,(z,_) in zip(rows,zs):
        d=get(r,policy)
        at_floor=stats(d,pp,z,min(THRESHOLDS),False)['F']
        if smoke and prefix(at_floor,.3)!=stats(d,pp,z,.3,False)['F']:
            raise ValueError('Official threshold prefix failed')
        traces.append(at_floor)
    trials=[]
    for threshold in THRESHOLDS:
        a=summary([dict(F=prefix(f,threshold)) for f in traces])
        trials.append(dict(threshold=threshold,F=a))
    return dict(base=base,trials=trials,selected=choose(trials,base))


def matched_count(z,prob,threshold=.2):
    # Transfer scores only to the same native eligible pool; no low-score admissions.
    return np.where(prob>threshold,z,-30).astype(np.float32)


def paired(rows,base,draws=150):
    from opencood.utils.eval_utils_opv2v import calculate_ap
    rng=np.random.default_rng(20261017)
    scenes=list(dict.fromkeys(r['scene'] for r in rows))
    by={s:np.flatnonzero([r['scene']==s for r in rows]) for s in scenes}
    blocks={s:[x[i:i+16] for i in range(0,len(x),16)] for s,x in by.items()}
    def values(rs,ids):
        out=[]
        for t in IOU:
            f=[rs[i]['F'][t] for i in ids]
            stat={k:[x for a in f for x in a[k]] for k in ('tp','fp','score')}
            stat['gt']=sum(a['gt'] for a in f)
            out.append(calculate_ap({t:stat},t,False)[0] if stat['gt'] else np.nan)
        return np.asarray([*out,np.mean(out)])
    paired_draw=[]
    for _ in range(draws):
        ids=np.concatenate([parts[i] for parts in blocks.values()
                            for i in rng.integers(len(parts),size=len(parts))])
        paired_draw.append(100*(values(rows,ids)-values(base,ids)))
    delta=np.asarray(paired_draw)
    return dict(names=['AP30','AP50','AP70','MacroAP'],draws=draws,block=16,
                interval_pp=[np.quantile(d[np.isfinite(d)],[.025,.975]).tolist() if np.isfinite(d).any() else None
                             for d in delta.T])


def formal(rows,policy,zs,pp,threshold,base,smoke=False,matched=False,details=False):
    result,low=[],[]
    staging=np.zeros((4,4),np.int64)
    for row,(z,p) in zip(rows,zs):
        d=get(row,policy)
        if matched:
            z=matched_count(z,d['prob'])
        f=stats(d,pp,z,1e-7 if matched else threshold)
        result.append(f)
        b=np.searchsorted([.3,.5,.7],d['iou'],side='right')
        for k in range(1,5):
            np.add.at(staging[:,k-1],b[f['stage']>=k],1)
        if details:
            low.append(f if threshold==min(THRESHOLDS) else stats(d,pp,z,min(THRESHOLDS)))
    result_base=base or result
    out={p:summary(result,p) for p in ('F','O','O*')}
    out.update(stage_counts=staging.tolist(),targets=target_changes(result,result_base),
               scenes={s:summary([r for r in result if r['scene']==s]) for s in
                       dict.fromkeys(r['scene'] for r in result)},none=sum(r['none'] for r in result))
    if details and base is not None:
        out['equal_fp']=equal_fp_frontier(low,base)
        if not smoke:
            out['paired']=paired(result,base)
    return out,result


def quality_metrics(rows,policy,zs):
    iou=np.concatenate([get(r,policy)['iou'] for r in rows])
    p=np.concatenate([p for _,p in zs])
    p=p.clip(1e-7,1-1e-7)
    vals=score(p)
    intervals=[vals[(iou>=a)&(iou<b)] for a,b in ((0,.3),(.3,.5),(.5,.7),(.7,1.001))]
    return dict(calibration=[binary(iou>=t,p[:,j]) for j,t in enumerate(IOU)] if p.shape[1]==3 else
                [binary(iou>=.7,p[:,0])], score_quantiles=[
                np.quantile(v,[.1,.5,.9]).tolist() if len(v) else None for v in intervals])
