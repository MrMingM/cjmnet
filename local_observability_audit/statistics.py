"""Scene-cluster uncertainty, fixed-alpha leave-one-scene-out descriptive probes."""
import numpy as np

def ranks(x):
    x=np.asarray(x,float); order=np.argsort(x,kind='stable'); out=np.empty(len(x),float)
    start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:
            end+=1
        out[order[start:end]]=(start+end-1)/2
        start=end
    return out

def rho(x,y):
    if len(x)<3:
        return None
    a,b=ranks(x),ranks(y); a-=a.mean(); b-=b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-14 else None

def cluster_ci(values, scenes, statistic, cfg, seed):
    values=np.asarray(values); scenes=np.asarray(scenes); unique=np.unique(scenes)
    estimate=statistic(values)
    result={'estimate':estimate,'ci95':None,'rows':len(values),'scenes':len(unique),'valid':False}
    if len(values)<cfg['minimum_rows'] or len(unique)<cfg['minimum_scenes'] or estimate is None:
        return result
    groups=[np.flatnonzero(scenes==s) for s in unique]
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(cfg['bootstrap_replicates']):
        idx=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))])
        v=statistic(values[idx])
        if v is not None and np.isfinite(v):
            samples.append(v)
    result['valid_replicates']=len(samples)
    if len(samples)>=.9*cfg['bootstrap_replicates']:
        result.update(ci95=np.percentile(samples,[2.5,97.5]).tolist(),valid=True)
    return result

def association(x,y,scenes,cfg,seed):
    # Exact Spearman for a bootstrap multiset via weighted midranks. Sorting
    # once avoids thousands of repeated Python tie loops on large cohorts.
    x,y,scenes=np.asarray(x,float),np.asarray(y,float),np.asarray(scenes)
    unique,sg=np.unique(scenes,return_inverse=True)
    estimate=rho(x,y)
    result={'estimate':estimate,'ci95':None,'rows':len(x),'scenes':len(unique),'valid':False}
    if len(x)<cfg['minimum_rows'] or len(unique)<cfg['minimum_scenes'] or estimate is None:
        return result
    _,gx=np.unique(x,return_inverse=True); _,gy=np.unique(y,return_inverse=True)
    def wrank(g,w):
        bins=np.bincount(g,weights=w)
        return (np.cumsum(bins)-bins/2)[g]
    samples=[]; rng=np.random.default_rng(seed)
    for _ in range(cfg['bootstrap_replicates']):
        multiplicity=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
        w=multiplicity[sg]; a=wrank(gx,w); b=wrank(gy,w)
        a-=np.average(a,weights=w); b-=np.average(b,weights=w)
        den=np.sqrt(np.dot(w,a*a)*np.dot(w,b*b))
        if den>1e-14:
            samples.append(float(np.dot(w,a*b)/den))
    result['valid_replicates']=len(samples)
    if len(samples)>=.9*cfg['bootstrap_replicates']:
        result.update(ci95=np.percentile(samples,[2.5,97.5]).tolist(),valid=True)
    return result

def performance(y,p):
    y,p=np.asarray(y),np.asarray(p)
    sst=float(np.sum((y-y.mean())**2))
    return {'r2':float(1-np.sum((y-p)**2)/sst) if sst>1e-14 else None,
            'mae':float(np.abs(y-p).mean()),'rho':rho(y,p)}

def ridge_oof(x,y,scenes,alpha):
    x,y,scenes=np.asarray(x,float),np.asarray(y,float),np.asarray(scenes)
    out=np.empty_like(y)
    for scene in np.unique(scenes):
        test=scenes==scene; train=~test
        if train.sum()<2:
            return None
        mean=x[train].mean(0); sd=x[train].std(0); sd[sd<1e-10]=1
        a=(x[train]-mean)/sd; b=(x[test]-mean)/sd
        ym=y[train].mean(0)
        weights=np.linalg.solve(a.T@a+alpha*np.eye(x.shape[1]),a.T@(y[train]-ym))
        out[test]=b@weights+ym
    return out

def incremental(base,extra,y,scenes,cfg,seed):
    scenes=np.asarray(scenes)
    if len(y)<cfg['minimum_rows'] or len(np.unique(scenes))<cfg['minimum_scenes']:
        return {'valid':False,'reason':'insufficient scene-separated cohort','rows':len(y),'scenes':len(np.unique(scenes))}
    a=ridge_oof(base,y,scenes,cfg['ridge_alpha'])
    b=ridge_oof(np.c_[base,extra],y,scenes,cfg['ridge_alpha'])
    if a is None or b is None:
        return {'valid':False,'reason':'insufficient folds'}
    def diff(v):
        pa,pb=performance(v[:,0],v[:,1]),performance(v[:,0],v[:,2])
        return None if pa['r2'] is None else pb['r2']-pa['r2']
    ci=cluster_ci(np.c_[y,a,b],scenes,diff,cfg,seed)
    pa,pb=performance(y,a),performance(y,b)
    return {'valid':ci['valid'],'baseline':pa,'augmented':pb,'delta_r2':ci,
            'mae_reduction':pa['mae']-pb['mae'],
            'rank_correlation_delta':None if pa['rho'] is None or pb['rho'] is None else pb['rho']-pa['rho'],
            'uncertainty':'scene bootstrap of fixed OOF predictions; models not refitted in each replicate',
            'folds':sorted(np.unique(scenes).tolist()), 'alpha':cfg['ridge_alpha']}

def strata_indices(rows,cfg,field):
    edges=cfg['count_bin_edges'] if field=='valid_point_count' else cfg['reliability_bin_edges']
    groups={}
    for i,r in enumerate(rows):
        key=(r['weather'],int(np.digitize(r[field],edges)))
        groups.setdefault(key,[]).append(i)
    return list(groups.values())

def matched_correlation(x,y,rows,cfg,field):
    # Within-weather, within-fixed-bin ranked deviations; no fitted threshold.
    xx,yy,ss=[],[],[]
    for ids in strata_indices(rows,cfg,field):
        if len(ids)<3:
            continue
        a,b=ranks(np.asarray(x)[ids]),ranks(np.asarray(y)[ids])
        xx.extend(a-a.mean()); yy.extend(b-b.mean()); ss.extend(rows[i]['scene_id'] for i in ids)
    if len(xx)<3:
        return {'valid':False,'rows':len(xx),'estimate':None,'ci95':None}
    def pearson(v):
        a=v[:,0]-v[:,0].mean(); b=v[:,1]-v[:,1].mean(); den=np.linalg.norm(a)*np.linalg.norm(b)
        return float(a@b/den) if den>1e-14 else None
    return cluster_ci(np.c_[xx,yy],ss,pearson,cfg,cfg['seed']+44)

def view_shuffle(rows,cfg,rng):
    groups={}
    for i,r in enumerate(rows):
        key=(r['weather'],int(np.digitize(r['source_distance'],cfg['distance_bin_edges'])))
        groups.setdefault(key,[]).append(i)
    order=np.arange(len(rows))
    for ids in groups.values():
        order[ids]=rng.permutation(ids)
    return order

def null_summary(values,observed,cfg):
    vals=[v for v in values if v is not None and np.isfinite(v)]
    return {'valid':bool(vals),'replicates':len(vals),'median':float(np.median(vals)) if vals else None,
            'q95':float(np.percentile(vals,cfg['decision']['null_percentile'])) if vals else None,
            'observed':observed,
            'above_q95':bool(observed is not None and vals and observed>np.percentile(vals,cfg['decision']['null_percentile'])),
            'interpretation':'diagnostic shuffled null, not an independent-sample significance test'}
