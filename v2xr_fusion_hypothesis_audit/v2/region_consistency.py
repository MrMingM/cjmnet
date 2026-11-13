"""Test structural information beyond coordinate and appearance baselines."""
from collections import defaultdict
from itertools import combinations
import json

import numpy as np
import shapely
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from shared import cache
from shared.geometry import cosine
from shared.input_audit import OUT, WEATHERS, atomic_json
from shared.statistics import ratio_interval

COLUMNS=('coordinate','same_grid','local','structure','shuffled','cross_scene','cycle')


def labels(xy, targets):
    polygons=shapely.polygons(np.asarray([t['corners'][:4] for t in targets])[:,:,:2]) if targets else []
    inside=shapely.intersects(shapely.points(xy)[:,None],polygons)
    ids=np.full(len(xy),-1)
    unique=inside.sum(1)==1
    if unique.any(): ids[unique]=inside[unique].argmax(1)
    ids[inside.sum(1)>1]=-2
    return ids


def neighbor_distances(xy):
    distances=cdist(xy,xy)
    return np.sort(distances,axis=1)[:,1:5]


def loop_matrices(xy, descriptors):
    result={}
    for i,j in combinations(range(len(xy)),2):
        distance=cdist(xy[i],xy[j])
        appearance=descriptors[i]@descriptors[j].T
        logits=np.exp(3*appearance-distance/3)*(distance<=12)
        row=lambda a:a/np.maximum(a.sum(1,keepdims=True),1e-12)
        result[i,j]=row(logits)
        result[j,i]=row(logits.T)
    return result


def measures(y,p,decision):
    y,p,decision=np.asarray(y),np.asarray(p),np.asarray(decision)
    tp=int((y&decision).sum())
    fp=int((~y&decision).sum())
    positive=int(y.sum())
    ratio=lambda a,b:a/b if b else None
    return dict(n=len(y), positives=positive, accuracy=ratio(int((y==decision).sum()),len(y)),
                precision=ratio(tp,tp+fp), recall=ratio(tp,positive), f1=ratio(2*tp,positive+tp+fp),
                auroc=roc_auc_score(y,p) if 0<positive<len(y) else None,
                auprc=average_precision_score(y,p) if positive else None,
                recall_at_fpr_10=ratio(int((y&(p>np.quantile(p[~y],.9))).sum()),positive)
                if positive<len(y) else None)


def reports(rows):
    grouped=defaultdict(list)
    for row in rows:
        if row['label'] is None: continue
        for k in ('all',row['status'],f"cars{row['cars']}"):
            grouped[row['weather'],row['scale'],row['region'],k].append(row)
    output={}
    for key, group in grouped.items():
        y=np.asarray([r['label'] for r in group],dtype=bool)
        x=np.asarray([r['scores'] for r in group])
        scenes=np.asarray([r['scene'] for r in group])
        item={}
        for i,name in enumerate(COLUMNS):
            mask=np.isfinite(x[:,i])
            if mask.any():
                p=x[mask,i]
                item[name]=measures(y[mask],p,p>=np.median(p))
        pred={name:np.full(len(y),np.nan) for name in ('baseline','plus_structure','plus_shuffled')}
        for scene in sorted(set(scenes)):
            test=scenes==scene
            train=~test
            if len(np.unique(y[train]))!=2: continue
            for name, cols in (('baseline',[0,1,2]),('plus_structure',[0,1,2,3]),('plus_shuffled',[0,1,2,4])):
                model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=300))
                pred[name][test]=model.fit(x[train][:,cols],y[train]).predict_proba(x[test][:,cols])[:,1]
        valid=np.isfinite(pred['baseline'])
        if valid.any():
            for name,p in pred.items():
                item[name]=measures(y[valid],p[valid],p[valid]>=.5)
            diff={}
            for scene in sorted(set(scenes[valid])):
                use=valid&(scenes==scene)
                if not (y[use].any() and (~y[use]).any()): continue
                delta=average_precision_score(y[use],pred['plus_structure'][use])-average_precision_score(y[use],pred['baseline'][use])
                diff[scene]=[delta,1]
            item['scene_macro_auprc_delta']=ratio_interval(diff)
        item['scenes']=len(set(scenes))
        output['/'.join(map(str,key))]=item
    return output


def analyze(smoke=False):
    root=OUT/('V2_smoke' if smoke else 'V2')
    root.mkdir(exist_ok=True)
    rows=[]
    past={}
    background=defaultdict(lambda:[0,0.,0])
    with (root/'RELATION_DETAILS.tmp').open('w',encoding='utf-8') as stream:
        for meta,arr in cache.frames(smoke):
            scene=meta['frame'][0]
            for weather in WEATHERS:
                d=meta['weather'][weather]
                cars=len(d['cars'])
                if cars<2: continue
                targets=meta['targets']
                for scale in range(2):
                    h,w=arr[f'{weather}_shape{scale}'][2:]
                    step=.8*2**scale
                    indices=arr[f'{weather}_peaks{scale}']
                    xy=np.stack((indices%w,indices//w),-1)*step+d['origin']
                    vectors=arr[f'{weather}_descriptors{scale}'].astype(np.float32)
                    compact=arr[f'{weather}_feature{scale}'].astype(np.float32)
                    clean=arr[f'clean_feature{scale}'].astype(np.float32)
                    def sampled(values,i,grid):
                        return values[i,:,round((grid//w)/4) if round((grid//w)/4)<values.shape[2] else values.shape[2]-1,
                                      min(round((grid%w)/4),values.shape[3]-1)]
                    for region in ('fixed_blocks','gt_auxiliary'):
                        positions, feats=xy,vectors
                        if region=='gt_auxiliary':
                            if len(targets)<2: continue
                            centers=np.asarray([t['box'][:2] for t in targets])
                            grids=np.rint((centers-d['origin'])/step).astype(int)
                            valid=((grids>=0)&(grids<[w,h])).all(1)
                            if valid.sum()<2: continue
                            grids=grids[valid]
                            centers=centers[valid]
                            loc=grids[:,1]*w+grids[:,0]
                            positions=np.broadcast_to(centers,(cars,*centers.shape))
                            feats=np.asarray([[sampled(compact,i,g) for g in loc] for i in range(cars)])
                        identities=np.asarray([labels(p,targets) for p in positions])
                        layouts=[neighbor_distances(p) for p in positions]
                        paths=loop_matrices(positions,feats) if cars>=3 else {}
                        for a,b in combinations(range(cars),2):
                            distance=cdist(positions[a],positions[b])
                            local=feats[a]@feats[b].T
                            if region=='gt_auxiliary':
                                norm=np.linalg.norm(feats,axis=2)
                                local/=np.maximum(norm[a,:,None]*norm[b,None,:],1e-12)
                            for i,j in np.argwhere(distance<=12):
                                ida,idb=identities[a,i],identities[b,j]
                                coordinate=-float(distance[i,j])
                                structure=-float(np.abs(layouts[a][i]-layouts[b][j]).mean())
                                shuffled=-float(np.abs(layouts[a][i]-layouts[b][(j+17)%len(layouts[b])]).mean())
                                grid=int(indices[a,i]) if region=='fixed_blocks' else int(loc[i])
                                same=cosine(sampled(compact,a,grid),sampled(compact,b,grid))
                                change=float(np.linalg.norm(sampled(compact,a,grid)-sampled(clean,a,grid)))
                                if ida<0 or idb<0:
                                    key=f'{weather}/{scale}/{region}'
                                    background[key][0]+=1
                                    background[key][1]+=change
                                    background[key][2]+=int(ida==-2 or idb==-2)
                                    continue
                                cycle=max(((paths[a,k][i]@paths[k,b][:,j]) for k in range(cars) if k not in (a,b)),default=np.nan)
                                prior=past.get((weather,scale,region))
                                cross=cosine(feats[a,i],prior[j]) if prior is not None and len(prior)>j else np.nan
                                row=dict(scene=scene,frame=meta['frame'][1],weather=weather,scale=scale,region=region,
                                         cars=cars, label=bool(ida==idb), status='p0_detected' if str(targets[ida]['object_id']) in d['multi_reference_hits'] else 'p0_missed',
                                         weather_change=change, scores=[coordinate,same,float(local[i,j]),structure,shuffled,cross,float(cycle)])
                                rows.append(row)
                                stream.write(json.dumps(dict(row,scores=[None if not np.isfinite(x) else x for x in row['scores']]))+'\n')
                        # Only use a vehicle from a different scene for the cross-scene control.
                        key=(weather,scale,region)
                        if (key,'scene') not in past:
                            past[key,'scene']=scene
                            past[key,'source']=feats[-1]
                        elif past[key,'scene']!=scene:
                            past[key]=past[key,'source']
            if (meta['index']+1)%100==0: print('V2',meta['index']+1,flush=True)
    (root/'RELATION_DETAILS.tmp').replace(root/'RELATION_DETAILS.jsonl')
    stats=reports(rows)
    result=dict(metrics=stats,background=background,
                notes=['Same-grid feature uses the shared 4-cell sampled projection.',
                       'Cross-scene descriptor columns are comparison scores on the same labelled rows.',
                       'Cycle score uses actual triples; corresponding-coordinate confounds remain.'])
    atomic_json(root/'REGION_MATCHING.json',result)
    lines=['# V2 区域结构诊断','','同格特征使用每四格抽样的八维投影。几何额外收益以留出场景的 AUPRC 差值衡量。','',
           '| 分组 | 场景 | 坐标 AUPRC | 基线 AUPRC | 加结构 AUPRC | 场景平均差及区间 |','|---|---:|---:|---:|---:|---|']
    for key, r in sorted(stats.items()):
        lines.append(f"| {key} | {r['scenes']} | {r['coordinate']['auprc']} | {r.get('baseline',{}).get('auprc')} | {r.get('plus_structure',{}).get('auprc')} | {r.get('scene_macro_auprc_delta')} |")
    (root/'GEOMETRY_CONSISTENCY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
