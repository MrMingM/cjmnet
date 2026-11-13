"""Paired feature changes, loose box association and held-out error prediction."""
from collections import Counter, defaultdict
from itertools import combinations
import json

import numpy as np
import shapely
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared import cache
from shared.geometry import error
from shared.input_audit import OUT, WEATHERS, atomic_json, read_json, lr
from shared.metrics import overlap
from shared.statistics import correlation_blocks, ratio_interval

RULES=((2.,0.),(4.,.05),(6.,.10),(6.,0.))


def associate(arr, prefix, targets, radius, minimum):
    boxes,corners,scores,top=(arr[f'{prefix}_{k}'] for k in ('box','corners','score','top'))
    boxes,corners,scores=boxes[top],corners[top],scores[top]
    if not len(targets): return {},Counter()
    gt=np.asarray([t['box'] for t in targets],dtype=float)
    distance=np.linalg.norm(boxes[:,None,:2]-gt[None,:,:2],axis=-1)
    iou=np.zeros_like(distance)
    for j,target in enumerate(targets):
        ind,values=overlap(corners,np.asarray(target['corners']))
        iou[ind,j]=values
    matches={}
    counts=Counter()
    assigned=set()
    for i in range(len(boxes)):
        options=np.flatnonzero((distance[i]<=radius)&(iou[i]>=minimum))
        if len(options)>1:
            best=options[np.argsort(distance[i,options],kind='stable')]
            if distance[i,best[1]]-distance[i,best[0]]<.5:
                counts['ambiguous_candidates']+=1
                continue
            options=best[:1]
        if not len(options): continue
        j=int(options[0])
        if j in assigned:
            counts['duplicate_association']+=1
            continue
        assigned.add(j)
        counts['low_iou_associated']+=iou[i,j]<.3
        matches[j]=dict(error=error(boxes[i],gt[j]),box=boxes[i],score=scores[i],iou=iou[i,j])
    counts['targets']+=len(gt)
    counts['associated']+=len(matches)
    counts['missing']+=len(gt)-len(matches)
    return matches,counts


def feature_records(meta, arr, output, prior):
    scene=meta['frame'][0]
    targets=meta['targets']
    polygons=shapely.polygons(np.asarray([x['corners'] for x in targets])[:,:4,:2]) if targets else []
    for scale in range(3):
        clean=arr[f'clean_feature{scale}'].astype(np.float32)
        n,d,h,w=clean.shape
        if n<2: continue
        y,x=np.indices((h,w))
        origin=meta['weather']['clean']['origin']
        xy=np.stack((x,y),-1).reshape(-1,2)*(3.2*2**scale)+origin
        points=shapely.points(xy)
        distance=shapely.distance(points[:,None],polygons)
        nearest=distance.min(1) if targets else np.full(h*w,np.inf)
        for weather in ('fog','snow'):
            change=(arr[f'{weather}_feature{scale}'].astype(np.float32)-clean).transpose(0,2,3,1).reshape(n,h*w,d)
            other='snow' if weather=='fog' else 'fog'
            swapped=(arr[f'{other}_feature{scale}'].astype(np.float32)-clean).transpose(0,2,3,1).reshape(n,h*w,d)
            support=(arr[f'{weather}_support{scale}']>0).any(-1).reshape(n,h*w)
            active=support.any(0)|(np.linalg.norm(change,axis=-1).max(0)>1e-3)
            for a,b in combinations(range(n),2):
                categories={'target':nearest==0,'neighborhood':(nearest>0)&(nearest<=4),
                            'background':nearest>4,'ego_unique':support[0]&(support.sum(0)==1),
                            'neighbor_unique':~support[0]&(support.sum(0)==1),'multiple':support.sum(0)>=2}
                for label,mask in categories.items():
                    ids=np.flatnonzero(mask&active)
                    if not len(ids): continue
                    # Fixed deterministic subsampling bounds memory. Empty zero cells are excluded.
                    ids=ids[np.linspace(0,len(ids)-1,min(12,len(ids)),dtype=int)]
                    dst=output[weather,scale,label]
                    dist=np.linalg.norm(np.asarray(meta['weather'][weather]['cars'][a]['pose'][:2])-
                                        meta['weather'][weather]['cars'][b]['pose'][:2])
                    dst.append((scene,change[a,ids],change[b,ids],swapped[b,ids],
                                change[b,(ids+37)%(h*w)],clean[b].reshape(d,-1).T[ids],float(dist)))
            k=(weather,scale)
            if k not in prior:
                prior[k]=(scene,change[1].copy())


def feature_stats(output, prior):
    results={}
    for (weather,scale,region),rows in output.items():
        scenes=np.concatenate([np.repeat(row[0],len(row[1])) for row in rows])
        col=lambda i:np.concatenate([row[i] for row in rows])
        a,b=col(1),col(2)
        first_scene, first=prior[weather,scale]
        matched=scenes!=first_scene
        crossed=first[np.arange(matched.sum())%len(first)]
        result={'same':correlation_blocks(a,b,scenes),
                'weather_swapped':correlation_blocks(a,col(3),scenes),
                'spatial_shift':correlation_blocks(a,col(4),scenes),
                'clean_reference_control':correlation_blocks(a,col(5),scenes),
                'cross_scene':correlation_blocks(a[matched],crossed,scenes[matched]),
                'vehicle_pairs':len(rows)}
        distance=np.concatenate([np.repeat(row[6],len(row[1])) for row in rows])
        for label,mask in [('within_30m',distance<30),('beyond_30m',distance>=30)]:
            result[label]=correlation_blocks(a[mask],b[mask],scenes[mask])
        if len(a)>2:
            cross=(a-a.mean(0)).T@(b-b.mean(0))/(len(a)-1)
            result['cross_covariance_singular_values']=np.linalg.svd(cross,compute_uv=False).tolist()
        results[f'{weather}/{scale}/{region}']=result
    return results


def error_stats(rows, mode):
    if not rows: return dict(n=0,scenes=0)
    a,b,scenes=zip(*[(r[mode][0],r[mode][1],r['scene']) for r in rows])
    a,b=np.asarray(a),np.asarray(b)
    stats=correlation_blocks(a,b,scenes)
    stats['targets']=len({(r['scene'],r['frame'],r['target']) for r in rows})
    if len(a)>2:
        cov=(a-a.mean(0)).T@(b-b.mean(0))/(len(a)-1)
        stats.update(cross_covariance=cov.tolist(), sign_agreement=(np.sign(a)==np.sign(b)).mean(0).tolist(),
                     centered_within_scene=correlation_blocks(
                         np.concatenate([a[np.asarray(scenes)==s]-a[np.asarray(scenes)==s].mean(0) for s in sorted(set(scenes))]),
                         np.concatenate([b[np.asarray(scenes)==s]-b[np.asarray(scenes)==s].mean(0) for s in sorted(set(scenes))]),
                         sorted(scenes)))
    return stats


def predict(rows):
    if not rows: return dict(n=0,scenes=0)
    x=np.asarray([r['predictors'] for r in rows])
    e=np.asarray([r['total'] for r in rows])
    scene=np.asarray([r['scene'] for r in rows])
    results=defaultdict(dict)
    for name in sorted(set(scene)):
        train=scene!=name
        test=~train
        if train.sum()<10: continue
        errors=e[train,:,:2]
        mean=errors.mean((0,1))
        scale=np.maximum(errors.std((0,1)),.1)
        standardized=(errors-mean)/scale
        target=np.clip(standardized[:,0]*standardized[:,1],-5,5)
        observed=np.clip(np.prod((e[test,:,:2]-mean)/scale,axis=1),-5,5)
        fixed=np.broadcast_to(target.mean(0),observed.shape)
        conditional=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(x[train],target).predict(x[test])
        variance=np.maximum(errors.var(0),.01)
        estimate={'independent':np.zeros_like(observed),'fixed':fixed,'conditional':conditional}
        losses={}
        for label,c in estimate.items():
            cross=np.clip(c*scale**2,-.9*np.sqrt(variance.prod(0)),.9*np.sqrt(variance.prod(0)))
            weight=np.clip((variance[1]-cross)/(variance.sum(0)-2*cross),0,1)
            fused=(e[test,0,:2]*weight+e[test,1,:2]*(1-weight))
            losses[label]=((c-observed)**2, fused**2)
            results[label][name]=[float((c-observed).astype(float).__pow__(2).sum()),observed.size]
            results[label+'_oracle_xy_mse'][name]=[float((fused**2).sum()),fused.size]
        for base in ('independent','fixed'):
            for j,field in enumerate(('product_mse','oracle_xy_mse')):
                delta=losses['conditional'][j]-losses[base][j]
                results[f'conditional_minus_{base}_{field}'][name]=[float(delta.sum()),delta.size]
    return dict(n=len(rows),scenes=len(set(scene)),metrics={key:ratio_interval(v) for key,v in results.items()})


def analyze(smoke=False):
    root=OUT/('V3_smoke' if smoke else 'V3')
    root.mkdir(exist_ok=True)
    features=defaultdict(list)
    prior={}
    errors=defaultdict(list)
    counts=defaultdict(Counter)
    history=read_json(lr.OUT/'p0-frozen-failure-v1/COHORT.json')
    with (root/'BOX_ERROR_DETAILS.tmp').open('w',encoding='utf-8') as stream:
        for meta,arr in cache.frames(smoke):
            feature_records(meta,arr,features,prior)
            scene,frame=meta['frame']
            targets=meta['targets']
            for rule_id,(radius,minimum) in enumerate(RULES):
                clean=None
                for weather in WEATHERS:
                    details=meta['weather'][weather]
                    cars=details['cars']
                    matches=[]
                    for i in range(len(cars)):
                        m,c=associate(arr,f'{weather}_{i}',targets,radius,minimum)
                        counts[f'{weather}/{rule_id}'].update(c)
                        matches.append(m)
                    if weather=='clean': clean=matches
                    for a,b in combinations(range(len(cars)),2):
                        usable=matches[a].keys()&matches[b].keys()
                        counts[f'{weather}/{rule_id}']['possible_pair_targets']+=len(targets)
                        counts[f'{weather}/{rule_id}']['both_associated']+=len(usable)
                        for j in sorted(usable):
                            A,B=matches[a][j],matches[b][j]
                            oid=str(targets[j]['object_id'])
                            diff=np.abs(A['box']-B['box'])
                            separation=np.linalg.norm(np.asarray(cars[a]['pose'][:2])-cars[b]['pose'][:2])
                            row=dict(scene=scene,frame=frame,target=oid,vehicles=[cars[a]['id'],cars[b]['id']],
                                     p0=oid in details['multi_reference_hits'],support=details['support'][oid]['source'],
                                     distance=details['support'][oid]['distance'],
                                     historical=oid in history.get(weather,{}).get(str(meta['index']),[]),
                                     iou=[float(A['iou']),float(B['iou'])],
                                     predictors=[float(A['score']),float(B['score']),float(separation),*map(float,diff[:6]),
                                                 float(np.linalg.norm((A['box'][:2]+B['box'][:2])/2))],
                                     total=[A['error'].tolist(),B['error'].tolist()])
                            if weather!='clean' and j in clean[a] and j in clean[b]:
                                row['induced']=[error(x['error'],clean[k][j]['error']).tolist() for x,k in ((A,a),(B,b))]
                            errors[weather,rule_id].append(row)
                            stream.write(json.dumps(dict(row,weather=weather,rule=rule_id))+'\n')
            if (meta['index']+1)%100==0: print('V3',meta['index']+1,flush=True)
    (root/'BOX_ERROR_DETAILS.tmp').replace(root/'BOX_ERROR_DETAILS.jsonl')
    feature=feature_stats(features,prior)
    statistics={}
    prediction={}
    for (weather,rule),rows in errors.items():
        total={'all':rows}
        for key in ('support','distance','p0','historical'):
            for value in {row[key] for row in rows}:
                total[f'{key}/{value}']=[r for r in rows if r[key]==value]
        statistics[f'{weather}/{rule}']={key:dict(total=error_stats(vals,'total'),
            induced=error_stats([r for r in vals if 'induced' in r],'induced')) for key,vals in total.items()}
        if rule==1:
            prediction[weather]=predict(rows)
    boxes=dict(errors=statistics,counts=counts,rules=RULES)
    for name,obj in [('FEATURE_CORRELATIONS',feature),('BOX_ERROR_CORRELATIONS',boxes),('PREDICTABILITY',prediction)]:
        atomic_json(root/(name+'.json'),obj)
    lines=['# V3 跨车误差诊断','','框误差顺序为 x/y/z/h/w/l/朝向。目标身份通过宽松离线关联，仅作真值辅助诊断。','',
           '| 天气/规则 | 可关联目标对 | 候选可能缺失的目标对 | 总误差相关 | 天气诱导误差相关 |','|---|---:|---:|---|---|']
    for key,value in statistics.items():
        c=counts[key]
        lines.append(f"| {key} | {c['both_associated']} | {c['possible_pair_targets']-c['both_associated']} | {value['all']['total'].get('r')} | {value['all']['induced'].get('r')} |")
    lines+=['','留出场景的独立、固定和条件预测见 PREDICTABILITY.json。']
    (root/'CORRELATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(features=feature,boxes=boxes,prediction=prediction)
