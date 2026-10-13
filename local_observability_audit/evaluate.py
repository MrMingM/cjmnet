"""Predeclared H1/H2/H3 analyses; all failures/cohort exclusions remain counted."""
import numpy as np
from .observability import complementarity
from .statistics import (rho, association, incremental, matched_correlation,
                         view_shuffle, null_summary, ridge_oof, performance)

def variants(row,matrix='geo',variant='count_normalized'):
    return row['observability'][matrix]['variants'][variant]

def scalar(row,matrix,cfg):
    m=variants(row,matrix,'raw')
    return [np.log1p(row['point_count']),np.log1p(row['valid_point_count']),
        row['mean_gspr_reliability'],row['mean_gspr_uncertainty'],
        np.log1p(m['trace']),m['logdet_regularized'],row['source_distance'],
        *[float(row['weather']==w) for w in cfg['weathers'][1:]],
        row['gt_box'][3],row['gt_box'][4],np.cos(2*row['gt_box'][6]),np.sin(2*row['gt_box'][6])]

def evaluate_h1(rows,cfg):
    result={}
    rng=np.random.default_rng(cfg['seed']+1)
    for name in ('geo','gspr'):
        usable=[r for r in rows if variants(r,name)['directional_valid']]
        strata={}
        for weather in ('pooled',*cfg['weathers']):
            rr=[r for r in usable if weather=='pooled' or r['weather']==weather]
            strata[weather]={'rows':len(rr), 'rank_histogram':{str(k):sum(variants(r,name)['rank']==k for r in rr) for k in range(6)},
                'median_eigenvalues':np.median([variants(r,name)['eigenvalues'] for r in rr],axis=0) if rr else None,
                'median_anisotropy':float(np.median([variants(r,name,'shape')['anisotropy'] for r in rr])) if rr else None}
        x=np.asarray([[np.cos(2*r['viewing_angle']),np.sin(2*r['viewing_angle']),
                       *scalar(r,name,cfg)] for r in usable])
        y=np.asarray([variants(r,name,'shape')['diagonal'] for r in usable])
        scenes=[r['scene_id'] for r in usable]
        view={'valid':False,'reason':'insufficient rows/scenes'}
        if len(usable)>=cfg['minimum_rows'] and len(set(scenes))>=cfg['minimum_scenes']:
            full=ridge_oof(x,y,scenes,cfg['ridge_alpha'])
            base=ridge_oof(x[:,2:],y,scenes,cfg['ridge_alpha'])
            def score(a,b):
                ss=np.sum((a-a.mean(0))**2)
                return float(1-np.sum((a-b)**2)/ss) if ss>1e-14 else None
            rf,rb=score(y,full),score(y,base)
            null=[]
            for _ in range(cfg['permutation_replicates']):
                xp=x.copy(); xp[:,:2]=x[view_shuffle(usable,cfg,rng),:2]
                value=score(y,ridge_oof(xp,y,scenes,cfg['ridge_alpha']))
                null.append(value)
            view={'valid':rf is not None,'r2':rf,'scalar_r2':rb,
                  'delta_r2':rf-rb if rf is not None and rb is not None else None,
                  'null':null_summary(null,rf,cfg)}
        result[name]={'strata':strata,'view_relation':view,
                     'warning':'Anisotropy alone partly follows from the fixed-face model; H2/H3 required.'}
    return result

def evaluate_h2(rows,cfg):
    result={}; controls={}
    rng=np.random.default_rng(cfg['seed']+2)
    for matrix in ('geo','gspr'):
        for variant in ('raw','count_normalized','shape'):
            key=matrix+'/'+variant; result[key]={}
            rr=[r for r in rows if variants(r,matrix,variant)['directional_valid'] and r['mean_gspr_reliability'] is not None]
            diag=np.asarray([variants(r,matrix,variant)['diagonal'] for r in rr]).reshape(-1,5)
            for k,param in enumerate(cfg['state']):
                entries={}
                for weather in ('pooled',*cfg['weathers']):
                    ids=[i for i,r in enumerate(rr) if weather=='pooled' or r['weather']==weather]
                    subset=[rr[i] for i in ids]
                    y=np.asarray([r['utility'][k] for r in subset]); x=diag[ids,k]
                    scenes=[r['scene_id'] for r in subset]
                    baselines={}
                    for feature in ('point_count','valid_point_count','mean_gspr_reliability','mean_gspr_uncertainty','trace','logdet_regularized'):
                        values=[variants(r,matrix,'raw')[feature] if feature in ('trace','logdet_regularized') else r[feature] for r in subset]
                        baselines[feature]=association(values,y,scenes,cfg,cfg['seed']+k)
                    entries[weather]={'directional':association(x,y,scenes,cfg,cfg['seed']+k), 'scalar_baselines':baselines}
                y=np.asarray([r['utility'][k] for r in rr]); scenes=[r['scene_id'] for r in rr]
                entries['incremental']=incremental([scalar(r,matrix,cfg) for r in rr],diag[:,k:k+1],y,scenes,cfg,cfg['seed']+k)
                entries['point_count_matched']=matched_correlation(diag[:,k],y,rr,cfg,'valid_point_count')
                entries['reliability_matched']=matched_correlation(diag[:,k],y,rr,cfg,'mean_gspr_reliability')
                result[key][param]=entries
                if variant=='count_normalized':
                    shuffled,rotated=[],[]
                    for _ in range(cfg['permutation_replicates']):
                        shuffled.append(rho(diag[view_shuffle(rr,cfg,rng),k],y))
                        # Conjugation by random axis permutation preserves every eigenvalue exactly.
                        columns=rng.integers(0,5,len(rr))
                        rotated.append(rho(diag[np.arange(len(rr)),columns],y))
                    observed=rho(diag[:,k],y)
                    controls[matrix+'/'+param]={'view_permutation':null_summary(shuffled,observed,cfg),
                                               'eigenvector_permutation':null_summary(rotated,observed,cfg)}
            print('H2 analyzed',key,'rows=',len(rr),flush=True)
    return result,controls

def pair_base(r,cfg,matrix):
    a,b=r['a'],r['b']
    qa,qb=scalar(a,matrix,cfg),scalar(b,matrix,cfg)
    # Total counts, both qualities, uncertainty, distance and weather, symmetric under source swap.
    return [np.log1p(a['point_count']+b['point_count']),np.log1p(a['valid_point_count']+b['valid_point_count']),
        (a['mean_gspr_reliability']+b['mean_gspr_reliability'])/2,
        abs(a['mean_gspr_reliability']-b['mean_gspr_reliability']),
        (a['mean_gspr_uncertainty']+b['mean_gspr_uncertainty'])/2,
        (a['source_distance']+b['source_distance'])/2,abs(a['source_distance']-b['source_distance']),
        qa[4]+qb[4],abs(qa[4]-qb[4]),qa[5]+qb[5],abs(qa[5]-qb[5]),
        *[float(a['weather']==w) for w in cfg['weathers'][1:]],
        a['gt_box'][3],a['gt_box'][4],np.cos(2*a['gt_box'][6]),np.sin(2*a['gt_box'][6])]

def evaluate_h3(rows,cfg):
    result={}; controls={}; rng=np.random.default_rng(cfg['seed']+3)
    for matrix in ('geo','gspr'):
        rr=[r for r in rows if all(variants(r[t],matrix)['directional_valid'] and r[t]['mean_gspr_reliability'] is not None for t in ('a','b'))]
        scenes=[r['a']['scene_id'] for r in rr]; y=np.asarray([r['gain'] for r in rr])
        base=np.asarray([pair_base(r,cfg,matrix) for r in rr])
        values={}
        for variant in ('raw','count_normalized','shape'):
            matrices=[(np.asarray(variants(r['a'],matrix,variant)['matrix']),np.asarray(variants(r['b'],matrix,variant)['matrix'])) for r in rr]
            metrics=[complementarity(a,b,cfg['matrix_ridge']) for a,b in matrices]
            for key in ('C_logdet','C_weak'):
                name=variant+'_'+key; x=np.asarray([m[key] for m in metrics]); values[name]=x
                strata={}
                for weather in ('pooled',*cfg['weathers']):
                    ix=[i for i,r in enumerate(rr) if weather=='pooled' or r['a']['weather']==weather]
                    strata[weather]=association(x[ix],y[ix],[scenes[i] for i in ix],cfg,cfg['seed']+3)
                result[matrix+'/'+name]={'associations':strata,
                    'incremental':incremental(base,x[:,None],y,scenes,cfg,cfg['seed']+3),
                    'zero_fraction':float(np.mean(np.abs(x)<1e-10)) if len(x) else None}
            if variant=='shape':
                null=[]
                for _ in range(cfg['permutation_replicates']):
                    xx=[]
                    for a,b in matrices:
                        p=rng.permutation(5)
                        xx.append(complementarity(a,b[np.ix_(p,p)],cfg['matrix_ridge'])['C_logdet'])
                    null.append(rho(xx,y))
                controls[matrix]={'eigenvector_permutation':null_summary(null,rho(values['shape_C_logdet'],y),cfg)}
        # Parameter-specific pair effects are secondary, with all-three matched cohort unchanged.
        result[matrix+'/parameter_gains']={p:association(values.get('shape_C_logdet',[]),[r['utility'][k] for r in rr],scenes,cfg,cfg['seed']+k)
                                           for k,p in enumerate(cfg['state'])}
        print('H3 analyzed',matrix,'pairs=',len(rr),flush=True)
    return result,controls

def positive_ci(item):
    return bool(item.get('valid') and item.get('ci95') and item['ci95'][0]>0)

def increment_pass(item,cfg):
    return bool(item.get('valid') and positive_ci(item['delta_r2'])
                and item['delta_r2']['estimate']>=cfg['decision']['minimum_delta_r2'] and item['mae_reduction']>0)

def decide(h1,h2,h3,controls,cfg,mode,failed_frames):
    primary=cfg['decision']['primary_matrix']; v=h1[primary]['view_relation']
    anis=h1[primary]['strata']['pooled']['median_anisotropy']
    a=bool(anis is not None and anis>=cfg['decision']['anisotropy_median_min'] and v.get('valid')
           and v.get('delta_r2',-1)>=cfg['decision']['h1_view_r2_min'] and v['null']['above_q95'])
    params=[]
    for p,e in h2[primary+'/'+cfg['decision']['primary_variant']].items():
        positive=sum((e[w]['directional']['estimate'] or 0)>0 for w in cfg['weathers'])
        null=controls['h2'][primary+'/'+p]
        if (positive_ci(e['pooled']['directional']) and positive>=cfg['decision']['majority_weathers']
            and increment_pass(e['incremental'],cfg)
            and (e['point_count_matched'].get('estimate') or 0)>0
            and (e['reliability_matched'].get('estimate') or 0)>0
            and all(null[k]['above_q95'] for k in ('view_permutation','eigenvector_permutation'))):
            params.append(p)
    b=len(params)>=cfg['decision']['multiple_parameters']
    pair=h3[primary+'/'+cfg['decision']['primary_pair_metric']]
    d=bool(positive_ci(pair['associations']['pooled']) and increment_pass(pair['incremental'],cfg)
           and controls['h3'][primary]['eigenvector_permutation']['above_q95'])
    enough=all(e['pooled']['directional'].get('valid',False) for e in h2[primary+'/count_normalized'].values()) and pair['associations']['pooled'].get('valid',False)
    if mode=='smoke':
        verdict='SMOKE_ONLY'
    elif failed_frames or not enough:
        verdict='INCONCLUSIVE'
    elif a and b and d:
        verdict='CONTINUE_RESEARCH'
    elif params:
        verdict='AUXILIARY_ONLY'
    else:
        verdict='STOP_V1_PREMISE'
    return {'verdict':verdict,'A_directionality':a,'BC_parameters':params,'BC_multiple_parameters':b,
            'D_complementarity':d,'adequate_cohort':enough,'failed_frames':failed_frames,
            'scope':'GT-assisted fixed-face surrogate, frozen GSPR+AttFuse, validation development only; no AP claim',
            'stop_scope':'Stop this operational hypothesis under this detector/residual; not a universal impossibility theorem.'}

def evaluate(frames,cfg,mode):
    all_sources=[]; singles=[]; pairs=[]; failures=[]; counts={'regression_valid':0,'regression_invalid':0,'recovered':0,'lost':0,'pair_valid':0,'pair_invalid':0}
    for frame in frames:
        if frame.get('failure'):
            failures.append(frame['failure']); continue
        all_sources.extend(frame['sources'])
        lookup={(r['gt_object_id'],r['source_index']):r for r in frame['sources']}
        for c in frame['contributions']['singles']:
            counts['recovered']+=int(c['recovery']); counts['lost']+=int(c['lost'])
            counts['regression_valid' if c['regression_valid'] else 'regression_invalid']+=1
            if c['regression_valid']:
                singles.append(dict(lookup[c['gt_object_id'],c['source_index']],utility=c['utility']))
        for c in frame['contributions']['pairs']:
            counts['pair_valid' if c['regression_valid'] else 'pair_invalid']+=1
            if c['regression_valid']:
                pairs.append(dict(c,a=lookup[c['gt_object_id'],c['i']],b=lookup[c['gt_object_id'],c['j']]))
    print('Analysis cohorts',counts,flush=True)
    h1=evaluate_h1(all_sources,cfg)
    h2,c2=evaluate_h2(singles,cfg)
    h3,c3=evaluate_h3(pairs,cfg)
    controls={'h2':c2,'h3':c3}
    verdict=decide(h1,h2,h3,controls,cfg,mode,len(failures))
    worst=sorted(singles,key=lambda r:float(np.mean(r['utility']/np.asarray(cfg['parameter_scales']))))[:20]
    return {'decision':verdict,'cohorts':counts,'source_rows':len(all_sources),'H1':h1,'H2':h2,'H3':h3,
            'negative_controls':controls,'frame_failures':failures,
            'failure_cases':[{'scene':r['scene_id'],'frame':r['frame_id'],'weather':r['weather'],
                              'object':r['gt_object_id'],'source':r['source_id'],'utility':r['utility']} for r in worst]}
