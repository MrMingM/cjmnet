"""Independent scoring experiments using the shared manifest."""
import gzip
import json
import os
import sys
import time

import numpy as np
import torch

from v2xr_griq_v2.dataset import F0,OUT,SEEDS,V1,WEATHERS,bind,get,import_runtime,read,root_for,save,sha
from v2xr_griq_v2.train_common import pack,predict,restore,rows_for,train
from v2xr_griq_v2.evaluate import aps,cached_logits,calibrate,formal,paired,quality_metrics
from v2xr_griq_v2.local_evidence import DL
from v2xr_griq_experiments.native import processor
from v2xr_griq_experiments.worker import official_archive_check


def setup_arm(audit, group_dir, name, subset, mode, variant, budget, seed,
              device, smoke, ablation='none', weights=(1,1,1), fold=None):
    parts=rows_for(audit,subset,fold)
    packs={}
    local=variant in ('cross','independent','lidar','radar')
    fold_name='main' if fold is None else 'fold_'+'_'.join(str(audit['split']['fit'].index(s)) for s in fold['fit'])
    for k in ('fit','select'):
        key=subset if k=='fit' else 's32'
        packs[k]=pack(parts[k],audit['policy'],group_dir/'packed'/fold_name/f'{k}_{key}_{"local" if local else "global"}',local)
    # Hold the historical strict-positive proportion constant across nested sizes.
    reference=rows_for(audit,'s32',fold)['fit']
    baseline=pack(reference,audit['policy'],group_dir/'packed'/fold_name/'fit_s32_global',False)
    ratio=float(np.clip(baseline['y'][:,2].mean(),.03,.80))
    return train(packs['fit'],packs['select'],group_dir/'checkpoints'/name,
                 dict(mode=mode,variant=variant,budget=budget,seed=seed,ablation=ablation,
                      weights=list(weights),ratio=ratio),device,smoke)


def contributions(rows,policy,ck,net,device,destination):
    accum=np.zeros((4,6,3),np.float64)
    signed=np.zeros_like(accum)
    n=np.zeros((4,6),int)
    with gzip.open(destination.with_suffix('.tmp'),'wt') as stream:
        for i,row in enumerate(rows):
            d=get(row,policy,True)
            _,changes=predict(net,dict(x=d['x'],l=d['local']),ck,device,seed=ck['seed']+i,contribution=True)
            vals=np.stack([np.concatenate([a for j,a in changes if j==k]) for k in range(6)],1) if len(d['x']) else np.empty((0,6,3))
            l=d['local'][:,:,0]>0
            r=d['local'][:,:,DL]>0
            state=2*l.astype(int)+r.astype(int)
            for k in range(4):
                active=state==k
                n[k]+=active.sum(0)
                accum[k]+=(abs(vals)*active[:,:,None]).sum(0)
                signed[k]+=(vals*active[:,:,None]).sum(0)
            stream.write(json.dumps(dict(key=d['meta']['key'],ids=d['ids'].tolist(),state=state.tolist(),
                                       iou=d['iou'].tolist(),delta=vals.round(5).tolist()),separators=(',',':'))+'\n')
    destination.with_suffix('.tmp').replace(destination)
    return dict(states=['neither','radar_only','lidar_only','both'],count=n.tolist(),
                mean_abs=(accum/n[:,:,None].clip(1)).tolist(),mean=(signed/n[:,:,None].clip(1)).tolist())


def group_run(group,smoke):
    import_runtime()
    from v2xr_griq_v2 import e1_multi_precision as e1,e2_data_scaling as e2,e3_local_complement as e3,e4_stability as e4
    root=root_for(smoke)
    audit=read(root/'DATA_AUDIT.json')
    signature=sha(root/'DATA_AUDIT.json')
    out=root/group
    out.mkdir(exist_ok=True)
    completed=out/'COMPLETE.json'
    if completed.exists():
        done=read(completed)
        if done['data']!=signature or any(sha(out/k)!=v for k,v in done['files'].items()):
            raise ValueError('Completed experiment files changed')
        return
    if not torch.version.hip or torch.cuda.device_count()!=1:
        raise RuntimeError('One visible HaiGuang device required')
    device=torch.device('cuda:0')
    gpu=os.environ['ROCR_VISIBLE_DEVICES']
    if gpu!=str(int(group[-1])):
        raise ValueError('Wrong physical device')
    print('DEVICE',gpu,torch.cuda.get_device_name(device),flush=True)
    pp=processor(F0/'config.yaml')
    start=time.monotonic()
    from v2xr_griq_experiments.training import Scorer
    historical=read(V1/'M1/COMPLETE.json')
    if sha(V1/'M1/best.pt')!=historical['artifacts']['best.pt'] or sha(V1/'M1/CALIBRATION.json')!=historical['artifacts']['CALIBRATION.json']:
        raise ValueError('Historical additive files differ')
    old_ck=torch.load(V1/'M1/best.pt',map_location='cpu',weights_only=False)['fitted']['additive']
    old_model=Scorer(old_ck)
    old_cut=read(V1/'M1/CALIBRATION.json')['arms']['additive']['primary']['prob']
    calibration=rows_for(audit,'s32')['calibrate']
    selection=rows_for(audit,'s32')['select']
    weather_rows={w:[r for r in audit['files'] if r['split']=='validate' and r['weather']==w] for w in WEATHERS}
    policy=audit['policy']
    models,lock,results={}, {}, {}
    base={}
    for w,rows in weather_rows.items():
        z=cached_logits(rows,policy,None,None,device)
        stats,base[w]=formal(rows,policy,z,pp,.2,None,smoke)
        results.setdefault('F0',{})[w]=stats
        if not smoke:
            official_archive_check(stats,w)
    results['F0_threshold']={}
    raw_cal=calibrate(calibration,policy,cached_logits(calibration,policy,None,None,device),pp,smoke)
    lock['F0_threshold']=raw_cal
    if group=='E1':
        results['E1-0']={}
        lock['E1-0']=dict(selected=dict(threshold=old_cut),source=historical['artifacts']['best.pt'])
    for w,rows in weather_rows.items():
        z=cached_logits(rows,policy,None,None,device)
        results['F0_threshold'][w],_=formal(rows,policy,z,pp,raw_cal['selected']['threshold'],base[w],smoke)
        if group=='E1':
            old=[(old_model(get(r,policy)['x'])[0],None) for r in rows]
            results['E1-0'][w],_=formal(rows,policy,old,pp,old_cut,base[w],smoke,details=True)
    primary={'E1':'E1-2','E2':'all-additive-fixed','E3':'E3-4','E4':'cross-20261017'}[group]
    traces={}
    def apply(name,ck,intervention=None,fixed_cut=None):
        arm=out/'evaluations'/f'{name}.json'
        net=restore(ck,device)
        if arm.exists():
            committed=read(arm)
            if committed['signature']!=signature:
                raise ValueError('Evaluation source changed')
            lock[name],results[name]=committed['calibration'],committed['results']
            traces[name]=committed['frames']
            return
        lock[name]=calibrate(calibration,policy,cached_logits(calibration,policy,net,ck,device,intervention),pp,smoke
                            ) if fixed_cut is None else dict(selected=dict(threshold=fixed_cut),paired_fixed_threshold=True)
        cut=lock[name]['selected']['threshold']
        vals={}
        details=(group!='E4' or name in ('cross-20261017','additive-20261017',
                                          'joint_local_capacity-20261017','independent-20261017'))
        all_traces={}
        for w,rows in weather_rows.items():
            tick=time.monotonic()
            zs=cached_logits(rows,policy,net,ck,device,intervention)
            seconds=time.monotonic()-tick
            r,trace=formal(rows,policy,zs,pp,cut,base[w],smoke,details=details)
            r['quality']=quality_metrics(rows,policy,zs)
            r['scoring_seconds']=seconds
            r['candidates']=sum(len(z) for z,_ in zs)
            if group=='E1':
                r['fixed_native_count'],_=formal(rows,policy,zs,pp,cut,base[w],smoke,matched=True)
            if group=='E3' and name=='E3-4':
                r['local_evidence']=contributions(rows,policy,ck,net,device,out/f'{w}_local_contributions.jsonl.gz')
            vals[w]=r
            if group=='E3' and name in ('E3-0','E3-3','E3-4','E3-5'):
                all_traces[w]=[dict(scene=f['scene'],F={str(k):v for k,v in f['F'].items()}) for f in trace]
        results[name]=vals
        traces[name]=all_traces
        bind(arm,dict(signature=signature,calibration=lock[name],results=vals,frames=all_traces))

    def fit_apply(name,subset,mode,variant,budget='converge',seed=SEEDS[0],source=None,ablation='none'):
        print('ARM',group,name,flush=True)
        ck=setup_arm(source or audit,out,name,subset,mode,variant,budget,seed,device,smoke,ablation)
        models[name]=ck
        apply(name,ck)

    if group=='E1':
        for args in e1.arms(smoke):
            fit_apply(*args)
        if not smoke:
            choices=[]
            for w in e1.tuning():
                name='E1-3-'+''.join(map(str,w))
                ck=models['E1-2'] if w==(1,1,1) else setup_arm(audit,out,name,'s32','ordinal','additive',
                                                             'converge',SEEDS[0],device,smoke,weights=w)
                val=calibrate(selection,policy,cached_logits(selection,policy,restore(ck,device),ck,device),pp)
                choices.append((val['selected']['F']['macro'],name,ck,val['selected']))
            _,name,ck,_=max(choices,key=lambda z:(z[0],z[1]=='E1-3-111',z[1]))
            models['E1-3']=ck
            apply('E1-3',ck)
            save(out/'LOSS_SELECTION.json',dict(selected=name,trials={n:a for _,n,_,a in choices}))
    elif group=='E2':
        from v2xr_griq_v2.prepare_shared import prepare
        for args in e2.arms(smoke):
            if args[1]=='s32':
                fit_apply(*args)
        prepare('full',smoke)
        full=read(root/'FULL_DATA.json')
        for args in e2.arms(smoke):
            if args[1]!='s32':
                fit_apply(*args,source=full if args[1]=='all' else audit)
        if smoke:
            primary='s32-additive-fixed'
    elif group=='E3':
        for args in e3.arms(smoke):
            fit_apply(*args)
    else:
        for args in e4.arms(smoke):
            fit_apply(*args)
        if not smoke:
            for kind in ('no_gspr','no_radar','points','permute','flatten_radar','one_cell'):
                fit_apply('ablate_'+kind,'s128','ordinal','cross',ablation=kind)
        cross=models[primary]
        cut=lock[primary]['selected']['threshold']
        for kind in ('permute','flatten_radar','one_cell'):
            apply('intervene_'+kind,cross,kind,cut)
        off=cross.copy()
        off['weights_state']=cross['weights_state'] | {'alpha':torch.zeros_like(cross['weights_state']['alpha'])}
        apply('intervene_residual_off',off,fixed_cut=cut)
        if not smoke:
            cv={}
            for i,fold in enumerate(e4.folds(audit['split'])):
                for variant in ('cross','additive','independent','joint_local_capacity'):
                    name=f'cv{i}_{variant}'
                    ck=setup_arm(audit,out,name,'s128','ordinal',variant,'converge',SEEDS[0],device,False,fold=fold)
                    net=restore(ck,device)
                    cal=calibrate(calibration,policy,cached_logits(calibration,policy,net,ck,device),pp)
                    cut=cal['selected']['threshold']
                    cv[name]=dict(fold=fold,threshold=cut,weather={})
                    for w in ('clean','fog'):
                        rr=[r for r in audit['files'] if r['split']=='train' and r['weather']==w and r['scene'] in fold['test']]
                        value,_=formal(rr,policy,cached_logits(rr,policy,net,ck,device),pp,cut,None)
                        cv[name]['weather'][w]=value
            save(out/'CV_RESULTS.json',cv)
    if group=='E3' and not smoke:
        for name in ('E3-0','E3-3','E3-5'):
            for w in WEATHERS:
                aa,bb=traces['E3-4'][w],traces[name][w]
                for frames in (aa,bb):
                    for f in frames:
                        f['F']={float(k):v for k,v in f['F'].items()}
                results['E3-4'][w].setdefault('mechanism_paired',{})[name]=paired(aa,bb)
    if group=='E4' and not smoke:
        stability={}
        for v in e4.VARIANTS:
            stability[v]={}
            for w in WEATHERS:
                a=np.array([[*aps(results[f'{v}-{s}'][w]['F']),results[f'{v}-{s}'][w]['F']['macro']] for s in SEEDS])
                stability[v][w]=dict(mean=a.mean(0).tolist(),sd=a.std(0,ddof=1).tolist(),per_seed=a.tolist())
        save(out/'STABILITY.json',stability)
    train_meta={n:{k:v for k,v in c.items() if k not in ('weights_state','mean','sd')} for n,c in models.items()}
    if group=='E1':
        train_meta['E1-0']=dict(parameters=old_ck['parameters'],source='v1 official additive')
    save(out/'metrics.json',dict(primary=primary,arms=results,training=train_meta,seconds=time.monotonic()-start,
                                 gpu=gpu,shared=signature))
    save(out/'CALIBRATION.json',lock)
    lines=[f'# {group}', '', '| 天气 | 对照 | F AP30 / AP50 / AP70 | F MacroAP |', '|---|---|---|---:|']
    for n,vals in results.items():
        for w,r in vals.items():
            lines.append(f'| {w} | {n} | '+ ' / '.join(f'{100*x:.3f}' for x in aps(r['F']))+
                         f' | {100*r["F"]["macro"]:.3f} |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and
           p.name not in ('progress.pt',) and p.suffix not in ('.npy','.tmp')}
    bind(completed,dict(data=signature,files=files))
    print('COMPLETE',group,flush=True)


if __name__=='__main__':
    group_run(sys.argv[1],'--smoke' in sys.argv)
