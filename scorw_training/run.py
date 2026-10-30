"""Serial preflight -> independent S0/S1/S2 -> Validate -> lock -> Test -> report."""
import argparse,os,platform,traceback
from pathlib import Path
from scorw_diagnostics.core import REPO,read,write,text,digest,identity,lock,stage

HERE=Path(__file__).resolve().parent


def sources():
    pins=read(HERE/'source_snapshot.json')['sources']
    for name,sha in pins.items():
        if digest(REPO/name)!=sha:raise ValueError('Source mismatch: '+name)
    native={p.relative_to(REPO).as_posix():digest(p) for d in ('opencood','attfuse_code','attfuse_gspr','scorw_dataset','scorw_diagnostics') for p in sorted((REPO/d).rglob('*.py')) if '__pycache__' not in p.parts}
    return dict(delivered=pins,actual_server=native,git_required=False)


def preflight(c,h,out):
    import numpy as np,torch
    from .data import Data,schedule
    from .model import create,state_hash
    if read(Path(c['dataset'])/'VALIDATION_GATES.json')['status']!='DATASET_COMPLETE':raise ValueError('Dataset incomplete')
    if not torch.cuda.is_available():raise ValueError('GPU required; no silent CPU fallback')
    identities={};counts={}
    for split,count in [('train',6374),('validate',1980),('test',2170)]:
        manifest=read(Path(c['dataset'])/'manifests'/(split.upper()+'_SELECTION.json'))
        if manifest['frames']!=count:raise ValueError('Dataset scope changed')
        counts[split]=dict(frames=count,clouds=len(manifest['rows']))
        identities[split]=digest(Path(c['dataset'])/'manifests'/(split.upper()+'_SELECTION.json'))
        # Read point/metadata bytes, never Test target annotations during preflight.
        for i,r in enumerate(manifest['rows'],1):
            receipt=read(Path(c['dataset'])/'records'/split/r['scene']/'clouds'/r['cav']/r['timestamp']/'RESULT.json')
            for p,sha in [(Path(r['pcd']),r['pcd_sha256']),(Path(r['yaml']),r['yaml_sha256']),
                          (Path(c['dataset'])/'fog'/r['relative'],receipt['output_sha256'])]:
                if digest(p)!=sha:raise ValueError('Frozen input hash changed '+str(p))
            if i%1000==0:print('INPUT %s %d/%d'%(split,i,len(manifest['rows'])),flush=True)
    for epoch in range(c['epochs']):
        a,w=schedule(c,epoch,6374)
        if sorted(a)!=list(range(6374)) or sum(w)!=3187:raise ValueError('Unbalanced frame schedule')
    model=create(c,h,'S0');base={k:v.detach().clone() for k,v in model.state_dict().items()}
    from .training import save_torch
    save_torch(out/'INITIAL_BACKBONE.pth',base)
    for group in ('S1','S2'):
        m=create(c,h,group,base)
        if any(not torch.equal(m.state_dict()[k],v) for k,v in base.items()):raise ValueError('Initial backbone not equal')
        del m
    from .tests import checks
    checks(c,out)
    # Two independent constructions must yield identical S1/S2 data, augmentation and voxel slots.
    a=Data(c,h,'train',True);b=Data(c,h,'train',True)
    fa=a.frame(0,'fog');fb=b.frame(0,'fog')
    if fa['input_hash']!=fb['input_hash'] or fa['active']!=fb['active']:raise ValueError('Mixed data pairing failed')
    for x,y in zip(fa['features'],fb['features']):
        if any(not np.array_equal(x[k],y[k]) for k in x):raise ValueError('Mixed voxel pairing failed')
    write(out/'PREFLIGHT.json',dict(status='passed',counts=counts,input_manifests=identities,
          initial_backbone_sha256=state_hash(base),mixed_frame_weather_exact_half=True,mixed_voxels_exact=True,
          optimizer=c['optimizer'],augmentation=h['data_augment'],GSPR='original attfuse_gspr; no point targets; keep/evidence regularizers only',
          GPU=torch.cuda.get_device_name(0),torch=torch.__version__,hip=torch.version.hip,selection=c['selection'],test_GT_interpreted=False))


def summarize(c,out,smoke):
    results={};selected={}
    for v in ('S0','S1','S2'):
        selected[v]=read(out/v/'SELECTED.json')
        results[v]={'validate':read(out/'validate'/v/'RESULT.json')}
        if not smoke:results[v]['test']=read(out/'test'/v/'RESULT.json')
    delta={}
    for split in ('validate',) if smoke else ('validate','test'):
        delta[split]={}
        for name,a,b in [('S1-S0','S1','S0'),('S2-S1','S2','S1')]:
            delta[split][name]={w:{q:100*(results[a][split]['metrics']['F'][w][q]['AP']-results[b][split]['metrics']['F'][w][q]['AP'])
                                   for q in ('AP30','AP50','AP70')} for w in ('clean','fog')}
    far={}
    for v,sp in results.items():
        far[v]={}
        for split,r in sp.items():
            far[v][split]={}
            for weather in ('clean','fog'):
                buckets=[b for b in r['metrics']['F'][weather]['AP70']['by_distance'] if b['low']>=40]
                tp=sum(b['TP'] for b in buckets);gt=sum(b['GT'] for b in buckets)
                far[v][split][weather]=dict(TP=tp,GT=gt,recall=tp/gt if gt else None)
    consistency={}
    if not smoke:
        for comparison in ('S1-S0','S2-S1'):
            consistency[comparison]={w:{q:dict(validate_delta_pp=delta['validate'][comparison][w][q],
                test_delta_pp=delta['test'][comparison][w][q],same_direction=(delta['validate'][comparison][w][q]*delta['test'][comparison][w][q]>0))
                for q in ('AP30','AP50','AP70')} for w in ('clean','fog')}
    value=dict(status='SMOKE_COMPLETE' if smoke else 'COMPLETE',selected=selected,results=results,delta_percentage_points=delta,
               far_40m_and_beyond_AP70_recall=far,validation_test_direction=consistency,
               single_seed=c['seed'],REAL_WORLD_VALIDATION_MISSING=True,smoke_not_scientific_result=smoke)
    write(out/'FINAL_RESULTS.json',value)
    lines=['# SCOR-W S0–S2训练与评测','',value['status'],'','|组别|划分|天气|协议|AP30|AP50|AP70|','|---|---|---|---|---:|---:|---:|']
    for v,sp in results.items():
        for split,r in sp.items():
            for protocol,weathers in r['metrics'].items():
                for w,ms in weathers.items():lines.append('|%s|%s|%s|%s|%.10f|%.10f|%.10f|'%(v,split,w,protocol,*(ms[q]['AP'] for q in ('AP30','AP50','AP70'))))
    lines+=['','|划分|对照|天气|AP30变化/pp|AP50变化/pp|AP70变化/pp|','|---|---|---|---:|---:|---:|']
    for split,comparisons in delta.items():
        for name,weathers in comparisons.items():
            for weather,ms in weathers.items():lines.append('|%s|%s|%s|%+.6f|%+.6f|%+.6f|'%(split,name,weather,ms['AP30'],ms['AP50'],ms['AP70']))
    lines+=['','|组别|划分|天气|40m及以上TP|GT|AP70召回率|','|---|---|---|---:|---:|---:|']
    for v,sp in far.items():
        for split,weathers in sp.items():
            for weather,a in weathers.items():lines.append('|%s|%s|%s|%d|%d|%s|'%(v,split,weather,a['TP'],a['GT'],'null' if a['recall'] is None else '%.8f'%a['recall']))
    lines+=['','S1−S0及S2−S1的百分点差、分距离召回、Validate/Test方向记录和标注分母见FINAL_RESULTS.json。',
            'F使用固定Clean GT和按帧累计AP；补充项为实际原生点云处理、GT及推理流程，原生通信前截断与点顺序规则保留。',
            '单seed；真实雾天校验缺失。SMOKE不读取Test GT，不能作为训练效果结论。']
    text(out/'FINAL_REPORT.md','\n'.join(lines)+'\n')


def execute(mode,resume,gpu):
    if platform.system()!='Linux' or REPO!=Path('/home/cjm/OpenCOOD-main/cjmnet'):raise ValueError('Research runs only on designated server')
    import torch
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from .model import hypes,create
    from .training import train,load
    from .evaluate import evaluate
    c=read(HERE/'config.json');h=hypes(c);src=sources();out=Path(c['output'])/mode
    dataset_ids={name:digest(Path(c['dataset'])/name) for name in ('DATASET_PROTOCOL.json','VALIDATION_GATES.json',
        'manifests/TRAIN_SELECTION.json','manifests/VALIDATE_SELECTION.json','manifests/TEST_SELECTION.json')}
    request=dict(config=c,hypes=h,sources=src,mode=mode,gpu=gpu,torch=torch.__version__,dataset_identities=dataset_ids,
                 cudnn_benchmark=False,cudnn_deterministic=True)
    # YAML parser inserts numpy arrays: normalize before canonical identity.
    import json
    request=json.loads(json.dumps(request,default=lambda x:x.tolist()))
    with lock(out):
        contract=dict(identity=identity(request),request=request);p=out/'CONTRACT.json'
        if p.exists():
            if not resume or read(p)!=contract:raise ValueError('Resume contract/source/config mismatch')
        else:
            if any(v.name!='.lock' for v in out.iterdir()):raise ValueError('Unregistered nonempty output')
            write(p,contract)
        parent=contract['identity'];active='preflight';gates=dict(status='RUNNING',mode=mode,stages={})
        def phase(name,callback):
            nonlocal active
            active=name;print('START '+name,flush=True)
            folder=out/name;stage(folder,dict(parent=parent,name=name),lambda:callback(folder))
            gates['stages'][name]='passed';write(out/'VALIDATION_GATES.json',gates);print('PASS '+name,flush=True)
        try:
            phase('preflight',lambda p:preflight(c,h,p))
            if mode=='preflight':gates['status']='PREFLIGHT_COMPLETE'
            else:
                base=load(out/'preflight/INITIAL_BACKBONE.pth');smoke=mode=='smoke'
                # Smoke never opens test GT. All groups train independently from shared raw initialization.
                for v in ('S0','S1','S2'):
                    def training_phase(p,v=v):
                        chosen=train(c,h,v,p,base,parent,smoke)
                        if smoke:
                            restored=train(c,h,v,p,base,parent,True)
                            if chosen!=restored:raise ValueError('Completed training resume changed selection')
                            write(p/'RESUME_TEST.json',dict(status='passed',selected_same=True))
                    phase(v,training_phase)
                # Verify S1/S2 exact paired samples before any formal test.
                for epoch in range(1,(1 if smoke else c['epochs'])+1):
                    a=read(out/'S1/epochs'/('%02d'%epoch)/'SAMPLING.json');b=read(out/'S2/epochs'/('%02d'%epoch)/'SAMPLING.json')
                    if [r['frames'] for r in a['rows']]!=[r['frames'] for r in b['rows']]:raise ValueError('S1/S2 weather/input ordering differs')
                selections={v:read(out/v/'SELECTED.json') for v in ('S0','S1','S2')}
                for split in ('validate','test'):
                    if split=='test' and smoke:break
                    if split=='test':
                        for v,r in selections.items():
                            if digest(r['checkpoint'])!=r['sha256']:raise ValueError('Selected checkpoint changed before Test')
                        write(out/'MODEL_LOCK.json',dict(parent=parent,selection_rule=c['selection'],models=selections,Test_not_used_for_selection=True))
                    for v in ('S0','S1','S2'):
                        def evaluate_selected(p,v=v,split=split):
                            chosen=selections[v]
                            if digest(chosen['checkpoint'])!=chosen['sha256']:raise ValueError('Selected weights changed')
                            model=create(c,h,v,base);model.load_state_dict(load(chosen['checkpoint']),strict=True);model.cuda()
                            evaluate(c,h,model,split,p,identity([parent,v,split,chosen['sha256']]),c['smoke_eval_frames'] if smoke else 0,True)
                            model.cpu();del model;torch.cuda.empty_cache()
                        phase(split+'/'+v,evaluate_selected)
                    gt=[read(out/split/v/'RESULT.json')['GT_identity'] for v in ('S0','S1','S2')]
                    if gt[0]!=gt[1] or gt[1]!=gt[2]:raise ValueError('Cross-model evaluation GT differs')
                summarize(c,out,smoke);gates['status']='SMOKE_COMPLETE' if smoke else 'COMPLETE'
            write(out/'VALIDATION_GATES.json',gates)
        except BaseException as e:
            gates.update(status='FAILED',stage=active,reason=str(e));write(out/'VALIDATION_GATES.json',gates)
            text(out/'FAILURE_TRACEBACK.txt',traceback.format_exc());raise
        finally:print('Reports: %s; status=%s'%(out,gates['status']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','smoke','full'],required=True);p.add_argument('--resume',action='store_true');p.add_argument('--gpu',required=True)
    a=p.parse_args();execute(a.mode,a.resume,a.gpu)
