"""Measured results only; E0/E1/E2 comparisons use identical fixed target identities."""
import json
import math
from collections import defaultdict
from pathlib import Path
from auditlib import atomic_json,complete,read_json,sha256
from bridge import load

VARIANTS={'E1':'E1','E2':'E2'}
write_evaluation=load('original_evaluation_report','gspr_repe/results.py').write_evaluation

def stage1(root):
    import torch
    tests=complete(root/'self_test/summary.json')
    ref=complete(root/'baseline_reference.json')
    rows={}
    for v in VARIANTS:
        folder=root/'smoke'/v
        cp=complete(folder/'smoke_checkpoint.json');hist=read_json(folder/'history.json')['epochs']
        ev=complete(folder/'evaluation_smoke12/summary.json')
        init=complete(folder/'initialization_audit.json')
        mechanism=complete(folder/'diagnostic_smoke2/summary.json')
        if (not mechanism['model_and_BN_unchanged'] or mechanism['optimization_steps']!=0 or
            mechanism['contract']['indices']!=[0,828] or not mechanism['contract']['smoke_only']):
            raise ValueError('Incomplete real mechanism switch smoke: '+v)
        if (len(hist)!=1 or hist[0]['train_updates']!=2 or len(hist[0]['repe'])!=2 or
            ev['frame_count']!=12 or ev['conditions']!=['clean','fog','snow'] or
            ev['checkpoint_sha256']!=cp['checkpoint_sha256'] or sha256(cp['checkpoint'])!=cp['checkpoint_sha256'] or
            not init['shared_tensors_exact'] or init['trained_E0_weight_used'] or
            init['initial_state_sha256']!=ref['initial_state_sha256']):
            raise ValueError('Incomplete real-data smoke/initialization: '+v)
        diag=hist[0]['repe']
        if v=='E1':
            if not any(any(k.startswith('s.') and x>0 for k,x in r['gradient_l2'].items()) for r in diag):
                raise ValueError('S never received a real-data gradient')
            if not any(any(k.startswith('s.') and x>0 for k,x in r['update_l2'].items()) for r in diag):
                raise ValueError('S parameters never updated on real data')
        else:
            if not all(any(x>0 for x in r['C_gradient_l2'].values()) for r in diag):raise ValueError('Missing real C gradients')
            if not all(any(x>0 for x in r['C_update_l2'].values()) for r in diag):raise ValueError('Missing real C updates')
            if not any(r['propagation']['empty']['nonzero']>0 for r in diag):raise ValueError('C never enhanced an empty real grid')
        state=torch.load(hist[0]['resume_state'],map_location='cpu',weights_only=False)
        if state['epoch']!=1 or not state['optimizer']['state'] or not all(k in state for k in ('model','scheduler','criterion','accumulators','rng')):
            raise ValueError('Missing full real smoke resume state')
        rows[v]=dict(checkpoint=cp,initialization=init,train_updates=2,evaluation=ev,mechanism_smoke=mechanism,
            full_state_saved=True,diagnostics=diag)
    atomic_json(root/'stage1/acceptance.json',dict(status='complete',tests=tests,smoke=rows,
        stage2_allowed=True,E0_retrained=False))

def final(root):
    from baseline import FAIR_FIELDS
    from evaluation import selection
    (root/'reports').mkdir(exist_ok=True)
    ref=complete(root/'baseline_reference.json')
    folders={'E0':Path(ref['evaluation_root']),**{v:root/v/'final_evaluation' for v in VARIANTS}}
    summaries={v:complete(f/'summary.json') for v,f in folders.items()}
    proto=read_json(folders['E0']/'protocol.json')
    fairness={}
    for v in VARIANTS:
        c=read_json(root/v/'contract.json')
        for k in FAIR_FIELDS:
            if c[k]!=ref['original_training_contract'][k]:raise ValueError('Final fairness mismatch: '+v+':'+k)
        if c['e0_initialization']['initial_state_sha256']!=ref['initial_state_sha256']:
            raise ValueError('Initial A1 state mismatch')
        chosen=selection(root/v)
        p=read_json(folders[v]/'protocol.json');s=summaries[v]
        if (s['checkpoint_sha256']!=chosen['checkpoint_sha256'] or s['frame_count']!=829 or
            s['conditions']!=['clean','fog','snow'] or not s['full_validation'] or
            p['protocols']!=proto['protocols'] or p['input_manifest_identity']!=proto['input_manifest_identity']):
            raise ValueError('Formal evaluation/selection mismatch: '+v)
        fairness[v]=dict(contract_sha256=sha256(root/v/'contract.json'),initialization=complete(root/v/'initialization_audit.json'),
            selected=chosen,mechanism=complete(root/'mechanism'/v/'summary.json'))
    gt_root=Path(read_json(root/'run_identity.json')['roots']['gt_audit_root'])
    gt=[json.loads(l) for l in (gt_root/'per_frame.jsonl').read_text().splitlines()]
    strata=defaultdict(lambda:dict(gt=0,tp=0));matches={};counts={}
    for v,folder in folders.items():
        for line in (folder/'per_frame_metrics.jsonl').open():
            row=json.loads(line);w=row['condition'];i=row['index'];frame=gt[i]
            if row['frame_key']!=frame['frame_key']:raise ValueError('Formal target frame identity mismatch')
            pairs=row['protocols']['F']['matched_pairs']['0.7']
            ids={str(p['object_id']) for p in pairs}
            matches[v,w,i]={str(p['object_id']):p['iou'] for p in pairs}
            counts[v,w,i]=row['protocols']['F']['thresholds']['0.7']
            for target in frame['conditions']['clean']['final_gt']:
                oid=str(target['object_id']);corners=target['corners']
                distance=math.hypot(*(sum(c[k] for c in corners)/8 for k in (0,1)))
                native=[]
                for cav in frame['conditions'][w]['cavs']:
                    # Original audit IDs may be numeric or strings; normalize identifiers only.
                    before=[str(s) for s in cav['before_point_filter_ids']]
                    if oid in before:native.append(cav['official_points_in_box'][before.index(oid)])
                n=sum(native)
                for group,b in [('target_distance_m','0-30' if distance<30 else '30-60' if distance<60 else '60+'),
                    ('target_native_points_sum_cavs','0' if n==0 else '1-5' if n<=5 else '6-20' if n<=20 else '21+')]:
                    item=strata[v,w,group,b];item['gt']+=1;item['tp']+=oid in ids
    changes=[]
    for v in VARIANTS:
        for w in ('clean','fog','snow'):
            for i in range(829):
                left,right=matches[v,w,i],matches['E0',w,i]
                changes.append(dict(contrast=v+'-E0',condition=w,index=i,frame_key=gt[i]['frame_key'],
                    recovered=sorted(set(left)-set(right)),lost=sorted(set(right)-set(left)),
                    common_iou_delta={k:left[k]-right[k] for k in left.keys()&right.keys()},
                    fp_delta=counts[v,w,i]['fp']-counts['E0',w,i]['fp'],
                    fn_delta=counts[v,w,i]['fn']-counts['E0',w,i]['fn']))
    delta={v+'-E0':{w:{p:{k:100*(summaries[v]['results'][w]['AP'][p][k]-summaries['E0']['results'][w]['AP'][p][k])
        for k in ('ap30','ap50','ap70')} for p in ('O','O*','F')} for w in ('clean','fog','snow')} for v in VARIANTS}
    old=ref['historical_B0_B2']
    interaction={w:{p:{k:delta['E2-E0'][w][p][k]-100*(old['B2']['evaluation']['results'][w]['AP'][p][k]
        -old['B0']['evaluation']['results'][w]['AP'][p][k]) for k in ('ap30','ap50','ap70')}
        for p in ('O','O*','F')} for w in ('clean','fog','snow')}
    histories={'E0':read_json(Path(ref['e0_root'])/'history.json'),**{v:read_json(root/v/'history.json') for v in VARIANTS}}
    traces={v:[r['trace_sha256'] for r in h['epochs']] for v,h in histories.items()}
    report=dict(status='complete',E0_retrained=False,baseline_reference=ref,results=summaries,AP_delta_pp=delta,
        fairness=fairness,sampling_trace_equal=all(t==traces['E0'] for t in traces.values()),
        C_interaction_difference_pp=interaction,
        C_interaction_definition='(E2-E0)-(old B2-old B0), matched-recipe single-seed descriptive difference; no significance claim',
        strata=[dict(model=k[0],weather=k[1],group=k[2],bin=k[3],**a,recall=a['tp']/a['gt']) for k,a in strata.items()],
        target_changes=changes,histories=histories,
        limitations=['single seed','validation also used for selection; no test evaluation',
            'mechanism controls are fixed-weight inference interventions, not a retrained control',
            'target 1-5 points is native point count summed across CAVs, not pillar occupancy',
            'E2-E0 measures C added to A1; strict interaction against B2 alone also requires matching initialization/recipe audit'])
    atomic_json(root/'reports/FINAL_RESULTS.json',report)
    lines=['# E0/E1/E2正式结果','','AP单位%，差值单位百分点；未据Snow成绩调参。','',
        '| 模型 | 天气 | 协议 | AP30 | AP50 | AP70 |','|---|---|---|---:|---:|---:|']
    for v,s in summaries.items():
        for w,row in s['results'].items():
            for p,a in row['AP'].items():lines.append('| '+v+' | '+w+' | '+p+' | '+' | '.join(f'{100*a[k]:.6f}' for k in ('ap30','ap50','ap70'))+' |')
    lines+=['','逐目标恢复/丢失、距离/目标点数召回、门控开关及传播统计见FINAL_RESULTS.json和mechanism/。',
        '实际训练采样轨迹一致：'+str(report['sampling_trace_equal'])]
    (root/'reports/FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
