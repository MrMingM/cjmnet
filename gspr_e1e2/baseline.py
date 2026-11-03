"""Read-only E0=A1 audit and pristine E0/B2 initialization. No trained-weight fallback."""
from pathlib import Path
from auditlib import (CONFIG,METHOD,ROOT,SELECTION,SETTINGS,atomic_json,bound_json,
    complete,frozen_c,identity,read_json,sha256)

FAIR_FIELDS=('epochs','policy','train_manifest_sha256','train_manifest_identity','train_data_acceptance',
    'pair_identity','config_sha256','input_identity','frozen_gt','loader','scheduler','training_GT','gspr_frozen_settings')

def check_identity(row):
    if row.get('identity')!=identity({k:v for k,v in row.items() if k!='identity'}):
        raise ValueError('Artifact content identity mismatch')

def verify_initial(folder):
    import torch
    path=folder/'resume_states/00.pth'
    marker=read_json(folder/'resume_states/00.json')
    if sha256(path)!=marker['sha256']:raise ValueError('Pristine initialization hash mismatch')
    s=torch.load(path,map_location='cpu',weights_only=False)
    contract=read_json(folder/'contract.json')
    if s['epoch']!=0 or s['contract']!=contract['identity'] or s['optimizer']['state']:
        raise ValueError('Initialization must be epoch zero with empty optimizer')
    if not all(k in s for k in ('model','optimizer','scheduler','criterion','accumulators','rng')):
        raise ValueError('Incomplete initial training state')
    return path,marker['sha256'],s

def audit(args,manifest,evidence):
    folder=Path(args.e0_root)
    c=read_json(folder/'contract.json');check_identity(c)
    if c['variant']!='A1' or c['smoke_only'] or c['epochs']!=20:
        raise ValueError('E0 must be the complete original A1, never retrained')
    if c['policy']!=SELECTION or c['gspr_frozen_settings']!=SETTINGS:
        raise ValueError('Original A1 protocol/GSPR differs')
    for k in ('seed','epochs','batch_size','workers','sampling','train_frames'):
        if c['settings'][k]!=CONFIG[k]:raise ValueError('A1 training recipe changed: '+k)
    for n,h in c['sources']['files'].items():
        if sha256(ROOT/'gspr_ggca'/n)!=h:raise ValueError('Original A1 source changed: '+n)
    from training import config
    destination=Path(args.output_root);destination.mkdir(parents=True,exist_ok=True)
    config(args,destination)
    if sha256(destination/'config.yaml')!=c['config_sha256'] or sha256(folder/'config.yaml')!=c['config_sha256']:
        raise ValueError('Training YAML differs from E0')
    from data import paired_inventory
    if (c['train_manifest_identity']!=manifest['identity'] or
        c['train_manifest_sha256']!=sha256(Path(args.train_root)/'manifest.json') or
        c['pair_identity']!=paired_inventory(manifest,args.train_root)['identity'] or
        c['input_identity']!=evidence['input_manifest']['identity'] or
        c['frozen_gt']!=frozen_c(args)['gt']['identity']):
        raise ValueError('Training/validation/weather partition differs from E0')
    from checkpoint import committed
    rows=committed(folder,20,c['identity'])
    if len(rows)!=20 or any(r['train_updates']!=1010 for r in rows):raise ValueError('E0 training incomplete')
    if complete(folder/'training_summary.json')['epochs']!=20:raise ValueError('Missing E0 completion')
    choice=complete(folder/'selected.json');check_identity(choice)
    best=min(rows,key=lambda r:(-r['AP70'],r['epoch']))
    if (choice['epoch'],choice['checkpoint_sha256'],choice['clean_O_AP70'])!=(best['epoch'],best['checkpoint_sha256'],best['AP70']):
        raise ValueError('E0 Clean-only selection mismatch')
    if sha256(choice['checkpoint'])!=choice['checkpoint_sha256'] or choice['training_contract_sha256']!=sha256(folder/'contract.json'):
        raise ValueError('E0 checkpoint/contract mismatch')
    ev=folder/'final_evaluation';summary=complete(ev/'summary.json');p=read_json(ev/'protocol.json')
    from support import DEFINITIONS
    if (not summary['full_validation'] or summary['frame_count']!=829 or
        summary['conditions']!=['clean','fog','snow'] or summary['checkpoint_sha256']!=choice['checkpoint_sha256'] or
        p['protocols']!=DEFINITIONS or p['input_manifest_identity']!=evidence['input_manifest']['identity'] or
        summary['frozen_d1c_gt_identity']!=c['frozen_gt']):
        raise ValueError('E0 formal evaluation is incomplete or incompatible; no retraining allowed')
    for n,h in summary['output_hashes'].items():
        if sha256(ev/n)!=h:raise ValueError('E0 evaluation artifact changed: '+n)
    initial,ih,s=verify_initial(folder)
    cpath,ch,cs=verify_initial(Path(args.b2_root))
    cc=read_json(Path(args.b2_root)/'contract.json')
    if cc['variant']!='B2' or cc['smoke_only']:raise ValueError('C initialization must be the original B2 epoch zero')
    for k in FAIR_FIELDS:
        if cc[k]!=c[k]:raise ValueError('Original B2/E0 recipe mismatch: '+k)
    if not any(k.startswith('repe_c.') for k in cs['model']):raise ValueError('Missing original C initialization')
    import torch
    common={k:v for k,v in s['model'].items() if not k.startswith('ggca.')}
    if any(k not in cs['model'] or not torch.equal(v,cs['model'][k]) for k,v in common.items()):
        raise ValueError('E0/B2 pristine common GSPR/backbone initialization differs')
    names=['contract.json','config.yaml','selected.json','history.json','training_summary.json','resume_states/00.json','final_evaluation/summary.json','final_evaluation/protocol.json']
    artifacts={str(folder/n):sha256(folder/n) for n in names}
    artifacts.update({str(initial):ih,str(cpath):ch,str(Path(args.b2_root)/'contract.json'):sha256(Path(args.b2_root)/'contract.json')})
    historical={}
    for label,oldroot in (('B0',Path(args.a0_root)),('B2',Path(args.b2_root))):
        oc=read_json(oldroot/'contract.json');check_identity(oc)
        for key in FAIR_FIELDS:
            if oc[key]!=c[key]:raise ValueError('Historical factorial recipe mismatch: '+label+':'+key)
        oldrows=committed(oldroot,20,oc['identity'])
        selected=complete(oldroot/'selected.json');check_identity(selected)
        bestold=min(oldrows,key=lambda r:(-r['AP70'],r['epoch']))
        if len(oldrows)!=20 or selected['checkpoint_sha256']!=bestold['checkpoint_sha256']:
            raise ValueError('Historical factorial checkpoint selection mismatch')
        es=complete(oldroot/'final_evaluation/summary.json')
        ep=read_json(oldroot/'final_evaluation/protocol.json')
        if (es['frame_count']!=829 or es['conditions']!=['clean','fog','snow'] or not es['full_validation'] or
            es['checkpoint_sha256']!=selected['checkpoint_sha256'] or ep['protocols']!=p['protocols'] or
            ep['input_manifest_identity']!=p['input_manifest_identity'] or
            sha256(selected['checkpoint'])!=selected['checkpoint_sha256']):
            raise ValueError('Historical factorial evaluation mismatch')
        historical[label]=dict(root=str(oldroot),selected=selected,evaluation=es)
        for name in ('contract.json','selected.json','final_evaluation/summary.json','final_evaluation/protocol.json'):
            artifacts[str(oldroot/name)]=sha256(oldroot/name)
        artifacts[selected['checkpoint']]=selected['checkpoint_sha256']
    ref=dict(status='complete',e0_root=str(folder),checkpoint=choice['checkpoint'],checkpoint_sha256=choice['checkpoint_sha256'],
        selected=choice,initial_state=str(initial),initial_state_sha256=ih,C_initial_state=str(cpath),C_initial_state_sha256=ch,
        original_training_contract=c,evaluation_root=str(ev),evaluation_summary_sha256=sha256(ev/'summary.json'),
        artifacts=artifacts,training_forbidden=True,reused=True,initial_model_keys=sorted(s['model']),
        historical_B0_B2=historical)
    ref['identity']=identity(ref)
    bound_json(Path(args.experiment_root)/'baseline_reference.json',ref)
    atomic_json(destination/'E0_REUSE_AUDIT.json',ref)
    (destination/'E0_REUSE_AUDIT.md').write_text('# E0复用审计\n\n原A1已完成20轮和三天气829帧正式评测；直接复用，不重新训练。\n\n选用权重：`'+choice['checkpoint']+'`\n\n初始化仅使用A1与B2原始第0轮，未使用训练后最佳权重。\n',encoding='utf-8')
    return ref

def guard_reference(args):
    ref=complete(Path(args.experiment_root)/'baseline_reference.json');check_identity(ref)
    for n,h in ref['artifacts'].items():
        if sha256(n)!=h:raise ValueError('E0/initial artifact changed: '+n)
    return ref

def initialization_reference(args):
    ref=guard_reference(args)
    return dict(e0_root=ref['e0_root'],source_epoch=0,initial_state_sha256=ref['initial_state_sha256'],
        C_initial_state_sha256=ref['C_initial_state_sha256'] if args.variant=='E2' else None,
        trained_E0_weight_used=False,baseline_reference_identity=ref['identity'])

def initialize_model(model,args):
    import torch
    if args.variant not in ('E1','E2'):raise ValueError('E0 training forbidden')
    ref=guard_reference(args)
    source=torch.load(ref['initial_state'],map_location='cpu',weights_only=False)['model']
    target=model.state_dict()
    extra={k for k in target if k not in source}
    expected={'ggca.s.logits'} if args.variant=='E1' else {k for k in target if k.startswith('repe_c.')}
    if extra!=expected or not set(source).issubset(target):raise ValueError('Original A1 tensor keys changed')
    for k,v in source.items():
        if v.shape!=target[k].shape or v.dtype!=target[k].dtype:raise ValueError('A1 initial tensor shape/dtype mismatch')
        target[k]=v.to(target[k].device)
    if args.variant=='E2':
        c=torch.load(ref['C_initial_state'],map_location='cpu',weights_only=False)['model']
        for k in expected:target[k]=c[k].to(target[k].device)
    model.load_state_dict(target,strict=True)
    if any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in source.items()):raise ValueError('A1 initial tensors differ')
    atomic_json(Path(args.output_root)/'initialization_audit.json',dict(status='complete',source=ref['initial_state'],
        initial_state_sha256=ref['initial_state_sha256'],source_epoch=0,shared_tensors_exact=True,
        shared_tensor_count=len(source),extra_keys=sorted(extra),trained_E0_weight_used=False,
        C_initial_state_sha256=ref['C_initial_state_sha256'] if args.variant=='E2' else None))
    return model

def restore_initial_rng(args):
    import torch
    from checkpoint import restore_rng
    ref=guard_reference(args)
    restore_rng(torch.load(ref['initial_state'],map_location='cpu',weights_only=False)['rng'])
