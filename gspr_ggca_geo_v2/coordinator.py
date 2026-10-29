"""One multi-version launcher: preserved original A1, new A2/A3 only."""
import argparse
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import traceback
from bridge import load
from auditlib import DEFAULTS, atomic_json, bound_json, complete, gpu, identity, lock, paths, read_json, server_only, sources
from lineage import plan, selected

Pipeline = load('orchestrator', 'gspr_repe/orchestrator.py').Pipeline


def command(args, action, output, variant='A2', old=False, **flags):
    p=plan(args.output_root)
    source=Path(p['old_source']) if old else Path(__file__).resolve().parent
    cmd=[sys.executable,'-B',str(source/'main.py'),action,'--variant',variant,'--resume']
    for key in DEFAULTS:
        value=output if key=='output_root' else (p['old_root'] if old and key=='experiment_root' else getattr(args,key))
        cmd+=['--'+key.replace('_','-'),str(value)]
    for key,value in flags.items():
        if value is True:cmd.append('--'+key.replace('_','-'))
        elif value is not False and value is not None:cmd+=['--'+key.replace('_','-'),str(value)]
    if not old and flags.get('smoke') and hasattr(args,'test_gpu'):
        cmd=['/usr/bin/env','ROCR_VISIBLE_DEVICES='+str(args.test_gpu)]+cmd
    return cmd


def test_device(root):
    """Choose a free extra card, otherwise retain the sequential single-card path."""
    file=root/'test_device.json'
    if file.exists():return read_json(file)['physical_gpu']
    preferred=os.environ.get('GGCA_TEST_GPU','1')
    if not preferred.isdigit():raise ValueError('GGCA_TEST_GPU must be a physical index')
    formal=os.environ['ROCR_VISIBLE_DEVICES']
    use=formal
    if preferred!=formal:
        r=subprocess.run(['rocm-smi','--showmeminfo','vram','-d',preferred],capture_output=True,text=True,check=True)
        import re
        matches=re.findall(r'Total Used Memory \(MiB\):\s*(\d+)',r.stdout)
        if len(matches)==1 and int(matches[0])<=16:use=preferred
    bound_json(file,dict(status='complete',physical_gpu=use,formal_gpu=formal,
        policy='free extra GPU only; otherwise sequential with original A1 GPU'))
    return use


def require_free_test_device(device):
    import re
    for attempt in range(6):
        r=subprocess.run(['rocm-smi','--showmeminfo','vram','-d',str(device)],capture_output=True,text=True,check=True)
        values=re.findall(r'Total Used Memory \(MiB\):\s*(\d+)',r.stdout)
        if len(values)==1 and int(values[0])<=16:return
        if attempt<5:time.sleep(2)
    raise RuntimeError('Test GPU '+str(device)+' is occupied; no other process was stopped')


def stage1(root):
    p=plan(root)
    tests=complete(root/'self_test/summary.json')
    mechanism=complete(root/'self_test/GEOMETRY_MECHANISM_TESTS.json')
    restore=complete(root/'self_test/A1_STATE_RESTORE_TEST.json')
    records={}
    for variant in ('A2','A3'):
        folder=root/'smoke'/variant
        history=read_json(folder/'history.json');row=history['epochs'][0]
        summary=complete(folder/'evaluation_smoke12/summary.json')
        cp=complete(folder/'smoke_checkpoint.json')
        if (len(history['epochs'])!=1 or row['train_updates']!=2 or len(row['repe'])!=2
                or summary['frame_count']!=12 or summary['conditions']!=['clean','fog','snow']
                or summary['checkpoint_sha256']!=cp['checkpoint_sha256']):
            raise ValueError('New-revision real smoke incomplete: '+variant)
        if not any(any(v>0 for v in r['update_l2'].values()) for r in row['repe']):
            raise ValueError('No actual GGCA parameter update in '+variant)
        if row['contract']!=read_json(folder/'contract.json')['identity']:
            raise ValueError('Smoke contract mismatch')
        records[variant]=dict(checkpoint=cp,evaluation=summary)
    original=complete(Path(p['old_root'])/'stage1/acceptance.json')
    bound_json(root/'stage1/acceptance.json',dict(status='complete',tests=tests,mechanism=mechanism,
        A1_restore=restore,new_smoke=records,original_A1_acceptance=original,
        original_A1_acceptance_source=str(Path(p['old_root'])/'stage1/acceptance.json'),
        old_A2_A3_smoke_accepted_for_new_version=False,stage2_allowed=True))


def stop_complete(root):
    path=Path(read_json(root/'migration_plan_v2.json')['boundary_record'])
    if not path.exists():return False
    stop=read_json(path)
    if stop['status']!='stopped_after_complete_epoch':return False
    for pid,old in stop['processes'].items():
        try:
            raw=(Path('/proc')/pid/'stat').read_text().rsplit(')',1)[1].split()
            if raw[19]==old['start'] and raw[0]!='Z':raise ValueError('Old GGCA process still alive')
        except FileNotFoundError:pass
    from checkpoint import committed
    p=plan(root);folder=Path(p['old_root'])/'A1'
    rows=committed(folder,20,read_json(folder/'contract.json')['identity'])
    if len(rows)<stop['completed_epoch']:raise ValueError('Stopped epoch state missing')
    return True


def workflow(args, run):
    root=Path(args.output_root);migration=plan(root)
    print('Waiting for verified original A1 epoch boundary; running sources stay unchanged.',flush=True)
    while not stop_complete(root):time.sleep(2)
    pipeline=Pipeline(args,run)
    pipeline.stage('preflight',root/'preflight',command(args,'preflight',root/'preflight'))
    pipeline.stage('A0_reference',root/'A0',command(args,'baseline-evaluate',root/'A0','A0'))
    if args.mode=='preflight':return
    pipeline.stage('self-test',root/'self_test',command(args,'self-test',root/'self_test'))
    # After the stopped checkpoint is actually reconstructed and verified, A1
    # can continue while independent new smoke tests use an otherwise free card.
    # With no free card, retain the sequential behavior and do not claim a card.
    args.test_gpu=test_device(root)
    from a1_lane import A1Lane
    old_folder=Path(migration['old_root'])/'A1'
    lane=A1Lane(root,command(args,'train',old_folder,'A1',old=True))
    parallel=args.mode=='all' and args.test_gpu!=os.environ['ROCR_VISIBLE_DEVICES']
    if parallel:lane.ensure()
    for variant in ('A2','A3'):
        folder=root/'smoke'/variant
        if not (root/'stages'/('smoke_'+variant)/'complete.json').exists():
            require_free_test_device(args.test_gpu)
        pipeline.stage('smoke_'+variant,folder,command(args,'train',folder,variant,smoke=True))
        weight=complete(folder/'smoke_checkpoint.json')['checkpoint'];dest=folder/'evaluation_smoke12'
        if not (root/'stages'/('smoke_eval_'+variant)/'complete.json').exists():
            require_free_test_device(args.test_gpu)
        pipeline.stage('smoke_eval_'+variant,dest,command(args,'evaluate',folder,variant,
            checkpoint=weight,evaluation_root=dest,smoke=True))
    pipeline.stage('stage1_acceptance',root/'stage1',callback=lambda:stage1(root))
    if args.mode=='smoke':return
    # Keep original code PATH, output directory, settings and full-state contract.
    # Do not restart the old orchestrator, which would also launch obsolete A2.
    folder=old_folder
    pipeline.stage('preserved_A1_train',folder,callback=lane.wait)
    dest=folder/'final_evaluation'
    pipeline.stage('preserved_A1_evaluate',dest,command(args,'evaluate',folder,'A1',old=True,
        checkpoint=selected(root,'A1')['checkpoint'],evaluation_root=dest))
    for variant in ('A2','A3'):
        folder=root/variant
        pipeline.stage(variant+'_train',folder,command(args,'train',folder,variant))
    for variant in ('A2','A3'):
        folder=root/variant;dest=folder/'final_evaluation'
        pipeline.stage(variant+'_evaluate',dest,command(args,'evaluate',folder,variant,
            checkpoint=selected(root,variant)['checkpoint'],evaluation_root=dest))
    pipeline.stage('mechanism',root/'mechanism',command(args,'diagnose',root/'mechanism'))
    from results import final
    pipeline.stage('final',root/'reports',callback=lambda:final(root))
    atomic_json(root/'current_stage.json',dict(status='complete',stage='final'))


def main():
    server_only()
    def interrupted(sig,frame):raise InterruptedError('Unified GGCA signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('preflight','smoke','all'),default='all')
    parser.add_argument('--resume',action='store_true')
    for key,value in DEFAULTS.items():parser.add_argument('--'+key.replace('_','-'),default=os.environ.get(key.upper(),value))
    args=parser.parse_args();args.experiment_root=args.output_root
    paths(args);gpu()
    root=Path(args.output_root);root.mkdir(parents=True,exist_ok=True);migration=plan(root)
    if (root/'run_identity.json').exists() and not args.resume:
        raise ValueError('Existing versioned run requires RESUME=1')
    run=dict(experiment='GSPR_GGCA_GEOMETRY_V2',sources=sources(),migration_identity=migration['identity'],
        roots={k:getattr(args,k) for k in DEFAULTS},backend='rocm',physical_gpu=os.environ['ROCR_VISIBLE_DEVICES'],
        PYTHONHASHSEED=os.environ['PYTHONHASHSEED'])
    run['identity']=identity(run)
    with lock(root/'.pipeline.lock'):
        try:
            bound_json(root/'run_identity.json',run);(root/'logs').mkdir(exist_ok=True)
            workflow(args,run)
        except BaseException as error:
            failure=dict(status='TECHNICAL_FAILURE',error=repr(error),traceback=traceback.format_exc())
            if (root/'A1_lane/state.json').exists():
                failure['preserved_A1_lane']=read_json(root/'A1_lane/state.json')
                failure['A1_policy']='Original A1 is an independent adopted worker; inspect its PID/log. New-test failure does not terminate it.'
            atomic_json(root/'STOP.json',failure)
            raise


if __name__=='__main__':main()
