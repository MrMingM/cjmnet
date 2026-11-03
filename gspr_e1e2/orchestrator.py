"""Single POSIX launcher -> audited E0 -> acceptance -> two independent GPU lanes."""
import argparse
import copy
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from bridge import load
from auditlib import DEFAULTS,HERE,atomic_json,bound_json,complete,identity,lock,paths,read_json,server_only,sha256,sources

Pipeline=load('pipeline','gspr_repe/orchestrator.py').Pipeline

def command(args,action,output,variant='E1',**flags):
    cmd=[sys.executable,'-B',str(HERE/'main.py'),action,'--variant',variant,'--resume']
    for k in DEFAULTS:cmd+=['--'+k.replace('_','-'),str(output if k=='output_root' else getattr(args,k))]
    for k,v in flags.items():
        if v is True:cmd.append('--'+k.replace('_','-'))
        elif v is not False and v is not None:cmd+=['--'+k.replace('_','-'),str(v)]
    return cmd

def free_device(device):
    result=subprocess.run(['rocm-smi','--showmeminfo','vram','-d',device],capture_output=True,text=True,check=True)
    used=re.findall(r'Total Used Memory \(MiB\):\s*(\d+)',result.stdout)
    if len(used)!=1 or int(used[0])>16:
        raise ValueError('Requested GPU is not free; no other task will be stopped: '+device)
    return dict(physical_gpu=device,used_memory_MiB=int(used[0]))

def worker_env(gpu):
    env=dict(os.environ)
    for k in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'):env.pop(k,None)
    env['ROCR_VISIBLE_DEVICES']=gpu
    return env

def parallel(args,run,phase):
    root=Path(args.output_root);folder=root/('parallel_'+phase);folder.mkdir(exist_ok=True)
    processes=[]
    try:
        for variant,gpu in zip(('E1','E2'),args.gpus.split(',')):
            # Verify only lanes which still need computation; completed lanes are hash-replayed.
            lane_done=root/'lanes'/variant/'stages'/('smoke_diagnose' if phase=='smoke' else 'diagnose')/'complete.json'
            if not lane_done.exists():free_device(gpu)
            cmd=[sys.executable,'-B',str(HERE/'orchestrator.py'),'--lane',variant,'--mode',phase,'--gpus',args.gpus,'--resume']
            for k in DEFAULTS:cmd+=['--'+k.replace('_','-'),str(getattr(args,k))]
            log=(folder/(variant+'.log')).open('a',encoding='utf-8')
            child=subprocess.Popen(cmd,env=worker_env(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((variant,child,log))
            atomic_json(folder/(variant+'_process.json'),dict(pid=child.pid,gpu=gpu,command=cmd))
            print('PARALLEL',phase,variant,'GPU',gpu,'PID',child.pid,flush=True)
        pending={v for v,_,_ in processes}
        while pending:
            for variant,child,log in processes:
                code=child.poll()
                if code is not None:
                    pending.discard(variant)
                    if code:raise RuntimeError('Parallel '+phase+' lane '+variant+' failed: '+str(code))
            if pending:time.sleep(1)
        atomic_json(folder/'summary.json',dict(status='complete',phase=phase,lanes=['E1','E2'],run_identity=run['identity']))
    finally:
        for variant,child,log in processes:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
            log.close()

def lane(args,run):
    from evaluation import selection
    v=args.lane;root=Path(args.experiment_root);out=root/'lanes'/v
    lane_args=copy.copy(args);lane_args.output_root=str(out)
    out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    with lock(out/'.lane.lock'):
        p=Pipeline(lane_args,run)
        smoke=root/'smoke'/v
        p.stage('smoke_train',smoke,command(args,'train',smoke,v,smoke=True))
        choice=complete(smoke/'smoke_checkpoint.json')
        p.stage('smoke_evaluate',smoke/'evaluation_smoke12',command(args,'evaluate',smoke,v,
            checkpoint=choice['checkpoint'],evaluation_root=smoke/'evaluation_smoke12',smoke=True))
        p.stage('smoke_diagnose',smoke/'diagnostic_smoke2',command(args,'diagnose',smoke/'diagnostic_smoke2',v,smoke=True))
        if args.mode=='smoke':return
        complete(root/'stage1/acceptance.json')
        formal=root/v
        p.stage('train',formal,command(args,'train',formal,v))
        p.stage('evaluate',formal/'final_evaluation',command(args,'evaluate',formal,v,
            checkpoint=selection(formal)['checkpoint'],evaluation_root=formal/'final_evaluation'))
        p.stage('diagnose',root/'mechanism'/v,command(args,'diagnose',root/'mechanism'/v,v))

def main():
    server_only()
    def interrupted(sig,frame):raise InterruptedError('E12 controller interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    parser=argparse.ArgumentParser()
    for k,v in DEFAULTS.items():parser.add_argument('--'+k.replace('_','-'),default=os.environ.get(k.upper(),v))
    parser.add_argument('--mode',choices=('preflight','smoke','all'),default=os.environ.get('MODE','smoke'))
    parser.add_argument('--resume',action='store_true');parser.add_argument('--lane',choices=('E1','E2'))
    parser.add_argument('--gpus',default=os.environ.get('GPUS','2,4'))
    args=parser.parse_args();args.experiment_root=args.output_root;paths(args);sources()
    gpus=args.gpus.split(',')
    if len(gpus)!=2 or len(set(gpus))!=2 or not all(s.isdigit() for s in gpus):raise ValueError('Two distinct physical GPU indices required')
    root=Path(args.output_root);root.mkdir(parents=True,exist_ok=True)
    run=dict(experiment='A1_SELF_RETENTION_AND_B2_C',sources=sources(),roots={k:getattr(args,k) for k in DEFAULTS},
        backend='rocm',gpus=dict(zip(('E1','E2'),gpus)),PYTHONHASHSEED=os.environ['PYTHONHASHSEED'])
    run['identity']=identity(run)
    if args.lane:
        lane(args,run);return
    with lock(root/'.pipeline.lock'):
        try:
            if (root/'run_identity.json').exists() and not args.resume:raise ValueError('Existing run requires RESUME=1')
            bound_json(root/'run_identity.json',run);(root/'logs').mkdir(exist_ok=True)
            p=Pipeline(args,run)
            # The common preflight/test uses E1's physical card, then releases it.
            os.environ['ROCR_VISIBLE_DEVICES']=gpus[0]
            if not (root/'stages/preflight/complete.json').exists():free_device(gpus[0])
            p.stage('preflight',root/'preflight',command(args,'preflight',root/'preflight'))
            if args.mode=='preflight':return
            if not (root/'stages/self-test/complete.json').exists():free_device(gpus[0])
            p.stage('self-test',root/'self_test',command(args,'self-test',root/'self_test'))
            p.stage('parallel_smoke',root/'parallel_smoke',callback=lambda:parallel(args,run,'smoke'))
            from results import stage1,final
            p.stage('stage1_acceptance',root/'stage1',callback=lambda:stage1(root))
            if args.mode=='smoke':
                atomic_json(root/'current_stage.json',dict(stage='stage1_acceptance',status='complete',formal_training_started=False));return
            p.stage('parallel_formal',root/'parallel_all',callback=lambda:parallel(args,run,'all'))
            p.stage('final',root/'reports',callback=lambda:final(root))
            atomic_json(root/'current_stage.json',dict(stage='final',status='complete'))
        except BaseException as error:
            atomic_json(root/'STOP.json',dict(status='technical_failure',error=repr(error),traceback=traceback.format_exc()))
            raise

if __name__=='__main__':main()
