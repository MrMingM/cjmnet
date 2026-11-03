"""Internal worker; the only user launcher is run_all.sh."""
import argparse
import os
from pathlib import Path
import traceback
from auditlib import DEFAULTS,atomic_json,complete,gpu,lock,paths,server_only,sha256,sources

def main():
    server_only()
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('preflight','self-test','train','evaluate','diagnose'))
    for k,v in DEFAULTS.items():parser.add_argument('--'+k.replace('_','-'),default=v)
    parser.add_argument('--variant',choices=('E0','E1','E2'),default='E1')
    for flag in ('resume','smoke','clean-only'):parser.add_argument('--'+flag,action='store_true')
    parser.add_argument('--checkpoint');parser.add_argument('--evaluation-root');parser.add_argument('--parent-root')
    args=parser.parse_args();sources();paths(args);gpu()
    if args.command=='self-test':os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    root=Path(args.evaluation_root or args.output_root);root.mkdir(parents=True,exist_ok=True)
    with lock(root.with_name(root.name+'.worker.lock')):
        try:
            if args.command=='evaluate':
                if not args.checkpoint or not args.evaluation_root:raise ValueError('Explicit weight/destination required')
                from auditlib import separate
                if not root.resolve().is_relative_to(Path(args.experiment_root)):raise ValueError('Evaluation outside new experiment')
                for p in (args.e0_root,args.a0_root,args.b2_root,args.runtime_root):separate(root,p)
                if not args.smoke and not args.clean_only:
                    from evaluation import selection
                    chosen=selection(Path(args.output_root))
                    if sha256(args.checkpoint)!=chosen['checkpoint_sha256']:raise ValueError('Weather weights must be Clean-selected')
                from evaluation import evaluate
                evaluate(args,args.checkpoint,root,args.clean_only,args.smoke)
            elif args.command=='diagnose':
                from provenance import verify_inputs
                from auditlib import runtime,frozen_c
                from baseline import guard_reference
                evidence=verify_inputs(args);frozen_c(args);runtime(args,evidence['a']);guard_reference(args)
                args.prepared=(None,evidence)
                from diagnose import run
                run(args)
            else:
                from training import prepare
                args.prepared=prepare(args)
                if args.command=='preflight':from preflight import run
                elif args.command=='self-test':from server_tests import run
                else:from training import run
                run(args)
        except BaseException as error:
            atomic_json(root/'failure.json',dict(status='technical_failure',error=repr(error),traceback=traceback.format_exc()))
            raise

if __name__=='__main__':main()
