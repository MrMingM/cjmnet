"""One invocation: preflight -> paired training -> validation -> OPV2V-W."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',default='local_fusion_v3/experiment.yaml')
    p.add_argument('--frontend-config',required=True)
    p.add_argument('--frontend-checkpoint',required=True)
    p.add_argument('--run',required=True)
    args = p.parse_args()
    import yaml
    from gspr_evidence import runtime as er, benchmark
    from gspr_communication.runtime import ROOT,new_output,write_json,sha256,verify_frozen
    from . import runtime as rt
    verify_frozen()
    options,_ = er.load_config(args.config,args.frontend_config)
    rt.settings(options)
    # Read only paths here, never test predictions; fail before expensive training.
    for name,path in benchmark.ROOTS.items():
        if not Path(path).is_dir():
            raise FileNotFoundError(f'{name} benchmark root missing: {path}')
    run = new_output(Path(args.run).resolve())
    config = run/'resolved_experiment.yaml'
    config.write_text(yaml.safe_dump(options,sort_keys=False),encoding='utf-8')
    specification = rt.contract(options,args.frontend_config,sha256(args.frontend_checkpoint))
    write_json(run/'source_contract.json',specification)
    for relative in {*specification['pipeline_sources'],*specification['method_sources']}:
        destination = run/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,destination)
    journal = []
    def execute(name,command):
        start = time.time()
        print(f'[{name}] START',flush=True)
        with (run/(name+'.log')).open('w',encoding='utf-8') as stream:
            process = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                     env=dict(os.environ,PYTHONUNBUFFERED='1'))
        journal.append(dict(stage=name,command=command,returncode=process.returncode,seconds=time.time()-start))
        write_json(run/'pipeline_status.json',journal)
        if process.returncode:
            raise RuntimeError(f'{name} failed; later stages stopped. See {run/(name+".log")}')
        print(f'[{name}] DONE',flush=True)
    execute('unit_tests',[sys.executable,'-m','unittest','local_fusion_v3.test_core','-v'])
    common = ['--config',str(config),'--frontend-config',args.frontend_config,
              '--frontend-checkpoint',args.frontend_checkpoint]
    for variant in ('attention','residual'):
        execute('train_'+variant,[sys.executable,'-m','local_fusion_v3.train',*common,
                                 '--variant',variant,'--output-dir',str(run/variant)])
    for phase in ('calibration','development','benchmark'):
        execute(phase,[sys.executable,'-m','local_fusion_v3.evaluate',*common,'--run',str(run),'--phase',phase])
    write_json(run/'all_results.json',{phase:json.loads((run/phase/'protocol.json').read_text(encoding='utf-8'))['conditions']
                                     for phase in ('development','benchmark')})
    print('ALL STAGES COMPLETE including OPV2V clean test and OPV2V-W:',run/'all_results.json',flush=True)


if __name__ == '__main__':
    main()
