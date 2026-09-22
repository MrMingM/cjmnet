"""One command: prepare -> train -> validation calibration -> development -> test.

Every child must finish successfully before the next starts. This launcher never
reads test scores to choose a model or threshold and never overwrites an old run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='local_fusion_utility_v2/experiment.yaml')
    p.add_argument('--frontend-config', required=True)
    p.add_argument('--frontend-checkpoint', required=True)
    p.add_argument('--run', required=True)
    args = p.parse_args()
    from gspr_communication.runtime import ROOT, new_output, write_json
    run = new_output(args.run)
    python = sys.executable
    journal = []

    def execute(stage, command):
        print(f'[{stage}] START', flush=True)
        began = time.time()
        log = run/(stage+'.log')
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.run(command, cwd=str(ROOT), stdout=stream,
                                     stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED='1'))
        journal.append(dict(stage=stage, command=command, returncode=process.returncode,
                            seconds=time.time()-began, log=str(log)))
        write_json(run/'pipeline_status.json', journal)
        if process.returncode:
            raise RuntimeError(f'{stage} failed; see {log}. Later stages were NOT started.')
        print(f'[{stage}] DONE; log={log}', flush=True)

    execute('unit_tests', [python, '-m', 'unittest', 'local_fusion_utility_v2.test_core', '-v'])
    frontend = ['--frontend-config', args.frontend_config,
                '--frontend-checkpoint', args.frontend_checkpoint]
    for split in ('train', 'validation'):
        execute('prepare_'+split, [python, '-m', 'local_fusion_utility_v2.prepare',
                '--config', args.config, *frontend, '--split', split,
                '--output-dir', str(run/(split+'_cache'))])
    # Save the resolved training options verbatim, including absolute fog paths.
    # Reusing this exact config prevents the v1 relative/absolute path failure.
    import yaml
    manifest = json.loads((run/'train_cache/manifest.json').read_text(encoding='utf-8'))
    resolved = run/'resolved_experiment.yaml'
    resolved.write_text(yaml.safe_dump(manifest['contract']['options'], sort_keys=False), encoding='utf-8')
    for method in ('utility', 'loss_gain'):
        execute('train_'+method, [python, '-m', 'local_fusion_utility_v2.train',
                '--train-cache', str(run/'train_cache'), '--validation-cache', str(run/'validation_cache'),
                '--variant', method, '--output-dir', str(run/method)])
    trained = ['--utility-checkpoint', str(run/'utility/best.pth'),
               '--loss-gain-checkpoint', str(run/'loss_gain/best.pth')]
    common = ['--config', str(resolved), *frontend, *trained]
    execute('calibration', [python, '-m', 'local_fusion_utility_v2.calibrate', *common,
            '--output-dir', str(run/'calibration')])
    calibration = run/'calibration/calibration.json'
    for phase in ('development', 'benchmark'):
        execute(phase, [python, '-m', 'local_fusion_utility_v2.evaluate', *common,
                '--calibration', str(calibration), '--phase', phase,
                '--output-dir', str(run/phase),
                *(['--scale-ablation', '--no-keep-ablation'] if phase == 'development' else [])])
    report = {}
    for phase in ('development', 'benchmark'):
        report[phase] = json.loads((run/phase/'protocol.json').read_text(encoding='utf-8'))['conditions']
    write_json(run/'all_results.json', report)
    print('ALL STAGES COMPLETE, including OPV2V-W:', run/'all_results.json', flush=True)


if __name__ == '__main__':
    main()
