"""Internal stages; the single public launcher is run_all.sh."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from e0_common import (DEFAULT_ORIGINAL, DEFAULT_OUTPUT, HERE, PROJECT, atomic,
    finish, identity, lock, read, receipt, safe, sources)

STAGES = ('preflight', 'self-test', 'smoke-paired', 'smoke-R0', 'smoke-R1', 'smoke-R2',
          'smoke-R3', 'smoke-analysis', 'baseline', 'paired', 'reliability',
          'intervention-R1', 'intervention-R2', 'intervention-R3', 'error-analysis', 'final')
FOLDERS = {'self-test': 'self_test', 'smoke-paired': 'smoke/paired',
           'smoke-analysis': 'smoke/analysis', 'reliability': 'reliability/R0',
           'error-analysis': 'error_analysis'}
FOLDERS.update({'smoke-'+r: 'smoke/'+r for r in ('R0', 'R1', 'R2', 'R3')})
FOLDERS.update({'intervention-'+r: 'interventions/'+r for r in ('R1', 'R2', 'R3')})


def stage_folder(output, stage):
    return output / 'stages' / FOLDERS.get(stage, stage)


def publish_reports(output):
    final = stage_folder(output, 'final')
    names = ['E0_FINAL_RESULTS.json', 'E0_RESULTS.md', 'E0_CONCLUSIONS.md', 'E0_PROTOCOL.json',
        'E0_RELIABILITY_STATS.json', 'E0_POINT_DEGRADATION.json', 'E0_INTERVENTIONS.json',
        'E0_ERROR_ANALYSIS.json', 'E0_EXECUTION_AUDIT.json', 'figures']
    for name in names:
        link, target = output / name, final / name
        if link.is_symlink():
            if link.resolve(strict=True) != target.resolve(strict=True):
                raise ValueError('Incompatible published E0 report link: '+str(link))
        elif link.exists():
            raise ValueError('Existing unowned E0 report would be overwritten: '+str(link))
        else:
            link.symlink_to(Path('stages/final') / name, target_is_directory=name == 'figures')


def validate_paths(cli):
    cli.output, cli.original = safe(cli.output).resolve(), safe(cli.original).resolve(strict=True)
    from e0_common import historical
    audit, _, _ = historical()
    for path in (PROJECT, cli.original, *[Path(v).resolve() for k, v in audit.DEFAULTS.items()
            if k not in ('output_root', 'seven_zip')]):
        if cli.output == path or cli.output in path.parents or path in cli.output.parents:
            raise ValueError('Output overlaps read-only input: ' + str(path))
    if Path('/data') not in cli.output.parents:
        raise ValueError('Research output must be on /data')
    return cli


def request(cli):
    record = dict(family='GSPR_E0_DIAGNOSIS', revision=1, sources=sources(),
        protocol=read(HERE / 'protocol.json'), output=str(cli.output), original=str(cli.original),
        physical_gpu=os.environ.get('ROCR_VISIBLE_DEVICES'),
        PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED'), executable=sys.executable,
        threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    record['identity'] = identity(record)
    path = cli.output / 'request.json'
    if not path.exists() and cli.output.exists() and any(p.name not in ('logs', '.controller.lock', '.worker.lock') for p in cli.output.iterdir()):
        raise ValueError('Nonempty output has no compatible E0 request; nothing overwritten')
    atomic(path, record, immutable=True)
    return record['identity']


def execute(cli, folder, request_id):
    from e0_checks import preflight, baseline, self_test
    if cli.stage == 'preflight':
        return preflight(cli, folder)
    inventory = read(cli.output / 'stages/preflight/inventory.json')
    if cli.stage == 'self-test':
        return self_test(cli, folder, inventory)
    if cli.stage in ('paired', 'smoke-paired'):
        from e0_runtime import setup
        from e0_data import audit_pairs
        _, evidence, _, _, _, _ = setup(cli, gpu=False)
        return audit_pairs(cli, folder, request_id, evidence, limit=cli.stage == 'smoke-paired')
    if cli.stage == 'baseline':
        return baseline(cli, folder, inventory)
    if cli.stage.startswith('smoke-R') or cli.stage.startswith('intervention-R') or cli.stage == 'reliability':
        from e0_inference import run_arm
        arm = 'R0' if cli.stage == 'reliability' else cli.stage.rsplit('-', 1)[1]
        return run_arm(cli, folder, arm, request_id, inventory, smoke=cli.stage.startswith('smoke-'))
    if cli.stage == 'smoke-analysis':
        from e0_analysis import smoke_check
        return smoke_check(cli, folder)
    if cli.stage == 'error-analysis':
        from e0_analysis import error_analysis
        return error_analysis(cli, folder, request_id)
    if cli.stage == 'final':
        from e0_report import report
        return report(cli, folder, request_id)
    raise ValueError('Unimplemented E0 stage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='gspr-e0-diagnosis 2026-10-21.1')
    parser.add_argument('stage', choices=(*STAGES, 'paths-check'))
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--original', type=Path, default=Path(DEFAULT_ORIGINAL))
    cli = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Research stages run on the server only; use check_static.py locally')
    validate_paths(cli)
    if cli.stage == 'paths-check':
        return
    folder = stage_folder(cli.output, cli.stage)
    with lock(cli.output / '.worker.lock'):
        request_id = request(cli)
        for stage in STAGES[:STAGES.index(cli.stage)]:
            if receipt(stage_folder(cli.output, stage), request_id) is None:
                raise ValueError('Previous stage has not completed: ' + stage)
        if receipt(folder, request_id):
            if cli.stage == 'preflight':
                # Every launch rehashes all real inputs and original results before reusing stages.
                from e0_runtime import setup, model_inventory, cache_audit, runtime_info
                _, evidence, frozen, audit, _, _ = setup(cli)
                actual = model_inventory(cli, audit)
                saved = read(folder / 'inventory.json')
                if actual != saved['models'] or runtime_info() != saved['environment']:
                    raise ValueError('Original model identity or execution environment changed on resume')
                for model in actual.values():
                    cache_audit(model, evidence, frozen)
            print('REUSE verified '+cli.stage, flush=True)
            if cli.stage == 'final':
                publish_reports(cli.output)
            return
        folder.mkdir(parents=True, exist_ok=True)
        def interrupted(signum, frame):
            raise InterruptedError('Signal '+str(signum))
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, interrupted)
        try:
            print('START '+cli.stage, flush=True)
            execute(cli, folder, request_id)
            finish(folder, request_id)
            if cli.stage == 'final':
                publish_reports(cli.output)
            print('COMPLETE '+cli.stage, flush=True)
        except BaseException as error:
            failure = dict(status='technical_failure', stage=cli.stage, error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(), request=request_id)
            stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'_'+str(os.getpid())
            atomic(cli.output / 'failures' / (stamp+'_'+cli.stage+'.json'), failure, immutable=True)
            print('FAILED '+cli.stage+': '+str(error), file=sys.stderr, flush=True)
            raise


if __name__ == '__main__':
    main()
