"""Serial hash-journaled experiment. A0 has no training stage."""
import argparse
from pathlib import Path
import os
import signal
import traceback
from bridge import load
from auditlib import DEFAULTS, atomic_json, bound_json, complete, gpu, identity, lock, paths, server_only, sources

_old = load('orchestrator', 'gspr_repe/orchestrator.py')
Pipeline = _old.Pipeline


def command(args, action, output, variant='A1', **flags):
    return _old.command(args, action, output, variant, **flags)


def workflow(args, run):
    from results import final, stage1
    from evaluation import selection
    root = Path(args.output_root)
    pipeline = Pipeline(args, run)
    pipeline.stage('preflight', root / 'preflight', command(args, 'preflight', root / 'preflight'))
    pipeline.stage('A0_reference', root / 'A0', command(args, 'baseline-evaluate', root / 'A0', 'A0'))
    if args.mode == 'preflight':
        return
    pipeline.stage('self-test', root / 'self_test', command(args, 'self-test', root / 'self_test'))
    for variant in ('A1', 'A2', 'A3'):
        folder = root / 'smoke' / variant
        pipeline.stage('smoke_' + variant, folder, command(args, 'train', folder, variant, smoke=True))
        weight = complete(folder / 'smoke_checkpoint.json')['checkpoint']
        evaluation = folder / 'evaluation_smoke12'
        pipeline.stage('smoke_eval_' + variant, evaluation, command(args, 'evaluate', folder, variant,
            checkpoint=weight, evaluation_root=evaluation, smoke=True))
    pipeline.stage('stage1_acceptance', root / 'stage1', callback=lambda: stage1(root))
    if args.mode == 'smoke':
        return
    for variant in ('A1', 'A2', 'A3'):
        folder = root / variant
        pipeline.stage(variant + '_train', folder, command(args, 'train', folder, variant))
    for variant in ('A1', 'A2', 'A3'):
        folder = root / variant
        evaluation = folder / 'final_evaluation'
        pipeline.stage(variant + '_evaluate', evaluation, command(args, 'evaluate', folder, variant,
            checkpoint=selection(folder)['checkpoint'], evaluation_root=evaluation))
    pipeline.stage('mechanism', root / 'mechanism', command(args, 'diagnose', root / 'mechanism'))
    pipeline.stage('final', root / 'reports', callback=lambda: final(root))
    atomic_json(root / 'current_stage.json', dict(status='complete', stage='final'))


def main():
    server_only()
    def interrupted(signum, frame):
        raise InterruptedError('GGCA pipeline received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'all'), default='all')
    parser.add_argument('--resume', action='store_true')
    for key, default in DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'), default=os.environ.get(key.upper(), default))
    args = parser.parse_args()
    if Path(args.train_root).name == 'train' and (Path(args.train_root).parent / 'manifest.json').is_file():
        args.train_root = str(Path(args.train_root).parent)
    args.experiment_root = args.output_root
    paths(args); gpu()
    root = Path(args.output_root)
    if root.exists() and any(root.iterdir()) and (not args.resume or not (root / 'run_identity.json').is_file()):
        raise ValueError('Nonempty output requires matching GGCA identity and RESUME=1')
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / '.pipeline.lock'):
        try:
            run = dict(experiment='GSPR_GGCA', sources=sources(), roots={k: getattr(args, k) for k in DEFAULTS},
                backend='rocm', physical_gpu=os.environ['ROCR_VISIBLE_DEVICES'],
                PYTHONHASHSEED=os.environ['PYTHONHASHSEED'])
            run['identity'] = identity(run)
            bound_json(root / 'run_identity.json', run)
            (root / 'logs').mkdir(exist_ok=True)
            workflow(args, run)
        except BaseException as error:
            atomic_json(root / 'STOP.json', dict(status='TECHNICAL_FAILURE', error=repr(error), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
