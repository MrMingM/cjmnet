"""Fresh-process internal server stages; run_all.sh is the public entry."""
import argparse
from pathlib import Path
import os
import sys
import traceback
from e01_common import atomic, identity, lock, read, safe, server_only, sha, sources


def worker_lock_path(cli):
    # The trainer waits for a child Clean evaluation. They must not take the same lock.
    # Keep this lock outside the evaluator directory: its frozen request requires an empty new output.
    if cli.stage == 'evaluate':
        if cli.destination is None or cli.checkpoint is None:
            raise ValueError('Evaluation needs checkpoint and destination')
        return cli.model_root / ('.evaluation_'+identity(str(cli.destination.resolve()))[:16]+'.lock')
    return cli.model_root / '.worker.lock'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('probe', 'preflight', 'self-test', 'train', 'evaluate', 'targets', 'E1-report', 'E0-import', 'synthesis'))
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--train-root', type=Path, default=Path('/data/cjm/datasets/v2xr-official-train-v2'))
    parser.add_argument('--original-root', type=Path, default=Path('/data/cjm/datasets/logs/v2xr-weather-l012-v1'))
    parser.add_argument('--recipe', choices=('T1', 'T2'), default='T1')
    parser.add_argument('--variant', choices=('C', 'D'), default='C')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--clean-only', action='store_true')
    parser.add_argument('--existing', type=Path)
    cli = parser.parse_args()
    server_only()
    sources()
    safe(cli.model_root)
    with lock(worker_lock_path(cli)):
        try:
            if cli.stage == 'probe':
                from preflight import probe
                probe(cli.destination)
            elif cli.stage == 'preflight':
                from preflight import inputs
                inputs(cli)
            elif cli.stage == 'train':
                from e1_runtime import train
                train(cli)
            elif cli.stage == 'self-test':
                from server_tests import check
                check(cli)
            elif cli.stage == 'evaluate':
                from e1_runtime import evaluate
                evaluate(cli)
            elif cli.stage == 'targets':
                from e1_targets import target_analysis
                atomic(cli.model_root / 'target_analysis.json', target_analysis(cli.destination, 12 if cli.smoke else 829))
            elif cli.stage == 'E1-report':
                from reports import e1_report
                e1_report(cli.model_root, cli.original_root, smoke=cli.smoke)
            elif cli.stage == 'E0-import':
                import_e0(cli)
            elif cli.stage == 'synthesis':
                from reports import synthesis
                synthesis(cli.model_root)
        except BaseException:
            atomic(cli.model_root / 'last_failure.json', dict(status='technical_failure',
                stage=cli.stage, traceback=traceback.format_exc(), pid=os.getpid()))
            raise


def verified_e0_source(source, original_root):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gspr_e0_diagnosis'))
    from e0_common import sources as e0_sources, receipt
    from e0_main import STAGES, stage_folder
    source = safe(source).resolve(strict=True)
    request = read(source / 'request.json')
    if request['identity'] != identity({k: v for k, v in request.items() if k != 'identity'}):
        raise ValueError('Existing E0 request content hash differs')
    if request['sources'] != e0_sources() or Path(request['original']).resolve() != Path(original_root).resolve():
        raise ValueError('Existing E0 sources or original experiment differ')
    for stage in STAGES:
        if receipt(stage_folder(source, stage), request['identity']) is None:
            raise ValueError('Existing E0 has not completed: '+stage)
    return source, source / 'stages/final'


def import_e0(cli):
    source, final = verified_e0_source(cli.existing, cli.original_root)
    names = dict(E0_RESULTS='E0_FINAL_RESULTS', E0_RELIABILITY='E0_RELIABILITY_STATS',
        E0_TARGET_ANALYSIS='E0_ERROR_ANALYSIS', E0_AUDIT='E0_EXECUTION_AUDIT',
        E0_INTERVENTIONS='E0_INTERVENTIONS', E0_POINT_DEGRADATION='E0_POINT_DEGRADATION')
    provenance = dict(status='complete', reuse_existing=True, root=str(source),
        request_sha256=sha(source / 'request.json'), files={})
    for target, original in names.items():
        value = read(final / (original+'.json'))
        atomic(cli.model_root / (target+'.json'), value, immutable=True)
        provenance['files'][original+'.json'] = sha(final / (original+'.json'))
    for name in ('E0_RESULTS.md', 'E0_CONCLUSIONS.md'):
        link = cli.model_root / name
        target = final / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError('Report link differs')
        elif link.exists():
            raise ValueError('Unowned report exists')
        else:
            link.symlink_to(target)
        provenance['files'][name] = sha(target)
    figures = cli.model_root / 'figures'
    if figures.is_symlink():
        if figures.resolve() != (final / 'figures').resolve():
            raise ValueError('E0 figure link differs')
    elif figures.exists():
        raise ValueError('Unowned E0 figure directory exists')
    else:
        figures.symlink_to(final / 'figures', target_is_directory=True)
    atomic(cli.model_root / 'E0_REUSE.json', provenance, immutable=True)


if __name__ == '__main__':
    main()
