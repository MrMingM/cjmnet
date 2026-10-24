"""Fresh-process stages for the approved formal first-two-epochs startup."""
import argparse
from pathlib import Path
import traceback

from warm_common import bridge, frozen_worker, ipc_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('probe', 'preflight', 'self-test', 'warmup', 'train',
        'evaluate', 'targets', 'E0-import', 'E1-report', 'synthesis'))
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--recipe', choices=('T1', 'T2'), default='T1')
    parser.add_argument('--variant', choices=('C', 'D'), default='C')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--existing', type=Path)
    parser.add_argument('--clean-only', action='store_true')
    cli = parser.parse_args()
    base = bridge()
    base.server_only()
    ipc = ipc_probe()
    print('IPC_PREFLIGHT=complete socket_path_bytes=' + str(ipc['socket_path_bytes']), flush=True)
    cli.smoke = False
    frozen = frozen_worker()
    with base.lock(frozen.worker_lock_path(cli)):
        try:
            if cli.stage in ('warmup', 'train', 'evaluate'):
                from warm_runtime import activate, publish_warmup
                runtime = activate(cli)
                if cli.stage == 'evaluate':
                    runtime.evaluate(cli)
                else:
                    runtime.train(cli)
                    if cli.stage == 'warmup':
                        publish_warmup(cli.model_root, cli.recipe, cli.variant)
            elif cli.stage == 'probe':
                from preflight import probe
                probe(cli.destination)
            elif cli.stage == 'preflight':
                from preflight import inputs
                inputs(cli)
            elif cli.stage == 'self-test':
                from server_tests import check
                check(cli)
            elif cli.stage == 'E0-import':
                frozen.import_e0(cli)
            elif cli.stage == 'targets':
                from e1_targets import target_analysis
                base.atomic(cli.model_root / 'target_analysis.json', target_analysis(cli.destination, 829))
            elif cli.stage == 'E1-report':
                from reports import e1_report
                e1_report(cli.model_root, cli.original_root, smoke=False)
            elif cli.stage == 'synthesis':
                from reports import synthesis
                synthesis(cli.model_root)
        except BaseException:
            base.atomic(cli.model_root / 'last_failure.json', dict(status='technical_failure',
                stage=cli.stage, traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
