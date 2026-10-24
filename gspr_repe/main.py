"""Internal serial subprocess entry. Users start run_all.sh."""
import argparse
from pathlib import Path
import traceback
from auditlib import (DEFAULTS, atomic_json, evaluation_path, gpu, paths,
                      server_only, sources, lock, no_test, read_json)


def main():
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'self-test', 'train', 'evaluate'))
    for key, default in DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'), default=default)
    parser.add_argument('--variant', choices=('B0', 'B1', 'B2', 'B3', 'B4'), default='B4')
    parser.add_argument('--parent-root')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--clean-only', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--evaluation-root')
    args = parser.parse_args()
    sources()
    paths(args)
    gpu()
    root = Path(args.evaluation_root or args.output_root)
    # Surviving child processes retain this lock even if the launcher is SIGKILLed.
    # A new launcher must fail rather than concurrently rewrite an active worker's output.
    worker_lock = lock(root.with_name(root.name + '.worker.lock'))
    worker_lock.__enter__()
    try:
        if args.command == 'evaluate':
            if not args.checkpoint or not args.evaluation_root:
                raise ValueError('Explicit evaluation checkpoint/destination required')
            args.checkpoint = str(no_test(args.checkpoint))
            evaluation_path(args, args.evaluation_root)
            if not args.clean_only and not args.smoke:
                from auditlib import complete, sha256
                from evaluation import selection
                parent = (complete(Path(args.output_root) / 'artifact.json') if args.variant.endswith('_BN')
                          else selection(Path(args.output_root)))
                if args.variant.endswith('_BN') and read_json(Path(args.output_root) / 'contract.json')['smoke_only']:
                    raise ValueError('Smoke BN cannot enter formal evaluation')
                if sha256(args.checkpoint) != parent['checkpoint_sha256']:
                    raise ValueError('Weather evaluation requires the locked selected/calibrated checkpoint')
            from evaluation import evaluate
            evaluate(args, args.checkpoint, Path(args.evaluation_root), args.clean_only, args.smoke)
        else:
            from training import prepare
            args.prepared = prepare(args)
            if args.command == 'preflight':
                from preflight import run
            elif args.command == 'train':
                from training import run
            else:
                from server_tests import run
            run(args)
    except BaseException as error:
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / 'failure.json', dict(status='technical_failure', command=args.command,
            variant=args.variant, error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        worker_lock.__exit__(None, None, None)


if __name__ == '__main__':
    main()
