"""Worker entry; users launch run_all.sh once."""
import argparse
from pathlib import Path
import traceback
from auditlib import DEFAULTS, METHOD, atomic_json, complete, evaluation_path, gpu, lock, paths, server_only, sha256, sources


def main():
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'baseline-evaluate', 'self-test', 'train', 'evaluate', 'diagnose'))
    for key, default in DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'), default=default)
    parser.add_argument('--variant', choices=('A0', 'A1', 'A2', 'A3'), default='A1')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--clean-only', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--evaluation-root')
    parser.add_argument('--parent-root')
    args = parser.parse_args()
    sources(); paths(args); gpu()
    root = Path(args.evaluation_root or args.output_root)
    with lock(root.with_name(root.name + '.worker.lock')):
        try:
            if args.command == 'evaluate':
                if not args.checkpoint or not args.evaluation_root:
                    raise ValueError('Explicit checkpoint and destination required')
                evaluation_path(args, args.evaluation_root)
                if not args.clean_only and not args.smoke:
                    from evaluation import selection
                    chosen = (complete(Path(args.experiment_root) / 'baseline_reference.json') if args.variant == 'A0'
                              else selection(Path(args.output_root)))
                    if sha256(args.checkpoint) != chosen['checkpoint_sha256']:
                        raise ValueError('Weather inference requires Clean-locked selected checkpoint')
                from evaluation import evaluate
                evaluate(args, args.checkpoint, root, args.clean_only, args.smoke)
            elif args.command == 'baseline-evaluate':
                from preflight import reference_evaluation
                reference_evaluation(args)
            else:
                # Reuse original data/environment/native-source validation unchanged.
                from training import prepare
                args.prepared = prepare(args)
                if args.command == 'preflight':
                    from preflight import run
                elif args.command == 'self-test':
                    from server_tests import run
                elif args.command == 'diagnose':
                    from diagnose import run
                else:
                    if args.variant not in METHOD['trainable_variants']:
                        raise ValueError('A0 training forbidden')
                    from training import run
                run(args)
        except BaseException as error:
            atomic_json(root / 'failure.json', dict(status='technical_failure', error=repr(error), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
