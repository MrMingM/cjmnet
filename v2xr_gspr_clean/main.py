"""Server-only Clean GSPR training and validation using frozen V2X-R official O."""
import argparse
from pathlib import Path
import traceback

from auditlib import DEFAULTS, atomic_json, evaluation_path, gpu, lock, paths, server_only, sources


def main():
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('self-test', 'train', 'clean-ap', 'select', 'final'))
    for name, default in DEFAULTS.items():
        parser.add_argument('--' + name.replace('_', '-'), default=default)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--limit-frames', type=int, choices=(0, 12), default=0)
    parser.add_argument('--checkpoint')
    parser.add_argument('--evaluation-root')
    args = parser.parse_args()
    own = sources()
    paths(args)
    if args.smoke and args.command != 'train':
        raise ValueError('SMOKE=1 is for training only; inference uses LIMIT_FRAMES=12')
    if args.limit_frames and args.command not in ('clean-ap', 'final'):
        raise ValueError('Partial AP cannot select the full model')
    if args.command != 'select':
        gpu()
    target = Path(args.output_root)
    if args.command == 'clean-ap':
        if not args.checkpoint or not args.evaluation_root:
            raise ValueError('Clean AP requires explicit checkpoint/evaluation-root')
        target = evaluation_path(args, args.evaluation_root)
    if args.command == 'self-test':
        target = target.with_name(target.name + '-self-test')
    with lock(target.with_suffix('.gspr.lock')):
        try:
            if args.command == 'train':
                from training import run
                run(args, own)
            elif args.command == 'clean-ap':
                from auditlib import no_test
                checkpoint = no_test(args.checkpoint).resolve(strict=True)
                root = Path(args.output_root)
                if root not in checkpoint.parents:
                    raise ValueError('Clean AP accepts only this new training run checkpoint')
                from evaluation import evaluate
                evaluate(args, own, checkpoint, target, conditions=('clean',), limit=args.limit_frames)
            elif args.command == 'select':
                from evaluation import selection
                print(selection(args), flush=True)
            elif args.command == 'final':
                from evaluation import final
                final(args, own)
            else:
                from server_tests import run
                run(args, target, own)
        except Exception as error:
            target.mkdir(parents=True, exist_ok=True)
            atomic_json(target / 'last_failure.json', dict(status='failed', command=args.command,
                error=repr(error), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
