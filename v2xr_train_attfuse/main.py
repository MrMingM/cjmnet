"""Server-only D-1D extraction, Clean random-init training and frozen protocol evaluation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import traceback

from auditlib import (DEFAULTS, atomic_json, evaluation_path, gpu, lock, paths, server_only,
                      sources, verify_train)


def main():
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory', 'inspect', 'extract', 'verify', 'train', 'select', 'final',
        'clean-ap', 'evaluate-checkpoint', 'compare', 'self-test'))
    for name, default in DEFAULTS.items():
        parser.add_argument('--' + name.replace('_', '-'), default=default)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--limit-frames', type=int, default=0)
    parser.add_argument('--checkpoint')
    parser.add_argument('--evaluation-root')
    parser.add_argument('--opv2v-checkpoint', default='/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/net_epoch13.pth')
    args = parser.parse_args()
    own = sources()
    paths(args)
    if not 0 <= args.limit_frames <= 829:
        raise ValueError('limit-frames must be 0 (all) or 1..829')
    if args.limit_frames and args.command not in ('clean-ap', 'evaluate-checkpoint', 'final'):
        raise ValueError('Partial AP is allowed only for evaluation smoke, never full training selection')
    if args.command not in ('inspect', 'extract', 'train') and args.smoke:
        raise ValueError('--smoke applies only to extraction/training; AP smoke uses --limit-frames')
    target = Path(args.train_root) if args.command in ('extract', 'verify') else Path(args.output_root)
    if args.command == 'inspect':
        target = Path(args.train_root).with_name(Path(args.train_root).name + '-inspection')
    if args.command == 'inventory':
        target = Path(args.train_root).with_name(Path(args.train_root).name + '-inventory')
    if args.command in ('clean-ap', 'evaluate-checkpoint'):
        if not args.checkpoint or not args.evaluation_root:
            raise ValueError('Explicit checkpoint and independent evaluation-root required')
        target = evaluation_path(args, args.evaluation_root)
    if args.command in ('train', 'final', 'clean-ap', 'evaluate-checkpoint', 'compare', 'self-test'):
        gpu()
    with lock(target.with_suffix('.d1d.lock')):
        try:
            if args.command == 'inventory':
                from extract import inspect
                attempt = target / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                attempt.mkdir(parents=True, exist_ok=True)
                print('Train frame inventory output:', attempt, flush=True)
                inspect(args, attempt, inventory_only=True)
            elif args.command == 'inspect':
                from archive_listing import inspect_only
                attempt = target / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                attempt.mkdir(parents=True, exist_ok=True)
                print('Archive inspection output:', attempt, flush=True)
                inspect_only(args, attempt)
            elif args.command == 'extract':
                from extract import run
                run(args, own)
            elif args.command == 'verify':
                manifest = verify_train(args)
                # Re-decode after hash verification; no cached completion can bypass content checks.
                from observe import bootstrap, module_inventory
                runtime_root = bootstrap(args.runtime_root)
                from extract import content
                for row in manifest['files']:
                    content(Path(args.train_root) / 'train' / row['relative'])
                module_inventory(runtime_root)
                print('FULL TRAIN VERIFIED:', manifest['counts'], flush=True)
            elif args.command == 'train':
                from training import run
                run(args, own)
            elif args.command == 'select':
                from evaluation import selection
                print(selection(args), flush=True)
            elif args.command == 'final':
                from evaluation import final
                final(args, own)
            elif args.command in ('clean-ap', 'evaluate-checkpoint'):
                from evaluation import evaluate, weather_checkpoint_gate
                if args.command == 'evaluate-checkpoint':
                    weather_checkpoint_gate(args, args.checkpoint)
                evaluate(args, own, args.checkpoint, target,
                    conditions=('clean',) if args.command == 'clean-ap' else ('clean', 'fog', 'snow'),
                    limit=args.limit_frames)
            elif args.command == 'compare':
                from evaluation import compare
                compare(args, own)
            else:
                from server_tests import run
                run(args)
        except Exception as error:
            failure = dict(status='failed', command=args.command, error=repr(error), traceback=traceback.format_exc(),
                time_utc=datetime.now(timezone.utc).isoformat())
            # Selection failures must not modify historical datasets or successful completion records.
            target.mkdir(parents=True, exist_ok=True)
            atomic_json(target / 'last_failure.json', failure)
            raise


if __name__ == '__main__':
    main()
