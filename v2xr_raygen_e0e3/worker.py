"""Internal stage entry; no recursive shell dispatch or final-test call."""
import argparse
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2xr_raygen_e0e3.common import lock, read, server, verify_sources, write


def main():
    server()
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['preflight', 'content', 'smoke-prepare', 'checks', 'probe', 'labels',
                                     'train', 'evaluate', 'resume-check', 'report', 'local-test'])
    p.add_argument('--output', required=True)
    p.add_argument('--group', choices=['E0', 'E1', 'E2', 'E3'])
    p.add_argument('--epoch', type=int, default=1)
    p.add_argument('--micro', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--replay', action='store_true')
    p.add_argument('--interrupt-eval-once', action='store_true')
    args = p.parse_args()
    root = Path(args.output)
    verify_sources()
    with lock(root / '.worker.lock'):
        if args.action == 'preflight':
            from v2xr_raygen_e0e3.preflight import run
            run(root)
            return
        if args.action == 'report':
            from v2xr_raygen_e0e3.report import run
            result = run(root, args.smoke)
            if result['status'] != 'complete':
                raise ValueError('Report has incomplete evidence')
            return
        if args.action == 'local-test':
            from v2xr_raygen_e0e3.tests_local import run
            run()
            return
        from v2xr_raygen_e0e3.runtime import bootstrap, input_stamps
        input_stamps(root)
        bootstrap(root, gpu=args.action not in ('content', 'smoke-prepare', 'labels'))
        if args.action == 'content':
            from v2xr_raygen_e0e3.preflight import verify_training_content
            verify_training_content(root)
        elif args.action == 'smoke-prepare':
            from v2xr_raygen_e0e3.smoke import prepare
            prepare(root)
        elif args.action == 'checks':
            from v2xr_raygen_e0e3.smoke import checks
            checks(root)
        elif args.action == 'labels':
            from v2xr_raygen_e0e3.cache import build, prepare_windows
            build(root)
            prepare_windows(root)
        elif args.action in ('train', 'probe'):
            from v2xr_raygen_e0e3.training import run
            try:
                run(root, args.group, args.epoch, args.smoke, args.micro, args.replay)
            except __import__('torch').cuda.OutOfMemoryError:
                if args.action == 'probe':
                    write(root / 'probe' / ('micro' + str(args.micro)) / 'oom.json',
                          dict(status='oom', microbatch=args.micro, traceback=traceback.format_exc()))
                    raise SystemExit(42)
                raise
        elif args.action == 'evaluate':
            from v2xr_raygen_e0e3.evaluation import run, SmokeInterruption
            try:
                run(root, args.group, args.epoch, args.smoke, args.interrupt_eval_once)
            except SmokeInterruption:
                raise SystemExit(43)
        elif args.action == 'resume-check':
            from v2xr_raygen_e0e3.smoke import resume_check
            resume_check(root, args.group)


if __name__ == '__main__':
    main()
