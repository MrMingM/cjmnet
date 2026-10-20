"""One action per process; model training exits before evaluation claims the GPU."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uecp_gspr_v2xr.common import lock, read, server, sources


def main():
    server()
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['preflight', 'content', 'self-test', 'train', 'dev', 'select', 'final', 'diagnose', 'report', 'resume-check'])
    p.add_argument('--output', required=True)
    p.add_argument('--variant', default='E0')
    p.add_argument('--epoch', type=int, default=1)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--replay', action='store_true')
    p.add_argument('--interrupt-once', action='store_true')
    p.add_argument('--interrupt-train-once', action='store_true')
    args = p.parse_args(); root = Path(args.output)
    sources()
    if args.variant not in read(root / 'config.json')['variants']:
        raise ValueError('Unknown detector variant')
    with lock(root / '.worker.lock'):
        if args.action == 'preflight':
            from uecp_gspr_v2xr.preflight import run
            run(root)
            return
        if args.action == 'report':
            from uecp_gspr_v2xr.report import run
            result = run(root, args.smoke)
            if result['status'] != 'COMPLETE':
                raise ValueError('Final report still has missing evidence')
            return
        from uecp_gspr_v2xr.runtime import input_stamps, bootstrap
        input_stamps(root); bootstrap(root, gpu=args.action not in ('content', 'select'))
        if args.action == 'content':
            from uecp_gspr_v2xr.preflight import content
            content(root)
        elif args.action == 'self-test':
            from uecp_gspr_v2xr.server_checks import run
            run(root)
        elif args.action == 'train':
            from uecp_gspr_v2xr.train import run
            run(root, args.variant, args.epoch, args.smoke, args.replay, args.interrupt_train_once)
        elif args.action in ('dev', 'select', 'final'):
            from uecp_gspr_v2xr import evaluate
            if args.action == 'dev':
                evaluate.dev(root, args.variant, args.epoch, args.smoke)
            elif args.action == 'select':
                evaluate.select(root, args.variant, args.smoke)
            else:
                evaluate.final(root, args.variant, args.smoke, args.interrupt_once)
        elif args.action == 'diagnose':
            from uecp_gspr_v2xr.diagnose import run
            run(root, args.smoke)
        elif args.action == 'resume-check':
            from uecp_gspr_v2xr.server_checks import resume_check
            resume_check(root, args.variant, args.epoch)


if __name__ == '__main__':
    main()
