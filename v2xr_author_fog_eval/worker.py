"""A fresh interpreter per stage prevents OpenCOOD namespace leakage."""
import argparse
from pathlib import Path
import sys
import traceback

from v2xr_author_fog_eval.core import identity, read, verify_sources, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--stage', required=True,
        choices=('preflight', 'checkpoint-audit', 'smoke', 'author-evaluation', 'report'))
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Research execution is server-only')
    root = Path(args.root)
    request = read(root / 'run_identity.json')
    if identity({k: v for k, v in request.items() if k != 'identity'}) != request['identity']:
        raise ValueError('Run identity record invalid')
    if verify_sources() != request['sources']:
        raise ValueError('Sources changed since controller start')
    try:
        if args.stage == 'report':
            from v2xr_author_fog_eval.comparison import report
            report(root, request)
        else:
            from v2xr_author_fog_eval.pipeline import preflight, load_audit, evaluate
            if args.stage == 'preflight':
                preflight(root, request)
            elif args.stage == 'checkpoint-audit':
                load_audit(root, request)
            else:
                evaluate(root, request, smoke=args.stage == 'smoke')
    except Exception:
        write(root / ('failure_' + args.stage + '.json'), dict(status='failed', stage=args.stage,
            run_identity=request['identity'], traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
