"""Exit zero only when every stage and final artifact has passed integrity."""
from pathlib import Path
import sys
from .common import Manifest, parser, read_json, external_run


def main():
    args = parser(__doc__).parse_args()
    try:
        run = external_run(args.run)
        from .pipeline import integrity
        integrity(run, require_integrity=True)
        if read_json(run/'exit_status.json')['exit_code'] != 0:
            raise ValueError('Driver has not exited successfully')
        for name in ('R0', 'GEO', 'SEM', 'ABLATIONS', 'REPLAY', 'FINAL'):
            print(name+' complete')
    except Exception as exc:
        try:
            if not (external_run(args.run)/'manifest.json').is_file():
                raise FileNotFoundError('No manifest')
            manifest = Manifest(args.run)
            for name in ('R0', 'GEO', 'SEM', 'ABLATIONS', 'REPLAY', 'FINAL'):
                row = manifest.value['entries'].get(manifest.key(name), {})
                print(name+' '+row.get('status', 'pending'))
        except Exception:
            pass
        print('INCOMPLETE: '+str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
