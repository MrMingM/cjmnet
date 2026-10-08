"""Read-only completion check; offline success is never full completion."""
from pathlib import Path
from .common import cli, digest, read_json, safe_file
from .pipeline import FULL_STAGES, OFFLINE_STAGES


def main():
    parser = cli(__doc__)
    parser.add_argument('--mode', choices=('offline', 'all'), default='all')
    args = parser.parse_args()
    run = Path(args.run)
    manifest = read_json(run / 'manifest.json')
    complete = True
    for stage in OFFLINE_STAGES if args.mode == 'offline' else FULL_STAGES:
        row = manifest['entries'].get(stage + '/-/-/-', {})
        done = row.get('status') == 'complete'
        for name, expected in row.get('artifacts', {}).items():
            path = safe_file(run, name)
            done = done and path.is_file() and digest(path) == expected
        state = 'complete' if done else ('corrupted' if row.get('status') == 'complete' else row.get('status', 'pending'))
        print(stage + ' ' + state)
        complete &= done
    print('OFFLINE ONLY' if args.mode == 'offline' else ('CONTROL COMPLETE' if complete else 'CONTROL INCOMPLETE'))
    raise SystemExit(0 if complete else 1)


if __name__ == '__main__':
    main()
