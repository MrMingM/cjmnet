"""Compact completion check; no Torch/OpenCOOD required."""
from .common import Manifest, parser


def main():
    args = parser(__doc__).parse_args()
    manifest = Manifest(args.run)
    complete = True
    for stage in ('S0', 'S1', 'S2', 'S3', 'FINAL'):
        done = manifest.complete(stage)
        row = manifest.value['entries'].get(manifest.key(stage), {})
        print(stage + ' ' + ('complete' if done else row.get('status', 'pending')))
        complete &= done
    if not manifest.complete('integrity'):
        complete = False
        print('integrity ' + manifest.value['entries'].get(manifest.key('integrity'), {}).get('status', 'pending'))
    raise SystemExit(0 if complete else 1)


if __name__ == '__main__':
    main()
