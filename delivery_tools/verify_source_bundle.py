"""Check the transferred source bundle with standard Python only, without Git."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def member(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError('Unsafe source path: ' + name)
    path = root.joinpath(*relative.parts)
    if root not in path.resolve().parents:
        raise ValueError('Source path escapes the bundle: ' + name)
    return path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'SOURCE_BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    failures = []
    for name, row in manifest['files'].items():
        path = member(root, name)
        if not path.is_file():
            failures.append('missing: ' + name)
        elif path.stat().st_size != row['size'] or sha256(path) != row['sha256']:
            failures.append('changed: ' + name)
    # The original packages reject both missing and extra files. Keep those rules.
    for package in manifest['snapshot_packages']:
        folder = root / package
        expected = json.loads((folder / 'source_snapshot.json').read_text(encoding='utf-8'))['files']
        actual = {p.relative_to(folder).as_posix(): sha256(p) for p in folder.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
        if actual != expected:
            failures.append('original package snapshot differs: ' + package)
    for name, digest in json.loads((root / 'v2xr_official_repro/frozen_baseline.json').read_text(encoding='utf-8')).items():
        path = member(root, name)
        if not path.is_file() or sha256(path) != digest:
            failures.append('frozen audit dependency differs: ' + name)
    for package, field in (('v2xr_weather_l012', 'files'), ('v2xr_gspr_clean', 'files'),
                           ('v2xr_train_attfuse', 'dependencies')):
        if package not in manifest['snapshot_packages']:
            continue
        pins = json.loads((root / package / 'PINS.json').read_text(encoding='utf-8'))[field]
        for name, digest in pins.items():
            path = member(root, name)
            if not path.is_file() or sha256(path) != digest:
                failures.append('pinned dependency differs: ' + name)
    if failures:
        raise ValueError('\n'.join(failures))
    return dict(status='source_integrity_passed', files=len(manifest['files']),
                snapshot_packages=len(manifest['snapshot_packages']),
                training_executed=False, cuda_migration_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
