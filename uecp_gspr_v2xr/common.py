"""Shared atomic I/O reused read-only; this experiment has its own source/contract namespace."""
import os
from pathlib import Path

from v2xr_raygen_e0e3.common import (atomic_bytes, bound, identity, lock, plain, progress,
                                    read, safe, server, sha, stamp, stream_seed, write)

__all__ = ['atomic_bytes', 'bound', 'identity', 'lock', 'plain', 'progress', 'read', 'safe',
           'server', 'sha', 'stamp', 'stream_seed', 'write', 'HERE', 'ROOT', 'settings',
           'sources', 'verify_dependencies', 'variant_path', 'commit', 'complete']

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def settings():
    c = read(HERE / 'settings.json')
    c['seed'] = int(os.environ.get('SEED', c['seed']))
    c['epochs'] = int(os.environ.get('EPOCHS', c['epochs']))
    c['microbatch'] = int(os.environ.get('MICROBATCH', c['microbatch']))
    if not 0 <= c['seed'] <= 4294967295 or c['epochs'] < 1 or c['microbatch'] not in (1, 2, 4, 8):
        raise ValueError('Invalid seed/epochs/microbatch; effective batch is fixed8')
    if os.environ.get('AMP', '0') != '0':
        raise ValueError('This preregistration uses FP32/AMP=0 for every model')
    return c


def verify_dependencies(pins, root=ROOT):
    failures = []
    for name, digest in pins.items():
        try:
            actual = sha(Path(root) / name)
        except OSError as exc:
            failures.append(name + ': unavailable (' + str(exc) + ')')
            continue
        if actual != digest:
            failures.append(name + ': expected=' + digest + ' actual=' + actual)
    if failures:
        raise ValueError('Read-only runtime dependency mismatch (' + str(len(failures)) + ' files):\n'
                         + '\n'.join(failures) + '\nKeep the pinned code versions; do not reseal server files to bypass this check.')


def sources():
    expected = read(HERE / 'source_snapshot.json')['files']
    actual = {p.relative_to(HERE).as_posix(): sha(p) for p in HERE.rglob('*')
              if p.is_file() and p.name != 'source_snapshot.json' and '__pycache__' not in p.parts}
    if expected != actual:
        raise ValueError('Synchronize the entire new package including source_snapshot.json')
    pins = read(HERE / 'PINS.json')['files']
    verify_dependencies(pins)
    from .dependency_scope import validate_scope
    validate_scope(ROOT, pins)
    return dict(own=actual, dependencies=pins)


def variant_path(output, variant, smoke=False):
    paths = {'E0': 'E0/baseline', 'E2': 'E2/gspr_assisted_uecp', 'E3_joint': 'E3/dual_quality_fusion/joint'}
    name = paths.get(variant, 'ablations/' + variant)
    return Path(output) / ('smoke' if smoke else '') / name


def commit(path, outputs, **fields):
    p = Path(path)
    record = dict(status='COMPLETE', output_hashes={str(Path(f).relative_to(p.parent)): sha(f) for f in outputs}, **fields)
    bound(p, record)
    return record


def complete(path):
    p = Path(path)
    r = read(p)
    if r['status'] != 'COMPLETE':
        raise ValueError('Incomplete artifact: ' + str(p))
    for name, digest in r.get('output_hashes', {}).items():
        if sha(p.parent / name) != digest:
            raise ValueError('Committed artifact bytes changed: ' + str(p.parent / name))
    if 'prediction_manifest.json' in r.get('output_hashes', {}):
        for row in read(p.parent / 'prediction_manifest.json')['files']:
            if sha(row['path']) != row['sha256']:
                raise ValueError('Committed frame prediction changed: '+row['path'])
    if 'shard_manifest.json' in r.get('output_hashes', {}):
        for row in read(p.parent / 'shard_manifest.json'):
            if sha(row['path']) != row['sha256'] or sha(Path(row['path']).with_suffix('.json')) != row['commit_sha256']:
                raise ValueError('Committed quality region shard changed')
    return r
