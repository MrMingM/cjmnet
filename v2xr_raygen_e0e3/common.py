"""Standard library only. Atomic artifacts and identities do not require Git."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    return value


def identity(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                                   allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write(path, value):
    atomic_bytes(path, (json.dumps(plain(value), indent=2, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False) + '\n').encode())


def bound(path, value):
    if Path(path).exists() and read(path) != plain(value):
        raise ValueError('Immutable identity changed: ' + str(path))
    write(path, value)


def safe(path):
    p = Path(path).absolute()
    for q in (p, p.resolve()):
        if any(x.lower() in ('test', 'testing') or 'opv2v-w' in x.lower() for x in q.parts):
            raise ValueError('Formal test path refused: ' + str(q))
    return p


def stamp(path):
    p = safe(path)
    s = p.stat()
    return dict(path=str(p), resolved=str(p.resolve(strict=True)), size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def stream_seed(seed, epoch, uid, stream):
    return int(identity([seed, epoch, uid, stream])[:8], 16)


def settings():
    c = read(HERE / 'config.json')
    c['seed'] = int(os.environ.get('SEED', c['seed']))
    c['epochs'] = int(os.environ.get('EPOCHS', c['epochs']))
    c['amp'] = bool(int(os.environ.get('AMP', '0')))
    if not 0 <= c['seed'] <= 4294967295 or c['epochs'] < 1 or c['amp']:
        raise ValueError('SEED must be uint32; EPOCHS positive; this FP32 contract requires AMP=0')
    return c


def verify_sources():
    snap = read(HERE / 'source_snapshot.json')
    actual = {p.relative_to(ROOT).as_posix(): sha(p) for p in HERE.rglob('*')
              if p.is_file() and p.name != 'source_snapshot.json' and '__pycache__' not in p.parts}
    actual['launch_v2xr_e0e3.sh'] = sha(ROOT / 'launch_v2xr_e0e3.sh')
    if snap['files'] != actual:
        raise ValueError('New package changed; synchronize complete package and snapshot to a NEW output')
    for name, digest in read(HERE / 'PINS.json')['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Read-only dependency changed: ' + name)
    return dict(files=actual, dependencies=read(HERE / 'PINS.json')['files'])


@contextlib.contextmanager
def lock(path):
    import fcntl
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()) + '\n')
        f.flush()
        yield


def server():
    if sys.platform != 'linux':
        raise RuntimeError('Real data, Torch and OpenCOOD are server-only')


def progress(path, **fields):
    write(path, dict(time=time.time(), pid=os.getpid(), **fields))
