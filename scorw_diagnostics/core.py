"""Own locked, atomic stages. Historical modules and data are read-only."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def read(path):
    def bad(x):
        raise ValueError('Nonfinite JSON: ' + x)
    return json.loads(Path(path).read_text('utf-8'), parse_constant=bad)


def identity(x):
    # Exactly the canonical encoding used by historical provenance.identity.
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    def convert(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        raise TypeError('Unsupported report value: '+type(x).__name__)
    text(p, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=convert) + '\n')


def text(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.partial')
    with q.open('w', encoding='utf-8', newline='\n') as f:
        f.write(value); f.flush(); os.fsync(f.fileno())
    os.replace(q, p)


@contextmanager
def lock(p):
    import fcntl
    p = Path(p); p.mkdir(parents=True, exist_ok=True)
    with (p / '.lock').open('a+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def files(p):
    return {v.relative_to(p).as_posix(): digest(v) for v in sorted(Path(p).rglob('*'))
            if v.is_file() and v.name not in ('.lock', 'COMMIT.json') and not v.name.endswith('.partial')}


def stage(p, request, callback):
    p = Path(p)
    with lock(p):
        contract = dict(identity=identity(request), request=request)
        if (p / 'CONTRACT.json').exists() and read(p / 'CONTRACT.json') != contract:
            raise ValueError('Stage contract changed: ' + str(p))
        write(p / 'CONTRACT.json', contract)
        if (p / 'COMMIT.json').exists():
            old = read(p / 'COMMIT.json')
            if old != dict(identity=contract['identity'], hashes=files(p), status='passed'):
                raise ValueError('Committed output changed: ' + str(p))
            print('REUSE ' + str(p), flush=True)
            return
        callback()
        write(p / 'COMMIT.json', dict(identity=contract['identity'], hashes=files(p), status='passed'))


def server_only():
    if platform.system() != 'Linux' or REPO != Path('/home/cjm/OpenCOOD-main/cjmnet'):
        raise RuntimeError('Run research checks only on the designated Linux server')


def sources():
    pins = read(HERE / 'source_snapshot.json')['sources']
    for p, h in pins.items():
        if digest(REPO / p) != h:
            raise ValueError('Delivered source mismatch: ' + p)
    own = {p.relative_to(REPO).as_posix() for p in HERE.rglob('*')
           if p.is_file() and p.suffix in ('.py', '.json', '.sh', '.md') and p.name != 'source_snapshot.json'}
    if own != {p for p in pins if p.startswith('scorw_diagnostics/')}:
        raise ValueError('Incomplete delivered source set')
    # Freeze actual server native sources: basedataset has a known historical byte difference.
    actual = {p.relative_to(REPO).as_posix(): digest(p) for folder in
              ('opencood', 'attfuse_code', 'scorw_dataset', 'scorw_r3', 'shared_weather_benchmark', 'reference_weather')
              for p in sorted((REPO / folder).rglob('*.py')) if '__pycache__' not in p.parts}
    versions = {}
    for name in ('numpy', 'torch', 'spconv', 'spconv-cu120', 'cumm', 'open3d', 'shapely'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(delivered=pins, actual_server_sources=actual, packages=versions,
                python_version=platform.python_version(), python_hash_seed=os.environ.get('PYTHONHASHSEED'),
                git_required=False, native_policy='freeze actual bytes; no historical files rewritten')
