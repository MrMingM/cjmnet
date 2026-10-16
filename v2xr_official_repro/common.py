"""Small standard-library helpers; never import the project's OpenCOOD."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
COMMIT = 'f3ebbad95561444a53a17358735b0f0dc4ba89f7'
REFERENCE = {'ap30': 0.8669, 'ap50': 0.8258, 'ap70': 0.6656}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def identity(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False,
        sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    os.replace(temporary, path)


def server_only():
    if sys.platform != 'linux':
        raise RuntimeError('Server only. Do not run Torch/OpenCOOD/data preparation locally.')


def no_test(path):
    # Check both spelling and resolved destinations, including linked files.
    path = Path(path).absolute()
    for candidate in (path, path.resolve()):
        if any(part.lower() in ('test', 'testing') or 'opv2v-w' in part.lower()
               for part in candidate.parts):
            raise ValueError(f'Test paths are refused: {candidate}')
    return path


@contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def link(source, destination):
    source = no_test(source).resolve(strict=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve(strict=True) == source:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Refusing to replace {destination}')
    destination.symlink_to(source)


def verify_package():
    expected = read_json(PACKAGE / 'source_snapshot.json')['files']
    actual = {p.relative_to(PACKAGE).as_posix(): sha256(p)
        for p in PACKAGE.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if actual != expected:
        raise ValueError('Package differs from source_snapshot.json; synchronize the entire directory.')
    return expected


def install_runtime(runtime):
    """Copy byte-identical upstream once; all compiled artifacts stay on data disk."""
    import shutil
    runtime = Path(runtime)
    expected = read_json(PACKAGE / 'UPSTREAM.json')['files']
    runtime.mkdir(parents=True, exist_ok=True)
    with lock(runtime.with_suffix('.lock')):
        for name, digest in expected.items():
            source, target = PACKAGE / 'upstream' / name, runtime / name
            if sha256(source) != digest:
                raise ValueError(f'Bundled upstream changed: {name}')
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            if sha256(target) != digest:
                raise ValueError(f'Runtime source changed: {target}; do not patch upstream silently.')
        extras = [str(p) for p in (runtime / 'opencood').rglob('*.py')
                  if p.relative_to(runtime).as_posix() not in expected
                  and 'build' not in p.parts]
        if extras:
            raise ValueError(f'Unexpected Python sources in official runtime: {extras}')
    return expected
