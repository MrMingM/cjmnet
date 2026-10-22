"""Standard-library identity, source and artifact checks (safe to test locally)."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHECKPOINT_SHA = '195a873a814cfeb9e392ea7e3b6ae12a142b009719194d3c0a5980cbb4d089f3'
CHECKPOINT_URL = 'https://huggingface.co/hx24/V2X-R_Benchmark/blob/main/benchmark/l%2Br_train_with_fog_sim/AttFuse_LiDAR_train_w_fog.pth'
FROZEN_GT = 'd5244c4a8fa78dc1d26df1a7ee1f4592c1dd24f91a3f9c575313a081bbeab776'
CONDITIONS = ('clean', 'fog', 'snow')
PROTOCOLS = ('O', 'O*', 'F')
METRICS = ('ap30', 'ap50', 'ap70')
TOTALS = dict(clean=6224, fog=3673, snow=5854)
DEFAULTS = dict(
    output_root='/data/cjm/datasets/logs/v2xr-author-fog-attfuse-val-v1',
    checkpoint='/home/cjm/OpenCOOD-main/cjmnet/logs/AttFuse_LiDAR_train_w_fog.pth',
    runtime_root='/data/cjm/datasets/v2xr-official-runtime-v1',
    model_dir='/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1',
    d1a_root='/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1',
    gt_audit_root='/data/cjm/datasets/logs/v2xr-official-gt-audit-v1',
    d1c_root='/data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1',
    view_root='/data/cjm/datasets/v2xr-official-validate-v1',
    cd_root='/data/cjm/datasets/logs/v2xr-weather-l012-v1',
    cd_results=str(ROOT / 'logs/train/FINAL_RESULTS.json'))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(value):
    # Match the frozen evaluator's serialization, including spaces and Unicode.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               allow_nan=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    os.replace(temp, path)


def text_file(path, text):
    Path(path).write_text(text.rstrip() + '\n', encoding='utf-8', newline='\n')


def no_test(path):
    path = Path(path).absolute()
    for item in (path, path.resolve()):
        if any(p.lower() in ('test', 'testing') or 'opv2v-w' in p.lower() for p in item.parts):
            raise ValueError('Test path refused: ' + str(item))
    return path


def verify_sources():
    own = {p.relative_to(HERE).as_posix(): sha256(p) for p in HERE.rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if own != read(HERE / 'source_snapshot.json')['files']:
        raise ValueError('Synchronize the entire v2xr_author_fog_eval directory without edits')
    deps = read(HERE / 'DEPENDENCIES.json')['files']
    for name, digest in deps.items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Frozen dependency changed: ' + name)
    return dict(own=own, dependencies=deps)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_modules():
    """Use legacy imports in a fresh worker; never import weather/GSPR model adapters."""
    if 'auditlib' in sys.modules:
        raise RuntimeError('Frozen evaluator must be imported in a fresh process')
    base = ROOT / 'v2xr_train_attfuse'
    sys.path.insert(0, str(base))
    import auditlib
    if Path(auditlib.__file__).resolve() != base / 'auditlib.py':
        raise RuntimeError('Wrong legacy auditlib imported')
    auditlib.sources()
    evaluator = load_file('_author_frozen_evaluation', base / 'evaluation.py')
    import evaluate as replay_module
    if Path(replay_module.__file__).resolve() != ROOT / 'v2xr_protocol_audit/evaluate.py':
        raise RuntimeError('Wrong replay module imported')
    return auditlib, evaluator


def checkpoint_file(path):
    path = no_test(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError('Author checkpoint must be an existing nonempty regular file')
    actual = sha256(path)
    if actual != CHECKPOINT_SHA:
        raise ValueError('Author checkpoint SHA256 mismatch: actual=' + actual + ', expected=' + CHECKPOINT_SHA)
    return path


def completed(path):
    result = read(path)
    if result.get('status') != 'complete' or not result.get('output_hashes'):
        raise ValueError('Missing completion/hash evidence: ' + str(path))
    for name, digest in result['output_hashes'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid artifact name')
        if sha256(Path(path).parent / relative) != digest:
            raise ValueError('Completed artifact changed: ' + name)
    return result


def bind(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError('Resume identity changed: ' + str(path))
    write(path, value)


@contextmanager
def lock(path):
    import fcntl
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another author evaluation controller owns this output; do not delete its lock') from error
        yield stream.fileno()


def reference_source(requested):
    expected = read(HERE / 'reference/provenance.json')['sha256']
    requested = no_test(requested)
    if requested.exists() and not requested.is_file():
        raise ValueError('C/D results path exists but is not a regular JSON file')
    source = requested if requested.is_file() else HERE / 'reference/FINAL_RESULTS.json'
    if sha256(source) != expected:
        raise ValueError('C/D archive differs from the reviewed full-precision FINAL_RESULTS.json: ' + str(source))
    return dict(path=str(source.resolve()), sha256=expected,
                origin='requested archive' if source == requested else 'byte-identical bundled archive',
                requested=str(requested), provenance=read(HERE / 'reference/provenance.json'))
