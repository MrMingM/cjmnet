"""Independent experiment identity; immutable historical packages remain dependencies."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'v2xr_train_attfuse'
sys.path.append(str(BASE))


def load_frozen(name, filename, folder=BASE):
    key = '_weather_l012_frozen_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, folder / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


legacy = load_frozen('auditlib', 'auditlib.py')
atomic_json = legacy.atomic_json
bound_json = legacy.bound_json
complete = legacy.complete
evaluation_path = legacy.evaluation_path
frozen_c = legacy.frozen_c
gpu = legacy.gpu
identity = legacy.identity
lock = legacy.lock
no_test = legacy.no_test
patched = legacy.patched
read_json = legacy.read_json
request = legacy.request
server_only = legacy.server_only
sha256 = legacy.sha256
verify_train = legacy.verify_train
SETTINGS = read_json(ROOT / 'v2xr_gspr_clean/settings.json')
CONFIG = read_json(HERE / 'settings.json')
DEFAULTS = dict(legacy.DEFAULTS,
    output_root='/data/cjm/datasets/logs/gspr-repe-v1',
    baseline_root='/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1')
SELECTION = dict(metric='Clean validation O AP70', direction='maximum',
    tie='earliest epoch on exact full-precision ties', validation_frames=829,
    weather_used=False, random_seed=20261017)


def jsonable(record):
    value = legacy.jsonable(record)
    if isinstance(value, dict) and value.get('stage') == 'D-1D':
        value['stage'] = 'V2XR_WEATHER_L012'
        value['main_protocol'] = 'O'
        value['model_variant'] = value.get('source_hashes', {}).get('variant')
        associated = value.get('associated_d1d_training')
        if associated:
            associated['interpretation'] = 'Associated independent weather L0/L1/L2 training or calibration contract; historical field name retained by frozen evaluator.'
    return value


def sources():
    legacy.sources()
    pins = read_json(HERE / 'PINS.json')
    for name, digest in pins['files'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Frozen dependency changed: ' + name)
    for package in ('v2xr_gspr_clean', 'v2xr_weather_l012'):
        folder = ROOT / package
        expected = read_json(folder / 'source_snapshot.json')['files']
        actual = {p.relative_to(folder).as_posix(): sha256(p) for p in folder.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
        if actual != expected:
            raise ValueError('Frozen package changed: ' + package)
    expected = read_json(HERE / 'source_snapshot.json')['files']
    actual = {p.relative_to(HERE).as_posix(): sha256(p) for p in HERE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if actual != expected:
        raise ValueError('Synchronize the complete gspr_repe package and snapshot')
    slim_tools=ROOT/'v2xr_pointcloud_only'
    tool_sources={p.relative_to(ROOT).as_posix():sha256(p) for p in slim_tools.iterdir()
                 if p.is_file() and p.suffix in ('.py','.md','.sh')}
    return dict(files=actual, dependencies=pins['files'], pointcloud_tools=tool_sources)


def paths(args):
    legacy.paths(args)
    args.baseline_root = str(no_test(args.baseline_root).resolve())
    output, baseline = Path(args.output_root), Path(args.baseline_root)
    if output == baseline or output in baseline.parents or baseline in output.parents:
        raise ValueError('Output overlaps frozen baseline')


def runtime(args, evidence):
    root = legacy.runtime(args, evidence)
    from models import install
    install(args.variant)
    return root


def baseline(args):
    pins = read_json(ROOT / 'v2xr_gspr_clean/PINS.json')['baseline']
    root = Path(args.baseline_root)
    for name, digest in pins['files'].items():
        if sha256(no_test(root / name)) != digest:
            raise ValueError('Frozen Clean-only baseline changed: ' + name)
    selected = read_json(root / 'selected.json')
    if sha256(no_test(selected['checkpoint'])) != pins['checkpoint_sha256']:
        raise ValueError('Actual baseline checkpoint differs')
    history = read_json(root / 'history.json')['epochs']
    if min(history, key=lambda r: (-r['clean_AP']['O']['ap70'], r['epoch']))['epoch'] != selected['epoch']:
        raise ValueError('Historical baseline O selection differs')
    summary = complete(root / 'final_evaluation/summary.json')
    expected = CONFIG['clean_only_O_AP70']
    if (not summary['full_validation'] or summary['frame_count'] != 829 or
            any(summary['results'][c]['AP']['O']['ap70'] != expected[c] for c in expected)):
        raise ValueError('Baseline AP/frame contract differs')
    return summary


# A distinct CUDA contract. Historical ROCm checks and sources remain intact.
import os
if os.environ.get('BACKEND') == 'cuda':
    DEFAULTS.update(runtime_root='/workspace/cjmnet_cuda_env/official-runtime-cu124',
        output_root='/data/cjm/datasets/logs/gspr-repe-cu124-v1',
        paired_root='/data/cjm/datasets/v2xr-lidar-weather-validate-v1',
        gt_audit_root=os.environ.get('CUDA_INPUT_ROOT','/data/cjm/datasets/gspr-repe-cu124-inputs-v1'))
    def gpu():
        from cuda_runtime import gpu as check
        return check()
    def paths(args):
        from cuda_runtime import paths as check
        return check(args)
    def frozen_c(args):
        from cuda_runtime import frozen_gt
        return frozen_gt(args)
