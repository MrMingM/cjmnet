"""Explicit adapters to hash-pinned D-1D infrastructure; no historical edits."""
import copy
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'v2xr_train_attfuse'
sys.path.append(str(BASE))


def load_frozen(name, filename):
    key = '_v2xr_gspr_inherited_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, BASE / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


legacy = load_frozen('auditlib', 'auditlib.py')
for _name in ('atomic_json', 'bound_json', 'complete', 'evaluation_path', 'frozen_c', 'gpu',
              'identity', 'lock', 'no_test', 'patched', 'read_json', 'request', 'server_only',
              'sha256', 'verify_train'):
    globals()[_name] = getattr(legacy, _name)
SETTINGS = read_json(HERE / 'settings.json')
DEFAULTS = dict(legacy.DEFAULTS, output_root='/data/cjm/datasets/logs/v2xr-gspr-clean-train-v2',
                baseline_root='/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1')
SELECTION = copy.deepcopy(legacy.SELECTION)
SELECTION.update(metric='Clean validation O AP70', experiment=SETTINGS,
                 initialization=SETTINGS['initialization'])


def sources():
    old = legacy.sources()  # Checks all historical dependency snapshots without modifying them.
    pins = read_json(HERE / 'PINS.json')
    for name, digest in pins['files'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Frozen GSPR/infrastructure file changed: ' + name)
    expected = read_json(HERE / 'source_snapshot.json')['files']
    actual = {p.relative_to(HERE).as_posix(): sha256(p) for p in HERE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if actual != expected:
        raise ValueError('Synchronize the complete v2xr_gspr_clean directory and snapshot')
    return dict(experiment=actual, frozen_dependencies=pins['files'], inherited_d1d=old)


def paths(args):
    legacy.paths(args)
    args.baseline_root = str(no_test(args.baseline_root).resolve())
    baseline, output = Path(args.baseline_root), Path(args.output_root)
    if output == baseline or output in baseline.parents or baseline in output.parents:
        raise ValueError('GSPR output must be independent of the completed plain AttFuse run')


def jsonable(record):
    value = legacy.jsonable(record)
    if isinstance(value, dict) and value.get('stage') == 'D-1D':
        value['stage'] = 'V2XR_GSPR_CLEAN_O'
        value['gspr_adapter'] = SETTINGS
        value['actual_model_class'] = 'integration.PointPillarGsprOfficial'
        if 'initializer' in value:
            value['initializer'] = 'official base random initialization + frozen GSPR architecture, seed 20261017'
        if 'reused_functions' in value:
            value['model_factory_adaptation'] = 'explicit GSPR subclass; official dataset/postprocess/AP unchanged'
    return value


def runtime(args, evidence):
    root = legacy.runtime(args, evidence)
    from integration import install
    install()
    return root
