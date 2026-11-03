"""Independent E0/E1/E2 source and path contracts; no Git runtime dependency."""
import os
from pathlib import Path
import sys
from bridge import ROOT, load

HERE = Path(__file__).resolve().parent
sys.path.append(str(ROOT/'v2xr_train_attfuse'))
legacy = load('base_auditlib', 'v2xr_train_attfuse/auditlib.py')
for name in ('atomic_json','bound_json','complete','frozen_c','identity','jsonable','lock',
             'no_test','patched','read_json','request','server_only','sha256','verify_train'):
    globals()[name] = getattr(legacy, name)
SETTINGS = read_json(ROOT/'v2xr_gspr_clean/settings.json')
CONFIG = dict(read_json(ROOT/'gspr_repe/settings.json'), variants=['E1','E2'])
METHOD = read_json(HERE/'settings.json')
SELECTION = dict(metric='Clean validation O AP70',direction='maximum',
    tie='earliest epoch on exact full-precision ties',validation_frames=829,
    weather_used=False,random_seed=20261017)
DEFAULTS = dict(legacy.DEFAULTS,
    baseline_root='/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1',
    a0_root='/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0',
    e0_root='/home/cjm/datasets/logs/gspr-ggca-v3/A1',
    b2_root='/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B2',
    output_root='/home/cjm/datasets/logs/gspr-e1e2-20261101-v5',
    experiment_root='/home/cjm/datasets/logs/gspr-e1e2-20261101-v5')

def load_frozen(name, filename, folder=None):
    return load(name,(Path(folder or ROOT/'v2xr_train_attfuse')/filename).relative_to(ROOT))

def sources():
    pins=read_json(HERE/'PINS.json')['files']
    for name,digest in pins.items():
        if sha256(ROOT/name)!=digest:
            raise ValueError('Frozen E12 dependency changed: '+name)
    actual={p.relative_to(HERE).as_posix():sha256(p) for p in HERE.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.name!='source_snapshot.json'}
    if actual!=read_json(HERE/'source_snapshot.json')['files']:
        raise ValueError('E12 source differs from its delivered snapshot')
    legacy.sources()
    return dict(files=actual,dependencies=pins,method=METHOD)

def separate(destination, source):
    a,b=Path(destination).resolve(),Path(source).resolve()
    if a==b or a in b.parents or b in a.parents:
        raise ValueError('Output overlaps protected input: '+str(b))

def paths(args):
    for k in DEFAULTS:
        setattr(args,k,str(no_test(getattr(args,k)).resolve()))
    for destination in (args.output_root,args.experiment_root):
        if not Path(destination).is_relative_to(Path('/home/cjm/datasets/logs')):
            raise ValueError('E12 outputs must stay under /home/cjm/datasets/logs')
        for source in [ROOT]+[getattr(args,k) for k in DEFAULTS if k not in ('output_root','experiment_root','seven_zip')]:
            separate(destination,source)
    if not Path(args.output_root).is_relative_to(Path(args.experiment_root)):
        raise ValueError('Worker output must belong to the new experiment')
    no_test(os.environ['TMPDIR'])

def gpu():
    if os.environ.get('BACKEND','rocm')!='rocm':
        raise ValueError('Matched E0 experiment requires ROCm')
    legacy.gpu()

def evaluation_path(args,destination):
    path=no_test(destination).resolve()
    if not path.is_relative_to(Path(args.experiment_root)):
        raise ValueError('Evaluation outside the new experiment')
    for source in (ROOT,args.e0_root,args.b2_root,args.a0_root,args.train_root,args.runtime_root,
                   args.model_dir,args.gt_audit_root,args.d1c_root,args.view_root):
        separate(path,source)
    return path

def runtime(args,evidence):
    result=legacy.runtime(args,evidence)
    from models import install
    install(args.variant)
    return result
