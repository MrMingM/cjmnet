"""Bind the archived LR0 runtime, validation evidence and physical cards."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path.append(str(REPO))
sys.path[:0] = [str(REPO/'v2xr_fusion_pot'), str(REPO/'v2xr_fusion_pot/parallel')]
from v2xr_fusion_interaction.common import audit as devices
from common import atomic_json, identity, read_json, sha256
from inputs import VAL_MANIFEST
from diagnostics import verify_gt
from v2xr_lidar_radar_baseline.run import source_contract

lr = devices.lr
OUT = Path(os.environ.get('AUDIT_OUT', '/data/cjm/datasets/logs/v2xr-fusion-hypothesis-audit-v1'))
BASE = Path('/data/new_disk/cjm_logs/v2xr-fusion-interaction-v1/BASELINE/final_evaluation')
SINGLE = lr.OUT/'ablations/P0/single'
CLEAN = lr.LR0/'final_evaluation/predictions/clean'
WEATHERS = lr.WEATHERS
GPU = {'V1': (1,'0000:0a:00.0','71010022e8ce3841'), 'V2': devices.DEVICES['F2'],
       'V3': devices.DEVICES['F3']}
devices.DEVICES = GPU


def contract():
    import shutil

    expected = '0eb0ec8bd6a6e0bc368c7cc8aacd49a33cfe014413463f8b314fdbf76c6defd2'
    if not OUT.is_dir() or not os.access(OUT, os.W_OK) or shutil.disk_usage(OUT).free < 5*2**30:
        raise RuntimeError(f'Output directory must be writable with 5 GiB free: {OUT}')
    lr0 = read_json(lr.LR0/'CONTRACT.json')
    if sha256(lr.WEIGHT) != expected or sha256(lr.CONFIG) != lr0['config_sha256']:
        raise ValueError('P0 weight or config differs')
    if source_contract() != lr0['sources']:
        raise ValueError('Official source differs from LR0')
    data = read_json(lr.LR0/'DATA.json')['splits']['validate']
    if data['frames'] != 829 or sha256(VAL_MANIFEST) != data['input_manifest_sha256']:
        raise ValueError('Paired validation manifest differs')
    for ref in (BASE, SINGLE):
        verify_gt(ref)
    files = [BASE/'summary.json', BASE/'per_frame_metrics.jsonl', SINGLE/'per_frame_metrics.jsonl',
             lr.OUT/'p0-frozen-failure-v1/COHORT.json',
             REPO/'v2xr_fusion_pot/parallel/p0_failure.py', REPO/'v2xr_fusion_pot/diagnostics.py',
             REPO/'v2xr_fusion_interaction/common/support.py']
    record = dict(weight=expected, config=lr0['config_sha256'], data=data,
                  official_source=identity(lr0['sources']), cards=GPU,
                  references={str(p):sha256(p) for p in files},
                  source={p.relative_to(HERE).as_posix():sha256(p) for p in HERE.rglob('*')
                          if p.suffix in ('.py','.md','.sh')})
    path = OUT/'CONTRACT.json'
    if path.exists():
        old = read_json(path)
        if 'inherited_cache_identity' in old:
            record['inherited_cache_identity'] = old['inherited_cache_identity']
        if 'native_metadata_cache_identity' in old:
            record['native_metadata_cache_identity'] = old['native_metadata_cache_identity']
        if identity(old) != identity(record):
            allowed = {'shared/input_audit.py', 'shared/cache.py', 'shared/extract.py'}
            changed = {k for k in old['source'].keys() | record['source'].keys()
                       if old['source'].get(k) != record['source'].get(k)}
            if (identity(old) != '280284e0a59c92c0d39d4965c5781f67238a0dca7aa3578c46fb376bed0bc1ae'
                    or changed - allowed or identity({k:v for k,v in old.items() if k!='source'})
                    != identity({k:v for k,v in record.items() if k!='source'})):
                raise ValueError('Existing output has a different contract; select a fresh AUDIT_OUT')
            inherited = {p.name:sha256(p) for p in sorted((OUT/'cache').glob('*.npz'))}
            atomic_json(OUT/'CONTRACT_BEFORE_NATIVE_METADATA.json', old)
            atomic_json(OUT/'CACHE_BEFORE_NATIVE_METADATA.json', dict(contract=identity(old), files=inherited))
            record['native_metadata_cache_identity'] = sha256(OUT/'CACHE_BEFORE_NATIVE_METADATA.json')
    atomic_json(path, record)


def mapped(stage):
    devices.idle([stage])
    return devices.device(stage)
