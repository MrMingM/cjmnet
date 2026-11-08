"""One fail-closed, restartable workflow on the sea-light server."""
from importlib import metadata
import os
from pathlib import Path
import socket
import sys
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / name) for name in
               ('v2xr_official_repro', 'v2xr_protocol_audit', 'v2xr_weather_l012')]
sys.path.append(str(REPO))

from common import atomic_json, identity, lock, read_json, server_only, sha256
from inputs import DATA, TRAIN_MANIFEST, VAL_MANIFEST, inventory
from runtime import CONFIG, EMPTY_POLICY, RUNTIME, config

DATA_CHECK_ROOT = DATA / 'logs/v2xr-lr0-v1'
OUT = DATA / 'logs' / ('v2xr-lr0-v3-reject' if EMPTY_POLICY == 'reject' else 'v2xr-lr0-v3')


def source_contract():
    expected = read_json(HERE / 'source_snapshot.json')
    actual = {p.name: sha256(p) for p in HERE.iterdir()
              if p.is_file() and p.name != 'source_snapshot.json'}
    if actual != expected:
        raise ValueError('LR0 source snapshot differs; transfer the complete package')
    files = {}
    for package, paths in {
        'v2xr_official_repro': ['common.py'],
        'v2xr_protocol_audit': ['evaluate.py', 'predict.py', 'support.py'],
        'v2xr_weather_l012': ['auditlib.py', 'checkpoint.py', 'data.py', 'settings.json'],
        'v2xr_train_attfuse': ['auditlib.py'],
        'v2xr_gspr_clean': ['settings.json'],
    }.items():
        for path in paths:
            p = REPO / package / path
            files[f'{package}/{path}'] = sha256(p)
    runtime = {str(p.relative_to(RUNTIME)): sha256(p) for p in sorted((RUNTIME / 'opencood').rglob('*'))
               if p.is_file() and p.suffix in ('.py', '.so', '.yaml')}
    return dict(lr0=expected, legacy=files, runtime=runtime, official_config_sha256=sha256(CONFIG))


def bind(path, value):
    if path.is_file() and read_json(path) != value:
        raise ValueError(f'Experiment identity changed: {path}')
    if not path.is_file():
        atomic_json(path, value)


def workflow():
    server_only()
    if socket.gethostname().split('.')[0] != 'admin1' or os.environ.get('ROCR_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Run LR0 on admin1 physical card 1 via run_all.sh')
    os.chdir(HERE)
    sys.path.insert(0, str(RUNTIME))
    print('START DATA', flush=True)
    sources = source_contract()
    input_check = DATA_CHECK_ROOT / 'DATA.json'
    data = read_json(input_check) if input_check.is_file() else inventory(input_check)
    if data['status'] != 'complete' or any(
            sha256(p) != data['splits'][split]['input_manifest_sha256']
            for split, p in (('train', TRAIN_MANIFEST), ('validate', VAL_MANIFEST))):
        raise ValueError('Data check incomplete or input manifests changed')
    data_path = OUT / 'DATA.json'
    bind(data_path, data)
    import torch
    import yaml
    import opencood
    if not Path(opencood.__file__).is_relative_to(RUNTIME):
        raise RuntimeError('Imported the wrong OpenCOOD runtime')
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible HIP GPU is required')
    cfg = OUT / 'config.yaml'
    config_text = yaml.safe_dump(config(), sort_keys=False)
    if cfg.is_file() and cfg.read_text() != config_text:
        raise ValueError('Effective LR0 configuration changed')
    if not cfg.is_file():
        cfg.write_text(config_text, encoding='utf-8', newline='\n')
    contract = dict(sources=sources, data_sha256=sha256(data_path), config_sha256=sha256(cfg),
        lidar_gt_reference_sha256=sha256(DATA / 'logs/v2xr-dminus1c-protocol-v1/gt_manifest.json'),
        torch=str(torch.__version__), hip=torch.version.hip, gpu=torch.cuda.get_device_name(0),
        physical_gpu=1, python=sys.version, hash_seed=os.environ['PYTHONHASHSEED'],
        packages={p: metadata.version(p) for p in ('numpy', 'shapely', 'open3d', 'PyYAML')},
        empty_input=EMPTY_POLICY,
        invalid_box_policy='exclude nonfinite, invalid or zero-area candidates before native NMS')
    bind(OUT / 'CONTRACT.json', contract)
    from engine import finalize, smoke, train

    contract_id = identity(contract)
    print('START SMOKE', flush=True)
    smoke(OUT, cfg, contract_id)
    print('START FORMAL', flush=True)
    rows = train(OUT, cfg, contract_id)
    print('START SELECT AND EVALUATE', flush=True)
    finalize(OUT, cfg, contract_id, rows)
    print('COMPLETE LR0', flush=True)


if __name__ == '__main__':
    server_only()
    with lock(OUT / '.lock'):
        try:
            workflow()
        except Exception:
            atomic_json(OUT / 'FAILURE.json', dict(status='failed', traceback=traceback.format_exc()))
            raise

