"""Run LR-G -> LR1 on the exact LR0 protocol, without altering LR0 or old A1."""
import json
import os
from pathlib import Path
import socket
import sys
import traceback
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / name) for name in
    ('v2xr_lidar_radar_baseline', 'v2xr_official_repro',
     'v2xr_protocol_audit', 'v2xr_weather_l012',
     'gspr_ggca', 'v2xr_train_attfuse')]
sys.path.append(str(REPO))

from common import atomic_json, identity, lock, read_json, server_only, sha256
from runtime import RUNTIME, DATA, create_model as baseline_create_model
from v2xr_lidar_radar_baseline import engine as native_engine
from v2xr_lidar_radar_baseline import run as native_run

LR0 = DATA / 'logs/v2xr-lr0-v3'
ROOT = DATA / 'logs/v2xr-lr1-v1'
VARIANTS = ('LRG', 'LR1')


def source_contract():
    old = native_run.source_contract()
    saved = read_json(LR0 / 'CONTRACT.json')
    if old != saved['sources']:
        raise ValueError('LR0 sources or official runtime have changed since the completed baseline')
    for filename in ('FINAL_RESULTS.json', 'selected.json', 'config.yaml',
                     'final_evaluation/gt_manifest.json'):
        if not (LR0 / filename).is_file():
            raise ValueError('Required completed LR0 evidence is missing: ' + filename)
    if read_json(LR0 / 'FINAL_RESULTS.json')['status'] != 'complete':
        raise ValueError('LR0 baseline is not final/complete')
    return dict(
        lr0_sources=old,
        lr0_config_sha256=sha256(LR0 / 'config.yaml'),
        lr0_gt_sha256=sha256(LR0 / 'final_evaluation/gt_manifest.json'),
        lr0_selected=read_json(LR0 / 'selected.json')['sha256'],
        lr1_files={str(p.relative_to(HERE)): sha256(p) for p in sorted(HERE.iterdir())
                   if p.is_file() and p.suffix in ('.py', '.sh', '.md')},
        frozen_methods={name: sha256(REPO / name) for name in
                        ('attfuse_gspr/reliability.py',
                         'v2xr_gspr_clean/settings.json',
                         'gspr_ggca/modules.py',
                         'gspr_ggca/settings.json',
                         'gspr_ggca/models.py',
                         'gspr_ggca/bridge.py')})


def bind(path, obj):
    if path.is_file():
        if read_json(path) != obj:
            raise ValueError('Existing LR1 contract differs: ' + str(path))
    else:
        atomic_json(path, obj)


def make_model(cfg, variant):
    from empty_input import preserve_empty_vehicle_slots
    from lr1_model import LR1Model
    model = LR1Model(cfg['model']['args'], variant).to('cuda:0')
    preserve_empty_vehicle_slots(model)
    return model


def make_components(cfg, batches, variant):
    from opencood.tools import train_utils
    from lr1_model import LR1Loss
    model = make_model(cfg, variant)
    native = train_utils.create_loss(cfg).to('cuda:0')
    criterion = LR1Loss(native, model).to('cuda:0')
    optimizer = train_utils.setup_optimizer(cfg, model)
    scheduler = train_utils.setup_lr_schedular(cfg, optimizer, n_iter_per_epoch=batches)
    return model, criterion, optimizer, scheduler


@contextmanager
def use_variant(variant):
    """Use LR0's unchanged training, checkpointing and metric engine."""
    old_create, old_components = native_engine.create_model, native_engine.components
    native_engine.create_model = lambda cfg: make_model(cfg, variant)
    native_engine.components = lambda cfg, batches: make_components(cfg, batches, variant)
    try:
        yield
    finally:
        native_engine.create_model, native_engine.components = old_create, old_components


def preflight(cfg_path, destination):
    """Verify exact shared initialization and A1 zero-residual starting state."""
    import torch
    from checkpoint import seed
    from opencood.hypes_yaml import yaml_utils
    from runtime import build, loader
    from opencood.tools import train_utils

    cfg = yaml_utils.load_yaml(str(cfg_path))
    seed(20261017)
    original = baseline_create_model(cfg)
    base_params = {name: value.detach().cpu().clone()
                   for name, value in original.state_dict().items()}
    del original
    torch.cuda.empty_cache()
    variants = {}
    results = {}
    for variant in VARIANTS:
        seed(20261017)
        model = make_model(cfg, variant)
        params = model.state_dict()
        wrong = [k for k, v in base_params.items()
                 if k not in params or not torch.equal(v, params[k].detach().cpu())]
        if wrong:
            raise ValueError('Changed LR0 shared initialization: ' + str(wrong[:6]))
        variants[variant] = {k: p.detach().cpu().clone()
                             for k, p in model.gspr.state_dict().items()}
        if variant == 'LR1' and torch.count_nonzero(model.ggca.project.weight):
            raise ValueError('A1 output projection did not start at exact zero')
        results[variant] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                common_tensors=len(base_params))
        del model
        torch.cuda.empty_cache()
    if set(variants['LRG']) != set(variants['LR1']) or any(
            not torch.equal(v, variants['LR1'][k]) for k, v in variants['LRG'].items()):
        raise ValueError('GSPR parameters do not match between the two variants')
    # End-to-end equivalence on one real batch, without any optimizer step.
    dataset, _ = build(cfg_path, split='validate', condition='clean')
    batch = next(iter(loader(dataset, evaluation=True, limit=1)))
    ego = train_utils.to_device(batch['ego'], torch.device('cuda:0'))
    predictions = {}
    for variant in VARIANTS:
        seed(20261017)
        model = make_model(cfg, variant).eval()
        with torch.no_grad():
            predictions[variant] = {k: v.detach().cpu() for k, v in model(ego).items()}
        del model
        torch.cuda.empty_cache()
    differences = {k: float((predictions['LRG'][k] - predictions['LR1'][k]).abs().max())
                   for k in ('psm', 'rm')}
    if any(not torch.allclose(predictions['LRG'][k], predictions['LR1'][k],
                              atol=1e-6, rtol=1e-6) for k in ('psm', 'rm')):
        raise ValueError('A1 must start as the GSPR-only model on the same input')
    atomic_json(destination, dict(status='complete', shared=results,
                                  gspr_initialization_identical=True,
                                  zero_residual_initial_max_abs=differences))


def verify_dual_gt(output):
    """Same clean/weather GT IDs as LR0 on every cooperative frame, not just counts."""
    a = read_json(LR0 / 'final_evaluation/gt_manifest.json')['frames']
    b = read_json(output / 'final_evaluation/gt_manifest.json')['frames']
    if len(a) != len(b):
        raise ValueError('LR1 changed the number of LR0 evaluation frames')
    for x, y in zip(a, b):
        for key in ('condition', 'index', 'frame_key', 'official_ids'):
            if x.get(key) != y.get(key):
                raise ValueError('LR1 / LR0 dual-modal GT differs: ' +
                                 str((x.get('condition'), x.get('index'), key)))
    atomic_json(output / 'MATCHED_GT.json',
                dict(status='identical', frames=len(a),
                     lr0_gt_sha256=sha256(LR0 / 'final_evaluation/gt_manifest.json'),
                     lr1_gt_sha256=sha256(output / 'final_evaluation/gt_manifest.json')))


def run_variant(variant, sources, runtime_record):
    out = ROOT / variant
    out.mkdir(parents=True, exist_ok=True)
    config_source = LR0 / 'config.yaml'
    cfg = out / 'config.yaml'
    native_config = config_source.read_bytes()
    if cfg.exists():
        if cfg.read_bytes() != native_config:
            raise ValueError('LR1 model config differs from locked LR0 config')
    else:
        cfg.write_bytes(native_config)
    contract = dict(source=sources, variant=variant, run='from_scratch',
                    protocol='LR0: 8083 train; Clean4/7 Fog3/7 per-CAV; 20 epochs; Clean O AP70',
                    gpu=runtime_record, config_sha256=sha256(cfg))
    bind(out / 'CONTRACT.json', contract)
    contract_id = identity(contract)
    with lock(out / '.lock'), use_variant(variant):
        print('LR1 START', variant, 'SMOKE', flush=True)
        native_engine.smoke(out, cfg, contract_id)
        print('LR1 START', variant, '20-EPOCH TRAIN', flush=True)
        history = native_engine.train(out, cfg, contract_id)
        print('LR1 START', variant, 'FINAL EVALUATION', flush=True)
        native_engine.finalize(out, cfg, contract_id, history)
        verify_dual_gt(out)
        result = read_json(out / 'FINAL_RESULTS.json')
        result['experiment'] = variant
        result['base_experiment'] = 'LR0'
        result['matched_lr0_gt'] = read_json(out / 'MATCHED_GT.json')
        result['source_contract'] = sha256(out / 'CONTRACT.json')
        atomic_json(out / 'FINAL_RESULTS.json', result)
        report = out / 'RESULTS.md'
        report.write_text(report.read_text(encoding='utf-8').replace(
            '# LR0 正式结果', '# ' + variant + ' 正式结果'),
            encoding='utf-8', newline='\n')
        print('LR1 COMPLETE', variant, flush=True)
    return dict(variant=variant, result_sha256=sha256(out / 'FINAL_RESULTS.json'),
                selected=read_json(out / 'selected.json'),
                final_ap={weather: r['AP'] for weather, r in result['evaluation']['results'].items()})


def main():
    server_only()
    if socket.gethostname().split('.')[0] != 'admin1' or os.environ.get('ROCR_VISIBLE_DEVICES') != '1':
        raise ValueError('Run on admin1 with only physical GPU 1 visible')
    import torch
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one HIP GPU is required')
    if not Path(__import__('opencood').__file__).is_relative_to(RUNTIME):
        raise RuntimeError('Wrong OpenCOOD runtime')
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = source_contract()
    runtime_record = dict(physical_gpu=1, torch=str(torch.__version__),
                          hip=torch.version.hip, python=sys.version,
                          hash_seed=os.environ['PYTHONHASHSEED'])
    preflight_file = ROOT / 'PREFLIGHT.json'
    if preflight_file.is_file():
        done = read_json(preflight_file)
        if done.get('status') != 'complete' or not done.get('gspr_initialization_identical'):
            raise ValueError('Bad preflight record')
    else:
        preflight(LR0 / 'config.yaml', preflight_file)
    results = []
    for variant in VARIANTS:
        results.append(run_variant(variant, sources, runtime_record))
    atomic_json(ROOT / 'FINAL_COMPARISON.json',
                dict(status='complete', lr0=sha256(LR0 / 'FINAL_RESULTS.json'),
                     variants=results,
                     note='Use identical dual-modal GT for LR0, LRG, LR1; all results are validation only'))
    print('COMPLETE LR1 ALL VARIANTS', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(ROOT / 'FAILURE.json',
                    dict(status='failed', traceback=traceback.format_exc()))
        raise
