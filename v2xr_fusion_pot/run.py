"""One restartable entry point. LR0 owns data, training schedule and evaluation."""
from contextlib import contextmanager
from importlib import metadata
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[1:1] = [str(REPO / p) for p in ('v2xr_lidar_radar_baseline', 'v2xr_official_repro',
                                        'v2xr_protocol_audit', 'v2xr_weather_l012')]
sys.path.append(str(REPO))

from common import atomic_json, identity, lock, read_json, server_only, sha256
from runtime import DATA, RUNTIME

sys.path.insert(0, str(RUNTIME))
OUT = DATA / 'logs/v2xr-fusion-pot-v1'
sys.path.append(str(OUT / 'deps'))

from device import map_device, pre_device

LR0 = DATA / 'logs/v2xr-lr0-v3'
WEIGHT = LR0 / 'epochs/18/weights.pth'
CONFIG = LR0 / 'config.yaml'
WEATHERS = ('clean', 'fog', 'snow')
GROUPS = {'P1': ('attention', 'matched_only'), 'P2': ('unbalanced', 'matched_only'),
          'P3': ('unbalanced', 'matched_plus_unmatched')}
SETTINGS = read_json(HERE / 'settings.json')
STATUS = {'stage': 'entry'}


def status(stage, **extra):
    STATUS.update(stage=stage, **extra)
    atomic_json(OUT / 'STATUS.json', dict(status='running', **STATUS))
    print('STAGE', stage, extra, flush=True)


def bind(path, value):
    if path.is_file():
        if read_json(path) != value:
            raise ValueError(f'Pinned record differs: {path}')
    else:
        atomic_json(path, value)


def audit():
    import opencood
    from inputs import TRAIN_MANIFEST, VAL_MANIFEST
    from runtime import EMPTY_POLICY
    from v2xr_lidar_radar_baseline.run import source_contract

    if EMPTY_POLICY != 'zero' or not Path(opencood.__file__).is_relative_to(RUNTIME):
        raise RuntimeError('Incorrect official runtime or empty-observation policy')
    if metadata.version('POT') != '0.9.5':
        raise RuntimeError('Install POT==0.9.5 into the isolated deps directory')
    expected = {
        WEIGHT: '0eb0ec8bd6a6e0bc368c7cc8aacd49a33cfe014413463f8b314fdbf76c6defd2',
        CONFIG: '4b8a3626d6fa94fd4ef841ba67dddb01f6030f2c964d0ed2d142c60eca8d59b9',
        LR0 / 'FINAL_RESULTS.json': 'c98d327ab834535916e5f3bc5dfd0ba94afda09fb16730c06d81d46c598e9758',
        LR0 / 'CONTRACT.json': '5479e679e12f5dda760d55d8039c22d99e38e73a2f30ddc550372d705f4bf652',
        LR0 / 'DATA.json': '0615a36a5ad056cba1053b91ffd406af1b6c0d03cb589c6e0201a09d99a7beab',
    }
    for file, digest in expected.items():
        if sha256(file) != digest:
            raise ValueError(f'Pinned LR0 artifact differs: {file}')
    previous = read_json(LR0 / 'CONTRACT.json')
    if source_contract() != previous['sources']:
        raise ValueError('Official runtime, LR0 or reused evaluation source changed')
    data = read_json(LR0 / 'DATA.json')
    for split, path, count in (('train', TRAIN_MANIFEST, 8083), ('validate', VAL_MANIFEST, 829)):
        row = data['splits'][split]
        if row['frames'] != count or sha256(path) != row['input_manifest_sha256']:
            raise ValueError(f'{split} manifest differs from audited LR0 inputs')
    final = read_json(LR0 / 'FINAL_RESULTS.json')
    reference = LR0 / 'final_evaluation'
    for name in ('gt_manifest.json', 'per_frame_metrics.jsonl', 'prediction_manifest.json'):
        digest = sha256(reference / name)
        if digest != final['evaluation']['artifacts'][name]:
            raise ValueError(f'Frozen LR0 prediction/GT evidence changed: {name}')
        expected[reference / name] = digest
    for path in HERE.iterdir():
        if path.suffix in ('.sh', '.py', '.md', '.json') and b'\r' in path.read_bytes():
            raise ValueError(f'Expected LF line endings: {path}')
    # At least two full runs' worth of epoch checkpoints must fit before starting.
    if not (OUT / 'CONTRACT.json').is_file() and shutil.disk_usage(OUT).free < 36 * 2**30:
        raise RuntimeError('Less than 36 GiB free on the experiment data disk')
    bind(OUT / 'DATA.json', data)
    own = {p.name: sha256(p) for p in sorted(HERE.glob('*.py'))}
    own.update({p: sha256(HERE / p) for p in ('run_all.sh', 'settings.json')})
    return dict(sources=own, lr0={str(k): v for k, v in expected.items()},
                packages={name: metadata.version(name) for name in ('torch', 'POT', 'numpy', 'scipy',
                                                                     'open3d', 'shapely', 'PyYAML')},
                python=sys.version, hash_seed=os.environ['PYTHONHASHSEED'], empty_policy=EMPTY_POLICY)


def make_model(cfg, group):
    from model import Model

    solver, mode = GROUPS[group]
    return Model(cfg['model']['args'], SETTINGS, solver, mode).to('cuda:0')


def components(cfg, batches, group):
    from opencood.tools import train_utils

    model = make_model(cfg, group)
    criterion = train_utils.create_loss(cfg).to('cuda:0')
    optimizer = train_utils.setup_optimizer(cfg, model)
    scheduler = train_utils.setup_lr_schedular(cfg, optimizer, n_iter_per_epoch=batches)
    return model, criterion, optimizer, scheduler


def update(model, criterion, optimizer, batch, epoch, inspect=False, fusion_only=False):
    import torch
    import engine
    from opencood.tools import train_utils

    model.train(not fusion_only)
    if fusion_only:
        model.fusion.train()
    optimizer.zero_grad(set_to_none=True)
    ego = train_utils.to_device(batch['ego'], torch.device('cuda:0'))
    ego['epoch'] = epoch
    if inspect:
        prediction, shapes = engine.checked_forward(model, ego)
    else:
        prediction, shapes = model(ego), None
    loss = criterion(prediction, ego['label_dict'])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f'Nonfinite detection loss at epoch {epoch + 1}')
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
        raise FloatingPointError(f'Missing/nonfinite gradient at epoch {epoch + 1}')
    optimizer.step()
    return float(loss.detach()), shapes


@contextmanager
def use_group(group):
    import engine
    from support import patched

    with patched(engine, 'components', lambda cfg, batches: components(cfg, batches, group)), \
            patched(engine, 'train_step', update):
        yield


def pilot(destination, contract, group, steps, fusion_only):
    import torch
    import engine
    from checks import assert_layout, roundtrip
    from checkpoint import seed
    from runtime import BATCH, SEED, build, loader

    done = destination / 'complete.json'
    if done.is_file():
        evidence = read_json(done)
        if evidence['contract'] != contract or any(sha256(destination / name) != digest
                                                  for name, digest in evidence['files'].items()):
            raise ValueError(f'Invalid acceptance evidence at {done}')
        return evidence
    seed(SEED)
    dataset, cfg = build(CONFIG, split='train')
    batches = loader(dataset, training=True, limit=BATCH * steps)
    model, criterion, optimizer, scheduler = components(cfg, len(batches), group)
    missing, unexpected = model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True),
                                                strict=False)
    extras = {k: p for k, p in model.named_parameters() if k.startswith('backbone.fuse_modules.2.')}
    if set(missing) != set(extras) or unexpected:
        raise ValueError('Could not load identical LR0 epoch-18 shared weights')
    if fusion_only:
        for name, param in model.named_parameters():
            param.requires_grad_(name in extras)
    torch.cuda.reset_peak_memory_stats()
    losses, gradients, shapes = [], {}, None
    from collections import Counter
    from empty_input import empty_slots
    empty = Counter()
    initial = {k: p.detach().clone() for k, p in extras.items()}
    for i, batch in enumerate(batches):
        assert_layout(batch['ego'])
        empty.update(empty_slots(batch['ego']))
        tick = time.monotonic()
        value, shape = update(model, criterion, optimizer, batch, 0, inspect=True, fusion_only=fusion_only)
        shapes = shapes or shape
        losses.append(value)
        gradients = {k: float(p.grad.norm()) if p.grad is not None else 0. for k, p in extras.items()}
        print(f'{group} {"PROTOTYPE" if fusion_only else "SMOKE"} {i+1}/{steps} '
              f'loss={value:.6f} seconds={time.monotonic()-tick:.2f}', flush=True)
    required = ['matched.weight', 'correspondence.query.weight', 'correspondence.key.weight']
    if group == 'P3':
        required += ['unmatched.weight', 'gate.weight']
    for suffix in required:
        k = 'backbone.fuse_modules.2.' + suffix
        if not math.isfinite(gradients[k]) or gradients[k] == 0 or torch.equal(initial[k], extras[k]):
            raise ValueError(f'{group} {suffix} did not receive a finite nonzero gradient and update')
    if len(losses) != steps:
        raise ValueError('Acceptance update count differs')
    peak = torch.cuda.max_memory_allocated()
    resume = destination / 'resume.pth'
    scheduler.step(0)
    digest = roundtrip(resume, model, optimizer, scheduler, criterion, contract)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    evaluation = None
    files = {resume.name: digest}
    if not fusion_only:
        evaluation = engine.evaluate(model, criterion, CONFIG, destination / 'evaluation', WEATHERS,
                                      SETTINGS['smoke_frames_per_weather'], contract, digest, loss=True)
        files['evaluation/summary.json'] = sha256(destination / 'evaluation/summary.json')
    evidence = dict(status='complete', contract=contract, losses=losses, shapes=shapes, empty_slots=empty,
                    last_gradients=gradients, peak_gpu_allocated_bytes=peak,
                    resume_roundtrip=True, evaluation=evaluation, files=files)
    atomic_json(done, evidence)
    return evidence


def workflow(check_only=False):
    status('environment_and_source_audit')
    card = pre_device()
    contract = audit()
    mapped = map_device()
    contract['device'] = mapped
    contract_id = identity(contract)
    contract_path = OUT / 'CONTRACT.json'
    if contract_path.is_file() and read_json(contract_path) != contract:
        raise ValueError('A formal run exists under different sources or environment')
    print('DEVICE VERIFIED', mapped, flush=True)
    validation = OUT / 'acceptance' / contract_id[:16]
    status('numerical_tests')
    from checks import integration, numerical
    tested = validation / 'checks.json'
    if tested.is_file():
        checks = read_json(tested)
        if checks['contract'] != contract_id:
            raise ValueError('Acceptance check identity changed')
    else:
        numerical_report = numerical(SETTINGS)
        print('NUMERICAL PASSED', numerical_report, flush=True)
        native = integration(CONFIG, SETTINGS, GROUPS, make_model, WEIGHT)
        print('REAL INTEGRATION PASSED', native, flush=True)
        checks = dict(contract=contract_id, device=mapped, idle_before_launch=card, numerical=numerical_report,
                      integration=native)
        atomic_json(tested, checks)
    status('real_smoke')
    smoke = {group: pilot(validation / 'smoke' / group, contract_id, group, 2, False) for group in GROUPS}
    atomic_json(OUT / 'SMOKE_RESULTS.json', dict(status='smoke_complete', **checks, smoke=smoke))
    status('prototype')
    prototypes = {group: pilot(validation / 'prototype' / group, contract_id, group,
                               SETTINGS['prototype_steps'], True) for group in GROUPS}
    atomic_json(OUT / 'SMOKE_RESULTS.json', dict(status='complete', **checks, smoke=smoke, prototype=prototypes))
    print('SMOKE AND PROTOTYPE COMPLETE', flush=True)
    if check_only:
        atomic_json(OUT / 'STATUS.json', dict(status='validation_complete', **STATUS))
        return
    bind(contract_path, contract)

    import engine
    from diagnostics import evaluate_formal, report

    results = {'P0': read_json(LR0 / 'FINAL_RESULTS.json')}
    for group in GROUPS:
        destination = OUT / group
        status('formal_train', group=group)
        with use_group(group):
            rows = engine.train(destination, CONFIG, contract_id)
        status('formal_evaluate', group=group)
        results[group] = evaluate_formal(destination, CONFIG, contract_id, group, rows, make_model)
    status('mechanism_diagnostics', group=None)
    report(OUT, CONFIG, contract_id, results, make_model, SETTINGS)
    atomic_json(OUT / 'STATUS.json', dict(status='complete', stage='complete', contract=contract_id))
    print('COMPLETE POT-CF', flush=True)


if __name__ == '__main__':
    server_only()
    if sys.argv[1:] not in ([], ['--check-only']):
        raise SystemExit('Usage: sh run_all.sh [--check-only]')
    with lock(OUT / '.lock'):
        try:
            workflow(check_only='--check-only' in sys.argv)
        except Exception:
            failure = dict(status='failed', traceback=traceback.format_exc(), **STATUS)
            atomic_json(OUT / 'FAILURE.json', failure)
            atomic_json(OUT / 'STATUS.json', failure)
            raise
