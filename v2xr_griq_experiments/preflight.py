"""Verify frozen checkpoint, sources and manifests before opening the cache."""
import shutil

from v2xr_griq_experiments.dataset import DATA, F0, LEGACY_CONTRACT, OUT, REPO, ROOT, RUNTIME, SEED, bind, read, save, sha

def preflight():
    selected = read(F0 / 'selected.json')
    weight = sha(selected['checkpoint'])
    if weight != selected['sha256'] or selected['epoch'] != 20 or weight != 'a5e12d10cde649c5a198b814277134ae06f8069a5b8e4d32505438c19b1fe8b1':
        raise ValueError('Expected formal F0 epoch 20')
    historical = read(F0.parent / 'CONTRACT.json')
    for name, digest in historical['source'].items():
        if sha(REPO / 'v2xr_radar_ra_rb' / name) != digest:
            raise ValueError(f'F0 source differs: {name}')
    for name, digest in historical['frozen'].items():
        if sha(REPO / name) != digest:
            raise ValueError(f'Frozen F0 dependency differs: {name}')
    config = sha(F0 / 'config.yaml')
    if config != read(F0 / 'CONTRACT.json')['base_config']:
        raise ValueError('F0 config differs')
    files = [
        F0 / 'selected.json', F0 / 'final_evaluation/prediction_manifest.json',
        F0 / 'final_evaluation/gt_manifest.json',
        DATA / 'v2xr-official-train-v2/manifest.json',
        DATA / 'logs/v2xr-official-gt-audit-v1/input_manifest.json',
        RUNTIME / 'opencood/data_utils/post_processor/voxel_postprocessor.py',
        RUNTIME / 'opencood/utils/eval_utils_opv2v.py',
        REPO / 'v2xr_radar_evidence/collect.py', REPO / 'v2xr_radar_evidence/context.py',
        REPO / 'v2xr_lidar_radar_baseline/metrics.py', REPO / 'v2xr_protocol_audit/evaluate.py',
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(OUT).free < 5 * 2**30:
        raise RuntimeError('Need at least 5 GiB on the output filesystem')
    path = OUT / 'CONTRACT.json'
    previous = OUT / 'CONTRACT.v1.json'
    value = dict(weight=weight, config=config, seed=SEED,
                 source={str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*'))
                         if p.suffix in ('.py','.sh')},
                 inputs={str(p): sha(p) for p in files})
    if previous.exists() or (path.exists() and sha(path) == LEGACY_CONTRACT):
        old_file = previous if previous.exists() else path
        if sha(old_file) != LEGACY_CONTRACT:
            raise ValueError('Unrecognized legacy contract')
        old = read(old_file)
        if any(old[k] != value[k] for k in ('weight', 'config', 'seed', 'inputs')):
            raise ValueError('Cannot reuse cache with different frozen inputs')
        unchanged = set(old['source']) - {'dataset.py', 'prepare_candidates.py', 'audit.py',
                                          'preflight.py', 'launch.py', 'worker.py'}
        if set(old['source']) != set(value['source']) or any(
                old['source'][k] != value['source'][k] for k in unchanged):
            raise ValueError('Source changes exceed the cache-compatible GT repair')
        value['compatible_train_cache'] = LEGACY_CONTRACT
        if sha(path) == LEGACY_CONTRACT:
            if not previous.exists():
                previous.write_bytes(path.read_bytes())
            save(path, value)
            print('REUSE HASH-VERIFIED V1 TRAINING FRAMES', flush=True)
    bind(path, value)


if __name__ == '__main__':
    preflight()
