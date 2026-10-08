"""Strict resumable server driver: preflight -> baseline -> S0/S1/S2/S3."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import signal
import sys
import traceback
from .common import (ROOT, WEATHERS, Manifest, Runtime, assert_development_paths,
                     atomic_json, digest, object_hash, read_json, scene_split,
                     settings, source_identity, validate_run, verify_protocol_snapshot)


def prepare(args, manifest):
    import yaml
    from gspr_communication.runtime import verify_frozen
    from .features import feature_leakage_check, schema
    source = source_identity()
    print('SOURCE ' + source['provenance'] + ' branch=' + source['source_branch']
          + ' commit=' + source['source_commit'], flush=True)
    spec = settings(args.config)
    inputs = {name: str(Path(getattr(args, name)).resolve()) for name in
              ('config', 'v3_config', 'frontend_config', 'frontend_checkpoint', 'v3_checkpoint', 'b0_run')}
    # Inspect plain YAML BEFORE er.load_config() can enumerate dataset roots.
    v3_options = yaml.safe_load(Path(inputs['v3_config']).read_text(encoding='utf-8'))
    front = yaml.safe_load(Path(inputs['frontend_config']).read_text(encoding='utf-8'))
    paths = {key: v3_options.get(key) or front.get(key) for key in ('root_dir', 'validate_dir')}
    if any(not value for value in paths.values()):
        raise ValueError('Missing dataset root')
    assert_development_paths(*paths.values())
    b0_run = Path(inputs['b0_run'])
    b0 = read_json(b0_run / 'protocol.json')
    saved = read_json(b0_run / 'decision_results.json')
    if spec['baseline_seed'] != b0['pilot']['seed'] + 1:
        raise ValueError('baseline_seed must equal saved B0 evaluation seed')
    for split in ('train', 'validation'):
        indices = b0[split + '_indices']
        if not indices or any(type(i) is not int or i < 0 for i in indices) or len(indices) != len(set(indices)):
            raise ValueError('Invalid B0 ' + split + ' indices')
        if not b0.get(split + '_scenes'):
            raise ValueError('B0 must record scene identities')
    if b0['validation_indices'] != sorted(b0['validation_indices']):
        raise ValueError('B0 validation order must remain sorted')
    fit, calibration = scene_split(b0['train_scenes'], spec['calibration_scene_fraction'], spec['seed'])
    frozen = verify_frozen()
    source_files = set(b0.get('source_hashes', {}))
    source_files.update(p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.iterdir()
                        if p.suffix in ('.py', '.yaml', '.sh', '.md'))
    source_files.update(('local_fusion_task_source_oracle/oracle.py',
                         'local_fusion_task_split_pilot/proposal_constrained_oracle.py',
                         'qa_local_intervention/operators.py', 'qa_local_intervention/common.py',
                         'opencood/utils/box_utils.py', 'opencood/utils/common_utils.py'))
    snapshot = Path(__file__).parent / 'source_snapshot.json'
    if snapshot.is_file():
        source_files.add(snapshot.relative_to(ROOT).as_posix())
    source_hashes = {name: digest(ROOT / name) for name in sorted(source_files)}
    checkpoints = {name: digest(inputs[name]) for name in ('frontend_checkpoint', 'v3_checkpoint')}
    checkpoints['b0_shared'] = digest(b0_run / 'Shared.pth')
    identity = {'source_commit': source['source_commit'],
                'config_hash': object_hash({'spec': spec, 'inputs': inputs,
                                            'v3_config_hash': digest(inputs['v3_config']),
                                            'frontend_config_hash': digest(inputs['frontend_config']),
                                            'b0_protocol_hash': digest(b0_run / 'protocol.json'),
                                            'b0_result_hash': digest(b0_run / 'decision_results.json')}),
                'checkpoint_hashes': checkpoints, 'source_hash': object_hash(source_hashes)}
    if manifest.value['identity'] and manifest.value['identity'] != identity:
        raise ValueError('Resume rejected: commit/source/config/checkpoints changed; use a new RUN')
    manifest.value['identity'] = identity
    manifest.save()
    protocol = {'purpose': 'Direction B four-step action utility development audit',
                'source_provenance': source,
                'test_data_used': False, 'inference_gt_fields': 0,
                'inputs': inputs, 'dataset_paths': {k: str(Path(v).resolve()) for k, v in paths.items()},
                'config': spec, 'identity': identity, 'source_hashes': source_hashes,
                'frozen_sources': frozen, 'b0_saved_results': saved,
                'train_indices': b0['train_indices'], 'validation_indices': b0['validation_indices'],
                'train_scenes': b0['train_scenes'], 'validation_scenes': b0['validation_scenes'],
                'probe_fit_scenes': fit, 'probe_calibration_scenes': calibration,
                'weather_protocol': 'same B0 online weather, seed and selected loader; not fixed-weather benchmark',
                'checkpoint_selection': 'fixed last probe epoch; no validation selection',
                'counterfactual': 'each candidate/task/source from identical original Shared output',
                'association_duplicates': 'one label per associated GT; S3 deduplicates proposal positions',
                'conflict_policy': 'margin desc, Shared score desc, anchor asc; first owner per output cell',
                'reference_metadata': {
                    'AI_CONTEXT_sections': [25, 26, 28, 29, 33, 36, 37],
                    'target_task_oracle_adverse_mean_gain_pp': 7.7197,
                    'proposal_oracle_task_minus_same_adverse_mean_pp': 1.379,
                    'boundary': 'historical references use greedy GT hindsight and different ROI choices; no direct new ceiling ratio'}}
    path = manifest.run / 'protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Existing protocol differs')
    atomic_json(path, protocol)
    for task in ('cls', 'reg'):
        feature_leakage_check(schema(task))
    # Reuse all checkpoint/source/freeze checks; build loaders but do not infer yet.
    runtime = Runtime(manifest.run)
    for split in ('train', 'validation'):
        dataset, loader = runtime.loader(split, 'clean')
        del dataset, loader
    del runtime
    import torch
    torch.cuda.empty_cache()


def stages(run):
    base = [sys.executable, '-u', '-m']
    module = 'local_fusion_action_utility_audit.'
    suffix = ['--run', str(run)]
    return [
        ('test_core', base + [module + 'test_core', '--require-server']),
        ('baseline', base + [module + 's0_counterfactual'] + suffix + ['--mode', 'baseline']),
        ('S0-build', base + [module + 's0_counterfactual'] + suffix + ['--mode', 'build']),
        ('S0', base + [module + 's0_counterfactual'] + suffix + ['--mode', 'summarize']),
        ('S1-fit', base + [module + 's1_cls_rank'] + suffix + ['--mode', 'fit']),
        ('S1', base + [module + 's1_cls_rank'] + suffix + ['--mode', 'evaluate']),
        ('S2-fit', base + [module + 's2_reg_rank'] + suffix + ['--mode', 'fit']),
        ('S2', base + [module + 's2_reg_rank'] + suffix + ['--mode', 'evaluate']),
        ('S3-assisted', base + [module + 's3_replay'] + suffix + ['--mode', 'assisted']),
        ('S3-proposal', base + [module + 's3_replay'] + suffix + ['--mode', 'proposal']),
        ('S3', base + [module + 's3_replay'] + suffix + ['--mode', 'summarize']),
        ('FINAL', base + [module + 'summarize'] + suffix),
    ]


def stage_artifacts(run, stage):
    if stage in ('S0', 'S1', 'S2', 'S3'):
        paths = [run / (stage + '_RESULTS.' + ext) for ext in ('json', 'md')]
        return paths + ([run / 'feature_schema.json'] if stage == 'S0' else [])
    if stage == 'FINAL':
        return [run / 'final_results.json', run / 'FINAL_RESULTS.md']
    if stage == 'baseline':
        return [run / 'baseline_reproduction.json']
    if stage in ('S1-fit', 'S2-fit'):
        task = 'cls' if stage == 'S1-fit' else 'reg'
        paths = [run / f'linear_{task}.pt', run / f'linear_{task}_without_competition.pt']
        return paths + ([run / 'feature_normalization.json'] if stage == 'S2-fit' else [])
    return []


def integrity(run):
    verify_protocol_snapshot(run)
    required = ['protocol.json', 'manifest.json', 'driver.log', 'linear_cls.pt', 'linear_reg.pt',
                'linear_cls_without_competition.pt', 'linear_reg_without_competition.pt',
                'feature_normalization.json', 'feature_schema.json', 'final_results.json', 'FINAL_RESULTS.md']
    required += [f'{stage}_RESULTS.{ext}' for stage in ('S0', 'S1', 'S2', 'S3') for ext in ('json', 'md')]
    for name in required:
        path = run / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError('Missing final artifact: ' + name)
    manifest = Manifest(run)
    for stage, _ in stages(run):
        if not manifest.complete(stage):
            raise RuntimeError('Incomplete stage: ' + stage)
    from .common import cache_paths
    for split in ('train', 'validation'):
        for weather in WEATHERS:
            list(cache_paths(run, split, weather))
    for mode in ('assisted', 'proposal'):
        for weather in WEATHERS:
            for index in read_json(run / 'protocol.json')['validation_indices']:
                if not manifest.complete('S3-' + mode + '-frame', 'validation', weather, index):
                    raise RuntimeError('Incomplete S3 frame')
    from gspr_communication.runtime import verify_frozen
    verify_frozen()


def main():
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', required=True)
    cli.add_argument('--config', default='local_fusion_action_utility_audit/experiment.yaml')
    cli.add_argument('--v3-config', default='local_fusion_v3/experiment.yaml')
    for name in ('frontend-config', 'frontend-checkpoint', 'v3-checkpoint', 'b0-run'):
        cli.add_argument('--' + name, required=True)
    args = cli.parse_args()
    run = validate_run(args.run)
    # Lock releases automatically after death; stale PID files never block resume.
    with (run / '.driver.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another driver is using this RUN')
        manifest = Manifest(run)
        stage = 'preflight'
        try:
            # Always verify identity again, even for a completely finished RUN.
            with manifest.work(stage):
                prepare(args, manifest)
            manifest.mark(stage, 'complete', artifacts=[run / 'protocol.json'])
            for name, _ in stages(run):
                if manifest.key(name) not in manifest.value['entries']:
                    manifest.mark(name, 'pending')
            for stage, command in stages(run):
                verify_protocol_snapshot(run)
                if manifest.complete(stage):
                    print(f'{stage} reuse complete', flush=True)
                    continue
                print(f'STAGE START {stage}', flush=True)
                with manifest.work(stage):
                    subprocess.run(command, cwd=ROOT, check=True)
                    # Children write frame and ablation states; reload before merging stage state.
                    manifest = Manifest(run)
                manifest.mark(stage, 'complete', artifacts=stage_artifacts(run, stage))
                print(f'{stage} complete', flush=True)
            stage = 'integrity'
            with manifest.work(stage):
                integrity(run)
            manifest.mark(stage, 'complete')
            for stage in ('S0', 'S1', 'S2', 'S3', 'FINAL'):
                print(stage + ' complete', flush=True)
            print('FINAL_RESULTS=' + str(run / 'FINAL_RESULTS.md'), flush=True)
        except BaseException as exc:
            manifest = Manifest(run)
            manifest.mark(stage, 'failed', error=repr(exc))
            print(f'STAGE FAILED {stage}: {exc}', file=sys.stderr, flush=True)
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
