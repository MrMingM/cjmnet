"""Hash-checked immutable contracts; no runtime Git dependency."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from local_fusion_action_utility_audit.common import (
    ROOT, KEEP, WEATHERS, Manifest, atomic_json, atomic_text, atomic_torch,
    load_cache, object_hash, read_json)

PACKAGE = Path(__file__).resolve().parent
DEFAULT_BASE = '/data/cjm/datasets/logs/action_utility_audit_20261007_210243'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe_path(path, dataset=False):
    path = Path(path).resolve()
    if any('test' == part.lower() for part in path.parts):
        raise ValueError('Formal test access prohibited: ' + str(path))
    if dataset and path.name not in ('train', 'validate', 'validation'):
        raise ValueError('An explicit development split is required: ' + str(path))
    return path


def external_run(path, create=False):
    path = safe_path(path)
    allowed = Path('/data/cjm/datasets/logs').resolve()
    if allowed not in path.parents:
        raise ValueError('RUN and BASE_RUN must be below /data/cjm/datasets/logs')
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def guard_dataset_tree(root):
    """Resolve directory and file links before a loader can follow them."""
    root = safe_path(root, dataset=True)
    pending, seen = [root], set()
    while pending:
        parent = safe_path(pending.pop())
        if parent in seen:
            continue
        seen.add(parent)
        with os.scandir(parent) as entries:
            for entry in entries:
                # Regular files inherit checked parents. A symlink can escape
                # into test and must be resolved even when it is not a folder.
                if entry.is_symlink() or entry.name.lower() == 'test':
                    safe_path(entry.path)
                if entry.is_dir(follow_symlinks=True):
                    pending.append(safe_path(entry.path))


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(PACKAGE.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.yaml', '.sh', '.md')}


def export_snapshot():
    atomic_json(PACKAGE / 'source_snapshot.json', {'schema': 1, 'sources': source_hashes()})


def settings():
    import yaml
    return yaml.safe_load((PACKAGE / 'experiment.yaml').read_text(encoding='utf-8'))


def verify_base(base):
    from local_fusion_action_utility_audit.common import verify_protocol_snapshot, cache_paths
    from local_fusion_action_utility_audit.linear_ranker import load_probe
    base = external_run(base)
    recorded = read_json(safe_path(base/'protocol.json'))
    for value in recorded['inputs'].values():
        safe_path(value)
    for value in recorded['dataset_paths'].values():
        safe_path(value, dataset=True)
    protocol = verify_protocol_snapshot(base)
    manifest = Manifest(base)
    for stage in ('baseline', 'S0-build', 'S0', 'S1-fit', 'S1', 'S2-fit', 'S2', 'S3-assisted', 'S3-proposal', 'S3', 'FINAL', 'validation-reference', 'integrity'):
        if not manifest.complete(stage):
            raise ValueError('BASE_RUN is incomplete: ' + stage)
    # Verify every completed artifact, including epoch/models, labels and exact pools.
    for row in manifest.value['entries'].values():
        if row['status'] != 'complete':
            continue
        for name, expected in row.get('artifacts', {}).items():
            path = safe_path(base / name)
            if base not in path.parents or digest(path) != expected:
                raise ValueError('BASE_RUN artifact differs: ' + name)
    spec = protocol['config']
    fixed = {'shared_top_k': 256, 'roi_expansion': 1.5,
             'candidate_families': ['single', 'query'], 'probe_epochs': 30,
             'probe_learning_rate': .01, 'probe_weight_decay': .0001,
             'primary_features': 'with_competition_features',
             'regret_definition': 'normalized_distinct_outcome_rank',
             's1_pairwise_accuracy_gate': .60, 's2_pairwise_accuracy_gate': .60,
             's1_regret_reduction_gate': .20, 's2_regret_reduction_gate': .20,
             's3_weather_mean_ap70_gain_pp': .30, 's3_min_positive_adverse_weathers': 2,
             's3_clean_max_drop_pp': .20, 'calibration_min_modify_precision': .80,
             'calibration_max_lost_per_action': .01, 'calibration_max_new_fp_per_action': .05,
             'calibration_min_actions': 20}
    for name, expected in fixed.items():
        if spec.get(name) != expected:
            raise ValueError('Unsupported S0/S1/S2 protocol: ' + name)
    for key in ('root_dir', 'validate_dir'):
        # The recorded roots are checked again before loader construction.
        if key in protocol.get('dataset_paths', {}):
            safe_path(protocol['dataset_paths'][key], dataset=True)
    fit, cal = map(set, (protocol['probe_fit_scenes'], protocol['probe_calibration_scenes']))
    if fit & cal or fit | cal != set(protocol['train_scenes']):
        raise ValueError('Fit/calibration scenes overlap or are incomplete')
    if len(protocol['train_indices']) != len(set(protocol['train_indices'])) or len(protocol['validation_indices']) != len(set(protocol['validation_indices'])):
        raise ValueError('Duplicate frame indices')
    checkpoints = protocol['identity']['checkpoint_hashes']
    paths = {k: Path(protocol['inputs'][k]) for k in ('frontend_checkpoint', 'v3_checkpoint')}
    paths['b0_shared'] = Path(protocol['inputs']['b0_run']) / 'Shared.pth'
    for key, path in paths.items():
        if digest(safe_path(path)) != checkpoints[key]:
            raise ValueError('Frozen checkpoint changed: ' + key)
    files = {}
    for split in ('train', 'validation'):
        for weather in WEATHERS:
            for path in cache_paths(base, split, weather):
                files[path.relative_to(base).as_posix()] = digest(path)
    for weather in WEATHERS:
        for frame in protocol['validation_indices']:
            for stage in ('reference-frame', 'S3-assisted-frame', 'S3-proposal-frame'):
                if not manifest.complete(stage, 'validation', weather, frame):
                    raise ValueError(f'BASE_RUN is missing {stage}: {weather}/{frame}')
    for task in ('cls', 'reg'):
        for variant in ('with_competition_features', 'without_competition_features'):
            load_probe(base, task, variant)
    core = ['protocol.json', 'manifest.json', 'feature_schema.json', 'feature_normalization.json',
            'S0_RESULTS.json', 'S1_RESULTS.json', 'S2_RESULTS.json', 'S3_RESULTS.json',
            'final_results.json', 'FINAL_RESULTS.md']
    files.update({name: digest(base / name) for name in core})
    return protocol, {'base_run': str(base), 'files': files, 'files_hash': object_hash(files),
                      'base_identity': protocol['identity'], 'checkpoint_hashes': checkpoints}


def contract(run):
    run = external_run(run)
    protocol = read_json(run / 'protocol.json')
    if protocol['sources'] != source_hashes():
        raise ValueError('S4 source changed: use a new RUN and record the repair; no silent migration')
    for name, expected in protocol['dependency_sources'].items():
        if digest(safe_path(ROOT/name)) != expected:
            raise ValueError('Frozen/old dependency source changed during S4: '+name)
    if read_json(PACKAGE / 'source_snapshot.json')['sources'] != protocol['sources']:
        raise ValueError('Sync the entire S4 directory including source_snapshot.json')
    for name in ('feature_schema.json', 'coordinate_contract.json'):
        if digest(run / name) != protocol['identity'][name.replace('.json', '_hash')]:
            raise ValueError('Schema/coordinate contract changed: ' + name)
    if protocol['s4_config'] != settings():
        raise ValueError('S4 config changed')
    base = external_run(protocol['base']['base_run'])
    for name in ('protocol.json', 'manifest.json'):
        if digest(base / name) != protocol['base']['files'][name]:
            raise ValueError('BASE_RUN identity changed during S4: ' + name)
    if Manifest(run).value['identity'] != protocol['identity']:
        raise ValueError('S4 manifest/protocol mismatch')
    for name, value in protocol['dataset_roots'].items():
        safe_path(value, dataset=True)
    return protocol


def base_frame(run, split, weather, index):
    protocol = read_json(Path(run) / 'protocol.json')
    base = Path(protocol['base']['base_run'])
    name = f'cache/{split}/{weather}/{int(index):08d}.pt'
    path = safe_path(base / name)
    if digest(path) != protocol['base']['files'][name]:
        raise ValueError('Original S0 label/cache changed: ' + name)
    return load_cache(path)


def evidence_path(run, split, weather, frame):
    return Path(run) / 'cache' / 'evidence' / split / weather / f'{int(frame):08d}.pt'


def report(run, name, value, markdown):
    atomic_json(Path(run) / (name + '.json'), value)
    atomic_text(Path(run) / (name + '.md'), markdown)


def parser(doc):
    cli = argparse.ArgumentParser(description=doc)
    cli.add_argument('--run', required=True)
    return cli


if __name__ == '__main__':
    export_snapshot()
