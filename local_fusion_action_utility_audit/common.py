"""Small IO/protocol helpers; heavyweight server imports stay lazy."""
import argparse
from bisect import bisect_right
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
WEATHERS = ('clean', 'fog', 'rain', 'snow')
KEEP = 'KEEP_SHARED'
VARIANTS = ('without_competition_features', 'with_competition_features')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    from gspr_communication.runtime import sha256
    return sha256(path)


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2,
                                allow_nan=False) + '\n')


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def load_cache(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def assert_development_paths(*paths):
    """Resolve links before any loader or root directory enumeration."""
    for value in paths:
        path = Path(value).resolve()
        if 'test' in (part.lower() for part in path.parts):
            raise ValueError('Formal test access prohibited: ' + str(path))
        if path.name not in ('train', 'validate', 'validation'):
            raise ValueError('Expected an explicit development split: ' + str(path))


def validate_run(path):
    run = Path(path).resolve()
    allowed = Path('/data/cjm/datasets/logs').resolve()
    if allowed not in run.parents:
        raise ValueError('RUN must be below /data/cjm/datasets/logs')
    run.mkdir(parents=True, exist_ok=True)
    return run


def settings(path):
    import yaml
    spec = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('seed', 'baseline_seed', 'shared_top_k', 'probe_epochs',
                'pair_batch_size', 'local_top_k', 'calibration_min_actions'):
        if type(spec.get(key)) is not int or spec[key] < 1:
            raise ValueError('Invalid positive integer: ' + key)
    if type(spec.get('background_candidates_per_frame')) is not int or spec['background_candidates_per_frame'] < 0:
        raise ValueError('Invalid background budget')
    for key in ('minimum_gt_proposal_iou', 'target_iou', 'fp_identity_iou',
                'calibration_min_modify_precision', 's1_pairwise_accuracy_gate',
                's2_pairwise_accuracy_gate', 's1_regret_reduction_gate',
                's2_regret_reduction_gate', 'task_source_disagreement_gate'):
        if not 0 <= spec[key] <= 1:
            raise ValueError('Invalid fraction: ' + key)
    if not 0 < spec['calibration_scene_fraction'] < 1:
        raise ValueError('Need separate fit/calibration scenes')
    for key in ('roi_expansion', 'probe_learning_rate', 'nearby_distance_m'):
        if spec[key] <= 0:
            raise ValueError('Invalid ' + key)
    if spec['roi_expansion'] < 1 or spec['candidate_families'] != ['single', 'query']:
        raise ValueError('Fixed proposal expansion/source families required')
    if spec['primary_features'] not in VARIANTS:
        raise ValueError('Primary feature variant must be preregistered')
    if spec['regret_definition'] != 'normalized_distinct_outcome_rank':
        raise ValueError('Unknown regret definition')
    if not spec['calibration_quantiles'] or any(not 0 <= q <= 1 for q in spec['calibration_quantiles']):
        raise ValueError('Invalid margin quantiles')
    for key in ('probe_weight_decay', 'calibration_max_lost_per_action',
                'calibration_max_new_fp_per_action', 's3_weather_mean_ap70_gain_pp',
                's3_clean_max_drop_pp', 'assisted_meaningful_gain_pp'):
        if spec[key] < 0:
            raise ValueError('Invalid nonnegative constraint: ' + key)
    if type(spec['s3_min_positive_adverse_weathers']) is not int or not 1 <= spec['s3_min_positive_adverse_weathers'] <= 3:
        raise ValueError('Invalid weather count')
    return spec


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument('--run', required=True)
    return result


def audit_source_files():
    return sorted(p for p in (ROOT / 'local_fusion_action_utility_audit').rglob('*')
                  if p.is_file() and p.suffix in ('.py', '.yaml', '.sh', '.md'))


def audit_hashes():
    directory = ROOT / 'local_fusion_action_utility_audit'
    return {p.relative_to(directory).as_posix(): digest(p) for p in audit_source_files()}


def source_identity():
    """Version checks use source hashes only, regardless of .git availability."""
    actual = audit_hashes()
    path = ROOT / 'local_fusion_action_utility_audit' / 'source_snapshot.json'
    if not actual:
        raise ValueError('No audit source files found')
    if path.is_file() and actual != read_json(path).get('audit_source_hashes'):
        raise ValueError('Server audit files differ from source_snapshot.json; '
                         'sync the entire updated audit directory together')
    return {'source_commit': None, 'source_hash': object_hash(actual), 'provenance': 'source_sha256'}


def write_source_snapshot():
    """Export current code hashes; no Git, model or research data is needed."""
    path = ROOT / 'local_fusion_action_utility_audit' / 'source_snapshot.json'
    atomic_json(path, {'schema': 2, 'version_check': 'source_sha256_only',
                       'audit_source_hashes': audit_hashes()})
    return path


def verify_protocol_snapshot(run):
    """Reject source/config drift between stages, including direct module runs."""
    protocol = read_json(Path(run) / 'protocol.json')
    inputs, identity = protocol['inputs'], protocol['identity']
    if Manifest(run).value['identity'] != identity:
        raise ValueError('Manifest/protocol identity differs')
    current_sources = {name: digest(ROOT / name) for name in protocol['source_hashes']}
    if current_sources != protocol['source_hashes'] or object_hash(current_sources) != identity['source_hash']:
        changed = [name for name, value in current_sources.items() if value != protocol['source_hashes'][name]]
        raise ValueError('Source changed during this RUN: ' + ', '.join(changed))
    current_config = object_hash({
        'spec': settings(inputs['config']), 'inputs': inputs,
        'v3_config_hash': digest(inputs['v3_config']),
        'frontend_config_hash': digest(inputs['frontend_config']),
        'b0_protocol_hash': digest(Path(inputs['b0_run']) / 'protocol.json'),
        'b0_result_hash': digest(Path(inputs['b0_run']) / 'decision_results.json'),
    })
    source = source_identity()
    if (current_config != identity['config_hash']
            or ('source_provenance' in protocol and source != protocol['source_provenance'])):
        raise ValueError('Config/source changed during this RUN; use a new RUN')
    return protocol


class Manifest:
    """Only complete, hash-verified artifacts can be reused after interruption."""
    def __init__(self, run, identity=None):
        self.run = Path(run)
        self.path = self.run / 'manifest.json'
        if self.path.exists():
            self.value = read_json(self.path)
            if identity is not None and self.value['identity'] != identity:
                raise ValueError('Resume source/config/checkpoint identity changed')
        else:
            self.value = {'schema': 1, 'identity': identity or {}, 'entries': {}}
            self.save()

    def save(self):
        atomic_json(self.path, self.value)

    def key(self, stage, split='-', weather='-', frame='-'):
        return '/'.join(map(str, (stage, split, weather, frame)))

    def complete(self, stage, split='-', weather='-', frame='-'):
        row = self.value['entries'].get(self.key(stage, split, weather, frame), {})
        if row.get('status') != 'complete':
            return False
        for name, expected in row.get('artifacts', {}).items():
            path = self.run / name
            if not path.is_file() or digest(path) != expected:
                raise ValueError('Complete artifact missing/corrupted: ' + str(path))
        return True

    def mark(self, stage, status, split='-', weather='-', frame='-', artifacts=(), error=None):
        if status not in ('pending', 'in_progress', 'complete', 'failed'):
            raise ValueError('Unknown manifest status')
        # Stage children update this same manifest. Never overwrite their entries
        # with the parent's stale in-memory copy, especially on a failed child.
        if self.path.exists():
            self.value = read_json(self.path)
        self.value['entries'][self.key(stage, split, weather, frame)] = {
            'stage': stage, 'split': split, 'weather': weather, 'frame': frame,
            'status': status, 'updated_utc_epoch': time.time(),
            **self.value['identity'],
            'artifacts': {str(Path(p).relative_to(self.run)): digest(p) for p in artifacts},
            'error': error,
        }
        self.save()

    @contextmanager
    def work(self, stage, split='-', weather='-', frame='-'):
        self.mark(stage, 'in_progress', split, weather, frame)
        try:
            yield
        except BaseException as exc:
            self.mark(stage, 'failed', split, weather, frame, error=repr(exc))
            raise


def scene_split(train_scenes, fraction, seed):
    scenes = sorted(set(int(x) for x in train_scenes))
    if len(scenes) < 2:
        raise ValueError('At least two train scenes required')
    shuffled = list(scenes)
    random.Random(seed).shuffle(shuffled)
    count = min(len(scenes) - 1, max(1, round(len(scenes) * fraction)))
    return sorted(shuffled[count:]), sorted(shuffled[:count])


def frame_scene(dataset, index):
    return bisect_right(dataset.len_record, int(index))


def assert_loader_paths(dataset):
    # Check root aliases AND nested file links, before __getitem__ can load them.
    for scene in dataset.scenario_database.values():
        for cav in scene.values():
            for frame in cav.values():
                if isinstance(frame, dict):
                    for key in ('yaml', 'lidar'):
                        if key in frame and 'test' in (x.lower() for x in Path(frame[key]).resolve().parts):
                            raise ValueError('Dataset file resolves to formal test: ' + str(frame[key]))


class Runtime:
    """Reuse B0 model, loader, source contracts and frozen checks verbatim."""
    def __init__(self, run):
        from gspr_communication.runtime import device, verify_frozen
        from gspr_evidence import runtime as er
        from local_fusion_v3 import runtime as v3rt
        from local_fusion_task_source_oracle import oracle as oracle
        self.run = Path(run)
        self.protocol = verify_protocol_snapshot(self.run)
        self.spec = self.protocol['config']
        self.manifest = Manifest(self.run)
        verify_frozen()
        inputs = self.protocol['inputs']
        if digest(Path(inputs['b0_run']) / 'Shared.pth') != self.protocol['identity']['checkpoint_hashes']['b0_shared']:
            raise ValueError('B0 Shared checkpoint changed during this RUN')
        self.b0 = read_json(Path(inputs['b0_run']) / 'protocol.json')
        oracle._verify_b0_source_contract(self.b0)
        self.options, self.hypes = er.load_config(inputs['v3_config'], inputs['frontend_config'])
        assert_development_paths(self.hypes['root_dir'], self.hypes['validate_dir'])
        self.target = device()
        self.model, front_hash = er.load_model(self.hypes, self.options, inputs['frontend_checkpoint'], self.target)
        if front_hash != self.protocol['identity']['checkpoint_hashes']['frontend_checkpoint']:
            raise ValueError('Frontend checkpoint changed during this RUN')
        self.model.requires_grad_(False).eval()
        contract = v3rt.contract(self.options, inputs['frontend_config'], front_hash)
        if self.b0['frontend_sha256'] != front_hash or self.b0['v3_contract'] != contract:
            raise ValueError('B0 frontend/config/source contract differs')
        v3_hash = digest(inputs['v3_checkpoint'])
        if v3_hash != self.protocol['identity']['checkpoint_hashes']['v3_checkpoint']:
            raise ValueError('v3 checkpoint changed during this RUN')
        if self.b0['v3_checkpoint_sha256'] != v3_hash:
            raise ValueError('B0 v3 checkpoint differs')
        source, state = v3rt.load(inputs['v3_checkpoint'], contract, self.target)
        if state['variant'] != 'residual':
            raise ValueError('Expected residual v3 checkpoint')
        self.shared_arm = oracle._load_b0_arm(Path(inputs['b0_run']), 'Shared.pth', source,
                                              self.target, 'shared', v3_hash)
        self.lidar_range = self.hypes['postprocess']['anchor_args']['cav_lidar_range']
        if any(p.requires_grad for module in (self.model, self.shared_arm) for p in module.parameters()):
            raise RuntimeError('Frozen model contains trainable parameters')

    def loader(self, split, weather):
        from gspr_communication.runtime import seed_all
        from local_fusion_task_split_pilot.pipeline import selected_loader
        # Full loader iteration is preserved during resume, including skipped frames.
        seed_all(self.spec['baseline_seed'])
        indices = self.protocol[split + '_indices']
        dataset, loader = selected_loader(self.hypes, self.options, split, weather, indices)
        assert_loader_paths(dataset)
        actual = sorted({frame_scene(dataset, i) for i in indices})
        if actual != sorted(self.protocol[split + '_scenes']):
            raise ValueError('B0 scene mapping drift: ' + split)
        return dataset, loader

    def predict(self, batch, weather, verify=False, sources=True):
        from local_fusion_v3 import runtime as v3rt
        from local_fusion_task_source_oracle.oracle import build_candidate_pool
        context = v3rt.context(self.model, batch['ego'],
                               'clean' if weather == 'clean' else 'weather', verify=verify)
        shared, _ = self.shared_arm.predict(self.model.engine.base, context['levels'])
        pool = build_candidate_pool(self.model.engine.base, context['levels'], shared,
                                    self.spec['candidate_families']) if sources else None
        return context, shared, pool


def cache_paths(run, split, weather):
    protocol = read_json(Path(run) / 'protocol.json')
    manifest = Manifest(run)
    for index in protocol[split + '_indices']:
        if not manifest.complete('S0-frame', split, weather, index):
            raise RuntimeError(f'Incomplete S0 cache: {split}/{weather}/{index}')
        yield Path(run) / 'cache' / split / weather / f'{index:08d}.pt'


def write_report(run, stage, report, markdown):
    atomic_json(Path(run) / (stage + '_RESULTS.json'), report)
    atomic_text(Path(run) / (stage + '_RESULTS.md'), markdown)


def shell_check():
    for name in ('run_all.sh', 'launch.sh'):
        path = Path(__file__).parent / name
        if b'\r' in path.read_bytes():
            raise ValueError('Shell must use LF: ' + name)
        subprocess.run(['sh', '-n', str(path)], check=True)
