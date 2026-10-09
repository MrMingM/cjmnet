"""Read SOURCE_RUN without ever constructing its writable Manifest/Runtime."""
from pathlib import Path
import math
from .common import (ROOT, KEEP, WEATHERS, FEATURES, digest, load_cache, object_hash,
                     read_json, safe_file)
from local_fusion_action_utility_audit.common import assert_development_paths, scene_split
from local_fusion_action_utility_audit.features import schema, source_names, feature_leakage_check
from local_fusion_action_utility_audit.counterfactual import action_keys


class Source:
    def __init__(self, root, cache_index=None):
        self.root = Path(root).resolve()
        self.cache_index = cache_index
        self._reuse_entries = None
        self.protocol = read_json(self.root / 'protocol.json')
        self.manifest = read_json(self.root / 'manifest.json')
        if self.manifest.get('identity') != self.protocol['identity']:
            raise ValueError('SOURCE_RUN manifest/protocol identity mismatch')
        self.spec = self.protocol['config']

    def refresh(self):
        self.manifest = read_json(self.root / 'manifest.json')
        self._reuse_entries = None
        if self.manifest['identity'] != self.protocol['identity']:
            raise ValueError('Source identity changed')

    def entry(self, stage, split='-', weather='-', frame='-'):
        return self.manifest['entries'].get('/'.join(map(str, (stage, split, weather, frame))), {})

    def complete(self, stage, split='-', weather='-', frame='-'):
        row = self.entry(stage, split, weather, frame)
        if row.get('status') != 'complete':
            return False
        if any(row.get(k) != v for k, v in self.protocol['identity'].items()):
            key = '/'.join(map(str, (stage, split, weather, frame)))
            if self.verified_reuse_entries().get(key) != row:
                changed = [k for k, v in self.protocol['identity'].items() if row.get(k) != v]
                raise ValueError('Source entry identity mismatch: ' + key + '; fields=' + ','.join(changed))
        for name, expected in row.get('artifacts', {}).items():
            path = safe_file(self.root, name)
            if not path.is_file() or digest(path) != expected:
                raise ValueError('Source artifact missing/corrupted: ' + str(path))
        return True

    def verified_reuse_entries(self):
        """Accept only immutable producers retained by the completed source repair.

        Never run repair helpers here: they write SOURCE_RUN. Check their saved
        transaction, old manifest and exact approved entries using read-only IO.
        """
        if self._reuse_entries is not None:
            return self._reuse_entries
        repairs = self.protocol.get('repairs', [])
        if not repairs:
            return {}
        repair_id = 'shared_snapshot_v2'
        if (len(repairs) != 1 or repairs[0].get('repair_id') != repair_id
                or self.manifest.get('repairs') != repairs):
            raise ValueError('Unsupported or inconsistent source repair provenance')
        record = repairs[0]
        folder = self.root / 'repair_shared_drift'
        journal = read_json(safe_file(folder, 'transaction.json'))
        if journal.get('repair_id') != repair_id or journal.get('status') != 'complete':
            raise ValueError('Source repair transaction must be complete')
        for name in ('proposed_protocol.json', 'proposed_manifest.json'):
            if digest(safe_file(folder, name)) != journal.get('proposed_hashes', {}).get(name):
                raise ValueError('Source repair proposal hash differs: ' + name)
        proposed = read_json(safe_file(folder, 'proposed_manifest.json'))
        if (read_json(safe_file(folder, 'proposed_protocol.json')) != self.protocol
                or proposed['identity'] != self.protocol['identity']
                or proposed.get('repairs') != repairs):
            raise ValueError('Source repair proposal/protocol identity differs')
        old = read_json(safe_file(folder, 'original_protocol.json'))
        original = read_json(safe_file(folder, 'original_manifest.json'))
        if (old['identity'] != record.get('original_identity')
                or original['identity'] != old['identity']
                or object_hash(old['source_hashes']) != old['identity']['source_hash']
                or record.get('configuration_and_checkpoint_changes') is not False
                or record.get('retained_train_frames') != len(self.protocol['train_indices']) * len(WEATHERS)):
            raise ValueError('Invalid source repair original identity/retention record')
        for name in ('inputs', 'config', 'dataset_paths', 'train_indices', 'validation_indices',
                     'train_scenes', 'validation_scenes', 'probe_fit_scenes',
                     'probe_calibration_scenes', 'b0_saved_results', 'frozen_sources'):
            if old.get(name) != self.protocol.get(name):
                raise ValueError('Source repair changed scientific protocol: ' + name)
        for name in ('config_hash', 'checkpoint_hashes'):
            if old['identity'][name] != self.protocol['identity'][name]:
                raise ValueError('Source repair changed configuration/checkpoints')
        compatibility = read_json(ROOT / 'local_fusion_action_utility_audit' / 'repair_compatibility.json')
        legacy = compatibility['legacy_audit_sources']
        if compatibility['repair_id'] != repair_id or not set(legacy).issubset(old['source_hashes']):
            raise ValueError('Unrecognized original source version in repair')
        for name, expected in old['source_hashes'].items():
            allowed = legacy.get(name)
            if name == 'local_fusion_action_utility_audit/source_snapshot.json':
                allowed = compatibility['legacy_snapshot_sha256']
            if expected != (allowed if allowed is not None else self.protocol['source_hashes'].get(name)):
                raise ValueError('Source repair dependency differs: ' + name)
        if digest(safe_file(self.root, record['evidence'])) != record['evidence_sha256']:
            raise ValueError('Source repair evidence hash differs')
        approved = {}
        for key, row in original['entries'].items():
            stage, split = row['stage'], row['split']
            retained = (stage == 'baseline' or
                (stage in ('baseline-frame', 'baseline-weather') and split == 'validation') or
                (stage in ('S0-frame', 'S0-weather') and split == 'train'))
            if not retained or row.get('status') != 'complete':
                continue
            if any(row.get(k) != v for k, v in old['identity'].items()):
                raise ValueError('Retained source producer differs: ' + key)
            expected = {**row, 'reuse_approval': repair_id}
            if proposed['entries'].get(key) != expected:
                raise ValueError('Retained source entry differs from repair proposal: ' + key)
            approved[key] = expected
        self._reuse_entries = approved
        return approved

    def repair_provenance_files(self):
        if not self.protocol.get('repairs'):
            return []
        self.verified_reuse_entries()
        files = ['repair_shared_drift/' + name for name in ('transaction.json',
            'proposed_protocol.json', 'proposed_manifest.json', 'original_protocol.json', 'original_manifest.json')]
        return files + [self.protocol['repairs'][0]['evidence']]

    def artifact(self, name, stage, split='-', weather='-', frame='-'):
        if not self.complete(stage, split, weather, frame):
            raise RuntimeError('Required source stage incomplete: ' + stage)
        path = safe_file(self.root, name)
        artifacts = self.entry(stage, split, weather, frame).get('artifacts', {})
        if name not in artifacts:
            raise ValueError('Source file lacks completed manifest hash: ' + name)
        if digest(path) != artifacts[name]:
            raise ValueError('Source file hash mismatch: ' + name)
        if stage == 'S0-frame' and self.cache_index is not None:
            if self.cache_index.get(name) != artifacts[name]:
                raise ValueError('Source cache changed after control preflight: ' + name)
        return path

    def readiness(self, baseline=False):
        stages = ['S0', 'S1', 'S2'] + (['baseline'] if baseline else [])
        files = ['S0_RESULTS.json', 'S1_RESULTS.json', 'S2_RESULTS.json',
                 'feature_schema.json', 'feature_normalization.json', 'linear_cls.pt', 'linear_reg.pt']
        if baseline:
            files.append('baseline_reproduction.json')
        missing = ['stage ' + s for s in stages if not self.complete(s)]
        missing += ['file ' + s for s in files if not (self.root / s).is_file()]
        return missing

    def verify_dependencies(self):
        p, identity = self.protocol, self.protocol['identity']
        if not p.get('source_hashes'):
            raise ValueError('SOURCE_RUN lacks reliable dependency hashes; restore matching source')
        actual = {name: digest(safe_file(ROOT, name)) for name in p['source_hashes']}
        if actual != p['source_hashes'] or object_hash(actual) != identity['source_hash']:
            changed = [name for name, value in actual.items() if value != p['source_hashes'][name]]
            raise ValueError('Original source differs; restore SOURCE_RUN-matching files: ' + ', '.join(changed))
        inputs = p['inputs']
        from local_fusion_action_utility_audit.common import settings as original_settings
        spec = original_settings(inputs['config'])
        value = object_hash({'spec': spec, 'inputs': inputs,
            'v3_config_hash': digest(inputs['v3_config']),
            'frontend_config_hash': digest(inputs['frontend_config']),
            'b0_protocol_hash': digest(Path(inputs['b0_run']) / 'protocol.json'),
            'b0_result_hash': digest(Path(inputs['b0_run']) / 'decision_results.json')})
        if spec != self.spec or value != identity['config_hash']:
            raise ValueError('Original configuration identity differs')
        expected = identity['checkpoint_hashes']
        for key, path in (('frontend_checkpoint', inputs['frontend_checkpoint']),
                          ('v3_checkpoint', inputs['v3_checkpoint']),
                          ('b0_shared', Path(inputs['b0_run']) / 'Shared.pth')):
            if digest(path) != expected[key]:
                raise ValueError('Original checkpoint differs: ' + key)
        from gspr_communication.runtime import verify_frozen
        verify_frozen()
        import yaml
        v3 = yaml.safe_load(Path(inputs['v3_config']).read_text(encoding='utf-8'))
        front = yaml.safe_load(Path(inputs['frontend_config']).read_text(encoding='utf-8'))
        roots = {key: v3.get(key) or front.get(key) for key in ('root_dir', 'validate_dir')}
        assert_development_paths(*roots.values())
        if {k: str(Path(v).resolve()) for k, v in roots.items()} != p['dataset_paths']:
            raise ValueError('Original dataset roots differ')
        verify_scenes(p)
        if spec['primary_features'] != FEATURES:
            raise ValueError('SOURCE_RUN primary probe must use with_competition_features')

    def frame(self, split, weather, index):
        name = f'cache/{split}/{weather}/{index:08d}.pt'
        row = load_cache(self.artifact(name, 'S0-frame', split, weather, index))
        validate_frame(row, self.protocol, split, weather, index)
        return row

    def frames(self, split, weather):
        for index in self.protocol[split + '_indices']:
            yield self.frame(split, weather, index)

    def groups(self, split, weather, task, scenes=None):
        for row in self.frames(split, weather):
            if scenes is not None and row['scene'] not in scenes:
                continue
            for label in row['labels']:
                yield {'x': row['features'][task][label['proposal_position']],
                       'outcomes': label['outcomes'][task], 'names': row['source_names'],
                       'frame': row['frame'], 'scene': row['scene'], 'weather': weather,
                       'proposal': label['proposal_position'], 'background': label['background']}

    def original_state(self, task):
        stage = 'S1-fit' if task == 'cls' else 'S2-fit'
        state = load_cache(self.artifact(f'linear_{task}.pt', stage))
        validate_state(state, self.protocol, task)
        normal = read_json(self.root / 'feature_normalization.json')[task + '/' + FEATURES]
        for key in ('mean', 'std', 'feature_names', 'fit_scenes'):
            if normal[key] != state[key]:
                raise ValueError('Original normalization/probe mismatch: ' + task + '/' + key)
        return state


def control_source(run, p=None):
    """Pin S0 inputs to this control's initial index, even if source metadata changes."""
    from .common import protocol
    p = protocol(run) if p is None else p
    index = read_json(Path(run) / 'source_cache_index.json')
    if object_hash(index) != p['identity']['source_cache_index_hash']:
        raise ValueError('Control source cache index hash differs')
    return Source(p['source_run'], index)


def verify_scenes(p):
    for split in ('train', 'validation'):
        indices = p[split + '_indices']
        if not indices or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
            raise ValueError('Invalid source indices: ' + split)
        if not p[split + '_scenes']:
            raise ValueError('Missing source scene mapping')
    if p['validation_indices'] != sorted(p['validation_indices']):
        raise ValueError('Source validation frame order differs')
    fit, calibration = scene_split(p['train_scenes'], p['config']['calibration_scene_fraction'], p['config']['seed'])
    if fit != p['probe_fit_scenes'] or calibration != p['probe_calibration_scenes']:
        raise ValueError('Source scene protocol mismatch')


def validate_state(state, p, task):
    if (state['identity'] != p['identity'] or state['task'] != task or state['variant'] != FEATURES
            or state['fit_scenes'] != p['probe_fit_scenes']
            or state['calibration_scenes'] != p['probe_calibration_scenes']
            or state['feature_names'] != schema(task)):
        raise ValueError('Source probe identity/schema/scenes differ: ' + task)
    feature_leakage_check(state['feature_names'])
    for key in ('mean', 'std'):
        if len(state[key]) != len(schema(task)) or not all(math.isfinite(x) for x in state[key]):
            raise ValueError('Invalid original normalization: ' + task)
    if min(state['std']) <= 0:
        raise ValueError('Nonpositive original feature std')


def validate_frame(row, p, split, weather, index):
    import numpy as np
    names = row['source_names']
    if names != source_names(names) or not names or names[0] != KEEP or len(set(names)) != len(names):
        raise ValueError('Source order invalid')
    if len(names) < 3 or set(n[6:] for n in names if n.startswith('query:')) != set(n[7:] for n in names if n.startswith('single:')):
        raise ValueError('single/query pool differs')
    if (row['frame'] != index or row['split'] != split or row['weather'] != weather
            or row['scene'] not in p[split + '_scenes']):
        raise ValueError('Cache frame/split/weather/scene mismatch')
    proposal = row['proposals']
    count = len(proposal['anchor_ids'])
    if len(set(proposal['anchor_ids'])) != count or proposal['ranks'] != list(range(1, count + 1)):
        raise ValueError('Proposal IDs/ranks differ')
    if len(proposal['scores']) != count or tuple(proposal['corners'].shape) != (count, 8, 3):
        raise ValueError('Proposal geometry shape differs')
    for task in ('cls', 'reg'):
        value = row['features'][task]
        if tuple(value.shape) != (count, len(names), len(schema(task))) or not np.isfinite(value.numpy()).all():
            raise ValueError('Cache feature shape/value differs')
    for label in row['labels']:
        if not 0 <= label['proposal_position'] < count or label['background'] != (label['target_index'] is None):
            raise ValueError('Invalid label association')
        for task in ('cls', 'reg'):
            outcomes = label['outcomes'][task]
            action_keys(outcomes, names)
            for o in outcomes:
                for key in ('baseline_tp', 'action_tp', 'recovered', 'lost', 'baseline_fp', 'action_fp', 'new_fp'):
                    if type(o[key]) is not int or o[key] < 0:
                        raise ValueError('Invalid structured count: ' + key)
                if (o['action_tp'] != o['baseline_tp'] + o['recovered'] - o['lost']
                        or o['new_fp'] > o['action_fp']):
                    raise ValueError('Inconsistent structured outcome')
                if label['background'] and any(o[k] is not None for k in o if k.startswith('focal_')):
                    raise ValueError('Background contains focal target')
            keep = outcomes[0]
            if any(keep[k] for k in ('recovered', 'lost', 'new_fp')) or keep['action_fp'] != keep['baseline_fp']:
                raise ValueError('KEEP is not the original Shared baseline')
