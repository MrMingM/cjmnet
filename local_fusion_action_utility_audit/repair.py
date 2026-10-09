"""One explicitly requested, hash-checked migration of the spectral-drift run."""
import copy
from pathlib import Path

from .common import ROOT, WEATHERS, Manifest, atomic_json, digest, object_hash, read_json

REPAIR_ID = 'shared_snapshot_v2'


def finish_repair_transaction(run):
    """Idempotently finish a prepared metadata transaction after interruption."""
    run = Path(run)
    folder = run / 'repair_shared_drift'
    journal_path = folder / 'transaction.json'
    if not journal_path.exists():
        return
    journal = read_json(journal_path)
    if journal['status'] == 'complete':
        return
    if journal.get('repair_id') != REPAIR_ID or journal['status'] != 'prepared':
        raise RuntimeError('Unknown repair transaction state')
    for name in ('proposed_protocol.json', 'proposed_manifest.json'):
        if digest(folder / name) != journal['proposed_hashes'][name]:
            raise RuntimeError('Repair proposal was changed/corrupted')
    protocol = read_json(folder / 'proposed_protocol.json')
    for name, expected in protocol['source_hashes'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Source changed during repair transaction: ' + name)
    inputs = protocol['inputs']
    checkpoints = {'frontend_checkpoint': Path(inputs['frontend_checkpoint']),
                   'v3_checkpoint': Path(inputs['v3_checkpoint']),
                   'b0_shared': Path(inputs['b0_run']) / 'Shared.pth'}
    for name, path in checkpoints.items():
        if digest(path) != protocol['identity']['checkpoint_hashes'][name]:
            raise RuntimeError('Checkpoint changed during repair: ' + name)
    atomic_json(run / 'protocol.json', protocol)
    atomic_json(run / 'manifest.json', read_json(folder / 'proposed_manifest.json'))
    journal['status'] = 'complete'
    atomic_json(journal_path, journal)


def migrate_if_authorized(run, new_protocol, enabled=False):
    run = Path(run)
    manifest = Manifest(run)
    if not manifest.value['identity'] or manifest.value['identity'] == new_protocol['identity']:
        old_path = run / 'protocol.json'
        if old_path.exists() and read_json(old_path).get('repairs'):
            new_protocol['repairs'] = read_json(old_path)['repairs']
        return new_protocol
    if not enabled:
        raise ValueError('Resume source/config/checkpoint identity changed. For the diagnosed spectral-drift '
                         'incident only, use --repair-shared-drift; other changes require a new RUN')
    old = read_json(run / 'protocol.json')
    if old['identity'] != manifest.value['identity']:
        raise ValueError('Original manifest/protocol identities differ')
    if old['identity']['source_hash'] != object_hash(old['source_hashes']):
        raise ValueError('Original source identity is inconsistent')
    if (old['inputs'] != new_protocol['inputs'] or old['config'] != new_protocol['config']
            or old['identity']['config_hash'] != new_protocol['identity']['config_hash']
            or old['identity']['checkpoint_hashes'] != new_protocol['identity']['checkpoint_hashes']):
        raise ValueError('Repair cannot change experiment configuration, input paths or checkpoints')
    compatibility = read_json(ROOT / 'local_fusion_action_utility_audit' / 'repair_compatibility.json')
    legacy = compatibility['legacy_audit_sources']
    if compatibility['repair_id'] != REPAIR_ID or not set(legacy).issubset(old['source_hashes']):
        raise ValueError('Unrecognized pre-repair audit version')
    for name, expected in old['source_hashes'].items():
        allowed = legacy.get(name)
        if name == 'local_fusion_action_utility_audit/source_snapshot.json':
            allowed = compatibility['legacy_snapshot_sha256']
        if allowed is not None:
            if expected != allowed:
                raise ValueError('Unrecognized original audit file: ' + name)
        elif digest(ROOT / name) != expected:
            raise ValueError('Repair cannot change an existing dependency: ' + name)
    for name in ('train_indices', 'validation_indices', 'train_scenes', 'validation_scenes',
                 'probe_fit_scenes', 'probe_calibration_scenes', 'b0_saved_results', 'frozen_sources'):
        if old[name] != new_protocol[name]:
            raise ValueError('Repair changed recorded scientific protocol: ' + name)
    if not manifest.complete('baseline'):
        raise ValueError('Repair requires a completed original B0 baseline')
    print('REPAIR CHECK: verifying original version and completed artifact hashes', flush=True)
    for row in manifest.value['entries'].values():
        if row['status'] != 'complete':
            continue
        manifest.complete(row['stage'], row['split'], row['weather'], row['frame'])
        if ((row['stage'] == 'S0-frame' and row['split'] == 'validation')
                or row['stage'] in ('S0', 'S1-fit', 'S1', 'S2-fit', 'S2', 'S3', 'FINAL')
                or row['stage'].startswith(('cls-fit', 'reg-fit', 'S3-', 'reference-'))):
            raise ValueError('This repair only supports the failure before validation labels/probes')
    for weather in WEATHERS:
        if not manifest.complete('S0-weather', 'train', weather):
            raise ValueError('All four training weather caches must be complete')
        for index in old['train_indices']:
            if not manifest.complete('S0-frame', 'train', weather, index):
                raise ValueError('Missing or corrupted completed training frame')
        for index in old['validation_indices']:
            if not manifest.complete('baseline-frame', 'validation', weather, index):
                raise ValueError('Missing or corrupted original baseline frame')
    reports = sorted((run / 'diagnostics').glob('shared_drift_*.json'), reverse=True)
    if not reports:
        raise ValueError('The completed server drift diagnostic report is required')
    evidence_path = reports[0]
    evidence = read_json(evidence_path)
    flags = ('same_input_in_all_cases', 'raw_hash_varies_between_repeats',
             'repeat_raw_differences_within_existing_2e5_tolerance',
             'saved_baseline_tp_fp_sequences_unchanged', 'all_inputs_and_buffers_unchanged')
    if (not all(evidence.get('summary', {}).get(name) is True for name in flags)
            or evidence['summary'].get('first_variable_intermediate_on_same_batch') != 'spectral'
            or evidence.get('frame') not in old['validation_indices']):
        raise ValueError('Diagnostic evidence does not match the approved spectral-drift repair')
    for case in evidence['cases']:
        for item in case['stats_vs_saved_baseline'].values():
            if not item['scores'].get('allclose_2e5'):
                raise ValueError('Diagnostic baseline score differences exceed the existing tolerance')
    record = {'repair_id': REPAIR_ID, 'original_identity': old['identity'],
              'retained_train_frames': 4 * len(old['train_indices']),
              'evidence': str(evidence_path.relative_to(run)), 'evidence_sha256': digest(evidence_path),
              'reason': 'same input, frozen buffers, spectral floating-point drift; preserve labels, freeze validation inference',
              'configuration_and_checkpoint_changes': False}
    new_protocol['repairs'] = [record]
    proposed = copy.deepcopy(manifest.value)
    proposed['identity'] = new_protocol['identity']
    proposed['repairs'] = [record]
    for row in proposed['entries'].values():
        if row['status'] == 'complete':
            # Keep producer identity fields and artifact hashes intact.
            row['reuse_approval'] = REPAIR_ID
        if row['stage'] == 'test_core':
            row['status'] = 'pending'  # New source tests must run again.
    folder = run / 'repair_shared_drift'
    if (folder / 'transaction.json').exists():
        raise ValueError('A different/completed repair transaction already exists')
    atomic_json(folder / 'original_protocol.json', old)
    atomic_json(folder / 'original_manifest.json', manifest.value)
    atomic_json(folder / 'proposed_protocol.json', new_protocol)
    atomic_json(folder / 'proposed_manifest.json', proposed)
    journal = {'repair_id': REPAIR_ID, 'status': 'prepared',
               'proposed_hashes': {name: digest(folder / name) for name in
                                   ('proposed_protocol.json', 'proposed_manifest.json')}}
    atomic_json(folder / 'transaction.json', journal)
    finish_repair_transaction(run)
    print(f'REPAIR COMPLETE: retained {record["retained_train_frames"]} hash-verified training frames; no cache files rewritten', flush=True)
    return new_protocol
