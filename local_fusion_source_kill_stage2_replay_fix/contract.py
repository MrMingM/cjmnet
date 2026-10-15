"""A new immutable RUN inherits science settings and records the repair."""
from copy import deepcopy
import os
from pathlib import Path
import shutil

from local_fusion_action_utility_audit.reproducibility import POLICY
from local_fusion_source_kill_stage2.common import (
    ROOT, atomic_json, digest, external_run, load_contract, object_hash,
    jsonlines, read_json, resource_check, safe_path, settings)
from .reproduction import compare_trajectory

PACKAGE = Path(__file__).resolve().parent
DIAGNOSTIC_FILES = ('DIAGNOSIS.json', 'CPU_SUMMARY.json', 'stage1_expected.json',
    'original_stage1_adapter.json', 'original_stage1_adapter_repeated.json', 'stage2_without_perturbations.json')


def repair_sources():
    paths = list(PACKAGE.glob('*.py')) + list(PACKAGE.glob('*.sh')) + list(PACKAGE.glob('*.md'))
    paths += [ROOT/'local_fusion_source_kill_stage2_diagnostics'/name for name in ('differences.py', 'inspect.py', '__init__.py')]
    paths += [ROOT/'local_fusion_action_utility_audit/reproducibility.py']
    return {str(safe_path(p)): digest(p) for p in paths}


def environment():
    import torch  # remote entry only
    if not torch.cuda.is_available():
        raise ValueError('Remote GPU required for genuine action evaluation')
    return dict(torch=str(torch.__version__), hip=torch.version.hip,
        device='cuda:0', device_name=torch.cuda.get_device_name(0),
        **{k: os.environ.get(k) for k in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')})


def derive(parent, parent_run, diagnosis, evidence, source_hashes, env):
    protocol = deepcopy(parent)
    protocol.pop('identity')
    protocol['k2_replay_repair'] = dict(schema=1,
        reason='Replace whole-record bit equality by recorded S0 numeric policy plus exact decision/order equality',
        parent_run=str(parent_run), parent_identity=parent['identity'], diagnosis=str(diagnosis),
        evidence_hashes=evidence, repair_source_hashes=source_hashes, environment=env,
        legacy_policy=POLICY, scientific_thresholds_changed=False,
        legacy_policy_scope='Only named continuous key/focal/proxy coordinates; every other field and full9 order exact',
        k1_reused_read_only=True, no_old_k2_cache_reuse=True)
    protocol['input_hashes'].update(evidence)
    protocol['input_hashes'].update(source_hashes)
    protocol['input_hashes'][str(parent_run/'PROTOCOL.json')] = digest(parent_run/'PROTOCOL.json')
    protocol['identity'] = object_hash(protocol)
    return protocol


def initialize(run, parent_run, diagnosis):
    run, parent_run, diagnosis = external_run(run), safe_path(parent_run), safe_path(diagnosis)
    if run == parent_run:
        raise ValueError('New RUN required: never overwrite the failed or running parent')
    parent = load_contract(parent_run)
    resource_check(parent)
    if shutil.disk_usage(run).free < settings()['minimum_free_disk_gib'] * 1024**3:
        raise ValueError('Insufficient free disk: require 20 GiB')
    source_hashes = repair_sources()
    env = environment()
    s0 = read_json(Path(parent['s0_run'])/'protocol.json')
    if s0.get('shared_reproducibility') != POLICY:
        raise ValueError('Original S0 did not register this numerical policy; no automatic fallback permitted')
    evidence = {str(diagnosis/name): digest(diagnosis/name) for name in DIAGNOSTIC_FILES}
    diagnostic = read_json(diagnosis/'DIAGNOSIS.json')
    if diagnostic['root_identity'] != parent['identity']:
        raise ValueError('Diagnostic belongs to another parent protocol')
    actual = read_json(diagnosis/'original_stage1_adapter.json')
    expected = read_json(diagnosis/'stage1_expected.json')
    condition = read_json(Path(parent['stage1_run'])/'K1_0_ORACLE_DECOMPOSITION.json')['conditions'][
        f'{diagnostic["weather"]}_{diagnostic["mode"]}_A_ONLY']
    raw_path = safe_path(Path(parent['stage1_run'])/condition['frame_artifact'])
    if (Path(parent['stage1_run']) not in raw_path.parents or digest(raw_path) != condition['frame_sha256'] or
            str(raw_path) != diagnostic['saved_stage1_file'] or
            condition['frame_sha256'] != diagnostic['saved_stage1_file_sha256']):
        raise ValueError('Diagnostic historical trajectory provenance mismatch')
    historical = next((r for r in jsonlines(raw_path) if r['sample_index'] == diagnostic['frame']), None)
    if historical is None or expected != dict(records=historical['records'], current_state_audit=historical['current_state_audit']):
        raise ValueError('Diagnostic expected data differs from original Stage 1 trajectory')
    evidence[str(raw_path)] = condition['frame_sha256']
    check = compare_trajectory(actual, expected)
    if not check['accepted']:
        raise ValueError('Diagnostic has substantive differences; repair cannot accept it')
    if (actual != read_json(diagnosis/'original_stage1_adapter_repeated.json') or
            actual != read_json(diagnosis/'stage2_without_perturbations.json')):
        raise ValueError('Repeated current inputs or adapters are not exactly identical')
    for name in ('torch', 'hip', 'device', 'device_name', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        if env[name] != diagnostic['environment'][name]:
            raise ValueError('Use diagnosed GPU/software environment: '+name)
    protocol = derive(parent, parent_run, diagnosis, evidence, source_hashes, env)
    snapshot = run/'K2_REPLAY_SOURCE_SNAPSHOT.json'
    atomic_snapshot = dict(source_hashes=source_hashes, parent_identity=parent['identity'])
    if (run/'PROTOCOL.json').exists():
        saved = load_contract(run)
        if saved['k2_replay_repair'] != protocol['k2_replay_repair'] or read_json(snapshot) != atomic_snapshot:
            raise ValueError('Repair input/source/environment changed: a new RUN is required')
        return saved
    if any((run/name).exists() for name in ('k1', 'k2', 'source_reproduction', 'STAGE2_FINAL.json')):
        raise ValueError('Scientific state exists without immutable repair protocol')
    atomic_json(snapshot, atomic_snapshot)
    protocol['input_hashes'][str(snapshot)] = digest(snapshot)
    protocol['identity'] = object_hash({k:v for k,v in protocol.items() if k != 'identity'})
    for name in ('SOURCE_SNAPSHOT.json', 'source_snapshot.json'):
        atomic_json(run/name, protocol['source_hashes'])
    for name in ('protocol.json', 'protocol.yaml', 'PROTOCOL.json'):
        atomic_json(run/name, protocol)
    load_contract(run)
    return protocol
