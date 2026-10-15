"""Extend old immutable repair provenance and verify existing Source AP evidence."""
from copy import deepcopy
from pathlib import Path
import shutil

from local_fusion_source_kill_stage2.common import (
    WEATHERS, atomic_json, commit_unit, digest, external_run, load_contract,
    object_hash, read_json, resource_check, resume, safe_path, settings, worker)
from local_fusion_source_kill_stage2.source_adapter import reproduction
from local_fusion_source_kill_stage2_replay_fix.contract import environment
from .reproduction import compare_trajectory, original_compare

PACKAGE = Path(__file__).resolve().parent
CHECKED_EVIDENCE_SHA256 = '4b741e0472feebecfee4570289abd1a59d951a1bd4aa8b4ed42aa5e1c2ac3c18'


def extra_sources():
    return {str(safe_path(p)):digest(p) for p in PACKAGE.iterdir() if p.suffix in ('.py','.sh','.md')}


def verify_source_gate(previous, old):
    folder = previous/'source_reproduction'
    manifest = read_json(folder/'manifest.json')
    if manifest['status'] != 'complete' or manifest['identity'] != old['identity']:
        raise ValueError('Previous Source AP gate is not complete for this protocol')
    if resume(folder/'result.json', old['identity']) is None:
        raise ValueError('Previous Source AP gate lacks a verified completion receipt')
    gate = read_json(folder/'SOURCE_REPRODUCTION_GATE.json')
    if gate['status'] != 'complete' or gate['root_identity'] != old['identity'] or gate['tolerance'] != 1e-6:
        raise ValueError('Previous Source AP gate identity/tolerance mismatch')
    receipt = read_json(folder/'result.json')
    files = {str(folder/'manifest.json'):digest(folder/'manifest.json'),
             str(folder/'result.json'):digest(folder/'result.json')}
    for name, sha in receipt['artifacts'].items():
        path = safe_path(folder/name)
        if folder not in path.parents or digest(path) != sha:
            raise ValueError('Previous Source artifact changed: '+name)
        files[str(path)] = sha
    conf = object_hash(dict(purpose='Source-only repair AP gate',root=old['identity']))
    identity = object_hash(dict(root=old['identity'],null=conf))
    reference = read_json(Path(old['oracle_run'])/'death_test_results.json')['conditions']
    conditions = {}
    for weather in WEATHERS:
        for mode in ('same','task'):
            path = folder/'conditions'/weather/'Source'/'0'/(mode+'.json')
            if str(path) not in files:
                raise ValueError('Missing condition from Source gate receipt')
            row = resume(path,identity)
            if row is None or (row['weather'],row['family'],row['mode'],row['seed']) != (weather,'Source',mode,0):
                raise ValueError('Incomplete or mismatched original Source condition')
            saved = reference[weather]['results']
            reproduction(None,row['Shared'],saved['Shared'],weather+'/Shared')
            reproduction(None,row['AP'],saved['Oracle-'+mode.title()],weather+'/'+mode)
            name = f'{weather}/Source/0/{mode}'
            conditions[name] = dict(AP=row['AP'],Shared=row['Shared'])
            if gate['conditions'].get(name) != conditions[name]:
                raise ValueError('Source gate summary differs from original condition receipt')
            for artifact, sha in read_json(path)['artifacts'].items():
                raw = safe_path(path.parent/artifact)
                if files.get(str(raw)) != sha:
                    raise ValueError('Source raw frames missing from protected parent evidence')
    if set(gate['conditions']) != set(conditions):
        raise ValueError('Source gate does not contain exactly all eight conditions')
    return files, conditions


def initialize(run, previous):
    run, previous = external_run(run), safe_path(previous)
    if run == previous:
        raise ValueError('New RUN required; old failed results stay immutable')
    old = load_contract(previous)
    resource_check(old)
    if 'k2_replay_repair_v2' in old:
        raise ValueError('PREVIOUS_RUN must be the original v1 repair RUN')
    if environment() != old['k2_replay_repair']['environment']:
        raise ValueError('Use the same diagnosed GPU/software environment')
    if shutil.disk_usage(run).free < settings()['minimum_free_disk_gib']*1024**3:
        raise ValueError('Require 20 GiB free disk')
    evidence = previous/'k2/conditions/fog/same/00000608.reproduction.json'
    if digest(evidence) != CHECKED_EVIDENCE_SHA256:
        raise ValueError('Frame 608 evidence differs from the artifact inspected by the user')
    record = read_json(evidence)
    if record['root_identity'] != old['identity'] or record['accepted'] is not False:
        raise ValueError('Failed replay artifact has mismatched provenance')
    if original_compare(record['actual_trajectory'],record['historical_trajectory'])['accepted']:
        raise ValueError('Evidence does not demonstrate the expected original rejection')
    if not compare_trajectory(record['actual_trajectory'],record['historical_trajectory'])['accepted']:
        raise ValueError('Evidence does not qualify for the narrow unselectable-E repair')
    source_files, conditions = verify_source_gate(previous,old)
    extra = extra_sources()
    descriptor = dict(schema=2, previous_repair_run=str(previous), previous_identity=old['identity'],
        evidence_path=str(evidence), evidence_sha256=digest(evidence), extra_source_hashes=extra,
        scope='Only wholly unselectable homogeneous E candidates at KEEP targets may permute',
        scientific_thresholds_changed=False, old_k2_cache_reused=False,
        original_source_AP_gate_referenced=True)
    if (run/'PROTOCOL.json').exists():
        saved = load_contract(run)
        if saved.get('k2_replay_repair_v2') != descriptor:
            raise ValueError('V2 repair source/provenance changed; use a new RUN')
        return saved, conditions
    if any((run/name).exists() for name in ('k2','source_reproduction','STAGE2_FINAL.json')):
        raise ValueError('Scientific state exists without immutable protocol')
    protocol = deepcopy(old)
    protocol.pop('identity')
    protocol['k2_replay_repair_v2'] = descriptor
    protocol['input_hashes'].update(source_files)
    protocol['input_hashes'].update(extra)
    protocol['input_hashes'].update({str(previous/'PROTOCOL.json'):digest(previous/'PROTOCOL.json'),
                                   str(evidence):digest(evidence)})
    snapshot = run/'K2_REPLAY_V2_SOURCE_SNAPSHOT.json'
    atomic_json(snapshot,descriptor)
    protocol['input_hashes'][str(snapshot)] = digest(snapshot)
    protocol['identity'] = object_hash(protocol)
    for name in ('SOURCE_SNAPSHOT.json','source_snapshot.json'):
        atomic_json(run/name,protocol['source_hashes'])
    for name in ('protocol.json','protocol.yaml','PROTOCOL.json'):
        atomic_json(run/name,protocol)
    load_contract(run)
    return protocol, conditions


def reference_source_gate(run, protocol, conditions):
    with worker(run,'source_reproduction',protocol) as folder:
        if resume(folder/'result.json',protocol['identity']) is not None:
            return
        path=folder/'SOURCE_REPRODUCTION_GATE.json'
        atomic_json(path,dict(status='complete',root_identity=protocol['identity'],tolerance=1e-6,
            conditions=conditions, reference_only=True,
            origin_run=protocol['k2_replay_repair_v2']['previous_repair_run'],
            origin_identity=protocol['k2_replay_repair_v2']['previous_identity'],
            note='All eight original condition receipts/raw hashes and historical AP rechecked; no new inference'))
        commit_unit(folder/'result.json',protocol['identity'],dict(status='complete',reference_only=True),[path])
