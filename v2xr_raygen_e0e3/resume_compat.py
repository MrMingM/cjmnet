"""Audited resume for the single known pre-training smoke U-fixture defect.

Original request/contract/source snapshot/cache identities are never rewritten.
Actual new sources are verified normally and recorded in an additive repair lineage.
"""
import math
import os
from pathlib import Path
import sys
import time

from .common import HERE, bound, identity, lock, read, safe, sha, stamp, write

REPAIR = 'smoke_unknown_U_v1'
MARKER = 'source_repair/smoke_unknown_U_v1.json'
CHANGED = {'v2xr_raygen_e0e3/' + name for name in (
    'smoke.py', 'tests_local.py', 'README.md', 'LOCAL_VALIDATION.md',
    'controller.py', 'worker.py', 'report.py', 'runtime.py', 'training.py')}
BASE_ADDED = {'v2xr_raygen_e0e3/resume_compat.py', 'v2xr_raygen_e0e3/U_FIX_COMPAT.json'}
NUMERIC_ADDED = {'v2xr_raygen_e0e3/resume_compare.py', 'v2xr_raygen_e0e3/RESUME_CHECK_COMPAT.json'}
RMS_ADDED = {'v2xr_raygen_e0e3/RESUME_RMS_COMPAT.json'}
ADDED = BASE_ADDED | NUMERIC_ADDED | RMS_ADDED
GPU_CAPACITY = 'source_repair/gpu_capacity.json'
NUMERIC_MARKER = 'source_repair/resume_check_v2.json'
RMS_MARKER = 'source_repair/resume_rms_v3.json'


def probe_root(root, micro, physical_gpu=None):
    if physical_gpu is None:
        return Path(root) / 'probe' / ('micro' + str(micro))
    if not isinstance(physical_gpu, str) or not physical_gpu.isascii() or not physical_gpu.isdecimal():
        raise ValueError('Capacity probe requires one integer physical GPU')
    return Path(root) / 'source_repair/gpu_probe' / physical_gpu / ('micro' + str(micro))


def effective_execution(root, capacity_gpu=None, micro=None):
    """Keep the original execution immutable; a measured device move is additive."""
    root = Path(root)
    original = read(root / 'execution.json')
    marker = root / MARKER
    repair = read(marker) if marker.exists() else None
    target = repair.get('requested_physical_gpu', original['physical_gpu']) if repair else original['physical_gpu']
    if capacity_gpu is not None:
        if (repair is None or target == original['physical_gpu'] or capacity_gpu != target
                or capacity_gpu != os.environ.get('GPU') or micro != original['microbatch']):
            raise ValueError('GPU capacity recheck must use the enrolled target and unchanged microbatch')
        probe_root(root, micro, capacity_gpu)  # Validate before using a device value as a directory name.
        return dict(original, physical_gpu=target, gpu_name=None)
    if target == original['physical_gpu']:
        return original
    path = root / GPU_CAPACITY
    if not path.exists():
        raise ValueError('Selected GPU requires a completed capacity recheck before server checks or training')
    record = read(path)
    ep = probe_root(root, original['microbatch'], target) / 'E3/epochs/01'
    measured = read(ep / 'train_complete.json')
    selected = dict(original, physical_gpu=target, gpu_name=measured['gpu_name'])
    if (record['identity'] != identity({k: v for k, v in record.items() if k != 'identity'})
            or record['repair_identity'] != repair['identity'] or record['status'] != 'complete'
            or record['original_execution_sha256'] != sha(root / 'execution.json')
            or record['probe_commit_sha256'] != sha(ep / 'train_complete.json')
            or record['execution'] != selected or measured['physical_gpu'] != target
            or measured['status'] != 'complete' or measured['contract_identity'] != read(root / 'contract.json')['identity']):
        raise ValueError('GPU capacity recheck lineage or selected batch/device differs')
    return selected


def commit_gpu_capacity(root):
    root = Path(root)
    repair = read(root / MARKER)
    original = read(root / 'execution.json')
    target = repair['requested_physical_gpu']
    selected = effective_execution(root, capacity_gpu=target, micro=original['microbatch'])
    ep = probe_root(root, original['microbatch'], target) / 'E3/epochs/01'
    measured = read(ep / 'train_complete.json')
    selected['gpu_name'] = measured['gpu_name']
    _committed_probe(root, read(root / 'contract.json'), selected, epoch_root=ep)
    sampling = read(ep / 'sampling.json')
    if sampling['microbatch'] != original['microbatch'] or sampling['accumulation'] != original['accumulation']:
        raise ValueError('GPU capacity recheck changed the selected batch')
    record = dict(status='complete', repair_identity=repair['identity'], execution=selected,
                  original_execution_sha256=sha(root / 'execution.json'),
                  probe_commit_sha256=sha(ep / 'train_complete.json'))
    record['identity'] = identity(record)
    bound(root / GPU_CAPACITY, record)
    return effective_execution(root)


def check_sources(original, active):
    expected = read(HERE / 'U_FIX_COMPAT.json')['original_files']
    if original['files'] != expected or original['dependencies'] != active['dependencies']:
        raise ValueError('U-fix resume supports only the exact original sources and unchanged dependencies')
    if set(active['files']) not in (set(expected) | ADDED, set(expected) | BASE_ADDED | NUMERIC_ADDED,
                                    set(expected) | BASE_ADDED):
        raise ValueError('Unexpected files in U-fix source transition')
    for name, digest in expected.items():
        if name not in CHANGED and active['files'][name] != digest:
            raise ValueError('Scientific source/config changed; U-fix resume refused: ' + name)


def verify_active_sources(root, active):
    root = Path(root)
    original = read(root / 'request.json')
    marker = root / MARKER
    if not marker.exists():
        if original['sources'] != active:
            raise ValueError('Changed sources require an audited controller resume before starting a worker')
        return None
    repair = read(marker)
    if (repair['repair'] != REPAIR or repair['status'] != 'accepted'
            or repair['identity'] != identity({k: v for k, v in repair.items() if k != 'identity'})
            or (repair['active_sources'] != active and not (root / NUMERIC_MARKER).exists())
            or repair['original_request_sha256'] != sha(root / 'request.json')
            or repair['original_contract_sha256'] != sha(root / 'contract.json')
            or repair['execution_sha256'] != sha(root / 'execution.json')
            or repair['execution_commit_sha256'] != sha(root / 'execution_commit.json')):
        raise ValueError('Immutable U-fix repair lineage/source/config/microbatch differs')
    check_sources(original['sources'], active)
    for name, digest in repair['preserved_evidence'].items():
        if sha(root / name) != digest:
            raise ValueError('Archived original U-fix evidence changed: ' + name)
    if (root / NUMERIC_MARKER).exists():
        _verify_numeric_lineage(root, repair, active)
    return repair


def _verify_numeric_lineage(root, previous, active):
    from .resume_compare import POLICY, POLICY_ID
    record = read(root / NUMERIC_MARKER)
    if (record['identity'] != identity({k: v for k, v in record.items() if k != 'identity'})
            or record['status'] != 'accepted' or record['repair'] != 'resume_state_check_v2'
            or record['previous_marker_sha256'] != sha(root / MARKER)
            or record['previous_sources_identity'] != identity(previous['active_sources'])
            or (record['active_sources'] != active and not (root / RMS_MARKER).exists())
            or record['policy'] != POLICY or record['policy_identity'] != POLICY_ID):
        raise ValueError('Resume-check v2 source/policy/lineage differs')
    for name, digest in record['preserved_evidence'].items():
        if sha(root / name) != digest:
            raise ValueError('Resume-check v2 preserved evidence changed: ' + name)
    for name, expected in record['artifact_stamps'].items():
        if stamp(root / name) != expected:
            raise ValueError('Preserved E0 artifact metadata changed: ' + name)
    if (root / RMS_MARKER).exists():
        _verify_rms_lineage(root, record, active)
    return record


def _verify_rms_lineage(root, previous, active):
    from .resume_compare import POLICY_V3, POLICY_V3_ID
    record = read(root / RMS_MARKER)
    if (record['identity'] != identity({k: v for k, v in record.items() if k != 'identity'})
            or record['status'] != 'accepted' or record['repair'] != 'resume_rms_v3'
            or record['previous_marker_sha256'] != sha(root / NUMERIC_MARKER)
            or record['previous_sources_identity'] != identity(previous['active_sources'])
            or record['active_sources'] != active or record['policy'] != POLICY_V3 or record['policy_identity'] != POLICY_V3_ID):
        raise ValueError('RMS v3 source/policy/lineage differs')
    for name, digest in record['preserved_evidence'].items():
        if sha(root / name) != digest:
            raise ValueError('RMS v3 preserved evidence changed: ' + name)
    for name, expected in record['artifact_stamps'].items():
        if stamp(root / name) != expected:
            raise ValueError('Preserved E1 artifact metadata changed: ' + name)
    return record


def _accept_rms_resume(root, previous, active):
    """Keep exact-load checks and thresholds; record the aggregate absolute-budget correction."""
    from .resume_compare import POLICY, POLICY_ID, POLICY_V3, POLICY_V3_ID
    prior = previous['active_sources']
    compat = read(HERE / 'RESUME_RMS_COMPAT.json')
    if prior['files'] != compat['previous_files'] or prior['dependencies'] != active['dependencies']:
        raise ValueError('RMS v3 requires the exact delivered v2 sources')
    changed = {'v2xr_raygen_e0e3/' + name for name in ('resume_compare.py', 'smoke.py', 'resume_compat.py',
               'controller.py', 'report.py', 'tests_local.py', 'README.md', 'LOCAL_VALIDATION.md')}
    if set(active['files']) != set(prior['files']) | RMS_ADDED:
        raise ValueError('Unexpected files in RMS source transition')
    if any(active['files'][name] != digest for name, digest in prior['files'].items() if name not in changed):
        raise ValueError('Training/model/data or unrelated source changed in RMS repair')
    verify_active_sources(root, prior)
    execution = effective_execution(root)
    if execution['physical_gpu'] != os.environ.get('GPU'):
        raise ValueError('RMS repair must keep the frozen physical GPU')
    status, failure = read(root / 'stage_status.json'), read(root / 'failure.json')
    required = {'preflight', 'synthetic_checks', 'smoke_prepare', 'server_checks', 'smoke_E0_resume_check',
                'smoke_E0_restore_check_v2', 'smoke_E1_restore_check_v2'}
    for group in ('E0', 'E1'):
        required |= {f'smoke_{group}_{action}_{epoch}' for action in ('train', 'eval') for epoch in (1, 2)}
        required.add(f'smoke_{group}_resume_replay')
    allowed = required | {'smoke_E1_resume_check', 'smoke_eval_interruption', 'u_fix_regression', 'resume_check_v2_regression'}
    allowed |= {'capacity_micro' + str(micro) for micro in (8, 4, 2, 1)} | {'capacity_gpu' + execution['physical_gpu']}
    if (any(status.get(name, {}).get('status') != 'complete' for name in required)
            or any(name not in allowed and row['status'] != 'planned' for name, row in status.items())
            or any(p.is_file() for p in (root / 'groups').rglob('*'))):
        raise ValueError('RMS enrollment requires the diagnosed E1-only pre-formal boundary')
    if (failure.get('stage') != 'smoke_E1_resume_check' or failure.get('group') != 'E1' or failure.get('returncode') != 1
            or status.get('smoke_E1_resume_check', {}).get('status') != 'failed'
            or 'Resume exceeds v2 state validation policy' not in failure.get('traceback', '')):
        raise ValueError('Not the diagnosed E1 v2 resume failure')
    folders = [root / branch / 'E1/epochs' / f'{epoch:02d}'
               for epoch in (1, 2) for branch in ('smoke', 'smoke_replay')]
    preserved, artifacts = {}, {}
    for folder in folders:
        commit = read(folder / 'train_complete.json')
        if (commit['status'] != 'complete' or not commit['smoke_only'] or commit['group'] != 'E1'
                or commit['epoch'] != int(folder.name) or commit['contract_identity'] != read(root / 'contract.json')['identity']
                or commit['physical_gpu'] != execution['physical_gpu'] or commit['gpu_name'] != execution['gpu_name']
                or set(commit['output_hashes']) != {'detector.pth', 'resume.pth', 'sampling.json', 'actual_samples.json'}):
            raise ValueError('E1 smoke metadata differs')
        for name, digest in commit['output_hashes'].items():
            if sha(folder / name) != digest:
                raise ValueError('E1 smoke artifact changed: ' + str(folder / name))
            artifacts[(folder / name).relative_to(root).as_posix()] = digest
        preserved[(folder / 'train_complete.json').relative_to(root).as_posix()] = sha(folder / 'train_complete.json')
    for name in ('detector.pth', 'resume.pth', 'sampling.json', 'actual_samples.json'):
        if sha(folders[0] / name) != sha(folders[1] / name):
            raise ValueError('E1 starting checkpoint differs')
    for name in ('sampling.json', 'actual_samples.json'):
        if read(folders[2] / name) != read(folders[3] / name):
            raise ValueError('E1 data schedule differs')
    diagnostic_path = root / 'smoke/E1/resume_check_v2_diagnostic.json'
    restored_path = root / 'smoke/E1/restore_check_v2.json'
    diagnostic, restored = read(diagnostic_path), read(restored_path)
    expected_path = 'optimizer/state/17/exp_avg'
    failures = diagnostic['comparison']['failures']
    numeric = [row for row in diagnostic['comparison']['numeric'] if not row['passed']]
    if (diagnostic['status'] != 'failed' or diagnostic['policy'] != POLICY or diagnostic['policy_identity'] != POLICY_ID
            or not diagnostic['same_start_exact'] or not diagnostic['data_schedule_exact']
            or [row['path'] for row in failures] != [expected_path]
            or [row['path'] for row in numeric] != [expected_path]
            or any(not math.isfinite(numeric[0][key]) for key in ('l2_delta', 'l2_scale', 'maximum_absolute_delta', 'relative_l2'))
            or diagnostic['original_checkpoint_sha256'] != sha(folders[2] / 'resume.pth')
            or diagnostic['replay_checkpoint_sha256'] != sha(folders[3] / 'resume.pth')
            or diagnostic['exact_restore_report_sha256'] != sha(restored_path)
            or restored['status'] != 'complete' or not restored['exact'] or restored['policy_identity'] != POLICY_ID
            or restored['comparison']['status'] != 'complete' or not restored['comparison']['exact']
            or restored['comparison']['arrays_checked'] < 1 or restored['comparison']['failures']
            or restored['source_checkpoint_sha256'] != sha(folders[0] / 'resume.pth')
            or restored['source_commit_sha256'] != sha(folders[0] / 'train_complete.json')
            or restored['group'] != 'E1' or restored['physical_gpu'] != execution['physical_gpu']
            or restored['gpu_name'] != execution['gpu_name']):
        raise ValueError('RMS repair requires matching exact-load/data evidence and the single finite E1 Adam failure')
    for path in (diagnostic_path, restored_path):
        preserved[path.relative_to(root).as_posix()] = sha(path)
    from .runtime import input_stamps
    input_stamps(root)
    for name, value in (('failure', failure), ('stage_status', status)):
        path = root / 'source_repair/resume_rms_v3' / (name + '.json')
        bound(path, value)
        preserved[path.relative_to(root).as_posix()] = sha(path)
    record = dict(repair='resume_rms_v3', status='accepted', time=time.time(),
                  previous_marker_sha256=sha(root / NUMERIC_MARKER), previous_sources_identity=identity(prior),
                  active_sources=active, policy=POLICY_V3, policy_identity=POLICY_V3_ID,
                  preserved_evidence=preserved, verified_artifact_hashes=artifacts,
                  artifact_stamps={name: stamp(root / name) for name in artifacts},
                  interpretation='Correct per-element absolute-error units in aggregate Adam comparison; '
                                 'thresholds and exact loading path unchanged. Preserve E0/E1 outputs and old v2 failure. '
                                 'Only the new server comparison can accept the saved E1 results.')
    record['identity'] = identity(record)
    bound(root / RMS_MARKER, record)
    return record


def _accept_numeric_resume(root, previous, active):
    """Enroll only the diagnosed E0 failure; keep previous lineage and all E0 outputs."""
    from .resume_compare import POLICY, POLICY_ID
    compat = read(HERE / 'RESUME_CHECK_COMPAT.json')
    prior = previous['active_sources']
    if prior['files'] != compat['previous_files'] or prior['dependencies'] != active['dependencies']:
        raise ValueError('Resume-check v2 requires the exact delivered GPU-fix source version')
    changed = {'v2xr_raygen_e0e3/' + name for name in ('smoke.py', 'training.py', 'controller.py', 'worker.py',
               'report.py', 'resume_compat.py', 'tests_local.py', 'README.md', 'LOCAL_VALIDATION.md')}
    if set(active['files']) != set(prior['files']) | NUMERIC_ADDED:
        raise ValueError('Unexpected files in resume-check v2 source transition')
    for name, digest in prior['files'].items():
        if name not in changed and active['files'][name] != digest:
            raise ValueError('Unrelated scientific source changed in resume-check repair: ' + name)
    verify_active_sources(root, prior)
    selected = effective_execution(root)
    if selected['physical_gpu'] != os.environ.get('GPU'):
        raise ValueError('Resume-check v2 must keep the accepted physical GPU')
    status, failure = read(root / 'stage_status.json'), read(root / 'failure.json')
    required = {'preflight', 'synthetic_checks', 'smoke_prepare', 'server_checks', 'smoke_E0_train_1',
                'smoke_E0_eval_1', 'smoke_E0_train_2', 'smoke_E0_eval_2', 'smoke_E0_resume_replay'}
    allowed = required | {'u_fix_regression', 'smoke_eval_interruption', 'smoke_E0_resume_check'}
    allowed |= {'capacity_micro' + str(i) for i in (8, 4, 2, 1)} | {'capacity_gpu' + selected['physical_gpu']}
    if (any(status.get(name, {}).get('status') != 'complete' for name in required)
            or any(name not in allowed and row['status'] != 'planned' for name, row in status.items())
            or any(p.is_file() for p in (root / 'groups').rglob('*'))):
        raise ValueError('Resume-check v2 requires the E0-only pre-formal failure boundary')
    if (failure.get('stage') != 'smoke_E0_resume_check' or failure.get('group') != 'E0'
            or failure.get('returncode') != 1 or status.get('smoke_E0_resume_check', {}).get('status') != 'failed'
            or 'HIP resume exceeds registered tolerance: optimizer/state/' not in failure.get('traceback', '')):
        raise ValueError('Not the diagnosed E0 optimizer tolerance failure')
    folders = [root / branch / 'E0/epochs' / f'{epoch:02d}'
               for epoch in (1, 2) for branch in ('smoke', 'smoke_replay')]
    commits, preserved, artifacts = [], {}, {}
    for folder in folders:
        commit = read(folder / 'train_complete.json')
        if (commit['status'] != 'complete' or not commit['smoke_only'] or commit['group'] != 'E0'
                or commit['epoch'] != int(folder.name) or commit['contract_identity'] != read(root / 'contract.json')['identity']
                or commit['physical_gpu'] != selected['physical_gpu'] or commit['gpu_name'] != selected['gpu_name']
                or set(commit['output_hashes']) != {'detector.pth', 'resume.pth', 'sampling.json', 'actual_samples.json'}):
            raise ValueError('E0 committed smoke metadata differs')
        for name, digest in commit['output_hashes'].items():
            if sha(folder / name) != digest:
                raise ValueError('E0 committed smoke artifact changed: ' + str(folder / name))
            artifacts[(folder / name).relative_to(root).as_posix()] = digest
        preserved[(folder / 'train_complete.json').relative_to(root).as_posix()] = sha(folder / 'train_complete.json')
        commits.append(commit)
    for name in ('detector.pth', 'resume.pth', 'sampling.json', 'actual_samples.json'):
        if sha(folders[0] / name) != sha(folders[1] / name):
            raise ValueError('E0 replay starting checkpoint differs')
    for name in ('sampling.json', 'actual_samples.json'):
        if read(folders[2] / name) != read(folders[3] / name):
            raise ValueError('E0 replay data schedule differs')
    diagnosis_path = None
    allowed_logs = {'criterion_accumulators/' + prefix + field for prefix in ('', 'loss_dict/')
                    for field in ('conf_loss', 'reg_loss', 'diff_loss', 'total_loss')}
    for path in sorted((root / 'diagnostics').glob('resume_E0_*.json'), reverse=True):
        diagnostic = read(path)
        if (diagnostic.get('diagnostic_source_sha256') != compat['diagnostic_source_sha256']
                or diagnostic.get('checkpoint_commits') != commits or not diagnostic.get('artifact_hashes_verified')
                or not diagnostic.get('epoch1_start_identical') or not diagnostic.get('same_device')
                or not all(diagnostic.get('epoch2_schedules_identical', {}).get(k) for k in ('sampling.json', 'actual_samples.json'))
                or not all(diagnostic.get('sections', {}).get(k, {}).get('passed') for k in ('model', 'aux', 'rng', 'scheduler'))
                or any(row['path'] not in allowed_logs for row in diagnostic.get('exact_errors', []))
                or any(not row['path'].startswith('optimizer/state/') or row.get('relative_l2', float('inf')) > POLICY['rtol']
                       or row.get('nonfinite_left', 1) or row.get('nonfinite_right', 1)
                       for row in diagnostic.get('failed_floating_arrays', []))):
            continue
        diagnosis_path = path
        break
    if diagnosis_path is None:
        raise ValueError('Run the supplied CPU diagnostic first; expected identical data/RNG/model and bounded finite Adam differences')
    from .runtime import input_stamps
    input_stamps(root)
    _prepared_cache(root, read(root / 'contract.json'))
    for name, value in (('failure', failure), ('stage_status', status)):
        path = root / 'source_repair/resume_check_v2' / (name + '.json')
        bound(path, value)
        preserved[path.relative_to(root).as_posix()] = sha(path)
    preserved[diagnosis_path.relative_to(root).as_posix()] = sha(diagnosis_path)
    record = dict(repair='resume_state_check_v2', status='accepted', time=time.time(),
                  previous_marker_sha256=sha(root / MARKER), previous_sources_identity=identity(prior),
                  active_sources=active, policy=POLICY, policy_identity=POLICY_ID, preserved_evidence=preserved,
                  verified_artifact_hashes=artifacts,
                  artifact_stamps={name: stamp(root / name) for name in artifacts},
                  interpretation='Preserve all committed E0 results. Add exact real-loader roundtrip; '
                                 'model tolerance unchanged, Adam moments per-tensor L2, float loss logs toleranced. '
                                 'Original pointwise failure retained; recipe/model/data/RNG unchanged.')
    record['identity'] = identity(record)
    bound(root / NUMERIC_MARKER, record)
    return previous


def _committed_probe(root, contract, execution, epoch_root=None):
    micro = execution['microbatch']
    if micro not in (8, 4, 2, 1) or execution['accumulation'] != 8 // micro or execution['effective_batch'] != 8:
        raise ValueError('Invalid previously selected microbatch')
    ep = epoch_root or probe_root(root, micro) / 'E3/epochs/01'
    commit = read(ep / 'train_complete.json')
    expected_files = {'detector.pth', 'resume.pth', 'sampling.json', 'actual_samples.json'}
    if (commit['status'] != 'complete' or not commit['smoke_only'] or commit['group'] != 'E3'
            or commit['epoch'] != 1 or commit['contract_identity'] != contract['identity']
            or set(commit['output_hashes']) != expected_files
            or commit['physical_gpu'] != execution['physical_gpu'] or commit['gpu_name'] != execution['gpu_name']):
        raise ValueError('Completed capacity probe evidence differs')
    for name, digest in commit['output_hashes'].items():
        if sha(ep / name) != digest:
            raise ValueError('Completed capacity probe artifact changed: ' + name)


def _prepared_cache(root, contract):
    labels = root / 'labels'
    expected = identity(dict(input=contract['inputs'], config=contract['config'], sources=contract['sources'],
                             geometry=read(root / 'protocol.json')['geometry']))
    if read(labels / 'contract.json')['identity'] != expected:
        raise ValueError('Original label cache contract differs')
    selected = read(root / 'preflight/smoke_indices.json')['indices']
    for index in selected:
        sample_path = labels / 'samples' / f'{index:06d}.json'
        sample = read(sample_path)
        paths = [sample_path]
        paths += [labels / 'windows' / f'{epoch:02d}' / f'{index:06d}.json' for epoch in (1, 2)]
        for path in paths:
            row = read(safe(path))
            if row['status'] != 'complete' or sha(safe(path.with_suffix('.npz'))) != row['sha256']:
                raise ValueError('Prepared smoke cache missing/changed: ' + str(path))
            if path.parent.name == 'samples' and row['cache_identity'] != expected:
                raise ValueError('Prepared ray sample belongs to another cache')
            if path.parent.name != 'samples':
                window_identity = identity(dict(base=sha(sample_path), config=contract['config'],
                                                uid=sample['uid'], epoch=int(path.parent.name)))
                if row['identity'] != window_identity:
                    raise ValueError('Prepared window config/UID/source identity differs')
    return len(selected)


def accept_resume(root, original_request, current_request):
    root = Path(root)
    if ({k: v for k, v in original_request.items() if k != 'sources'}
            != {k: v for k, v in current_request.items() if k != 'sources'}):
        raise ValueError('U-fix resume requires identical configuration, seed and paths')
    if (root / MARKER).exists():
        previous = read(root / MARKER)
        if (root / NUMERIC_MARKER).exists() and not (root / RMS_MARKER).exists():
            numeric = read(root / NUMERIC_MARKER)
            if numeric['active_sources'] != current_request['sources']:
                with lock(root / '.worker.lock'):
                    _accept_rms_resume(root, numeric, current_request['sources'])
        if previous['active_sources'] != current_request['sources'] and not (root / NUMERIC_MARKER).exists():
            with lock(root / '.worker.lock'):
                _accept_numeric_resume(root, previous, current_request['sources'])
        repair = verify_active_sources(root, current_request['sources'])
        target = repair.get('requested_physical_gpu', read(root / 'execution.json')['physical_gpu'])
        if os.environ.get('GPU', target) != target:
            raise ValueError('Resume must keep the GPU selected when enrolling this repair: ' + target)
        return repair
    check_sources(original_request['sources'], current_request['sources'])
    # Controller lock is already held. Refuse enrollment while any orphan worker still runs.
    with lock(root / '.worker.lock'):
        contract = read(root / 'contract.json')
        if (contract['identity'] != identity({k: v for k, v in contract.items() if k != 'identity'})
                or contract['sources'] != original_request['sources']
                or contract['config'] != current_request['config']
                or read(root / 'config.json') != contract['config']):
            raise ValueError('Original training contract/source/config identity differs')
        status, failure = read(root / 'stage_status.json'), read(root / 'failure.json')
        allowed = {'preflight', 'synthetic_checks', 'smoke_prepare', 'server_checks'}
        allowed |= {'capacity_micro' + str(i) for i in (8, 4, 2, 1)}
        if any(k not in allowed and v['status'] != 'planned' for k, v in status.items()):
            raise ValueError('U-fix enrollment is limited to the reported pre-training failure boundary')
        if any(status.get(k, {}).get('status') != 'complete' for k in ('preflight', 'synthetic_checks', 'smoke_prepare')):
            raise ValueError('Preflight/synthetic/prepared smoke stages are not committed')
        if (status.get('server_checks', {}).get('status') != 'failed'
                or failure.get('stage') != 'server_checks' or failure.get('returncode') != 1
                or "KeyError: 'U'" not in failure.get('traceback', '')
                or 'zero_unknown' not in failure['traceback']):
            raise ValueError('Failure is not the exact supported all-unknown U fixture error')
        for name in ('groups', 'smoke', 'smoke_replay'):
            if any(p.is_file() for p in (root / name).rglob('*')):
                raise ValueError('Four-group smoke/formal artifacts already exist; narrow U-fix enrollment refused')
        execution = read(root / 'execution.json')
        commit = read(root / 'execution_commit.json')
        if (commit['sha256'] != sha(root / 'execution.json') or commit['parent_contract'] != contract['identity']
                or status.get('capacity_micro' + str(execution['microbatch']), {}).get('status') != 'complete'):
            raise ValueError('Original capacity execution commit differs')
        _committed_probe(root, contract, execution)
        target = os.environ.get('GPU', execution['physical_gpu'])
        probe_root(root, execution['microbatch'], target)
        count = _prepared_cache(root, contract)
        # All source/input/path checks remain active. This is metadata/source verification, no model/data decoding.
        from .runtime import input_stamps
        input_stamps(root)
        evidence = {'source_repair/original_failure.json': failure,
                    'source_repair/original_stage_status.json': status}
        for name, value in evidence.items():
            bound(root / name, value)
        record = dict(repair=REPAIR, status='accepted', time=time.time(),
            original_request_sha256=sha(root / 'request.json'), original_contract_sha256=sha(root / 'contract.json'),
            execution_sha256=sha(root / 'execution.json'), execution_commit_sha256=sha(root / 'execution_commit.json'),
            active_sources=current_request['sources'], original_sources_identity=identity(original_request['sources']),
            requested_physical_gpu=target,
            preserved_evidence={name: sha(root / name) for name in evidence}, preserved_smoke_samples=count,
            reused_stages=['preflight', 'synthetic_checks', 'smoke_prepare', 'capacity_micro' + str(execution['microbatch'])],
            repeated_stages=['u_fix_regression'] + (['capacity_gpu' + target] if target != execution['physical_gpu'] else []) + ['server_checks'],
            interpretation='Exact self-test fixture repair plus auditable resume/device plumbing; training recipe unchanged. '
                           'Original contract/cache provenance retained; subsequent stage execution uses active_sources.')
        record['identity'] = identity(record)
        bound(root / MARKER, record)
        return record


def record_worker(root, active):
    repair = verify_active_sources(root, active)
    if repair is not None:
        numeric = Path(root) / NUMERIC_MARKER
        rms = Path(root) / RMS_MARKER
        write(Path(root) / 'source_repair/stage_runs' / (str(time.time_ns()) + '.json'),
              dict(repair_identity=repair['identity'], active_sources_identity=identity(active),
                   resume_check_repair_identity=read(numeric)['identity'] if numeric.exists() else None,
                   rms_repair_identity=read(rms)['identity'] if rms.exists() else None,
                   original_contract_sha256=repair['original_contract_sha256'], command=sys.argv,
                   pid=os.getpid(), time=time.time()))
    return repair
