"""Controlled E0 continuation/restart probe, outside the sealed research package.

Server only: one two-update continuous run creates a common optimizer-boundary
checkpoint; three independent workers resume its second update. A separate strict
determinism worker records an unsupported operation or finishes its diagnostic run.
Original artifacts and the FAILED research gate are preserved. No AP is evaluated.
"""
import argparse
from datetime import datetime, timezone
import gc
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback
import zipfile

from uecp_gspr_resume_diagnose import (compare, committed_epoch, diagnostic_sources,
                                     read, sha, stopped_pipeline)


ARMS = ('continuous', 'resume1', 'resume2', 'resume3', 'strict')
COPY_FILES = ('config.json', 'contract.json', 'data_manifest.json', 'request.json',
              'protocol.json', 'source_snapshot.json', 'execution.json',
              'preflight/evidence.json', 'preflight/train_manifest.json',
              'preflight/input_stamps.json', 'preflight/gspr_source_config.json')
STATE_FIELDS = ('model', 'optimizer', 'scheduler', 'criterion', 'rng', 'global_step',
                'position', 'totals', 'elapsed', 'actual', 'gradients', 'distributions',
                'contract_identity', 'variant', 'epoch', 'microbatch',
                'order_identity', 'accumulation_position', 'amp_scaler')
META_FIELDS = ('contract_identity', 'variant', 'epoch', 'position', 'global_step',
               'microbatch', 'order_identity', 'accumulation_position', 'amp_scaler')


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(data)


def summary(result):
    changed = result['changed_floating_arrays']
    return dict(exact=result['exact'], within_registered_tolerance=result['within_registered_tolerance'],
        structural_or_exact_errors=result['structural_or_exact_errors'],
        failed_floating_arrays=len(result['failed_floating_arrays']), changed_floating_arrays=len(changed),
        maximum_absolute_difference=max((r['maximum_absolute_difference'] for r in changed), default=0.),
        largest_relative_changes=sorted(changed, key=lambda r: r['relative_l2'], reverse=True)[:3])


def classify(pairs, restored_exact):
    expected = {(a, b) for i, a in enumerate(ARMS[:4]) for b in ARMS[i+1:4]}
    if len(pairs) != len(expected) or {(p['left'], p['right']) for p in pairs} != expected:
        return 'INCOMPLETE_CONTROLLED_COMPARISONS'
    if not restored_exact or any(not p['exact_inputs_rng_metadata'] for p in pairs):
        return 'RESTORE_OR_INPUT_MISMATCH'
    repeats = [p for p in pairs if p['left'].startswith('resume') and p['right'].startswith('resume')]
    if any(not p['numerical_within_registered_tolerance'] for p in repeats):
        return 'REPEATED_CALCULATION_EXCEEDS_ORIGINAL_TOLERANCE'
    if any(not p['numerical_within_registered_tolerance'] for p in pairs):
        return 'RESTART_ASSOCIATED_DIFFERENCE_REQUIRES_FURTHER_CHECK'
    return 'CONTROLLED_COMPARISONS_WITHIN_ORIGINAL_TOLERANCE'


def verify_archive():
    path = Path(sys.argv[0]).resolve()
    if not zipfile.is_zipfile(path):
        raise ValueError('Run the delivered executable ZIP, without unpacking it')
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if set(archive.namelist()) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Diagnostic archive member set differs')
        import hashlib
        for name, expected in manifest['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Diagnostic archive member changed: '+name)
    return path


def environment(root, project):
    request = read(root / 'request.json')
    if request.get('family') != 'uecp_gspr_v2xr':
        raise ValueError('Wrong experiment family')
    config = read(root / 'config.json')
    if request['config'] != config or config['microbatch'] not in (1, 2, 4, 8):
        raise ValueError('Original config differs')
    if config['smoke_updates'] != 2 or config['effective_batch'] != 8:
        raise ValueError('This probe requires the original two-update smoke budget')
    commit = committed_epoch(root / 'smoke/E0/baseline/epochs/01', 1)
    gpu = str(commit['physical_gpu'])
    if not gpu.isdecimal() or request['gpu'] != gpu:
        raise ValueError('Original physical GPU identity differs')
    env = dict(os.environ)
    for key, value in request['paths'].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    if Path(env.get('PROJECT_ROOT', project)).resolve() != project:
        raise ValueError('Project differs from the original request')
    for key in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        env.pop(key, None)
    env.update(GPU=gpu, ROCR_VISIBLE_DEVICES=gpu, PYTHONHASHSEED=str(config['seed']),
               SEED=str(config['seed']), EPOCHS=str(config['epochs']),
               MICROBATCH=str(config['microbatch']), AMP='0', PYTHONDONTWRITEBYTECODE='1')
    return env


def overlay(root, probe, arm):
    target = probe / 'arms' / arm
    target.mkdir(parents=True, exist_ok=False)
    for name in COPY_FILES:
        source = root / name
        if source.exists():
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            if sha(source) != sha(dest):
                raise ValueError('Diagnostic metadata copy differs: '+name)
    return target


def numpy_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, dict):
        return {k: numpy_tree(v, torch) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(numpy_tree(v, torch) for v in value)
    return value


def backend_flags(torch):
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_enabled=torch.backends.cudnn.enabled, cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads())


def run_locals(function):
    frame = inspect.currentframe().f_back
    try:
        while frame is not None and frame.f_code is not function.__code__:
            frame = frame.f_back
        if frame is None:
            raise ValueError('Observer was not called by the original training function')
        return dict(frame.f_locals)
    finally:
        del frame


def exact_live_state(local, train):
    result = {k: local[k] for k in ('variant', 'epoch', 'position', 'global_step', 'totals',
                                   'elapsed', 'actual', 'gradients', 'distributions')}
    result.update(model=local['model'].state_dict(), optimizer=local['opt'].state_dict(),
        scheduler=local['scheduler'].state_dict(), criterion=local['crit'].state_dict(), rng=train.rng_state(),
        contract_identity=local['contract'], microbatch=local['micro'], order_identity=train.identity(local['order']),
        accumulation_position=0, amp_scaler=None)
    return result


def worker(root, project, probe, arm):
    sys.path.insert(0, str(project))
    reviewed = diagnostic_sources(project, read(root / 'request.json')['sources'])
    from uecp_gspr_v2xr.runtime import bootstrap, input_stamps
    input_stamps(root)
    bootstrap(root, gpu=True)
    import torch
    from uecp_gspr_v2xr import train
    torch.use_deterministic_algorithms(False)
    original_commit = read(root / 'smoke/E0/baseline/epochs/01/train_complete.json')
    if torch.cuda.get_device_name(0) != original_commit['gpu_name']:
        raise ValueError('Original GPU model differs')
    free, total = torch.cuda.mem_get_info()
    if free / total < .9:
        raise RuntimeError('Original GPU has less than 90% free memory')
    target = probe / 'arms' / arm
    evidence = probe / 'evidence' / arm
    evidence.mkdir(parents=True, exist_ok=False)
    shared = probe / 'shared_boundary.pth'
    if arm != 'continuous':
        checkpoint = torch.load(shared, map_location='cpu', weights_only=False)
        if checkpoint['global_step'] != 1 or checkpoint['position'] != 8 or checkpoint['epoch'] != 1:
            raise ValueError('Shared checkpoint is not the first complete optimizer boundary')
        folder = target / 'smoke/E0/baseline/recovery'
        folder.mkdir(parents=True)
        shutil.copyfile(shared, folder / 'slot0.pth')
        train.write(folder / 'latest.json', dict(slot=0, sha256=sha(shared), epoch=1, global_step=1))
        del checkpoint
    original_recovery, original_restore, original_losses = train.recovery, train.restore_rng, train.losses
    original_adam = torch.optim.AdamW
    loaded = []

    def recovery(branch, state):
        if branch.resolve() != (target / 'smoke/E0/baseline').resolve():
            raise ValueError('Training attempted to write outside the diagnostic arm')
        original_recovery(branch, state)
        if arm == 'continuous' and state['global_step'] == 1:
            if shared.exists():
                raise ValueError('Common boundary already exists; preserved')
            train.atomic_torch(shared, state)
            saved = torch.load(shared, map_location='cpu', weights_only=False)
            result = compare(numpy_tree(state, torch), numpy_tree(saved, torch), 'boundary', 0., 0.)
            write_new(evidence / 'boundary.json', dict(shared_sha256=sha(shared),
                exact=result['exact'], comparison=summary(result), backend=backend_flags(torch)))
            if not result['exact']:
                raise ValueError('Saved common boundary differs from the live continuous state')

    def restore(rng):
        original_restore(rng)
        local = run_locals(train.run)
        expected = torch.load(shared, map_location='cpu', weights_only=False)
        actual = exact_live_state(local, train)
        result = compare(numpy_tree({k: expected[k] for k in STATE_FIELDS}, torch),
                         numpy_tree(actual, torch), 'post_load', 0., 0.)
        record = dict(exact=result['exact'], shared_sha256=sha(shared),
                      comparison=summary(result), backend=backend_flags(torch))
        write_new(evidence / 'restore.json', record)
        loaded.append(result['exact'])
        if not result['exact']:
            raise ValueError('Actual original training loader did not restore the common state exactly')
        if arm == 'strict':
            torch.use_deterministic_algorithms(True)

    def losses(*args, **kwargs):
        result = original_losses(*args, **kwargs)
        local = run_locals(train.run)
        if local['offset'] == 8:
            train.atomic_torch(evidence / 'first_micro_forward.pth',
                               {k: v.detach().cpu() for k, v in args[1].items()})
        return result

    class ObservedAdamW(original_adam):
        def step(self, *args, **kwargs):
            local = run_locals(train.run)
            if local['global_step'] == 1:
                train.atomic_torch(evidence / 'second_update_gradients.pth',
                    {n: None if p.grad is None else p.grad.detach().cpu()
                     for n, p in local['model'].named_parameters()})
            return super().step(*args, **kwargs)

    train.recovery, train.restore_rng, train.losses = recovery, restore, losses
    torch.optim.AdamW = ObservedAdamW
    try:
        train.run(target, 'E0', 1, smoke=True)
        status = dict(status='COMPLETE', arm=arm, actual_original_train_run=True,
                      source_review=reviewed, backend=backend_flags(torch), post_load_exact=loaded)
    except Exception as exc:
        known = isinstance(exc, RuntimeError) and any(word in str(exc).lower()
                    for word in ('deterministic', 'determinism', 'cublas_workspace_config'))
        status = dict(status='STRICT_OPERATION_BLOCKED' if arm == 'strict' and loaded == [True] and known else 'ERROR',
            arm=arm, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
            backend=backend_flags(torch), post_load_exact=loaded)
        if status['status'] == 'ERROR':
            write_new(evidence / 'worker.json', status)
            raise
    finally:
        train.recovery, train.restore_rng, train.losses = original_recovery, original_restore, original_losses
        torch.optim.AdamW = original_adam
    if arm != 'continuous' and loaded != [True]:
        raise ValueError('Expected exactly one verified restore')
    if status['status'] == 'COMPLETE':
        original_init = read(root / 'smoke/E0/baseline/initialization.json')
        if read(target / 'smoke/E0/baseline/initialization.json') != original_init:
            raise ValueError('Shared model initialization differs from the original E0')
        if read(target / 'smoke/E0/baseline/epochs/01/sampling.json') != read(root / 'smoke/E0/baseline/epochs/01/sampling.json'):
            raise ValueError('Controlled sampling differs from the original smoke')
        if read(target / 'smoke/E0/baseline/epochs/01/actual_samples.json') != read(root / 'smoke/E0/baseline/epochs/01/actual_samples.json'):
            raise ValueError('Controlled input/label tensors differ from the original smoke')
    write_new(evidence / 'worker.json', status)
    print('ARM_FINISHED '+arm+' '+status['status'], flush=True)


def compare_arms(probe, left, right, torch, tolerance):
    def load(arm, name):
        if name == 'resume.pth':
            path = probe / 'arms' / arm / 'smoke/E0/baseline/epochs/01' / name
        else:
            path = probe / 'evidence' / arm / name
        return numpy_tree(torch.load(path, map_location='cpu', weights_only=False), torch)
    a, b = load(left, 'resume.pth'), load(right, 'resume.pth')
    sections = {key: summary(compare(a[key], b[key], key, **tolerance))
                for key in ('model', 'optimizer', 'totals', 'actual', 'rng', 'scheduler')}
    metadata = compare({k: a[k] for k in META_FIELDS}, {k: b[k] for k in META_FIELDS}, 'metadata', 0., 0.)
    del a, b
    for key, name in (('first_micro_forward', 'first_micro_forward.pth'),
                      ('second_update_gradients', 'second_update_gradients.pth')):
        a, b = load(left, name), load(right, name)
        sections[key] = summary(compare(a, b, key, **tolerance))
        del a, b
        gc.collect()
    return dict(left=left, right=right, sections=sections, metadata=summary(metadata),
        exact_inputs_rng_metadata=metadata['exact'] and all(sections[k]['exact'] for k in ('actual', 'rng', 'scheduler')),
        numerical_within_registered_tolerance=all(v['within_registered_tolerance'] for v in sections.values()))


def protected_files(root, project):
    paths = [root / name for name in ('request.json', 'contract.json', 'config.json',
             'protocol.json', 'source_snapshot.json', 'pipeline_status.json', 'failure.json')]
    for branch in (root / 'smoke/E0/baseline', root / 'smoke_replay/E0'):
        paths.extend(branch / 'epochs/01' / name for name in ('train_complete.json', 'resume.pth', 'detector.pth'))
    paths.extend(project / name for name in read(project / 'uecp_gspr_v2xr/PINS.json')['files'])
    paths.extend(p for p in (project / 'uecp_gspr_v2xr').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p): sha(p) for p in paths if p.is_file()}


def controller(root, project, archive, env):
    sys.path.insert(0, str(project))
    with stopped_pipeline(root):
        reviewed = diagnostic_sources(project, read(root / 'request.json')['sources'])
        before = protected_files(root, project)
        checkpoint_bytes = (root / 'smoke/E0/baseline/epochs/01/resume.pth').stat().st_size
        required = 24 * checkpoint_bytes + 1024**3
        if shutil.disk_usage(root).free < required:
            raise RuntimeError('Controlled probe needs at least '+str(required)+' free bytes')
        tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        probe = root / 'diagnostics' / ('controlled_resume_'+tag+'_'+str(os.getpid()))
        probe.mkdir(parents=True, exist_ok=False)
        write_new(probe / 'descriptor.json', dict(family='UECP_CONTROLLED_RESUME_DIAGNOSIS',
            original_output=str(root), source_review=reviewed, archive_sha256=sha(archive),
            original_request_sha256=sha(root / 'request.json'), original_contract_sha256=sha(root / 'contract.json'),
            original_tolerance=read(root / 'config.json')['resume_replay'], arms=list(ARMS),
            fixed_budget='continuous:2 updates; resume1/2/3:1 each; strict:at most1',
            ap_evaluation=False, automatic_policy_change=False, formal_recovery_gate_passed=False))
        print('PROBE_ROOT='+str(probe), flush=True)
        child = None
        def stop(signum, frame):
            raise InterruptedError('Controlled diagnosis interrupted by signal '+str(signum))
        old_signals = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            for arm in ARMS:
                overlay(root, probe, arm)
                log = probe / (arm+'.log')
                print('START '+arm+' LOG='+str(log), flush=True)
                with log.open('xb') as stream:
                    child = subprocess.Popen([sys.executable, '-B', str(archive), '--output', str(root),
                        '--project', str(project), '--probe', str(probe), '--worker', arm], env=env,
                        cwd=project, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    while True:
                        try:
                            code = child.wait(timeout=20)
                            break
                        except subprocess.TimeoutExpired:
                            print('RUNNING '+arm+' LOG='+str(log), flush=True)
                if code:
                    raise RuntimeError('Diagnostic worker failed: '+arm+'; see '+str(log))
                print('FINISHED '+arm, flush=True)
            import torch
            torch.set_num_threads(1)
            config = read(root / 'config.json')
            pairs = [compare_arms(probe, a, b, torch, config['resume_replay']) for a, b in (
                ('continuous', 'resume1'), ('continuous', 'resume2'), ('continuous', 'resume3'),
                ('resume1', 'resume2'), ('resume1', 'resume3'), ('resume2', 'resume3'))]
            restores = {arm: read(probe / 'evidence' / arm / 'restore.json') for arm in ARMS[1:]}
            boundary = read(probe / 'evidence/continuous/boundary.json')
            exact = boundary['exact'] and all(row['exact'] for row in restores.values())
            same_backends = all(row['backend'] == boundary['backend'] for row in restores.values())
            if protected_files(root, project) != before:
                raise ValueError('Protected original artifacts changed during the probe')
            report = dict(status='DIAGNOSIS_FINISHED', diagnosis=classify(pairs, exact)
                if same_backends else 'BACKEND_CONFIGURATION_DIFFERS', backend_flags_equal=same_backends,
                original_artifact_hashes_unchanged=True, formal_recovery_gate_passed=False,
                automatic_policy_change=False, common_boundary=boundary, restores=restores, comparisons=pairs,
                strict_worker=read(probe / 'evidence/strict/worker.json'),
                limitation='Three restarts measure a tiny fixed workload; they do not establish formal '
                           'statistical equivalence, long-training stability, or permission to relax tolerances.')
            write_new(probe / 'RESULTS.json', report)
            compact = dict(diagnosis=report['diagnosis'], exact_common_boundary_and_loads=exact,
                backend_flags_equal=same_backends,
                original_artifact_hashes_unchanged=True, formal_recovery_gate_passed=False,
                pairs=[dict(left=p['left'], right=p['right'], exact_inputs_rng_metadata=p['exact_inputs_rng_metadata'],
                    failed_arrays={k: v['failed_floating_arrays'] for k, v in p['sections'].items()}) for p in pairs],
                strict_status=report['strict_worker']['status'], strict_error=report['strict_worker'].get('error'),
                results=str(probe / 'RESULTS.json'))
            print('PROBE_SUMMARY_BEGIN\n'+json.dumps(compact, ensure_ascii=False, indent=2)+'\nPROBE_SUMMARY_END', flush=True)
            print('DIAGNOSIS_FINISHED '+str(probe / 'RESULTS.json'), flush=True)
        except BaseException:
            write_new(probe / 'ERROR.json', dict(traceback=traceback.format_exc(),
                original_artifact_hashes_unchanged=protected_files(root, project) == before,
                formal_recovery_gate_passed=False))
            raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            for sig, handler in old_signals.items():
                signal.signal(sig, handler)


def main():
    if sys.platform != 'linux':
        raise RuntimeError('Research probe runs only on the Linux server')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--project', default='/home/cjm/OpenCOOD-main/cjmnet')
    parser.add_argument('--worker', choices=ARMS)
    parser.add_argument('--probe')
    args = parser.parse_args()
    root, project = Path(args.output).resolve(strict=True), Path(args.project).resolve(strict=True)
    if any(p.lower() in ('test', 'testing') for p in root.parts):
        raise ValueError('Formal test path refused')
    archive = verify_archive()
    env = environment(root, project)
    if os.environ.get('PYTHONHASHSEED') != env['PYTHONHASHSEED']:
        os.execve(sys.executable, [sys.executable, '-B', str(archive), *sys.argv[1:]], env)
    os.environ.clear()
    os.environ.update(env)
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'):
        os.environ.pop(key, None)
    if args.worker:
        probe = Path(args.probe).resolve(strict=True)
        if probe.parent != root / 'diagnostics' or not probe.name.startswith('controlled_resume_'):
            raise ValueError('Worker probe directory is outside this diagnostic namespace')
        descriptor = read(probe / 'descriptor.json')
        if descriptor['archive_sha256'] != sha(archive) or descriptor['original_output'] != str(root):
            raise ValueError('Worker diagnostic identity differs')
        worker(root, project, probe, args.worker)
    else:
        if args.probe:
            raise ValueError('--probe belongs to diagnostic workers only')
        controller(root, project, archive, env)


if __name__ == '__main__':
    main()
