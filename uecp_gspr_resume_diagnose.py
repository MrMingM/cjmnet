"""Server diagnosis of saved UECP smoke checkpoints; never repairs/passes a run.

This standalone tool lives outside the sealed package. It reads existing artifacts,
Default comparison is CPU-only. --restore-load additionally checks the actual loader
on the original HIP device, without a forward/backward pass or optimizer update.
Local verification is limited to NumPy fixtures and static compilation.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


KNOWN_TRAINING_PATH = 'v2xr_raygen_e0e3/training.py'
ORIGINAL_TRAINING_SHA256 = 'b6577e83734086bc760cde237eb99c9086ac2a6d514e3fa0bf1ae7a44ba6c129'
REVIEWED_TRAINING_SHA256 = 'cc5f523e64cb34814208a5901ef30be45712701ef0f00a0e4892accf9a683793'
HELPERS = ('tensor_hash', 'atomic_torch', 'rng_state', 'restore_rng')
# Copied from the byte-verified original delivery ZIP. Parsed on the server's own
# Python version so AST representation differences across Python versions are harmless.
ORIGINAL_HELPER_SOURCE = '''
"""One training epoch per process. Evaluation starts only after this process releases its GPU."""
import hashlib
import os
from pathlib import Path
import random
import time

import numpy as np

from .common import atomic_bytes, bound, identity, progress, read, sha, stream_seed, write


def tensor_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(k.encode()); h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def atomic_torch(path, value):
    import torch
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    torch.save(value, tmp)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, p)


def rng_state():
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                hip=torch.cuda.get_rng_state_all())


def restore_rng(rng):
    import torch
    random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state_all(rng['hip'])
'''


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def review_helpers(source):
    """Check all reused function bodies and every module-level statement without importing them."""
    current, original = ast.parse(source), ast.parse(ORIGINAL_HELPER_SOURCE)
    def components(tree):
        return ({n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)},
                [ast.dump(n) for n in tree.body if not isinstance(n, ast.FunctionDef)])
    functions, top_level = components(current)
    expected, expected_top_level = components(original)
    if top_level != expected_top_level or any(functions.get(k) != expected[k] for k in HELPERS):
        raise ValueError('Reused training helper or module-level statement differs from original delivery')
    return dict(helpers=list(HELPERS), all_helper_bodies_unchanged=True, module_level_statements_unchanged=True)


def diagnostic_sources(project, recorded):
    """Strict diagnosis-only source check, with one byte-identified reviewed legacy update.

    This does not call/replace/modify the experiment's sources() gate or change any
    recorded expected hash. Unknown changes still fail before model construction.
    """
    here = project / 'uecp_gspr_v2xr'
    expected_own = read(here / 'source_snapshot.json')['files']
    actual_own = {p.relative_to(here).as_posix(): sha(p) for p in here.rglob('*')
                  if p.is_file() and p.name != 'source_snapshot.json' and '__pycache__' not in p.parts}
    if actual_own != expected_own or expected_own != recorded['own']:
        raise ValueError('Sealed UECP package differs from original experiment sources')
    pins = read(here / 'PINS.json')['files']
    if pins != recorded['dependencies']:
        raise ValueError('Recorded dependency pins were changed; diagnosis refused')
    differences, failures = [], []
    for name, digest in pins.items():
        path = project / name
        try:
            current = sha(path)
        except OSError as exc:
            failures.append(name+': '+str(exc))
            continue
        if current == digest:
            continue
        if (name == KNOWN_TRAINING_PATH and digest == ORIGINAL_TRAINING_SHA256
                and current == REVIEWED_TRAINING_SHA256):
            reviewed = review_helpers(path.read_text(encoding='utf-8'))
            differences.append(dict(path=name, original_sha256=digest, current_sha256=current,
                scope='DIAGNOSIS_ONLY; original pipeline gate and request remain unchanged', **reviewed))
        else:
            failures.append(name+': expected='+digest+' actual='+current)
    if failures:
        raise ValueError('Unreviewed source differences; diagnosis refused:\n'+'\n'.join(failures))
    return dict(sealed_package_exact=True, original_pins_unchanged=True, dependencies_checked=len(pins),
                diagnosis_only_reviewed_differences=differences, original_pipeline_gate_modified=False)


def restore_block(source):
    """Extract the existing training loader; fail if its reviewed structure changes."""
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    candidates = [n for n in function.body if isinstance(n, ast.If)
                  and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                          and isinstance(c.func.value, ast.Name) and c.func.value.id == 'model'
                          and c.func.attr == 'load_state_dict' for c in ast.walk(n))]
    if len(candidates) != 1:
        raise ValueError('Cannot identify one existing checkpoint restore block')
    block = candidates[0]
    if ast.dump(block.test) != ast.dump(ast.parse('restored is not None', mode='eval').body):
        raise ValueError('Existing checkpoint restore condition changed')
    expected = ast.parse('''
model.load_state_dict(restored['model'], strict=True)
opt.load_state_dict(restored['optimizer'])
scheduler.load_state_dict(restored['scheduler'])
crit.load_state_dict(restored['criterion'], strict=True)
global_step = restored['global_step']
if restored['epoch'] == epoch:
    position, totals, elapsed = restored['position'], restored['totals'], restored['elapsed']
    actual, gradients, distributions = restored['actual'], restored['gradients'], restored['distributions']
restore_rng(restored['rng'])
del restored
''')
    selected = ast.Module(body=block.body, type_ignores=[])
    if ast.dump(selected) != ast.dump(expected):
        raise ValueError('Training restore block differs from reviewed loader; inspect before running')
    return compile(ast.fix_missing_locations(selected), '<existing train.run restore block>', 'exec')


@contextlib.contextmanager
def stopped_pipeline(root):
    """Acquire existing kernel locks without rewriting lock files or PID records."""
    import fcntl
    with contextlib.ExitStack() as stack:
        for name in ('.controller.lock', '.worker.lock'):
            stream = stack.enter_context((root / name).open('r'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Pipeline/worker is active; stopped state required for diagnosis') from exc
        yield


def restore_load(root, project, paths, commits, torch, numpy_tree):
    """Audit freshly constructed live objects using the exact existing restore statements."""
    sys.path.insert(0, str(project))
    request = read(root / 'request.json')
    if request.get('family') != 'uecp_gspr_v2xr':
        raise ValueError('Output does not belong to this experiment')
    reviewed_sources = diagnostic_sources(project, request['sources'])
    from uecp_gspr_v2xr.dependency_scope import validate_scope
    validate_scope(project, request['sources']['dependencies'])
    from uecp_gspr_v2xr.runtime import bootstrap
    bootstrap(root, gpu=True)
    if not torch.version.hip or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one original HIP device must be selected')
    original_gpu = str(commits[0]['physical_gpu'])
    if os.environ.get('GPU') != original_gpu or os.environ.get('ROCR_VISIBLE_DEVICES') != original_gpu:
        raise ValueError('Explicit GPU selection differs from original smoke device')
    if any(c['physical_gpu'] != commits[0]['physical_gpu'] or
           c['gpu_name'] != torch.cuda.get_device_name(0) for c in commits):
        raise ValueError('Original/replay GPU identity differs')
    free, total = torch.cuda.mem_get_info()
    if free / total < .9:
        raise RuntimeError('Original GPU has less than 90% free memory; no objects constructed')

    from uecp_gspr_v2xr import train
    from uecp_gspr_v2xr.model_adapter import make, criterion
    from uecp_gspr_v2xr.state_logic import verify_resume
    loader_path = project / 'uecp_gspr_v2xr/train.py'
    code = restore_block(loader_path.read_text(encoding='utf-8'))
    reports = []
    for path in paths:
        print('RESTORE_LOAD_START: '+str(path), flush=True)
        saved = torch.load(path / 'resume.pth', map_location='cpu', weights_only=False)
        order = read(path / 'sampling.json')['order']
        verify_resume(saved, read(root / 'contract.json')['identity'], 'E0', saved['epoch'],
                      read(root / 'config.json')['microbatch'], order)
        keys = ('model', 'optimizer', 'scheduler', 'criterion', 'rng', 'global_step',
                'position', 'totals', 'elapsed', 'actual', 'gradients', 'distributions')
        expected = copy.deepcopy(numpy_tree({k: saved[k] for k in keys}))
        model = make(root, 'E0').cuda()
        crit = criterion(root).cuda()
        config = read(root / 'config.json')
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], **config['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, **config['scheduler'])
        env = dict(model=model, opt=opt, scheduler=scheduler, crit=crit,
                   restored=saved, epoch=saved['epoch'], restore_rng=train.restore_rng)
        # The compiled statements come from the hash-verified train.py; no training loop runs.
        exec(code, env)
        live = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                    criterion=crit.state_dict(), rng=train.rng_state(),
                    **{k: env[k] for k in keys if k not in ('model', 'optimizer', 'scheduler', 'criterion', 'rng')})
        result = compare(expected, numpy_tree(live), 'post_load', atol=0., rtol=0.)
        reports.append(dict(checkpoint=str(path / 'resume.pth'), exact=result['exact'],
            structural_or_exact_errors=result['structural_or_exact_errors'],
            changed_floating_arrays=result['changed_floating_arrays'],
            floating_arrays=result['floating_arrays']))
        del model, crit, opt, scheduler, env, saved, expected, live
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print('RESTORE_LOAD_END: exact='+str(result['exact']), flush=True)
    return dict(status='EXACT' if all(r['exact'] for r in reports) else 'DIFFERENT',
        loader_source_sha256=sha(loader_path), extracted_original_loader=True, checkpoints=reports,
        source_review=reviewed_sources,
        limitation='Checks existing epoch-end checkpoint loading only. Does not reproduce the original '
                   'mid-epoch interruption, run another optimizer step, or pass the failed replay gate.')


def compare(left, right, prefix, atol, rtol):
    """Compare NumPy trees without stopping at the first floating-point mismatch."""
    errors, floating = [], []

    def walk(x, y, path):
        if isinstance(x, dict):
            if not isinstance(y, dict) or x.keys() != y.keys():
                errors.append(dict(path=path, reason='dictionary keys differ'))
                return
            for key in x:
                walk(x[key], y[key], path+'/'+str(key))
        elif isinstance(x, (list, tuple)):
            if type(x) is not type(y) or len(x) != len(y):
                errors.append(dict(path=path, reason='sequence type/length differs'))
                return
            for index, (a, b) in enumerate(zip(x, y)):
                walk(a, b, path+'/'+str(index))
        elif isinstance(x, np.ndarray) or isinstance(x, (float, np.floating)):
            a, b = np.asarray(x), np.asarray(y)
            if a.shape != b.shape or a.dtype != b.dtype:
                errors.append(dict(path=path, reason='array shape/dtype differs',
                    left_shape=list(a.shape), right_shape=list(b.shape), left_dtype=str(a.dtype), right_dtype=str(b.dtype)))
                return
            if not np.issubdtype(a.dtype, np.floating):
                if not np.array_equal(a, b):
                    errors.append(dict(path=path, reason='integer/bool array differs'))
                return
            finite_a, finite_b = np.isfinite(a), np.isfinite(b)
            finite = bool(finite_a.all() and finite_b.all())
            if not finite:
                errors.append(dict(path=path, reason='nonfinite values',
                    nonfinite_left=int((~finite_a).sum()), nonfinite_right=int((~finite_b).sum())))
                return
            a, b = a.astype(np.float64), b.astype(np.float64)
            delta = np.abs(a-b)
            allowed = atol+rtol*np.abs(b)
            count = int((delta > allowed).sum())
            norm_a, norm_b = float(np.linalg.norm(a.ravel())), float(np.linalg.norm(b.ravel()))
            difference_norm = float(np.linalg.norm((a-b).ravel()))
            floating.append(dict(path=path, shape=list(a.shape), elements=int(a.size),
                exactly_equal=bool(np.array_equal(a, b)), mismatched_elements=count,
                maximum_absolute_difference=float(delta.max()) if delta.size else 0.,
                relative_l2=difference_norm/max(norm_a, norm_b, 1e-30),
                left_l2=norm_a, right_l2=norm_b,
                left_scalar=float(a) if a.ndim == 0 else None,
                right_scalar=float(b) if b.ndim == 0 else None))
        elif type(x) is not type(y) or x != y:
            errors.append(dict(path=path, reason='exact scalar/type differs', left=repr(x), right=repr(y)))

    walk(left, right, prefix)
    failed = [r for r in floating if r['mismatched_elements']]
    return dict(exact=not errors and all(r['exactly_equal'] for r in floating),
        within_registered_tolerance=not errors and not failed,
        structural_or_exact_errors=errors, floating_arrays=len(floating),
        failed_floating_arrays=failed,
        changed_floating_arrays=[r for r in floating if not r['exactly_equal']])


def committed_epoch(path, epoch):
    commit = read(path / 'train_complete.json')
    if commit['status'] != 'COMPLETE' or not commit['smoke_only'] or commit['variant'] != 'E0' or commit['epoch'] != epoch:
        raise ValueError('Not a completed E0 smoke checkpoint: '+str(path))
    for name, expected in commit['output_hashes'].items():
        if sha(path / name) != expected:
            raise ValueError('Committed artifact bytes changed: '+str(path / name))
    return commit


def run(output, epoch, project=None, audit_restore=False):
    if sys.platform != 'linux':
        raise RuntimeError('Saved research checkpoints are read only on the Linux server')
    import torch
    torch.set_num_threads(1)
    root = Path(output).resolve(strict=True)
    if any(part.lower() in ('test', 'testing') for part in root.parts):
        raise ValueError('Formal test path refused')
    original = root / 'smoke/E0/baseline'
    replay = root / 'smoke_replay/E0'
    paths = [branch / 'epochs' / f'{epoch:02d}' for branch in (original, replay)]
    commits = [committed_epoch(p, epoch) for p in paths]
    states = [torch.load(p / 'resume.pth', map_location='cpu', weights_only=False) for p in paths]
    config = read(root / 'config.json')
    tolerance = config['resume_replay']

    def numpy_tree(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return {k: numpy_tree(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(numpy_tree(v) for v in value)
        return value

    states = [numpy_tree(s) for s in states]
    fields = ('contract_identity', 'variant', 'epoch', 'position', 'global_step',
              'order_identity', 'microbatch', 'accumulation_position')
    metadata = compare({k: states[0][k] for k in fields}, {k: states[1][k] for k in fields}, 'metadata', **tolerance)
    sections = {key: compare(states[0][key], states[1][key], key, **tolerance)
                for key in ('actual', 'rng', 'scheduler', 'model', 'optimizer', 'totals', 'gradients')}
    summaries = {key: {k: v for k, v in value.items() if k not in ('changed_floating_arrays', 'failed_floating_arrays')}
                 for key, value in sections.items()}
    for key, value in summaries.items():
        value['failed_floating_arrays'] = len(sections[key]['failed_floating_arrays'])
        value['changed_floating_arrays'] = len(sections[key]['changed_floating_arrays'])
    model_changes = sorted(sections['model']['changed_floating_arrays'],
        key=lambda row: (-row['relative_l2'], -row['maximum_absolute_difference']))
    optimizer_changes = sorted(sections['optimizer']['changed_floating_arrays'],
        key=lambda row: (-row['relative_l2'], -row['maximum_absolute_difference']))
    same_device = all(commits[0].get(k) == commits[1].get(k) for k in ('gpu_name', 'physical_gpu'))
    schedules = {name: read(paths[0] / name) == read(paths[1] / name)
                 for name in ('sampling.json', 'actual_samples.json')}
    report = dict(status='DIAGNOSIS_ONLY', formal_recovery_gate_passed=False,
        checkpoint_commits=[{k: c[k] for k in ('variant', 'epoch', 'contract_identity', 'global_step',
            'gpu_name', 'physical_gpu', 'final_tensor_sha256')} for c in commits],
        checkpoint_commit_sha256=[sha(p / 'train_complete.json') for p in paths],
        verified_checkpoint_artifact_hashes=True,
        saved_contract_matches=all(s['contract_identity'] == read(root / 'contract.json')['identity'] for s in states),
        saved_initialization_equal=read(original / 'initialization.json') == read(replay / 'initialization.json'),
        same_device=same_device, schedules_equal=schedules, registered_tolerance=tolerance,
        metadata=metadata, sections=summaries,
        first_failed_model_array=next(iter(sections['model']['failed_floating_arrays']), None),
        largest_model_changes=model_changes[:5], largest_optimizer_changes=optimizer_changes[:5],
        first_update_gradient_norm_changes=sections['gradients'],
        note='No GPU/data-loader execution, parameter update, checkpoint rewrite, or tolerance change. '
             'First-update gradient norms were recorded before the injected interruption. '
             'Their difference with identical initialization/inputs is evidence of an earlier divergence, '
             'but does not by itself certify that checkpoint restore is correct.',
        environment=dict(python=sys.version, torch=torch.__version__, hip=torch.version.hip))
    report['diagnostic_source_sha256'] = sha(Path(__file__))
    if audit_restore:
        if project is None:
            raise ValueError('--project is required with --restore-load')
        project = Path(project).resolve(strict=True)
        protected = [root / name for name in ('request.json', 'contract.json', 'config.json',
                     'pipeline_status.json', 'failure.json') if (root / name).exists()]
        protected += [p / name for p in paths for name in ('train_complete.json', 'resume.pth', 'detector.pth')]
        protected += list((project / 'uecp_gspr_v2xr').glob('*'))
        protected += [project / name for name in read(project / 'uecp_gspr_v2xr/PINS.json')['files']]
        protected = [p for p in protected if p.is_file()]
        try:
            with stopped_pipeline(root):
                before = {str(p): sha(p) for p in protected}
                try:
                    report['restore_load_audit'] = restore_load(root, project, paths, commits, torch, numpy_tree)
                finally:
                    unchanged = before == {str(p): sha(p) for p in protected}
                    report['protected_artifact_hashes_unchanged'] = unchanged
                    if not unchanged:
                        raise RuntimeError('Protected experiment/source artifact changed during audit')
        except Exception as exc:
            report['restore_load_audit'] = dict(status='ERROR', error_type=type(exc).__name__, error=str(exc))
        report['note'] += ' Optional restore audit constructs HIP objects only; no forward/backward/update.'
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print('EVIDENCE_SUMMARY_BEGIN', flush=True)
    compact = dict(initialization_equal=report['saved_initialization_equal'],
        same_device=report['same_device'], schedules_equal=schedules, metadata_exact=metadata['exact'],
        sections=report['sections'], first_failed_model_array=report['first_failed_model_array'],
        largest_model_changes=report['largest_model_changes'],
        largest_optimizer_changes=report['largest_optimizer_changes'],
        first_update_gradient_norm_changes=report['first_update_gradient_norm_changes']['changed_floating_arrays'],
        restore_load_audit=report.get('restore_load_audit', dict(status='NOT_REQUESTED')),
        protected_artifact_hashes_unchanged=report.get('protected_artifact_hashes_unchanged'))
    # One JSON line keeps a complete pasteable summary within tail -n 30.
    print(json.dumps(compact, ensure_ascii=False, allow_nan=False), flush=True)
    print('EVIDENCE_SUMMARY_END', flush=True)
    if audit_restore:
        print('RESTORE_RESULT_BEGIN', flush=True)
        print(json.dumps(report['restore_load_audit'], ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        print('RESTORE_RESULT_END', flush=True)
    print('DIAGNOSIS_FINISHED: existing FAILED recovery gate was retained.', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='uecp-resume-diagnose 2026-10-20.2')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--project', help='Existing server project containing the sealed package')
    parser.add_argument('--restore-load', action='store_true', help='Also audit exact original HIP checkpoint loader')
    args = parser.parse_args()
    # Select the recorded physical device before importing torch, with no GPU guessed.
    if args.restore_load:
        root = Path(args.output)
        saved_commit = read(root / 'smoke/E0/baseline/epochs' / f'{args.epoch:02d}' / 'train_complete.json')
        gpu = str(saved_commit['physical_gpu'])
        if not gpu.isdecimal():
            raise ValueError('Invalid original physical GPU identifier')
        os.environ.pop('HIP_VISIBLE_DEVICES', None)
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        os.environ['GPU'] = os.environ['ROCR_VISIBLE_DEVICES'] = gpu
    result = run(args.output, args.epoch, args.project, args.restore_load)
    if args.restore_load and result['restore_load_audit']['status'] != 'EXACT':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
