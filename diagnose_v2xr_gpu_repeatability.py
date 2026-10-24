"""Server-only fixed-input forward/backward probe. No optimizer update or gate change.

Kept outside the frozen experiment package so its source identity remains unchanged.
Local tests use only synthetic NumPy arrays and backend substitutes.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

import numpy as np

from v2xr_raygen_e0e3.common import ROOT, identity, read, safe, sha, write


def array_difference(a, b):
    if a is None or b is None:
        if a is not None or b is not None:
            raise ValueError('Used/unused parameter gradient changed')
        return dict(equal=True, elements=0, max_abs=0., relative_l2=0., rms_delta=0.)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('Gradient shape or dtype changed')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError('Nonfinite gradient in repeatability probe')
    x, y = a.astype(np.float64), b.astype(np.float64)
    delta = x - y
    norm = float(np.linalg.norm(delta.ravel()))
    scale = max(float(np.linalg.norm(x.ravel())), float(np.linalg.norm(y.ravel())))
    return dict(equal=bool(np.array_equal(a, b)), elements=int(a.size),
                max_abs=float(np.abs(delta).max()) if a.size else 0.,
                relative_l2=norm / max(scale, 1e-30), rms_delta=norm / math.sqrt(max(a.size, 1)))


def choose_batch(order, micro, supervised):
    if micro not in (1, 2, 4, 8) or len(order) % micro:
        raise ValueError('Expected complete committed microbatches')
    for batch_index, offset in enumerate(range(0, len(order), micro)):
        indices = order[offset:offset + micro]
        if any(supervised(index) for index in indices):
            return batch_index, indices
    raise ValueError('No supervised window in committed smoke batches; no auxiliary probe executed')


def backend_state(torch):
    return dict(enabled=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_enabled=torch.backends.cudnn.enabled,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark)


@contextlib.contextmanager
def backend_mode(torch, mode):
    before = backend_state(torch)
    try:
        if mode == 'prefer_deterministic':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
        elif mode != 'current':
            raise ValueError('Unknown probe mode')
        yield backend_state(torch)
    finally:
        torch.use_deterministic_algorithms(before['enabled'], warn_only=before['warn_only'])
        torch.backends.cudnn.enabled = before['cudnn_enabled']
        torch.backends.cudnn.deterministic = before['cudnn_deterministic']
        torch.backends.cudnn.benchmark = before['cudnn_benchmark']


def compare_gradient_maps(reference_gradients, gradients):
    if reference_gradients.keys() != gradients.keys():
        raise ValueError('Parameter registration changed')
    rows = [dict(name=name, **array_difference(reference_gradients[name], gradients[name])) for name in gradients]
    changed = [row for row in rows if not row['equal']]
    return dict(changed_gradient_tensors=len(changed),
                top_changed_gradients=sorted(changed, key=lambda row: (-row['rms_delta'], row['name']))[:12],
                all_gradient_differences=rows)


def crop_arguments(batch):
    """Match the owner/center order used by the frozen auxiliary_loss, including zero-weight windows."""
    pairs = [(owner, row['center']) for owner, rows in enumerate(batch['auxiliary']) for row in rows]
    if not pairs:
        raise ValueError('No crop windows in the selected batch')
    return [owner for owner, center in pairs], np.asarray([center for owner, center in pairs])


def compare_crop_trial(reference, current, reference_gradient, gradient):
    for key in ('input', 'owners', 'centers', 'upstream', 'rng_before', 'rng_after'):
        if reference[key] != current[key]:
            raise ValueError('Crop isolation controls drifted: ' + key)
    return dict(forward_equal=reference['output'] == current['output'],
                gradient=array_difference(reference_gradient, gradient))


def verify_prior_probe(prior, current):
    if prior.get('status') != 'complete':
        raise ValueError('The referenced prior probe is incomplete')
    for key in ('group', 'execution', 'sources', 'source_checkpoint_sha256', 'source_commit_sha256', 'selected_microbatch'):
        if prior.get(key) != current.get(key):
            raise ValueError('Prior probe identity differs: ' + key)


def compare_trials(reference, current, reference_gradients, gradients, target_name):
    if (reference['rng_before'] != current['rng_before'] or reference['rng_after'] != current['rng_after']
            or reference['batch_digest'] != current['batch_digest']):
        raise ValueError('Fixed input/RNG controls drifted during diagnosis')
    gradients_report = compare_gradient_maps(reference_gradients, gradients)
    rows = gradients_report['all_gradient_differences']
    captures = ('detector_outputs', 'auxiliary_output', 'fused_features', 'target_input', 'target_output',
                'detection_loss', 'generation_loss', 'joint_loss')
    if any('condition' in row['capture'] for row in (reference, current)):
        captures += ('condition',)
    if any(key not in row['capture'] for row in (reference, current) for key in captures):
        raise ValueError('A required forward observer did not execute')
    if reference_gradients[target_name] is None or gradients[target_name] is None:
        raise ValueError('The target parameter did not receive a gradient')
    forward = {key: reference['capture'][key] == current['capture'][key] for key in captures}
    def joint(capture, key):
        return [row['tensor'] for row in capture.get(key, []) if row['phase'] == 'joint_backward']
    upstream_a = joint(reference['capture'], 'target_output_gradients')
    upstream_b = joint(current['capture'], 'target_output_gradients')
    upstream_equal = bool(upstream_a) and upstream_a == upstream_b
    target = next(row for row in rows if row['name'] == target_name)
    if not all(forward.values()):
        observed = 'forward_bitwise_variation_observed'
    elif gradients_report['changed_gradient_tensors']:
        observed = 'backward_numeric_variation_observed'
    else:
        observed = 'no_numeric_variation_in_selected_probe'
    return dict(observed=observed, **gradients_report, target_gradient=target,
                forward_fields_equal=forward, target_upstream_gradient_equal=upstream_equal,
                target_weight_backward_candidate=bool(forward['target_input'] and forward['target_output']
                    and upstream_equal and not target['equal']))


def prepare_environment(root, gpu):
    config, request = read(root / 'config.json'), read(root / 'request.json')
    if os.environ.get('PYTHONHASHSEED') != str(config['seed']):
        raise ValueError('Start Python with PYTHONHASHSEED=' + str(config['seed']))
    for key, value in request['paths'].items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError('Path/environment differs from the frozen run: ' + key)
        os.environ[key] = value
    if Path(os.environ.get('PROJECT_ROOT', str(ROOT))).resolve() != ROOT:
        raise ValueError('Run the standalone probe beside the original frozen package')
    for key, value in dict(SEED=str(config['seed']), EPOCHS=str(config['epochs']), AMP='0').items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError('Frozen configuration differs: ' + key)
        os.environ[key] = value
    os.environ['GPU'] = gpu
    os.environ['ROCR_VISIBLE_DEVICES'] = gpu
    os.environ.pop('HIP_VISIBLE_DEVICES', None)
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)


def execute(root, folder, args, report):
    from v2xr_raygen_e0e3.common import verify_sources
    from v2xr_raygen_e0e3.resume_compat import effective_execution, verify_active_sources
    from v2xr_raygen_e0e3.runtime import bootstrap, input_stamps, audit_imports, hypes
    from v2xr_raygen_e0e3.training import completion, make_models, load_resume_state, restore_rng, rng_state, tensor_hash
    own = verify_sources()
    verify_active_sources(root, own)
    prior = None
    if args.prior_report:
        prior_path = safe(args.prior_report).resolve(strict=True)
        if root / 'diagnostics' not in prior_path.parents:
            raise ValueError('The prior report must belong to this run diagnostics directory')
        prior = read(prior_path)
        report['prior_report'] = dict(path=str(prior_path), sha256=sha(prior_path),
                                     diagnostic_source_sha256=prior.get('diagnostic_source_sha256'))
    print('核对冻结源码和输入身份；仅在独立诊断目录写报告。', flush=True)
    input_stamps(root)
    bootstrap(root, gpu=True)
    import torch
    from opencood.tools import train_utils
    from v2xr_raygen_e0e3.data import Dataset
    from v2xr_raygen_e0e3.cache import load_windows
    from v2xr_raygen_e0e3.geometry import loss_weights
    from v2xr_raygen_e0e3.model import FeatureTap, crop_condition
    from v2xr_raygen_e0e3.losses import auxiliary_loss
    from v2xr_raygen_e0e3.smoke import tensor_digest
    execution, config = effective_execution(root), read(root / 'config.json')
    source = root / 'smoke' / args.group / 'epochs/01'
    second = root / 'smoke' / args.group / 'epochs/02'
    for path in (source, second):
        committed = completion(root, path)
        if (committed['group'] != args.group or committed['epoch'] != int(path.name) or not committed['smoke_only']
                or committed['physical_gpu'] != args.gpu or committed['gpu_name'] != torch.cuda.get_device_name(0)):
            raise ValueError('Selected committed smoke/group/device differs')
    pristine = torch.load(source / 'resume.pth', map_location='cpu', weights_only=False)
    if (pristine['group'] != args.group or pristine['epoch'] != 1 or pristine['accumulation_position'] != 0
            or pristine['config'] != config or not pristine['completed_training']):
        raise ValueError('Invalid frozen epoch1 checkpoint')
    sampling = read(second / 'sampling.json')
    micro = execution['microbatch']
    if sampling['microbatch'] != micro or sampling['accumulation'] != 8 // micro:
        raise ValueError('Selected batch size differs from committed smoke')
    supervised = lambda index: any(loss_weights(args.group, row['target']).any()
                                   for row in load_windows(root, index, 2))
    batch_index, indices = choose_batch(sampling['used'], micro, supervised)
    restore_rng(pristine['rng'])
    dataset = Dataset(root, epoch=2, auxiliary=True)
    fixed = dataset.collate([dataset[index] for index in indices])
    recorded = read(second / 'actual_samples.json')
    input_digest = tensor_digest(dict(inputs=fixed['inputs'], labels=fixed['labels']))
    if (input_digest != recorded['smoke_input_and_target_digests'][batch_index]
            or fixed['audits'] != recorded['samples'][batch_index * micro:(batch_index + 1) * micro]):
            raise ValueError('Chosen inputs/labels do not match the committed smoke batch')
    full_digest = tensor_digest(fixed)
    detector, aux = make_models(root, args.group, config)
    detector, aux = detector.cuda(), aux.cuda()
    detector.train(); aux.train()
    criterion = train_utils.create_loss(hypes(root)).cuda()
    named = [('detector.' + name, parameter) for name, parameter in detector.named_parameters()]
    named += [('aux.' + name, parameter) for name, parameter in aux.named_parameters()]
    optimizer = torch.optim.Adam([parameter for name, parameter in named], **config['optimizer'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['scheduler'])
    ids = [value for group in pristine['optimizer']['param_groups'] for value in group['params']]
    if len(ids) != len(named) or args.param_id not in ids:
        raise ValueError('Checkpoint optimizer parameter registration differs')
    target_name, target_parameter = named[ids.index(args.param_id)]
    if not target_name.startswith('detector.'):
        raise ValueError('This probe targets the detector parameter named in the reported failure')
    for field in ('exp_avg', 'exp_avg_sq'):
        if pristine['optimizer']['state'][args.param_id][field].shape != target_parameter.shape:
            raise ValueError('Target parameter shape differs from saved Adam state')
    module_name = target_name[len('detector.'):].rsplit('.', 1)[0]
    module = detector.get_submodule(module_name)
    expected_weights = {name: pristine['model'][name[len('detector.'):]] if name.startswith('detector.')
                        else pristine['aux'][name[len('aux.'):]] for name, parameter in named}
    expected_weight_hash = tensor_hash(expected_weights)
    tap = FeatureTap(detector)
    capture, phase, tensor_handles = {}, ['forward'], []
    condition_holder, crop_anchor = [None], {}
    owners, centers = crop_arguments(fixed) if args.isolate_paths else (None, None)
    def fingerprint(value):
        if not isinstance(value, torch.Tensor):
            raise ValueError('Expected tensor at the target operator boundary')
        array = value.detach().cpu().contiguous().numpy()
        if not np.isfinite(array).all():
            raise FloatingPointError('Nonfinite operator tensor')
        return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest())
    def target_hook(layer, inputs, output):
        capture['target_input'], capture['target_output'] = fingerprint(inputs[0]), fingerprint(output)
        for key, value in (('target_input_gradients', inputs[0]), ('target_output_gradients', output)):
            if value.requires_grad:
                def observe(gradient, key=key):
                    capture.setdefault(key, []).append(dict(phase=phase[0], tensor=fingerprint(gradient)))
                tensor_handles.append(value.register_hook(observe))
    def aux_hook(layer, inputs, output):
        capture['auxiliary_output'] = fingerprint(output)
        if args.isolate_paths:
            condition_holder[0] = inputs[2]
            capture['condition'] = fingerprint(inputs[2])
    module_handle, aux_handle = module.register_forward_hook(target_hook), aux.register_forward_hook(aux_hook)
    report.update(status='running', sources=own, execution=execution,
        source_checkpoint_sha256=sha(source / 'resume.pth'), source_commit_sha256=sha(source / 'train_complete.json'),
        selected_microbatch=dict(index=batch_index, indices=indices, microbatch=micro, accumulation=8 // micro,
            audits=fixed['audits'], committed_input_digest=input_digest,
            policy='First committed smoke microbatch with at least one supervised window'),
        target=dict(optimizer_id=args.param_id, parameter=target_name, shape=list(target_parameter.shape),
            operator=type(module).__module__ + '.' + type(module).__name__, module=module_name),
        environment=dict(torch=torch.__version__, hip=torch.version.hip, python=sys.version,
            gpu=torch.cuda.get_device_name(0), physical_gpu=args.gpu, initial_backend=backend_state(torch),
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            threads={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            workspace={key: os.environ.get(key) for key in ('CUBLAS_WORKSPACE_CONFIG', 'HIPBLAS_WORKSPACE_CONFIG')}),
        imported_modules=audit_imports(root), modes=[])
    if prior is not None:
        verify_prior_probe(prior, report)
    print('TARGET=' + target_name + ' OPERATOR=' + report['target']['operator'] + ' SHAPE=' + str(list(target_parameter.shape)), flush=True)
    write(folder / 'report.json', report)
    def trial(mode, number):
        capture.clear()
        condition_holder[0] = None
        phase[0] = 'forward'
        started = time.monotonic()
        print(f'BEGIN {mode} trial={number} (0为预热；不更新权重)', flush=True)
        optimizer.zero_grad(set_to_none=True)
        _, checked = load_resume_state(detector, aux, criterion, optimizer, scheduler, copy.deepcopy(pristine))
        if tensor_hash(dict(named)) != expected_weight_hash:
            raise ValueError('Starting weights differ')
        batch_before = tensor_digest(fixed)
        if batch_before != full_digest:
            raise ValueError('Fixed host batch was mutated')
        rng_before = identity(rng_state())
        inputs = train_utils.to_device(fixed['inputs'], torch.device('cuda:0'))
        labels = train_utils.to_device(fixed['labels'], torch.device('cuda:0'))
        try:
            output = detector(inputs)
            capture['detector_outputs'] = tensor_digest(output)
            capture['fused_features'] = fingerprint(tap.value)
            detection = criterion(output, labels)
            generation, counts = auxiliary_loss(args.group, aux, tap.value, fixed, 2, config)
            if counts['valid_windows'] < 1:
                raise ValueError('Selected batch did not activate auxiliary supervision')
            diagnostics = None
            if batch_index == 0:
                shared = [p for name, p in detector.named_parameters() if p.requires_grad and
                          (name.startswith('pillar_vfe') or 'fuse_modules' in name)]
                phase[0] = 'diagnostic_detection_gradient'
                gd = torch.autograd.grad(detection, shared, retain_graph=True, allow_unused=True)
                phase[0] = 'diagnostic_generation_gradient'
                gg = torch.autograd.grad(generation, shared, retain_graph=True, allow_unused=True)
                phase[0] = 'diagnostic_feature_gradient'
                feature = torch.autograd.grad(generation, tap.value, retain_graph=True, allow_unused=True)[0]
                diagnostics = dict(
                    detection_norm=float(torch.sqrt(sum((g.square().sum() for g in gd if g is not None), detection.new_zeros(()))).item()),
                    generation_norm=float(torch.sqrt(sum((g.square().sum() for g in gg if g is not None), detection.new_zeros(()))).item()),
                    fused_gradient_norm=0. if feature is None else float(feature.norm().item()))
                del gd, gg, feature
            isolated = {}
            if args.isolate_paths:
                condition = condition_holder[0]
                if condition is None or not condition.requires_grad:
                    raise ValueError('The auxiliary condition boundary was not observed')
                active = [(name, p) for name, p in named if p.requires_grad]
                for path, objective in (('detection', detection), ('generation', generation)):
                    phase[0] = 'isolate_' + path
                    boundaries = [tap.value] + ([condition] if path == 'generation' else [])
                    raw = torch.autograd.grad(objective, [p for name, p in active] + boundaries,
                                              retain_graph=True, allow_unused=True)
                    values = {name: None for name, p in named}
                    for (name, p), value in zip(active, raw[:len(active)]):
                        values[name] = None if value is None else value.detach().cpu().numpy().copy()
                    for key, value in zip(('boundary.fused_features', 'boundary.condition'), raw[len(active):]):
                        if value is None:
                            raise ValueError('An isolated loss did not reach ' + key)
                        values[key] = value.detach().cpu().numpy().copy()
                    for value in values.values():
                        array_difference(value, value)
                    isolated[path] = values
                    if path == 'generation' and number == 0:
                        crop_anchor.update(bev=tap.value.detach().cpu(), upstream=raw[-1].detach().cpu(),
                            expected_output=capture['condition'],
                            original_bev_stride=list(tap.value.stride()),
                            original_upstream_stride=list(raw[-1].stride()))
                    del raw
            phase[0] = 'joint_backward'
            loss = (detection + config['lambda_gen'] * generation) / (8 // micro)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            capture.update(detection_loss=fingerprint(detection), generation_loss=fingerprint(generation),
                           joint_loss=fingerprint(loss))
            loss.backward()
            torch.cuda.synchronize()
            gradients = {name: None if p.grad is None else p.grad.detach().cpu().numpy().copy() for name, p in named}
            for gradient in gradients.values():
                array_difference(gradient, gradient)
            if tensor_hash(dict(named)) != expected_weight_hash:
                raise ValueError('Probe unexpectedly updated model parameters')
            if tensor_digest(fixed) != full_digest:
                raise ValueError('Input batch changed during forward/backward')
            row = dict(trial=number, seconds=time.monotonic() - started, exact_restore=checked,
                batch_digest=batch_before, rng_before=rng_before, rng_after=identity(rng_state()),
                loss=dict(detection=float(detection.detach()), generation=float(generation.detach()), joint=float(loss.detach())),
                supervision=counts, step0_gradient_diagnostics=diagnostics, capture=copy.deepcopy(capture),
                weights_unchanged=True)
            print(f'END {mode} trial={number} seconds={row["seconds"]:.1f} loss={row["loss"]["joint"]:.8g}', flush=True)
            return row, dict(joint=gradients, paths=isolated)
        finally:
            tap.clear()
            condition_holder[0] = None
            for handle in tensor_handles:
                handle.remove()
            tensor_handles.clear()

    def crop_trials(mode, entry):
        """Replay the unchanged crop block after the model graphs have been freed, with fixed upstream."""
        optimizer.zero_grad(set_to_none=True)
        probe = dict(status='running', scope='crop_condition: owner indexing plus bilinear grid_sample',
                     expected_output=crop_anchor['expected_output'], trials=[],
                     original_bev_stride=crop_anchor['original_bev_stride'],
                     original_upstream_stride=crop_anchor['original_upstream_stride'])
        entry['crop_isolation'] = probe
        bev = crop_anchor['bev'].to('cuda:0').requires_grad_(True)
        upstream = crop_anchor['upstream'].to('cuda:0')
        owner_tensor = torch.as_tensor(owners, device=bev.device)
        controls = dict(input=fingerprint(bev), owners=fingerprint(owner_tensor),
                        centers=identity(centers), upstream=fingerprint(upstream))
        probe.update(controls=controls, probe_bev_stride=list(bev.stride()),
                     probe_upstream_stride=list(upstream.stride()))
        write(folder / 'report.json', report)
        baseline_gradient, baseline_row = None, None
        for number in range(args.repeats + 1):
            print(f'CROP_BEGIN {mode} trial={number} (0为预热；固定输入和上游梯度)', flush=True)
            started = time.monotonic()
            rng_before = identity(rng_state())
            result = crop_condition(bev, owner_tensor, centers)
            output_identity = fingerprint(result)
            if output_identity != crop_anchor['expected_output']:
                raise ValueError('Isolated crop forward differs from the captured real crop')
            gradient = torch.autograd.grad(result, bev, grad_outputs=upstream)[0]
            current = gradient.detach().cpu().numpy().copy()
            array_difference(current, current)
            row = dict(trial=number, **controls, rng_before=rng_before, rng_after=identity(rng_state()),
                       output=output_identity, seconds=time.monotonic() - started)
            if (fingerprint(bev) != controls['input'] or fingerprint(upstream) != controls['upstream']
                    or fingerprint(owner_tensor) != controls['owners'] or identity(centers) != controls['centers']):
                raise ValueError('Fixed crop tensors were mutated')
            if number > 0:
                probe['trials'].append(row)
                if number == 1:
                    baseline_gradient, baseline_row = current, row
                else:
                    row['comparison'] = compare_crop_trial(baseline_row, row, baseline_gradient, current)
            print(f'CROP_END {mode} trial={number} seconds={row["seconds"]:.1f}', flush=True)
            write(folder / 'report.json', report)
            del result, gradient
        probe['status'] = 'complete'
    try:
        for mode in ('current', 'prefer_deterministic'):
            crop_anchor.clear()
            entry = dict(mode=mode, status='running', trials=[])
            report['modes'].append(entry)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    with backend_mode(torch, mode) as backend:
                        entry['backend'] = backend
                        trial(mode, 0)
                        baseline, baseline_gradients = trial(mode, 1)
                        entry['trials'].append(baseline)
                        write(folder / 'report.json', report)
                        for number in range(2, args.repeats + 1):
                            row, gradients = trial(mode, number)
                            entry['trials'].append(row)
                            row['comparison'] = compare_trials(baseline, row, baseline_gradients['joint'],
                                                               gradients['joint'], target_name)
                            if args.isolate_paths:
                                row['path_comparisons'] = {path: compare_gradient_maps(baseline_gradients['paths'][path],
                                    gradients['paths'][path]) for path in ('detection', 'generation')}
                            print('OBSERVED ' + mode + ' ' + row['comparison']['observed'] + ' TARGET=' +
                                  str(row['comparison']['target_gradient']), flush=True)
                            write(folder / 'report.json', report)
                        del baseline_gradients
                        if args.isolate_paths:
                            crop_trials(mode, entry)
                    entry['status'] = 'complete'
                except (RuntimeError, ValueError, FloatingPointError) as error:
                    entry.update(status='probe_error', error=str(error), traceback=traceback.format_exc())
                    if entry.get('crop_isolation', {}).get('status') == 'running':
                        entry['crop_isolation'].update(status='probe_error', error=str(error))
                    print('MODE_ERROR ' + mode + ': ' + str(error), flush=True)
                finally:
                    crop_anchor.clear()
                    entry['warnings'] = sorted({str(row.message) for row in caught})
                    entry['backend_restored'] = backend_state(torch) == report['environment']['initial_backend']
                    write(folder / 'report.json', report)
        if verify_sources() != own or sha(source / 'resume.pth') != report['source_checkpoint_sha256']:
            raise ValueError('Source/checkpoint changed during diagnosis')
        if prior is not None and sha(prior_path) != report['prior_report']['sha256']:
            raise ValueError('The prior report changed during diagnosis')
        if sha(Path(__file__)) != report['diagnostic_source_sha256']:
            raise ValueError('The standalone diagnostic source changed during diagnosis')
        report['status'] = 'complete' if all(row['status'] == 'complete' for row in report['modes']) else 'partial'
    finally:
        module_handle.remove(); aux_handle.remove(); tap.handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', required=True, choices=[str(index) for index in range(64)])
    parser.add_argument('--group', choices=['E1', 'E2', 'E3'], default='E2')
    parser.add_argument('--param-id', type=int, default=66)
    parser.add_argument('--repeats', type=int, choices=[3, 4, 5], default=3)
    parser.add_argument('--isolate-paths', action='store_true',
                        help='Also compare unweighted detection/generation gradients and replay the fixed crop block')
    parser.add_argument('--prior-report', help='Bind the same source/checkpoint/microbatch to an existing completed probe')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Real inputs/Torch/OpenCOOD/GPU diagnostics must run on the server')
    root = safe(args.output).resolve(strict=True)
    if Path('/data') not in root.parents or root == ROOT or ROOT in root.parents:
        parser.error('Use the existing separate accepted run directory on /data')
    prepare_environment(root, args.gpu)
    import fcntl
    with contextlib.ExitStack() as stack:
        for name, kind in (('.controller.lock', fcntl.LOCK_SH), ('.worker.lock', fcntl.LOCK_EX)):
            stream = stack.enter_context((root / name).open('rb'))
            try:
                fcntl.flock(stream, kind | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Run/worker is still active; wait for this task to exit before the probe') from None
        prefix = 'gpu_paths_' if args.isolate_paths else 'gpu_repeatability_'
        folder = root / 'diagnostics' / (prefix + args.group + '_' + str(time.time_ns()))
        folder.mkdir(parents=True, exist_ok=False)
        report = dict(status='starting', diagnostic_source_sha256=sha(Path(__file__)),
            output=str(folder), group=args.group, repeats=args.repeats, started=time.time(),
            optimizer_updates=0, experiment_gate_changed=False, original_run_status_changed=False,
            isolate_paths=args.isolate_paths,
            limitations=['One process and one selected committed microbatch; not the full two-update trajectory.',
                'Forward/backward observers copy tensors to CPU and can affect timing; a negative probe does not exclude other variation.',
                'prefer_deterministic uses warn_only=True; unsupported operators can remain nondeterministic.',
                'Operator-boundary evidence suggests a location, not a vendor-library fault proof.',
                'No run is approved or resumed by this diagnostic.'])
        if args.isolate_paths:
            report['limitations'] += ['Split-path probes add autograd calls to the original step0/joint diagnostics.',
                'Detection/generation path gradients are unweighted and unscaled; joint gradients retain original scaling.',
                'The crop replay covers owner-index accumulation and grid_sample together, not either kernel in isolation.',
                'Crop anchors are captured once per mode; comparisons are only within each mode.']
        print('REPORT=' + str(folder / 'report.json'), flush=True)
        write(folder / 'report.json', report)
        try:
            execute(root, folder, args, report)
        except BaseException:
            report.update(status='failed', traceback=traceback.format_exc())
            raise
        finally:
            report['ended'] = time.time()
            write(folder / 'report.json', report)
        print('STATUS=' + report['status'], flush=True)
        for mode in report['modes']:
            comparisons = [row['comparison'] for row in mode['trials'] if 'comparison' in row]
            summary = dict(mode=mode['mode'], status=mode['status'], warnings=mode.get('warnings', []),
                observed=[row['observed'] for row in comparisons],
                target_gradient=[row['target_gradient'] for row in comparisons],
                target_weight_backward_candidate=any(row['target_weight_backward_candidate'] for row in comparisons))
            print('SUMMARY=' + json.dumps(summary, ensure_ascii=False), flush=True)
            if args.isolate_paths:
                for path in ('detection', 'generation'):
                    runs = []
                    for trial_row in mode['trials']:
                        if 'path_comparisons' not in trial_row:
                            continue
                        c = trial_row['path_comparisons'][path]
                        runs.append(dict(trial=trial_row['trial'],
                            forward_equal=all(trial_row['comparison']['forward_fields_equal'].values()),
                            changed_tensors=c['changed_gradient_tensors'],
                            target=next(row for row in c['all_gradient_differences'] if row['name'] == report['target']['parameter']),
                            boundaries=[row for row in c['all_gradient_differences'] if row['name'].startswith('boundary.')],
                            top_changed=c['top_changed_gradients'][:5]))
                    print('PATH_SUMMARY=' + json.dumps(dict(mode=mode['mode'], path=path,
                          status=mode['status'], runs=runs), ensure_ascii=False), flush=True)
                probe = mode.get('crop_isolation', {})
                print('CROP_SUMMARY=' + json.dumps(dict(mode=mode['mode'], status=probe.get('status', 'not_run'),
                      scope=probe.get('scope'), comparisons=[row['comparison'] for row in probe.get('trials', [])
                      if 'comparison' in row]), ensure_ascii=False), flush=True)
        print('诊断结束；实验门禁和原有权重保持原样。请回传SUMMARY/路径摘要及MODE_ERROR行。', flush=True)


if __name__ == '__main__':
    main()
