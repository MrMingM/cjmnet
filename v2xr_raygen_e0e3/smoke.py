"""Server-only real-interface checks; local tests never import this module."""
import copy
from pathlib import Path
import random

import numpy as np

from .common import bound, identity, read, stream_seed, write


def tensor_digest(value):
    from .training import tensor_hash
    import torch
    fields = {}
    def walk(x, key):
        if isinstance(x, dict):
            for k in sorted(x):
                walk(x[k], key + '/' + str(k))
        elif isinstance(x, (tuple, list)):
            for i, v in enumerate(x):
                walk(v, key + '/' + str(i))
        elif isinstance(x, np.ndarray):
            fields[key] = torch.from_numpy(x.copy())
        elif isinstance(x, torch.Tensor):
            fields[key] = x
    walk(value, '')
    return tensor_hash(fields)


def prepare(output):
    from .cache import build, prepare_windows
    root = Path(output)
    m = read(root / 'preflight/train_manifest.json')
    c = read(root / 'config.json')
    sizes = {r['relative']: r['size'] for r in m['files']}
    weights, keys = [], []
    for s in m['scenes']:
        for t in s['timestamps']:
            keys.append([s['scene'], t])
            weights.append(sum(sizes[s['scene'] + '/' + a + '/' + t + suffix + '.pcd']
                               for a in s['agents'][:7] for suffix in c['weather'].values()))
    heavy = sorted(range(len(keys)), key=lambda i: (-weights[i], i))[:8]
    rest = [i for i in np.linspace(0, len(keys) - 1, 32).astype(int).tolist() if i not in heavy][:8]
    selected = heavy + rest
    bound(root / 'preflight/smoke_indices.json', dict(indices=selected, uid=[keys[i] for i in selected],
        policy='largest aggregate weather PCD size (8) plus spread over the accepted frame order (8)',
        memory_probe_limitation='empirical E3 real-input check; later OOM is fatal, no group-specific adjustment'))
    build(root, selected)
    prepare_windows(root, selected, epochs=[1, 2])


def checks(output):
    import torch
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from .data import Dataset, rng_scope
    from .geometry import loss_weights
    from .losses import diffusion_inputs, auxiliary_loss
    from .model import FeatureTap
    from .runtime import hypes
    from .training import make_models, tensor_hash
    root = Path(output)
    c = read(root / 'config.json')
    # Reuse the existing official None/empty/matching synthetic contract tests on the server.
    import importlib.util
    from .common import ROOT
    spec = importlib.util.spec_from_file_location('_raygen_frozen_ap_tests', ROOT / 'v2xr_protocol_audit/test_contract.py')
    ap_tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ap_tests)
    ap_tests.run_tests()
    indices = read(root / 'preflight/smoke_indices.json')['indices']
    data = Dataset(root, epoch=1, auxiliary=True)
    model, aux = make_models(root, 'E1', c)
    init_host, init_aux = tensor_hash(model.state_dict()), tensor_hash(aux.state_dict())
    for g in ('E0', 'E1', 'E2', 'E3'):
        candidate, head = make_models(root, g, c)
        if tensor_hash(candidate.state_dict()) != init_host:
            raise ValueError('Initial host tensors differ: ' + g)
        if head is not None and tensor_hash(head.state_dict()) != init_aux:
            raise ValueError('Initial auxiliary tensors differ: ' + g)
        if g == 'E0' and head is not None:
            raise ValueError('E0 constructed an auxiliary head')
        del candidate, head
    # Compare normal, nonempty frames to untouched official item/voxel/GT functions.
    native = build_dataset(hypes(root), visualize=False, train=True)
    compared = False
    for index in indices:
        adapted = data.get(index, force='clean')
        if adapted['ego']['object_bbx_mask'].sum() == 0:
            continue
        if any(len(x) == 0 for x in adapted['ego']['projected_lidar']):
            continue
        with rng_scope(stream_seed(c['seed'], 1, data.keys[index], 'base')):
            ordinary = native[index]
        fields = ('processed_lidar', 'label_dict', 'object_bbx_center', 'object_bbx_mask', 'pairwise_t_matrix')
        if tensor_digest({k: ordinary['ego'][k] for k in fields}) != tensor_digest({k: adapted['ego'][k] for k in fields}):
            raise ValueError('Auxiliary-disabled adapter changed a normal native sample')
        compared = True
        break
    if not compared:
        raise ValueError('Smoke has no ordinary nonempty frame for native-equivalence check')
    # Group-independent schedule and actual augmented/voxelized inputs.
    sample = data.get(index)
    repeated = data.get(index)
    if sample['_audit'] != repeated['_audit'] or tensor_digest(sample['ego']) != tensor_digest(repeated['ego']):
        raise ValueError('Stateless input schedule is not repeatable')
    original_auxiliary = data.auxiliary
    data.auxiliary = False
    without_aux = data.get(index)
    data.auxiliary = original_auxiliary
    if tensor_digest(sample['ego']) != tensor_digest(without_aux['ego']) or sample['_audit'] != without_aux['_audit']:
        raise ValueError('Auxiliary work changed native input or weather/CAV schedule')
    batch = data.collate([data.get(index, force='clean')])
    inputs = train_utils.to_device(batch['inputs'], torch.device('cuda:0'))
    model = model.cuda().eval()
    aux = aux.cuda()
    tap = FeatureTap(model)
    with torch.no_grad():
        first = model(inputs)
        altered = dict(inputs, object_bbx_center=torch.randn(1, 100, 7, device='cuda'),
                       label_dict={'unused': torch.ones(1, device='cuda')})
        second = model(altered)
    if any(not torch.allclose(first[k], second[k], rtol=1e-5, atol=1e-6) for k in first):
        raise ValueError('GT modification changed fixed-input detector outputs')
    clean = train_utils.create_model(hypes(root)).cuda().eval()
    clean.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        official_out = clean(inputs)
    if any(not torch.allclose(first[k], official_out[k], rtol=1e-5, atol=1e-6) for k in first):
        raise ValueError('Feature observer changed native forward outputs')
    del clean
    model.train()
    result = model(inputs)
    gen, counts = auxiliary_loss('E3', aux, tap.value, batch, 1, c)
    if counts['valid_windows']:
        grad = torch.autograd.grad(gen, tap.value, retain_graph=True)[0]
        if not torch.isfinite(grad).all() or grad.abs().sum() == 0:
            raise ValueError('No auxiliary gradient to fused features')
    empty_batch = dict(batch, auxiliary=[[]])
    zero, _ = auxiliary_loss('E3', aux, tap.value, empty_batch, 1, c)
    if zero.item() != 0 or not torch.isfinite(train_utils.create_loss(hypes(root)).cuda()(result, train_utils.to_device(batch['labels'], torch.device('cuda:0')))):
        raise ValueError('Empty auxiliary target changed detection validity')
    # Loss-position gradients: unknown output coordinates receive exactly zero direct loss gradient.
    lab = dict(x0=-np.ones((8, 32, 32), np.float32))
    for k in ('O_raw', 'O', 'E', 'ego', 'peer'):
        lab[k] = np.zeros((8, 32, 32), bool)
    lab['O_raw'][0, 0, 0] = lab['O'][0, 0, 0] = lab['ego'][0, 0, 0] = True
    lab['E'][0, 0, 1] = True
    prediction = torch.zeros(8, 32, 32, requires_grad=True, device='cuda')
    w = torch.from_numpy(loss_weights('E2', lab)).cuda()
    ((prediction - 1).square() * w).sum().backward()
    if torch.count_nonzero(prediction.grad[w == 0]).item() or torch.count_nonzero(prediction.grad[w > 0]).item() != 2:
        raise ValueError('Unknown output position received a direct loss gradient')
    all_unknown = copy.deepcopy(lab)
    for key in ('O_raw', 'O', 'E', 'ego', 'peer'):
        all_unknown[key][:] = False
    all_unknown['U'] = ~(all_unknown['O'] | all_unknown['E'])
    unknown_batch = dict(batch, auxiliary=[[dict(center=np.array([0., 0., -1.]), target=all_unknown, object_index=-1)]])
    zero_unknown, _ = auxiliary_loss('E2', aux, tap.value, unknown_batch, 1, c)
    if zero_unknown.item() != 0:
        raise ValueError('All-unknown selected window must yield zero auxiliary loss')
    a = diffusion_inputs(lab['x0'], c['seed'], 1, data.keys[index], 0, c)
    for g in ('E1', 'E2', 'E3'):
        b = diffusion_inputs(lab['x0'], c['seed'], 1, data.keys[index], 0, c)
        if any(not np.array_equal(x, y) for x, y in zip(a, b)):
            raise ValueError('Diffusion input differs between groups')
    # Real native zero-GT and empty-CAV paths: no recursion, no fabricated projected points.
    from support import patched
    native_object = data.native.post_processor.generate_object_center
    def no_objects(*args, **kwargs):
        centers, mask, ids = native_object(*args, **kwargs)
        return np.zeros_like(centers), np.zeros_like(mask), []
    retrieve = data.native.retrieve_base_data
    def empty_clouds(*args, **kwargs):
        base, scene, timestamp = retrieve(*args, **kwargs)
        for row in base.values():
            row['lidar_np'] = np.empty((0, 4), dtype=np.float32)
            row['radar_np'] = np.empty((0, 4), dtype=np.float32)
        return base, scene, timestamp
    with patched(data.native.post_processor, 'generate_object_center', no_objects), patched(data.native, 'retrieve_base_data', empty_clouds):
        background = data.get(index)
    if background['ego']['object_bbx_mask'].sum() or any(len(x) for x in background['ego']['projected_lidar']):
        raise ValueError('Empty/background frame acquired synthetic observations')
    bg = data.collate([background])
    model.eval()
    with torch.no_grad():
        bgout = model(train_utils.to_device(bg['inputs'], torch.device('cuda:0')))
        bgloss = train_utils.create_loss(hypes(root)).cuda()(bgout, train_utils.to_device(bg['labels'], torch.device('cuda:0')))
    if not torch.isfinite(bgloss):
        raise ValueError('Zero-GT/empty-CAV loss is not finite')
    report = dict(status='complete', native_sample_equivalence=True, native_forward_equivalence=True,
        minimal_detector_input=True, modified_GT_output_unchanged=True, strict_detector_reload=True,
        empty_aux_detection_retained=True, zero_GT_empty_CAV=True, unknown_output_gradient_zero=True,
        host_initial_tensor_hash=init_host, auxiliary_initial_tensor_hash=init_aux,
        identical_diffusion_inputs=True, stateless_voxels_weather_CAV=True,
        auxiliary_gradient_to_fusion=counts['valid_windows'] > 0,
        official_AP_None_empty_trace=True, all_unknown_selected_window_zero=True,
        forward_tolerance=dict(rtol=1e-5, atol=1e-6),
        hip_determinism='Strict deterministic backend; bilinear index_select crop; separate real exact-gradient check; v3 resume acceptance unchanged')
    write(root / 'smoke/server_checks.json', report)
    print('Server interface checks passed', flush=True)


def resume_check(output, group):
    import torch
    from .training import completion
    from .resume_compare import KEYS, POLICY_ID, POLICY_V3, POLICY_V3_ID, compare_states
    from .common import sha
    root = Path(output)
    original = root / 'smoke' / group / 'epochs/02'
    repeated = root / 'smoke_replay' / group / 'epochs/02'
    start = root / 'smoke' / group / 'epochs/01'
    replay_start = root / 'smoke_replay' / group / 'epochs/01'
    for folder in (start, replay_start, original, repeated):
        completion(root, folder)
    for name in ('resume.pth', 'detector.pth', 'sampling.json', 'actual_samples.json'):
        if sha(start / name) != sha(replay_start / name):
            raise ValueError('Resume branches did not start at an identical checkpoint: ' + name)
    from .resume_compat import effective_execution
    execution = effective_execution(root)
    for folder in (start, replay_start, original, repeated):
        committed = read(folder / 'train_complete.json')
        if (committed['physical_gpu'] != execution['physical_gpu'] or committed['gpu_name'] != execution['gpu_name']
                or committed['group'] != group or committed['epoch'] != int(folder.name) or not committed['smoke_only']):
            raise ValueError('Resume smoke group/epoch/device changed')
    restored = read(root / 'smoke' / group / 'restore_check_v2.json')
    if (restored['status'] != 'complete' or not restored['exact'] or restored['policy_identity'] != POLICY_ID
            or restored['comparison']['status'] != 'complete' or not restored['comparison']['exact']
            or restored['comparison']['arrays_checked'] < 1
            or restored['source_checkpoint_sha256'] != sha(start / 'resume.pth')
            or restored['source_commit_sha256'] != sha(start / 'train_complete.json')
            or restored['group'] != group or restored['physical_gpu'] != execution['physical_gpu']
            or restored['gpu_name'] != execution['gpu_name']):
        raise ValueError('Exact post-load check for this boundary/device is required')
    for name in ('sampling.json', 'actual_samples.json'):
        if read(original / name) != read(repeated / name):
            raise ValueError('Data schedule drifted on resume: ' + name)
    a = torch.load(original / 'resume.pth', map_location='cpu', weights_only=False)
    b = torch.load(repeated / 'resume.pth', map_location='cpu', weights_only=False)
    def as_array(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value if isinstance(value, np.ndarray) else None
    result = compare_states({key: a[key] for key in KEYS}, {key: b[key] for key in KEYS}, as_array, policy=POLICY_V3)
    report = dict(status=result['status'], policy=POLICY_V3, policy_identity=POLICY_V3_ID,
        comparison=result, exact_restore_report_sha256=sha(root / 'smoke' / group / 'restore_check_v2.json'),
        original_checkpoint_sha256=sha(original / 'resume.pth'), replay_checkpoint_sha256=sha(repeated / 'resume.pth'),
        maximum_absolute_delta=max((row['maximum_absolute_delta'] for row in result['numeric']), default=0.),
        tolerance=dict(rtol=1e-4, atol=1e-5), data_schedule_exact=True, same_start_exact=True,
        scope='Exact actual-loader roundtrip + two real epoch2 replay runs from the same epoch1 checkpoint')
    # A failed comparison is diagnostic evidence, never a successful stage commit.
    write(root / 'smoke' / group / 'resume_check_v3_diagnostic.json', report)
    if result['status'] != 'complete':
        raise ValueError('Resume exceeds v3 state validation policy: ' + repr(result['failures'][:8]))
    bound(root / 'smoke' / group / 'resume_check.json', report)
    print('Resume validation v3 passed: ' + group + '; exact restore, original model tolerance, per-element RMS Adam check', flush=True)
