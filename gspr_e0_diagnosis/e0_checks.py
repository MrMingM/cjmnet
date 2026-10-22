"""Read-only baseline verification and real-model equivalence gates (server only)."""
import copy
from pathlib import Path
import shutil
import numpy as np
from e0_common import HERE, WEATHERS, atomic, read, safe, sha
from e0_runtime import (cache_audit, model_inventory, npz_write, runtime_info,
                        seed, setup, smoke_indices, state_hash, tensor_numpy)


def preflight(cli, folder):
    args, evidence, frozen, audit, evaluation, models = setup(cli)
    inventory = model_inventory(cli, audit)
    verified = {v: cache_audit(m, evidence, frozen) for v, m in inventory.items()}
    if inventory['C']['contract']['pair_identity'] != inventory['D']['contract']['pair_identity']:
        raise ValueError('C/D training paired inventory differs')
    if inventory['C']['contract']['config_sha256'] != inventory['D']['contract']['config_sha256']:
        raise ValueError('C/D original official model configurations differ')
    from opencood.hypes_yaml import yaml_utils
    from observe import normalize_state, module_inventory
    import torch
    loaded = {}
    for variant, record in inventory.items():
        h = yaml_utils.load_yaml(str(Path(record['root']) / 'config.yaml'))
        if (not h['train_sim'] or h['eval_sim'] or h['sim_weather'] != '_fog_0.060'
                or h['model']['args']['use_modality'] != 'processed_lidar'):
            raise ValueError('Actual trained model configuration differs from mixed-weather LiDAR protocol')
        model = models.create(h, variant)
        state = normalize_state(torch.load(record['selected']['checkpoint'], map_location='cpu', weights_only=False))
        actual = model.state_dict()
        if actual.keys() != state.keys() or any(actual[k].shape != state[k].shape or actual[k].dtype != state[k].dtype for k in actual):
            raise ValueError('Selected model/checkpoint key/shape/dtype mismatch')
        model.load_state_dict(state, strict=True)
        if any(not torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items()):
            raise ValueError('Checkpoint load is not exact')
        loaded[variant] = dict(class_name=type(model).__module__+'.'+type(model).__name__,
            state_sha256=state_hash(model), exact=True)
        del model, state
    space = shutil.disk_usage(cli.output)
    if space.free < 25*1024**3:
        raise ValueError('Need at least 25 GiB free for paired/color evidence and diagnostics')
    result = dict(status='complete', models=inventory, cache_checks=verified,
        loading=loaded, environment=runtime_info(), sample_indices=smoke_indices(evidence),
        source_modules=module_inventory(Path(args.runtime_root)), free_bytes=space.free,
        archived_results_sha256=sha(HERE / 'original_FINAL_RESULTS.json'),
        point_schema_probe=point_schema(evidence),
        differences_from_description=[
            'Actual D adapter reuses official VFE and weights decorated features before PFN; attfuse_gspr/pillar_vfe.py is not the V2X-R execution class.',
            'Mixed training samples Fog per CAV at probability 3/7; inference is fixed single weather.',
            'Reliability class docstring mentions calibration; this V2X-R checkpoint has no point labels and cannot substantiate calibration.',
            'Official BaseDataset reads the green channel and constructs de_lidar_np with label==0; C/D use_DeLidar=False and their loss has no point-label supervision. Audit that mask separately from spatial-nearest proxies.',
            'Source context contains historical Clean-only results; only selected L0_C/L1_D contracts and their final caches are used here.'],
        no_test=True, no_training=True)
    atomic(folder / 'inventory.json', result, immutable=True)
    atomic(folder / 'evidence.json', evidence, immutable=True)
    atomic(folder / 'E0_PROTOCOL.json', dict(**read(HERE / 'protocol.json'),
        actual_inputs={k: getattr(args, k) for k in ('runtime_root', 'gt_audit_root', 'd1a_root', 'd1c_root', 'view_root', 'model_dir')},
        original_root=str(cli.original), output_root=str(cli.output),
        original_models={v: r['selected'] for v, r in inventory.items()},
        actual_data_overlay=evidence['input_manifest']['overlay'],
        official_protocol_definitions=frozen['protocol']['protocols'],
        diagnostic_space='processed official voxel slots for reliability; raw and real retained points for paired geometry'), immutable=True)


def point_schema(evidence):
    from opencood.utils.pcd_utils import pcd_to_np
    result = {}
    for weather in WEATHERS:
        path = safe(evidence['frames'][0]['conditions'][weather]['context']['cavs'][0]['lidar_file'])
        header = {}
        with path.open('rb') as stream:
            for _ in range(64):
                line = stream.readline().decode('ascii').strip()
                if not line or line.startswith('#'):
                    continue
                key, *value = line.split()
                header[key] = value
                if key == 'DATA':
                    break
            else:
                raise ValueError('Unrecognized PCD header')
        if header.get('FIELDS') != ['x', 'y', 'z', 'rgb']:
            raise ValueError('Point schema changed; review potential labels/correspondence before using proxies')
        points, green = pcd_to_np(str(path), get_weather=True)
        values, counts = np.unique(green, return_counts=True)
        result[weather] = dict(path=str(path), sha256=sha(path), header=header, points=len(points),
            green_channel_values=values.tolist(), green_channel_counts=counts.tolist(),
            official_loader_usage='label==0 constructs de_lidar_np; LiDAR-only C/D do not use the DeLidar branch',
            green_semantics='unverified color channel; never treated as a weather/noise ground truth')
    return dict(scope='first verified CAV file in each weather; actual original path/suffix', examples=result,
        label_provenance='Official zero-green mask is observable; accepted contracts supply no verified point IDs/generator-origin mapping. Keep author-mask and spatial proxies separate.')


def clone_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target):
            raise ValueError('Incompatible copied baseline evidence')
    else:
        temp = Path(str(target)+'.tmp')
        shutil.copyfile(source, temp)
        temp.replace(target)


def baseline(cli, folder, inventory):
    _, evidence, _, _, _, _ = setup(cli, gpu=False)
    import evaluate as frozen_evaluate
    for variant in ('C', 'D'):
        original = Path(inventory['models'][variant]['root']) / 'final_evaluation'
        target = folder / variant
        manifests = {}
        for weather in WEATHERS:
            source = original / 'predictions' / weather
            for path in sorted(source.glob('*.json')):
                clone_file(path, target / 'predictions' / weather / path.name)
            clone_file(original / 'official_runs' / weather / 'official_result_stat.json',
                target / 'official_runs' / weather / 'official_result_stat.json')
            manifests[weather] = read(source / 'manifest.json')
        contract = read(original / 'contract.json')
        result, prediction, gt = frozen_evaluate.replay(target, evidence, contract, manifests)
        if result != inventory['models'][variant]['summary']['results']:
            raise ValueError('Original full baseline replay differs from FINAL_RESULTS.json: ' + variant)
        atomic(target / 'summary.json', dict(status='complete', results=result,
            checkpoint_sha256=inventory['models'][variant]['selected']['checkpoint_sha256'],
            frame_count=829, full_validation=True, origin=str(original)), immutable=True)
    atomic(folder / 'E0_BASELINE.json', dict(status='complete',
        validation_frames_per_weather=829, AP_replayed_exactly=True,
        source='existing immutable final prediction caches; no repeated baseline inference'), immutable=True)


def match_signature(boxes, scores, fixed):
    import torch
    from evaluate import official_match_trace, statistics
    from support import corner_array
    stat, pairs = statistics(), {}
    for t in (.3, .5, .7):
        p, ties = official_match_trace(boxes, scores, torch.from_numpy(corner_array(fixed)),
            [r['object_id'] for r in fixed], stat, t)
        pairs[str(t)] = sorted((r['object_id'], r['prediction_index']) for r in p)
    return pairs, stat


def self_test(cli, folder, inventory):
    args, evidence, _, _, _, models = setup(cli)
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils, inference_utils
    from opencood.utils import eval_utils_opv2v
    from observe import normalize_state
    from e0_observe import Observer
    from e0_inference import compare_predictions
    from torch.utils.data import DataLoader, Subset
    import torch
    protocol = read(HERE / 'protocol.json')
    indices = smoke_indices(evidence)
    results = []
    for variant in ('C', 'D'):
        for weather in WEATHERS:
            seed()
            hypes = copy.deepcopy(evidence['b']['evaluated_configs'][weather])
            ds = build_dataset(hypes, visualize=True, train=False)
            loader = DataLoader(Subset(ds, indices), batch_size=1, num_workers=16,
                collate_fn=ds.collate_batch_test, shuffle=False, pin_memory=False, drop_last=False)
            model = models.create(hypes, variant).to('cuda:0').eval()
            state = normalize_state(torch.load(inventory['models'][variant]['selected']['checkpoint'], map_location='cpu', weights_only=False))
            model.load_state_dict(state, strict=True)
            for p in model.parameters():
                p.requires_grad_(False)
            initial = state_hash(model)
            aggregates = {name: {p: {t: dict(tp=[], fp=[], gt=0, score=[]) for t in (.3, .5, .7)}
                for p in ('O', 'O*', 'F')} for name in ('native', 'observed')}
            for position, cpu_batch in enumerate(loader):
                index = indices[position]
                batch = train_utils.to_device(cpu_batch, torch.device('cuda:0'))
                voxels = batch['ego']['processed_lidar']
                from e0_numeric import array_hash
                before_inputs = {k: array_hash(tensor_numpy(v)) for k, v in voxels.items()}
                heads = []
                hook = model.register_forward_hook(lambda m, i, out: heads.append({k: tensor_numpy(out[k]).copy() for k in ('psm', 'rm')}))
                with torch.no_grad():
                    native = inference_utils.inference_intermediate_fusion(batch, model, ds)
                    repeated = inference_utils.inference_intermediate_fusion(batch, model, ds)
                hook.remove()
                ctx = dict(weather=weather, frame_key=evidence['b']['selected_frame_keys'][index],
                    cavs=evidence['frames'][index]['conditions'][weather]['context']['participating_cavs'],
                    state_before_forward=initial)
                observer = Observer(model, 'R0', ctx, branch_support=True)
                with torch.no_grad():
                    observed = inference_utils.inference_intermediate_fusion(batch, model, ds)
                observer.verify_unchanged()
                observer.close()
                observed_heads = ctx['heads']
                deltas = {}
                for name in ('psm', 'rm'):
                    np.testing.assert_allclose(heads[0][name], heads[1][name], atol=protocol['head_atol'], rtol=protocol['head_rtol'])
                    np.testing.assert_allclose(heads[0][name], ctx['heads'][name], atol=protocol['head_atol'], rtol=protocol['head_rtol'])
                    deltas[name] = dict(native_repeat_max=float(np.max(abs(heads[0][name]-heads[1][name]))),
                        observer_max=float(np.max(abs(heads[0][name]-ctx['heads'][name]))))
                record = dict(prediction_is_none=native[0] is None,
                    prediction_boxes=None if native[0] is None else tensor_numpy(native[0]).tolist(),
                    scores=None if native[1] is None else tensor_numpy(native[1]).tolist(), box_dtype='float32', score_dtype='float32')
                compare_predictions(observed[0], observed[1], record, protocol)
                compare_predictions(repeated[0], repeated[1], record, protocol)
                if not torch.equal(native[2], observed[2]):
                    raise ValueError('Observer changed official GT')
                from predict import live_gt_observer
                _, gt_rows = live_gt_observer(ds.post_processor, batch, ds.post_processor.generate_gt_bbx)
                fixed = read(Path(inventory['models']['C']['root']) / 'final_evaluation/predictions/clean' / f'{index:06d}.json')['official_gt']
                for p in ('O', 'O*', 'F'):
                    objects = fixed if p == 'F' else gt_rows
                    sig_a, stat_a = match_signature(native[0], native[1], objects)
                    sig_b, stat_b = match_signature(observed[0], observed[1], objects)
                    sig_repeat, _ = match_signature(repeated[0], repeated[1], objects)
                    if sig_a != sig_b or sig_a != sig_repeat:
                        raise ValueError('Observed/repeated forward changed frozen '+p+' assignments')
                    for key, value in (('native', stat_a), ('observed', stat_b)):
                        if p == 'O' and native[0] is None:
                            continue
                        for t in value:
                            for field in ('tp', 'fp', 'score'):
                                aggregates[key][p][t][field].extend(value[t][field])
                            aggregates[key][p][t]['gt'] += value[t]['gt']
                if variant == 'D':
                    for arm in ('R1', 'R2', 'R3'):
                        intervention = Observer(model, arm, ctx)
                        with torch.no_grad():
                            inference_utils.inference_intermediate_fusion(batch, model, ds)
                        intervention.verify_unchanged()
                        if not intervention.last['pfn_weight_verified']:
                            raise ValueError('Intervention PFN gate missing')
                        intervention.close()
                if before_inputs != {k: array_hash(tensor_numpy(v)) for k, v in voxels.items()}:
                    raise ValueError('Observer/intervention modified original input voxels')
                if position == 0:
                    npz_write(folder / f'{variant}_{weather}_heads.npz',
                        native_psm=heads[0]['psm'], native_rm=heads[0]['rm'],
                        observed_psm=observed_heads['psm'], observed_rm=observed_heads['rm'])
                results.append(dict(variant=variant, weather=weather, index=index, deltas=deltas,
                    original_input_unchanged=True, matched_ids_equal=True, original_same_input_GT_exact=True,
                    frozen_state_sha256=initial))
            ap = {name: {p: {str(t): float(eval_utils_opv2v.calculate_ap(copy.deepcopy(stat), t, False)[0]) for t in stat}
                for p, stat in protocols.items()} for name, protocols in aggregates.items()}
            if any(abs(ap['native'][p][str(t)]-ap['observed'][p][str(t)]) > protocol['ap_atol'] for p in ('O', 'O*', 'F') for t in (.3, .5, .7)):
                raise ValueError('Small-sample frozen evaluation differs')
            if state_hash(model) != initial:
                raise ValueError('Self-test changed original checkpoint parameters or buffers')
            atomic(folder / f'{variant}_{weather}_metrics.json', ap, immutable=True)
            del model, state
            torch.cuda.empty_cache()
    atomic(folder / 'E0_SELF_TEST.json', dict(status='complete', records=results,
        original_and_observed_heads_checked=True, final_predictions_checked=True,
        fixed_GT_matching_checked=True, repeat_forward_variation_measured=True,
        scope='12 paired frames across all six scenes per weather per model; full inference still gated separately'), immutable=True)
