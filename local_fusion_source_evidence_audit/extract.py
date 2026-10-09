"""Read-only frozen hooks and per-frame evidence extraction; labels are never recomputed."""
from pathlib import Path
import numpy as np
from .common import (KEEP, WEATHERS, Manifest, atomic_torch, atomic_json, base_frame,
                     contract, evidence_path, load_cache, object_hash, read_json)
from . import coordinates as coord
from .features import BLOCKS, block_names, vector, geometric, distribution, semantic
from .reproduction import output_reproduction


def removal_plan(count, removed, query=0):
    if not 0 <= removed < count or not 0 <= query < count:
        raise ValueError('Invalid source removal')
    retained = [i for i in range(count) if i != removed]
    return retained, retained.index(query) if query in retained else None


def frozen_outputs(runtime, batch, weather, verify):
    """Hook returns from this very GSPR forward; never run a second reliability pass."""
    import torch
    from local_fusion_task_split_pilot.model import _route, _mean_weights, _fuse_from_weights
    capture = {}
    key = 'processed_lidar' if weather == 'clean' else 'processed_lidar_weather'
    processed = {k: v.detach().clone() for k, v in batch['ego'][key].items()}

    def hook(_module, _inputs, output):
        if capture:
            raise RuntimeError('Unexpected multiple GSPR calls')
        capture.update({k: v.detach().clone() for k, v in output.items()})

    handle = runtime.model.engine.base.gspr.register_forward_hook(hook)
    try:
        context, shared, pool = runtime.predict(batch, weather, verify=verify)
    finally:
        handle.remove()
    required = {'point_reliability', 'point_uncertainty', 'point_evidence',
                'point_valid_mask', 'pillar_reliability', 'pillar_uncertainty'}
    if not required <= capture.keys():
        raise ValueError('Frozen GSPR did not expose required outputs')
    levels, base = context['levels'], runtime.model.engine.base
    _, a, _ = _route(runtime.shared_arm.router_a, levels)
    _, b, _ = _route(runtime.shared_arm.router_b, levels)
    shared_levels = _fuse_from_weights(levels, _mean_weights(a, b))
    return context, pool, processed, capture, shared_levels


def leave_one_out(runtime, levels):
    from local_fusion_utility_v2.fusion import attention_fusion, predict_from_levels
    # Recompute attention and learned B0 routes with each peer absent. Removing
    # ego cannot preserve the original query and therefore has no substitute.
    full_att = predict_from_levels(runtime.model.engine.base, [attention_fusion(x)[0] for x in levels])
    output = {}
    for removed in range(len(levels[0])):
        retained, query = removal_plan(len(levels[0]), removed)
        if query is None or not retained:
            output[removed] = None
            continue
        subset = [x[retained] for x in levels]
        prediction, _ = runtime.shared_arm.predict(runtime.model.engine.base, subset)
        attention = predict_from_levels(runtime.model.engine.base, [attention_fusion(x, query)[0] for x in subset])
        output[removed] = {'shared': prediction, 'attfuse': attention}
    return full_att, output


def numpy(value):
    return value.detach().cpu().numpy()


def sensitivity(full, removed, full_proxy, removed_proxy, anchor, mask):
    from local_fusion_action_utility_audit.features import geometry, pred_iou
    if removed is None:
        return {'applicable': 0}
    def logit(pred):
        return numpy(pred['psm']).transpose(0, 2, 3, 1).reshape(-1)
    a, b = logit(full), logit(removed)
    channels = int(full['psm'].shape[1])
    roi = np.repeat(mask[..., None], channels, -1).reshape(-1)
    def probability(value):
        return 1 / (1 + np.exp(-np.clip(np.asarray(value, dtype=np.float64), -80, 80)))
    values = dict(applicable=1, full_logit=a[anchor], removed_logit=b[anchor],
                  delta_logit=a[anchor]-b[anchor], delta_probability=probability(a[anchor])-probability(b[anchor]))
    if roi.any():
        probability_a, probability_b = probability(a[roi]), probability(b[roi])
        values.update(roi_max_delta=probability_a.max()-probability_b.max(),
                      roi_mean_delta=probability_a.mean()-probability_b.mean())
    ca, sa, ya = geometry(full_proxy['corners'][anchor])
    cb, sb, yb = geometry(removed_proxy['corners'][anchor])
    values.update(zip(('delta_x', 'delta_y', 'delta_z'), ca-cb))
    values.update(zip(('delta_length', 'delta_width', 'delta_height'), sa-sb))
    values.update(delta_yaw_sin=np.sin(ya-yb), delta_yaw_cos=np.cos(ya-yb),
                  before_after_pred_iou=pred_iou(full_proxy['corners'][anchor:anchor+1], removed_proxy['corners'][anchor:anchor+1])[0, 0])
    anchor_num = full['psm'].shape[1]
    def regression(pred):
        return numpy(pred['rm']).transpose(0, 2, 3, 1).reshape(-1, anchor_num, 7).reshape(-1, 7)[anchor]
    delta = regression(full)-regression(removed)
    values.update(rm_l1=np.abs(delta).mean(), rm_l2=np.linalg.norm(delta))
    values.update({f'rm_delta_{i}': v for i, v in enumerate(delta)})
    return values


def source_blocks(corners, names, processed, reliability, levels, shared_levels,
                  transforms, lidar_range, expansion, full_pool, proxies, loo, loo_proxies,
                  att_full, att_proxy, ids, masks):
    from local_fusion_action_utility_audit.features import geometry
    features = numpy(processed['voxel_features'])
    coords = numpy(processed['voxel_coords']).astype(int)
    num = numpy(processed['voxel_num_points']).astype(int)
    valid = np.arange(features.shape[1])[None, :] < num[:, None]
    if not np.array_equal(valid, numpy(reliability['point_valid_mask']).astype(bool)):
        raise ValueError('GSPR mask differs from retained slots')
    rel = {k: numpy(v) for k, v in reliability.items()}
    all_levels = [numpy(x) for x in levels[:2]]
    shared_levels = [numpy(x)[0] for x in shared_levels[:2]]
    count = len(levels[0])
    output = {b: np.zeros((len(corners), len(names), 2*len(keys)), np.float32) for b, keys in BLOCKS.items()}
    # Full tensors are converted once; sensitivity inputs remain tiny slices below.
    cpu_pool = {name: {k: v.detach().cpu() for k, v in pred.items()} for name, pred in full_pool.items()}
    cpu_att = {k: v.detach().cpu() for k, v in att_full.items()}
    cpu_loo = {i: None if pred is None else {ref: {k: v.detach().cpu() for k, v in p.items()} for ref, p in pred.items()} for i, pred in loo.items()}
    voxel_h, voxel_w = (int(round((lidar_range[4]-lidar_range[1])/float(processed['_voxel_size'][1]))),
                         int(round((lidar_range[3]-lidar_range[0])/float(processed['_voxel_size'][0]))))
    pillar_grid = coord.grid_centers(voxel_h, voxel_w, lidar_range)
    all_centers = pillar_grid[coords[:, 2], coords[:, 3]]
    for p, box in enumerate(corners):
        # Restrict work before touching padded slots: a vehicle ROI normally
        # contains a small fraction of the frame's retained pillars.
        center = box.mean(0)
        expanded = center+(box-center)*expansion
        margin = np.asarray(processed['_voxel_size'])[:2]
        low, high = expanded[:, :2].min(0)-margin, expanded[:, :2].max(0)+margin
        nearby = np.flatnonzero(((all_centers[:, :2] >= low) & (all_centers[:, :2] <= high)).all(1))
        local_coords, local_valid = coords[nearby], valid[nearby]
        valid_roi = local_valid & coord.inside(features[nearby, :, :3], box, expansion)
        pillar_roi = coord.inside(pillar_grid, box, expansion, bev=True)
        centers = all_centers[nearby]
        pillar_selected = coord.inside(centers, box, expansion, bev=True)
        possible = int(pillar_roi.sum())
        slot_ids = np.repeat(local_coords[:, None, [0, 2, 3]], features.shape[1], axis=1)
        for s, name in enumerate(names):
            underlying = None if name == KEEP else int(name.split(':')[1])
            if name.startswith('query:'):
                single_position = names.index(f'single:{underlying}')
                for block in BLOCKS:
                    output[block][p, s] = output[block][p, single_position]
                query_flag = BLOCKS['loo'].index('query_self_applicable')
                output['loo'][p, s, query_flag] = 0
                output['loo'][p, s, len(BLOCKS['loo'])+query_flag] = 1
                continue
            rows = np.ones(len(local_coords), bool) if underlying is None else local_coords[:, 0] == underlying
            point_mask = valid_roi & rows[:, None]
            pillar_mask = pillar_selected & rows & (num[nearby] > 0)
            support_ids = slot_ids[point_mask]
            geo = geometric(features[nearby, :, :3][point_mask], support_ids, box, expansion,
                            possible*(count if underlying is None else 1), None if underlying is None else transforms[underlying])
            output['geometry'][p, s] = vector('geometry', geo)
            gs = {'point_coverage': point_mask.sum()/max(1, int((local_valid & rows[:, None]).sum())),
                  'pillar_coverage': pillar_mask.sum()/max(1, possible*(count if underlying is None else 1)),
                  'observation_valid': int(point_mask.any() or pillar_mask.any())}
            for field in ('point_reliability', 'point_uncertainty', 'point_evidence_reliable', 'point_evidence_unreliable', 'pillar_reliability', 'pillar_uncertainty'):
                if field.startswith('point_evidence'):
                    value = rel['point_evidence'][nearby, :, 0 if field.endswith('_reliable') else 1][point_mask]
                else:
                    value = rel[field][nearby][point_mask if field.startswith('point_') else pillar_mask]
                gs.update({field+'_'+k: v for k, v in distribution(value).items()})
            output['gspr'][p, s] = vector('gspr', gs)
            sem = {}
            for scale, all_sources in enumerate(all_levels):
                selected = shared_levels[scale] if underlying is None else all_sources[underlying]
                roi = coord.inside(coord.grid_centers(*selected.shape[-2:], lidar_range), box, expansion, bev=True)
                sem.update({f's{scale}_'+k: v for k, v in semantic(selected, shared_levels[scale], all_sources[0], all_sources, roi, underlying).items()})
            output['semantic'][p, s] = vector('semantic', sem)
            lo = {'query_self_applicable': 0} if name.startswith('query:') else {}
            for ref in ('shared', 'attfuse'):
                removed = None if underlying is None else cpu_loo[underlying]
                full = cpu_pool[KEEP] if ref == 'shared' else cpu_att
                proxy = proxies[KEEP] if ref == 'shared' else att_proxy
                summary = sensitivity(full, None if removed is None else removed[ref], proxy,
                                      None if removed is None else loo_proxies[underlying][ref], int(ids[p]), masks[p])
                lo.update({ref+'_'+k: v for k, v in summary.items()})
            output['loo'][p, s] = vector('loo', lo)
            sq = {}
            if underlying is not None:
                a, b, sh = (proxies[f'single:{underlying}']['scores'], proxies[f'query:{underlying}']['scores'], proxies[KEEP]['scores'])
                anchor_num = proxies[KEEP]['anchor_num']
                roi = np.repeat(masks[p][..., None], anchor_num, -1).reshape(-1)
                for scope, values in [('anchor', [x[ids[p]] for x in (a, b, sh)])] + ([(key, [getattr(x[roi], reduction)() for x in (a, b, sh)]) for key, reduction in (('roi_max', 'max'), ('roi_mean', 'mean'))] if roi.any() else []):
                    single, query, shared = values
                    sq.update({scope+'_'+k: v for k, v in zip(('single', 'query', 'query_minus_single', 'query_minus_shared', 'single_minus_shared'), (single, query, query-single, query-shared, single-shared))})
            output['singlequery'][p, s] = vector('singlequery', sq)
    return output


def extract(run, split):
    import torch
    from local_fusion_action_utility_audit.common import Runtime
    from local_fusion_action_utility_audit.s0_counterfactual import proposals
    from local_fusion_action_utility_audit.reproducibility import load_reference, numeric_difference, tensor_tree_hash, inference_input_hash
    from local_fusion_task_source_oracle.oracle import _proxy_cache
    from opencood.tools.train_utils import to_device
    protocol = contract(run)
    runtime = Runtime(protocol['base']['base_run'])
    manifest = Manifest(run)
    for weather in WEATHERS:
        dataset, loader = runtime.loader(split, weather)
        seen = []
        with torch.no_grad():
            for ordinal, batch in enumerate(loader):
                index = int(batch['ego']['communication_sample_index'][0])
                seen.append(index)
                path = evidence_path(run, split, weather, index)
                if manifest.complete('evidence-frame', split, weather, index):
                    continue
                with manifest.work('evidence-frame', split, weather, index):
                    cached = base_frame(run, split, weather, index)
                    batch = to_device(batch, runtime.target)
                    before = inference_input_hash(batch, weather)
                    context, generated, processed, reliability, shared_levels = frozen_outputs(runtime, batch, weather, ordinal == 0)
                    transforms = coord.validate(runtime, batch, processed, context['levels'])
                    differences = {}
                    if split == 'validation':
                        pool, reference = load_reference(runtime, batch, weather, index)
                        if set(pool) != set(generated):
                            raise ValueError('Source family/cardinality drift')
                        for name in pool:
                            for key in ('psm', 'rm'):
                                delta = numeric_difference(numpy(pool[name][key]), numpy(generated[name][key]))
                                if not delta.get('allclose_2e5'):
                                    raise ValueError('Repeated frozen output exceeds existing FP32 tolerance: '+name+'/'+key)
                                differences[name+'/'+key] = delta
                    else:
                        pool = generated
                    ids, scores, corners, masks, names, old = proposals(dataset, batch, pool[KEEP], pool, context, runtime)
                    if names != cached['source_names'] or ids.tolist() != cached['proposals']['anchor_ids']:
                        raise ValueError('S4 candidate/source pool differs from S0; labels cannot be reused')
                    failures = []
                    for task in ('cls', 'reg'):
                        check = output_reproduction(old[task], numpy(cached['features'][task]),
                                                    task, names, require_exact=split == 'validation',
                                                    regression_actual=old['reg'],
                                                    regression_expected=numpy(cached['features']['reg']))
                        differences['output_only/'+task] = check
                        if not check['accepted']:
                            failures.append(task+': '+check['reason'])
                        elif check['rank_ties']:
                            print(f'evidence {split}/{weather} frame={index} verified near-tie ranks: '
                                  f"{check['rank_ties']}", flush=True)
                        if check['accepted'] and check['geometry_product']:
                            print(f'evidence {split}/{weather} frame={index} verified distance product: '
                                  f"{check['geometry_product']}", flush=True)
                    if failures:
                        diagnostic = Path(run)/'diagnostics'/f'output_reproduction_{split}_{weather}_{index:08d}.json'
                        atomic_json(diagnostic, {'split': split, 'weather': weather, 'frame': index,
                                                'source_names': names, 'checks': differences})
                        raise ValueError(f'Output-only evidence reproduction failed {split}/{weather} '
                                         f'frame={index}: '+ '; '.join(failures)+f'; diagnostics={diagnostic}')
                    att_full, loo = leave_one_out(runtime, context['levels'])
                    proxies = {name: _proxy_cache(dataset, batch, pred) for name, pred in pool.items()}
                    att_proxy = _proxy_cache(dataset, batch, att_full)
                    loo_proxies = {i: None if preds is None else {ref: _proxy_cache(dataset, batch, pred) for ref, pred in preds.items()} for i, preds in loo.items()}
                    processed['_voxel_size'] = runtime.hypes['preprocess']['args']['voxel_size']
                    blocks = source_blocks(numpy(cached['proposals']['corners']), names, processed, reliability,
                                           context['levels'], shared_levels, transforms, runtime.lidar_range,
                                           runtime.spec['roi_expansion'], pool, proxies, loo, loo_proxies,
                                           att_full, att_proxy, ids, masks)
                    if before != inference_input_hash(batch, weather):
                        raise RuntimeError('Frozen extraction mutated inference inputs')
                    value = {'schema': 1, 'identity': protocol['identity'], 'frame': index,
                             'scene': cached['scene'], 'split': split, 'weather': weather,
                             'source_names': names, 'proposal_ids': ids.tolist(),
                             'input_hash': before, 'blocks': {k: torch.from_numpy(v) for k, v in blocks.items()},
                             'masks': torch.from_numpy(np.asarray(masks, dtype=bool)),
                             'original_labels_hash': tensor_tree_hash(cached['labels']),
                             'output_reproduction': differences, 'inference_gt_fields': 0}
                    atomic_torch(path, value)
                    del context, generated, pool, shared_levels, reliability, loo, att_full, blocks
                manifest.mark('evidence-frame', 'complete', split, weather, index, [path])
                print(f'evidence {split}/{weather} {ordinal+1}/{len(loader)} frame={index} complete', flush=True)
        if sorted(seen) != sorted(runtime.protocol[split+'_indices']):
            raise RuntimeError('Incomplete evidence frame pass')
        del dataset, loader


if __name__ == '__main__':
    from .common import parser
    cli = parser(__doc__)
    cli.add_argument('--split', choices=('train', 'validation'), required=True)
    args = cli.parse_args()
    extract(args.run, args.split)
