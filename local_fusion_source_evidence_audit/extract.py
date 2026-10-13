"""Read-only frozen hooks and per-frame evidence extraction; labels are never recomputed."""
from pathlib import Path
import numpy as np
from .common import (KEEP, WEATHERS, Manifest, atomic_torch, atomic_json, base_frame,
                     contract, evidence_path, load_cache, object_hash, read_json)
from . import coordinates as coord
from .features import BLOCKS, block_names, vector, geometric, distribution, semantic
from .reproduction import (output_reproduction, proposal_reproduction, candidate_snapshot_proxies,
                           source_score_reproduction, source_score_snapshot_proxies)


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


def fixed_proposals(batch, proxies, context, runtime, cached, require_exact=False):
    """Use original S0 anchors/order/ROIs, checking their current frozen outputs."""
    from local_fusion_action_utility_audit.features import source_names, extract_features, schema
    from qa_local_intervention.operators import masks_for_frame
    reference = dict(cached['proposals'])
    reference['corners'] = numpy(reference['corners'])
    names = source_names(proxies)
    shared = proxies[KEEP]
    check = proposal_reproduction(reference, cached['source_names'], names,
                                  shared['scores'], shared['corners'],
                                  runtime.spec['shared_top_k'], require_exact)
    if not check['accepted']:
        return None, check
    ids = np.asarray(reference['anchor_ids'], np.int64)
    # Original S0 ROIs, including its empty-ROI fallback. Repeated inference is
    # checked, but cannot silently change which region an old label describes.
    masks = [masks_for_frame(batch['ego']['anchor_box'], corner, context['levels'],
                             runtime.lidar_range, runtime.spec['roi_expansion'])[0]
             for corner in reference['corners']]
    # Raw repeated values have already passed the check above. The candidate
    # definition also includes its score/box, not just its anchor ID and mask.
    # Restore KEEP at selected anchors as well: otherwise the KEEP-specific
    # higher-score count can still flip against an unfrozen anchor probability.
    audit_proxies = candidate_snapshot_proxies(proxies, reference)
    expected_cls = numpy(cached['features']['cls'])
    if expected_cls.shape != (len(ids), len(names), len(schema('cls'))):
        return None, dict(check, accepted=False, reason='invalid S0 classification feature snapshot shape')
    anchor_probabilities = expected_cls[..., schema('cls').index('anchor_probability')]
    score_check = source_score_reproduction(audit_proxies, ids, names, anchor_probabilities, require_exact)
    check['source_anchor_scores'] = score_check
    if not score_check['accepted']:
        return None, dict(check, accepted=False, reason=score_check['reason'])
    audit_proxies = source_score_snapshot_proxies(audit_proxies, ids, names, anchor_probabilities)
    # top_shared_proposals promotes the selected scores to FP64 in S0. Keeping
    # FP32 here changes the accumulator in competition_local_score_density.
    scores = np.asarray(reference['scores'], dtype=np.float64)
    corners = np.asarray(reference['corners'], dtype=np.float32)
    names, features = extract_features(audit_proxies, ids, scores, corners, masks, runtime.spec)
    return (ids, scores, corners, masks, names, features), check


def reference_pool(runtime, batch, weather, index, split, generated):
    """One validation boundary shared by preaudit and evidence extraction."""
    from local_fusion_action_utility_audit.reproducibility import load_reference, numeric_difference
    differences, failures = {}, []
    if split == 'train':
        return generated, differences, failures
    pool, _ = load_reference(runtime, batch, weather, index)
    if set(pool) != set(generated):
        return pool, differences, ['source family/cardinality drift']
    for name in pool:
        for key in ('psm', 'rm'):
            delta = numeric_difference(numpy(generated[name][key]), numpy(pool[name][key]))
            differences[name+'/'+key] = delta
            if not delta.get('allclose_2e5', False):
                failures.append('repeated frozen output exceeds original tolerance: '+name+'/'+key)
    return pool, differences, failures


def check_outputs(proposals, cached, split, differences):
    """Check both tasks and every field, without touching cached model inputs."""
    failures = []
    names, actual = proposals[4:]
    expected = {task: numpy(cached['features'][task]) for task in ('cls', 'reg')}
    for task in ('cls', 'reg'):
        check = output_reproduction(actual[task], expected[task], task, names,
            require_exact=split == 'validation', regression_actual=actual['reg'],
            regression_expected=expected['reg'], classification_actual=actual['cls'],
            classification_expected=expected['cls'])
        differences['output_only/'+task] = check
        if not check['accepted']:
            failures.append(task+': '+check['reason'])
    return failures


def source_count_diagnostics(proposals, cached, proxies):
    """Expose score and IoU decisions for residual count failures, without exemptions."""
    from local_fusion_action_utility_audit.features import schema, pred_iou
    ids, scores, corners, _, names, actual = proposals
    field = 'competition_source_higher_overlap_count'
    expected = numpy(cached['features']['cls'])
    j = schema('cls').index(field)
    positions = np.argwhere(actual['cls'][..., j] != expected[..., j])
    probability = schema('cls').index('anchor_probability')
    records = []
    for p, s in positions[:16]:
        p, s = int(p), int(s)
        name, anchor = names[s], int(ids[p])
        source_corner = corners[p:p+1] if name == KEEP else proxies[name]['corners'][anchor:anchor+1]
        ious = pred_iou(source_corner, corners)[0]
        others = np.arange(len(ids)) != p
        old_score = float(expected[p, s, probability])
        raw_score = float(proxies[name]['scores'][anchor])
        overlap = others & (ious > .1)
        records.append({'proposal_position': p, 'source': name, 'anchor_id': anchor,
            'expected_count': int(expected[p, s, j]), 'actual_count': int(actual['cls'][p, s, j]),
            'original_source_probability': old_score, 'repeated_source_probability': raw_score,
            'count_current_geometry_original_probability': int((overlap & (scores > old_score)).sum()),
            'count_current_geometry_repeated_probability': int((overlap & (scores > raw_score)).sum()),
            'current_prediction_iou': ious.tolist(),
            'score_comparison_changes': np.flatnonzero(others & ((scores > old_score) != (scores > raw_score))).tolist(),
            'iou_within_2e5_of_boundary': np.flatnonzero(others & (np.abs(ious-.1) <= 2e-5)).tolist()})
    return {'different_rows': len(positions), 'recorded_rows': len(records), 'rows': records,
            'candidate_anchor_ids': ids.tolist(), 'candidate_scores': scores.tolist(),
            'limit': 'old full train source boxes unavailable; these are current IoUs, not proof of old pairwise IoUs'}


def check_frame(run, split, weather, index, batch, proxies, context, runtime, cached, differences):
    """Return checked candidates or write one complete frame diagnostic."""
    try:
        proposals, check = fixed_proposals(batch, proxies, context, runtime, cached,
                                          require_exact=split == 'validation')
        differences['proposal_pool'] = check
        failures = [check['reason']] if proposals is None else check_outputs(proposals, cached, split, differences)
    except ValueError as exc:
        proposals = None
        failures = ['feature reproduction computation failed: '+str(exc)]
    if failures:
        if proposals is not None and any('competition_source_higher_overlap_count' in f for f in failures):
            differences['source_count_decisions'] = source_count_diagnostics(proposals, cached, proxies)
        prefix = 'proposal' if proposals is None else 'output'
        path = Path(run)/'diagnostics'/f'{prefix}_reproduction_{split}_{weather}_{index:08d}.json'
        atomic_json(path, {'split': split, 'weather': weather, 'frame': index,
                          'source_names': cached['source_names'], 'checks': differences,
                          'reasons': failures})
        return None, {'split': split, 'weather': weather, 'frame': index,
                      'reasons': failures, 'diagnostics': str(path)}
    return proposals, None


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
    from local_fusion_action_utility_audit.reproducibility import tensor_tree_hash, inference_input_hash
    from local_fusion_task_source_oracle.oracle import _proxy_cache
    from opencood.tools.train_utils import to_device
    protocol = contract(run)
    runtime = Runtime(protocol['base']['base_run'])
    manifest = Manifest(run)
    if not manifest.complete('reproduction-audit'):
        raise RuntimeError('Complete all-frame reproduction-audit before expensive evidence extraction')
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
                    pool, differences, failures = reference_pool(runtime, batch, weather, index, split, generated)
                    if failures:
                        diagnostic = Path(run)/'diagnostics'/f'raw_reproduction_{split}_{weather}_{index:08d}.json'
                        atomic_json(diagnostic, {'split': split, 'weather': weather, 'frame': index,
                                                'checks': differences, 'reasons': failures})
                        raise ValueError(f'Raw reproduction failed: {failures}; diagnostics={diagnostic}')
                    proxies = {name: _proxy_cache(dataset, batch, pred) for name, pred in pool.items()}
                    proposals, failure = check_frame(run, split, weather, index, batch, proxies,
                                                     context, runtime, cached, differences)
                    if failure:
                        raise ValueError('Output reproduction failed: '+str(failure))
                    ids, scores, corners, masks, names, old = proposals
                    pool_check = differences['proposal_pool']
                    if pool_check['reordered_positions_in_repeated_scores']:
                        print(f'evidence {split}/{weather} frame={index} retained S0 proposal order; '
                              f"repeated-score reordered positions={pool_check['reordered_positions_in_repeated_scores']}", flush=True)
                    for task in ('cls', 'reg'):
                        check = differences['output_only/'+task]
                        if check['rank_ties']:
                            print(f'evidence {split}/{weather} frame={index} verified near-tie ranks: '
                                  f"{check['rank_ties']}", flush=True)
                        if check['geometry_product']:
                            print(f'evidence {split}/{weather} frame={index} verified distance product: '
                                  f"{check['geometry_product']}", flush=True)
                    att_full, loo = leave_one_out(runtime, context['levels'])
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
