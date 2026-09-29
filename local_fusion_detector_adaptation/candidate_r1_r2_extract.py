"""Server-only R1 independent-agent proposals and R2 same-anchor F/FD export.

The existing F top256 rows are immutable. This overlay contains inference-visible
features only; GT remains in the older training/evaluation extraction.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gspr_communication.runtime import (device, seed_all, seed_worker, sha256,
                                        verify_frozen, write_json)
from gspr_evidence import runtime as er
from local_fusion_utility_v2.fusion import predict_each_source
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import box_utils, common_utils, eval_utils

from .candidate_audit import _load_arm, _same_box_iou
from .candidate_ranker_pilot import _root


WEATHERS = ('clean', 'fog', 'rain', 'snow')
R1_NAMES = ('ego_best_iou', 'ego_best_score', 'peer_count',
            'peer_support_01', 'peer_support_03', 'peer_support_05',
            'peer_best_iou', 'peer_mean_best_iou', 'peer_best_match_score',
            'peer_mean_match_score', 'peer_best_score_iou',
            'peer_alternate_anchor_iou', 'peer_alternate_support_03',
            'peer_postnms_support_03', 'peer_postnms_best_iou',
            'peer_postnms_best_score', 'peer_scorepass_support_03',
            'nearest_matching_peer_distance_m', 'peer_matched_with_nearby_points',
            'ego_points_near_center', 'peer_max_points_near_center',
            'peer_sum_points_near_center',
            'nearest_peer_distance_m', 'peer_count_within_30m')
R1_MATCH_WIDTH = 19
R2_NAMES = ('fd_score', 'fd_minus_f_score', 'abs_score_gap',
            'fd_f_bev_iou', 'fd_f_center_shift', 'fd_f_size_shift_l1',
            'fd_f_yaw_shift_abs')


def _base_rows(root, weather, indices):
    path = root / weather / 'candidate_rows.jsonl'
    rows = {index: [] for index in indices}
    seen = set()
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            if row['arm'] != 'F':
                continue
            sample, cid = int(row['sample_index']), int(row['candidate_id'])
            if row['weather'] != weather or sample not in rows or (sample, cid) in seen:
                raise ValueError(f'{path}:{line_no}: wrong or duplicate F candidate')
            seen.add((sample, cid))
            rows[sample].append(row)
    if any(not 1 <= len(value) <= 256 for value in rows.values()):
        raise ValueError(f'{weather}: expected 1–256 F candidates per frame')
    return rows, sha256(path)


def _project(pp, decoded, matrix):
    corners = box_utils.boxes_to_corners_3d(decoded, order=pp.params['order'])
    corners = box_utils.project_box3d(corners, matrix)
    if not torch.isfinite(corners).all():
        raise ValueError('Non-finite projected source/FD corners')
    return corners


def _source_pools(pp, local, ego, source_topk, preselect):
    """Each CAV independently ranks its own anchors; no fused-anchor filtering."""
    scores = torch.sigmoid(local['psm'].permute(0, 2, 3, 1).reshape(
        local['psm'].shape[0], -1))
    decoded = pp.delta_to_boxes3d(local['rm'], ego['anchor_box'])
    if decoded.shape[:2] != scores.shape or not torch.isfinite(scores).all():
        raise ValueError('Independent source score/decoded layout differs')
    pools = []
    for source in range(len(scores)):
        n = scores.shape[1]
        width = min(n, max(source_topk, preselect))
        while True:
            ids = torch.topk(scores[source], width, sorted=True).indices
            corners = _project(pp, decoded[source, ids], ego['transformation_matrix'])
            keep = (box_utils.remove_large_pred_bbx(corners) &
                    box_utils.remove_bbx_abnormal_z(corners) &
                    box_utils.get_mask_for_boxes_within_range_torch(corners))
            chosen = torch.nonzero(keep, as_tuple=False).reshape(-1)[:source_topk]
            if len(chosen) >= source_topk or width == n:
                ids, corners = ids[chosen], corners[chosen]
                break
            width = min(n, width * 2)
        postnms = np.zeros(len(ids), dtype=bool)
        if len(ids):
            nms_ids = box_utils.nms_rotated(corners[:, :4, :2],
                                             scores[source, ids],
                                             pp.params['nms_thresh'])
            postnms[np.asarray(nms_ids, dtype=np.int64)] = True
        pools.append({
            'candidate_id': ids.detach().cpu().numpy().astype(np.int64),
            'score': scores[source, ids].detach().cpu().numpy().astype(np.float32),
            'corners': corners.detach().cpu().numpy()[:, :4, :2].astype(np.float32),
            'postnms': postnms,
        })
    return pools


def _best_match(candidate, candidate_id, pool):
    corners = pool['corners']
    if not len(corners):
        return (0., 0., -1, 0., 0., 0., -1, -1, -1)
    low, high = candidate.min(0), candidate.max(0)
    nearby = np.flatnonzero(((corners.max(1) > low) &
                             (corners.min(1) < high)).all(1))
    if not len(nearby):
        return (0., 0., -1, 0., 0., 0., -1, -1, -1)
    polygon = common_utils.convert_format(candidate[None])[0]
    polygons = common_utils.convert_format(corners[nearby])
    ious = np.asarray(common_utils.compute_iou(polygon, polygons), dtype=np.float32)
    if not np.isfinite(ious).all():
        raise ValueError('Non-finite independent candidate overlap')
    if float(ious.max()) <= 0:
        return (0., 0., -1, 0., 0., 0., -1, -1, -1)
    best = int(nearby[int(np.argmax(ious))])
    different = nearby[pool['candidate_id'][nearby] != candidate_id]
    alternate = 0.
    if len(different):
        alternate = float(np.max(ious[np.isin(nearby, different)]))
    post_iou, post_score, post_id, post_item = 0., 0., -1, -1
    post = np.flatnonzero(pool['postnms'][nearby])
    if len(post):
        post_best = int(post[int(np.argmax(ious[post]))])
        if float(ious[post_best]) > 0:
            item = int(nearby[post_best])
            post_item = item
            post_iou = float(ious[post_best])
            post_score = float(pool['score'][item])
            post_id = int(pool['candidate_id'][item])
    return (float(ious.max()), float(pool['score'][best]),
            int(pool['candidate_id'][best]), alternate,
            post_iou, post_score, post_id, best, post_item)


def _point_trees(ego, branch, count):
    """Inference-visible 3 m local-point proxy; not a visibility/ray claim."""
    from scipy.spatial import cKDTree

    key = 'evidence_clouds' if branch == 'clean' else 'evidence_weather_clouds'
    clouds = ego.get(key)
    if clouds is None or len(clouds) != count:
        raise ValueError('Per-source projected evidence clouds missing or misaligned')
    result = []
    for cloud in clouds:
        points = np.asarray(cloud.detach().cpu().numpy(), dtype=np.float32)
        if not points.size:
            result.append(None)
            continue
        if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points).all():
            raise ValueError('Invalid per-source point cloud')
        result.append(cKDTree(points[:, :2]))
    return result


def _r1(candidate, cid, pools, trees, source_positions):
    matches = [_best_match(candidate, cid, pool) for pool in pools]
    if len(trees) != len(matches) or source_positions.shape != (len(matches), 2):
        raise ValueError('Per-source point/pose/match layout differs')
    center = candidate.mean(0)
    distances = np.linalg.norm(source_positions - center, axis=1)
    nearby = np.asarray([len(tree.query_ball_point(center, 3.0))
                         if tree is not None else 0 for tree in trees],
                        dtype=np.int32)
    ego = matches[0]
    peers = matches[1:]
    ious = np.asarray([x[0] for x in peers], dtype=np.float32)
    scores = np.asarray([x[1] for x in peers], dtype=np.float32)
    alternate = np.asarray([x[3] for x in peers], dtype=np.float32)
    post_ious = np.asarray([x[4] for x in peers], dtype=np.float32)
    post_scores = np.asarray([x[5] for x in peers], dtype=np.float32)
    matched_distances = distances[1:][ious >= .3]
    vector = (ego[0], ego[1], len(peers), int((ious >= .1).sum()),
              int((ious >= .3).sum()), int((ious >= .5).sum()),
              float(ious.max(initial=0.)), float(ious.mean()) if len(ious) else 0.,
              float(scores.max(initial=0.)), float(scores.mean()) if len(scores) else 0.,
              float((scores * ious).max(initial=0.)),
              float(alternate.max(initial=0.)), int((alternate >= .3).sum()),
              int((post_ious >= .3).sum()), float(post_ious.max(initial=0.)),
              float(post_scores.max(initial=0.)),
              int(((post_ious >= .3) & (post_scores > .2)).sum()),
              float(matched_distances.min()) if len(matched_distances) else 1000.,
              int(((ious >= .3) & (nearby[1:] >= 3)).sum()),
              int(nearby[0]), int(nearby[1:].max(initial=0)),
              int(nearby[1:].sum()),
              float(distances[1:].min()) if len(peers) else 1000.,
              int((distances[1:] <= 30.).sum()))
    if len(vector) != len(R1_NAMES):
        raise AssertionError('R1 feature contract changed')
    return vector, [dict(source=i, iou=match[0], score=match[1],
                         independent_candidate_id=match[2],
                         alternate_anchor_iou=match[3],
                         postnms_iou=match[4], postnms_score=match[5],
                         postnms_candidate_id=match[6],
                         best_source_bev_corners=(pools[i]['corners'][match[7]].tolist()
                                                  if match[7] >= 0 else None),
                         postnms_source_bev_corners=(pools[i]['corners'][match[8]].tolist()
                                                     if match[8] >= 0 else None),
                         points_near_center=int(nearby[i]),
                         candidate_distance_to_source_m=float(distances[i]))
                    for i, match in enumerate(matches)]


def _r2(f_score, fd_score, f_box, fd_box, f_corners, fd_corners):
    pair_iou = _same_box_iou(f_corners[None], fd_corners[None])[0]
    yaw = np.arctan2(np.sin(fd_box[6] - f_box[6]),
                     np.cos(fd_box[6] - f_box[6]))
    vector = (fd_score, fd_score - f_score, abs(fd_score - f_score),
              pair_iou, float(np.linalg.norm(fd_box[:2] - f_box[:2])),
              float(np.abs(fd_box[3:6] - f_box[3:6]).sum()), float(abs(yaw)))
    return vector


def _sync(target):
    if target.type == 'cuda':
        torch.cuda.synchronize(target)


def export(args):
    verify_frozen()
    base = _root(args.base_root)
    base_meta = json.loads((base / 'candidate_audit.json').read_text(encoding='utf-8'))
    split = 'train' if base_meta.get('split') == 'train' else 'validation'
    indices = [int(x) for x in base_meta['sample_indices' if split == 'train'
                                          else 'validation_indices']]
    scene_map = base_meta['scene_map' if split == 'train' else 'validation_scene_map']
    if not indices or indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError('Base extraction indices must be unique and ordered')
    if not base_meta.get('candidate_feature_cache') or base_meta.get('audit_pool') != 'top_256':
        raise ValueError('R1/R2 requires F top256 rows and feature cache')
    run = Path(args.run).resolve()
    pilot = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    options, hypes = er.load_config(args.v3_config, args.frontend_config)
    if split == 'train':
        split_seed = int(base_meta['selection_seed'])
        loader_seed = split_seed
    else:
        split_seed = int(pilot['pilot']['seed']) + 1
        loader_seed = int(options['seed'])
    target = device()
    model, digest = er.load_model(hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    contract = v3rt.contract(options, args.frontend_config, digest)
    if (digest != pilot['frontend_sha256'] or digest != base_meta['frontend_sha256']
            or contract != pilot['v3_contract'] or
            sha256(args.v3_checkpoint) != pilot['v3_checkpoint_sha256'] or
            sha256(args.v3_checkpoint) != base_meta['v3_checkpoint_sha256']):
        raise ValueError('Frontend/v3 frozen contract differs from base extraction')
    source, checkpoint = v3rt.load(args.v3_checkpoint, contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Expected frozen residual v3 checkpoint')
    arm_paths = {name: run / (name + '.pth') for name in ('F', 'F+D')}
    for name, path in arm_paths.items():
        expected = base_meta['arm_sha256'].get(name) or pilot.get('arm_sha256', {}).get(name)
        if expected is not None and sha256(path) != expected:
            raise ValueError(f'{name} checkpoint differs from pilot/base extraction')
    arms = {name: _load_arm(arm_paths[name], source, model.engine.base,
                            target, name == 'F+D') for name in arm_paths}
    for arm in arms.values():
        arm.requires_grad_(False).eval()
    del source
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        'split': split, 'base_root': str(base),
        'base_audit_sha256': sha256(base / 'candidate_audit.json'),
        'sample_indices': indices, 'scene_map': scene_map,
        'frontend_sha256': digest,
        'v3_checkpoint_sha256': sha256(args.v3_checkpoint),
        'arm_sha256': {name: sha256(path) for name, path in arm_paths.items()},
        'implementation_sha256': sha256(Path(__file__).resolve()),
        'r1_names': R1_NAMES, 'r2_names': R2_NAMES,
        'source_topk': args.source_topk, 'source_preselect': args.source_preselect,
        'base_split_seed': split_seed, 'base_loader_seed': loader_seed,
        'point_proxy_radius_m': 3.0,
        'source_semantics': ('Each source uses its own frozen prediction on aligned '
                             'features, independently ranked across all anchors. '
                             'Post-NMS is the same rotated NMS within its bounded pool. '
                             'Missing match means no match in this bounded pool, not '
                             'proven absence or visibility.'),
        'point_proxy_semantics': ('Projected per-source LiDAR points within 3m '
                                  'of fused BEV center; local sampling proxy, '
                                  'not a ray visibility or occlusion estimate.'),
        'distance_semantics': ('Source-to-ego pose translations and fused BEV center '
                               'provide candidate-to-source distances; 1000m is the '
                               'missing matched-peer distance sentinel.'),
        'conditions': {},
    }
    write_json(out / 'protocol.json', {k: v for k, v in report.items()
                                       if k != 'conditions'})
    for weather in WEATHERS:
        seed_all(split_seed + (WEATHERS.index(weather) if split == 'train' else 0))
        rows, base_hash = _base_rows(base, weather, indices)
        ds, _, _ = er.make_loader(hypes, options, train=split == 'train',
                                  weather=weather)
        ends = np.asarray(ds.len_record, dtype=np.int64)
        actual_scenes = {str(i): int(np.searchsorted(ends, i, side='right'))
                         for i in indices}
        if actual_scenes != {str(k): int(v) for k, v in scene_map.items()}:
            raise ValueError('Scene mapping changed since base extraction')
        generator = torch.Generator().manual_seed(loader_seed)
        loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False,
                            num_workers=int(options['workers']),
                            collate_fn=ds.collate_batch_test,
                            worker_init_fn=seed_worker, generator=generator)
        condition = {'frames': 0, 'candidates': 0, 'source_pool_shortfalls': 0,
                     'base_rows_sha256': base_hash,
                     'extra_compute_seconds': {'r1_source_head_pool_match': 0.,
                                               'r2_fd_head_decode': 0.}}
        stats = {name: {v: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
                        for v in (.3, .5, .7)} for name in ('F', 'F+D')}
        folder = out / weather
        folder.mkdir()
        with torch.no_grad(), (folder / 'overlay_rows.jsonl').open(
                'w', encoding='utf-8') as stream:
            for pos, batch in enumerate(loader):
                batch = to_device(batch, target)
                sample = int(batch['ego']['communication_sample_index'][0])
                if sample != indices[pos]:
                    raise RuntimeError('DataLoader changed base candidate frame order')
                ctx = v3rt.context(model, batch['ego'],
                                   'clean' if weather == 'clean' else 'weather',
                                   verify=pos == 0)
                f, _ = arms['F'].predict(None, ctx['levels'])
                frame_rows = rows[sample]
                ids = torch.as_tensor([int(r['candidate_id']) for r in frame_rows],
                                      dtype=torch.long, device=target)
                pp = ds.post_processor
                f_scores = torch.sigmoid(f['psm'].permute(0, 2, 3, 1).reshape(-1))
                f_boxes = pp.delta_to_boxes3d(f['rm'], batch['ego']['anchor_box'])[0, ids]
                _sync(target)
                r2_start = time.perf_counter()
                fd, _ = arms['F+D'].predict(None, ctx['levels'])
                fd_scores = torch.sigmoid(fd['psm'].permute(0, 2, 3, 1).reshape(-1))
                fd_boxes = pp.delta_to_boxes3d(fd['rm'], batch['ego']['anchor_box'])[0, ids]
                fd_corners = _project(pp, fd_boxes, batch['ego']['transformation_matrix'])
                _sync(target)
                condition['extra_compute_seconds']['r2_fd_head_decode'] += (
                    time.perf_counter() - r2_start)
                r1_start = time.perf_counter()
                local = predict_each_source(model.engine.base, ctx['levels'])
                pools = _source_pools(pp, local, batch['ego'],
                                      args.source_topk, args.source_preselect)
                branch = 'clean' if weather == 'clean' else 'weather'
                trees = _point_trees(batch['ego'], branch, len(pools))
                transforms = batch['ego']['communication_transforms']
                if transforms.shape != (len(pools), 4, 4):
                    raise ValueError('Source pose/feature count differs')
                source_positions = transforms[:, :2, 3].detach().cpu().numpy()
                if not np.isfinite(source_positions).all():
                    raise ValueError('Non-finite source positions')
                _sync(target)
                condition['extra_compute_seconds']['r1_source_head_pool_match'] += (
                    time.perf_counter() - r1_start)
                condition['source_pool_shortfalls'] += sum(
                    len(pool['candidate_id']) < args.source_topk for pool in pools)
                if len(pools) != int(frame_rows[0]['source_count']):
                    raise ValueError('Source count differs from base candidate row')
                for j, raw in enumerate(frame_rows):
                    f_score = float(f_scores[ids[j]])
                    f_box = f_boxes[j].detach().cpu().numpy()
                    base_box = np.asarray(raw['fused_decoded_box'], dtype=np.float32)
                    fc = np.asarray(raw['fused_bev_corners'], dtype=np.float32)
                    if (base_box.shape != (7,) or fc.shape != (4, 2) or
                            not np.isfinite(fc).all() or not np.isfinite(base_box).all()):
                        raise ValueError(f'{weather}/{sample}: invalid base F geometry')
                    score_delta = abs(f_score - float(raw['score']))
                    box_delta = float(np.max(np.abs(f_box - base_box)))
                    if (score_delta > 1e-5 or
                            not np.allclose(f_box, base_box, atol=1e-4, rtol=1e-5)):
                        raise ValueError(f'{weather}/{sample}/anchor={int(ids[j])}: '
                                         f'F differs from base; score_delta={score_delta:.8g}, '
                                         f'decoded_box_max_abs_delta={box_delta:.8g}, '
                                         f'base_seed={split_seed}, loader_seed={loader_seed}')
                    fd_box = fd_boxes[j].detach().cpu().numpy()
                    fd_score = float(fd_scores[ids[j]])
                    match_start = time.perf_counter()
                    r1, matches = _r1(fc, int(ids[j]), pools, trees,
                                      source_positions)
                    condition['extra_compute_seconds']['r1_source_head_pool_match'] += (
                        time.perf_counter() - match_start)
                    r2 = _r2(f_score, fd_score, f_box, fd_box,
                             fc, fd_corners[j, :4, :2].detach().cpu().numpy())
                    if not np.isfinite(r1).all() or not np.isfinite(r2).all():
                        raise ValueError('Non-finite R1/R2 inference features')
                    stream.write(json.dumps({
                        'weather': weather, 'sample_index': sample,
                        'candidate_id': int(ids[j]), 'source_count': len(pools),
                        'f_score_check': f_score, 'r1': r1, 'r2': r2,
                        'matches': matches,
                    }, ensure_ascii=False) + '\n')
                if split == 'validation':
                    for name, prediction in (('F', f), ('F+D', fd)):
                        output = ds.post_process(batch, {'ego': prediction})
                        for threshold in stats[name]:
                            eval_utils.caluclate_tp_fp(*output, stats[name], threshold)
                condition['frames'] += 1
                condition['candidates'] += len(frame_rows)
                if pos == 0 or (pos + 1) % 20 == 0:
                    print(f'{split}/{weather}: {pos + 1}/{len(indices)} R1/R2 export',
                          flush=True)
        if split == 'validation':
            from ceif_audit.scoring import ap_values
            condition['original_ap'] = {name: ap_values(stats[name], eval_utils)
                                        for name in stats}
            for name in stats:
                expected = base_meta['conditions'][weather]['swap_ap'][
                    'F_fusion_F_detector' if name == 'F' else
                    'FD_fusion_FD_detector']
                for metric in ('ap30', 'ap50', 'ap70'):
                    if abs(condition['original_ap'][name][metric] -
                           expected[metric]) > args.reproduction_tolerance:
                        raise RuntimeError(f'{weather}/{name}: original AP differs')
        if condition['frames'] != len(indices):
            raise RuntimeError('Incomplete source/model overlay extraction')
        report['conditions'][weather] = condition
        write_json(folder / 'summary.json', condition)
        del ds, loader
    if sha256(Path(__file__).resolve()) != report['implementation_sha256']:
        raise RuntimeError('Extractor source changed while running')
    write_json(out / 'overlay.json', report)
    verify_frozen()
    print('R1/R2 OVERLAY COMPLETE:', out / 'overlay.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('base-root', 'run', 'frontend-config', 'frontend-checkpoint',
                  'v3-checkpoint', 'output'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--source-topk', type=int, default=256)
    parser.add_argument('--source-preselect', type=int, default=2048)
    parser.add_argument('--reproduction-tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    if args.source_topk < 1 or args.source_preselect < args.source_topk:
        parser.error('Source pool sizes must be positive and preselect >= topk')
    export(args)


if __name__ == '__main__':
    main()
