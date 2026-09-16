"""Collect all first-round evidence needed to test Q-A hypotheses H-A1..H-A6.

Development validation only. The frozen GSPR frontend is never modified. Each frame is
encoded once; ego-only and full-fusion predictions are replayed from the same encoding.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def _sorted_polygon(corners):
    xy = np.asarray(corners, dtype=np.float64)[:, :2]
    rounded = np.round(xy, 5)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    poly = xy[np.sort(unique_idx)]
    center = poly.mean(axis=0)
    order = np.argsort(np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0]))
    return poly[order], center


def _box_geometry(corners):
    c = np.asarray(corners, dtype=np.float64)
    poly, center = _sorted_polygon(c)
    if len(poly) != 4:
        raise ValueError(f"Expected 4 unique XY corners, got {len(poly)}")
    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.linalg.norm(edges, axis=1)
    a = edges[0] / max(lengths[0], 1e-9)
    b = (poly[-1] - poly[0])
    b = b / max(np.linalg.norm(b), 1e-9)
    half_a = 0.5 * lengths[0]
    half_b = 0.5 * np.linalg.norm(poly[-1] - poly[0])
    zmin, zmax = float(c[:, 2].min()), float(c[:, 2].max())
    return dict(poly=poly, center=center, axis_a=a, axis_b=b,
                half_a=half_a, half_b=half_b, zmin=zmin, zmax=zmax,
                length=float(max(2 * half_a, 2 * half_b)),
                width=float(min(2 * half_a, 2 * half_b)), height=zmax-zmin)


def _torch_box_mask(points, geom, scale_xy=1.0):
    import torch
    if points.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=points.device)
    center = torch.as_tensor(geom['center'], device=points.device, dtype=points.dtype)
    a = torch.as_tensor(geom['axis_a'], device=points.device, dtype=points.dtype)
    b = torch.as_tensor(geom['axis_b'], device=points.device, dtype=points.dtype)
    delta = points[:, :2] - center
    pa = (delta * a).sum(-1).abs()
    pb = (delta * b).sum(-1).abs()
    return ((pa <= geom['half_a'] * scale_xy + 1e-5)
            & (pb <= geom['half_b'] * scale_xy + 1e-5)
            & (points[:, 2] >= geom['zmin'] - 1e-3)
            & (points[:, 2] <= geom['zmax'] + 1e-3))


def _normalized_structure(points, weights, geom):
    """4x4 target-local footprint coverage/entropy from retained reliable slots."""
    if len(points) == 0:
        return dict(coverage4x4=0.0, quadrants=0.0, entropy4x4=0.0, span=0.0)
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    w = np.asarray(weights, dtype=np.float64)
    center = geom['center']
    u = (xy - center) @ geom['axis_a'] / max(geom['half_a'], 1e-6)
    v = (xy - center) @ geom['axis_b'] / max(geom['half_b'], 1e-6)
    keep = (np.abs(u) <= 1.001) & (np.abs(v) <= 1.001)
    u, v, w = u[keep], v[keep], w[keep]
    if not len(u):
        return dict(coverage4x4=0.0, quadrants=0.0, entropy4x4=0.0, span=0.0)
    ix = np.clip(((u + 1.0) * 2.0).astype(np.int64), 0, 3)
    iy = np.clip(((v + 1.0) * 2.0).astype(np.int64), 0, 3)
    hist = np.zeros((4, 4), dtype=np.float64)
    np.add.at(hist, (iy, ix), np.maximum(w, 0.0))
    occupied = hist > 1e-6
    coverage = float(occupied.mean())
    q = np.zeros((2, 2), dtype=bool)
    q[np.clip((v >= 0).astype(int), 0, 1), np.clip((u >= 0).astype(int), 0, 1)] = True
    quadrants = float(q.mean())
    p = hist.ravel()
    p = p[p > 0]
    if p.sum() > 0:
        p = p / p.sum()
        entropy = float(-(p * np.log(p + 1e-12)).sum() / math.log(16.0))
    else:
        entropy = 0.0
    span_u = min(2.0, float(u.max() - u.min())) / 2.0 if len(u) > 1 else 0.0
    span_v = min(2.0, float(v.max() - v.min())) / 2.0 if len(v) > 1 else 0.0
    return dict(coverage4x4=coverage, quadrants=quadrants,
                entropy4x4=entropy, span=float(math.sqrt(max(span_u * span_v, 0.0))))


def _angular_occlusion(geometries, target):
    """Cheap GT-only 2D angular-overlap proxy: closer boxes occluding the target bearing."""
    g = geometries[target]
    center = g['center']
    d = float(np.linalg.norm(center))
    if d <= 1e-4:
        return 0.0
    bearing = math.atan2(center[1], center[0])
    radius = 0.5 * math.hypot(g['length'], g['width'])
    half = math.asin(min(0.999, radius / max(d, radius + 1e-6)))
    if half <= 1e-6:
        return 0.0
    cover = 0.0
    for j, other in enumerate(geometries):
        if j == target:
            continue
        od = float(np.linalg.norm(other['center']))
        if od >= d - 1e-3 or od <= 1e-4:
            continue
        ob = math.atan2(other['center'][1], other['center'][0])
        oradius = 0.5 * math.hypot(other['length'], other['width'])
        oh = math.asin(min(0.999, oradius / max(od, oradius + 1e-6)))
        delta = abs(math.atan2(math.sin(ob - bearing), math.cos(ob - bearing)))
        overlap = max(0.0, half + oh - delta)
        cover += overlap / (2.0 * half)
    return float(min(1.0, cover))


def _grid_target_cells(geom, grid, lidar_range, scale=1.0):
    h, w = int(grid[0]), int(grid[1])
    xmin, ymin, _, xmax, ymax, _ = [float(x) for x in lidar_range]
    ys = ymin + (np.arange(h) + 0.5) * (ymax - ymin) / h
    xs = xmin + (np.arange(w) + 0.5) * (xmax - xmin) / w
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    delta = pts - geom['center'][None]
    pa = np.abs(delta @ geom['axis_a'])
    pb = np.abs(delta @ geom['axis_b'])
    mask = (pa <= geom['half_a'] * scale + 1e-6) & (pb <= geom['half_b'] * scale + 1e-6)
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        cx = int(np.clip((geom['center'][0] - xmin) / (xmax - xmin) * w, 0, w - 1))
        cy = int(np.clip((geom['center'][1] - ymin) / (ymax - ymin) * h, 0, h - 1))
        ids = np.asarray([cy * w + cx], dtype=np.int64)
    return ids


def _extract_retained(processed, rel, agents):
    evidence = rel['point_evidence'] + 1
    p = evidence[..., 0] / evidence.sum(-1)
    u = rel['point_uncertainty']
    valid = rel['point_valid_mask'].bool()
    coords = processed['voxel_coords'].long()
    xyz = processed['voxel_features'][..., :3]
    rows = []
    for agent in range(agents):
        m = valid & (coords[:, 0] == agent)[:, None]
        rows.append((xyz[m], p[m], u[m]))
    return rows


def _target_agent_stats(raw_cloud, retained, geom, region_ids, obs_agent, semantics_agent):
    import torch
    xyz, p, u = retained
    inside = _torch_box_mask(xyz, geom)
    box_xyz = xyz[inside]
    box_p = p[inside]
    box_u = u[inside]
    raw = raw_cloud[:, :3] if raw_cloud.numel() else raw_cloud.new_zeros((0, 3))
    raw_inside = _torch_box_mask(raw, geom)
    flat_obs = obs_agent.flatten(1)
    ids = torch.as_tensor(region_ids, device=obs_agent.device, dtype=torch.long)
    region_neff = torch.expm1(flat_obs[2, ids] * 8.0).sum()
    endpoint = flat_obs[4:8, ids]
    traversal = flat_obs[8:12, ids]
    conf = flat_obs[12, ids]
    sem = semantics_agent.flatten(1).index_select(1, ids)
    sem_norm = torch.linalg.vector_norm(sem, dim=0)
    structure = _normalized_structure(box_xyz.detach().cpu().numpy(), box_p.detach().cpu().numpy(), geom)
    return dict(
        raw_count=int(raw_inside.sum()),
        retained_count=int(inside.sum()),
        reliable_count=int((box_p >= 0.5).sum()) if box_p.numel() else 0,
        box_neff=float(box_p.sum()) if box_p.numel() else 0.0,
        box_mean_reliability=float(box_p.mean()) if box_p.numel() else 0.0,
        box_mean_uncertainty=float(box_u.mean()) if box_u.numel() else 1.0,
        region_neff=float(region_neff),
        coverage4x4=structure['coverage4x4'], quadrants=structure['quadrants'],
        entropy4x4=structure['entropy4x4'], span=structure['span'],
        endpoint_mean=float(endpoint.mean()) if endpoint.numel() else 0.0,
        endpoint_max=float(endpoint.max()) if endpoint.numel() else 0.0,
        traversal_mean=float(traversal.mean()) if traversal.numel() else 0.0,
        traversal_max=float(traversal.max()) if traversal.numel() else 0.0,
        known_fraction=float(flat_obs[3, ids].mean()) if ids.numel() else 0.0,
        confidence_mean=float(conf.mean()) if conf.numel() else 0.0,
        confidence_max=float(conf.max()) if conf.numel() else 0.0,
        semantic_norm_mean=float(sem_norm.mean()) if sem_norm.numel() else 0.0,
    )


def _background_summary(obs, geometries, grid, lidar_range):
    h, w = int(grid[0]), int(grid[1])
    gt_mask = np.zeros(h * w, dtype=bool)
    for geom in geometries:
        gt_mask[_grid_target_cells(geom, grid, lidar_range, 1.0)] = True
    import torch
    gt_mask_t = torch.as_tensor(gt_mask, device=obs.device)
    flat = obs.flatten(1)
    known = flat[3] > 0.5
    traversal = flat[8:12].mean(0)
    endpoint = flat[4:8].amax(0)
    background_zero = (~gt_mask_t) & (~known)
    target_zero = gt_mask_t & (~known)

    def summarize(mask):
        values = traversal[mask]
        endpoints = endpoint[mask]
        return dict(
            count=int(mask.sum()),
            traversal_positive=int((values > 0).sum()) if values.numel() else 0,
            traversal_ge_025=int((values >= .25).sum()) if values.numel() else 0,
            traversal_ge_050=int((values >= .50).sum()) if values.numel() else 0,
            traversal_ge_075=int((values >= .75).sum()) if values.numel() else 0,
            endpoint_positive=int((endpoints > 0).sum()) if endpoints.numel() else 0,
            traversal_sum=float(values.sum()) if values.numel() else 0.0,
        )
    return dict(background_zero=summarize(background_zero), target_zero=summarize(target_zero),
                total_background=int((~gt_mask_t).sum()), total_target_cells=int(gt_mask_t.sum()))


def main():
    parser = argparse.ArgumentParser(description='Q-A observation insufficiency diagnostic collector')
    parser.add_argument('--config', default='qa_observation_diagnostic/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--weather', required=True, choices=('clean', 'fog', 'rain', 'snow'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', type=int, default=0, help='First N validation frames; 0 means full split')
    parser.add_argument('--region-scale', type=float, default=1.0,
                        help='Communication-cell footprint scale used only to select cells overlapping a target')
    args = parser.parse_args()
    if args.smoke < 0 or args.region_scale <= 0:
        parser.error('--smoke must be >=0 and --region-scale >0')

    import torch
    from opencood.tools.train_utils import to_device
    from gspr_communication.evaluate import matched_objects
    from gspr_evidence import runtime as rt

    rt.verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    fixed_validation = Path('/data/scd/datasets/opv2v_official_data_dumping/validate').resolve()
    if Path(hypes['validate_dir']).resolve() != fixed_validation:
        raise ValueError('Q-A diagnostic is development-only and requires fixed OPV2V validation root')
    rt.seed_all(options['seed'])
    device = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, device)
    model.eval()
    rt.seed_all(options['seed'])
    ds, loader, indices = rt.make_loader(hypes, options, weather=args.weather, smoke=args.smoke)
    out = rt.new_output(args.output_dir)
    protocol = dict(
        schema=1, purpose='Q-A hypothesis diagnosis H-A1..H-A6', development_only=True,
        weather=args.weather, data_root=hypes['validate_dir'], sample_indices=indices,
        scene_ends=ds.len_record, online_weather=args.weather != 'clean', smoke=args.smoke,
        frontend_sha256=digest, frontend_config_sha256=rt.sha256(args.frontend_config),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        region_definition='communication-grid cell centers inside GT OBB; exact point metrics use GT OBB',
        reliability_definition='GSPR retained voxel slots; p=(evidence_reliable+1)/sum(alpha)',
        visibility_definition='existing endpoint/traversal proxy from gspr_evidence.geometry; NOT occupancy truth',
        occlusion_definition='GT-only 2D angular overlap from closer annotated vehicle boxes; proxy, not true occlusion label',
        detections='same frozen encoding; ego-only mask vs all-CAV full mask; IoU70 greedy match',
        test_data_used=False)
    rt.write_json(out / 'protocol.json', protocol)
    print(f'Q-A development {args.weather}: {len(indices)} frames, {len(ds.len_record)} scenes -> {out}', flush=True)

    start = time.perf_counter()
    target_count = 0
    number = 0
    with torch.no_grad(), (out / 'targets.jsonl').open('w', encoding='utf-8') as target_log, \
            (out / 'frames.jsonl').open('w', encoding='utf-8') as frame_log:
        for number, batch in enumerate(loader, 1):
            batch = to_device(batch, device)
            ego = batch['ego']
            index = int(ego['communication_sample_index'][0])
            scene = bisect.bisect_right(ds.len_record, index)
            inp = rt.input_branch(ego, args.weather)
            processed = inp['processed_lidar']
            base = model.engine.base
            rel = base.gspr(processed['voxel_features'], processed['voxel_num_points'], processed['voxel_coords'])
            encoded = model.encode(inp)
            agents = int(ego['record_len'].sum())
            retained = _extract_retained(processed, rel, agents)

            none_mask = model.empty_masks(encoded)
            full_mask = torch.ones_like(none_mask)
            none_pred, _ = model.detect(encoded, none_mask, serialize=False)
            full_pred, _ = model.detect(encoded, full_mask, serialize=False)
            if number == 1:
                original = base(inp)
                for output_key in ('psm', 'rm'):
                    torch.testing.assert_close(full_pred[output_key], original[output_key], atol=2e-4, rtol=2e-4)
            none_boxes, none_scores, gt = ds.post_process(batch, {'ego': none_pred})
            full_boxes, full_scores, full_gt = ds.post_process(batch, {'ego': full_pred})
            torch.testing.assert_close(gt, full_gt)
            none_match, none_fp = matched_objects(none_boxes, none_scores, gt)
            full_match, full_fp = matched_objects(full_boxes, full_scores, gt)

            gt_np = gt.detach().cpu().numpy()
            geometries = [_box_geometry(c) for c in gt_np]
            occ = [_angular_occlusion(geometries, j) for j in range(len(geometries))]
            background = _background_summary(encoded['obs'][0], geometries, model.grid, model.lidar_range)
            frame_log.write(json.dumps(dict(sample_index=index, scene=scene, weather=args.weather,
                agents=agents, gt=len(gt_np), ego_matched=len(none_match), full_matched=len(full_match),
                ego_fp=none_fp, full_fp=full_fp, background=background)) + '\n')

            for j, geom in enumerate(geometries):
                cell_ids = _grid_target_cells(geom, model.grid, model.lidar_range, args.region_scale)
                per_agent = []
                for a in range(agents):
                    per_agent.append(_target_agent_stats(inp['clouds'][a], retained[a], geom, cell_ids,
                        encoded['obs'][a], encoded['semantics'][a]))
                center = geom['center']
                row = dict(
                    sample_index=index, scene=scene, weather=args.weather, target_index=j,
                    center_x=float(center[0]), center_y=float(center[1]),
                    distance=float(np.linalg.norm(center)), box_length=geom['length'], box_width=geom['width'],
                    box_height=geom['height'], box_area=float(geom['length'] * geom['width']),
                    occlusion_proxy=occ[j], target_cells=len(cell_ids), agents=agents,
                    ego_detected=j in none_match, full_detected=j in full_match,
                    ego=per_agent[0], peers=per_agent[1:])
                target_log.write(json.dumps(row) + '\n')
                target_count += 1

            if number == 1 or number % 20 == 0:
                elapsed = time.perf_counter() - start
                print(f'{args.weather}: {number}/{len(indices)}; {elapsed/number:.2f}s/frame; '
                      f'ETA {(len(indices)-number)*elapsed/number/3600:.2f}h; targets={target_count}', flush=True)
                target_log.flush(); frame_log.flush()

    if number != len(indices) or target_count == 0:
        raise RuntimeError('Incomplete Q-A collection or no GT targets')
    rt.write_json(out / 'summary.json', dict(frames=number, targets=target_count, weather=args.weather))
    rt.verify_frozen()
    print(f'Complete {args.weather}: {out}', flush=True)


if __name__ == '__main__':
    main()
