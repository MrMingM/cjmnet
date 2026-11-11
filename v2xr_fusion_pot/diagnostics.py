"""Formal evaluation, inference interventions, and fixed-Clean target attribution."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from common import atomic_json, identity, read_json, sha256
from checkpoint import seed
from runtime import DATA, build, loader
from metrics import evaluate


LR0 = DATA / 'logs/v2xr-lr0-v3'
WEATHERS = ('clean', 'fog', 'snow')
MASS_BINS = [0., .01, .1, .25, .5, .75, 1., 1.5, 2., 3., 4., 8.]


def verify_gt(out):
    reference = read_json(LR0 / 'final_evaluation/gt_manifest.json')['frames']
    actual = read_json(out / 'gt_manifest.json')['frames']
    fields = ('condition', 'index', 'frame_key', 'fixed_ids', 'fixed_gt_identity')
    if (len(actual) != 3*829 or any(
            any(row[key] != old[key] for key in fields) for row, old in zip(actual, reference, strict=True))
            or any(sum(row['fixed_count'] for row in actual if row['condition'] == w) != 6235 for w in WEATHERS)):
        raise ValueError(f'Fixed Clean target IDs or box identities changed: {out}')


def evaluate_formal(destination, config, contract, group, rows, factory):
    from opencood.hypes_yaml import yaml_utils

    if len(rows) != 20:
        raise ValueError('Formal evaluation requires 20 committed epochs')
    best = min(rows, key=lambda row: (-row['clean_AP']['O']['ap70'], row['epoch']))
    weight = Path(best['checkpoint'])
    digest = sha256(weight)
    if digest != best['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    selected = dict(epoch=best['epoch'], checkpoint=str(weight), sha256=digest,
                    clean_selection_ap70=best['clean_AP']['O']['ap70'])
    atomic_json(destination / 'selected.json', selected)
    seed(20261016)
    model = factory(yaml_utils.load_yaml(str(config)), group)
    model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
    out = destination / 'final_evaluation'
    metrics = evaluate(model, None, config, out, WEATHERS, 829, contract, digest)
    verify_gt(out)
    result = dict(status='complete', experiment=group, contract=contract, selected=selected,
                  parameters=sum(p.numel() for p in model.parameters()),
                  peak_train_gpu_allocated_bytes=max(row['peak_train_gpu_allocated_bytes'] for row in rows),
                  updates=sum(row['train_updates'] for row in rows), evaluation=metrics)
    atomic_json(destination / 'FINAL_RESULTS.json', result)
    return result


def histogram(counter, label, array):
    values = np.histogram(array, bins=MASS_BINS)[0]
    counter.update({f'{label}_{i}': int(v) for i, v in enumerate(values)})
    counter[label + '_over_8'] += int(np.count_nonzero(array > MASS_BINS[-1]))


def distance_bin(distance, cfg):
    first, second = cfg['distance_edges_m']
    return f'0-{first}' if distance < first else f'{first}-{second}' if distance < second else f'{second}+'


def transport_counts(tensors, distance, stats, cfg):
    values = {k: v.cpu().numpy() for k, v in tensors.items() if torch.is_tensor(v)}
    for side, mass, residual in (('ego', 'row', 'row_remaining'), ('neighbor', 'col', 'col_remaining')):
        observed = values[side] > 0
        lidar, radar = observed[..., 0], observed[..., 1]
        for dist in np.unique(distance):
            for source, mask in (('lidar_only', lidar & ~radar), ('radar_only', radar & ~lidar),
                                 ('both', lidar & radar)):
                select = mask & (distance == dist)
                row = stats[f'{side}/{dist}/{source}']
                m, u = values[mass][select], values[residual][select]
                row['positions'] += len(m)
                row['matched'] += int(np.count_nonzero(m >= cfg['match_quality_threshold']))
                row['low_quality'] += int(np.count_nonzero((m >= cfg['match_rejected_threshold'])
                                                           & (m < cfg['match_quality_threshold'])))
                row['rejected_or_unmatched'] += int(np.count_nonzero(m < cfg['match_rejected_threshold']))
                row['mass_sum'] += float(m.sum())
                row['remaining_mass_sum'] += float(u.sum())
                row['expanded_mass_sum'] += float(np.maximum(m - 1., 0.).sum())
                histogram(row, 'mass', m)
                histogram(row, 'remaining', u)
                if side == 'neighbor':
                    eligible = values['candidate'][select]
                    row['unmatched_candidates'] += int(np.count_nonzero(eligible & (u > 0)))
                    row['candidate_without_ego_support'] += int(np.count_nonzero(
                        eligible & ~values['near_ego'][select] & (u > 0)))
                    histogram(row, 'unmatched_weight', values['unmatched_weight'][select])
                else:
                    n = values['allowed_edges'][select]
                    ambiguous = n > 1
                    peak = values['peak_fraction'][select]
                    effective = values['effective_edges'][select]
                    row['multi_candidate_rows'] += int(ambiguous.sum())
                    row['multi_candidate_concentrated'] += int(np.count_nonzero(ambiguous & (peak >= .9)))
                    row['multi_candidate_near_uniform'] += int(np.count_nonzero(
                        ambiguous & (effective >= .9*n)))
    return values


def object_support(ego, reference, residual, geometry, cfg):
    """Inspect real per-vehicle points inside fixed boxes after predictions are made."""
    from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

    boxes = np.asarray([r['box'] for r in reference], dtype=np.float32).reshape(-1, 7)
    box_xyz_lwh = boxes[:, [0, 1, 2, 5, 4, 3, 6]]
    counts = {}
    for sensor in ('lidar', 'radar'):
        per_vehicle = []
        for points in ego[f'projected_{sensor}'][0]:
            xyz = points[:, :3]
            if torch.is_tensor(xyz):
                xyz = xyz.detach().cpu().numpy()
            inside = points_in_boxes_cpu(xyz.astype(np.float32, copy=False), box_xyz_lwh) if len(xyz) and len(boxes) else None
            per_vehicle.append(np.asarray(inside).sum(1) if inside is not None else np.zeros(len(boxes), dtype=np.int64))
        counts[sensor] = np.stack(per_vehicle)
    xmin, ymin, step_x, step_y, height, width = geometry
    x = np.rint((boxes[:, 0] - xmin) / step_x).astype(int).clip(0, width-1)
    y = np.rint((boxes[:, 1] - ymin) / step_y).astype(int).clip(0, height-1)
    rows = {}
    for i, obj in enumerate(reference):
        lidar, radar = counts['lidar'][:, i], counts['radar'][:, i]
        ego_points = int(lidar[0] + radar[0])
        neighbor_points = int(lidar[1:].sum() + radar[1:].sum())
        r = float(residual[0, y[i]*width+x[i]]) if residual is not None else 0.
        rows[str(obj['object_id'])] = dict(
            source=('neighbor_only' if neighbor_points and not ego_points else 'both' if neighbor_points
                    else 'ego_only' if ego_points else 'no_points'), ego_lidar=int(lidar[0]),
            ego_radar=int(radar[0]), neighbor_lidar=int(lidar[1:].sum()), neighbor_radar=int(radar[1:].sum()),
            distance=distance_bin(float(np.linalg.norm(boxes[i, :2])), cfg), center_unmatched_rms=r)
    return rows


def observe(model, config, out, cfg, collect_objects):
    from opencood.tools import train_utils

    model.eval()
    model.fusion.observe = model.fusion.profile = True
    height, width = model.coarse_shape
    overall, objects = {}, {}
    for weather in WEATHERS:
        dataset, native_cfg = build(config, condition=weather, visualize=True)
        minimum = native_cfg['model']['args']['lidar_range']
        vx, vy, _ = native_cfg['model']['args']['voxel_size']
        geometry = (minimum[0]+vx/2, minimum[1]+vy/2, model.stride*vx, model.stride*vy, height, width)
        distances = np.hypot(geometry[0] + np.arange(width)[None, :]*geometry[2],
                             geometry[1] + np.arange(height)[:, None]*geometry[3])
        strata = np.vectorize(lambda d: distance_bin(d, cfg))(distances.ravel())
        stats, totals = defaultdict(Counter), Counter()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for i, batch in enumerate(loader(dataset, evaluation=True, limit=829)):
                ego = train_utils.to_device(batch['ego'], torch.device('cuda:0'))
                model(ego)
                tensors = model.fusion.report
                if tensors is not None:
                    start, stop = model.fusion.elapsed
                    stop.synchronize()
                    elapsed = start.elapsed_time(stop)
                    totals['fusion_extra_gpu_ms'] += elapsed
                    totals['frames_with_pairs'] += 1
                    totals['pair_count'] += len(tensors['row'])
                    totals['max_solver_residual'] = max(totals['max_solver_residual'], float(tensors['residual']))
                    totals['max_pair_mass_ratio'] = max(totals['max_pair_mass_ratio'], float(tensors['mass_ratio_max']))
                    totals['iterations'] = tensors['iterations']
                    totals['nonfinite_count'] = 0  # The solver raises before returning any nonfinite result.
                    values = transport_counts(tensors, strata, stats, cfg)
                    for path in ('matched_rms', 'unmatched_rms'):
                        if path in values:
                            rms = values[path]
                            totals[path + '_sum'] += float(rms.sum())
                            totals[path + '_cells'] += rms.size
                            totals[path + '_nonzero_cells'] += int(np.count_nonzero(rms))
                    for col in values['col']:
                        mass = col.sum()
                        if mass <= 0:
                            totals['zero_transport_pairs'] += 1
                        else:
                            totals['top_1_percent_col_share_sum'] += float(
                                np.sort(col)[-max(1, col.size//100):].sum()/mass)
                            active = max(1, int(np.count_nonzero(col)))
                            totals['top_1_percent_active_col_share_sum'] += float(
                                np.sort(col)[-max(1, active//100):].sum()/mass)
                if collect_objects:
                    clean = read_json(LR0 / 'final_evaluation/predictions/clean' / f'{i:06d}.json')
                    if list(clean['frame_key']) != list(dataset.keys[i]):
                        raise ValueError('Diagnostic frame differs from fixed target reference')
                    residual = tensors['unmatched_rms'] if tensors is not None else None
                    objects[f'{weather}/{i}'] = object_support(batch['ego'], clean['official_gt'],
                        residual.cpu().numpy() if residual is not None else None, geometry, cfg)
                if (i + 1) % 100 == 0:
                    print(f'DIAG {out.name} {weather} {i+1}/829', flush=True)
        if i + 1 != 829:
            raise ValueError('Incomplete weather diagnosis')
        for row in stats.values():
            if row['positions']:
                for key in ('matched', 'low_quality', 'rejected_or_unmatched', 'unmatched_candidates'):
                    row[key + '_fraction'] = row[key] / row['positions']
        overall[weather] = dict(by_distance_and_source=stats, totals=totals,
            extra_fusion_ms_per_multicar_frame=(totals['fusion_extra_gpu_ms'] /
                                               max(1, totals['frames_with_pairs'])),
            instrumented_peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    return overall, objects


def fixed_metrics(out):
    rows = {}
    with (out / 'per_frame_metrics.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            rows[f"{row['condition']}/{row['index']}"] = row['protocols']['F']
    return rows


def false_boxes(out, frame, metrics):
    weather, index = frame.split('/')
    saved = read_json(out / 'predictions' / weather / f'{int(index):06d}.json')
    matched = {pair['prediction_index'] for pair in metrics['matched_pairs']['0.7']}
    return [box for idx, box in enumerate(saved['prediction_boxes'] or ()) if idx not in matched]


def added_false_boxes(before, after):
    """Spatial identity proxy: greedy one-to-one BEV IoU >= .7 between false boxes."""
    from opencood.utils import common_utils

    if not before or not after:
        return len(after)
    old = common_utils.convert_format(np.asarray(before))
    new = common_utils.convert_format(np.asarray(after))
    used = set()
    for box in new:
        best, best_iou = None, .7
        for i, other in enumerate(old):
            if i in used or not box.intersects(other):
                continue
            union = box.union(other).area
            iou = box.intersection(other).area / union if union else 0.
            if iou >= best_iou:
                best, best_iou = i, iou
        if best is not None:
            used.add(best)
    return len(after) - len(used)


def compare_targets(old_dir, new_dir, context, detailed=False):
    old, new = fixed_metrics(old_dir), fixed_metrics(new_dir)
    if old.keys() != new.keys():
        raise ValueError('Evaluation frames differ')
    counts, changed = defaultdict(lambda: defaultdict(Counter)), []
    for frame, row in new.items():
        weather = frame.split('/')[0]
        old_ids = {str(p['object_id']) for p in old[frame]['matched_pairs']['0.7']}
        new_ids = {str(p['object_id']) for p in row['matched_pairs']['0.7']}
        counts[weather]['all']['old_fp70'] += old[frame]['thresholds']['0.7']['fp']
        counts[weather]['all']['new_fp70'] += row['thresholds']['0.7']['fp']
        if detailed:
            counts[weather]['all']['added_fp70_spatial'] += added_false_boxes(
                false_boxes(old_dir, frame, old[frame]), false_boxes(new_dir, frame, row))
        for oid, obj in context[frame].items():
            a, b = oid in old_ids, oid in new_ids
            event = 'recovered' if b and not a else 'lost' if a and not b else 'kept' if a else 'missed_both'
            for label in ('all', obj['source'], obj['distance'], obj['source']+'/'+obj['distance']):
                counts[weather][label][event] += 1
            if detailed and a != b:
                changed.append(dict(frame=frame, object_id=oid, event=event, **obj))
    return dict(counts=counts, changed=changed)


def report(root, config, contract, results, factory, settings):
    from opencood.hypes_yaml import yaml_utils

    cfg = yaml_utils.load_yaml(str(config))
    ablations, transport_stats, contexts = {}, {}, {}
    evaluation_dirs = {}
    for group in ('P0', 'P1', 'P2', 'P3'):
        torch.cuda.empty_cache()
        seed(20261016)
        model = factory(cfg, group if group != 'P0' else 'P2')
        weight = (LR0 / 'epochs/18/weights.pth' if group == 'P0'
                  else Path(results[group]['selected']['checkpoint']))
        digest = sha256(weight)
        state = torch.load(weight, map_location='cpu', weights_only=True)
        if group == 'P0':
            missing, unexpected = model.load_state_dict(state, strict=False)
            assert not unexpected and all(k.startswith('backbone.fuse_modules.2.') for k in missing)
            model.fusion.mode = 'module_off'
        else:
            model.load_state_dict(state, strict=True)
        del state
        default = model.fusion.mode, model.fusion.solver, dict(model.fusion.cfg)
        full = (LR0 / 'final_evaluation' if group == 'P0' else root / group / 'final_evaluation')
        evaluation_dirs[group] = full
        if group != 'P0':
            path = root / group / 'TRANSPORT.json'
            diagnostic_id = identity([contract, digest, settings])
            if path.exists():
                observed = read_json(path)
                if observed['identity'] != diagnostic_id:
                    raise ValueError(f'{group} mechanism evidence identity changed')
            else:
                stats, targets = observe(model, config, full, settings, group == 'P3')
                observed = dict(identity=diagnostic_id, mass_bins=MASS_BINS, stats=stats, targets=targets)
                atomic_json(path, observed)
            transport_stats[group] = observed['stats']
            if group == 'P3':
                contexts = observed['targets']
        model.fusion.observe = model.fusion.profile = False

        variants = {'single': dict(single=True)}
        if group == 'P0':
            variants['same_weights_gpu4'] = {}
        if group == 'P2':
            variants['dustbin_inference'] = dict(solver='dustbin')
        if group == 'P3':
            variants.update(matched_only=dict(mode='matched_only'), unmatched_only=dict(mode='unmatched_only'),
                module_off=dict(mode='module_off'), window5=dict(window=5), radar_off=dict(radar_off=True))
            variants.update({f'epsilon_{value}': dict(epsilon=value, iterations=40)
                             for value in settings['ablation_epsilon']})
        for name, changes in variants.items():
            model.single_vehicle = changes.get('single', False)
            model.radar_off = changes.get('radar_off', False)
            model.fusion.mode = changes.get('mode', default[0])
            model.fusion.solver = changes.get('solver', default[1])
            if model.fusion.solver == 'dustbin':
                model.fusion.bin_score = torch.nn.Parameter(torch.tensor(
                    settings['dustbin_score'], device='cuda:0'), requires_grad=False)
            model.fusion.cfg.clear()
            model.fusion.cfg.update(default[2])
            model.fusion.cfg.update({k: changes[k] for k in ('epsilon', 'iterations', 'window') if k in changes})
            seed(20261016)
            out = root / 'ablations' / group / name
            ablations[f'{group}/{name}'] = evaluate(model, None, config, out, WEATHERS, 829,
                identity([contract, group, name, changes]), digest)
            verify_gt(out)
            evaluation_dirs[f'{group}/{name}'] = out
            print(f'ABLATION COMPLETE {group}/{name}', flush=True)
        del model

    transitions = {}
    for group in ('P0', 'P1', 'P2', 'P3'):
        transitions[f'{group} single to multi'] = compare_targets(
            evaluation_dirs[f'{group}/single'],
            evaluation_dirs['P0/same_weights_gpu4'] if group == 'P0' else evaluation_dirs[group], contexts)
    transitions['P2 to P3'] = compare_targets(evaluation_dirs['P2'], evaluation_dirs['P3'], contexts, True)
    transitions['same P3 weights, unmatched off to on'] = compare_targets(
        evaluation_dirs['P3/matched_only'], evaluation_dirs['P3'], contexts, True)
    atomic_json(root / 'FINAL_RESULTS.json', dict(status='complete', contract=contract, groups=results,
        ablations=ablations, transport=transport_stats, target_transitions=transitions))
    write_report(root, results, ablations, transitions, transport_stats)


def write_report(root, results, ablations, transitions, transport):
    lines = ['# POT-CF 正式对照', '', '各天气 829 帧。F 对照固定双模态 Clean 的 6,235 个目标。',
             'P0 使用海光卡 1 的已归档正式结果。相同权重在卡 4 的复测见消融表。',
             '', '| 组别 | 天气 | 协议 | AP30 | AP50 | AP70 | 推理帧/秒 | 峰值显存 GiB |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    tables = {group: value['evaluation'] for group, value in results.items()} | ablations
    for label, result in tables.items():
        for weather in WEATHERS:
            perf = result['performance'][weather]
            for protocol, ap in result['results'][weather]['AP'].items():
                lines.append(f'| {label} | {weather} | {protocol} | {ap["ap30"]:.6f} | {ap["ap50"]:.6f} | '
                             f'{ap["ap70"]:.6f} | {perf["inference_fps"]:.2f} | '
                             f'{perf["peak_gpu_allocated_bytes"]/2**30:.3f} |')
    lines += ['', '## 同一目标集合的逐目标变化', '',
              '| 对照 | 天气 | 恢复 | 丢失 | 恢复且仅邻车有实点 | 仅邻车有实点的全部目标 | 误检数变化 | 新增误检（空间去重） |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for label, report in transitions.items():
        for weather, groups in report['counts'].items():
            a, n = groups['all'], groups.get('neighbor_only', {})
            lines.append(f'| {label} | {weather} | {a["recovered"]} | {a["lost"]} | {n.get("recovered",0)} | '
                         f'{sum(n.values())} | {a["new_fp70"]-a["old_fp70"]} | '
                         f'{a.get("added_fp70_spatial", "—")} |')
    lines += ['', '“仅邻车有实点”按当帧两路点是否落在固定真值框内统计。逐目标变化和框中心处的实际残差在 JSON 中。'
              'P3 同权重路径开关用于区分未匹配路径的影响，P2 与 P3 的训练差异单独计。',
              '', '## 求解器与额外开销', '',
              '| 组别 | 天气 | 多车帧平均新增融合毫秒 | 最大求解残差 | 零传输车对 | 车对数 |',
              '|---|---|---:|---:|---:|---:|']
    for group, weather_stats in transport.items():
        for weather, row in weather_stats.items():
            total = row['totals']
            lines.append(f'| {group} | {weather} | {row["extra_fusion_ms_per_multicar_frame"]:.3f} | '
                         f'{total["max_solver_residual"]:.6g} | {total.get("zero_transport_pairs",0)} | '
                         f'{total["pair_count"]} |')
    lines += ['', '各组 TRANSPORT.json 保存按天气、距离和传感器来源拆分的质量直方图，'
              '并记录局部权重集中、接近均匀、质量扩张和实际路径残差。']
    target = root / 'FINAL_COMPARISON.md'
    temporary = target.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    temporary.replace(target)
