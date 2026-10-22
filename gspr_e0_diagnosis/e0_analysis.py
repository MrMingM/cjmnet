"""Streaming reliability summaries and frozen-match object-level error analysis."""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
from e0_common import WEATHERS, ARMS, atomic, read, rows, sha
from e0_numeric import covariance_summary, merge_distribution, range_group


def arm_folder(output, arm, smoke=False):
    return output / ('stages/smoke/'+arm if smoke else
        'stages/reliability/R0' if arm == 'R0' else 'stages/interventions/'+arm)


def reliability_summary(output, smoke=False):
    source = arm_folder(output, 'R0', smoke)
    count = 12 if smoke else 829
    result = {}
    for weather in WEATHERS:
        distribution_groups, saturation, sums, gram, n, names = {}, {}, None, None, 0, None
        author_n, author_sums, author_gram, author_names = 0, None, None, None
        author_counts = {str(v): 0 for v in (-1, 0, 1, 2)}
        for index in range(count):
            value = read(source / 'observations' / weather / f'{index:06d}.json')['statistics']
            for group, stats in value['distributions'].items():
                for variable, stat in stats.items():
                    variables = distribution_groups.setdefault(group, {})
                    variables[variable] = merge_distribution([variables[variable], stat]) if variable in variables else stat
            for field, x in value['saturation'].items():
                if field.startswith('attainable_'):
                    if field in saturation and saturation[field] != x:
                        raise ValueError('Reliability bound changed')
                    saturation[field] = x
                else:
                    saturation[field] = saturation.get(field, 0)+x
            corr = value['correlation']
            if names is not None and names != corr['names']:
                raise ValueError('Diagnostic evidence branch schema changed')
            names = corr['names']
            n += corr['n']
            sums = np.asarray(corr['sums']) if sums is None else sums+np.asarray(corr['sums'])
            gram = np.asarray(corr['gram']) if gram is None else gram+np.asarray(corr['gram'])
            ac = value['author_color_correlation']
            if author_names is not None and author_names != ac['names']:
                raise ValueError('Author color diagnostic schema changed')
            author_names = ac['names']
            author_n += ac['n']
            author_sums = np.asarray(ac['sums']) if author_sums is None else author_sums+np.asarray(ac['sums'])
            author_gram = np.asarray(ac['gram']) if author_gram is None else author_gram+np.asarray(ac['gram'])
            for k, x in value['author_color_mapping']['counts'].items():
                author_counts[k] += x
        result[weather] = dict(distributions={g: {v: merge_distribution([d]) for v, d in item.items()}
            for g, item in distribution_groups.items()}, saturation=saturation,
            association=covariance_summary(names, n, sums, gram))
        result[weather]['author_color_association'] = covariance_summary(author_names, author_n, author_sums, author_gram)
        result[weather]['author_color_mapping_counts'] = author_counts
    return dict(status='complete', weather=result, frames_per_weather=count,
        point_scope='all real valid slots entering the official voxelized model; capped/sampled voxel representation, not every raw return',
        quantiles='histogram approximations; exact first/second moments; quantile bin width recorded',
        distance_groups=['[0,20)', '[20,40)', '[40,70)', '[70,infinity)'],
        point_labels='nearest retained Clean proxy at 0.1m, not real noise labels',
        author_color='additional official green==0 de_lidar mask proxy, mapped at 1e-4m without crossing CAVs; ambiguous/unmapped explicit; not asserted as physical truth',
        probability_calibration='not established; no point-level V2X-R weather truth',
        independence='linear residual associations control density, sensor range and intensity; nonlinear dependence remains possible')


def interventions(output, smoke=False):
    summaries = {a: read(arm_folder(output, a, smoke) / 'summary.json') for a in ARMS}
    count = 12 if smoke else 829
    if any(s['frame_count'] != count or s['conditions'] != list(WEATHERS) or s['status'] != 'complete' for s in summaries.values()):
        raise ValueError('Missing full intervention result')
    if len({s['checkpoint_sha256'] for s in summaries.values()}) != 1:
        raise ValueError('Interventions used different checkpoints')
    differences = {a: {w: {p: {m: 100*(summaries[a]['results'][w]['AP'][p][m]-summaries['R0']['results'][w]['AP'][p][m])
        for m in ('ap30', 'ap50', 'ap70')} for p in ('O', 'O*', 'F')} for w in WEATHERS} for a in ARMS}
    detection_counts = {}
    frame_sets = {}
    for arm in ARMS:
        frames = rows(arm_folder(output, arm, smoke) / 'per_frame_metrics.jsonl')
        frame_sets[arm] = {(r['condition'], r['index'], str(t)):
            {p['object_id'] for p in r['protocols']['F']['matched_pairs'][str(t)]} for r in frames for t in (.3, .5, .7)}
        detection_counts[arm] = {w: {p: {str(t): {field: sum(r['protocols'][p]['thresholds'][str(t)][field]
            for r in frames if r['condition'] == w) for field in ('tp', 'fp', 'fn')}
            for t in (.3, .5, .7)} for p in ('O', 'O*', 'F')} for w in WEATHERS}
    target_changes = {arm: {w: {str(t): dict(
        recovered=sum(len(frame_sets[arm][(w, i, str(t))]-frame_sets['R0'][(w, i, str(t))]) for i in range(count)),
        lost=sum(len(frame_sets['R0'][(w, i, str(t))]-frame_sets[arm][(w, i, str(t))]) for i in range(count)))
        for t in (.3, .5, .7)} for w in WEATHERS} for arm in ARMS}
    return dict(status='complete', frames_per_weather=count,
        results={a: s['results'] for a, s in summaries.items()}, delta_from_R0_pp=differences,
        checkpoint_sha256=summaries['R0']['checkpoint_sha256'], detection_counts=detection_counts,
        fixed_GT_target_changes_from_R0=target_changes,
        raw_summaries={a: dict(path=str(arm_folder(output, a, smoke) / 'summary.json'), sha256=sha(arm_folder(output, a, smoke) / 'summary.json')) for a in ARMS})


def smoke_check(cli, folder):
    summary = interventions(cli.output, smoke=True)
    stats = reliability_summary(cli.output, smoke=True)
    # Exercise full error analysis on the same cached prefix using original frozen matching.
    detail = error_core(cli, folder / 'errors', count=12, smoke=True)
    atomic(folder / 'E0_SMOKE.json', dict(status='complete', interventions=summary,
        reliability=stats, error_analysis=detail, full_validation=False,
        eligible_for_scientific_conclusions=False), immutable=True)


def grouped(rows_, key):
    result = {}
    for row in rows_:
        label = str(row[key])
        entry = result.setdefault(label, dict(targets=0, C_detected=0, D_detected=0, recovered=0, lost=0,
            clean_cells=0, missing_clean_cells=0, reliability_sum=0., reliability_points=0))
        entry['targets'] += 1
        for field in ('C_detected', 'D_detected', 'recovered', 'lost'):
            entry[field] += bool(row[field])
        entry['clean_cells'] += row['coverage']['clean_occupied']
        entry['missing_clean_cells'] += row['coverage']['missing_clean_cells']
        entry['reliability_sum'] += row['r']['sum']
        entry['reliability_points'] += row['r']['n']
    for entry in result.values():
        entry['low_sample'] = entry['targets'] < 30
        entry['mean_reliability'] = entry['reliability_sum']/entry['reliability_points'] if entry['reliability_points'] else None
    return result


def geometric_pairs(left, right, threshold=.5):
    """Descriptive final-box correspondence only; never changes official AP matching."""
    from opencood.utils import common_utils
    if left['prediction_is_none'] or right['prediction_is_none']:
        return [], 0
    a = np.asarray(left['prediction_boxes'], dtype=np.float32).reshape(-1, 8, 3)
    b = np.asarray(right['prediction_boxes'], dtype=np.float32).reshape(-1, 8, 3)
    if not len(a) or not len(b):
        return [], 0
    ap, bp = common_utils.convert_format(a), common_utils.convert_format(b)
    matrix = np.asarray([common_utils.compute_iou(p, bp) for p in ap])
    if matrix.shape != (len(a), len(b)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid final-box geometry comparison')
    matches, ambiguous = [], 0
    for i in range(len(a)):
        j = int(np.argmax(matrix[i]))
        if matrix[i, j] < threshold:
            continue
        if (np.sum(matrix[i] >= threshold) != 1 or np.sum(matrix[:, j] >= threshold) != 1
                or int(np.argmax(matrix[:, j])) != i):
            ambiguous += 1
        else:
            matches.append((i, j, float(matrix[i, j])))
    return matches, ambiguous


def scene_bootstrap(objects, replicates=2000):
    scenes = sorted({r['scene'] for r in objects})
    by_scene = np.asarray([[sum(r['recovered'] for r in objects if r['scene'] == s),
        sum(r['lost'] for r in objects if r['scene'] == s)] for s in scenes], dtype=float)
    if len(scenes) < 2:
        return dict(status='insufficient_scenes')
    rng = np.random.Generator(np.random.PCG64(20261021))
    values = []
    for _ in range(replicates):
        chosen = rng.integers(0, len(scenes), size=len(scenes))
        sampled = by_scene[chosen].sum(0)
        values.append(sampled[0]-sampled[1])
    return dict(status='descriptive', scenes=len(scenes), replicates=replicates,
        net_recovered_95_percentile_interval=np.quantile(values, [.025, .975]).tolist(),
        limitation='few correlated scenes, one training seed; interval is for target counts, not AP or a causal effect')


def error_core(cli, folder, count=829, smoke=False):
    from e0_runtime import setup
    _, evidence, _, _, _, _ = setup(cli, gpu=False)
    from evaluate import official_match_trace, statistics
    from support import corner_array, prediction_tensors
    import torch
    from opencood.utils import eval_utils_opv2v
    inventory = read(cli.output / 'stages/preflight/inventory.json')
    r0 = arm_folder(cli.output, 'R0', smoke)
    pair_root = cli.output / ('stages/smoke/paired' if smoke else 'stages/paired')
    results, all_objects = {}, []
    baseline = {v: Path(inventory['models'][v]['root']) / 'final_evaluation' for v in ('C', 'D')}
    # Clean matched IDs define observable cross-weather detection categories.
    clean_detection = {}
    fixed_by_frame = [read(baseline['C'] / 'predictions/clean' / f'{i:06d}.json')['official_gt'] for i in range(count)]
    for index in range(count):
        fixed = fixed_by_frame[index]
        clean_detection[index] = {}
        for v in ('C', 'D'):
            rec = read(baseline[v] / 'predictions/clean' / f'{index:06d}.json')
            boxes, scores = prediction_tensors(rec)
            pairs, _ = official_match_trace(boxes, scores, torch.from_numpy(corner_array(fixed)),
                [o['object_id'] for o in fixed], statistics(), .7)
            clean_detection[index][v] = {p['object_id'] for p in pairs}
    for weather in WEATHERS:
        objects, frame_results = [], []
        totals = {str(t): dict(recovered=0, lost=0, both=0, neither=0, C_tp=0, D_tp=0, C_fp=0, D_fp=0,
            new_FP_geometric_proxy=0, ambiguous_box_correspondence=0) for t in (.3, .5, .7)}
        score_localization = []
        score_probe = {k: statistics() for k in ('C_fixed_final_boxes', 'C_boxes_with_matched_D_scores')}
        for index in range(count):
            fixed = fixed_by_frame[index]
            gt = torch.from_numpy(corner_array(fixed))
            records = {v: read(baseline[v] / 'predictions' / weather / f'{index:06d}.json') for v in ('C', 'D')}
            geometry, ambiguity = geometric_pairs(records['C'], records['D'])
            paired_by_d = {b: a for a, b, iou in geometry}
            ranked = {}
            for variant in ('C', 'D'):
                order = np.argsort(-np.asarray(records[variant]['scores'] or [], dtype=np.float32))
                ranked[variant] = {int(j): int(i) for i, j in enumerate(order)}
            correspondence = [dict(C_prediction_index=i, D_prediction_index=j, geometry_iou=iou,
                C_score=records['C']['scores'][i], D_score=records['D']['scores'][j],
                C_rank=ranked['C'][i], D_rank=ranked['D'][j], rank_delta=ranked['D'][j]-ranked['C'][i]) for i, j, iou in geometry]
            trace, frame = {}, dict(index=index, weather=weather, thresholds={},
                final_box_correspondence_proxy=correspondence)
            c_boxes, c_scores = prediction_tensors(records['C'])
            rescored = None if c_scores is None else c_scores.clone()
            if rescored is not None:
                for i, j, iou in geometry:
                    rescored[i] = records['D']['scores'][j]
            for t in (.3, .5, .7):
                eval_utils_opv2v.caluclate_tp_fp(c_boxes, c_scores, gt, score_probe['C_fixed_final_boxes'], t)
                eval_utils_opv2v.caluclate_tp_fp(c_boxes, rescored, gt, score_probe['C_boxes_with_matched_D_scores'], t)
            for t in (.3, .5, .7):
                matched, fp_sets = {}, {}
                for v in ('C', 'D'):
                    boxes, scores = prediction_tensors(records[v])
                    stat = statistics()
                    pairs, ties = official_match_trace(boxes, scores, gt,
                        [o['object_id'] for o in fixed], stat, t)
                    matched[v] = {p['object_id']: p for p in pairs}
                    fp_sets[v] = set(range(records[v]['prediction_count']))-{p['prediction_index'] for p in pairs}
                a, b = set(matched['C']), set(matched['D'])
                row = dict(recovered=len(b-a), lost=len(a-b), both=len(a & b), neither=len(fixed)-len(a | b),
                    C_tp=len(a), D_tp=len(b), C_fp=len(fp_sets['C']), D_fp=len(fp_sets['D']),
                    new_FP_geometric_proxy=sum(j not in paired_by_d or paired_by_d[j] not in fp_sets['C'] for j in fp_sets['D']),
                    ambiguous_box_correspondence=ambiguity)
                for k, v in row.items():
                    totals[str(t)][k] += v
                frame['thresholds'][str(t)] = row
                trace[str(t)] = matched
            pair = read(pair_root / 'frames' / f'{index:06d}.json')['conditions'][weather]
            geo = {str(r['object_id']): r for r in pair['targets']}
            rel = read(r0 / 'observations' / weather / f'{index:06d}.json')['statistics']
            reliabilities = {str(r['object_id']): r for r in rel['targets']}
            matches = trace['0.7']
            for target in fixed:
                target_id = target['object_id']
                key = str(target_id)
                c, d = target_id in matches['C'], target_id in matches['D']
                if c and d:
                    left, right = matches['C'][target_id], matches['D'][target_id]
                    cscore = records['C']['scores'][left['prediction_index']]
                    dscore = records['D']['scores'][right['prediction_index']]
                    score_localization.append(dict(index=index, object_id=target_id,
                        score_delta=dscore-cscore, matched_iou_delta=right['iou']-left['iou'],
                        C_score=cscore, D_score=dscore, C_iou=left['iou'], D_iou=right['iou']))
                box = target['box']
                distance = float(np.linalg.norm(box[:3]))
                size = float(np.prod(box[3:6]))
                rel_row = reliabilities[key]
                row = dict(weather=weather, index=index, frame_key=evidence['b']['selected_frame_keys'][index],
                    scene=evidence['b']['selected_frame_keys'][index][0], object_id=target_id,
                    C_detected=c, D_detected=d, recovered=d and not c, lost=c and not d,
                    C_clean_detected=target_id in clean_detection[index]['C'], D_clean_detected=target_id in clean_detection[index]['D'],
                    distance=distance, range_group=int(range_group(np.asarray([distance]))[0]),
                    volume=size, size_group='small_<8' if size < 8 else 'medium_8_30' if size < 30 else 'large_>=30',
                    cav_count=len(pair['agents']), vehicle_source='not_identifiable_from_fused_detection',
                    retained_points=geo[key]['points'], coverage=geo[key]['coverage'],
                    nearby_coverage=geo[key]['nearby_coverage'], r=rel_row['r'], u=rel_row['u'],
                    matches={v: matches[v].get(target_id) for v in ('C', 'D')})
                objects.append(row)
                all_objects.append(row)
            frame_results.append(frame)
        probe_metrics = {k: dict(AP={f'ap{round(t*100)}': float(eval_utils_opv2v.calculate_ap(copy.deepcopy(stat), t, False)[0])
            for t in (.3, .5, .7)}, tp={str(t): sum(stat[t]['tp']) for t in stat}) for k, stat in score_probe.items()}
        if not smoke and any(abs(probe_metrics['C_fixed_final_boxes']['AP'][m]-inventory['models']['C']['summary']['results'][weather]['AP']['F'][m]) > 1e-12 for m in ('ap30', 'ap50', 'ap70')):
            raise ValueError('Score-analysis original C replay differs from its frozen F AP')
        results[weather] = dict(counts=totals, groups={k: grouped(objects, k) for k in ('range_group', 'size_group', 'scene', 'cav_count')},
            final_output_score_probe=dict(results=probe_metrics,
                definition='Hold C post-NMS final boxes fixed; replace scores only for unique mutual final-box IoU>=0.5 correspondences with D; use frozen F matcher/AP',
                limitation='descriptive post-NMS ranking sensitivity; not an original model result or a full causal decomposition of C/D/AP/NMS'),
            score_and_localization=score_localization,
            clean_detectable_weather_missed={v: sum(r[v+'_clean_detected'] and not r[v+'_detected'] for r in objects) for v in ('C', 'D')},
            clean_and_weather_detectable={v: sum(r[v+'_clean_detected'] and r[v+'_detected'] for r in objects) for v in ('C', 'D')},
            missing_cells_in_lost_targets=sum(r['coverage']['missing_clean_cells'] for r in objects if r['lost']),
            completely_unobserved_targets=sum(r['retained_points'] == 0 for r in objects),
            count_uncertainty=scene_bootstrap(objects),
            frames=frame_results)
    folder.mkdir(parents=True, exist_ok=True)
    detail_path = folder / 'objects.jsonl.gz'
    # Fixed gzip timestamp permits exact resume verification.
    temp = Path(str(detail_path)+'.tmp')
    with temp.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for row in all_objects:
                # Full per-target reliability histograms remain in R0 frame observations.
                compact = dict(row)
                compact['r'] = {k: row['r'][k] for k in ('n', 'sum', 'sumsq', 'min', 'max')}
                compact['u'] = {k: row['u'][k] for k in ('n', 'sum', 'sumsq', 'min', 'max')}
                stream.write((json.dumps(compact, ensure_ascii=False, allow_nan=False)+'\n').encode())
    temp.replace(detail_path)
    return dict(status='complete', frames_per_weather=count, weather=results,
        target_records=dict(path=str(detail_path), sha256=sha(detail_path)),
        matching='frozen official score-sorted one-to-one BEV assignments; fixed Clean GT',
        additional_FP='geometric final-box correspondence proxy (unique mutual IoU>=0.5); report net FP separately; ambiguous identities not resolved',
        unobserved=['full C/D pre-NMS candidate correspondence', 'NMS-internal causal contribution',
            'physical point return truth', 'which CAV caused a fused detection'],
        interpretation='final-output score and localization associations; no complete internal causal decomposition')


def error_analysis(cli, folder, request_id):
    result = error_core(cli, folder)
    atomic(folder / 'E0_ERROR_ANALYSIS.json', result, immutable=True)
