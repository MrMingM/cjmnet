"""Replay the original official AP functions against immutable prediction caches."""
import copy
from pathlib import Path

from support import (CONDITIONS, DEFINITIONS, THRESHOLDS, atomic_json, atomic_jsonl,
                     compare_gt, corner_array, identity, load_prediction, patched,
                     prediction_tensors, read_json, sha256)


def statistics():
    return {t: dict(tp=[], fp=[], gt=0, score=[]) for t in THRESHOLDS}


def official_match_trace(boxes, scores, gt, gt_ids, stat, threshold):
    """Observe accepted pairs in the actual AP call; never run a second matcher."""
    import numpy as np
    from opencood.utils import common_utils, eval_utils_opv2v
    original_format, original_iou = common_utils.convert_format, common_utils.compute_iou
    formats, pred_map, gt_map, matched = [], {}, {}, []
    calls, ties = [], []
    before = len(stat[threshold]['tp'])

    def convert(array):
        result = original_format(array)
        formats.append(result)  # Keep polygons alive while matching pops GT references.
        mapping = pred_map if len(formats) == 1 else gt_map
        if len(formats) > 2:
            raise ValueError('Official matcher changed polygon conversion dispatch')
        mapping.update({id(poly): i for i, poly in enumerate(result)})
        return result

    def iou(polygon, available_gt):
        result = original_iou(polygon, available_gt)
        calls.append(pred_map[id(polygon)])
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite official BEV IoU')
        # These are precisely the conditions in original caluclate_tp_fp.
        if len(available_gt) and not np.max(result) < threshold:
            j = int(np.argmax(result))
            target = gt_map[id(available_gt[j])]
            matched.append(dict(prediction_index=pred_map[id(polygon)], object_id=gt_ids[target],
                                iou=float(result[j])))
            if np.count_nonzero(result == result[j]) > 1:
                ties.append(dict(prediction_index=pred_map[id(polygon)], max_iou=float(result[j])))
        return result

    with patched(common_utils, 'convert_format', convert), patched(common_utils, 'compute_iou', iou):
        eval_utils_opv2v.caluclate_tp_fp(boxes, scores, gt, stat, threshold)
    delta_tp = stat[threshold]['tp'][before:]
    if (len(matched) != sum(delta_tp) or len({r['object_id'] for r in matched}) != len(matched)
            or len({r['prediction_index'] for r in matched}) != len(matched)):
        raise ValueError('Trace is inconsistent with official one-to-one TP assignments')
    if boxes is not None and (len(formats) != 2 or len(calls) != len(boxes)):
        raise ValueError('Incomplete trace of official predictions')
    return matched, ties


def replay(output, evidence, contract, manifests):
    import numpy as np
    import torch
    from opencood.utils import eval_utils_opv2v
    all_results, per_frame, removed_rows = {}, [], []
    gt_rows, pred_rows = [], []
    selected = contract['frame_keys']
    clean_cache = [load_prediction(output / 'predictions/clean' / f'{i:06d}.json', contract, 'clean', i, key)
                   for i, key in enumerate(selected)]
    fixed_by_frame = [r['official_gt'] for r in clean_cache]
    fixed_total = sum(len(r) for r in fixed_by_frame)
    for condition in CONDITIONS:
        stat = {p: statistics() for p in DEFINITIONS}
        counts = dict(official_gt=0, fixed_gt=fixed_total, removed_by_weather=0,
            none_frames=0, empty_non_none_frames=0, official_gt_in_none_frames=0,
            fixed_gt_in_none_frames=0, prediction_count=0, removed_gt=0, retained_gt=0,
            removed_detected={str(t): 0 for t in THRESHOLDS}, retained_detected={str(t): 0 for t in THRESHOLDS},
            matching_max_iou_ties={str(t): 0 for t in THRESHOLDS},
            paired_geometry_max_corner_delta_m=0.0, paired_corners_bitwise_equal=True)
        for index, key in enumerate(selected):
            path = output / 'predictions' / condition / f'{index:06d}.json'
            digest = sha256(path)
            if digest != manifests[condition]['entries'][str(index)]['sha256']:
                raise ValueError('Prediction bytes changed before evaluation')
            record = load_prediction(path, contract, condition, index, key)
            official, fixed = record['official_gt'], fixed_by_frame[index]
            official_ids, fixed_ids = {r['object_id'] for r in official}, {r['object_id'] for r in fixed}
            if not official_ids <= fixed_ids:
                raise ValueError('Live official weather GT is not a subset of live Clean')
            geometry = compare_gt(official, [r for r in fixed if r['object_id'] in official_ids], f'live paired {condition}/{key}')
            counts['paired_geometry_max_corner_delta_m'] = max(counts['paired_geometry_max_corner_delta_m'], geometry['max_corner_delta_m'])
            counts['paired_corners_bitwise_equal'] &= geometry['corners_bitwise_equal']
            removed = fixed_ids - official_ids
            boxes, scores = prediction_tensors(record)
            official_tensor = torch.from_numpy(corner_array(official))
            fixed_tensor = torch.from_numpy(corner_array(fixed))
            counts['official_gt'] += len(official)
            counts['removed_by_weather'] += len(removed)
            counts['none_frames'] += record['prediction_is_none']
            counts['empty_non_none_frames'] += record['prediction_empty']
            counts['prediction_count'] += record['prediction_count']
            counts['official_gt_in_none_frames'] += len(official) if record['prediction_is_none'] else 0
            counts['fixed_gt_in_none_frames'] += len(fixed) if record['prediction_is_none'] else 0
            counts['removed_gt'] += len(removed)
            counts['retained_gt'] += len(official_ids)
            trace_by_threshold = {}
            metrics = dict(index=index, frame_key=key, scene=key[0], timestamp=key[1], condition=condition,
                prediction_cache_sha256=digest, prediction_state=record['prediction_state'],
                prediction_count=record['prediction_count'], official_gt_count=len(official), fixed_gt_count=len(fixed),
                removed_ids=sorted(removed), protocols={}, paired_geometry=geometry)
            for protocol in DEFINITIONS:
                gt = fixed_tensor if protocol == 'F' else official_tensor
                skipped = protocol == 'O' and record['prediction_is_none']
                row = dict(skipped=skipped, counted_gt=0 if skipped else len(gt),
                    prediction_cache_sha256=digest, thresholds={})
                for t in THRESHOLDS:
                    before = len(stat[protocol][t]['tp'])
                    if not skipped:
                        if protocol == 'F':
                            pairs, ties = official_match_trace(boxes, scores, gt, [r['object_id'] for r in fixed], stat[protocol], t)
                            trace_by_threshold[str(t)] = {r['object_id']: r for r in pairs}
                            counts['matching_max_iou_ties'][str(t)] += len(ties)
                            counts['removed_detected'][str(t)] += sum(r['object_id'] in removed for r in pairs)
                            counts['retained_detected'][str(t)] += sum(r['object_id'] in official_ids for r in pairs)
                            row.setdefault('matched_pairs', {})[str(t)] = pairs
                        else:
                            eval_utils_opv2v.caluclate_tp_fp(boxes, scores, gt, stat[protocol], t)
                    tp, fp = stat[protocol][t]['tp'][before:], stat[protocol][t]['fp'][before:]
                    row['thresholds'][str(t)] = dict(tp=sum(tp), fp=sum(fp),
                        fn=0 if skipped else len(gt) - sum(tp), tp_flags=tp, fp_flags=fp)
                metrics['protocols'][protocol] = row
            for object_id in sorted(removed):
                removed_rows.append(dict(condition=condition, index=index, frame_key=key, object_id=object_id,
                    prediction_cache_sha256=digest, clean_gt_identity=identity(next(r for r in fixed if r['object_id'] == object_id)),
                    detection={str(t): dict(detected=object_id in trace_by_threshold[str(t)],
                        match=trace_by_threshold[str(t)].get(object_id)) for t in THRESHOLDS}))
            per_frame.append(metrics)
            pred_rows.append(dict(condition=condition, index=index, frame_key=key, path=str(path), sha256=digest,
                                  prediction_state=record['prediction_state']))
            gt_rows.append(dict(condition=condition, index=index, frame_key=key,
                official_gt_identity=identity(official), fixed_gt_identity=identity(fixed),
                official_ids=[r['object_id'] for r in official], fixed_ids=[r['object_id'] for r in fixed],
                official_count=len(official), fixed_count=len(fixed), removed_ids=sorted(removed)))
        if counts['official_gt'] != sum(len(f['conditions'][condition]['final_gt']) for f in evidence['frames'][:len(selected)]):
            raise ValueError('Actual official condition GT total differs from D-1B')
        original = read_json(output / 'official_runs' / condition / 'official_result_stat.json')
        if identity(stat['O']) != identity(original):
            raise ValueError('Cached replay O statistics differ from original official inference.main')
        ap = {}
        for protocol, current in stat.items():
            if protocol == 'O' and current[.7]['gt'] != counts['official_gt'] - counts['official_gt_in_none_frames']:
                raise ValueError('O None-frame denominator mismatch')
            expected_gt = fixed_total if protocol == 'F' else counts['official_gt']
            if protocol != 'O' and current[.7]['gt'] != expected_gt:
                raise ValueError('No-skip GT denominator mismatch')
            ap[protocol] = {f'ap{round(t*100)}': float(eval_utils_opv2v.calculate_ap(copy.deepcopy(current), t, False)[0]) for t in THRESHOLDS}
            if any(not np.isfinite(v) or not 0 <= v <= 1 for v in ap[protocol].values()):
                raise ValueError('Invalid official AP')
            atomic_json(output / f'stat_{condition}_{"Ostar" if protocol == "O*" else protocol}.json', current)
        if any(abs(ap['O'][k] - manifests[condition]['official_exact_AP'][k]) > 1e-12 for k in ap['O']):
            raise ValueError('Replay O AP differs from original official result')
        if condition == 'clean' and ap['O*'] != ap['F']:
            raise ValueError('Clean O* and F must be exactly identical')
        if any(stat['O'][t]['tp'] != stat['O*'][t]['tp'] or stat['O'][t]['fp'] != stat['O*'][t]['fp']
               or stat['O'][t]['score'] != stat['O*'][t]['score'] for t in THRESHOLDS):
            raise ValueError('O vs O* changed matching instead of only the None-frame denominator')
        if any(counts['removed_detected'][str(t)] + counts['retained_detected'][str(t)] != sum(stat['F'][t]['tp'])
               or counts['removed_detected'][str(t)] > counts['removed_gt']
               or counts['retained_detected'][str(t)] > counts['retained_gt'] for t in THRESHOLDS):
            raise ValueError('Removed/retained recall does not partition F one-to-one TPs')
        for group in ('removed', 'retained'):
            denominator = counts[group + '_gt']
            counts[group + '_recall'] = {str(t): counts[group + '_detected'][str(t)] / denominator if denominator else None for t in THRESHOLDS}
        all_results[condition] = dict(counts=counts, AP=ap, counted_gt={p: stat[p][.7]['gt'] for p in stat})
        print(f'D-1C replay {condition}: {ap}', flush=True)
    prediction_manifest = dict(status='complete', contract_identity=contract['identity'],
        generation='exactly one final prediction per condition/frame; all protocols reference the same immutable file',
        frame_count_per_condition=len(selected), files=pred_rows)
    prediction_manifest['identity'] = identity(prediction_manifest)
    gt_manifest = dict(status='complete', contract_identity=contract['identity'],
        definition=DEFINITIONS, source_d1b_hashes=evidence['d1b_artifact_sha256'],
        fixed_total_per_condition=fixed_total, fixed_source='live official Clean GT verified against complete D-1B by object ID and geometry',
        frames=gt_rows)
    gt_manifest['identity'] = identity(gt_manifest)
    atomic_json(output / 'prediction_manifest.json', prediction_manifest)
    atomic_json(output / 'gt_manifest.json', gt_manifest)
    atomic_jsonl(output / 'per_frame_metrics.jsonl', per_frame)
    atomic_jsonl(output / 'removed_gt_metrics.jsonl', removed_rows)
    return all_results, prediction_manifest, gt_manifest
