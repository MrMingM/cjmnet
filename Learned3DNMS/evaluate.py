"""Evaluate frozen Learned 3D NMS using cached candidate-to-GT IoU.

The extraction stage already stores the exact top256-vs-GT BEV IoU matrix.
Evaluation therefore reuses that matrix for AP and recovered/lost diagnostics
instead of repeating Shapely polygon intersections for every method and every
IoU threshold. Only the original-F output (which is not stored with candidate
IDs) needs one polygon-IoU computation per frame. Rotated NMS itself remains
the repository's original implementation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from ceif_audit.scoring import empty_stats
from gspr_communication.runtime import sha256, write_json
from gspr_evidence.stage3_analysis import greedy_assignment
from gspr_evidence.stage3_trace import polygon_ious
from opencood.utils import box_utils, eval_utils

from .data import FrameCache, WEATHERS, IOU_LEVELS
from .model import D2DRescore


METHODS = ('original_f', 'top256_original_nms', 'd2d_topk',
           'd2d_original_nms', 'gt_iou_oracle')


def select(corners, scores, threshold, budget, *, mode, in_range=None):
    """Keep the original range filter; only native TopK replaces original NMS."""
    if not len(scores) or budget <= 0:
        return np.empty(0, dtype=np.int64)
    boxes = torch.as_tensor(corners, dtype=torch.float32)
    values = torch.as_tensor(scores, dtype=torch.float32)
    if in_range is None:
        in_range = box_utils.get_mask_for_boxes_within_range_torch(
            boxes).cpu().numpy().astype(bool)
    else:
        in_range = np.asarray(in_range, dtype=bool)
        if in_range.shape != (len(scores),):
            raise ValueError('cached range mask has the wrong size')
    if mode == 'topk':
        order = np.argsort(-np.asarray(scores), kind='stable')
        return np.asarray(
            [int(i) for i in order if in_range[int(i)]][:budget],
            dtype=np.int64)
    if mode == 'nms':
        keep = box_utils.nms_rotated(boxes, values, threshold)
        return np.asarray([
            int(i) for i in keep if in_range[int(i)]
        ][:budget], dtype=np.int64)
    raise ValueError(f'unknown postprocess mode: {mode}')


def _validate_iou(scores, ious):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    ious = np.asarray(ious, dtype=np.float32)
    if ious.ndim != 2 or ious.shape[0] != len(scores):
        raise ValueError('scores and candidate-to-GT IoU matrix disagree')
    if not np.isfinite(scores).all() or not np.isfinite(ious).all():
        raise ValueError('non-finite score/IoU in evaluation')
    if (ious < 0).any() or (ious > 1 + 1e-5).any():
        raise ValueError('candidate-to-GT IoU outside [0,1]')
    return scores, ious


def _record_from_iou(stats, scores, ious):
    """Exact eval_utils greedy TP/FP semantics using a precomputed IoU matrix."""
    scores, ious = _validate_iou(scores, ious)
    gt_count = int(ious.shape[1])
    for threshold in IOU_LEVELS:
        fp, tp = [], []
        if len(scores):
            # eval_utils.caluclate_tp_fp uses NumPy's default argsort here.
            order = np.argsort(-scores)
            ordered_scores = scores[order]
            remaining = list(range(gt_count))
            for candidate in order:
                if not remaining:
                    fp.append(1)
                    tp.append(0)
                    continue
                values = ious[int(candidate), remaining]
                if len(values) == 0 or float(np.max(values)) < threshold:
                    fp.append(1)
                    tp.append(0)
                    continue
                fp.append(0)
                tp.append(1)
                remaining.pop(int(np.argmax(values)))
            stats[threshold]['score'].extend(ordered_scores.tolist())
        stats[threshold]['fp'].extend(fp)
        stats[threshold]['tp'].extend(tp)
        stats[threshold]['gt'] += gt_count


def _matches_from_iou(scores, ious):
    """Preserve the old recovered/lost diagnostic's stable greedy matching."""
    scores, ious = _validate_iou(scores, ious)
    if not len(scores) or ious.shape[1] == 0:
        return set(), int(len(scores))
    assignment = greedy_assignment(scores, ious, threshold=.7)
    return set(map(int, assignment[assignment >= 0])), int(
        (assignment < 0).sum())



def _assert_cached_replay(corners, scores, gt, ious, selected):
    """One-frame gate: cached fast path must match original OpenCOOD evaluation."""
    fresh = polygon_ious(corners, gt)
    if fresh.shape != ious.shape or not np.allclose(
            fresh, ious, atol=1e-6, rtol=1e-6):
        difference = (
            float(np.max(np.abs(fresh - ious)))
            if fresh.shape == ious.shape and fresh.size else None)
        raise AssertionError(
            f'cached candidate-to-GT IoU differs from fresh polygons: {difference}')

    selected = np.asarray(selected, dtype=np.int64)
    cached_stats = empty_stats()
    _record_from_iou(
        cached_stats, scores[selected], ious[selected])

    reference = empty_stats()
    boxes = torch.as_tensor(
        corners[selected], dtype=torch.float32)
    values = torch.as_tensor(
        scores[selected], dtype=torch.float32)
    target = torch.as_tensor(gt, dtype=torch.float32)
    for threshold in IOU_LEVELS:
        eval_utils.caluclate_tp_fp(
            boxes, values, target, reference, threshold)
    if cached_stats != reference:
        raise AssertionError(
            'cached IoU TP/FP replay differs from OpenCOOD eval_utils')


def _ap(stats, global_sort):
    return {f'ap{int(t * 100)}': float(eval_utils.calculate_ap(
        copy.deepcopy(stats), t, global_sort)[0]) for t in IOU_LEVELS}


def evaluate(args):
    root = Path(args.cache).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(
        args.checkpoint, map_location='cpu', weights_only=True)
    metadata = json.loads(
        (root / 'manifest.json').read_text(encoding='utf-8'))
    if checkpoint['extract_manifest_sha256'] != sha256(
            root / 'manifest.json'):
        raise ValueError('evaluation cache differs from training extraction')
    for key in ('f_checkpoint_sha256', 'frontend_sha256',
                'v3_checkpoint_sha256', 'weather_root'):
        if checkpoint.get(key) != metadata.get(key):
            raise ValueError(f'checkpoint/extraction mismatch: {key}')
    if metadata.get('status') != 'complete':
        raise ValueError('cannot evaluate incomplete extraction')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = D2DRescore(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    report = {
        'protocol': 'fixed physics validate, no online weather',
        'model': 'adapted ' + checkpoint['model_config']['variant'],
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'extract_manifest_sha256': checkpoint['extract_manifest_sha256'],
        'evaluator': (
            'cached top256-to-GT IoU for all candidate methods; '
            'one original-F polygon-IoU computation per frame'),
        'paper_native_note': (
            'TopK on fixed top256 with original-frame output budget; '
            'OPV2V has no velocity and IoU-matched labels replace '
            'nuScenes center-distance labels.'),
        'methods': list(METHODS),
        'conditions': {},
    }
    with torch.no_grad(), (out / 'frames.jsonl').open(
            'w', encoding='utf-8') as stream:
        for weather in WEATHERS:
            started = time.perf_counter()
            dataset = FrameCache(root, 'validate', (weather,))
            stats = {method: empty_stats() for method in METHODS}
            counters = {
                method: dict(recovered_gt=0, lost_gt=0, fp_count_delta=0)
                for method in METHODS if method != 'original_f'
            }
            for ordinal in range(len(dataset)):
                _, frame = dataset[ordinal]
                corners = frame['corners']
                scores = frame['scores']
                gt = frame['gt']
                cached_ious = frame['gt_ious']
                budget = frame['original_budget']
                threshold = frame['nms_threshold']

                if len(scores):
                    boxes_gpu = torch.from_numpy(
                        frame['boxes'])[None].to(device)
                    score_gpu = torch.from_numpy(
                        scores)[None].to(device)
                    mask_gpu = torch.ones_like(
                        score_gpu, dtype=torch.bool)
                    rescored = model.rescore(
                        boxes_gpu, score_gpu, mask_gpu)[0].cpu().numpy()
                    in_range = box_utils.get_mask_for_boxes_within_range_torch(
                        torch.as_tensor(corners, dtype=torch.float32)
                    ).cpu().numpy().astype(bool)
                else:
                    rescored = scores.copy()
                    in_range = np.zeros(0, dtype=bool)

                # Three genuinely different rotated-NMS orderings. The old
                # evaluator also recomputed raw NMS a second time as a
                # tautological regression check; that duplicate call is gone.
                raw_selected = select(
                    corners, scores, threshold, budget,
                    mode='nms', in_range=in_range)
                top_selected = select(
                    corners, rescored, threshold, budget,
                    mode='topk', in_range=in_range)
                reranked_selected = select(
                    corners, rescored, threshold, budget,
                    mode='nms', in_range=in_range)

                quality = (
                    cached_ious.max(axis=1)
                    if len(gt) else np.zeros(len(scores), dtype=np.float32))
                oracle_ids = np.flatnonzero(quality > 0)
                oracle_in_range = in_range[oracle_ids]
                oracle_local = select(
                    corners[oracle_ids], quality[oracle_ids],
                    threshold, budget, mode='nms',
                    in_range=oracle_in_range)
                oracle_selected = oracle_ids[oracle_local]

                if ordinal == 0:
                    _assert_cached_replay(
                        corners, scores, gt, cached_ious, raw_selected)

                # Candidate methods reuse the exact IoU matrix saved by
                # extraction. Original F has no saved candidate-ID mapping,
                # so compute its IoU matrix once and reuse it for all AP
                # thresholds plus recovered/lost diagnostics.
                original_scores = np.asarray(
                    frame['original_scores'], dtype=np.float32)
                original_corners = np.asarray(
                    frame['original_corners'], dtype=np.float32)
                original_ious = polygon_ious(original_corners, gt)

                method_data = {
                    'original_f': (
                        original_scores, original_ious),
                    'top256_original_nms': (
                        scores[raw_selected],
                        cached_ious[raw_selected]),
                    'd2d_topk': (
                        rescored[top_selected],
                        cached_ious[top_selected]),
                    'd2d_original_nms': (
                        rescored[reranked_selected],
                        cached_ious[reranked_selected]),
                    'gt_iou_oracle': (
                        quality[oracle_selected],
                        cached_ious[oracle_selected]),
                }

                orig_matches, orig_fp = _matches_from_iou(
                    *method_data['original_f'])
                for method, values in method_data.items():
                    _record_from_iou(stats[method], *values)
                    if method != 'original_f':
                        matched, fp = _matches_from_iou(*values)
                        counters[method]['recovered_gt'] += len(
                            matched - orig_matches)
                        counters[method]['lost_gt'] += len(
                            orig_matches - matched)
                        counters[method]['fp_count_delta'] += fp - orig_fp

                frame_log = dict(
                    weather=weather, index=ordinal,
                    sample_index=frame['sample_index'],
                    original_count=budget,
                    candidates=int(len(scores)),
                    selected_candidate_ids={
                        'top256_original_nms':
                            frame['candidate_ids'][raw_selected].tolist(),
                        'd2d_topk':
                            frame['candidate_ids'][top_selected].tolist(),
                        'd2d_original_nms':
                            frame['candidate_ids'][reranked_selected].tolist(),
                        'gt_iou_oracle':
                            frame['candidate_ids'][oracle_selected].tolist(),
                    },
                    learned_score_min=(
                        float(rescored.min()) if len(rescored) else None),
                    learned_score_max=(
                        float(rescored.max()) if len(rescored) else None),
                )
                stream.write(json.dumps(frame_log) + '\n')
                if ordinal == 0 or (ordinal + 1) % 100 == 0:
                    elapsed = time.perf_counter() - started
                    fps = (ordinal + 1) / max(elapsed, 1e-9)
                    print(
                        f'evaluate {weather} {ordinal + 1}/{len(dataset)} '
                        f'({fps:.2f} frame/s)',
                        flush=True)

            elapsed = time.perf_counter() - started
            ap_frame = {
                m: _ap(stats[m], False) for m in METHODS}
            ap_global = {
                m: _ap(stats[m], True) for m in METHODS}
            recovery = {}
            for order, values in (
                    ('frame_order', ap_frame), ('global_sort', ap_global)):
                origin = values['original_f']['ap70']
                ceiling = values['gt_iou_oracle']['ap70']
                denominator = ceiling - origin
                recovery[order] = {
                    method: (
                        (values[method]['ap70'] - origin) / denominator
                        if denominator > 1e-12 else None)
                    for method in METHODS
                }
            report['conditions'][weather] = dict(
                frames=len(dataset),
                evaluation_seconds=float(elapsed),
                frames_per_second=float(
                    len(dataset) / max(elapsed, 1e-9)),
                frame_order_ap=ap_frame,
                global_sort_ap=ap_global,
                oracle_recovery_ratio=recovery,
                frame_counts=counters)
            write_json(out / 'results.json', report)

    lines = [
        '# Learned 3D NMS (' + checkpoint['model_config']['variant']
        + ') fixed physics validation',
        '',
        'AP70 values use identical frozen F/top256 candidate sets.',
        'Frame-order and global-sort AP must never be compared directly.',
        '',
        '| Weather | Method | Frame AP70 | Global AP70 | Global Oracle Recovery |',
        '|---|---|---:|---:|---:|',
    ]
    for weather, summary in report['conditions'].items():
        for method in METHODS:
            ratio = summary['oracle_recovery_ratio']['global_sort'][method]
            if ratio is None:
                ratio_text = 'n/a'
            else:
                ratio_text = f'{ratio:.4f}'
            lines.append(
                f"| {weather} | {method} | "
                f"{summary['frame_order_ap'][method]['ap70']:.6f} | "
                f"{summary['global_sort_ap'][method]['ap70']:.6f} | "
                f"{ratio_text} |")
    (out / 'results.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    write_json(out / 'results.json', report)
    print(
        f'LEARNED NMS EVALUATION COMPLETE: {out / "results.md"}',
        flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    evaluate(p.parse_args())


if __name__ == '__main__':
    main()
