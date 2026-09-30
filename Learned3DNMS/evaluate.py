"""Evaluate frozen D2D on fixed OPV2V physics validate, with original AP."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

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


def _as_output(corners, scores, gt, selected):
    selected = np.asarray(selected, dtype=np.int64)
    return (
        torch.as_tensor(corners[selected], dtype=torch.float32),
        torch.as_tensor(scores[selected], dtype=torch.float32),
        torch.as_tensor(gt, dtype=torch.float32),
    )


def select(corners, scores, threshold, budget, *, mode):
    """Keep the original range filter; only native TopK replaces original NMS."""
    if not len(scores) or budget <= 0:
        return np.empty(0, dtype=np.int64)
    boxes = torch.as_tensor(corners, dtype=torch.float32)
    values = torch.as_tensor(scores, dtype=torch.float32)
    in_range = box_utils.get_mask_for_boxes_within_range_torch(boxes)
    if mode == 'topk':
        order = np.argsort(-np.asarray(scores), kind='stable')
        return np.asarray(
            [int(i) for i in order if bool(in_range[int(i)])][:budget],
            dtype=np.int64)
    if mode == 'nms':
        keep = box_utils.nms_rotated(boxes, values, threshold)
        return np.asarray([
            int(i) for i in keep if bool(in_range[int(i)])
        ][:budget], dtype=np.int64)
    raise ValueError(f'unknown postprocess mode: {mode}')


def _record(stats, output):
    for threshold in IOU_LEVELS:
        eval_utils.caluclate_tp_fp(*output, stats, threshold)


def _ap(stats, global_sort):
    return {f'ap{int(t * 100)}': float(eval_utils.calculate_ap(
        copy.deepcopy(stats), t, global_sort)[0]) for t in IOU_LEVELS}


def _matches(output):
    corners, scores, gt = output
    if not len(corners) or not len(gt):
        return set(), int(len(corners))
    ious = polygon_ious(
        corners.detach().cpu().numpy(), gt.detach().cpu().numpy())
    assignment = greedy_assignment(
        scores.detach().cpu().numpy(), ious, threshold=.7)
    return set(map(int, assignment[assignment >= 0])), int(
        (assignment < 0).sum())


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
                budget = frame['original_budget']
                threshold = frame['nms_threshold']
                original = (
                    torch.as_tensor(frame['original_corners']),
                    torch.as_tensor(frame['original_scores']),
                    torch.as_tensor(gt),
                )
                if len(scores):
                    boxes_gpu = torch.from_numpy(
                        frame['boxes'])[None].to(device)
                    score_gpu = torch.from_numpy(
                        scores)[None].to(device)
                    mask_gpu = torch.ones_like(
                        score_gpu, dtype=torch.bool)
                    rescored = model.rescore(
                        boxes_gpu, score_gpu, mask_gpu)[0].cpu().numpy()
                else:
                    rescored = scores.copy()
                raw_selected = select(
                    corners, scores, threshold, budget, mode='nms')
                top_selected = select(
                    corners, rescored, threshold, budget, mode='topk')
                reranked_selected = select(
                    corners, rescored, threshold, budget, mode='nms')
                quality = (
                    frame['gt_ious'].max(axis=1)
                    if len(gt) else np.zeros(len(scores), dtype=np.float32))
                oracle_ids = np.flatnonzero(quality > 0)
                oracle_local = select(
                    corners[oracle_ids], quality[oracle_ids],
                    threshold, budget, mode='nms')
                oracle_selected = oracle_ids[oracle_local]
                outputs = {
                    'original_f': original,
                    'top256_original_nms': _as_output(
                        corners, scores, gt, raw_selected),
                    'd2d_topk': _as_output(
                        corners, rescored, gt, top_selected),
                    'd2d_original_nms': _as_output(
                        corners, rescored, gt, reranked_selected),
                    'gt_iou_oracle': _as_output(
                        corners, quality, gt, oracle_selected),
                }
                # Identity-rescoring must reproduce the ordinary top256 NMS
                # up to score floating-point tolerance.
                base_ids = select(
                    corners, scores.copy(), threshold, budget, mode='nms')
                if not np.array_equal(base_ids, raw_selected):
                    raise AssertionError('disabled-method NMS regression failed')
                orig_matches, orig_fp = _matches(original)
                frame_log = dict(weather=weather, index=ordinal,
                                 sample_index=frame['sample_index'],
                                 original_count=budget,
                                 candidates=int(len(scores)),
                                 selected_candidate_ids={},
                                 learned_score_min=(
                                     float(rescored.min()) if len(rescored) else None),
                                 learned_score_max=(
                                     float(rescored.max()) if len(rescored) else None))
                for method, output in outputs.items():
                    _record(stats[method], output)
                    if method != 'original_f':
                        matched, fp = _matches(output)
                        counters[method]['recovered_gt'] += len(
                            matched - orig_matches)
                        counters[method]['lost_gt'] += len(
                            orig_matches - matched)
                        counters[method]['fp_count_delta'] += fp - orig_fp
                for method, selected in (
                    ('top256_original_nms', raw_selected),
                    ('d2d_topk', top_selected),
                    ('d2d_original_nms', reranked_selected),
                    ('gt_iou_oracle', oracle_selected),
                ):
                    frame_log['selected_candidate_ids'][method] = (
                        frame['candidate_ids'][selected].tolist())
                stream.write(json.dumps(frame_log) + '\n')
                if ordinal == 0 or (ordinal + 1) % 100 == 0:
                    print(f'evaluate {weather} {ordinal + 1}/{len(dataset)}',
                          flush=True)
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
                frames=len(dataset), frame_order_ap=ap_frame,
                global_sort_ap=ap_global,
                oracle_recovery_ratio=recovery,
                frame_counts=counters)
            write_json(out / 'results.json', report)
    lines = [
        '# Learned 3D NMS (' + checkpoint['model_config']['variant'] + ') fixed physics validation',
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
            lines.append(
                f"| {weather} | {method} | "
                f"{summary['frame_order_ap'][method]['ap70']:.6f} | "
                f"{summary['global_sort_ap'][method]['ap70']:.6f} | "
                f"{ratio:.4f}" if ratio is not None else
                f"| {weather} | {method} | "
                f"{summary['frame_order_ap'][method]['ap70']:.6f} | "
                f"{summary['global_sort_ap'][method]['ap70']:.6f} | n/a")
            lines[-1] += ' |'
    (out / 'results.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    write_json(out / 'results.json', report)
    print(f'LEARNED NMS EVALUATION COMPLETE: {out / "results.md"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    evaluate(p.parse_args())


if __name__ == '__main__':
    main()
