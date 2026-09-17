"""Stage-3A: trace where source-valid/full-miss targets disappear in detection post-processing.

This is a frozen-model diagnostic on OPV2V validation + online weather only.
It locates the *last observable disappearance stage*; it does not claim the
causal root of the failure is that post-processing stage.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gspr_evidence.stage3_common import (
    candidate_rows,
    failure_stage,
    key,
    load_stage1_map,
)


def _source_only_prediction(model, encoded, source):
    """Same source-only decoder definition used by Stage-2."""
    import torch
    base = model.engine.base
    joined = torch.cat(
        [deblock(level[source:source + 1])
         for deblock, level in zip(base.backbone.deblocks, encoded["levels"])],
        dim=1,
    )
    return {"psm": base.cls_head(joined), "rm": base.reg_head(joined)}


def _box_signature(box, order):
    from opencood.utils import box_utils
    arr = box.detach().cpu().numpy()[None]
    center = box_utils.corner_to_center(arr, order=order)[0]
    if order == "hwl":
        x, y, z, h, w, l, yaw = center
    elif order == "lwh":
        x, y, z, l, w, h, yaw = center
    else:
        raise ValueError(f"Unsupported box order: {order}")
    return {
        "center_x": float(x), "center_y": float(y), "center_z": float(z),
        "length": float(l), "width": float(w), "height": float(h),
        "yaw": float(yaw),
    }


def _aabb_overlap_candidates(boxes, gt_box):
    """Exact-safe prefilter: rotated polygons cannot overlap if their AABBs do not."""
    import torch
    from opencood.utils import box_utils
    if boxes is None or len(boxes) == 0:
        return torch.zeros((0,), dtype=torch.long, device=gt_box.device)
    bb = box_utils.corner_to_standup_box_torch(boxes)
    gg = box_utils.corner_to_standup_box_torch(gt_box[None])[0]
    keep = (
        (bb[:, 2] >= gg[0]) & (bb[:, 0] <= gg[2]) &
        (bb[:, 3] >= gg[1]) & (bb[:, 1] <= gg[3])
    )
    return keep.nonzero().flatten()


def _ious_to_gt(boxes, gt_box):
    """Planar polygon IoUs, with an exact-safe AABB prefilter."""
    from opencood.utils import common_utils as cu
    ids = _aabb_overlap_candidates(boxes, gt_box)
    if ids.numel() == 0:
        return ids, np.zeros((0,), dtype=np.float64)
    subset = boxes.index_select(0, ids).detach().cpu().numpy()
    gt_poly = cu.convert_format(gt_box[None].detach().cpu().numpy())[0]
    polys = cu.convert_format(subset)
    ious = np.asarray(cu.compute_iou(gt_poly, polys), dtype=np.float64)
    return ids, ious


def _stage_target_summary(state, gt_box, order, threshold=0.7):
    boxes, scores, ids = state["boxes"], state["scores"], state["ids"]
    summary = {
        "count": int(len(ids)),
        "has_iou70": False,
        "best_iou": 0.0,
        "best_iou_score": None,
        "best_iou_anchor_id": None,
        "best_iou_box": None,
        "qualifying_count": 0,
        "best_qualifying_score": None,
        "best_qualifying_iou": None,
        "best_qualifying_anchor_id": None,
        "best_qualifying_box": None,
    }
    if len(ids) == 0:
        return summary
    local, ious = _ious_to_gt(boxes, gt_box)
    if len(ious) == 0:
        return summary
    best_pos = int(np.argmax(ious))
    best_local = int(local[best_pos])
    summary.update({
        "best_iou": float(ious[best_pos]),
        "best_iou_score": float(scores[best_local]),
        "best_iou_anchor_id": int(ids[best_local]),
        "best_iou_box": _box_signature(boxes[best_local], order),
    })
    qual_positions = np.flatnonzero(ious >= threshold)
    summary["qualifying_count"] = int(len(qual_positions))
    summary["has_iou70"] = bool(len(qual_positions))
    if len(qual_positions):
        qual_local = local[qual_positions]
        qual_scores = scores.index_select(0, qual_local)
        chosen_in_qual = int(qual_scores.argmax().item())
        chosen_local = int(qual_local[chosen_in_qual])
        chosen_prefilter_pos = int(qual_positions[chosen_in_qual])
        summary.update({
            "best_qualifying_score": float(scores[chosen_local]),
            "best_qualifying_iou": float(ious[chosen_prefilter_pos]),
            "best_qualifying_anchor_id": int(ids[chosen_local]),
            "best_qualifying_box": _box_signature(boxes[chosen_local], order),
        })
    return summary


def _nms_trace(state, threshold):
    """Replicate checkout's nms_rotated, exposing top-1000 and suppressor IDs."""
    from opencood.utils import common_utils as cu
    boxes, scores, ids = state["boxes"], state["scores"], state["ids"]
    if len(ids) == 0:
        empty = {"boxes": boxes, "scores": scores, "ids": ids}
        return empty, empty, {}

    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    polygons = cu.convert_format(boxes_np)
    ixs = scores_np.argsort()[::-1][:1000]
    top_ixs = np.array(ixs, copy=True)
    suppressor = {}
    pick = []

    while len(ixs) > 0:
        i = int(ixs[0])
        pick.append(i)
        rest = ixs[1:]
        if len(rest):
            iou = np.asarray(cu.compute_iou(polygons[i], polygons[rest]), dtype=np.float64)
            remove_positions = np.where(iou > threshold)[0]
            for pos in remove_positions:
                victim = int(rest[int(pos)])
                suppressor[int(ids[victim])] = int(ids[i])
            keep_rest = np.ones(len(rest), dtype=bool)
            keep_rest[remove_positions] = False
            ixs = rest[keep_rest]
        else:
            ixs = np.asarray([], dtype=np.int64)

    import torch
    top_t = torch.as_tensor(top_ixs, device=boxes.device, dtype=torch.long)
    pick_t = torch.as_tensor(pick, device=boxes.device, dtype=torch.long)
    top_state = {
        "boxes": boxes.index_select(0, top_t),
        "scores": scores.index_select(0, top_t),
        "ids": ids.index_select(0, top_t),
    }
    nms_state = {
        "boxes": boxes.index_select(0, pick_t),
        "scores": scores.index_select(0, pick_t),
        "ids": ids.index_select(0, pick_t),
    }
    return top_state, nms_state, suppressor


def _trace_prediction(ds, batch, prediction):
    """Replay the exact VoxelPostprocessor path while preserving anchor identities."""
    import torch
    import torch.nn.functional as F
    from opencood.utils import box_utils

    pp = ds.post_processor
    if not hasattr(pp, "delta_to_boxes3d"):
        raise TypeError("Stage-3A currently requires the checkout's VoxelPostprocessor semantics")
    ego = batch["ego"]
    anchor_box = ego["anchor_box"]
    transform = ego["transformation_matrix"]

    prob = F.sigmoid(prediction["psm"].permute(0, 2, 3, 1)).reshape(1, -1)[0]
    decoded_center = pp.delta_to_boxes3d(prediction["rm"], anchor_box)[0]
    decoded = box_utils.boxes_to_corners_3d(decoded_center, order=pp.params["order"])
    decoded = box_utils.project_box3d(decoded, transform)
    ids = torch.arange(len(prob), device=prob.device, dtype=torch.long)

    states = {
        "decoded": {"boxes": decoded, "scores": prob, "ids": ids},
    }

    score_threshold = float(pp.params["target_args"]["score_threshold"])
    score_mask = prob > score_threshold
    states["score"] = {
        "boxes": decoded[score_mask],
        "scores": prob[score_mask],
        "ids": ids[score_mask],
    }

    s = states["score"]
    if len(s["ids"]):
        keep_large = box_utils.remove_large_pred_bbx(s["boxes"])
        keep_z = box_utils.remove_bbx_abnormal_z(s["boxes"])
        keep = torch.logical_and(keep_large, keep_z)
    else:
        keep = torch.zeros((0,), dtype=torch.bool, device=prob.device)
    states["geometry"] = {
        "boxes": s["boxes"][keep],
        "scores": s["scores"][keep],
        "ids": s["ids"][keep],
    }

    nms_thresh = float(pp.params["nms_thresh"])
    top_state, nms_state, suppressor = _nms_trace(states["geometry"], nms_thresh)
    states["nms_top1000"] = top_state
    states["nms"] = nms_state

    s = states["nms"]
    if len(s["ids"]):
        range_keep = box_utils.get_mask_for_boxes_within_range_torch(s["boxes"])
    else:
        range_keep = torch.zeros((0,), dtype=torch.bool, device=prob.device)
    states["range"] = {
        "boxes": s["boxes"][range_keep],
        "scores": s["scores"][range_keep],
        "ids": s["ids"][range_keep],
    }
    return states, suppressor, {
        "score_threshold": score_threshold,
        "nms_threshold": nms_thresh,
    }


def _greedy_assignment(boxes, scores, gt, threshold=0.7):
    """Same descending-score greedy semantics, plus prediction/GT assignment details."""
    from opencood.utils import common_utils as cu
    if boxes is None:
        return {}, {}, 0
    box_np = boxes.detach().cpu().numpy()
    score_np = scores.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    pred_polys = list(cu.convert_format(box_np))
    remaining_ids = list(range(len(gt_np)))
    remaining_polys = list(cu.convert_format(gt_np))
    matched_by_gt = {}
    assignment_by_pred = {}
    fp = 0

    for pi in np.argsort(-score_np, kind="stable"):
        pi = int(pi)
        if not remaining_polys:
            assignment_by_pred[pi] = {"kind": "fp_after_all_gt_matched"}
            fp += 1
            continue
        ious = np.asarray(cu.compute_iou(pred_polys[pi], remaining_polys), dtype=np.float64)
        if not len(ious) or float(np.max(ious)) < threshold:
            assignment_by_pred[pi] = {"kind": "fp", "best_remaining_iou": float(np.max(ious)) if len(ious) else 0.0}
            fp += 1
            continue
        local = int(np.argmax(ious))
        original_gt = int(remaining_ids.pop(local))
        matched_iou = float(ious[local])
        remaining_polys.pop(local)
        matched_by_gt[original_gt] = {"pred_index": pi, "score": float(score_np[pi]), "iou": matched_iou}
        assignment_by_pred[pi] = {"kind": "tp", "gt_index": original_gt, "iou": matched_iou}
    return matched_by_gt, assignment_by_pred, fp


def _assert_final_equal(trace_state, boxes, scores):
    import torch
    tb, ts = trace_state["boxes"], trace_state["scores"]
    if boxes is None:
        if len(tb) != 0:
            raise AssertionError("Manual post-process produced boxes while ds.post_process returned None")
        return
    if len(tb) != len(boxes):
        raise AssertionError(f"Manual/final box count differs: {len(tb)} vs {len(boxes)}")
    torch.testing.assert_close(tb, boxes, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ts, scores, atol=1e-6, rtol=1e-6)


def _angle_delta(a, b):
    return float(math.atan2(math.sin(a - b), math.cos(a - b)))


def _localization_delta(peer_box, full_box):
    if peer_box is None or full_box is None:
        return None
    return {
        "center_xy_shift": float(math.hypot(
            full_box["center_x"] - peer_box["center_x"],
            full_box["center_y"] - peer_box["center_y"],
        )),
        "center_z_shift": float(full_box["center_z"] - peer_box["center_z"]),
        "length_delta": float(full_box["length"] - peer_box["length"]),
        "width_delta": float(full_box["width"] - peer_box["width"]),
        "height_delta": float(full_box["height"] - peer_box["height"]),
        "yaw_delta": _angle_delta(full_box["yaw"], peer_box["yaw"]),
    }


def _suppression_details(top_summary, suppressor, geometry_state, gt_box, order):
    anchor_id = top_summary.get("best_qualifying_anchor_id")
    if anchor_id is None or int(anchor_id) not in suppressor:
        return None
    suppressor_id = int(suppressor[int(anchor_id)])
    ids = geometry_state["ids"]
    victim_pos = (ids == int(anchor_id)).nonzero().flatten()
    sup_pos = (ids == suppressor_id).nonzero().flatten()
    if victim_pos.numel() != 1 or sup_pos.numel() != 1:
        return None
    v = int(victim_pos[0]); s = int(sup_pos[0])

    from opencood.utils import common_utils as cu
    victim = geometry_state["boxes"][v]
    killer = geometry_state["boxes"][s]
    pv = cu.convert_format(victim[None].detach().cpu().numpy())[0]
    pk = cu.convert_format(killer[None].detach().cpu().numpy())[0]
    overlap = float(cu.compute_iou(pv, [pk])[0])
    _, killer_iou_arr = _ious_to_gt(killer[None], gt_box)
    killer_gt_iou = float(killer_iou_arr[0]) if len(killer_iou_arr) else 0.0
    return {
        "victim_anchor_id": int(anchor_id),
        "victim_score": float(geometry_state["scores"][v]),
        "victim_box": _box_signature(victim, order),
        "suppressor_anchor_id": suppressor_id,
        "suppressor_score": float(geometry_state["scores"][s]),
        "suppressor_gt_iou": killer_gt_iou,
        "suppressor_box": _box_signature(killer, order),
        "victim_suppressor_overlap": overlap,
    }


def main():
    p = argparse.ArgumentParser(description="Stage-3A H-A6 disappearance audit")
    p.add_argument("--stage1-root", required=True)
    p.add_argument("--stage2-root", required=True)
    p.add_argument("--config", default="qa_observation_diagnostic/experiment.yaml")
    p.add_argument("--frontend-config", required=True)
    p.add_argument("--frontend-checkpoint", required=True)
    p.add_argument("--weather", required=True, choices=("fog", "rain", "snow"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-candidate-frames", type=int, default=0,
                   help="0=all. Positive values are smoke/debug only and cannot support the final Stage-3 gate.")
    args = p.parse_args()
    if args.max_candidate_frames < 0:
        p.error("--max-candidate-frames must be >=0")

    import torch
    from opencood.tools.train_utils import to_device
    from gspr_evidence import runtime as rt

    rt.verify_frozen()
    stage2_rows = candidate_rows(args.stage2_root, args.weather)
    if not stage2_rows:
        raise RuntimeError(f"No source-valid/full-miss Stage-2 candidates for {args.weather}")
    stage1_map = load_stage1_map(args.stage1_root, args.weather)

    by_frame = defaultdict(list)
    for row in stage2_rows:
        by_frame[int(row["sample_index"])].append(row)

    options, hypes = rt.load_config(args.config, args.frontend_config)
    fixed_validation = Path("/data/scd/datasets/opv2v_official_data_dumping/validate").resolve()
    if Path(hypes["validate_dir"]).resolve() != fixed_validation:
        raise ValueError("Stage-3A is development-only and requires fixed OPV2V validation")
    rt.seed_all(options["seed"])
    device = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, device)
    model.eval()
    rt.seed_all(options["seed"])
    ds, loader, indices = rt.make_loader(hypes, options, weather=args.weather)

    s2_protocol_path = Path(args.stage2_root) / "peer" / args.weather / "protocol.json"
    s2_protocol = json.loads(s2_protocol_path.read_text(encoding="utf-8"))
    if s2_protocol.get("test_data_used", True) or not s2_protocol.get("development_only", False):
        raise ValueError("Stage-3A accepts only development Stage-2 logs")
    if list(indices) != list(s2_protocol["sample_indices"]):
        raise ValueError("Current validation frame queue differs from Stage-2")
    if digest != s2_protocol["frontend_sha256"]:
        raise ValueError("Frontend checkpoint differs from Stage-2")
    if Path(s2_protocol["stage1_root"]).resolve() != Path(args.stage1_root).resolve():
        raise ValueError("Stage-1 root differs from the one used by Stage-2")

    out = rt.new_output(args.output_dir)
    protocol = {
        "schema": 1,
        "purpose": "Stage-3A source-valid/full-miss detection-path disappearance audit",
        "development_only": True,
        "test_data_used": False,
        "weather": args.weather,
        "stage1_root": str(Path(args.stage1_root).resolve()),
        "stage2_root": str(Path(args.stage2_root).resolve()),
        "candidate_target_occurrences": len(stage2_rows),
        "candidate_frames": len(by_frame),
        "max_candidate_frames": args.max_candidate_frames,
        "smoke_or_debug": bool(args.max_candidate_frames),
        "frontend_sha256": digest,
        "frontend_config_sha256": rt.sha256(args.frontend_config),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_definition": "Stage-2 any-peer-alone detected + full-fusion final miss",
        "failure_stage_definition":
            "first post-decoding stage at which no IoU>=0.7 candidate remains; "
            "this is the last observable disappearance location, NOT a causal root label",
        "stages": [
            "decoded", "score_threshold", "geometry_sanity",
            "nms_top1000", "nms_suppression", "range_filter", "final_greedy_matching",
        ],
        "iou_score_pairing":
            "IoU and score are always recorded from the same candidate box",
        "nms_note":
            "Replicates checkout nms_rotated top-1000 truncation and suppression order and records suppressor IDs",
    }
    rt.write_json(out / "protocol.json", protocol)

    stage_counts = Counter()
    scene_counts = Counter()
    distance_counts = Counter()
    processed_targets = 0
    processed_candidate_frames = 0
    scanned_frames = 0
    start = time.perf_counter()

    with torch.no_grad(), (out / "targets.jsonl").open("w", encoding="utf-8") as stream:
        for batch in loader:
            scanned_frames += 1
            index = int(batch["ego"]["communication_sample_index"][0])
            frame_rows = by_frame.get(index)
            if not frame_rows:
                continue
            if args.max_candidate_frames and processed_candidate_frames >= args.max_candidate_frames:
                break

            batch = to_device(batch, device)
            ego = batch["ego"]
            inp = rt.input_branch(ego, args.weather)
            encoded = model.encode(inp)
            agents = int(ego["record_len"].sum())

            full_mask = torch.ones_like(model.empty_masks(encoded))
            full_pred, _ = model.detect(encoded, full_mask, serialize=False)
            final_boxes, final_scores, gt = ds.post_process(batch, {"ego": full_pred})
            trace, suppressor, thresholds = _trace_prediction(ds, batch, full_pred)
            _assert_final_equal(trace["range"], final_boxes, final_scores)
            full_match, assignment, full_fp = _greedy_assignment(final_boxes, final_scores, gt)

            needed_peers = set()
            chosen_peer_for_target = {}
            for audit in frame_rows:
                detected = [p for p in audit["peers"] if p.get("matched")]
                if not detected:
                    raise RuntimeError(f"Stage-2 source-valid row has no detected peer: {key(audit)}")
                chosen = max(
                    detected,
                    key=lambda p: (
                        -1e30 if p.get("matched_score") is None else float(p["matched_score"]),
                        -int(p["peer_index"]),
                    ),
                )
                peer = int(chosen["peer_index"])
                if peer >= agents:
                    raise RuntimeError(f"Stage-2 peer index {peer} outside current agent count {agents}")
                chosen_peer_for_target[int(audit["target_index"])] = peer
                needed_peers.add(peer)

            peer_cache = {}
            for peer in sorted(needed_peers):
                pred = _source_only_prediction(model, encoded, peer)
                boxes, scores, peer_gt = ds.post_process(batch, {"ego": pred})
                torch.testing.assert_close(gt, peer_gt)
                matched, assign, fp = _greedy_assignment(boxes, scores, gt)
                peer_cache[peer] = {
                    "boxes": boxes, "scores": scores,
                    "matched": matched, "assignment": assign, "fp": fp,
                }

            scene = bisect.bisect_right(ds.len_record, index)
            order = ds.post_processor.params["order"]
            for audit in frame_rows:
                target = int(audit["target_index"])
                stage1 = stage1_map.get((index, target))
                if stage1 is None:
                    raise RuntimeError(f"Stage-1 target missing: {(index, target)}")
                if bool(audit["full"]["matched"]):
                    raise RuntimeError("Stage-3A candidate unexpectedly marked full-detected in Stage-2")
                if target in full_match:
                    raise RuntimeError(f"Stage-2 full miss did not reproduce: {(index, target)}")

                gt_box = gt[target]
                summaries = {
                    name: _stage_target_summary(state, gt_box, order)
                    for name, state in trace.items()
                }
                flags = {
                    "decoded": summaries["decoded"]["has_iou70"],
                    "score": summaries["score"]["has_iou70"],
                    "geometry": summaries["geometry"]["has_iou70"],
                    "nms_top1000": summaries["nms_top1000"]["has_iou70"],
                    "nms": summaries["nms"]["has_iou70"],
                    "range": summaries["range"]["has_iou70"],
                    "final_matched": target in full_match,
                }
                label = failure_stage(flags)

                peer = chosen_peer_for_target[target]
                cache = peer_cache[peer]
                peer_match = cache["matched"].get(target)
                if peer_match is None:
                    raise RuntimeError(
                        f"Chosen Stage-2 peer-alone detection did not reproduce: frame={index} target={target} peer={peer}"
                    )
                pi = int(peer_match["pred_index"])
                peer_box = cache["boxes"][pi]
                peer_reference = {
                    "peer_index": peer,
                    "matched_score": float(peer_match["score"]),
                    "matched_iou": float(peer_match["iou"]),
                    "box": _box_signature(peer_box, order),
                    "frame_fp_count": int(cache["fp"]),
                }

                decoded_best = summaries["decoded"]["best_iou_box"]
                row = {
                    "sample_index": index,
                    "scene": scene,
                    "weather": args.weather,
                    "target_index": target,
                    "distance": float(stage1["distance"]),
                    "agents": agents,
                    "peer_reference": peer_reference,
                    "full_final_fp_count": int(full_fp),
                    "thresholds": thresholds,
                    "stage_flags": flags,
                    "last_observable_disappearance_stage": label,
                    "stages": summaries,
                    "peer_to_full_best_decoded_localization_delta":
                        _localization_delta(peer_reference["box"], decoded_best),
                    "nms_suppression":
                        _suppression_details(
                            summaries["nms_top1000"], suppressor,
                            trace["geometry"], gt_box, order
                        ) if label == "nms_suppression" else None,
                    "final_matching_competition": None,
                }

                if label == "final_matching_competition":
                    qualifying = []
                    final_state = trace["range"]
                    local, ious = _ious_to_gt(final_state["boxes"], gt_box)
                    for local_idx, iou in zip(local.tolist(), ious.tolist()):
                        if iou < 0.7:
                            continue
                        info = assignment.get(int(local_idx), {"kind": "unknown"})
                        qualifying.append({
                            "final_pred_index": int(local_idx),
                            "anchor_id": int(final_state["ids"][local_idx]),
                            "score": float(final_state["scores"][local_idx]),
                            "iou_to_target": float(iou),
                            "greedy_assignment": info,
                        })
                    row["final_matching_competition"] = qualifying

                stream.write(json.dumps(row) + "\n")
                processed_targets += 1
                stage_counts[label] += 1
                scene_counts[str(scene)] += 1
                from qa_observation_diagnostic.metrics import distance_bin
                distance_counts[distance_bin(float(stage1["distance"]))] += 1

            processed_candidate_frames += 1
            if processed_candidate_frames == 1 or processed_candidate_frames % 10 == 0:
                elapsed = time.perf_counter() - start
                expected_frames = min(
                    len(by_frame),
                    args.max_candidate_frames if args.max_candidate_frames else len(by_frame),
                )
                rate = elapsed / processed_candidate_frames
                print(
                    f"{args.weather}: candidate frames {processed_candidate_frames}/{expected_frames}; "
                    f"targets={processed_targets}; scanned={scanned_frames}; "
                    f"{rate:.2f}s/candidate-frame; ETA {(expected_frames-processed_candidate_frames)*rate/3600:.2f}h",
                    flush=True,
                )
                stream.flush()

    expected_frames = min(
        len(by_frame),
        args.max_candidate_frames if args.max_candidate_frames else len(by_frame),
    )
    if processed_candidate_frames != expected_frames:
        raise RuntimeError(
            f"Incomplete Stage-3A candidate frame coverage: {processed_candidate_frames}/{expected_frames}"
        )
    if not args.max_candidate_frames and processed_targets != len(stage2_rows):
        raise RuntimeError(
            f"Incomplete Stage-3A target coverage: {processed_targets}/{len(stage2_rows)}"
        )

    summary = {
        "weather": args.weather,
        "candidate_target_occurrences": processed_targets,
        "candidate_unique_frames": processed_candidate_frames,
        "scanned_validation_frames": scanned_frames,
        "last_observable_disappearance_stage": dict(stage_counts),
        "candidate_occurrences_by_scene": dict(scene_counts),
        "candidate_occurrences_by_distance": dict(distance_counts),
        "interpretation":
            "Stage labels locate where IoU>=0.7 support last disappears in the frozen detection path; "
            "score/NMS labels do not prove score/NMS is the causal root because fusion may have changed upstream features.",
    }
    rt.write_json(out / "summary.json", summary)
    rt.verify_frozen()
    print(f"Complete Stage-3A {args.weather}: {out}", flush=True)


if __name__ == "__main__":
    main()
