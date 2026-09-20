"""Candidate generation, repair application and paired outcome accounting.

GT is deliberately absent from candidate generation and inference functions. It is
used only by make_training_targets(), selector_action_label(), and evaluation
coverage functions.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


WEATHERS = ("clean", "fog", "rain", "snow")
ABLATIONS = ("score", "geometry", "joint")


def wrap_angle(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def candidate_feature_dim(config):
    c = config["candidate"]
    branches = 1 + int(c["max_sources"])
    side = 2 * int(c["context_radius"]) + 1
    # each branch: logit + sigmoid score + 7 regression deltas; branch-present mask;
    # full local logit/score patch; base box descriptor; source summary.
    return branches * 9 + branches + 2 * side * side + 8 + 6


def selector_feature_dim(config):
    # normalized 7-D repair residual + repaired score + score delta +
    # center/size/yaw magnitudes.
    return candidate_feature_dim(config) + 12


def _flatten_prediction(prediction):
    logits = prediction["psm"].permute(0, 2, 3, 1).reshape(-1)
    scores = logits.sigmoid()
    regression = prediction["rm"].permute(0, 2, 3, 1).reshape(-1, 7)
    return logits, scores, regression


@torch.no_grad()
def frozen_full_and_sources(frontend, model_input, max_sources, verify_full=False):
    """Run the unchanged full model plus source-only heads.

    The second encode is intentional in v1: it exposes already ego-aligned per-CAV
    pre-fusion feature levels without changing the frozen frontend.
    """
    frontend.eval()
    full_raw = frontend.base(model_input)
    full = {key: full_raw[key] for key in ("psm", "rm")}
    encoded = frontend.encode(model_input)
    levels = encoded[0] if isinstance(encoded, tuple) else encoded["levels"]
    source_count = int(model_input["record_len"].sum().item())
    if source_count < 1 or source_count > int(max_sources):
        raise ValueError("source count outside configured 1..max_sources")
    if any(level.shape[0] != source_count for level in levels):
        raise ValueError("per-source feature cardinality mismatch")
    base = frontend.base
    sources = []
    for source in range(source_count):
        joined = torch.cat(
            [deblock(level[source:source + 1])
             for deblock, level in zip(base.backbone.deblocks, levels)],
            dim=1,
        )
        sources.append({"psm": base.cls_head(joined), "rm": base.reg_head(joined)})
    if verify_full:
        from gspr_communication.masked_attfuse import fuse
        masks = levels[0].new_ones((source_count, 1, 1, 1))
        fused = [fuse(level, masks) for level in levels]
        joined = torch.cat([deblock(level) for deblock, level
                            in zip(base.backbone.deblocks, fused)], dim=1)
        replay = {"psm": base.cls_head(joined), "rm": base.reg_head(joined)}
        for key in ("psm", "rm"):
            torch.testing.assert_close(replay[key], full[key], atol=2e-4, rtol=2e-4)
    return full, sources


def _decode(pp, prediction, anchors):
    decoded = pp.delta_to_boxes3d(prediction["rm"], anchors)[0]
    if not torch.isfinite(decoded).all():
        raise FloatingPointError("non-finite decoded model boxes")
    return decoded


def _base_box_descriptor(boxes, lidar_range):
    bounds = boxes.new_tensor(lidar_range)
    center = (bounds[:3] + bounds[3:]) / 2
    extent = (bounds[3:] - bounds[:3]).clamp_min(1e-3)
    xyz = (boxes[:, :3] - center) / extent
    size = boxes[:, 3:6].clamp_min(1e-3).log() / 4.0
    yaw = boxes[:, 6]
    return torch.cat([xyz, size, yaw.sin()[:, None], yaw.cos()[:, None]], dim=1)


def _context_features(full_prediction, candidate_ids, radius):
    psm = full_prediction["psm"][0]
    anchors, height, width = psm.shape
    side = 2 * radius + 1
    values = []
    for cid in candidate_ids.detach().cpu().tolist():
        y, rem = divmod(int(cid), width * anchors)
        x, anchor = divmod(rem, anchors)
        logits, scores = [], []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width:
                    value = psm[anchor, yy, xx]
                    logits.append(value)
                    scores.append(value.sigmoid())
                else:
                    logits.append(psm.new_zeros(()))
                    scores.append(psm.new_zeros(()))
        if len(logits) != side * side:
            raise AssertionError("local context shape mismatch")
        values.append(torch.stack(logits + scores))
    return torch.stack(values) if values else psm.new_zeros((0, 2 * side * side))


@torch.no_grad()
def generate_candidates(ds, batch, full_prediction, source_predictions, config, lidar_range):
    """GT-free, model-only candidate generation before score filtering/NMS."""
    cfg = config["candidate"]
    max_sources = int(cfg["max_sources"])
    if len(source_predictions) > max_sources:
        raise ValueError("too many source predictions")
    branches = [full_prediction] + list(source_predictions)
    flat = [_flatten_prediction(pred) for pred in branches]
    total_anchors = flat[0][0].numel()
    if any(item[0].numel() != total_anchors for item in flat):
        raise ValueError("branch anchor layouts differ")

    proposed = set()
    floor = float(cfg["score_floor"])
    topk = int(cfg["topk_per_source"])
    for _, scores, _ in flat:
        valid = torch.nonzero(scores >= floor, as_tuple=False).flatten()
        if valid.numel():
            order = torch.argsort(scores[valid], descending=True, stable=True)
            for cid in valid[order[:topk]].detach().cpu().tolist():
                proposed.add(int(cid))
    device = flat[0][0].device
    if proposed:
        ids = torch.tensor(sorted(proposed), device=device, dtype=torch.long)
        rank = torch.stack([item[1][ids] for item in flat]).amax(0)
        order = torch.argsort(rank, descending=True, stable=True)
        ids = ids[order[:int(cfg["max_candidates"])]]
        rank = rank[order[:int(cfg["max_candidates"])]]
    else:
        ids = torch.empty(0, device=device, dtype=torch.long)
        rank = torch.empty(0, device=device)

    pp = ds.post_processor
    anchors = batch["ego"]["anchor_box"]
    decoded = [_decode(pp, pred, anchors) for pred in branches]
    branch_boxes = torch.stack([boxes[ids] for boxes in decoded], dim=0)
    branch_logits = torch.stack([item[0][ids] for item in flat], dim=0)
    branch_scores = torch.stack([item[1][ids] for item in flat], dim=0)
    branch_regression = torch.stack([item[2][ids] for item in flat], dim=0)

    k = len(ids)
    padded_branches = 1 + max_sources
    branch_values = flat[0][0].new_zeros((k, padded_branches, 9))
    branch_mask = flat[0][0].new_zeros((k, padded_branches))
    if k:
        actual = len(branches)
        branch_values[:, :actual, 0] = branch_logits.T
        branch_values[:, :actual, 1] = branch_scores.T
        branch_values[:, :actual, 2:] = branch_regression.permute(1, 0, 2)
        branch_mask[:, :actual] = 1

    context = _context_features(full_prediction, ids, int(cfg["context_radius"]))
    base_boxes = branch_boxes[0] if k else decoded[0].new_zeros((0, 7))
    box_descriptor = _base_box_descriptor(base_boxes, lidar_range)

    if k:
        source_scores = branch_scores[1:]
        max_score, proposer = source_scores.max(0)
        summary = torch.stack([
            max_score,
            source_scores.mean(0),
            source_scores.std(0, unbiased=False),
            max_score - branch_scores[0],
            proposer.to(max_score.dtype) / max(max_sources - 1, 1),
            max_score.new_full((k,), len(source_predictions) / max_sources),
        ], dim=1)
    else:
        summary = flat[0][0].new_zeros((0, 6))

    features = torch.cat([
        branch_values.flatten(1), branch_mask, context, box_descriptor, summary
    ], dim=1)
    expected = candidate_feature_dim(config)
    if features.shape != (k, expected):
        raise AssertionError(f"candidate feature shape {tuple(features.shape)} != {(k, expected)}")
    if not torch.isfinite(features).all():
        raise FloatingPointError("non-finite candidate features")
    return {
        "candidate_ids": ids,
        "rank_scores": rank,
        "features": features,
        "base_boxes": base_boxes,
        "base_scores": branch_scores[0] if k else rank,
        "branch_boxes": branch_boxes,
        "branch_scores": branch_scores,
        "source_count": len(source_predictions),
        "local_region_count": k,
    }


def _project_centers(centers, order, transformation):
    from opencood.utils import box_utils as bu
    corners = bu.boxes_to_corners_3d(centers, order=order)
    return bu.project_box3d(corners, transformation)


@torch.no_grad()
def aligned_gt_centers(ds, batch, gt_corners):
    """Recover GT 7-D centers in the detector coordinate system without changing GT order."""
    ego = batch["ego"]
    centers = ego["object_bbx_center"][0]
    mask = ego["object_bbx_mask"][0].bool()
    centers = centers[mask]
    if len(gt_corners) == 0:
        return centers[:0]
    if len(centers) < len(gt_corners):
        raise ValueError("fewer labeled centers than postprocessed GT boxes")
    pp = ds.post_processor
    projected = _project_centers(centers, pp.params["order"], ego["transformation_matrix"])
    pxy = projected[:, :4, :2].mean(1)
    gxy = gt_corners[:, :4, :2].mean(1)
    available = set(range(len(centers)))
    order_ids = []
    for target in range(len(gt_corners)):
        choices = sorted(available)
        distances = torch.linalg.norm(pxy[choices] - gxy[target], dim=1)
        local = int(torch.argmin(distances))
        if float(distances[local]) > 1e-2:
            raise ValueError("GT center/corner alignment changed")
        chosen = choices[local]
        available.remove(chosen)
        order_ids.append(chosen)
    return centers[torch.as_tensor(order_ids, device=centers.device)]


def polygon_iou_matrix(boxes, targets, order):
    """Project-free planar IoU in the common decoded coordinate system."""
    from opencood.utils import box_utils as bu, common_utils as cu
    result = np.zeros((len(boxes), len(targets)), dtype=np.float32)
    if not len(boxes) or not len(targets):
        return result
    bc = bu.boxes_to_corners_3d(boxes, order=order).detach().cpu().numpy()
    tc = bu.boxes_to_corners_3d(targets, order=order).detach().cpu().numpy()
    bp = cu.convert_format(bc)
    tp = cu.convert_format(tc)
    for i, polygon in enumerate(bp):
        result[i] = np.asarray(cu.compute_iou(polygon, tp), dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite candidate/GT IoU")
    return result


def geometry_residual(base, target, repair_config):
    scale = base.new_tensor(repair_config["center_scale_m"])
    center = ((target[:, :3] - base[:, :3]) / scale).clamp(-1, 1) * scale
    limit = float(repair_config["max_log_size"])
    size = torch.log(target[:, 3:6].clamp_min(1e-3) /
                     base[:, 3:6].clamp_min(1e-3)).clamp(-limit, limit)
    yaw_limit = float(repair_config["max_yaw_delta"])
    yaw = wrap_angle(target[:, 6] - base[:, 6]).clamp(-yaw_limit, yaw_limit)
    return torch.cat([center, size, yaw[:, None]], dim=1)


def normalize_geometry_residual(residual, repair_config):
    scale = residual.new_tensor([
        *repair_config["center_scale_m"],
        repair_config["max_log_size"], repair_config["max_log_size"],
        repair_config["max_log_size"], repair_config["max_yaw_delta"],
    ])
    return residual / scale.clamp_min(1e-6)


def apply_geometry_residual(base, residual):
    if base.shape != residual.shape or base.shape[-1] != 7:
        raise ValueError("base/residual must be Nx7")
    result = base.clone()
    result[:, :3] = base[:, :3] + residual[:, :3]
    result[:, 3:6] = (base[:, 3:6].clamp_min(1e-3) *
                      torch.exp(residual[:, 3:6])).clamp(1e-3, 50.0)
    result[:, 6] = wrap_angle(base[:, 6] + residual[:, 6])
    if not torch.isfinite(result).all() or (result[:, 3:6] <= 0).any():
        raise FloatingPointError("invalid repaired boxes")
    return result


@torch.no_grad()
def make_training_targets(ds, batch, candidates, gt_corners, config):
    """GT-only supervision; candidate IDs and source identities were already fixed model-only."""
    k = len(candidates["candidate_ids"])
    device = candidates["base_boxes"].device
    if not k:
        return {
            "labels": torch.empty(0, device=device, dtype=torch.long),
            "matched_gt": torch.empty(0, device=device, dtype=torch.long),
            "geometry": torch.empty((0, 7), device=device),
            "score": torch.empty(0, device=device),
            "max_iou": torch.empty(0, device=device),
            "target_clipped": torch.empty(0, device=device, dtype=torch.bool),
        }
    gt_centers = aligned_gt_centers(ds, batch, gt_corners)
    branches, _, _ = candidates["branch_boxes"].shape
    if len(gt_centers):
        iou = polygon_iou_matrix(
            candidates["branch_boxes"].reshape(-1, 7),
            gt_centers,
            ds.post_processor.params["order"],
        ).reshape(branches, k, len(gt_centers))
        best_branch = iou.max(axis=0)
        max_iou = best_branch.max(axis=1)
        matched = best_branch.argmax(axis=1)
    else:
        max_iou = np.zeros(k, dtype=np.float32)
        matched = np.zeros(k, dtype=np.int64)
    max_iou_t = torch.as_tensor(max_iou, device=device)
    matched_t = torch.as_tensor(matched, device=device, dtype=torch.long)
    positive = max_iou_t >= float(config["candidate"]["positive_iou"])
    negative = max_iou_t < float(config["candidate"]["negative_iou"])
    labels = torch.full((k,), -1, device=device, dtype=torch.long)
    labels[negative] = 0
    labels[positive] = 1

    geometry = candidates["base_boxes"].new_zeros((k, 7))
    clipped = torch.zeros(k, device=device, dtype=torch.bool)
    if positive.any():
        selected_gt = gt_centers[matched_t[positive]]
        raw_center = selected_gt[:, :3] - candidates["base_boxes"][positive, :3]
        raw_size = torch.log(selected_gt[:, 3:6].clamp_min(1e-3) /
                             candidates["base_boxes"][positive, 3:6].clamp_min(1e-3))
        raw_yaw = wrap_angle(selected_gt[:, 6] - candidates["base_boxes"][positive, 6])
        cfg = config["repair"]
        limits = candidates["base_boxes"].new_tensor(cfg["center_scale_m"])
        clipped[positive] = (
            (raw_center.abs() > limits).any(1) |
            (raw_size.abs() > float(cfg["max_log_size"])).any(1) |
            (raw_yaw.abs() > float(cfg["max_yaw_delta"]))
        )
        geometry[positive] = geometry_residual(
            candidates["base_boxes"][positive], selected_gt, cfg
        )
    # Soft quality target: background=0; positives are supervised by the best
    # source-localization IoU that caused the model-only candidate to be retained.
    score_target = torch.zeros(k, device=device)
    score_target[positive] = max_iou_t[positive].clamp(0, 1)
    return {
        "labels": labels,
        "matched_gt": matched_t,
        "geometry": geometry,
        "score": score_target,
        "max_iou": max_iou_t,
        "target_clipped": clipped,
    }


def repair_loss(network, output, targets, config):
    labels = targets["labels"]
    keep = labels >= 0
    positive = labels == 1
    zero = output["score_logit"].sum() * 0
    score_loss = zero
    geometry_loss = zero
    if keep.any():
        score_loss = F.binary_cross_entropy_with_logits(
            output["score_logit"][keep], targets["score"][keep]
        )
    if positive.any():
        predicted = network.bounded_geometry(output["geometry_raw"][positive])
        predicted = normalize_geometry_residual(predicted, config["repair"])
        expected = normalize_geometry_residual(
            targets["geometry"][positive], config["repair"]
        )
        geometry_loss = F.smooth_l1_loss(predicted, expected)
    total = (float(config["repair"]["score_weight"]) * score_loss +
             float(config["repair"]["geometry_weight"]) * geometry_loss)
    return total, {
        "loss": float(total.detach()),
        "score_loss": float(score_loss.detach()),
        "geometry_loss": float(geometry_loss.detach()),
        "positive": int(positive.sum()),
        "negative": int((labels == 0).sum()),
        "ignored": int((labels < 0).sum()),
        "clipped": int(targets["target_clipped"].sum()),
    }


def repaired_outputs(network, candidates, output, config, ablation):
    if ablation not in ABLATIONS:
        raise ValueError(ablation)
    residual = network.bounded_geometry(output["geometry_raw"])
    repaired_boxes = apply_geometry_residual(candidates["base_boxes"], residual)
    repaired_scores = output["score_logit"].sigmoid()
    boxes = repaired_boxes if ablation in ("geometry", "joint") else candidates["base_boxes"]
    scores = repaired_scores if ablation in ("score", "joint") else candidates["base_scores"]
    return boxes, scores, residual, repaired_scores


def build_selector_features(candidates, residual, repaired_scores, config):
    normalized = normalize_geometry_residual(residual, config["repair"])
    score_delta = repaired_scores - candidates["base_scores"]
    center_mag = torch.linalg.norm(normalized[:, :3], dim=1)
    size_mag = torch.linalg.norm(normalized[:, 3:6], dim=1)
    yaw_mag = normalized[:, 6].abs()
    extra = torch.cat([
        normalized,
        repaired_scores[:, None],
        score_delta[:, None],
        center_mag[:, None],
        size_mag[:, None],
        yaw_mag[:, None],
    ], dim=1)
    result = torch.cat([candidates["features"], extra], dim=1)
    if result.shape[1] != selector_feature_dim(config):
        raise AssertionError("selector feature dimension mismatch")
    return result


def post_process_with_extras(ds, batch, full_prediction, extra_boxes, extra_scores):
    """Insert repaired decoded boxes before the original score filter/NMS chain."""
    from opencood.utils import box_utils as bu
    pp = ds.post_processor
    reference_boxes, reference_scores, gt = ds.post_process(
        batch, {"ego": full_prediction}
    )
    dense_scores = full_prediction["psm"].permute(0, 2, 3, 1).reshape(-1).sigmoid()
    dense_boxes = pp.delta_to_boxes3d(
        full_prediction["rm"], batch["ego"]["anchor_box"]
    )[0]
    if extra_boxes is None:
        extra_boxes = dense_boxes.new_zeros((0, 7))
    if extra_scores is None:
        extra_scores = dense_scores.new_zeros((0,))
    if len(extra_boxes) != len(extra_scores):
        raise ValueError("extra box/score count mismatch")
    if not torch.isfinite(extra_boxes).all() or not torch.isfinite(extra_scores).all():
        raise FloatingPointError("non-finite repaired candidates")
    decoded = torch.cat([dense_boxes, extra_boxes], dim=0)
    scores = torch.cat([dense_scores, extra_scores], dim=0)
    selected = scores > pp.params["target_args"]["score_threshold"]
    ids = torch.nonzero(selected, as_tuple=False).flatten()
    if not len(ids):
        return None, None, gt

    corners = bu.boxes_to_corners_3d(decoded[ids], order=pp.params["order"])
    corners = bu.project_box3d(corners, batch["ego"]["transformation_matrix"])
    keep = torch.logical_and(
        bu.remove_large_pred_bbx(corners), bu.remove_bbx_abnormal_z(corners)
    )
    corners = corners[keep]
    selected_scores = scores[ids][keep]
    if not len(corners):
        return None, None, gt
    keep_nms = bu.nms_rotated(corners, selected_scores, pp.params["nms_thresh"])
    corners = corners[keep_nms]
    selected_scores = selected_scores[keep_nms]
    keep_range = bu.get_mask_for_boxes_within_range_torch(corners)
    corners = corners[keep_range]
    selected_scores = selected_scores[keep_range]
    if not len(corners):
        return None, None, gt
    return corners, selected_scores, gt


def assert_disabled_matches_baseline(ds, batch, full_prediction):
    boxes, scores, gt = post_process_with_extras(
        ds, batch, full_prediction,
        full_prediction["rm"].new_zeros((0, 7)),
        full_prediction["psm"].new_zeros((0,)),
    )
    reference_boxes, reference_scores, reference_gt = ds.post_process(
        batch, {"ego": full_prediction}
    )
    torch.testing.assert_close(gt, reference_gt)
    if reference_boxes is None:
        if boxes is not None or scores is not None:
            raise AssertionError("disabled repair changed None/empty baseline semantics")
    else:
        torch.testing.assert_close(boxes, reference_boxes, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(scores, reference_scores, atol=1e-7, rtol=1e-6)


def _match_frame(boxes, scores, gt, threshold):
    from opencood.utils import common_utils as cu
    remaining_ids = list(range(len(gt)))
    remaining = list(cu.convert_format(gt.detach().cpu().numpy()))
    matched = set()
    fp_indices = []
    if boxes is None:
        return matched, fp_indices
    polygons = list(cu.convert_format(boxes.detach().cpu().numpy()))
    order = np.argsort(-scores.detach().cpu().numpy(), kind="stable")
    for index in order:
        if not remaining:
            fp_indices.append(int(index))
            continue
        ious = np.asarray(cu.compute_iou(polygons[int(index)], remaining))
        if not len(ious) or float(ious.max()) < threshold:
            fp_indices.append(int(index))
            continue
        best = int(ious.argmax())
        matched.add(remaining_ids.pop(best))
        remaining.pop(best)
    return matched, fp_indices


def _new_fp_count(base_boxes, base_scores, base_fp, action_boxes, action_scores,
                  action_fp, threshold):
    from opencood.utils import common_utils as cu
    if not action_fp:
        return 0
    if not base_fp:
        return len(action_fp)
    base_polygons = list(cu.convert_format(
        base_boxes[base_fp].detach().cpu().numpy()
    ))
    action_polygons = list(cu.convert_format(
        action_boxes[action_fp].detach().cpu().numpy()
    ))
    action_rank = np.argsort(
        -action_scores[action_fp].detach().cpu().numpy(), kind="stable"
    )
    available = list(range(len(base_polygons)))
    new = 0
    for local in action_rank:
        if not available:
            new += 1
            continue
        candidates = [base_polygons[i] for i in available]
        ious = np.asarray(cu.compute_iou(action_polygons[int(local)], candidates))
        if len(ious) and float(ious.max()) >= threshold:
            available.pop(int(ious.argmax()))
        else:
            new += 1
    return new


def paired_outcome(base_boxes, base_scores, action_boxes, action_scores, gt, threshold):
    base_match, base_fp = _match_frame(base_boxes, base_scores, gt, threshold)
    action_match, action_fp = _match_frame(action_boxes, action_scores, gt, threshold)
    if action_boxes is None:
        new_fp = 0
    elif base_boxes is None:
        new_fp = len(action_fp)
    else:
        new_fp = _new_fp_count(
            base_boxes, base_scores, base_fp,
            action_boxes, action_scores, action_fp, threshold
        )
    return {
        "recovered": sorted(action_match - base_match),
        "lost": sorted(base_match - action_match),
        "base_tp": len(base_match),
        "action_tp": len(action_match),
        "base_fp": len(base_fp),
        "action_fp": len(action_fp),
        "new_fp": int(new_fp),
    }


def selector_action_label(ds, batch, full_prediction, candidate_box, candidate_score,
                          threshold):
    base_boxes, base_scores, gt = ds.post_process(batch, {"ego": full_prediction})
    action_boxes, action_scores, action_gt = post_process_with_extras(
        ds, batch, full_prediction, candidate_box[None], candidate_score[None]
    )
    torch.testing.assert_close(gt, action_gt)
    outcome = paired_outcome(
        base_boxes, base_scores, action_boxes, action_scores, gt, float(threshold)
    )
    safe_repair = bool(outcome["recovered"] and not outcome["lost"] and
                       outcome["new_fp"] == 0)
    # Harmful and unchanged cases are both trained as KEEP.
    return int(safe_repair), outcome


@torch.no_grad()
def candidate_coverage(ds, batch, candidates, gt_corners):
    gt_centers = aligned_gt_centers(ds, batch, gt_corners)
    result = {
        "gt": len(gt_centers),
        "candidate_count": len(candidates["candidate_ids"]),
        "source_covered_50": 0,
        "source_covered_70": 0,
        "full_covered_50": 0,
        "full_covered_70": 0,
    }
    if not len(gt_centers) or not len(candidates["candidate_ids"]):
        return result
    branch = candidates["branch_boxes"]
    iou = polygon_iou_matrix(
        branch.reshape(-1, 7), gt_centers, ds.post_processor.params["order"]
    ).reshape(branch.shape[0], branch.shape[1], len(gt_centers))
    source_best = iou.max(axis=(0, 1))
    full_best = iou[0].max(axis=0)
    for threshold, suffix in ((0.5, "50"), (0.7, "70")):
        result["source_covered_" + suffix] = int((source_best >= threshold).sum())
        result["full_covered_" + suffix] = int((full_best >= threshold).sum())
    return result
