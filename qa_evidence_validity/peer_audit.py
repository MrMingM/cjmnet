"""GPU audit of task-usable per-source evidence.

For each Stage-1 weather-induced ego miss, run every CAV source feature by itself
through the same frozen deblocks/detection heads in the ego-aligned frame. This
tests whether a peer that looks "strong" by N_eff/coverage can actually produce
a correctly localized post-processed detection.

This is a diagnostic, not a deployable inference mode.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from qa_evidence_validity.offline import key, read_jsonl


def _source_only_prediction(model, encoded, source):
    """Decode one already ego-aligned source without AttFuse."""
    import torch
    base = model.engine.base
    joined = torch.cat(
        [deblock(level[source:source + 1])
         for deblock, level in zip(base.backbone.deblocks, encoded["levels"])],
        dim=1,
    )
    return {"psm": base.cls_head(joined), "rm": base.reg_head(joined)}


def _greedy_target_details(boxes, scores, gt, threshold=0.7):
    """Per-GT details under the same descending-score greedy matching semantics."""
    from opencood.utils import common_utils as cu

    n_gt = int(len(gt))
    result = [
        {
            "matched": False,
            "matched_score": None,
            "matched_iou": None,
            "best_post_iou": 0.0,
            "score_at_best_post_iou": None,
        }
        for _ in range(n_gt)
    ]
    if boxes is None or len(boxes) == 0 or n_gt == 0:
        return result, 0 if boxes is None else int(len(boxes))

    gt_polys_all = list(cu.convert_format(gt.detach().cpu().numpy()))
    pred_polys = list(cu.convert_format(boxes.detach().cpu().numpy()))
    score_np = scores.detach().cpu().numpy()

    for gi, gp in enumerate(gt_polys_all):
        best_iou, best_score = 0.0, None
        for pi, pp in enumerate(pred_polys):
            iou = float(cu.compute_iou(pp, [gp])[0])
            if iou > best_iou:
                best_iou = iou
                best_score = float(score_np[pi])
        result[gi]["best_post_iou"] = best_iou
        result[gi]["score_at_best_post_iou"] = best_score

    remaining_ids = list(range(n_gt))
    remaining_polys = list(gt_polys_all)
    false_positives = 0
    for pi in np.argsort(-score_np, kind="stable"):
        if not remaining_polys:
            false_positives += 1
            continue
        ious = cu.compute_iou(pred_polys[int(pi)], remaining_polys)
        if not len(ious) or float(np.max(ious)) < threshold:
            false_positives += 1
            continue
        local = int(np.argmax(ious))
        original = remaining_ids.pop(local)
        remaining_polys.pop(local)
        result[original]["matched"] = True
        result[original]["matched_score"] = float(score_np[int(pi)])
        result[original]["matched_iou"] = float(ious[local])
    return result, false_positives


def _load_candidates(stage1_root, weather):
    root = Path(stage1_root)
    clean = {key(r): r for r in read_jsonl(root / "clean" / "targets.jsonl")}
    rows = read_jsonl(root / weather / "targets.jsonl")
    protocol = json.loads((root / weather / "protocol.json").read_text(encoding="utf-8"))
    if not protocol.get("development_only", False) or protocol.get("test_data_used", True):
        raise ValueError("Peer audit accepts only Stage-1 development logs")
    candidates = defaultdict(dict)
    for r in rows:
        c = clean.get(key(r))
        if c is not None and c["ego_detected"] and not r["ego_detected"]:
            candidates[int(r["sample_index"])][int(r["target_index"])] = r
    if not candidates:
        raise RuntimeError(f"No weather-induced ego-miss candidates for {weather}")
    return candidates, protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-root", required=True)
    p.add_argument("--config", default="qa_observation_diagnostic/experiment.yaml")
    p.add_argument("--frontend-config", required=True)
    p.add_argument("--frontend-checkpoint", required=True)
    p.add_argument("--weather", required=True, choices=("fog", "rain", "snow"))
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    import torch
    from opencood.tools.train_utils import to_device
    from gspr_evidence import runtime as rt

    rt.verify_frozen()
    candidates, stage1_protocol = _load_candidates(args.stage1_root, args.weather)
    options, hypes = rt.load_config(args.config, args.frontend_config)
    fixed_validation = Path("/data/scd/datasets/opv2v_official_data_dumping/validate").resolve()
    if Path(hypes["validate_dir"]).resolve() != fixed_validation:
        raise ValueError("Peer audit is development-only and requires fixed OPV2V validation")
    rt.seed_all(options["seed"])
    device = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, device)
    model.eval()
    rt.seed_all(options["seed"])
    ds, loader, indices = rt.make_loader(hypes, options, weather=args.weather)

    if list(indices) != list(stage1_protocol["sample_indices"]):
        raise ValueError("Current frame queue differs from Stage-1")
    if digest != stage1_protocol["frontend_sha256"]:
        raise ValueError("Frontend checkpoint differs from Stage-1")

    out = rt.new_output(args.output_dir)
    protocol = {
        "schema": 1,
        "purpose": "Stage-2 peer-alone task-usable evidence audit",
        "weather": args.weather,
        "development_only": True,
        "test_data_used": False,
        "stage1_root": str(Path(args.stage1_root).resolve()),
        "candidate_frames": len(candidates),
        "candidate_targets": sum(len(v) for v in candidates.values()),
        "sample_indices": indices,
        "frontend_sha256": digest,
        "frontend_config_sha256": rt.sha256(args.frontend_config),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_only_definition":
            "One CAV's pre-fusion multiscale levels, already projected into ego coordinates, "
            "decoded by the same frozen deblocks/cls/reg heads; no AttFuse and no ego feature.",
        "detection_definition":
            "Final post-processed IoU>=0.7 greedy target match; best final proposal IoU/score also recorded.",
        "interpretation":
            "peer-alone detectable is stronger evidence of task usability than N_eff/coverage, "
            "but source-valid/full-miss is still a fusion candidate, not proof of attention causality.",
    }
    rt.write_json(out / "protocol.json", protocol)

    processed_frames = 0
    processed_targets = 0
    first_checked = False
    with torch.no_grad(), (out / "peer_targets.jsonl").open("w", encoding="utf-8") as stream:
        for batch in loader:
            index = int(batch["ego"]["communication_sample_index"][0])
            frame_candidates = candidates.get(index)
            if not frame_candidates:
                continue

            batch = to_device(batch, device)
            ego = batch["ego"]
            inp = rt.input_branch(ego, args.weather)
            encoded = model.encode(inp)
            agents = int(ego["record_len"].sum())

            ego_pred = _source_only_prediction(model, encoded, 0)
            full_mask = torch.ones_like(model.empty_masks(encoded))
            full_pred, _ = model.detect(encoded, full_mask, serialize=False)

            if not first_checked:
                original = model.engine.base(inp)
                for output_key in ("psm", "rm"):
                    torch.testing.assert_close(
                        full_pred[output_key], original[output_key], atol=2e-4, rtol=2e-4
                    )
                none_mask = model.empty_masks(encoded)
                none_pred, _ = model.detect(encoded, none_mask, serialize=False)
                for output_key in ("psm", "rm"):
                    torch.testing.assert_close(
                        ego_pred[output_key], none_pred[output_key], atol=2e-4, rtol=2e-4
                    )
                first_checked = True

            ego_boxes, ego_scores, gt = ds.post_process(batch, {"ego": ego_pred})
            full_boxes, full_scores, full_gt = ds.post_process(batch, {"ego": full_pred})
            torch.testing.assert_close(gt, full_gt)
            ego_details, ego_fp = _greedy_target_details(ego_boxes, ego_scores, gt)
            full_details, full_fp = _greedy_target_details(full_boxes, full_scores, gt)

            peer_outputs = []
            for peer_index in range(1, agents):
                prediction = _source_only_prediction(model, encoded, peer_index)
                boxes, scores, peer_gt = ds.post_process(batch, {"ego": prediction})
                torch.testing.assert_close(gt, peer_gt)
                details, fp = _greedy_target_details(boxes, scores, gt)
                peer_outputs.append((peer_index, details, fp))

            scene = bisect.bisect_right(ds.len_record, index)
            gt_np = gt.detach().cpu().numpy()
            for target_index, stage1 in frame_candidates.items():
                if target_index >= len(gt_np):
                    raise RuntimeError(f"Stage-1 target index {target_index} outside current GT at frame {index}")
                center = gt_np[target_index, :4, :2].mean(axis=0)
                delta = float(np.linalg.norm(center - np.asarray(
                    [stage1["center_x"], stage1["center_y"]], dtype=np.float64)))
                if delta > 1e-3:
                    raise RuntimeError(
                        f"Stage-1/current GT target order changed at frame {index}, target {target_index}: {delta}"
                    )
                if bool(ego_details[target_index]["matched"]) != bool(stage1["ego_detected"]):
                    raise RuntimeError(
                        f"Stage-1 ego detection state did not reproduce at frame {index}, target {target_index}"
                    )
                if bool(full_details[target_index]["matched"]) != bool(stage1["full_detected"]):
                    raise RuntimeError(
                        f"Stage-1 full detection state did not reproduce at frame {index}, target {target_index}"
                    )

                peers = []
                for peer_index, details, fp in peer_outputs:
                    d = dict(details[target_index])
                    d["peer_index"] = peer_index
                    d["frame_fp_count"] = int(fp)
                    peers.append(d)
                detected_peers = [p for p in peers if p["matched"]]
                row = {
                    "sample_index": index,
                    "scene": scene,
                    "weather": args.weather,
                    "target_index": target_index,
                    "ego": dict(ego_details[target_index], frame_fp_count=int(ego_fp)),
                    "full": dict(full_details[target_index], frame_fp_count=int(full_fp)),
                    "peers": peers,
                    "any_peer_alone_detected": bool(detected_peers),
                    "peer_alone_detected_count": len(detected_peers),
                    "detected_peer_indices": [p["peer_index"] for p in detected_peers],
                    "best_peer_matched_score": max(
                        (p["matched_score"] for p in detected_peers if p["matched_score"] is not None),
                        default=None,
                    ),
                    "best_peer_post_iou": max((p["best_post_iou"] for p in peers), default=0.0),
                    "source_valid_full_miss":
                        bool(detected_peers) and not bool(full_details[target_index]["matched"]),
                    "source_valid_full_recovers":
                        bool(detected_peers) and bool(full_details[target_index]["matched"]),
                }
                stream.write(json.dumps(row) + "\n")
                processed_targets += 1

            processed_frames += 1
            if processed_frames == 1 or processed_frames % 20 == 0:
                print(
                    f"{args.weather}: candidate frames {processed_frames}/{len(candidates)}; "
                    f"targets={processed_targets}",
                    flush=True,
                )
                stream.flush()

    expected_targets = sum(len(v) for v in candidates.values())
    if processed_frames != len(candidates) or processed_targets != expected_targets:
        raise RuntimeError(
            f"Incomplete peer audit: frames {processed_frames}/{len(candidates)}, "
            f"targets {processed_targets}/{expected_targets}"
        )
    rt.write_json(
        out / "summary.json",
        {
            "weather": args.weather,
            "candidate_frames": processed_frames,
            "candidate_targets": processed_targets,
        },
    )
    rt.verify_frozen()
    print(f"Complete peer-alone audit: {out}")


if __name__ == "__main__":
    main()
