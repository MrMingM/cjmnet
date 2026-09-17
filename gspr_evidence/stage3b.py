"""Stage-3B: limited whole-agent source-subset Oracle for H-A6 candidates.

This keeps encoded features and the original AttFuse operator fixed, always keeps ego,
and enumerates peer inclusion subsets. It is a finite intervention-space Oracle, not a
fusion-theory upper bound and not a deployable selector.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from gspr_evidence.stage3_common import candidate_rows, choose_frame_oracles


def _subset_masks(model, encoded, included_peers):
    masks = model.empty_masks(encoded)
    for peer in included_peers:
        masks[int(peer)] = 1
    if not bool(masks[0].all()):
        raise AssertionError("Ego must remain fully available in Stage-3B")
    return masks


def _matched_objects(boxes, scores, gt, threshold=0.7):
    """Return matched GT set, FP count, and per-GT score/IoU details."""
    import numpy as np
    from opencood.utils import common_utils as cu
    gt_polys = list(cu.convert_format(gt.detach().cpu().numpy()))
    remaining_ids = list(range(len(gt_polys)))
    remaining_polys = list(gt_polys)
    matched = set()
    details = {}
    fp = 0
    if boxes is None:
        return matched, fp, details
    polys = list(cu.convert_format(boxes.detach().cpu().numpy()))
    score_np = scores.detach().cpu().numpy()
    for pi in np.argsort(-score_np, kind="stable"):
        pi = int(pi)
        if not remaining_polys:
            fp += 1
            continue
        ious = cu.compute_iou(polys[pi], remaining_polys)
        if not len(ious) or float(np.max(ious)) < threshold:
            fp += 1
            continue
        local = int(np.argmax(ious))
        gt_index = int(remaining_ids.pop(local))
        remaining_polys.pop(local)
        matched.add(gt_index)
        details[gt_index] = {
            "score": float(score_np[pi]),
            "iou": float(ious[local]),
            "pred_index": pi,
        }
    return matched, fp, details


def main():
    p = argparse.ArgumentParser(description="Stage-3B limited whole-agent subset Oracle")
    p.add_argument("--stage1-root", required=True,
                   help="Recorded for protocol parity; Stage-3B candidates themselves come from Stage-2")
    p.add_argument("--stage2-root", required=True)
    p.add_argument("--stage3a-root", required=True,
                   help="Must be a completed non-smoke Stage-3A run")
    p.add_argument("--config", default="qa_observation_diagnostic/experiment.yaml")
    p.add_argument("--frontend-config", required=True)
    p.add_argument("--frontend-checkpoint", required=True)
    p.add_argument("--weather", required=True, choices=("fog", "rain", "snow"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-candidate-frames", type=int, default=0,
                   help="0=all candidate frames; positive values are exploratory smoke/debug only")
    p.add_argument("--max-peers", type=int, default=4,
                   help="Safety cap. Current OPV2V setup is expected to have <=4 peers.")
    args = p.parse_args()
    if args.max_candidate_frames < 0 or args.max_peers < 1:
        p.error("Invalid frame/peer cap")

    import torch
    from opencood.tools.train_utils import to_device
    from gspr_evidence import runtime as rt

    rt.verify_frozen()
    rows = candidate_rows(args.stage2_root, args.weather)
    if not rows:
        raise RuntimeError(f"No Stage-2 source-valid/full-miss candidates for {args.weather}")
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[int(row["sample_index"])].append(row)

    options, hypes = rt.load_config(args.config, args.frontend_config)
    fixed_validation = Path("/data/scd/datasets/opv2v_official_data_dumping/validate").resolve()
    if Path(hypes["validate_dir"]).resolve() != fixed_validation:
        raise ValueError("Stage-3B is development-only and requires fixed OPV2V validation")
    rt.seed_all(options["seed"])
    device = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, device)
    model.eval()
    rt.seed_all(options["seed"])
    ds, loader, indices = rt.make_loader(hypes, options, weather=args.weather)

    s2_protocol = json.loads(
        (Path(args.stage2_root) / "peer" / args.weather / "protocol.json").read_text(encoding="utf-8")
    )
    if s2_protocol.get("test_data_used", True) or not s2_protocol.get("development_only", False):
        raise ValueError("Stage-3B accepts only development Stage-2 logs")
    if list(indices) != list(s2_protocol["sample_indices"]):
        raise ValueError("Current validation frame queue differs from Stage-2")
    if digest != s2_protocol["frontend_sha256"]:
        raise ValueError("Frontend checkpoint differs from Stage-2")
    if Path(s2_protocol["stage1_root"]).resolve() != Path(args.stage1_root).resolve():
        raise ValueError("Stage-1 root differs from Stage-2")

    s3a_protocol_path = Path(args.stage3a_root) / args.weather / "protocol.json"
    if not s3a_protocol_path.is_file():
        raise FileNotFoundError(f"Stage-3A protocol missing: {s3a_protocol_path}")
    s3a_protocol = json.loads(s3a_protocol_path.read_text(encoding="utf-8"))
    if s3a_protocol.get("smoke_or_debug", True):
        raise ValueError("Stage-3B requires a completed full Stage-3A run, not a smoke/debug run")
    if Path(s3a_protocol["stage2_root"]).resolve() != Path(args.stage2_root).resolve():
        raise ValueError("Stage-3A and Stage-3B do not reference the same Stage-2 root")
    if Path(s3a_protocol["stage1_root"]).resolve() != Path(args.stage1_root).resolve():
        raise ValueError("Stage-3A and Stage-3B do not reference the same Stage-1 root")
    if s3a_protocol["frontend_sha256"] != digest:
        raise ValueError("Stage-3A frontend differs from Stage-3B")

    out = rt.new_output(args.output_dir)
    protocol = {
        "schema": 1,
        "purpose": "Stage-3B limited whole-agent source-subset Oracle",
        "development_only": True,
        "test_data_used": False,
        "weather": args.weather,
        "stage1_root": str(Path(args.stage1_root).resolve()),
        "stage2_root": str(Path(args.stage2_root).resolve()),
        "stage3a_root": str(Path(args.stage3a_root).resolve()),
        "candidate_definition": "Stage-2 any-peer-alone detected + full-fusion final miss",
        "intervention_space":
            "fixed encoding + fixed original AttFuse + ego always included + whole-peer inclusion/exclusion",
        "not_claimed":
            "Not an upper bound for spatial source selection, modified fusion operators, query changes, "
            "feature replacement, or any other intervention outside whole-agent subset selection.",
        "target_vs_frame":
            "Target-wise recoverability is reported separately from a single realizable frame subset. "
            "Per-target best subsets are never merged into one fictitious frame result.",
        "frontend_sha256": digest,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_candidate_frames": args.max_candidate_frames,
        "smoke_or_debug": bool(args.max_candidate_frames),
    }
    rt.write_json(out / "protocol.json", protocol)

    frames = targets = 0
    target_recoverable = 0
    frame_best_recovered = 0
    frame_safe_recovered = 0
    best_with_zero_loss = 0
    subset_count_sum = 0
    scene_target_recovery = Counter()
    scene_target_total = Counter()
    start = time.perf_counter()

    with torch.no_grad(), (out / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for batch in loader:
            index = int(batch["ego"]["communication_sample_index"][0])
            frame_candidates = by_frame.get(index)
            if not frame_candidates:
                continue
            if args.max_candidate_frames and frames >= args.max_candidate_frames:
                break

            batch = to_device(batch, device)
            ego = batch["ego"]
            inp = rt.input_branch(ego, args.weather)
            encoded = model.encode(inp)
            agents = int(ego["record_len"].sum())
            peers = list(range(1, agents))
            if len(peers) > args.max_peers:
                raise RuntimeError(
                    f"Frame {index} has {len(peers)} peers > safety cap {args.max_peers}; "
                    "raise --max-peers deliberately if intended"
                )

            subset_results = []
            full_tuple = tuple(peers)
            full_matched = None
            full_fp = None
            gt_reference = None

            for r in range(len(peers) + 1):
                for subset in itertools.combinations(peers, r):
                    masks = _subset_masks(model, encoded, subset)
                    pred, _ = model.detect(encoded, masks, serialize=False)
                    boxes, scores, gt = ds.post_process(batch, {"ego": pred})
                    if gt_reference is None:
                        gt_reference = gt
                    else:
                        torch.testing.assert_close(gt_reference, gt)
                    matched, fp, _ = _matched_objects(boxes, scores, gt)
                    subset_results.append({
                        "subset": list(subset),
                        "matched": sorted(matched),
                        "fp": int(fp),
                    })
                    if tuple(subset) == full_tuple:
                        full_matched = matched
                        full_fp = int(fp)

            if full_matched is None:
                raise AssertionError("Full subset was not enumerated")
            candidate_targets = [int(r["target_index"]) for r in frame_candidates]
            unexpected = sorted(set(candidate_targets) & set(full_matched))
            if unexpected:
                raise RuntimeError(f"Stage-2 full misses did not reproduce in frame {index}: {unexpected}")

            oracle = choose_frame_oracles(subset_results, candidate_targets, full_matched, full_fp)
            scene = bisect.bisect_right(ds.len_record, index)
            for target in candidate_targets:
                scene_target_total[str(scene)] += 1
                if oracle["target_wise_recoverable"][target]:
                    scene_target_recovery[str(scene)] += 1

            best = oracle["best_frame_subset"]
            safe = oracle["safe_best_frame_subset"]
            recovered_target_count = oracle["target_wise_recoverable_count"]
            target_recoverable += recovered_target_count
            targets += len(candidate_targets)
            frame_best_recovered += best["recovered_candidate_count"] if best else 0
            frame_safe_recovered += safe["recovered_candidate_count"] if safe else 0
            if best and best["lost_full_count"] == 0:
                best_with_zero_loss += 1
            subset_count_sum += len(subset_results)

            row = {
                "sample_index": index,
                "scene": scene,
                "weather": args.weather,
                "agents": agents,
                "candidate_targets": candidate_targets,
                "full_matched_targets": sorted(full_matched),
                "full_fp": full_fp,
                "target_wise_recoverable": oracle["target_wise_recoverable"],
                "target_wise_recoverable_count": recovered_target_count,
                "best_frame_subset": best,
                "safe_best_frame_subset": safe,
                "all_subsets": oracle["all_subsets"],
            }
            stream.write(json.dumps(row) + "\n")
            frames += 1

            if frames == 1 or frames % 10 == 0:
                expected = min(
                    len(by_frame),
                    args.max_candidate_frames if args.max_candidate_frames else len(by_frame),
                )
                elapsed = time.perf_counter() - start
                per = elapsed / frames
                print(
                    f"{args.weather}: {frames}/{expected} candidate frames; "
                    f"targets={targets}; subsets={subset_count_sum}; "
                    f"{per:.2f}s/frame; ETA {(expected-frames)*per/3600:.2f}h",
                    flush=True,
                )
                stream.flush()

    expected_frames = min(
        len(by_frame),
        args.max_candidate_frames if args.max_candidate_frames else len(by_frame),
    )
    if frames != expected_frames:
        raise RuntimeError(f"Incomplete Stage-3B frame coverage: {frames}/{expected_frames}")
    if not args.max_candidate_frames and targets != len(rows):
        raise RuntimeError(f"Incomplete Stage-3B target coverage: {targets}/{len(rows)}")

    scene_rates = {}
    for scene, den in sorted(scene_target_total.items()):
        num = int(scene_target_recovery[scene])
        scene_rates[scene] = {
            "candidate_target_occurrences": int(den),
            "target_wise_recoverable_occurrences": num,
            "recoverability_rate": num / den if den else None,
        }

    summary = {
        "weather": args.weather,
        "candidate_unique_frames": frames,
        "candidate_target_occurrences": targets,
        "enumerated_subset_evaluations": subset_count_sum,
        "target_wise_recoverable_occurrences": target_recoverable,
        "target_wise_recoverability_rate": target_recoverable / targets if targets else None,
        "sum_best_frame_subset_recovered_occurrences": frame_best_recovered,
        "sum_safe_frame_subset_recovered_occurrences": frame_safe_recovered,
        "frames_whose_best_subset_loses_no_full_tp": best_with_zero_loss,
        "by_scene_target_wise_recoverability": scene_rates,
        "interpretation":
            "Target-wise recoverability is an opportunity statistic. Frame-level best/safe results use one "
            "whole-agent subset per frame and therefore are closer to realizable subset selection, but remain "
            "GT-hindsight Oracles on a restricted candidate-frame subset and are not AP.",
    }
    rt.write_json(out / "summary.json", summary)
    rt.verify_frozen()
    print(f"Complete Stage-3B {args.weather}: {out}", flush=True)


if __name__ == "__main__":
    main()
