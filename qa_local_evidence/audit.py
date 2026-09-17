"""H-A5 paired clean/weather ego-only detection-path audit.

Development-only diagnostic. It replays the exact Stage-1 clean-positive/weather-ego-miss
population and asks whether target-aligned detector evidence survives in the local ego
branch before final post-processing. No training and no fusion intervention are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from qa_evidence_validity.offline import (
    QUANTILES,
    build_thresholds,
    eligible_weather_misses,
    load_stage1,
    support_strong,
    threshold_for,
)

VALIDATION = Path("/data/scd/datasets/opv2v_official_data_dumping/validate").resolve()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_anchor(clean_trace, weather_trace, target, candidate_id, score_threshold):
    cid = int(candidate_id)
    if clean_trace["anchor_shape"] != weather_trace["anchor_shape"]:
        raise AssertionError("Clean/weather anchor layouts differ")
    if cid < 0 or cid >= len(weather_trace["scores"]):
        raise AssertionError("Clean matched anchor is outside weather prediction")
    clean_score = float(clean_trace["scores"][cid])
    weather_score = float(weather_trace["scores"][cid])
    clean_iou = float(clean_trace["ious"][cid, target])
    weather_iou = float(weather_trace["ious"][cid, target])
    clean_logit = float(clean_trace["logits"][cid])
    weather_logit = float(weather_trace["logits"][cid])
    clean_reg = np.asarray(clean_trace["regression_deltas"][cid], dtype=np.float64)
    weather_reg = np.asarray(weather_trace["regression_deltas"][cid], dtype=np.float64)
    return {
        "candidate_id": cid,
        "clean_score": clean_score,
        "weather_score": weather_score,
        "score_delta": weather_score - clean_score,
        "clean_logit": clean_logit,
        "weather_logit": weather_logit,
        "logit_delta": weather_logit - clean_logit,
        "clean_iou": clean_iou,
        "weather_iou": weather_iou,
        "iou_delta": weather_iou - clean_iou,
        "score_passes_threshold": bool(weather_score > score_threshold),
        "iou70": bool(weather_iou >= 0.7),
        "reg_l2_delta": float(np.linalg.norm(weather_reg - clean_reg)),
    }


def _proxy_labels(weather_row, thresholds):
    labels = {}
    for q in QUANTILES:
        name = f"q{int(round(q * 100)):02d}"
        labels[name] = bool(support_strong(weather_row["ego"], threshold_for(weather_row, thresholds[q])))
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", required=True)
    parser.add_argument("--config", default="qa_observation_diagnostic/experiment.yaml")
    parser.add_argument("--frontend-config", required=True)
    parser.add_argument("--frontend-checkpoint", required=True)
    parser.add_argument("--weather", choices=("fog", "rain", "snow"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", type=int, default=0,
                        help="First N candidate frames; 0 means all. Full loader queue is still replayed.")
    args = parser.parse_args()
    if args.smoke < 0:
        parser.error("--smoke must be >= 0")

    import torch
    from opencood.tools.train_utils import to_device
    from gspr_evidence import runtime as rt
    from gspr_evidence.stage3_runtime import check_sensing
    from gspr_evidence.stage3_trace import describe_target, trace_branch
    from qa_evidence_validity.peer_audit import _source_only_prediction

    rt.verify_frozen()
    targets, maps, _, protocols = load_stage1(args.stage1_root)
    weather_rows = eligible_weather_misses(maps["clean"], targets[args.weather])
    if not weather_rows:
        raise RuntimeError(f"No Stage-1 weather ego misses for {args.weather}")
    by_frame = defaultdict(dict)
    for row in weather_rows:
        by_frame[int(row["sample_index"])][int(row["target_index"])] = row
    selected_frames = sorted(by_frame)
    if args.smoke:
        selected_frames = selected_frames[:args.smoke]
    selected = {i: by_frame[i] for i in selected_frames}

    thresholds = {q: build_thresholds(targets["clean"], q) for q in QUANTILES}
    stage1_protocol = protocols[args.weather]
    clean_protocol = protocols["clean"]
    for p in (stage1_protocol, clean_protocol):
        if not p.get("development_only", False) or p.get("test_data_used", True):
            raise ValueError("H-A5 audit accepts development Stage-1 logs only")
        if Path(p["data_root"]).resolve() != VALIDATION:
            raise ValueError("H-A5 audit requires official OPV2V validation")
    if stage1_protocol["sample_indices"] != clean_protocol["sample_indices"]:
        raise ValueError("Clean/weather Stage-1 frame queues differ")
    if stage1_protocol["frontend_sha256"] != clean_protocol["frontend_sha256"]:
        raise ValueError("Clean/weather Stage-1 frontend checkpoints differ")
    if stage1_protocol["frontend_config_sha256"] != clean_protocol["frontend_config_sha256"]:
        raise ValueError("Clean/weather Stage-1 frontend configs differ")
    if stage1_protocol["collector_sha256"] != _sha("qa_observation_diagnostic/collect.py"):
        raise ValueError("Stage-1 collector source changed; refuse replay")

    options, hypes = rt.load_config(args.config, args.frontend_config)
    if Path(hypes["validate_dir"]).resolve() != VALIDATION:
        raise ValueError("Current config is not official OPV2V validation")
    if rt.sha256(args.frontend_config) != stage1_protocol["frontend_config_sha256"]:
        raise ValueError("Frontend config SHA differs from Stage-1")
    rt.seed_all(options["seed"])
    device = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, device)
    model.eval()
    if digest != stage1_protocol["frontend_sha256"]:
        raise ValueError("Frontend checkpoint SHA differs from Stage-1")
    rt.seed_all(options["seed"])
    ds, loader, indices = rt.make_loader(hypes, options, weather=args.weather)
    if indices != stage1_protocol["sample_indices"]:
        raise ValueError("Current loader queue differs from Stage-1")
    if list(ds.len_record) != stage1_protocol["scene_ends"]:
        raise ValueError("Current scene boundaries differ from Stage-1")

    out = rt.new_output(args.output_dir)
    score_threshold = float(ds.post_processor.params["target_args"]["score_threshold"])
    protocol = {
        "schema": 1,
        "purpose": "H-A5 local evidence audit: A5a proxy overestimate vs A5b local downstream loss",
        "development_only": True,
        "test_data_used": False,
        "weather": args.weather,
        "smoke": args.smoke,
        "unit": "target-frame occurrence",
        "candidate_definition": "Stage-1 clean ego detected AND same target weather ego missed",
        "paired_branch_definition": "Clean and online-weather ego-only branches from the same loader sample; no AttFuse",
        "score_threshold": score_threshold,
        "match_iou": 0.7,
        "selected_candidate_frames": selected_frames,
        "candidate_targets": sum(len(x) for x in selected.values()),
        "stage1_root": str(Path(args.stage1_root).resolve()),
        "frontend_sha256": digest,
        "frontend_config_sha256": rt.sha256(args.frontend_config),
        "experiment_config_sha256": rt.sha256(args.config),
        "collector_sha256": stage1_protocol["collector_sha256"],
        "implementation_sha256": {
            "qa_local_evidence/audit.py": _sha(__file__),
            "qa_local_evidence/analysis.py": _sha(Path(__file__).with_name("analysis.py")),
            "gspr_evidence/stage3_trace.py": _sha("gspr_evidence/stage3_trace.py"),
        },
        "interpretation_boundary": [
            "A final weather ego miss with an IoU>=0.7 decoded proposal demonstrates a late detection-path loss, not fusion failure.",
            "If the clean-matched anchor keeps weather classification confidence but loses IoU, this supports a regression/localization bottleneck.",
            "Metric-strong plus no decoded IoU>=0.7 and no same-anchor classification survival is only A5a-compatible/upstream-unresolved, not proof of A5a.",
            "No result here establishes the root cause inside PillarVFE/backbone; a deeper feature-level audit is required for unresolved cases.",
        ],
    }
    rt.write_json(out / "protocol.json", protocol)

    processed_frames = 0
    processed_targets = 0
    with torch.no_grad(), (out / "targets.jsonl").open("w", encoding="utf-8") as stream:
        for batch in loader:
            index = int(batch["ego"]["communication_sample_index"][0])
            frame_rows = selected.get(index)
            if not frame_rows:
                continue
            batch = to_device(batch, device)
            ego = batch["ego"]
            weather_inp = rt.input_branch(ego, args.weather)
            clean_inp = rt.input_branch(ego, "clean")
            weather_encoded = model.encode(weather_inp)
            clean_encoded = model.encode(clean_inp)
            weather_pred = _source_only_prediction(model, weather_encoded, 0)
            clean_pred = _source_only_prediction(model, clean_encoded, 0)
            weather_trace = trace_branch(ds, batch, weather_pred)
            clean_trace = trace_branch(ds, batch, clean_pred)
            if weather_trace["gt"].shape != clean_trace["gt"].shape:
                raise AssertionError("Clean/weather GT shapes differ")
            if not np.allclose(weather_trace["gt"], clean_trace["gt"], atol=1e-6, rtol=1e-6):
                raise AssertionError("Clean/weather GT differs within same sample")

            clean_frame_rows = {
                j: maps["clean"][(index, j)] for j in frame_rows
            }
            check_sensing(model, weather_inp, weather_encoded, weather_trace["gt"],
                          frame_rows, {"stats": maps[args.weather]})
            check_sensing(model, clean_inp, clean_encoded, clean_trace["gt"],
                          clean_frame_rows, {"stats": maps["clean"]})

            for target_index, weather_row in sorted(frame_rows.items()):
                clean_row = maps["clean"].get((index, target_index))
                if clean_row is None:
                    raise AssertionError("Paired clean Stage-1 target missing")
                weather_desc = describe_target(weather_trace, target_index)
                clean_desc = describe_target(clean_trace, target_index)
                if weather_desc["matched"] or not clean_desc["matched"]:
                    raise AssertionError("Stage-1 clean-hit/weather-miss detection state did not replay")
                if bool(weather_row["ego_detected"]) or not bool(clean_row["ego_detected"]):
                    raise AssertionError("Stage-1 candidate flags changed")
                clean_cid = clean_desc["matched_candidate_id"]
                if clean_cid is None:
                    raise AssertionError("Clean target is matched but has no candidate id")
                same = _same_anchor(clean_trace, weather_trace, target_index, clean_cid, score_threshold)
                center = weather_trace["gt"][target_index, :4, :2].mean(axis=0)
                stored = np.asarray([weather_row["center_x"], weather_row["center_y"]], dtype=np.float64)
                if float(np.linalg.norm(center - stored)) > 1e-3:
                    raise AssertionError("GT target order/center differs from Stage-1")
                row = {
                    "sample_index": index,
                    "target_index": target_index,
                    "scene": int(weather_row["scene"]),
                    "distance": float(weather_row["distance"]),
                    "weather": args.weather,
                    "unit": "target-frame occurrence",
                    "proxy_strong": _proxy_labels(weather_row, thresholds),
                    "proxy": {
                        "box_neff": float(weather_row["ego"]["box_neff"]),
                        "coverage4x4": float(weather_row["ego"]["coverage4x4"]),
                        "reliable_count": int(weather_row["ego"]["reliable_count"]),
                        "semantic_norm_mean": float(weather_row["ego"]["semantic_norm_mean"]),
                    },
                    "clean_final": {
                        "matched": True,
                        "matched_candidate_id": int(clean_cid),
                        "matched_score": float(clean_trace["scores"][clean_cid]),
                        "matched_iou": float(clean_trace["ious"][clean_cid, target_index]),
                    },
                    "weather_path": {
                        "matched": False,
                        "failure_stage": weather_desc["failure_stage"],
                        "stages": weather_desc["stages"],
                        "matching_competitors": weather_desc["matching_competitors"],
                    },
                    "clean_matched_anchor_in_weather": same,
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                processed_targets += 1
            stream.flush()
            processed_frames += 1
            print(
                f"A5 {args.weather}: candidate frames {processed_frames}/{len(selected)}; "
                f"targets={processed_targets}",
                flush=True,
            )

    expected_targets = sum(len(x) for x in selected.values())
    if processed_frames != len(selected) or processed_targets != expected_targets:
        raise RuntimeError(
            f"Incomplete H-A5 audit: frames {processed_frames}/{len(selected)}, "
            f"targets {processed_targets}/{expected_targets}"
        )
    rt.write_json(out / "summary.json", {
        "complete": True,
        "weather": args.weather,
        "smoke": args.smoke,
        "candidate_frames": processed_frames,
        "candidate_targets": processed_targets,
    })
    rt.verify_frozen()
    print(f"Complete H-A5 audit: {out}")


if __name__ == "__main__":
    main()
