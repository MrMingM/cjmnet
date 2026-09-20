"""Paired validation or fixed official benchmark for A/B/C repair modes."""
import argparse
import bisect
import copy
import json
from pathlib import Path
import time


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="local_detection_repair/experiment.yaml")
    p.add_argument("--frontend-config", required=True)
    p.add_argument("--frontend-checkpoint", required=True)
    p.add_argument("--repair-checkpoint")
    p.add_argument("--selector-checkpoint")
    p.add_argument("--protocol", choices=("development", "benchmark"), default="development")
    p.add_argument("--weather", choices=("clean", "fog", "rain", "snow"), required=True)
    p.add_argument("--ablation", choices=("score", "geometry", "joint"), default="joint")
    p.add_argument("--modes", nargs="+", choices=("baseline", "all", "selector"),
                   default=("baseline", "all", "selector"))
    p.add_argument("--preset", choices=("baseline", "score_only", "geometry_only",
                                       "joint_all", "joint_selector"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--smoke", type=int, default=0)
    args = p.parse_args()
    if args.preset:
        import yaml
        presets = yaml.safe_load(
            (Path(__file__).resolve().parent / "evaluation_presets.yaml").read_text(
                encoding="utf-8"
            )
        )
        args.modes = presets[args.preset]["modes"]
        args.ablation = presets[args.preset]["ablation"]
    return args


def _add_ap(eval_utils, store, boxes, scores, gt):
    for threshold in store:
        eval_utils.caluclate_tp_fp(boxes, scores, gt, store, threshold)


def main():
    args = _args()
    import torch
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from . import runtime as rt
    from .model import LocalRepairNet, RepairSelector
    from .pipeline import (
        assert_disabled_matches_baseline, build_selector_features,
        candidate_coverage, candidate_feature_dim, frozen_full_and_sources,
        frozen_source_predictions, generate_candidates, paired_outcome,
        post_process_with_extras, prepare_postprocess, repaired_outputs,
        selector_feature_dim,
    )

    if args.smoke < 0:
        raise ValueError("--smoke cannot be negative")
    modes = list(dict.fromkeys(["baseline"] + list(args.modes)))
    need_repair = any(mode in modes for mode in ("all", "selector"))
    if need_repair and not args.repair_checkpoint:
        raise ValueError("--repair-checkpoint is required for all/selector modes")
    if "selector" in modes and not args.selector_checkpoint:
        raise ValueError("--selector-checkpoint is required for selector mode")

    options, hypes = rt.load_config(args.config, args.frontend_config)
    rt.seed_all(options["seed"])
    target = rt.device()
    frontend, frontend_sha = rt.load_frontend(
        hypes, options, args.frontend_checkpoint, target
    )
    scene_ds, _, _ = rt.make_loader(
        hypes, options, "train", "clean", shuffle=False, smoke=1
    )
    repair_scenes, selector_scenes = rt.scene_split(
        len(scene_ds.len_record), options["repair_scene_fraction"], options["seed"]
    )
    contract = rt.contract(
        args.config, args.frontend_config, args.frontend_checkpoint,
        options, frontend_sha, repair_scenes, selector_scenes
    )

    repair = selector = None
    repair_state = selector_state = None
    if need_repair:
        repair_state = torch.load(
            args.repair_checkpoint, map_location=target, weights_only=False
        )
        rt.ensure_contract(repair_state["contract"], contract)
        if args.protocol == "benchmark" and repair_state.get("smoke"):
            raise ValueError("formal benchmark refuses a smoke repair checkpoint")
        repair = LocalRepairNet(
            candidate_feature_dim(options), options["repair"]
        ).to(target)
        repair.load_state_dict(repair_state["repair"], strict=True)
        repair.requires_grad_(False)
        repair.eval()
    if "selector" in modes:
        selector_state = torch.load(
            args.selector_checkpoint, map_location=target, weights_only=False
        )
        rt.ensure_contract(selector_state["contract"], contract)
        if selector_state["repair_sha256"] != rt.sha256(args.repair_checkpoint):
            raise ValueError("selector was trained with a different repair checkpoint")
        if selector_state["ablation"] != args.ablation:
            raise ValueError("selector checkpoint ablation differs from evaluation")
        if args.protocol == "benchmark" and selector_state.get("smoke"):
            raise ValueError("formal benchmark refuses a smoke selector checkpoint")
        selector = RepairSelector(
            selector_feature_dim(options), options["selector"]
        ).to(target)
        selector.load_state_dict(selector_state["selector"], strict=True)
        selector.requires_grad_(False)
        selector.eval()

    benchmark = args.protocol == "benchmark"
    split = "test" if benchmark else "validation"
    eval_seed = int(options["seed"]) + 120000
    rt.seed_all(eval_seed)
    ds, loader, indices = rt.make_loader(
        hypes, options, split, args.weather, scenes=None, shuffle=False,
        seed=eval_seed, smoke=args.smoke, benchmark=benchmark
    )
    out = rt.safe_output(args.output_dir)
    rt.write_json(out / "protocol.json", {
        "schema": 1,
        "evaluation_protocol": args.protocol,
        "development_only": not benchmark,
        "formal_test": benchmark,
        "weather": args.weather,
        "data_root": str(rt.TEST_ROOTS[args.weather] if benchmark
                         else rt.VALIDATION_ROOT),
        "online_weather_augmentation": not benchmark and args.weather != "clean",
        "fixed_weather_files": benchmark and args.weather != "clean",
        "sample_indices": indices,
        "modes": modes,
        "ablation": args.ablation,
        "score_threshold": ds.post_processor.params["target_args"]["score_threshold"],
        "nms_threshold": ds.post_processor.params["nms_thresh"],
        "repair_checkpoint": str(Path(args.repair_checkpoint).resolve())
            if args.repair_checkpoint else None,
        "repair_sha256": rt.sha256(args.repair_checkpoint)
            if args.repair_checkpoint else None,
        "selector_checkpoint": str(Path(args.selector_checkpoint).resolve())
            if args.selector_checkpoint else None,
        "selector_sha256": rt.sha256(args.selector_checkpoint)
            if args.selector_checkpoint else None,
        "frontend_checkpoint_sha256": frontend_sha,
        "contract": contract,
        "metric": "existing OpenCOOD planar polygon IoU; non-global-sort AP",
        "additional_communication_bytes": 0,
        "communication_scope":
            "Zero additional payload only under the current full intermediate-fusion baseline: "
            "the receiver already owns per-source pre-fusion features. Source-only heads add "
            "compute, not a new transmitted tensor. Re-measure for future compressed protocols.",
        "timing":
            "baseline forward, source/candidate preparation, repair/selector, and postprocess "
            "are recorded separately; first frame is marked warmup",
    })

    ap = {
        mode: {iou: {"tp": [], "fp": [], "gt": 0, "score": []}
               for iou in (0.3, 0.5, 0.7)}
        for mode in modes
    }
    sums = {
        mode: {
            "frames": 0, "recovered_50": 0, "lost_50": 0, "new_fp_50": 0,
            "recovered_70": 0, "lost_70": 0, "new_fp_70": 0,
            "postprocess_ms": 0.0,
        } for mode in modes
    }
    common = {
        "frames": 0, "candidates": 0, "executed_all": 0, "executed_selector": 0,
        "candidate_source_covered_50": 0, "candidate_source_covered_70": 0,
        "candidate_full_covered_50": 0, "candidate_full_covered_70": 0,
        "coverage_gt": 0, "baseline_model_ms": 0.0, "candidate_prepare_ms": 0.0,
        "repair_net_ms": 0.0, "selector_ms": 0.0,
        "peak_baseline_mib": 0.0, "peak_total_mib": 0.0,
    }

    start = time.perf_counter()
    with torch.no_grad(), (out / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for frame_number, batch in enumerate(loader):
            batch = to_device(batch, target)
            inp = rt.input_branch(batch, args.weather, benchmark=benchmark)
            torch.cuda.reset_peak_memory_stats(target)

            torch.cuda.synchronize()
            t = time.perf_counter()
            full_raw = frontend.base(inp)
            full = {key: full_raw[key] for key in ("psm", "rm")}
            torch.cuda.synchronize()
            baseline_ms = (time.perf_counter() - t) * 1000.0
            peak_baseline = torch.cuda.max_memory_allocated(target) / (1024.0 ** 2)

            torch.cuda.synchronize()
            t = time.perf_counter()
            candidates = None
            if need_repair:
                sources, _ = frozen_source_predictions(
                    frontend, inp, options["candidate"]["max_sources"]
                )
                candidates = generate_candidates(
                    ds.post_processor, batch["ego"]["anchor_box"],
                    full, sources, options,
                    hypes["model"]["args"]["lidar_range"]
                )
            torch.cuda.synchronize()
            candidate_ms = (time.perf_counter() - t) * 1000.0

            prepared = prepare_postprocess(ds, batch, full)
            baseline_boxes = prepared["reference_boxes"]
            baseline_scores = prepared["reference_scores"]
            gt = prepared["gt"]
            if frame_number == 0:
                assert_disabled_matches_baseline(ds, batch, full)
                if need_repair:
                    checked_full, _ = frozen_full_and_sources(
                        frontend, inp, options["candidate"]["max_sources"],
                        verify_full=True
                    )
                    for key in ("psm", "rm"):
                        torch.testing.assert_close(
                            checked_full[key], full[key], atol=0, rtol=0
                        )

            outputs = {"baseline": (baseline_boxes, baseline_scores)}
            repair_ms = selector_ms = 0.0
            selected_count = 0
            if need_repair:
                torch.cuda.synchronize()
                t = time.perf_counter()
                prediction = repair(candidates["features"])
                extra_boxes, extra_scores, residual, repaired_score = repaired_outputs(
                    repair, candidates, prediction, options, args.ablation
                )
                torch.cuda.synchronize()
                repair_ms = (time.perf_counter() - t) * 1000.0
                if "all" in modes:
                    pt = time.perf_counter()
                    outputs["all"] = post_process_with_extras(
                        ds, batch, full, extra_boxes, extra_scores, prepared=prepared
                    )[:2]
                    sums["all"]["postprocess_ms"] += (time.perf_counter() - pt) * 1000.0
                if "selector" in modes:
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    selection_features = build_selector_features(
                        candidates, residual, repaired_score, options, args.ablation
                    )
                    probability = selector(selection_features).sigmoid()
                    selected = probability >= float(options["selector"]["threshold"])
                    torch.cuda.synchronize()
                    selector_ms = (time.perf_counter() - t) * 1000.0
                    selected_count = int(selected.sum())
                    pt = time.perf_counter()
                    outputs["selector"] = post_process_with_extras(
                        ds, batch, full, extra_boxes[selected], extra_scores[selected],
                        prepared=prepared
                    )[:2]
                    sums["selector"]["postprocess_ms"] += (
                        time.perf_counter() - pt
                    ) * 1000.0

                coverage = candidate_coverage(ds, batch, candidates, gt)
                common["candidates"] += coverage["candidate_count"]
                common["coverage_gt"] += coverage["gt"]
                common["candidate_source_covered_50"] += coverage["source_covered_50"]
                common["candidate_source_covered_70"] += coverage["source_covered_70"]
                common["candidate_full_covered_50"] += coverage["full_covered_50"]
                common["candidate_full_covered_70"] += coverage["full_covered_70"]
                common["executed_all"] += len(candidates["candidate_ids"]) if "all" in modes else 0
                common["executed_selector"] += selected_count

            baseline_pt = time.perf_counter()
            # Baseline postprocess was prepared once before the mode loop.
            sums["baseline"]["postprocess_ms"] += 0.0
            _ = baseline_pt

            frame_row = {
                "sample_index": int(batch["ego"]["communication_sample_index"][0]),
                "scene": bisect.bisect_right(
                    ds.len_record, int(batch["ego"]["communication_sample_index"][0])
                ),
                "weather": args.weather,
                "candidate_count": 0 if candidates is None else len(candidates["candidate_ids"]),
                "selector_executed": selected_count,
                "baseline_model_ms": baseline_ms,
                "candidate_prepare_ms": candidate_ms,
                "repair_net_ms": repair_ms,
                "selector_ms": selector_ms,
                "warmup": frame_number == 0,
            }
            for mode in modes:
                boxes, scores = outputs[mode]
                _add_ap(eval_utils, ap[mode], boxes, scores, gt)
                sums[mode]["frames"] += 1
                if mode != "baseline":
                    for threshold, suffix in ((0.5, "50"), (0.7, "70")):
                        paired = paired_outcome(
                            baseline_boxes, baseline_scores, boxes, scores, gt, threshold
                        )
                        sums[mode]["recovered_" + suffix] += len(paired["recovered"])
                        sums[mode]["lost_" + suffix] += len(paired["lost"])
                        sums[mode]["new_fp_" + suffix] += paired["new_fp"]
                        frame_row[f"{mode}_{suffix}"] = paired

            common["frames"] += 1
            common["baseline_model_ms"] += baseline_ms
            common["candidate_prepare_ms"] += candidate_ms
            common["repair_net_ms"] += repair_ms
            common["selector_ms"] += selector_ms
            common["peak_baseline_mib"] += peak_baseline
            common["peak_total_mib"] += (
                torch.cuda.max_memory_allocated(target) / (1024.0 ** 2)
            )
            stream.write(json.dumps(frame_row, ensure_ascii=False) + "\n")

            done = frame_number + 1
            if done == 1 or done % 20 == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"{args.protocol} {args.weather}: {done}/{len(indices)}, "
                    f"{elapsed/done:.2f}s/frame",
                    flush=True,
                )

    if common["frames"] != len(indices):
        raise RuntimeError("incomplete evaluation")
    for mode in modes:
        folder = out / mode
        folder.mkdir()
        if not ap[mode][0.7]["gt"]:
            raise RuntimeError("evaluation has no GT")
        eval_utils.eval_final_results(copy.deepcopy(ap[mode]), str(folder), False)
        rt.write_json(folder / "ap_inputs.json", ap[mode])
        for key in ("postprocess_ms",):
            sums[mode]["mean_" + key] = sums[mode][key] / sums[mode]["frames"]
        rt.write_json(folder / "paired_metrics.json", sums[mode])

    frames = common["frames"]
    summary = {
        **common,
        "mean_candidates": common["candidates"] / frames,
        "repair_all_fraction":
            common["executed_all"] / max(common["candidates"], 1),
        "repair_selector_fraction":
            common["executed_selector"] / max(common["candidates"], 1),
        "source_candidate_coverage_50":
            common["candidate_source_covered_50"] / max(common["coverage_gt"], 1),
        "source_candidate_coverage_70":
            common["candidate_source_covered_70"] / max(common["coverage_gt"], 1),
        "full_candidate_coverage_50":
            common["candidate_full_covered_50"] / max(common["coverage_gt"], 1),
        "full_candidate_coverage_70":
            common["candidate_full_covered_70"] / max(common["coverage_gt"], 1),
        "mean_baseline_model_ms": common["baseline_model_ms"] / frames,
        "mean_candidate_prepare_ms": common["candidate_prepare_ms"] / frames,
        "mean_repair_net_ms": common["repair_net_ms"] / frames,
        "mean_selector_ms": common["selector_ms"] / frames,
        "mean_peak_baseline_mib": common["peak_baseline_mib"] / frames,
        "mean_peak_total_mib": common["peak_total_mib"] / frames,
        "mean_added_peak_mib":
            (common["peak_total_mib"] - common["peak_baseline_mib"]) / frames,
        "additional_communication_bytes": 0,
        "paired": sums,
    }
    rt.write_json(out / "summary.json", summary)
    rt.assert_frozen(frontend)
    print("DONE evaluation:", out, flush=True)


if __name__ == "__main__":
    main()
