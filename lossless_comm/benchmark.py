"""Evaluate raw full, exact lossless full, and level0+recompute.

Formal benchmark follows the project contract:
- OPV2V clean test + existing OPV2V-W fog/rain/snow test
- no online weather augmentation
- all frames
- non-global AP
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

METRICS = ("ap30", "ap50", "ap70")


def _ap(eval_utils, stats):
    return {
        "ap30": float(eval_utils.calculate_ap(copy.deepcopy(stats), .3, False)[0]),
        "ap50": float(eval_utils.calculate_ap(copy.deepcopy(stats), .5, False)[0]),
        "ap70": float(eval_utils.calculate_ap(copy.deepcopy(stats), .7, False)[0]),
    }


def _new_stats():
    return {t: dict(tp=[], fp=[], gt=0, score=[]) for t in (.3, .5, .7)}


def _accumulate(eval_utils, boxes, scores, gt, target):
    frame = _new_stats()
    for threshold in frame:
        eval_utils.caluclate_tp_fp(boxes, scores, gt, frame, threshold)
        target[threshold]["gt"] += frame[threshold]["gt"]
        for key in ("tp", "fp", "score"):
            target[threshold][key].extend(frame[threshold][key])
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="gspr_evidence/experiment.yaml")
    p.add_argument("--frontend-config", required=True)
    p.add_argument("--frontend-checkpoint", required=True)
    p.add_argument("--phase", choices=("development", "benchmark"), required=True)
    p.add_argument("--weather", choices=("clean", "fog", "rain", "snow"), required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backend", choices=("auto", "zstd", "zlib"), default="auto")
    p.add_argument("--compression-level", type=int, default=3)
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--atol", type=float, default=2e-4)
    p.add_argument("--rtol", type=float, default=2e-4)
    args = p.parse_args()

    import torch
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from gspr_evidence import runtime as rt
    from gspr_evidence.benchmark import load_config as benchmark_config
    from .transport import lossless_full, level0_recompute

    rt.verify_frozen()
    formal = args.phase == "benchmark"
    if formal and args.smoke:
        raise ValueError("Formal benchmark cannot use --smoke")

    options, hypes = (
        benchmark_config(args.config, args.frontend_config, args.weather)
        if formal else rt.load_config(args.config, args.frontend_config)
    )

    if not formal:
        expected = Path("/data/scd/datasets/opv2v_official_data_dumping/validate").resolve()
        if Path(hypes["validate_dir"]).resolve() != expected:
            raise ValueError("Development evaluation must use full OPV2V validation")

    rt.seed_all(options["seed"])
    device = rt.device()
    model, digest = rt.load_model(
        hypes, options, args.frontend_checkpoint, device
    )

    branch = "clean" if formal else args.weather
    ds, loader, indices = rt.make_loader(
        hypes, options, weather=branch, smoke=args.smoke
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    modes = ("raw_full", "lossless_full", "level0_recompute", "original_full")
    stats = {m: _new_stats() for m in modes}
    totals = {
        m: dict(
            bytes=0, peak_bytes=0, raw_feature_bytes=0,
            encode_ms=0.0, decode_ms=0.0, recompute_ms=0.0,
            max_level0_error=0.0, max_level1_error=0.0, max_level2_error=0.0
        ) for m in modes
    }

    contract = rt.contract(options, args.frontend_config, digest)
    protocol = {
        "phase": args.phase,
        "weather": args.weather,
        "data_root": hypes["validate_dir"],
        "online_weather": bool(not formal and args.weather != "clean"),
        "global_sort_detections": False,
        "all_frames": not bool(args.smoke),
        "sample_indices": indices,
        "frontend_sha256": digest,
        "codec_backend_requested": args.backend,
        "codec_level": args.compression_level,
        "modes": list(modes),
        "note": (
            "raw_full is existing three-scale serialized full communication; "
            "lossless_full sends all scales with exact bit-preserving compression; "
            "level0_recompute sends only exact level0 then reruns frozen backbone blocks 1/2."
        ),
        "comparison_contract": contract,
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    start = time.perf_counter()
    with torch.no_grad(), (out / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for number, batch in enumerate(loader, 1):
            batch = to_device(batch, device)
            ego = batch["ego"]
            inp = rt.input_branch(ego, branch)
            encoded = model.encode(inp)

            raw_pred, raw_diag = model.run(encoded, "full")
            original = model.engine.base(inp)
            original_pred = {k: original[k] for k in ("psm", "rm")}

            lossless_pred, lossless_diag = lossless_full(
                model, encoded, backend=args.backend, level=args.compression_level
            )
            recompute_pred, recompute_diag = level0_recompute(
                model, encoded, backend=args.backend, level=args.compression_level
            )

            # Existing project already tolerates tiny kernel-level numeric variation.
            for name, pred in (
                ("original_full", original_pred),
                ("lossless_full", lossless_pred),
                ("level0_recompute", recompute_pred),
            ):
                for key in ("psm", "rm"):
                    torch.testing.assert_close(
                        pred[key], raw_pred[key],
                        atol=args.atol, rtol=args.rtol,
                        msg=lambda msg, n=name, k=key: f"{n}/{k}: {msg}"
                    )

            predictions = {
                "raw_full": raw_pred,
                "lossless_full": lossless_pred,
                "level0_recompute": recompute_pred,
                "original_full": original_pred,
            }
            diagnostics = {
                "raw_full": {
                    "total_bytes": raw_diag["total_bytes"],
                    "raw_feature_bytes": raw_diag.get("feature_bytes", raw_diag["total_bytes"]),
                    "encode_ms": 0.0, "decode_ms": 0.0, "recompute_ms": 0.0,
                    "max_level0_error": 0.0, "max_level1_error": 0.0, "max_level2_error": 0.0,
                },
                "lossless_full": lossless_diag,
                "level0_recompute": recompute_diag,
                # Logical original-full baseline uses the same raw communication accounting.
                "original_full": {
                    "total_bytes": raw_diag["total_bytes"],
                    "raw_feature_bytes": raw_diag.get("feature_bytes", raw_diag["total_bytes"]),
                    "encode_ms": 0.0, "decode_ms": 0.0, "recompute_ms": 0.0,
                    "max_level0_error": 0.0, "max_level1_error": 0.0, "max_level2_error": 0.0,
                },
            }

            full_gt = None
            frame_row = {
                "sample_index": int(ego["communication_sample_index"][0])
                if "communication_sample_index" in ego else number - 1,
                "modes": {}
            }

            for mode in modes:
                boxes, scores, gt = ds.post_process(batch, {"ego": predictions[mode]})
                if full_gt is None:
                    full_gt = gt
                else:
                    torch.testing.assert_close(gt, full_gt)
                per_frame = _accumulate(eval_utils, boxes, scores, gt, stats[mode])

                d = diagnostics[mode]
                agg = totals[mode]
                agg["bytes"] += int(d["total_bytes"])
                agg["peak_bytes"] = max(agg["peak_bytes"], int(d["total_bytes"]))
                for key in (
                    "raw_feature_bytes", "encode_ms", "decode_ms", "recompute_ms"
                ):
                    agg[key] += float(d.get(key, 0.0))
                for key in (
                    "max_level0_error", "max_level1_error", "max_level2_error"
                ):
                    agg[key] = max(agg[key], float(d.get(key, 0.0)))

                frame_row["modes"][mode] = {
                    "bytes": int(d["total_bytes"]),
                    "raw_feature_bytes": int(d.get("raw_feature_bytes", 0)),
                    "encode_ms": float(d.get("encode_ms", 0.0)),
                    "decode_ms": float(d.get("decode_ms", 0.0)),
                    "recompute_ms": float(d.get("recompute_ms", 0.0)),
                    "max_level0_error": float(d.get("max_level0_error", 0.0)),
                    "max_level1_error": float(d.get("max_level1_error", 0.0)),
                    "max_level2_error": float(d.get("max_level2_error", 0.0)),
                    "ap_inputs": per_frame,
                }

            stream.write(json.dumps(frame_row) + "\n")
            if number == 1 or number % 20 == 0:
                elapsed = time.perf_counter() - start
                eta_h = (len(indices) - number) * elapsed / number / 3600.0
                print(
                    f"{args.phase}/{args.weather}: {number}/{len(indices)}, "
                    f"{elapsed/number:.2f}s/frame, ETA {eta_h:.2f}h",
                    flush=True,
                )
                stream.flush()

    if number != len(indices):
        raise RuntimeError("Evaluation did not finish all requested frames")

    results = {}
    raw_ap = None
    for mode in modes:
        folder = out / mode
        folder.mkdir(exist_ok=True)
        eval_utils.eval_final_results(copy.deepcopy(stats[mode]), str(folder), False)
        row = _ap(eval_utils, stats[mode])
        if mode == "raw_full":
            raw_ap = row.copy()
        row.update({
            "frames": number,
            "mean_bytes": totals[mode]["bytes"] / number,
            "peak_bytes": totals[mode]["peak_bytes"],
            "mean_raw_feature_bytes": totals[mode]["raw_feature_bytes"] / number,
            "mean_encode_ms": totals[mode]["encode_ms"] / number,
            "mean_decode_ms": totals[mode]["decode_ms"] / number,
            "mean_recompute_ms": totals[mode]["recompute_ms"] / number,
            "max_level0_error": totals[mode]["max_level0_error"],
            "max_level1_error": totals[mode]["max_level1_error"],
            "max_level2_error": totals[mode]["max_level2_error"],
        })
        results[mode] = row

    for mode, row in results.items():
        row["ap_delta_vs_raw_full"] = {
            k: row[k] - raw_ap[k] for k in METRICS
        }
        row["no_ap_drop_vs_raw_full"] = all(row[k] >= raw_ap[k] for k in METRICS)
        row["wire_reduction_vs_raw_full"] = (
            1.0 - row["mean_bytes"] / max(results["raw_full"]["mean_bytes"], 1.0)
        )

    summary = {
        "phase": args.phase,
        "weather": args.weather,
        "comparison_contract": contract,
        "results": results,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rt.verify_frozen()
    print(f"Complete: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
