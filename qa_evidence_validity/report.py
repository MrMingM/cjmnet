"""Unified Stage-2 evidence-validity report.

Combines Stage-1 target statistics, offline sensitivity/stratification, and the
new peer-alone detector audit. It deliberately avoids declaring mechanisms
from proxy thresholds alone.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from qa_observation_diagnostic.metrics import distance_bin
from qa_evidence_validity.offline import (
    ADVERSE,
    QUANTILES,
    build_thresholds,
    key,
    load_stage1,
    support_strong,
    threshold_for,
)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_ratio(num, den):
    return num / den if den else None


def confusion_metrics(tp, fp, fn, tn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "specificity": safe_ratio(tn, tn + fp),
        "accuracy": safe_ratio(tp + tn, tp + fp + fn + tn),
    }


def _top_share(counter, k=3):
    values = sorted(counter.values(), reverse=True)
    total = sum(values)
    return sum(values[:k]) / total if total else None


def load_peer_root(root):
    root = Path(root)
    rows, protocols = {}, {}
    for weather in ADVERSE:
        folder = root / weather
        rows[weather] = read_jsonl(folder / "peer_targets.jsonl")
        protocols[weather] = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        if protocols[weather].get("test_data_used", True) or not protocols[weather].get("development_only", False):
            raise ValueError("Peer audit must be development-only")
    return rows, protocols


def peer_metric_validity(targets, maps, peer_rows):
    thresholds = {q: build_thresholds(targets["clean"], q) for q in QUANTILES}
    peer_maps = {w: {key(r): r for r in peer_rows[w]} for w in ADVERSE}
    result = {}

    for weather in ADVERSE:
        weather_result = {"by_quantile": {}}
        for q in QUANTILES:
            ths = thresholds[q]
            source_counts = Counter()
            target_counts = Counter()
            metric_fusion_candidates = set()
            source_valid_fusion_candidates = set()
            metric_candidate_validated = set()
            scene_source_valid = Counter()
            distance_source_valid = Counter()

            for k, audit in peer_maps[weather].items():
                stage = maps[weather].get(k)
                clean = maps["clean"].get(k)
                if stage is None or clean is None or not clean["ego_detected"] or stage["ego_detected"]:
                    continue
                th = threshold_for(stage, ths)
                ego_metric_strong = support_strong(stage["ego"], th)

                metric_peer_flags = []
                actual_peer_flags = []
                if len(stage["peers"]) != len(audit["peers"]):
                    raise ValueError(f"Peer count mismatch for {weather} {k}")
                for stat, observed in zip(stage["peers"], audit["peers"]):
                    pred = support_strong(stat, th)
                    actual = bool(observed["matched"])
                    metric_peer_flags.append(pred)
                    actual_peer_flags.append(actual)
                    if pred and actual:
                        source_counts["tp"] += 1
                    elif pred and not actual:
                        source_counts["fp"] += 1
                    elif (not pred) and actual:
                        source_counts["fn"] += 1
                    else:
                        source_counts["tn"] += 1

                metric_any = any(metric_peer_flags)
                actual_any = any(actual_peer_flags)
                if metric_any and actual_any:
                    target_counts["tp"] += 1
                elif metric_any and not actual_any:
                    target_counts["fp"] += 1
                elif (not metric_any) and actual_any:
                    target_counts["fn"] += 1
                else:
                    target_counts["tn"] += 1

                if (not ego_metric_strong) and metric_any and not stage["full_detected"]:
                    metric_fusion_candidates.add(k)
                    if actual_any:
                        metric_candidate_validated.add(k)

                if actual_any and not stage["full_detected"]:
                    source_valid_fusion_candidates.add(k)
                    scene_source_valid[str(stage["scene"])] += 1
                    distance_source_valid[distance_bin(stage["distance"])] += 1

            source = confusion_metrics(
                source_counts["tp"], source_counts["fp"], source_counts["fn"], source_counts["tn"]
            )
            target = confusion_metrics(
                target_counts["tp"], target_counts["fp"], target_counts["fn"], target_counts["tn"]
            )
            qname = f"q{int(q*100):02d}"
            weather_result["by_quantile"][qname] = {
                "source_level_metric_vs_peer_alone": source,
                "target_level_any_peer_metric_vs_any_peer_alone": target,
                "metric_fusion_candidate_n": len(metric_fusion_candidates),
                "metric_fusion_candidate_validated_by_peer_alone_n": len(metric_candidate_validated),
                "metric_fusion_candidate_validation_fraction":
                    safe_ratio(len(metric_candidate_validated), len(metric_fusion_candidates)),
                "source_valid_full_miss_n": len(source_valid_fusion_candidates),
                "source_valid_full_miss_top3_scene_share": _top_share(scene_source_valid, 3),
                "source_valid_full_miss_by_scene": dict(scene_source_valid),
                "source_valid_full_miss_by_distance": dict(distance_source_valid),
            }

        all_audits = list(peer_maps[weather].values())
        source_valid_full_miss = [r for r in all_audits if r["any_peer_alone_detected"] and not r["full"]["matched"]]
        source_valid_full_recover = [r for r in all_audits if r["any_peer_alone_detected"] and r["full"]["matched"]]
        no_peer_detect = [r for r in all_audits if not r["any_peer_alone_detected"]]
        weather_result["task_observed"] = {
            "weather_induced_ego_miss_n": len(all_audits),
            "any_peer_alone_detected_n":
                len(source_valid_full_miss) + len(source_valid_full_recover),
            "any_peer_alone_detected_fraction":
                safe_ratio(len(source_valid_full_miss) + len(source_valid_full_recover), len(all_audits)),
            "source_valid_full_miss_n": len(source_valid_full_miss),
            "source_valid_full_miss_fraction_of_weather_ego_misses":
                safe_ratio(len(source_valid_full_miss), len(all_audits)),
            "source_valid_full_recover_n": len(source_valid_full_recover),
            "source_valid_full_recover_fraction_of_weather_ego_misses":
                safe_ratio(len(source_valid_full_recover), len(all_audits)),
            "no_peer_alone_detection_n": len(no_peer_detect),
            "no_peer_alone_detection_fraction": safe_ratio(len(no_peer_detect), len(all_audits)),
        }
        result[weather] = weather_result
    return result


def conclusions(offline, validity):
    lines = []
    for weather in ADVERSE:
        q25 = validity[weather]["by_quantile"]["q25"]
        src = q25["source_level_metric_vs_peer_alone"]
        task = validity[weather]["task_observed"]
        lines.append({
            "weather": weather,
            "metric_peer_precision_q25": src["precision"],
            "metric_peer_recall_q25": src["recall"],
            "metric_fusion_candidate_validation_fraction_q25":
                q25["metric_fusion_candidate_validation_fraction"],
            "source_valid_full_miss_n": task["source_valid_full_miss_n"],
            "source_valid_full_recover_n": task["source_valid_full_recover_n"],
            "ego_strong_fraction_range_q20_to_q40":
                offline["threshold_sensitivity"][weather]["ego_strong_fraction_range"],
            "interpretation":
                "peer-alone detection is stronger task-usable evidence than proxy strength; "
                "source-valid/full-miss remains a fusion candidate, not proof of attention causality.",
        })
    return lines


def markdown(result):
    off = result["offline"]
    val = result["peer_validity"]
    lines = [
        "# Stage-2 evidence validity audit",
        "",
        "> Development OPV2V validation + online weather only. This audit checks whether the "
        "Stage-1 evidence-strength ruler is trustworthy before any Oracle or new module is designed.",
        "",
        "## 1. What this audit can and cannot establish",
        "",
        "- `peer-alone detected` means one peer's pre-fusion multiscale features, by themselves, "
        "produce a final IoU>=0.7 matched box in the ego frame.",
        "- This is stronger evidence of task usability than high N_eff/coverage, but it is still "
        "conditioned on the same frozen detector head.",
        "- `peer-alone detected + full miss` is a **source-valid / fusion-invalid candidate**. "
        "It does not prove attention is the unique cause.",
        "- Q20/Q25/Q30/Q40 are sensitivity probes, not deployable thresholds.",
        "",
        "## 2. Threshold sensitivity of the Stage-1 strong/weak ruler",
        "",
        "| Weather | Ego-strong fraction range Q20-Q40 | Peer-strong fraction range Q20-Q40 |",
        "|---|---:|---:|",
    ]
    for weather in ADVERSE:
        r = off["threshold_sensitivity"][weather]
        lines.append(
            f"| {weather} | {r['ego_strong_fraction_range']:.6f} | "
            f"{r['peer_strong_fraction_range']:.6f} |"
        )

    lines += [
        "",
        "The smaller these ranges and the higher the Jaccard overlap with Q25, the less the conclusion "
        "depends on one arbitrary threshold. Stability does **not** prove semantic correctness.",
        "",
        "## 3. Q25 proxy strength vs actual peer-alone detection",
        "",
        "| Weather | Source precision | Source recall | Any-peer precision | Any-peer recall | "
        "Metric fusion candidates | Validated by peer-alone | Validation fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for weather in ADVERSE:
        q = val[weather]["by_quantile"]["q25"]
        s = q["source_level_metric_vs_peer_alone"]
        t = q["target_level_any_peer_metric_vs_any_peer_alone"]
        lines.append(
            f"| {weather} | {s['precision']} | {s['recall']} | {t['precision']} | {t['recall']} | "
            f"{q['metric_fusion_candidate_n']} | {q['metric_fusion_candidate_validated_by_peer_alone_n']} | "
            f"{q['metric_fusion_candidate_validation_fraction']} |"
        )

    lines += [
        "",
        "## 4. Task-observed source-valid cases",
        "",
        "| Weather | Weather ego misses | Any peer-alone detects | Full recovers | Full still misses | "
        "No peer-alone detection |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for weather in ADVERSE:
        t = val[weather]["task_observed"]
        lines.append(
            f"| {weather} | {t['weather_induced_ego_miss_n']} | {t['any_peer_alone_detected_n']} | "
            f"{t['source_valid_full_recover_n']} | {t['source_valid_full_miss_n']} | "
            f"{t['no_peer_alone_detection_n']} |"
        )

    lines += [
        "",
        "## 5. Ego -> full degradation already visible in Stage-1",
        "",
        "| Weather | Ego-detected -> full-miss fraction | Full recovery given ego miss | "
        "Top-3 scene share of ego->full degradation |",
        "|---|---:|---:|---:|",
    ]
    for weather in ("clean",) + ADVERSE:
        r = off["ego_full_transitions"][weather]
        lines.append(
            f"| {weather} | {r['ego_to_full_degradation_fraction_given_ego_detected']} | "
            f"{r['full_recovery_fraction_given_ego_miss']} | "
            f"{r['top3_scene_share_ego_to_full_degradation']} |"
        )

    lines += [
        "",
        "## 6. Corrected N_eff change audit",
        "",
        "The previous huge Snow relative-loss value came from near-zero clean denominators. "
        "This report keeps zero-clean cases explicit and adds absolute, bounded-symmetric, and log1p differences. "
        "The bounded metric is a **new metric** and is not renamed as the old relative-loss rate.",
        "",
        "| Weather | Group | N | Clean N_eff=0 frac | Absolute delta mean | Bounded delta mean | "
        "Log1p delta mean | Relative loss mean (clean>0 only) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for weather in ADVERSE:
        for group in ("low_occlusion", "high_occlusion"):
            r = off["corrected_neff_change"][weather][group]
            lines.append(
                f"| {weather} | {group} | {r['n']} | {r['clean_neff_zero_fraction']} | "
                f"{r['absolute_delta_mean']} | {r['bounded_symmetric_delta_mean']} | "
                f"{r['log1p_delta_mean']} | {r['relative_loss_clean_positive_mean']} |"
            )

    lines += [
        "",
        "## 7. Scene/distance concentration",
        "",
        "Use the machine-readable JSON for full per-scene/per-distance counts. High concentration means "
        "the phenomenon may be scenario-specific and should not be promoted to a general mechanism yet.",
        "",
        "## 8. Gate after Stage-2",
        "",
        "Do **not** design a new fusion/local module from proxy strength alone.",
        "",
        "- If Q20-Q40 labels are unstable or proxy precision against peer-alone detection is poor, refine the evidence ruler first.",
        "- If a non-trivial, cross-scene set of `peer-alone detected + full miss` cases survives, those cases qualify for a small source-combination/local-fusion intervention.",
        "- For local H-A5 candidates, Stage-2 still does not prove that high feature magnitude contains correct target semantics. "
        "Hierarchical clean-information intervention should only be framed as recoverability at an intervention site, not failure-origin localization.",
        "- No OPV2V-W test data should be used for these gates.",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-root", required=True)
    p.add_argument("--offline-results", required=True)
    p.add_argument("--peer-root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    offline = json.loads(Path(args.offline_results).read_text(encoding="utf-8"))
    targets, maps, _, _ = load_stage1(args.stage1_root)
    peer_rows, peer_protocols = load_peer_root(args.peer_root)

    for weather in ADVERSE:
        if Path(peer_protocols[weather]["stage1_root"]).resolve() != Path(args.stage1_root).resolve():
            raise ValueError(f"Peer audit {weather} used a different Stage-1 root")

    validity = peer_metric_validity(targets, maps, peer_rows)
    result = {
        "schema": 1,
        "stage1_root": str(Path(args.stage1_root).resolve()),
        "offline": offline,
        "peer_validity": validity,
        "summary": conclusions(offline, validity),
        "interpretation_boundaries": [
            "Proxy strong/weak is not task-usable evidence truth.",
            "Peer-alone detection is detector-conditioned and therefore a conservative task-usability test.",
            "Source-valid/full-miss is a fusion-bottleneck candidate, not proof that AttFuse attention is the unique cause.",
            "Threshold robustness is necessary for a stable proxy claim but not sufficient for semantic validity.",
            "All results are development diagnostics; no OPV2V-W test data are used.",
        ],
    }
    (out / "evidence_validity_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "evidence_validity_report.md").write_text(markdown(result), encoding="utf-8")
    print(f"Wrote {out/'evidence_validity_report.md'}")
    print(f"Wrote {out/'evidence_validity_results.json'}")


if __name__ == "__main__":
    main()
