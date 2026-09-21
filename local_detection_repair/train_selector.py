"""Phase 2: learn KEEP vs REPAIR from the phase-1 frozen cache.

No weather simulation or frozen detector forward pass is repeated here. The fixed
repairer is applied to cached model-only candidates, while exact baseline replay
state cached in phase 1 is used to build selector action labels.
"""
import argparse
import json
from pathlib import Path
import random
import shutil
import time


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="local_detection_repair/experiment.yaml")
    p.add_argument("--frontend-config", required=True)
    p.add_argument("--frontend-checkpoint", required=True)
    p.add_argument("--repair-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ablation", choices=("score", "geometry", "joint"), default="joint")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--smoke", type=int, default=0,
        help="limit cached frames per weather for label generation and run one epoch",
    )
    return p.parse_args()


def _selector_cache_file(root, split, weather):
    return Path(root) / split / (weather + ".pt")


def _valid_selector_cache(path, contract, repair_sha, ablation, smoke):
    if not path.is_file():
        return None
    import torch
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if (
        payload.get("schema") != 2
        or payload.get("contract") != contract
        or payload.get("repair_sha256") != repair_sha
        or payload.get("ablation") != ablation
        or bool(payload.get("smoke")) != bool(smoke)
    ):
        return None
    return payload


def _build_cache(args, out, options, repair, contract, target, replay_ds):
    import torch

    from . import runtime as rt
    from .cache import (
        load_cache,
        load_manifest,
        candidates_to_device,
        prepared_to_device,
    )
    from .pipeline import (
        WEATHERS,
        build_selector_features,
        repaired_outputs,
        selector_action_label,
    )

    repair_root = Path(args.repair_checkpoint).resolve().parent / "repair_cache"
    repair_manifest = load_manifest(repair_root, contract=contract)
    repair_sha = rt.sha256(args.repair_checkpoint)

    cache = out / "selector_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        (cache / split).mkdir(exist_ok=True)

    files = {"train": {}, "validation": {}}
    totals = {
        "train": {
            "frames": 0, "candidates": 0, "safe_repair": 0,
            "unchanged": 0, "lost_tp_candidates": 0, "new_fp_candidates": 0,
            "recovered_targets": 0, "lost_targets": 0, "new_fp": 0,
        },
        "validation": {
            "frames": 0, "candidates": 0, "safe_repair": 0,
            "unchanged": 0, "lost_tp_candidates": 0, "new_fp_candidates": 0,
            "recovered_targets": 0, "lost_targets": 0, "new_fp": 0,
        },
    }

    source_split = {"train": "selector_train", "validation": "validation"}

    for split_index, split in enumerate(("train", "validation")):
        for weather_index, weather in enumerate(WEATHERS):
            out_file = _selector_cache_file(cache, split, weather)
            existing = _valid_selector_cache(
                out_file, contract, repair_sha, args.ablation, args.smoke
            )
            if existing is not None:
                files[split][weather] = str(out_file)
                for key, value in existing["stats"].items():
                    totals[split][key] += int(value)
                print(
                    f"SELECTOR CACHE SKIP {split}/{weather}: "
                    f"{existing['stats']['frames']} frames",
                    flush=True,
                )
                continue

            if out_file.exists():
                out_file.unlink()

            source = load_cache(
                repair_root, source_split[split], weather, contract=contract
            )
            records = source["records"]
            if args.smoke:
                records = records[:int(args.smoke)]

            generated = []
            weather_stats = {
                "frames": 0, "candidates": 0, "safe_repair": 0,
                "unchanged": 0, "lost_tp_candidates": 0, "new_fp_candidates": 0,
                "recovered_targets": 0, "lost_targets": 0, "new_fp": 0,
            }
            started = time.perf_counter()

            for frame_no, record in enumerate(records):
                candidates = candidates_to_device(record, target)
                prepared = prepared_to_device(record, target)
                batch = {
                    "ego": {
                        "transformation_matrix":
                            record["transformation_matrix"].to(target)
                    }
                }

                with torch.no_grad():
                    prediction = repair(candidates["features"])
                    boxes, scores, residual, repaired_score = repaired_outputs(
                        repair, candidates, prediction, options, args.ablation
                    )
                    selector_features = build_selector_features(
                        candidates, residual, repaired_score, options, args.ablation
                    )

                    labels = []
                    outcomes = []
                    for index in range(len(candidates["candidate_ids"])):
                        label, outcome = selector_action_label(
                            replay_ds,
                            batch,
                            None,
                            boxes[index],
                            scores[index],
                            float(options["selector_label_iou"]),
                            prepared=prepared,
                        )
                        labels.append(label)
                        outcomes.append(outcome)

                        weather_stats["safe_repair"] += int(label)
                        weather_stats["recovered_targets"] += len(outcome["recovered"])
                        weather_stats["lost_targets"] += len(outcome["lost"])
                        weather_stats["new_fp"] += int(outcome["new_fp"])
                        weather_stats["lost_tp_candidates"] += int(bool(outcome["lost"]))
                        weather_stats["new_fp_candidates"] += int(outcome["new_fp"] > 0)
                        if (
                            not outcome["recovered"]
                            and not outcome["lost"]
                            and int(outcome["new_fp"]) == 0
                        ):
                            weather_stats["unchanged"] += 1

                generated.append({
                    "features": selector_features.detach().cpu().float(),
                    "labels": torch.tensor(labels, dtype=torch.float32),
                    "sample_index": int(record["sample_index"]),
                    "weather": weather,
                    "outcomes": outcomes,
                })
                weather_stats["frames"] += 1
                weather_stats["candidates"] += len(labels)

                done = frame_no + 1
                if done == 1 or done % 50 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"SELECTOR CACHE {split}/{weather}: {done}/{len(records)} "
                        f"frames, {elapsed/done:.2f}s/frame, "
                        f"labels={weather_stats['candidates']}",
                        flush=True,
                    )

            payload = {
                "schema": 2,
                "contract": contract,
                "repair_sha256": repair_sha,
                "repair_cache_schema": repair_manifest["schema"],
                "ablation": args.ablation,
                "smoke": bool(args.smoke),
                "split": split,
                "weather": weather,
                "records": generated,
                "stats": weather_stats,
            }
            rt.atomic_torch_save(payload, out_file)
            files[split][weather] = str(out_file)
            for key, value in weather_stats.items():
                totals[split][key] += int(value)
            print(
                f"SELECTOR CACHE DONE {split}/{weather}: "
                f"{weather_stats['frames']} frames in "
                f"{(time.perf_counter()-started)/3600:.2f}h",
                flush=True,
            )

    manifest = {
        "schema": 2,
        "complete": True,
        "contract": contract,
        "repair_sha256": repair_sha,
        "repair_cache_schema": repair_manifest["schema"],
        "ablation": args.ablation,
        "smoke": bool(args.smoke),
        "label_iou": float(options["selector_label_iou"]),
        "label_definition":
            "REPAIR iff this fixed repaired candidate alone recovers >=1 baseline-missed GT "
            "with zero baseline-TP loss and zero new FP; harmful and unchanged are KEEP",
        "files": files,
        "totals": totals,
        "compute_plan":
            "reuse phase-1 candidate and compact baseline caches; no weather simulation "
            "or frozen detector forward pass is repeated",
    }
    rt.write_json(cache / "manifest.json", manifest)
    return manifest


def main():
    args = _args()
    import torch
    import torch.nn.functional as F
    from torch import nn

    from . import runtime as rt
    from .cache import load_manifest
    from .model import LocalRepairNet, RepairSelector
    from .pipeline import WEATHERS, candidate_feature_dim, selector_feature_dim

    if args.smoke < 0:
        raise ValueError("--smoke cannot be negative")

    options, hypes = rt.load_config(args.config, args.frontend_config)
    rt.seed_all(options["seed"])
    target = rt.device()

    # No frozen frontend is loaded in phase 2. Its hash and the phase-1 cache
    # contract are enough to prove the selector is built from the same detector.
    frontend_sha = rt.sha256(args.frontend_checkpoint)
    scene_ds, _, _ = rt.make_loader(
        hypes, options, "train", "clean", shuffle=False, smoke=1
    )
    repair_scenes, selector_scenes = rt.scene_split(
        len(scene_ds.len_record), options["repair_scene_fraction"], options["seed"]
    )
    current_contract = rt.contract(
        args.config,
        args.frontend_config,
        args.frontend_checkpoint,
        options,
        frontend_sha,
        repair_scenes,
        selector_scenes,
    )

    repair_state = torch.load(
        args.repair_checkpoint, map_location=target, weights_only=False
    )
    rt.ensure_contract(repair_state["contract"], current_contract)
    if int(repair_state["feature_dim"]) != candidate_feature_dim(options):
        raise ValueError("repair checkpoint feature definition changed")

    repair = LocalRepairNet(
        candidate_feature_dim(options), options["repair"]
    ).to(target)
    repair.load_state_dict(repair_state["repair"], strict=True)
    repair.requires_grad_(False)
    repair.eval()

    repair_cache_root = Path(args.repair_checkpoint).resolve().parent / "repair_cache"
    load_manifest(repair_cache_root, contract=current_contract)

    selector = RepairSelector(
        selector_feature_dim(options), options["selector"]
    ).to(target)
    optimizer = torch.optim.AdamW(
        selector.parameters(),
        lr=float(options["selector_learning_rate"]),
        weight_decay=float(options["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(options["lr_step"]),
        gamma=float(options["lr_gamma"]),
    )
    rt.optimizer_only(optimizer, [selector])

    out = rt.safe_output(args.output_dir, resume=args.resume)
    protocol_path = out / "training_protocol.json"
    if not args.resume:
        shutil.copy2(args.config, out / "experiment.yaml")
        rt.write_json(protocol_path, {
            "phase": "selector",
            "smoke": bool(args.smoke),
            "ablation": args.ablation,
            "contract": current_contract,
            "repair_checkpoint": str(Path(args.repair_checkpoint).resolve()),
            "repair_sha256": rt.sha256(args.repair_checkpoint),
            "scene_rule":
                "repairer uses repair_scenes; selector action labels use disjoint "
                "selector_scenes; official validation only selects checkpoint; "
                "no test data generates labels",
            "compute_plan":
                "selector labels reuse phase-1 cached candidates and compact baseline "
                "replay; frozen detector and weather simulator are not rerun",
        })
    elif not protocol_path.is_file():
        raise FileNotFoundError("resume directory is missing training_protocol.json")

    manifest = _build_cache(
        args, out, options, repair, current_contract, target, scene_ds
    )

    begin, best = 0, float("inf")
    last_path = out / "last.pth"
    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location=target, weights_only=False)
        rt.ensure_contract(state["contract"], current_contract)
        if state["repair_sha256"] != rt.sha256(args.repair_checkpoint):
            raise ValueError("resume selector uses a different repairer")
        if state["ablation"] != args.ablation:
            raise ValueError("resume selector ablation changed")
        selector.load_state_dict(state["selector"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        begin = int(state["epoch"]) + 1
        best = float(state["best_validation_loss"])
        rt.restore_rng(state["rng"])
    elif args.resume:
        print("RESUME: selector cache exists but no selector checkpoint yet", flush=True)

    positive_weight = torch.tensor(
        float(options["selector"]["positive_weight"]), device=target
    )
    epochs = 1 if args.smoke else int(options["selector_epochs"])
    history_path = out / "history.jsonl"

    for epoch in range(begin, epochs):
        metrics = {}
        for split in ("train", "validation"):
            training = split == "train"
            selector.train(training)
            repair.eval()

            all_records = []
            for weather in WEATHERS:
                path = Path(manifest["files"][split][weather])
                payload = torch.load(path, map_location="cpu", weights_only=False)
                all_records.extend(payload["records"])

            order = list(range(len(all_records)))
            if training:
                random.Random(
                    int(options["seed"]) + epoch + 900000
                ).shuffle(order)

            loss_sum = total = correct = tp = fp = fn = 0
            for position, record_index in enumerate(order):
                record = all_records[record_index]
                features = record["features"].to(target)
                labels = record["labels"].to(target)
                if not len(labels):
                    continue

                with torch.set_grad_enabled(training):
                    logits = selector(features)
                    loss = F.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        pos_weight=positive_weight,
                        reduction="mean",
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite selector loss")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(selector.parameters(), 5.0)
                        if any(p.grad is not None for p in repair.parameters()):
                            raise AssertionError("frozen repairer accumulated gradients")
                        optimizer.step()

                prediction = logits.detach().sigmoid() >= float(
                    options["selector"]["threshold"]
                )
                truth = labels.bool()
                n = len(labels)
                loss_sum += float(loss.detach()) * n
                total += n
                correct += int((prediction == truth).sum())
                tp += int((prediction & truth).sum())
                fp += int((prediction & ~truth).sum())
                fn += int((~prediction & truth).sum())

                if position == 0 or (position + 1) % 500 == 0:
                    print(
                        f"selector-cache epoch={epoch+1} {split}: "
                        f"{position+1}/{len(order)} records",
                        flush=True,
                    )

            if not total:
                raise RuntimeError("selector cache contains no candidates")

            metrics[split] = {
                "loss": loss_sum / total,
                "candidates": total,
                "accuracy": correct / total,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "positive_labels": tp + fn,
                "predicted_repair": tp + fp,
            }

        scheduler.step()
        score = float(metrics["validation"]["loss"])
        improved = score < best
        best = min(best, score)
        state = {
            "phase": "selector",
            "epoch": epoch,
            "best_validation_loss": best,
            "selector": selector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rt.rng_state(),
            "feature_dim": selector_feature_dim(options),
            "contract": current_contract,
            "repair_sha256": rt.sha256(args.repair_checkpoint),
            "ablation": args.ablation,
            "metrics": metrics,
            "selector_cache_totals": manifest["totals"],
            "smoke": bool(args.smoke),
        }
        rt.atomic_torch_save(state, last_path)
        if improved:
            rt.atomic_torch_save(state, out / "selector_best.pth")

        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **metrics,
            }, ensure_ascii=False) + "\n")

        print(json.dumps({
            "epoch": epoch + 1,
            "best_validation_loss": best,
            **metrics,
        }, ensure_ascii=False), flush=True)

    print("DONE selector:", out / "selector_best.pth", flush=True)


if __name__ == "__main__":
    main()
