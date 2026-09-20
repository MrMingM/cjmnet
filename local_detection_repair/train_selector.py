"""Phase 2: freeze the chosen repairer and learn KEEP vs REPAIR.

Labels are generated only on the scene-disjoint selector subset of OPV2V train.
Each label measures the fixed repairer's actual one-candidate post-NMS outcome:
safe recovery -> REPAIR; harmful or unchanged -> KEEP.
"""
import argparse
import json
from pathlib import Path
import random
import shutil


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="local_detection_repair/experiment.yaml")
    p.add_argument("--frontend-config", required=True)
    p.add_argument("--frontend-checkpoint", required=True)
    p.add_argument("--repair-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ablation", choices=("score", "geometry", "joint"), default="joint")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", type=int, default=0,
                   help="limit frames per weather for cache generation and run one epoch")
    return p.parse_args()


def _build_cache(args, out, options, hypes, frontend, repair, contract, target):
    import torch
    from opencood.tools.train_utils import to_device
    from . import runtime as rt
    from .pipeline import (
        WEATHERS, build_selector_features, frozen_full_and_sources,
        generate_candidates, prepare_postprocess, repaired_outputs,
        selector_action_label,
    )

    cache = out / "selector_cache"
    manifest_path = cache / "manifest.json"
    repair_sha = rt.sha256(args.repair_checkpoint)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not manifest.get("complete") or manifest.get("contract") != contract or
                manifest.get("repair_sha256") != repair_sha or
                manifest.get("ablation") != args.ablation or
                bool(manifest.get("smoke")) != bool(args.smoke)):
            raise ValueError("existing selector cache contract differs")
        return manifest
    if cache.exists():
        raise RuntimeError(
            "incomplete selector_cache exists; keep it for diagnosis and use a new output directory"
        )
    (cache / "train").mkdir(parents=True)
    (cache / "validation").mkdir()
    records = {"train": [], "validation": []}
    totals = {"train": {"candidates": 0, "repair": 0},
              "validation": {"candidates": 0, "repair": 0}}
    label_iou = float(options["selector_label_iou"])

    for split in ("train", "validation"):
        scenes = contract["selector_scenes"] if split == "train" else None
        for weather_index, weather in enumerate(WEATHERS):
            run_seed = (int(options["seed"]) + 700000 +
                        weather_index * 1000 + (0 if split == "train" else 50000))
            rt.seed_all(run_seed)
            ds, loader, indices = rt.make_loader(
                hypes, options, split, weather, scenes=scenes,
                shuffle=False, seed=run_seed, smoke=args.smoke
            )
            for frame_no, batch in enumerate(loader):
                sample_index = int(batch["ego"]["communication_sample_index"][0])
                batch = to_device(batch, target)
                inp = rt.input_branch(batch, weather, benchmark=False)
                with torch.no_grad():
                    full, sources = frozen_full_and_sources(
                        frontend, inp, options["candidate"]["max_sources"],
                        verify_full=(split == "train" and weather_index == 0 and frame_no == 0),
                    )
                    candidates = generate_candidates(
                        ds, batch, full, sources, options,
                        hypes["model"]["args"]["lidar_range"]
                    )
                    prediction = repair(candidates["features"])
                    boxes, scores, residual, repaired_score = repaired_outputs(
                        repair, candidates, prediction, options, args.ablation
                    )
                    selector_features = build_selector_features(
                        candidates, residual, repaired_score, options
                    )
                    prepared = prepare_postprocess(ds, batch, full)
                    labels, outcomes = [], []
                    for index in range(len(candidates["candidate_ids"])):
                        label, outcome = selector_action_label(
                            ds, batch, full, boxes[index], scores[index],
                            label_iou, prepared=prepared
                        )
                        labels.append(label)
                        outcomes.append({
                            "candidate_id": int(candidates["candidate_ids"][index]),
                            "label": int(label),
                            **outcome,
                        })
                label_tensor = torch.tensor(labels, dtype=torch.float32)
                record = {
                    "features": selector_features.detach().cpu().float(),
                    "labels": label_tensor,
                    "sample_index": sample_index,
                    "weather": weather,
                }
                name = f"{weather}_{sample_index:08d}.pt"
                torch.save(record, cache / split / name)
                with (cache / split / "outcomes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({
                        "sample_index": sample_index,
                        "weather": weather,
                        "outcomes": outcomes,
                    }, ensure_ascii=False) + "\n")
                records[split].append(name)
                totals[split]["candidates"] += len(labels)
                totals[split]["repair"] += int(sum(labels))
                if len(records[split]) == 1 or len(records[split]) % 100 == 0:
                    print(
                        f"selector cache {split} {weather}: {len(records[split])} frames, "
                        f"{totals[split]['candidates']} candidate labels",
                        flush=True,
                    )

    manifest = {
        "complete": True,
        "schema": 1,
        "contract": contract,
        "repair_sha256": repair_sha,
        "ablation": args.ablation,
        "smoke": bool(args.smoke),
        "label_iou": label_iou,
        "label_definition":
            "REPAIR iff this fixed repaired candidate alone recovers >=1 baseline-missed GT "
            "with zero baseline-TP loss and zero new FP; harmful and unchanged are KEEP",
        "records": records,
        "totals": totals,
    }
    rt.write_json(manifest_path, manifest)
    return manifest


def main():
    args = _args()
    import torch
    import torch.nn.functional as F
    from torch import nn
    from . import runtime as rt
    from .model import LocalRepairNet, RepairSelector
    from .pipeline import candidate_feature_dim, selector_feature_dim

    if args.smoke < 0:
        raise ValueError("--smoke cannot be negative")
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
    current_contract = rt.contract(
        args.config, args.frontend_config, args.frontend_checkpoint,
        options, frontend_sha, repair_scenes, selector_scenes
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

    selector = RepairSelector(
        selector_feature_dim(options), options["selector"]
    ).to(target)
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=float(options["selector_learning_rate"]),
        weight_decay=float(options["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(options["lr_step"]),
        gamma=float(options["lr_gamma"])
    )
    rt.optimizer_only(optimizer, [selector])

    out = rt.safe_output(args.output_dir, resume=args.resume)
    if not args.resume:
        shutil.copy2(args.config, out / "experiment.yaml")
        rt.write_json(out / "training_protocol.json", {
            "phase": "selector",
            "smoke": bool(args.smoke),
            "ablation": args.ablation,
            "contract": current_contract,
            "repair_checkpoint": str(Path(args.repair_checkpoint).resolve()),
            "repair_sha256": rt.sha256(args.repair_checkpoint),
            "scene_rule":
                "repairer uses repair_scenes; selector action labels use disjoint selector_scenes; "
                "official validation only selects checkpoint; no test data generates labels",
        })

    manifest = _build_cache(
        args, out, options, hypes, frontend, repair, current_contract, target
    )
    begin, best = 0, float("inf")
    if args.resume and (out / "last.pth").is_file():
        state = torch.load(out / "last.pth", map_location=target, weights_only=False)
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
            frontend.eval()
            names = list(manifest["records"][split])
            if training:
                random.Random(int(options["seed"]) + epoch + 900000).shuffle(names)
            loss_sum = total = correct = tp = fp = fn = 0
            for index, name in enumerate(names):
                record = torch.load(
                    out / "selector_cache" / split / name,
                    map_location=target, weights_only=True
                )
                features = record["features"].to(target)
                labels = record["labels"].to(target)
                if not len(labels):
                    continue
                with torch.set_grad_enabled(training):
                    logits = selector(features)
                    loss = F.binary_cross_entropy_with_logits(
                        logits, labels, pos_weight=positive_weight, reduction="mean"
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite selector loss")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(selector.parameters(), 5.0)
                        rt.assert_frozen(frontend)
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
                if index == 0 or (index + 1) % 500 == 0:
                    print(
                        f"selector epoch={epoch+1} {split}: {index+1}/{len(names)} records",
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
            rt.assert_frozen(frontend)

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
            "smoke": bool(args.smoke),
        }
        rt.atomic_torch_save(state, out / "last.pth")
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
