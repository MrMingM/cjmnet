"""Phase 1: cache frozen detector work once, then train only the repairer."""
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
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--smoke", type=int, default=0,
        help="limit cached frames per weather and run one repair epoch",
    )
    return p.parse_args()


def _merge(total, row):
    for key, value in row.items():
        total[key] = total.get(key, 0.0) + float(value)


def main():
    args = _args()
    import torch
    from torch import nn

    from . import runtime as rt
    from .cache import build_repair_cache, load_cache
    from .model import LocalRepairNet
    from .pipeline import WEATHERS, candidate_feature_dim, repair_loss

    if args.smoke < 0:
        raise ValueError("--smoke cannot be negative")

    options, hypes = rt.load_config(args.config, args.frontend_config)
    rt.seed_all(options["seed"])
    target = rt.device()
    frontend, frontend_sha = rt.load_frontend(
        hypes, options, args.frontend_checkpoint, target
    )

    # Read official train scene boundaries once, then keep repair/selector scenes disjoint.
    scene_ds, _, _ = rt.make_loader(
        hypes, options, "train", "clean", shuffle=False, smoke=1
    )
    repair_scenes, selector_scenes = rt.scene_split(
        len(scene_ds.len_record),
        options["repair_scene_fraction"],
        options["seed"],
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

    out = rt.safe_output(args.output_dir, resume=args.resume)
    protocol_path = out / "training_protocol.json"
    if not args.resume:
        shutil.copy2(args.config, out / "experiment.yaml")
        rt.write_json(protocol_path, {
            "phase": "repair",
            "smoke": bool(args.smoke),
            "contract": current_contract,
            "candidate_definition":
                "union of model-only full/source top anchors before final score filter/NMS; "
                "deduplicated by anchor id and capped before any GT access",
            "supervision":
                "GT only after candidates are fixed: any-source IoU>=positive is positive, "
                "IoU<negative is background, middle is ignored; geometry target is from the "
                "unchanged full box to matched GT; score target is best source localization IoU",
            "compute_plan":
                "weather simulation + frozen GSPR/PointPillar/source decode + GT matching are "
                "cached once per frame; repair epochs reuse per-frame tensors and never rerun "
                "the frozen detector or weather simulator",
            "checkpoint_selection":
                "mean cached loss over official OPV2V validation clean/fog/rain/snow development conditions",
        })
    elif not protocol_path.is_file():
        raise FileNotFoundError("resume directory is missing training_protocol.json")

    # This is the expensive stage. It is resumable at split/weather granularity:
    # completed cache files are kept, while only an interrupted weather is rebuilt.
    cache_manifest = build_repair_cache(
        out,
        hypes,
        options,
        frontend,
        current_contract,
        target,
        repair_scenes,
        selector_scenes,
        smoke=args.smoke,
    )
    rt.assert_frozen(frontend)

    # The frozen detector is no longer needed for repair epochs.
    del frontend
    torch.cuda.empty_cache()

    feature_dim = candidate_feature_dim(options)
    repair = LocalRepairNet(feature_dim, options["repair"]).to(target)
    optimizer = torch.optim.AdamW(
        repair.parameters(),
        lr=float(options["learning_rate"]),
        weight_decay=float(options["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(options["lr_step"]),
        gamma=float(options["lr_gamma"]),
    )
    rt.optimizer_only(optimizer, [repair])

    begin, best = 0, float("inf")
    last_path = out / "last.pth"
    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location=target, weights_only=False)
        rt.ensure_contract(state["contract"], current_contract)
        if int(state["feature_dim"]) != feature_dim:
            raise ValueError("resume feature dimension changed")
        repair.load_state_dict(state["repair"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        begin = int(state["epoch"]) + 1
        best = float(state["best_validation_loss"])
        rt.restore_rng(state["rng"])
    elif args.resume:
        # A previous run may have stopped while generating the one-time cache.
        # Reuse the completed cache files and start repair training from epoch 0.
        print("RESUME: cache exists but no repair checkpoint yet; starting epoch 1", flush=True)

    epochs = 1 if args.smoke else int(options["repair_epochs"])
    history_path = out / "history.jsonl"
    cache_root = out / "repair_cache"

    for epoch in range(begin, epochs):
        epoch_metrics = {}
        for split in ("train", "validation"):
            training = split == "train"
            cache_split = "repair_train" if training else "validation"
            repair.train(training)
            totals = {}
            frames = updates = 0

            for weather_index, weather in enumerate(WEATHERS):
                payload = load_cache(
                    cache_root, cache_split, weather, contract=current_contract
                )
                records = payload["records"]
                order = list(range(len(records)))
                epoch_seed = (
                    int(options["seed"]) + epoch * 1000
                    + weather_index * 100 + (0 if training else 50000)
                )
                rt.seed_all(epoch_seed)
                if training:
                    random.Random(epoch_seed).shuffle(order)

                for local_index, record_index in enumerate(order):
                    record = records[record_index]
                    features = record["candidates"]["features"].to(
                        target, non_blocking=False
                    )
                    targets = {
                        key: value.to(target, non_blocking=False)
                        for key, value in record["targets"].items()
                    }

                    with torch.set_grad_enabled(training):
                        prediction = repair(features)
                        loss, stats = repair_loss(
                            repair, prediction, targets, options
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite repair loss")
                        if training and len(features):
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            nn.utils.clip_grad_norm_(repair.parameters(), 5.0)
                            optimizer.step()
                            updates += 1

                    stats.update(
                        candidates=len(features),
                        sources=record["candidates"]["source_count"],
                        frames=1,
                    )
                    _merge(totals, stats)
                    frames += 1
                    if frames == 1 or frames % 500 == 0:
                        print(
                            f"repair-cache epoch={epoch+1} {split} {weather}: "
                            f"{frames} accumulated frames, "
                            f"candidates={int(totals.get('candidates', 0))}",
                            flush=True,
                        )

            if not frames:
                raise RuntimeError("repair cache contains no frames")

            metrics = {
                key: value / frames
                for key, value in totals.items()
                if key not in (
                    "positive", "negative", "ignored", "clipped",
                    "candidates", "sources", "frames",
                )
            }
            metrics.update({
                "frames": frames,
                "updates": updates,
                "positive": int(totals.get("positive", 0)),
                "negative": int(totals.get("negative", 0)),
                "ignored": int(totals.get("ignored", 0)),
                "target_clipped": int(totals.get("clipped", 0)),
                "mean_candidates": totals.get("candidates", 0) / frames,
                "mean_sources": totals.get("sources", 0) / frames,
            })
            epoch_metrics[split] = metrics

        scheduler.step()
        score = float(epoch_metrics["validation"]["loss"])
        improved = score < best
        best = min(best, score)
        state = {
            "phase": "repair",
            "epoch": epoch,
            "best_validation_loss": best,
            "repair": repair.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rt.rng_state(),
            "feature_dim": feature_dim,
            "contract": current_contract,
            "cache_schema": cache_manifest["schema"],
            "metrics": epoch_metrics,
            "smoke": bool(args.smoke),
        }
        rt.atomic_torch_save(state, last_path)
        if improved:
            rt.atomic_torch_save(state, out / "repair_best.pth")

        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **epoch_metrics,
            }, ensure_ascii=False) + "\n")

        print(json.dumps({
            "epoch": epoch + 1,
            "best_validation_loss": best,
            **epoch_metrics,
        }, ensure_ascii=False), flush=True)

    print("DONE repair:", out / "repair_best.pth", flush=True)


if __name__ == "__main__":
    main()
