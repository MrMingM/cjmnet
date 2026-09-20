"""Phase 1: freeze the original detector/fusion model and train only the repairer."""
import argparse
import json
from pathlib import Path
import shutil


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="local_detection_repair/experiment.yaml")
    p.add_argument("--frontend-config", required=True)
    p.add_argument("--frontend-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", type=int, default=0,
                   help="limit frames per weather and run one epoch")
    return p.parse_args()


def _merge(total, row):
    for key, value in row.items():
        total[key] = total.get(key, 0.0) + float(value)


def main():
    args = _args()
    import torch
    from torch import nn
    from opencood.tools.train_utils import to_device
    from . import runtime as rt
    from .model import LocalRepairNet
    from .pipeline import (WEATHERS, candidate_feature_dim, frozen_full_and_sources,
                           generate_candidates, make_training_targets, repair_loss)

    if args.smoke < 0:
        raise ValueError("--smoke cannot be negative")
    options, hypes = rt.load_config(args.config, args.frontend_config)
    rt.seed_all(options["seed"])
    target = rt.device()
    frontend, frontend_sha = rt.load_frontend(
        hypes, options, args.frontend_checkpoint, target
    )

    # Build only to read official train scene boundaries.
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

    feature_dim = candidate_feature_dim(options)
    repair = LocalRepairNet(feature_dim, options["repair"]).to(target)
    optimizer = torch.optim.AdamW(
        repair.parameters(), lr=float(options["learning_rate"]),
        weight_decay=float(options["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(options["lr_step"]),
        gamma=float(options["lr_gamma"])
    )
    rt.optimizer_only(optimizer, [repair])

    out = rt.safe_output(args.output_dir, resume=args.resume)
    if not args.resume:
        shutil.copy2(args.config, out / "experiment.yaml")
        rt.write_json(out / "training_protocol.json", {
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
            "checkpoint_selection":
                "mean loss over official OPV2V validation clean/fog/rain/snow online development weather",
        })

    begin, best = 0, float("inf")
    if args.resume:
        state = torch.load(out / "last.pth", map_location=target, weights_only=False)
        rt.ensure_contract(state["contract"], current_contract)
        if int(state["feature_dim"]) != feature_dim:
            raise ValueError("resume feature dimension changed")
        repair.load_state_dict(state["repair"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        begin = int(state["epoch"]) + 1
        best = float(state["best_validation_loss"])
        rt.restore_rng(state["rng"])

    epochs = 1 if args.smoke else int(options["repair_epochs"])
    history_path = out / "history.jsonl"
    for epoch in range(begin, epochs):
        epoch_metrics = {}
        for split in ("train", "validation"):
            training = split == "train"
            repair.train(training)
            frontend.eval()
            scenes = repair_scenes if training else None
            totals = {}
            frames = updates = 0
            for weather_index, weather in enumerate(WEATHERS):
                epoch_seed = (int(options["seed"]) + epoch * 1000 +
                              weather_index * 100 + (0 if training else 50000))
                rt.seed_all(epoch_seed)
                ds, loader, indices = rt.make_loader(
                    hypes, options, split, weather, scenes=scenes,
                    shuffle=training, seed=epoch_seed, smoke=args.smoke
                )
                for batch_index, batch in enumerate(loader):
                    batch = to_device(batch, target)
                    inp = rt.input_branch(batch, weather, benchmark=False)
                    with torch.no_grad():
                        full, sources = frozen_full_and_sources(
                            frontend, inp, options["candidate"]["max_sources"],
                            verify_full=(epoch == begin and split == "train" and
                                         weather_index == 0 and batch_index == 0),
                        )
                        candidates = generate_candidates(
                            ds.post_processor, batch["ego"]["anchor_box"],
                            full, sources, options,
                            hypes["model"]["args"]["lidar_range"]
                        )
                        _, _, gt = ds.post_process(batch, {"ego": full})
                        targets = make_training_targets(
                            ds, batch, candidates, gt, options
                        )
                    with torch.set_grad_enabled(training):
                        prediction = repair(candidates["features"])
                        loss, stats = repair_loss(
                            repair, prediction, targets, options
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite repair loss")
                        if training and len(candidates["candidate_ids"]):
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            nn.utils.clip_grad_norm_(repair.parameters(), 5.0)
                            rt.assert_frozen(frontend)
                            optimizer.step()
                            updates += 1
                    stats.update(
                        candidates=len(candidates["candidate_ids"]),
                        sources=candidates["source_count"],
                        frames=1,
                    )
                    _merge(totals, stats)
                    frames += 1
                    if frames == 1 or frames % 200 == 0:
                        print(
                            f"repair epoch={epoch+1} {split} {weather}: "
                            f"{frames} accumulated frames, candidates={int(totals.get('candidates', 0))}",
                            flush=True,
                        )
            if not frames:
                raise RuntimeError("no frames processed")
            metrics = {
                key: value / frames for key, value in totals.items()
                if key not in ("positive", "negative", "ignored", "clipped",
                               "candidates", "sources", "frames")
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
            rt.assert_frozen(frontend)

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
            "metrics": epoch_metrics,
            "smoke": bool(args.smoke),
        }
        rt.atomic_torch_save(state, out / "last.pth")
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

    rt.assert_frozen(frontend)
    print("DONE repair:", out / "repair_best.pth", flush=True)


if __name__ == "__main__":
    main()
