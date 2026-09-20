"""One-frame-per-weather server smoke test; no long training and no test split."""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="local_detection_repair/experiment.yaml")
    p.add_argument("--frontend-config", required=True)
    p.add_argument("--frontend-checkpoint", required=True)
    args = p.parse_args()

    import torch
    from opencood.tools.train_utils import to_device
    from . import runtime as rt
    from .model import LocalRepairNet
    from .pipeline import (
        WEATHERS, assert_disabled_matches_baseline, candidate_feature_dim,
        frozen_full_and_sources, generate_candidates, make_training_targets,
        post_process_with_extras, repair_loss, repaired_outputs,
    )

    options, hypes = rt.load_config(args.config, args.frontend_config)
    rt.seed_all(options["seed"])
    target = rt.device()
    frontend, _ = rt.load_frontend(
        hypes, options, args.frontend_checkpoint, target
    )
    repair = LocalRepairNet(
        candidate_feature_dim(options), options["repair"]
    ).to(target)
    for weather_index, weather in enumerate(WEATHERS):
        run_seed = int(options["seed"]) + 300000 + weather_index
        rt.seed_all(run_seed)
        ds, loader, _ = rt.make_loader(
            hypes, options, "validation", weather,
            shuffle=False, seed=run_seed, smoke=1
        )
        batch = next(iter(loader))
        batch = to_device(batch, target)
        inp = rt.input_branch(batch, weather)
        with torch.no_grad():
            full, sources = frozen_full_and_sources(
                frontend, inp, options["candidate"]["max_sources"],
                verify_full=True
            )
            candidates = generate_candidates(
                ds.post_processor, batch["ego"]["anchor_box"],
                full, sources, options,
                hypes["model"]["args"]["lidar_range"]
            )
            assert_disabled_matches_baseline(ds, batch, full)
            _, _, gt = ds.post_process(batch, {"ego": full})
            targets = make_training_targets(ds, batch, candidates, gt, options)
        repair.train()
        output = repair(candidates["features"])
        loss, stats = repair_loss(repair, output, targets, options)
        repair.zero_grad(set_to_none=True)
        loss.backward()
        rt.assert_frozen(frontend)
        if len(candidates["candidate_ids"]) and not any(
            p.grad is not None for p in repair.parameters()
        ):
            raise AssertionError("repairer received no gradients")
        repair.eval()
        with torch.no_grad():
            boxes, scores, _, _ = repaired_outputs(
                repair, candidates, output, options, "joint"
            )
            repaired_boxes, repaired_scores, repaired_gt = post_process_with_extras(
                ds, batch, full, boxes, scores
            )
            torch.testing.assert_close(gt, repaired_gt)
            if repaired_scores is not None and not torch.isfinite(repaired_scores).all():
                raise FloatingPointError("non-finite smoke output")
        print(
            f"SMOKE {weather}: sources={len(sources)} "
            f"candidates={len(candidates['candidate_ids'])} "
            f"pos={stats['positive']} neg={stats['negative']} "
            f"output={0 if repaired_boxes is None else len(repaired_boxes)}",
            flush=True,
        )
    print("SMOKE PASS: train/validation code path only; no effectiveness claim", flush=True)


if __name__ == "__main__":
    main()
