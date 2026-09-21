"""One-time frozen-detector cache for local repair training.

The expensive path (weather simulation, GSPR/PointPillar encoding, source heads,
decode/post-process and GT geometry matching) is executed once per cached frame.
Repairer epochs and selector training then reuse compact CPU tensors.
"""
from pathlib import Path
import time

import torch

from .pipeline import (
    WEATHERS,
    frozen_full_and_sources,
    generate_candidates,
    make_training_targets,
    prepare_postprocess,
    post_process_with_extras,
)

CACHE_SCHEMA = 2
CACHE_SPLITS = ("repair_train", "selector_train", "validation")


def _cpu(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu()
    raise TypeError("cache values must be tensors or None")


def _candidate_record(candidates):
    return {
        "candidate_ids": _cpu(candidates["candidate_ids"]),
        "features": _cpu(candidates["features"]).float(),
        "base_boxes": _cpu(candidates["base_boxes"]).float(),
        "base_scores": _cpu(candidates["base_scores"]).float(),
        "source_count": int(candidates["source_count"]),
    }


def _target_record(targets):
    return {
        "labels": _cpu(targets["labels"]).long(),
        "geometry": _cpu(targets["geometry"]).float(),
        "score": _cpu(targets["score"]).float(),
        "max_iou": _cpu(targets["max_iou"]).float(),
        "target_clipped": _cpu(targets["target_clipped"]).bool(),
    }


def _compact_prepared(ds, prepared):
    """Keep only dense proposals that can survive the original score filter.

    post_process_with_extras first applies this same threshold before geometry
    filtering/NMS, so discarding sub-threshold dense proposals is exact for
    selector-label replay and greatly reduces cache size.
    """
    threshold = float(ds.post_processor.params["target_args"]["score_threshold"])
    keep = prepared["dense_scores"] > threshold
    return {
        "dense_scores": _cpu(prepared["dense_scores"][keep]).float(),
        "dense_boxes": _cpu(prepared["dense_boxes"][keep]).float(),
        "reference_boxes": _cpu(prepared["reference_boxes"]),
        "reference_scores": _cpu(prepared["reference_scores"]),
        "gt": _cpu(prepared["gt"]).float(),
    }


def _tensor_dict_to_device(values, device):
    return {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in values.items()
    }


def candidates_to_device(record, device):
    return _tensor_dict_to_device(record["candidates"], device)


def targets_to_device(record, device):
    if "targets" not in record:
        raise KeyError("cached record has no repair targets")
    return _tensor_dict_to_device(record["targets"], device)


def prepared_to_device(record, device):
    if "prepared" not in record:
        raise KeyError("cached record has no selector replay state")
    return _tensor_dict_to_device(record["prepared"], device)


def cache_path(cache_root, split, weather):
    if split not in CACHE_SPLITS or weather not in WEATHERS:
        raise ValueError("invalid cache split/weather")
    return Path(cache_root) / split / (weather + ".pt")


def load_cache(cache_root, split, weather, contract=None):
    path = cache_path(cache_root, split, weather)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("schema", -1)) != CACHE_SCHEMA:
        raise ValueError("repair cache schema mismatch: " + str(path))
    if payload.get("split") != split or payload.get("weather") != weather:
        raise ValueError("repair cache identity mismatch: " + str(path))
    if contract is not None and payload.get("contract") != contract:
        raise ValueError("repair cache contract mismatch: " + str(path))
    return payload


def load_manifest(cache_root, contract=None):
    import json
    path = Path(cache_root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("repair cache manifest missing: " + str(path))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or int(manifest.get("schema", -1)) != CACHE_SCHEMA:
        raise ValueError("repair cache manifest is incomplete or incompatible")
    if contract is not None and manifest.get("contract") != contract:
        raise ValueError("repair cache manifest contract mismatch")
    return manifest


def _existing_valid(path, split, weather, contract):
    if not path.is_file():
        return None
    try:
        return load_cache(path.parent.parent, split, weather, contract=contract)
    except Exception:
        return None


@torch.no_grad()
def build_repair_cache(out, hypes, options, frontend, contract, target,
                       repair_scenes, selector_scenes, smoke=0):
    """Build resumable per-weather cache files.

    repair_train stores candidate features + repair GT targets.
    selector_train stores candidate features + compact exact baseline replay.
    validation stores both, so the same fixed development frames serve repair
    checkpoint selection and selector checkpoint selection without re-running
    weather simulation or the frozen detector.
    """
    from opencood.tools.train_utils import to_device
    from . import runtime as rt

    cache_root = Path(out) / "repair_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    for split in CACHE_SPLITS:
        (cache_root / split).mkdir(exist_ok=True)

    plans = (
        ("repair_train", "train", repair_scenes, True, False),
        ("selector_train", "train", selector_scenes, False, True),
        ("validation", "validation", None, True, True),
    )
    files = {}
    totals = {}
    verified_full = False
    verified_compact = False

    for plan_index, (cache_split, data_split, scenes,
                     need_targets, need_prepared) in enumerate(plans):
        files[cache_split] = {}
        totals[cache_split] = {}
        for weather_index, weather in enumerate(WEATHERS):
            path = cache_path(cache_root, cache_split, weather)
            existing = _existing_valid(
                path, cache_split, weather, contract
            )
            if existing is not None:
                files[cache_split][weather] = str(path)
                totals[cache_split][weather] = existing["stats"]
                print(
                    f"CACHE SKIP {cache_split}/{weather}: "
                    f"{existing['stats']['frames']} frames",
                    flush=True,
                )
                continue

            if path.exists():
                path.unlink()

            seed = (
                int(options["seed"]) + 200000
                + plan_index * 10000 + weather_index * 1000
            )
            rt.seed_all(seed)
            ds, loader, indices = rt.make_loader(
                hypes, options, data_split, weather,
                scenes=scenes, shuffle=False, seed=seed, smoke=smoke,
                workers=int(options.get("cache_workers", 0)),
            )
            records = []
            candidate_total = positive_total = negative_total = ignored_total = 0
            started = time.perf_counter()

            for frame_no, batch in enumerate(loader):
                sample_index = int(batch["ego"]["communication_sample_index"][0])
                batch = to_device(batch, target)
                inp = rt.input_branch(batch, weather, benchmark=False)

                full, sources = frozen_full_and_sources(
                    frontend, inp, options["candidate"]["max_sources"],
                    verify_full=not verified_full,
                )
                verified_full = True
                candidates = generate_candidates(
                    ds.post_processor, batch["ego"]["anchor_box"],
                    full, sources, options,
                    hypes["model"]["args"]["lidar_range"],
                )

                prepared = None
                if need_prepared:
                    prepared = prepare_postprocess(ds, batch, full)
                    gt = prepared["gt"]
                elif need_targets:
                    # Repair supervision only needs GT corners. Calling
                    # ds.post_process here would also decode/filter/NMS every
                    # baseline prediction even though those predictions are not
                    # used by phase-1 targets.
                    gt = ds.post_processor.generate_gt_bbx(batch)
                else:
                    gt = None

                record = {
                    "sample_index": sample_index,
                    "weather": weather,
                    "candidates": _candidate_record(candidates),
                    "transformation_matrix":
                        _cpu(batch["ego"]["transformation_matrix"]).float(),
                }

                if need_targets:
                    targets = make_training_targets(
                        ds, batch, candidates, gt, options
                    )
                    record["targets"] = _target_record(targets)
                    labels = targets["labels"]
                    positive_total += int((labels == 1).sum())
                    negative_total += int((labels == 0).sum())
                    ignored_total += int((labels < 0).sum())

                if need_prepared:
                    compact = _compact_prepared(ds, prepared)
                    record["prepared"] = compact
                    if not verified_compact:
                        replay = _tensor_dict_to_device(compact, target)
                        zero_boxes = candidates["base_boxes"].new_zeros((0, 7))
                        zero_scores = candidates["base_scores"].new_zeros((0,))
                        boxes, scores, replay_gt = post_process_with_extras(
                            ds, batch, None, zero_boxes, zero_scores,
                            prepared=replay,
                        )
                        torch.testing.assert_close(replay_gt, prepared["gt"])
                        if prepared["reference_boxes"] is None:
                            if boxes is not None or scores is not None:
                                raise AssertionError(
                                    "compact baseline replay changed empty semantics"
                                )
                        else:
                            torch.testing.assert_close(
                                boxes, prepared["reference_boxes"],
                                atol=1e-6, rtol=1e-6,
                            )
                            torch.testing.assert_close(
                                scores, prepared["reference_scores"],
                                atol=1e-7, rtol=1e-6,
                            )
                        verified_compact = True

                candidate_total += len(candidates["candidate_ids"])
                records.append(record)

                done = frame_no + 1
                if done == 1 or done % 50 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"CACHE {cache_split}/{weather}: {done}/{len(indices)} "
                        f"frames, {elapsed/done:.2f}s/frame, "
                        f"candidates={candidate_total}",
                        flush=True,
                    )

            stats = {
                "frames": len(records),
                "candidates": candidate_total,
                "mean_candidates": candidate_total / max(len(records), 1),
                "positive": positive_total,
                "negative": negative_total,
                "ignored": ignored_total,
                "elapsed_seconds": time.perf_counter() - started,
            }
            payload = {
                "schema": CACHE_SCHEMA,
                "split": cache_split,
                "weather": weather,
                "seed": seed,
                "contract": contract,
                "records": records,
                "stats": stats,
            }
            rt.atomic_torch_save(payload, path)
            files[cache_split][weather] = str(path)
            totals[cache_split][weather] = stats
            print(
                f"CACHE DONE {cache_split}/{weather}: "
                f"{stats['frames']} frames in {stats['elapsed_seconds']/3600:.2f}h",
                flush=True,
            )

    manifest = {
        "schema": CACHE_SCHEMA,
        "complete": True,
        "contract": contract,
        "files": files,
        "stats": totals,
        "definition": (
            "Frozen frontend/weather/decode/GT geometry are computed once. "
            "Repair epochs reuse per-frame candidate tensors. Selector reuses "
            "compact pre-threshold baseline replay state; no weather/frontend "
            "rerun is needed for selector-label generation."
        ),
    }
    rt.write_json(cache_root / "manifest.json", manifest)
    rt.assert_frozen(frontend)
    return manifest
