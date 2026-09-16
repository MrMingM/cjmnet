"""Evaluate one explicitly frozen GSPR-AttFuse checkpoint on four conditions."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


CONDITIONS = {
    "clean": "/data/scd/datasets/opv2v_official_data_dumping/test",
    "fog": "/data/cjm/datasets/opv2v-w/fog/test",
    "rain": "/data/cjm/datasets/opv2v-w/rain/test",
    "snow": "/data/cjm/datasets/opv2v-w/snow/test",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS),
                        default=tuple(CONDITIONS))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def configure_imports():
    here = os.path.abspath(os.path.dirname(__file__))
    root = here if os.path.isdir(os.path.join(here, "opencood")) else \
        os.path.abspath(os.path.join(here, "..", "OpenCOOD-main"))
    sys.path[:0] = [root, here]
    import attfuse_gspr.point_pillar_gspr_attfuse as local_model
    sys.modules["opencood.models.point_pillar_gspr_attfuse"] = local_model


def enforce_output(path):
    output = Path(path).resolve()
    allowed = Path("/data/cjm/datasets").resolve()
    if allowed not in (output, *output.parents):
        raise ValueError("--output-root must be inside /data/cjm/datasets")
    output.mkdir(parents=True, exist_ok=True)
    return output


def empty_result_stat():
    return {threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
            for threshold in (0.3, 0.5, 0.7)}


def main():
    args = parse_args()
    configure_imports()
    from opencood.data_utils.datasets import build_dataset
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils

    config = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_root = enforce_output(args.output_root)
    shutil.copy2(config, output_root / "config.yaml")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    hypes = yaml_utils.load_yaml(str(config))
    model = train_utils.create_model(hypes)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint = checkpoint.get("state_dict", checkpoint) \
        if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(checkpoint, strict=True)
    model.to(device).eval()

    summary = {}
    for condition in args.conditions:
        condition_dir = output_root / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        condition_hypes = dict(hypes)
        condition_hypes["validate_dir"] = CONDITIONS[condition]
        dataset = build_dataset(condition_hypes, visualize=False, train=False)
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_batch_test)
        result_stat = empty_result_stat()
        with torch.inference_mode():
            for batch in tqdm(loader, desc="evaluate %s" % condition):
                batch = train_utils.to_device(batch, device)
                pred_box, pred_score, gt_box = \
                    inference_utils.inference_intermediate_fusion(
                        batch, model, dataset)
                for threshold in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(
                        pred_box, pred_score, gt_box,
                        result_stat, threshold)
        eval_utils.eval_final_results(result_stat, str(condition_dir), False)
        with (condition_dir / "eval.yaml").open("r", encoding="utf-8") as stream:
            result = yaml.safe_load(stream)
        summary[condition] = {
            "dataset": CONDITIONS[condition],
            "samples": len(dataset),
            "ap30": float(result["ap30"]),
            "ap_50": float(result["ap_50"]),
            "ap_70": float(result["ap_70"]),
        }
        print(json.dumps({condition: summary[condition]}, ensure_ascii=False))

    report = {
        "selection_source": "clean validation loss plus point validation",
        "checkpoint": str(checkpoint_path),
        "config": str(config),
        "results": summary,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
