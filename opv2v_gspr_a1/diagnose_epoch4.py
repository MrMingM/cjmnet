"""Same-input, fixed-checkpoint A1 ablation for OPV2V Clean validation."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path("/data/cjm/datasets/logs/opv2v-gspr-a1-v2")
PROJECT = Path("/home/cjm/OpenCOOD-main/cjmnet")
SOURCE = PROJECT / "opv2v_gspr_a1_v2"
JOINT = ROOT / "20260906/joint"
CONFIG = JOINT / "config.yaml"
THRESHOLDS = (.3, .5, .7)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, row):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Residual:
    def __init__(self):
        self.total = None

    def __call__(self, module, args, result):
        base = args[1].float()
        delta = result.float() - base
        parts = torch.stack((base.square().sum(), delta.square().sum(),
                             delta.new_tensor(delta.numel()),
                             (delta.abs().amax(dim=-1) > 0).sum().float(),
                             delta.new_tensor(delta.shape[0])))
        self.total = parts.double() if self.total is None else self.total + parts.double()

    def report(self):
        base, delta, elements, changed, pillars = self.total.tolist()
        return dict(base_rms=(base/elements)**.5, residual_rms=(delta/elements)**.5,
                    residual_to_base_rms=(delta/base)**.5, changed_pillars=int(changed),
                    pillars=int(pillars))


def main(output):
    if not torch.version.hip or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one ROCm device")
    sys.path.insert(0, str(SOURCE))
    from runtime import install
    install()
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils
    sys.path.insert(0, str(PROJECT))
    from train_gspr_multitask import configure_imports
    configure_imports()

    history = json.loads((JOINT/"history.json").read_text())
    if [row["epoch"] for row in history] != list(range(1, 6)):
        raise ValueError("Expected five completed joint epochs")
    input_audit = json.loads((ROOT/"inputs.json").read_text())
    historical_config = PROJECT/"gspr_attfuse_config.yaml"
    if sha(CONFIG) != input_audit["sources"][str(historical_config)]:
        raise ValueError("Joint config differs from registered input")
    hypes = load_yaml(str(CONFIG))
    dataset = build_dataset(hypes, visualize=False, train=False)
    if str(Path(hypes["validate_dir"])) != input_audit["datasets"]["validation"]["path"] or len(dataset) != 1980:
        raise ValueError("Unexpected Clean validation input")
    model = train_utils.create_model(hypes).cuda().eval()
    if not hasattr(model, "a1"):
        raise ValueError("A1 model factory not active")

    def loader(data):
        random.seed(20260906)
        np.random.seed(20260906)
        return DataLoader(data, batch_size=1, shuffle=False, num_workers=4,
                          generator=torch.Generator().manual_seed(20260906),
                          collate_fn=dataset.collate_batch_test)

    checkpoints = {epoch: JOINT/f"net_epoch{epoch}.pth" for epoch in range(1, 6)}
    hashes = {str(epoch):sha(path) for epoch,path in checkpoints.items()}
    sample_ids = np.linspace(0, len(dataset)-1, 64, dtype=int).tolist()
    probes = []
    for epoch, path in checkpoints.items():
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        model.enabled = True
        residual = Residual()
        handle = model.a1.register_forward_hook(residual)
        with torch.inference_mode():
            for batch in loader(Subset(dataset, sample_ids)):
                model(train_utils.to_device(batch, "cuda:0")["ego"])
        handle.remove()
        probes.append(dict(epoch=epoch, **residual.report()))
        print("RESIDUAL", json.dumps(probes[-1]), flush=True)
    write(output/"TRAINING_AND_RESIDUALS.json",dict(history=history, probes=probes,
          sample_indices=sample_ids, checkpoint_sha256=hashes))

    model.load_state_dict(torch.load(checkpoints[4], map_location="cpu", weights_only=True), strict=True)
    stats = {key:{t:{"tp":[],"fp":[],"gt":0,"score":[]} for t in THRESHOLDS}
             for key in ("enabled","disabled")}
    residual = Residual()
    handle = model.a1.register_forward_hook(residual)
    count = 0
    start = time.monotonic()
    with torch.inference_mode():
        for batch in loader(dataset):
            batch = train_utils.to_device(batch, "cuda:0")
            targets = {}
            for mode in stats:
                model.enabled = mode == "enabled"
                # Model and historical postprocessor see the same voxel input.
                boxes, scores, gt = inference_utils.inference_intermediate_fusion(
                    batch, model, dataset)
                targets[mode] = gt
                for threshold in THRESHOLDS:
                    eval_utils.caluclate_tp_fp(boxes, scores, gt, stats[mode], threshold)
            if not torch.equal(targets["enabled"], targets["disabled"]):
                raise ValueError("Paired GT mismatch")
            count += 1
            if count % 50 == 0:
                print("PAIRED", count, len(dataset), "seconds", round(time.monotonic()-start), flush=True)
    handle.remove()
    if count != len(dataset) or sha(checkpoints[4]) != hashes["4"]:
        raise ValueError("Incomplete evaluation or checkpoint changed")
    ap = {mode:{str(t):float(eval_utils.calculate_ap(copy.deepcopy(stat), t, False)[0])
                for t in THRESHOLDS} for mode,stat in stats.items()}
    write(output/"PAIRED_AP.json", dict(seed=20260906, epoch=4,
          checkpoint=str(checkpoints[4]), checkpoint_sha256=hashes["4"], frames=count,
          split=hypes["validate_dir"], global_sort_detections=False, ap=ap,
          enabled_minus_disabled_pp={str(t):100*(ap["enabled"][str(t)]-ap["disabled"][str(t)])
                                     for t in THRESHOLDS},
          residual_full=residual.report(),
          counts={mode:{str(t):dict(tp=sum(stat[t]["tp"]), fp=sum(stat[t]["fp"]), gt=stat[t]["gt"])
                        for t in THRESHOLDS} for mode,stat in stats.items()}))
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.output)
