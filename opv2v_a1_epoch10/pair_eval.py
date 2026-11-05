"""Compare three modes on each identical OPV2V test batch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import SOURCE, runtime

THRESHOLDS = (.3, .5, .7)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", required=True)
    parser.add_argument("--gspr", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    from attfuse_gspr import point_pillar_gspr_attfuse
    original_class = point_pillar_gspr_attfuse.PointPillarGsprAttfuse
    runtime.install()
    from evaluate import CONDITIONS
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils

    config = load_yaml(args.config)
    config["validate_dir"] = config["validate_dir"] if args.smoke else CONDITIONS[args.weather]
    dataset = build_dataset(config, visualize=False, train=False)
    random.seed(20260906)
    np.random.seed(20260906)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                        generator=torch.Generator().manual_seed(20260906),
                        collate_fn=dataset.collate_batch_test)
    a1 = train_utils.create_model(config)
    gspr = original_class(config["model"]["args"])
    hashes = {name:digest(path) for name,path in (("a1",args.a1),("gspr",args.gspr))}
    for model,path in ((a1,args.a1),(gspr,args.gspr)):
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        model.cuda().eval()
    if not args.smoke and len(dataset) != 2170:
        raise ValueError(f"Expected 2170 test frames, got {len(dataset)}")
    stats = {mode:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in THRESHOLDS}
             for mode in ("a1_on","a1_off","gspr_only")}
    seen = 0
    start = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            data = train_utils.to_device(batch, "cuda:0")
            reference = None
            for mode,model in (("a1_on",a1),("a1_off",a1),("gspr_only",gspr)):
                a1.enabled = mode == "a1_on"
                boxes,scores,gt = inference_utils.inference_intermediate_fusion(data,model,dataset)
                if reference is None:
                    reference = gt.clone()
                elif not torch.equal(reference,gt):
                    raise ValueError("Paired GT mismatch")
                if boxes is not None and (not torch.isfinite(boxes).all() or not torch.isfinite(scores).all()):
                    raise ValueError("Nonfinite prediction")
                for t in THRESHOLDS:
                    eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[mode],t)
            seen += 1
            if seen % 50 == 0:
                print("PAIRED",args.weather,seen,len(dataset),round(time.monotonic()-start),flush=True)
            if args.smoke and seen == 3:
                break
    expected = 3 if args.smoke else len(dataset)
    if seen != expected or hashes != {"a1":digest(args.a1),"gspr":digest(args.gspr)}:
        raise RuntimeError("Incomplete eval or changed checkpoint")
    result = dict(frames=seen, dataset=config["validate_dir"], hashes=hashes,
                  global_sort_detections=False, smoke=args.smoke)
    if not args.smoke:
        ap = {mode:float(eval_utils.calculate_ap(copy.deepcopy(stat),.7,False)[0])
              for mode,stat in stats.items()}
        result.update(ap70=ap, switch_gain_pp=100*(ap["a1_on"]-ap["a1_off"]),
                      trained_gain_pp=100*(ap["a1_on"]-ap["gspr_only"]))
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    temporary.replace(args.output)
    print("COMPLETE",json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
