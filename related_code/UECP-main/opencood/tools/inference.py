# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.utils import eval_utils

torch.multiprocessing.set_sharing_strategy("file_system")


def test_parser():
    parser = argparse.ArgumentParser(description="UECP evaluation")
    parser.add_argument("--model_dir", type=str, required=True, help="checkpoint/run directory")
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="intermediate",
        choices=["intermediate", "no", "single"],
        help="evaluation mode",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="dataloader workers")
    return parser.parse_args()


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)

    if "test_dir" not in hypes:
        assert "validate_dir" in hypes, "Please specify test_dir or validate_dir."
        hypes["test_dir"] = hypes["validate_dir"]
    hypes["validate_dir"] = hypes["test_dir"]

    if "box_align" in hypes:
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    print("Creating Model")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Model from checkpoint")
    resume_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    print(f"resume from {resume_epoch} epoch.")

    if torch.cuda.is_available():
        model.to(device)
    model.eval()

    print("Dataset Building")
    opencood_dataset = build_dataset(hypes, visualize=False, train=False)
    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    infer_info = f"{opt.fusion_method}_epoch{resume_epoch}"
    for idx, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{idx}")
        if batch_data is None:
            continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == "intermediate":
                infer_result = inference_utils.inference_intermediate_fusion(
                    batch_data,
                    model,
                    opencood_dataset,
                )
            elif opt.fusion_method == "no":
                infer_result = inference_utils.inference_no_fusion(
                    batch_data,
                    model,
                    opencood_dataset,
                )
            else:
                infer_result = inference_utils.inference_no_fusion(
                    batch_data,
                    model,
                    opencood_dataset,
                    single_gt=True,
                )

            pred_box_tensor = infer_result["pred_box_tensor"]
            gt_box_tensor = infer_result["gt_box_tensor"]
            pred_score = infer_result["pred_score"]

            for iou in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    gt_box_tensor,
                    result_stat,
                    iou,
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eval_utils.eval_final_results(result_stat, opt.model_dir, infer_info)


if __name__ == "__main__":
    main()
