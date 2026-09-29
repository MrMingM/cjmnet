# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from collections import OrderedDict


def inference_no_fusion(batch_data, model, dataset, single_gt=False):
    """Run ego-only inference."""
    output_dict_ego = OrderedDict()
    if single_gt:
        batch_data = {"ego": batch_data["ego"]}

    output_dict_ego["ego"] = model(batch_data["ego"])
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process_no_fusion(
        batch_data,
        output_dict_ego,
    )

    return {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "gt_box_tensor": gt_box_tensor,
    }


def inference_early_fusion(batch_data, model, dataset):
    """Run the model on the already-collated ego batch."""
    output_dict = OrderedDict()
    output_dict["ego"] = model(batch_data["ego"])
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

    return {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "gt_box_tensor": gt_box_tensor,
    }


def inference_intermediate_fusion(batch_data, model, dataset):
    return inference_early_fusion(batch_data, model, dataset)
