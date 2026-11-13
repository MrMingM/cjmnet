"""Run the official postprocessor on frozen decoded candidates."""
from collections import Counter
from copy import deepcopy

import numpy as np
import torch


def processor(config):
    from opencood.data_utils.post_processor import build_postprocessor
    from opencood.hypes_yaml.yaml_utils import load_yaml

    return build_postprocessor(load_yaml(str(config))['postprocess'], dataset='opv2v', train=False)


def replay(pp, decoded, logits, transform, threshold, trace=False):
    from metrics import valid_nms_candidates
    from opencood.utils import box_utils
    from support import patched

    logits = np.asarray(logits, dtype=np.float32)
    boxes = torch.from_numpy(decoded)
    out = {'ego': dict(psm=torch.from_numpy(logits.reshape(1, 1, 1, -1)), rm=None)}
    data = {'ego': dict(anchor_box=None, transformation_matrix=torch.from_numpy(transform))}
    params = deepcopy(pp.params)
    params['target_args']['score_threshold'] = float(threshold)
    stages = np.zeros(len(logits), dtype=np.uint8)
    above = torch.from_numpy(logits).sigmoid().numpy() > threshold
    stages[above] = 1
    source_ids = np.flatnonzero(above)

    def large(boxes, dataset):
        nonlocal large_mask
        large_mask = original_large(boxes, dataset)
        return large_mask

    def abnormal(boxes):
        nonlocal source_ids
        valid = original_abnormal(boxes)
        source_ids = source_ids[(large_mask & valid).numpy()]
        stages[source_ids] = 2
        return valid

    def within(boxes, limits):
        nonlocal source_ids
        mask = original_within(boxes, limits)
        source_ids = source_ids[mask.numpy()]
        stages[source_ids] = 4
        return mask

    original_large = box_utils.remove_large_pred_bbx
    original_abnormal = box_utils.remove_bbx_abnormal_z
    original_within = box_utils.get_mask_for_boxes_within_range_torch
    large_mask = None
    from contextlib import ExitStack
    with ExitStack() as stack:
        if trace:
            for name, func in [('remove_large_pred_bbx', large), ('remove_bbx_abnormal_z', abnormal),
                               ('get_mask_for_boxes_within_range_torch', within)]: 
                stack.enter_context(patched(box_utils, name, func))
            # For tracing, prevalidated geometry is handled below with the same guard.
            # The guard can drop candidates prior to the inner NMS and maps its return.
            def guarded_nms(boxes, scores, threshold):
                nonlocal source_ids
                keep_ids = source_ids
                idx = official_guard(boxes, scores, threshold)
                source_ids = keep_ids[idx]
                stages[keep_ids] = 2
                stages[source_ids] = 3
                return idx
            # The guard wraps the actual native NMS, preserving original input indices.
            stack.enter_context(valid_nms_candidates(Counter()))
            official_guard = box_utils.nms_rotated
            stack.enter_context(patched(box_utils, 'nms_rotated', guarded_nms))
        else:
            stack.enter_context(valid_nms_candidates(Counter()))
        stack.enter_context(patched(pp, 'params', params))
        stack.enter_context(patched(pp, 'delta_to_boxes3d', lambda *_: boxes.unsqueeze(0)))
        pred, scores = pp.post_process(data, out)
    if not trace:
        return pred, scores
    if pred is not None and len(source_ids) != len(pred):
        raise ValueError('Native stage IDs do not align')
    return pred, scores, stages, source_ids
