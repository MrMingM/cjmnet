"""Trace official candidate rows without recomputing the thresholded geometry."""
from contextlib import ExitStack

import numpy as np
import torch
from opencood.utils import box_utils
from evaluate import official_match_trace, statistics
from support import corner_array, patched


def trace(model, ego, processor, reference):
    outputs = model(ego)
    if 'dm' in outputs:
        raise ValueError('P0 unexpectedly has a direction head')
    scores = outputs['psm'].permute(0, 2, 3, 1).sigmoid().reshape(-1)
    decoded = processor.delta_to_boxes3d(outputs['rm'], ego['anchor_box'])[0]
    selected = torch.where(scores > processor.params['target_args']['score_threshold'])[0]
    corners = box_utils.project_box3d(box_utils.boxes_to_corners_3d(decoded, order=processor.params['order']),
                                      ego['transformation_matrix'])
    captured = {}

    def observe(name, take):
        original = getattr(box_utils, name)

        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            if name in captured:
                raise ValueError(f'Unexpected repeated native call: {name}')
            captured[name] = take(args, result)
            return result
        return patched(box_utils, name, call)

    result_only = lambda args, result: result
    with ExitStack() as stack:
        for name, take in (
            ('boxes_to_corners_3d', lambda args, result: args[0]),
            ('project_box3d', result_only),
            ('remove_large_pred_bbx', result_only),
            ('remove_bbx_abnormal_z', result_only),
            ('nms_rotated', lambda args, result: (args[0], args[1], result)),
            ('get_mask_for_boxes_within_range_torch', result_only),
        ):
            stack.enter_context(observe(name, take))
        final, final_scores = processor.post_process({'ego': ego}, {'ego': outputs})
    flags = np.zeros(len(decoded), dtype=np.uint8)
    if len(selected):
        native_boxes = captured['boxes_to_corners_3d']
        if not torch.equal(decoded[selected], native_boxes):
            raise ValueError('Native selected boxes do not match decoded anchor IDs')
        projected = captured['project_box3d']
        exact = captured['remove_large_pred_bbx'] & captured['remove_bbx_abnormal_z']
        nms_boxes, nms_scores, keep = captured['nms_rotated']
        indices = selected[exact]
        if not torch.equal(projected[exact], nms_boxes) or not torch.equal(scores[indices], nms_scores):
            raise ValueError('Native NMS input does not match traced anchors')
        if not torch.equal(corners[indices], nms_boxes):
            error = float((corners[indices] - nms_boxes).abs().max())
            print(f'NATIVE PROJECTION ROUNDING max_abs={error:.9g}', flush=True)
        # Batched corner projection may round differently when the number of boxes changes.
        # Store the values that actually entered native postprocessing.
        corners[selected] = projected
        flags[selected.cpu().numpy()] = 1
        flags[indices.cpu().numpy()] = 2
        indices = indices[keep]
        flags[indices.cpu().numpy()] = 3
        indices = indices[captured['get_mask_for_boxes_within_range_torch']]
        if not torch.equal(corners[indices], final) or not torch.equal(scores[indices], final_scores):
            raise ValueError('Native final boxes do not match traced anchors')
        flags[indices.cpu().numpy()] = 4
    elif captured or final is not None or final_scores is not None:
        raise ValueError('Native postprocessor disagrees with the empty score mask')
    pairs, _ = official_match_trace(final, final_scores, torch.from_numpy(corner_array(reference)),
                                    [r['object_id'] for r in reference], statistics(), .7)
    return (corners.cpu().numpy(), decoded.cpu().numpy(), None, scores.cpu().numpy(), flags), {
        str(p['object_id']) for p in pairs}
