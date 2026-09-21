"""Post-NMS GT/FP bookkeeping used only while producing labels or metrics."""
import numpy as np


def detection_outcome(boxes, scores, gt, threshold=.7):
    """Return matched GT IDs and the actual unmatched detection boxes."""
    from opencood.utils import common_utils as cu
    gt_array = gt.detach().cpu().numpy()
    gt_polygons = list(cu.convert_format(gt_array))
    remaining = list(range(len(gt_polygons)))
    matched, fp_indices = set(), []
    if boxes is None:
        return matched, np.empty((0, 8, 3), dtype=np.float32)
    box_array = boxes.detach().cpu().numpy()
    polygons = list(cu.convert_format(box_array))
    order = np.argsort(-scores.detach().cpu().numpy(), kind='stable')
    for index in order:
        ious = cu.compute_iou(polygons[int(index)], gt_polygons)
        if not len(ious) or float(np.max(ious)) < float(threshold):
            fp_indices.append(int(index))
            continue
        best = int(np.argmax(ious))
        matched.add(remaining.pop(best))
        gt_polygons.pop(best)
    return matched, box_array[fp_indices]


def count_new_false_positives(action_fp_boxes, baseline_fp_boxes, identity_iou=.7):
    """Count action FPs with no geometrically equivalent baseline FP.

    This is intentionally different from max(action_fp_count-base_fp_count, 0):
    replacing one old FP with a new FP is still one newly introduced FP.
    """
    from opencood.utils import common_utils as cu
    if not len(action_fp_boxes):
        return 0
    if not len(baseline_fp_boxes):
        return int(len(action_fp_boxes))
    old = list(cu.convert_format(np.asarray(baseline_fp_boxes)))
    new = list(cu.convert_format(np.asarray(action_fp_boxes)))
    return sum(not len(ious) or float(np.max(ious)) < float(identity_iou)
               for ious in (cu.compute_iou(poly, old) for poly in new))


def compare_predictions(baseline, action, target_iou=.7, fp_identity_iou=.7):
    base_boxes, base_scores, gt = baseline
    action_boxes, action_scores, action_gt = action
    if tuple(gt.shape) != tuple(action_gt.shape) or not np.allclose(
            gt.detach().cpu().numpy(), action_gt.detach().cpu().numpy()):
        raise RuntimeError('GT changed between paired predictions')
    base_match, base_fp = detection_outcome(base_boxes, base_scores, gt, target_iou)
    action_match, action_fp = detection_outcome(action_boxes, action_scores, gt, target_iou)
    return {
        'recovered': len(action_match-base_match),
        'lost': len(base_match-action_match),
        'new_fp': count_new_false_positives(action_fp, base_fp, fp_identity_iou),
        'baseline_tp': len(base_match),
        'action_tp': len(action_match),
        'baseline_fp': len(base_fp),
        'action_fp': len(action_fp),
    }

