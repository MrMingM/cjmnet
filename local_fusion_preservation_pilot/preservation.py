"""Training-only teacher mask; GT never enters the deployed v3 module."""
import numpy as np
import torch
import torch.nn.functional as F


def matched_teacher_pairs(boxes, scores, gt, threshold=0.7):
    """Return (prediction, GT) indices using the evaluation's greedy matching."""
    from opencood.utils import common_utils as cu

    if boxes is None or not len(boxes) or not len(gt):
        return []
    pred = list(cu.convert_format(boxes.detach().cpu().numpy()))
    truth = list(cu.convert_format(gt.detach().cpu().numpy()))
    remaining = list(range(len(truth)))
    order = np.argsort(-scores.detach().cpu().numpy(), kind='stable')
    matches = []
    for index in order:
        ious = cu.compute_iou(pred[int(index)], truth)
        if len(ious) and float(np.max(ious)) >= threshold:
            best = int(np.argmax(ious))
            matches.append((int(index), remaining.pop(best)))
            truth.pop(best)
    return matches


@torch.no_grad()
def correct_anchor_mask(dataset, batch, teacher, target_iou=0.7,
                        identity_iou=0.99):
    """Protect only positive anchors that produced a GT-matched teacher TP.

    A final TP can originate at an anchor not marked positive by the detector's
    training labels. Such TPs are counted as uncovered, not guessed into a mask.
    """
    from opencood.utils import box_utils, common_utils as cu

    labels = batch['ego']['label_dict']['pos_equal_one']
    if labels.shape[0] != 1 or labels.ndim != 4:
        raise ValueError('Pilot requires one frame and [B,H,W,A] labels')
    mask = torch.zeros_like(labels, dtype=torch.bool)
    boxes, scores, gt = dataset.post_process(batch, {'ego': teacher})
    matches = matched_teacher_pairs(boxes, scores, gt, target_iou)
    if not matches or not bool(labels.bool().any()):
        return mask, dict(teacher_tp=len(matches), protected=0)

    anchors = batch['ego']['anchor_box']
    decoded = dataset.post_processor.delta_to_boxes3d(teacher['rm'], anchors)[0]
    if len(decoded) != labels[0].numel():
        raise RuntimeError('Decoded anchor order differs from training labels')
    eligible = labels[0].reshape(-1).bool() & torch.isfinite(decoded).all(-1)
    positive_flat = eligible.nonzero(as_tuple=False).flatten()
    if not len(positive_flat):
        return mask, dict(teacher_tp=len(matches), protected=0)
    positive_corners = box_utils.boxes_to_corners_3d(
        decoded[positive_flat], order=dataset.post_processor.params['order'])
    positive_polygons = list(cu.convert_format(positive_corners.detach().cpu().numpy()))
    pred_polygons = list(cu.convert_format(boxes.detach().cpu().numpy()))
    gt_polygons = list(cu.convert_format(gt.detach().cpu().numpy()))
    chosen = set()
    for pred_index, gt_index in matches:
        same_pred = cu.compute_iou(pred_polygons[pred_index], positive_polygons)
        if not len(same_pred):
            continue
        order = np.argsort(-same_pred)
        for candidate in order:
            candidate = int(candidate)
            if float(same_pred[candidate]) < identity_iou:
                break
            if candidate in chosen:
                continue
            same_gt = cu.compute_iou(positive_polygons[candidate], [gt_polygons[gt_index]])
            if len(same_gt) and float(same_gt[0]) >= target_iou:
                chosen.add(candidate)
                break
    if chosen:
        mask[0].reshape(-1)[positive_flat[sorted(chosen)]] = True
    return mask, dict(teacher_tp=len(matches), protected=len(chosen))


def preservation_loss(student, teacher, anchor_mask):
    """Distill class logit and seven box deltas at verified teacher anchors."""
    psm, rm = student['psm'], student['rm']
    if anchor_mask.shape != (psm.shape[0], psm.shape[2], psm.shape[3], psm.shape[1]):
        raise ValueError('Mask and detector anchor dimensions differ')
    selected = anchor_mask.bool()
    if not bool(selected.any()):
        return (psm.sum() + rm.sum()) * 0.
    student_cls = psm.permute(0, 2, 3, 1)[selected]
    teacher_cls = teacher['psm'].permute(0, 2, 3, 1)[selected].detach()
    box_shape = (*selected.shape, 7)
    student_box = rm.permute(0, 2, 3, 1).reshape(box_shape)[selected]
    teacher_box = teacher['rm'].permute(0, 2, 3, 1).reshape(box_shape)[selected].detach()
    return (F.smooth_l1_loss(student_cls, teacher_cls) +
            F.smooth_l1_loss(student_box, teacher_box))
