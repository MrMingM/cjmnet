"""Read-only replica of classic VoxelPostprocessor, with stable anchor IDs.

Original code: opencood/data_utils/post_processor/voxel_postprocessor.py and
opencood/utils/box_utils.py. Uses original decoding, geometry and range helpers.
Never patches them. Every traced branch is checked against ds.post_process.
"""
import numpy as np
from .stage3_analysis import STAGES, greedy_assignment, target_summary, traced_nms


def polygon_ious(boxes, gt):
    """Exact original planar polygon IoU; disjoint AABBs safely shortcut to 0."""
    from opencood.utils import common_utils as cu
    result = np.zeros((len(boxes), len(gt)), dtype=np.float32)
    if not len(boxes) or not len(gt):
        return result
    lo, hi = boxes[:, :4, :2].min(1), boxes[:, :4, :2].max(1)
    glo, ghi = gt[:, :4, :2].min(1), gt[:, :4, :2].max(1)
    gp = cu.convert_format(gt)
    for j in range(len(gt)):
        ids = np.flatnonzero(((hi > glo[j]) & (lo < ghi[j])).all(1))
        if len(ids):
            pp = cu.convert_format(boxes[ids])
            result[ids, j] = [cu.compute_iou(p, [gp[j]])[0] for p in pp]
    if not np.isfinite(result).all():
        raise ValueError('Non-finite GT IoU; refuse an approximate report')
    return result


def assert_final_equal(boxes, scores, reference_boxes, reference_scores):
    import torch
    if reference_boxes is None:
        if boxes is not None or scores is not None:
            raise AssertionError('Tracer None/empty semantics differ from original')
        return
    if boxes is None or scores is None:
        raise AssertionError('Tracer unexpectedly returned None')
    if boxes.shape != reference_boxes.shape or scores.shape != reference_scores.shape:
        raise AssertionError('Tracer final counts differ from original')
    torch.testing.assert_close(boxes, reference_boxes, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(scores, reference_scores, atol=1e-7, rtol=1e-6)


def trace_branch(ds, batch, prediction):
    import torch
    from opencood.utils import box_utils as bu, common_utils as cu
    from qa_evidence_validity.peer_audit import _greedy_target_details
    pp = ds.post_processor
    if type(pp).__name__ != 'VoxelPostprocessor' or set(batch) != {'ego'}:
        raise ValueError('Tracer supports only classic single-output VoxelPostprocessor')
    if prediction['psm'].shape[0] != 1:
        raise ValueError('Only one frame per batch is supported')
    ego = batch['ego']
    logits = prediction['psm'].permute(0, 2, 3, 1).reshape(-1)
    scores = torch.nn.functional.sigmoid(prediction['psm'].permute(0, 2, 3, 1)).reshape(-1)
    decoded = pp.delta_to_boxes3d(prediction['rm'], ego['anchor_box'])[0]
    if not torch.isfinite(decoded).all() or not torch.isfinite(scores).all():
        raise ValueError('Non-finite decoded proposals; no silent filtering permitted')
    corners = bu.project_box3d(bu.boxes_to_corners_3d(decoded, order=pp.params['order']),
                               ego['transformation_matrix'])
    all_ids = torch.arange(len(scores), device=scores.device)
    score_ids = all_ids[scores > pp.params['target_args']['score_threshold']]
    # Operations on the SAME score-selected boxes as the original implementation.
    stage = dict(decoded=all_ids, score=score_ids)
    if len(score_ids):
        # Original projects only score-selected boxes. Match its operation sizes
        # too, avoiding size-dependent floating-point differences in matmul.
        selected_boxes = bu.project_box3d(
            bu.boxes_to_corners_3d(decoded[score_ids], order=pp.params['order']),
            ego['transformation_matrix'])
        corners[score_ids] = selected_boxes
        keep = torch.logical_and(bu.remove_large_pred_bbx(selected_boxes),
                                 bu.remove_bbx_abnormal_z(selected_boxes))
        geometry_ids = score_ids[keep]
    else:
        geometry_ids = score_ids
    stage['geometry'] = geometry_ids
    geom_boxes = corners[geometry_ids]
    polygons = cu.convert_format(geom_boxes.detach().cpu().numpy())
    pick, top, suppression = traced_nms(scores[geometry_ids].detach().cpu().numpy(),
        lambda i, js: cu.compute_iou(polygons[i], polygons[js]), pp.params['nms_thresh'])
    original_keep = bu.nms_rotated(geom_boxes, scores[geometry_ids], pp.params['nms_thresh'])
    if not np.array_equal(pick, original_keep):
        raise AssertionError('NMS trace diverged from original keep order')
    stage['nms_topk'] = geometry_ids[torch.as_tensor(top, device=scores.device)]
    nms_ids = geometry_ids[torch.as_tensor(pick, device=scores.device)]
    stage['nms'] = nms_ids
    final_ids = nms_ids[bu.get_mask_for_boxes_within_range_torch(corners[nms_ids])]
    stage['range'] = final_ids
    final_boxes = corners[final_ids] if len(score_ids) else None
    final_scores = scores[final_ids] if len(score_ids) else None
    reference_boxes, reference_scores, gt = ds.post_process(batch, {'ego': prediction})
    assert_final_equal(final_boxes, final_scores, reference_boxes, reference_scores)
    cn = corners.detach().cpu().numpy()
    sn = scores.detach().cpu().numpy()
    gn = gt.detach().cpu().numpy()
    ids = {name: tensor.detach().cpu().numpy() for name, tensor in stage.items()}
    ious = polygon_ious(cn, gn)
    assignment = greedy_assignment(sn[ids['range']], ious[ids['range']])
    reference_details, fp = _greedy_target_details(reference_boxes, reference_scores, gt)
    for j, d in enumerate(reference_details):
        if bool((assignment == j).any()) != d['matched']:
            raise AssertionError('Tracer final GT assignment differs from Stage-2 matching')
        if d['matched']:
            cid = ids['range'][np.flatnonzero(assignment == j)[0]]
            if not (np.isclose(sn[cid], d['matched_score'], atol=1e-7, rtol=1e-6) and
                    np.isclose(ious[cid, j], d['matched_iou'], atol=1e-7, rtol=1e-6)):
                raise AssertionError('Tracer matched proposal differs from Stage-2')
    if int((assignment == -1).sum()) != fp:
        raise AssertionError('Tracer FP count differs from Stage-2')
    geo_np = ids['geometry']
    suppressors = {int(geo_np[k]): (int(geo_np[v[0]]), v[1]) for k, v in suppression.items()}
    return dict(scores=sn, logits=logits.detach().cpu().numpy(),
                regression_deltas=prediction['rm'].permute(0, 2, 3, 1).reshape(-1, 7).detach().cpu().numpy(),
                decoded=decoded.detach().cpu().numpy(), corners=cn, gt=gn, ious=ious,
                ids=ids, suppressors=suppressors, assignment=assignment,
                details=reference_details, frame_fp=fp,
                anchor_shape=list(ego['anchor_box'].shape), box_order=pp.params['order'])


def proposal(trace, cid, target):
    if cid is None:
        return None
    cid = int(cid)
    shape = trace['anchor_shape'][:-1]
    return dict(candidate_id=cid, anchor_index=list(map(int, np.unravel_index(cid, shape))),
                score=float(trace['scores'][cid]), gt_iou=float(trace['ious'][cid, target]),
                decoded_box=trace['decoded'][cid].tolist(),
                ego_corners=trace['corners'][cid].tolist())


def describe_target(trace, target):
    out = target_summary(trace['scores'], trace['ious'][:, target], trace['ids'],
                         trace['assignment'], target)
    best = out['stages']['decoded']['best_iou_candidate_id']
    out['best_decoded_proposal'] = proposal(trace, best, target)
    out['matched_proposal'] = proposal(trace, out['matched_candidate_id'], target)
    suppressions = []
    for cid, (other, overlap) in trace['suppressors'].items():
        if float(trace['ious'][cid, target]) >= .7:
            suppressions.append(dict(suppressed=proposal(trace, cid, target),
                suppressor=proposal(trace, other, target), pair_iou=overlap,
                suppressor_best_gt=int(np.argmax(trace['ious'][other])) if len(trace['gt']) else None,
                suppressor_best_gt_iou=float(trace['ious'][other].max()) if len(trace['gt']) else None))
    out['qualifying_nms_suppressions'] = suppressions
    top_set = set(trace['ids']['nms_topk'].tolist())
    out['qualifying_topk_removed'] = [proposal(trace, cid, target)
        for cid in trace['ids']['geometry'] if cid not in top_set and float(trace['ious'][cid, target]) >= .7]
    return out


def geometry_comparison(peer, full, target):
    """Both same-anchor comparison and descriptive best-vs-best (not correspondence)."""
    pcid = peer['ids']['range'][np.flatnonzero(peer['assignment'] == target)[0]]
    fcid = int(np.argmax(full['ious'][:, target]))
    def compare(a, b):
        delta = b - a
        return dict(center_displacement=float(np.linalg.norm(delta[:3])),
                    center_delta=delta[:3].tolist(), size_delta_in_box_order=delta[3:6].tolist(),
                    yaw_delta_wrapped=float(np.arctan2(np.sin(delta[6]), np.cos(delta[6]))))
    return dict(peer_matched=proposal(peer, pcid, target),
                full_same_anchor=proposal(full, pcid, target),
                same_anchor_change=dict(compare(peer['decoded'][pcid], full['decoded'][pcid]),
                    iou_delta=float(full['ious'][pcid, target]-peer['ious'][pcid, target]),
                    score_delta=float(full['scores'][pcid]-peer['scores'][pcid])),
                full_best_iou=proposal(full, fcid, target),
                best_vs_best_change=compare(peer['decoded'][pcid], full['decoded'][fcid]),
                box_order=peer['box_order'],
                note='Best-vs-best boxes need not share an anchor; no root-cause label.')


def save_proposals(path, trace, targets):
    """All decoded proposals once per frame/branch; target rows link to this file."""
    targets = np.asarray(sorted(targets), dtype=np.int64)
    n = len(trace['scores'])
    removed_by = np.full(n, -1, dtype=np.int64)
    overlap = np.zeros(n, dtype=np.float32)
    for cid, (other, value) in trace['suppressors'].items():
        removed_by[cid], overlap[cid] = other, value
    np.savez_compressed(path, candidate_id=np.arange(n), anchor_shape=trace['anchor_shape'],
        score=trace['scores'], classification_logit=trace['logits'], decoded_box=trace['decoded'],
        regression_deltas=trace['regression_deltas'],
        ego_corners=trace['corners'], target_indices=targets, gt_iou=trace['ious'][:, targets],
        gt_boxes=trace['gt'], suppressor_id=removed_by, suppressor_pair_iou=overlap,
        final_assigned_gt=trace['assignment'],
        **{name+'_ids': trace['ids'][name] for name in STAGES})
