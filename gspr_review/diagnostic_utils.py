"""Read-only diagnostic interventions and GT region accounting; no training changes."""
import numpy as np


def near_boxes_xy(points, boxes, order='hwl', margin=1.):
    """Expanded oriented BEV boxes. Outside means outside annotated neighborhoods."""
    if order not in ('hwl', 'lwh') or margin < 0:
        raise ValueError('Invalid box order/margin')
    points, boxes = np.asarray(points), np.asarray(boxes)
    result = np.zeros(points.shape[:-1], dtype=bool)
    for box in boxes:
        length, width = (box[5], box[4]) if order == 'hwl' else (box[3], box[4])
        if not np.isfinite(box).all() or min(length, width) <= 0:
            raise ValueError('Invalid GT box')
        dx, dy = points[..., 0]-box[0], points[..., 1]-box[1]
        c, s = np.cos(box[6]), np.sin(box[6])
        result |= (np.abs(c*dx+s*dy) <= length/2+margin) & (np.abs(-s*dx+c*dy) <= width/2+margin)
    return result


def altered_inputs(inputs, cell, eligible, variant):
    """Alter evidence CONTENT only. The caller retains the original eligibility mask."""
    import torch
    if variant not in ('normal', 'permuted', 'constant', 'zero'):
        raise ValueError('Invalid intervention')
    changed = inputs.clone()
    if variant == 'normal' or not eligible.any():
        return changed
    # One source vector per visited geometric cell, not independently per point.
    selected_cells = cell[eligible]
    order = torch.argsort(selected_cells, stable=True)
    sorted_cells = selected_cells[order]
    first = torch.ones_like(sorted_cells, dtype=torch.bool)
    first[1:] = sorted_cells[1:] != sorted_cells[:-1]
    unique = sorted_cells[first]
    values = inputs[eligible][:, -4:][order][first]
    if variant == 'permuted':
        values = values.roll(1, 0)
    elif variant == 'constant':
        values = values.mean(0, keepdim=True).expand_as(values)
    else:
        values = torch.zeros_like(values)
    changed_peer = changed[..., -4:]
    changed_peer[eligible] = values[torch.searchsorted(unique, selected_cells)]
    return changed


def fixed_mask_weights(reviewer, inputs, original, eligible):
    import torch
    delta = reviewer.weight_delta(inputs)
    return torch.where(eligible, (original+delta).clamp(reviewer.floor, 1), original)


def point_totals(original, weights, reference, eligible, valid, region, epsilon):
    selected = eligible & region & valid
    d = weights[selected]-original[selected]
    r = weights[selected]-reference[selected]
    return {'valid_points': int((valid & region).sum()), 'eligible_points': int(selected.sum()),
            'changed_points': int((np.abs(d) > epsilon).sum()),
            'raised_points': int((d > epsilon).sum()), 'lowered_points': int((d < -epsilon).sum()),
            'abs_weight_change_sum': float(np.abs(d).sum()), 'signed_weight_change_sum': float(d.sum()),
            'abs_change_vs_normal_sum': float(np.abs(r).sum())}


def pillar_totals(original, updated, selected, region, epsilon):
    mask = selected & region
    before, after = original[mask], updated[mask]
    delta = np.abs(after-before)
    return {'eligible_pillars': int(mask.sum()),
            'changed_pillars': int((delta.max(-1) > epsilon).sum()) if len(delta) else 0,
            'feature_elements': int(delta.size), 'feature_abs_change_sum': float(delta.sum()),
            'feature_reference_abs_sum': float(np.abs(before).sum())}


def summarize_rows(rows):
    result = {}
    for branch in sorted({r['branch'] for r in rows}):
        branch_rows = [r for r in rows if r['branch'] == branch]
        result[branch] = {'frames': len(branch_rows), 'variants': {}}
        for variant in ('normal', 'permuted', 'constant', 'zero'):
            groups = {}
            for region in ('all', 'target_near', 'outside_target_near'):
                entries = [r['variants'][variant][region] for r in branch_rows]
                sums = {key: sum(e[key] for e in entries) for key in entries[0]}
                n, f, p = sums['eligible_points'], sums['feature_elements'], sums['eligible_pillars']
                sums.update(mean_abs_weight_change=sums['abs_weight_change_sum']/n if n else None,
                            mean_abs_change_vs_normal=sums['abs_change_vs_normal_sum']/n if n else None,
                            mean_abs_pillar_feature_change=sums['feature_abs_change_sum']/f if f else None,
                            relative_pillar_l1_change=sums['feature_abs_change_sum']/sums['feature_reference_abs_sum'] if sums['feature_reference_abs_sum'] else None,
                            changed_pillar_fraction=sums['changed_pillars']/p if p else None)
                groups[region] = sums
            entries = [r['variants'][variant] for r in branch_rows]
            n = groups['all']['eligible_points']
            groups['mean_peer_input_change_vs_normal'] = sum(e['peer_input_abs_change_sum'] for e in entries)/(n*4) if n else None
            result[branch]['variants'][variant] = groups
        result[branch]['baseline_repeat_max_error'] = max(r['baseline_repeat_max_error'] for r in branch_rows)
    return result
