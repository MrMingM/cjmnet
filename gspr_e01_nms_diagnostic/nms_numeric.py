"""NumPy-only descriptive observations; never used to score, filter or repair boxes."""
import hashlib
import math
import numpy as np


def array_hash(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256(str(value.dtype).encode() + repr(value.shape).encode())
    result.update(value.tobytes())
    return result.hexdigest()


def json_numbers(value):
    """Keep nonfinite evidence explicit while producing strict JSON (no NaN literals)."""
    if isinstance(value, np.ndarray):
        return json_numbers(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_numbers(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_numbers(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return 'NaN'
        if math.isinf(value):
            return '+Infinity' if value > 0 else '-Infinity'
        return float(value)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def summary(value):
    value = np.asarray(value)
    finite = np.isfinite(value)
    selected = value[finite].astype(np.float64)
    return dict(shape=list(value.shape), dtype=str(value.dtype), elements=int(value.size),
        sha256=array_hash(value), nan=int(np.isnan(value).sum()),
        positive_infinity=int(np.isposinf(value).sum()), negative_infinity=int(np.isneginf(value).sum()),
        finite_min=float(selected.min()) if selected.size else None,
        finite_max=float(selected.max()) if selected.size else None,
        finite_abs_max=float(np.abs(selected).max()) if selected.size else None,
        zero_count=int(np.count_nonzero(value == 0)), negative_count=int(np.count_nonzero(value < 0)))


def footprint_area(corners):
    """Diagnostic shoelace area of stored native corners; not an alternative IoU/AP."""
    xy = np.asarray(corners, dtype=np.float64)[:4, :2]
    if xy.shape != (4, 2) or not np.isfinite(xy).all():
        return None
    xy = xy - xy[0]
    return float(abs(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                           - xy[:, 1] * np.roll(xy[:, 0], -1))) / 2)


def box_findings(selected, raw_corners, projected):
    selected = np.asarray(selected)
    result = []
    if not np.isfinite(selected).all():
        result.append('nonfinite_selected_decoded_box')
    elif selected.shape == (7,) and (selected[4] <= 0 or selected[5] <= 0):
        result.append('native_decoded_nonpositive_width_or_length')
    elif footprint_area(raw_corners) == 0:
        result.append('positive_width_length_but_native_corners_have_zero_BEV_area')
    elif footprint_area(projected) == 0:
        result.append('positive_native_corner_area_but_projection_has_zero_BEV_area')
    return result


def map_indices(arrays, threshold):
    """Prove the native score-mask and existing filter mapping before attributing raw deltas."""
    required = ('native_prob', 'decoded', 'selected', 'projected', 'large_mask', 'z_mask', 'nms_boxes')
    if any(k not in arrays for k in required):
        return dict(status='unavailable', reason='Missing passive observation stage')
    if not all(len(arrays.get(k + '_calls', [0])) == 1 for k in ('selected', 'projected', 'decoded')):
        return dict(status='unavailable', reason='Unexpected multiple native decode/corner/projection calls')
    anchors = np.flatnonzero(arrays['native_prob'].reshape(-1) > threshold)
    decoded = arrays['decoded'].reshape(-1, 7)
    selected = arrays['selected']
    if len(anchors) != len(selected) or not np.array_equal(decoded[anchors], selected, equal_nan=True):
        return dict(status='unavailable', reason='Native score-to-decoded mapping did not match exactly')
    mask = arrays['large_mask'].reshape(-1).astype(bool) & arrays['z_mask'].reshape(-1).astype(bool)
    if len(mask) != len(selected) or not np.array_equal(arrays['projected'][mask], arrays['nms_boxes'], equal_nan=True):
        return dict(status='unavailable', reason='Native existing-filter-to-NMS mapping did not match exactly')
    return dict(status='exact', nms_to_selected=np.flatnonzero(mask), selected_to_anchor=anchors)
