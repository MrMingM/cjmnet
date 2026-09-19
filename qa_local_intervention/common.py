"""Pure CPU geometry, deterministic sampling and frame-level choice."""
from collections import Counter
import hashlib
import numpy as np

SEED = 20260919
COUNTS = {'fog': 16, 'rain': 16, 'snow': 32}
SCALES = ((0,), (1,), (0, 1))
RADII = (1., 1.5)
ALPHAS = (.5, 1.)


def compare_full_replay(current, historical):
    """Allow <=1e-6 score roundoff only after exact final decision agreement."""
    for key in ('matched_gt', 'final_candidate_ids', 'final_assigned_gt', 'frame_fp'):
        if current[key] != historical[key]:
            raise ValueError('Stage-3B full output changed: '+key)
    actual = np.asarray(current['final_scores'], dtype=np.float64)
    expected = np.asarray(historical['final_scores'], dtype=np.float64)
    if actual.shape != expected.shape or not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise ValueError('Stage-3B full score shape/nonfinite mismatch')
    # Absolute bound, not a relative tolerance growing with the score.
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0, equal_nan=False,
                               err_msg='Stage-3B full score drift exceeds 1e-6')
    np.testing.assert_allclose(current['final_boxes'], historical['final_boxes'], atol=1e-6, rtol=1e-6)
    delta = np.abs(actual-expected)
    return dict(score_atol=1e-6, score_rtol=0, max_abs_score_delta=float(delta.max()) if delta.size else 0.,
                scores_outside_previous_tolerance=int((delta > 1e-7+1e-6*np.abs(expected)).sum()),
                exact_final_decisions=True, box_tolerance_unchanged=True)


def priority(seed, weather, row):
    return hashlib.sha256(f'{seed}/{weather}/{row["sample_index"]}/{row["target_index"]}'.encode()).hexdigest()


def sample_rows(rows, count, seed, weather, excluded=()):
    """Balanced recovery status, scene, distance; one preselected focal target/frame."""
    chosen, used = [], set(excluded)
    status_counts, scene_counts, distance_counts = Counter(), Counter(), Counter()
    for _ in range(count):
        available = [r for r in rows if r['sample_index'] not in used]
        if not available:
            raise ValueError(f'{weather}: insufficient distinct frames for requested sample')
        def key(r):
            return (status_counts[r['global_recoverable']], scene_counts[str(r['scene'])],
                    distance_counts[r['distance_bin']], priority(seed, weather, r))
        r = min(available, key=key)
        chosen.append(r); used.add(r['sample_index'])
        status_counts[r['global_recoverable']] += 1
        scene_counts[str(r['scene'])] += 1; distance_counts[r['distance_bin']] += 1
    return chosen


def roi_mask(xy, corners, expansion):
    """Cell/anchor centers inside expanded oriented GT rectangle; explicit fallback."""
    points = np.asarray(xy, dtype=np.float64)
    c = np.asarray(corners, dtype=np.float64)[:4, :2]
    center = c.mean(0)
    c = c[np.argsort(np.arctan2(c[:, 1]-center[1], c[:, 0]-center[0]))]
    a, b = c[1]-c[0], c[-1]-c[0]
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    if min(la, lb) <= 0 or expansion < 1:
        raise ValueError('Invalid target box/expansion')
    d = points-center
    mask = (np.abs(d @ (a/la)) <= la*expansion/2+1e-6) & (np.abs(d @ (b/lb)) <= lb*expansion/2+1e-6)
    fallback = not bool(mask.any())
    if fallback:
        mask.flat[int(np.argmin((d*d).sum(-1)))] = True
    return mask, fallback


def grid_xy(shape, lidar_range):
    h, w = shape
    xmin, ymin, _, xmax, ymax, _ = lidar_range
    y = ymin+(np.arange(h)+.5)*(ymax-ymin)/h
    x = xmin+(np.arange(w)+.5)*(xmax-xmin)/w
    yy, xx = np.meshgrid(y, x, indexing='ij')
    return np.stack((xx, yy), -1)


def choose_action(rows, baseline, safe=False):
    options = [baseline]+list(rows)
    if safe:
        options = [r for r in options if not r['lost_gt'] and r['new_fp_count'] == 0]
    return min(options, key=lambda r: (-len(r['recovered_candidates']),
        -r['net_matched_gt'], len(r['lost_gt']), r['new_fp_count'], r['name'] != 'KEEP_FULL', r['name']))


def action_grid():
    for radius in RADII:
        for family in ('score', 'geometry'):
            yield dict(family=family, radius=radius, scales=[], alpha=1.)
        for scales in SCALES:
            for alpha in ALPHAS:
                yield dict(family='weights', radius=radius, scales=list(scales), alpha=alpha)


def global_key(family, scales, alpha):
    return family if family != 'weights' else 'weights_'+''.join(map(str, scales))+'_'+str(alpha)
