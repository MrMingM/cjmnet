"""Cluster resampling. AP is recomputed from detections, never mean scene AP."""
import copy
import numpy as np


def interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Undefined bootstrap samples')
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                lower=float(np.quantile(values, .025)), upper=float(np.quantile(values, .975)))


def scene_draws(scenes, repetitions=5000, seed=20261014):
    scenes = sorted(set(scenes))
    if len(scenes) < 2:
        raise ValueError('At least two scenes required for paired bootstrap')
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        yield [scenes[i] for i in rng.integers(len(scenes), size=len(scenes))]


def ordered_resample(frames, draw):
    # Fixed canonical scene-block order; original frame order WITHIN a scene.
    # Duplicate the whole scene block according to sampled multiplicity. All
    # weather/methods use exactly the same draw. No global score sort is added.
    from collections import Counter
    counts = Counter(draw)
    return [row for scene in sorted(counts) for _ in range(counts[scene])
            for row in frames if row['scene_id'] == scene]


def ap70(frames, calculate_ap):
    stats = {0.7: dict(tp=[], fp=[], score=[], gt=0)}
    for frame in frames:
        row = frame['stats']['0.7']
        stats[.7]['gt'] += row['gt']
        for field in ('tp', 'fp', 'score'):
            stats[.7][field].extend(row[field])
    if not stats[.7]['gt']:
        raise ValueError('Bootstrap draw has no GT')
    return float(calculate_ap(copy.deepcopy(stats), .7, False)[0])


def k1_bootstrap(source, null, calculate_ap, repetitions=5000, seed=20261014):
    """weather -> method -> frame rows; null is a list of seed mappings."""
    weather_names = ('fog', 'rain', 'snow')
    canonical = [(r['sample_index'], r['scene_id']) for r in source['fog']['same']]
    scenes = [r[1] for r in canonical]
    for table in [source] + null:
        for weather in weather_names:
            for method in ('shared', 'same', 'task'):
                if [(r['sample_index'], r['scene_id']) for r in table[weather][method]] != canonical:
                    raise ValueError('Unpaired scene/frame inputs')
    values = {'same': [], 'task': []}
    for draw in scene_draws(scenes, repetitions, seed):
        effects = []
        for table in [source] + null:
            gains = {'same': [], 'task': []}
            for weather in weather_names:
                ap = {method: ap70(ordered_resample(table[weather][method], draw), calculate_ap)
                      for method in ('shared', 'same', 'task')}
                gains['same'].append(100 * (ap['same'] - ap['shared']))
                gains['task'].append(100 * (ap['task'] - ap['same']))
            effects.append({k: float(np.mean(v)) for k, v in gains.items()})
        for layer in values:
            values[layer].append(effects[0][layer] - float(np.mean([x[layer] for x in effects[1:]])))
    return {layer: dict(**interval(v), distribution=v) for layer, v in values.items()}


def metric_bootstrap(rows, metric, variants, conditional=None, repetitions=5000, seed=20261014):
    """Targets and every perturbation of each target stay in their scene."""
    groups = {}
    for row in rows:
        groups.setdefault(row['scene_id'], []).append(row)
    values = []
    for draw in scene_draws(groups, repetitions, seed):
        selected = [row for scene in draw for row in groups[scene]]
        rates = []
        for variant in variants:
            subset = [r for r in selected if r['variant'] == variant and
                      (conditional is None or r[conditional])]
            # A draw with no retained Task positives contributes zero pair
            # retention, never an optimistic omission or fabricated success.
            rates.append(sum(r[metric] for r in subset) / len(subset) if subset else 0.)
        values.append(float(np.mean(rates)))
    return dict(**interval(values), distribution=values)
