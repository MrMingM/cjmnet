"""P0-A: exact discrete best sets, preserving action and CAV distinctions."""
from collections import Counter
from statistics import mean, median

from local_fusion_action_utility_audit.counterfactual import action_keys
from .common import KEEP, WEATHERS, ADVERSE, SPLITS, scene_summary


def action_identity(name):
    if name == KEEP:
        return ('keep', None)
    parts = name.split(':')
    if len(parts) != 2 or parts[0] not in ('single', 'query') or not parts[1].isdigit():
        raise ValueError('Unsupported original source action: ' + name)
    if str(int(parts[1])) != parts[1]:
        raise ValueError('Noncanonical source ID')
    return parts[0], int(parts[1])


def analyze_group(outcomes, names):
    for name in names:
        action_identity(name)
    keys = action_keys(outcomes, names)
    keep = tuple(keys[0][:6])
    best = max((tuple(k[:6]) for k in keys[1:]), default=keep)
    beneficial = best > keep
    indices = [i for i in range(1, len(keys)) if tuple(keys[i][:6]) == best] if beneficial else []
    selected = sorted(names[i] for i in indices)
    sources = sorted({action_identity(name)[1] for name in selected})
    families = sorted({action_identity(name)[0] for name in selected})
    family_by_source = {source: {action_identity(name)[0] for name in selected
                                 if action_identity(name)[1] == source} for source in sources}
    # No focal score/IoU or simplicity field is used to split this set.
    return dict(beneficial=beneficial, keep_discrete6=list(keep), best_discrete6=list(best),
                equivalent_best_actions=selected, equivalent_best_action_count=len(selected),
                best_source_ids=sources, unique_source_count=len(sources),
                family_category=('single+query' if len(families) == 2 else families[0]) if families else None,
                same_source_cross_family=any(len(value) == 2 for value in family_by_source.values()),
                cross_source_redundancy=len(sources) >= 2,
                MODIFY_STABLE_SOURCE_AMBIGUOUS=beneficial and len(sources) >= 2,
                stability_not_measured=True)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values):
    n = len(values)
    return dict(exactly_1=sum(v == 1 for v in values), exactly_2=sum(v == 2 for v in values),
                at_least_3=sum(v >= 3 for v in values),
                mean=mean(values) if n else None, median=median(values) if n else None,
                p90=percentile(values, .9), histogram=dict(Counter(values)))


def summarize_groups(rows):
    rows = list(rows)
    positives = [row for row in rows if row['beneficial']]
    n = len(positives)
    total = len(rows)
    def scenes(selected):
        return scene_summary(Counter(row['scene_id'] for row in selected))
    sets = dict(
        beneficial=positives,
        unique_source=[r for r in positives if r['unique_source_count'] == 1],
        multi_source=[r for r in positives if r['unique_source_count'] >= 2],
        single_only=[r for r in positives if r['family_category'] == 'single'],
        query_only=[r for r in positives if r['family_category'] == 'query'],
        single_and_query=[r for r in positives if r['family_category'] == 'single+query'],
        same_source_cross_family=[r for r in positives if r['same_source_cross_family']],
        same_source_cross_family_only=[r for r in positives if r['same_source_cross_family'] and r['unique_source_count'] == 1],
        MODIFY_STABLE_SOURCE_AMBIGUOUS=[r for r in positives if r['MODIFY_STABLE_SOURCE_AMBIGUOUS']])
    for field in ('equivalent_best_action_count', 'unique_source_count'):
        for suffix, predicate in (('exactly_1', lambda n: n == 1), ('exactly_2', lambda n: n == 2),
                                  ('at_least_3', lambda n: n >= 3)):
            sets[field + '_' + suffix] = [r for r in positives if predicate(r[field])]
    result = dict(total_groups=total, beneficial_discrete_groups=n,
                  positive_fraction=n / total if total else None,
                  equivalent_best_actions=distribution([r['equivalent_best_action_count'] for r in positives]),
                  unique_sources=distribution([r['unique_source_count'] for r in positives]),
                  unique_source_best_fraction=len(sets['unique_source']) / n if n else None,
                  multi_source_best_fraction=len(sets['multi_source']) / n if n else None,
                  unique_source_beneficial_groups=len(sets['unique_source']),
                  multi_source_equivalent_beneficial_groups=len(sets['multi_source']),
                  all_scene_coverage=scenes(rows), scene_coverage=scenes(positives),
                  category_scene_coverage={name: scenes(selected) for name, selected in sets.items()},
                  family_counts={name: len(sets[name]) for name in (
                      'single_only', 'query_only', 'single_and_query', 'same_source_cross_family',
                      'same_source_cross_family_only', 'MODIFY_STABLE_SOURCE_AMBIGUOUS')},
                  # Repeated weather realizations are not independent training cases.
                  unique_physical_positive_groups=len({tuple(r['physical_id']) for r in positives}),
                  unique_positive_frames=len({(r['split'], r['frame']) for r in positives}),
                  weather_distribution=dict(Counter(r['weather'] for r in positives)))
    return result


def build_cells(rows):
    result = {}
    for split in SPLITS:
        for weather in WEATHERS + ('adverse pooled',):
            for task in ('cls', 'reg'):
                for stratum in ('associated', 'background'):
                    selected = [r for r in rows if (split == 'train+validation' or r['split'] == split)
                        and (r['weather'] in ADVERSE if weather == 'adverse pooled' else r['weather'] == weather)
                        and (r['task'], r['stratum']) == (task, stratum)]
                    result[f'{split}/{weather}/{task}/{stratum}'] = summarize_groups(selected)
    for split in SPLITS:
        for task in ('cls', 'reg'):
            for stratum in ('associated', 'background'):
                cells = [result[f'{split}/{w}/{task}/{stratum}'] for w in ADVERSE]
                macro = {'summary_type': 'weather_equal_weight', 'not_a_sample_count': True}
                for field in ('positive_fraction', 'unique_source_best_fraction', 'multi_source_best_fraction',
                              'total_groups', 'beneficial_discrete_groups', 'unique_source_beneficial_groups',
                              'multi_source_equivalent_beneficial_groups'):
                    vals = [r[field] for r in cells]
                    macro[field] = mean(vals) if all(v is not None for v in vals) else None
                for field in ('equivalent_best_actions', 'unique_sources', 'family_counts'):
                    macro[field] = {name: mean(r[field][name] for r in cells)
                        if all(r[field][name] is not None for r in cells) else None
                        for name in cells[0][field] if name != 'histogram'}
                macro['undefined_weather_count'] = sum(r['beneficial_discrete_groups'] == 0 for r in cells)
                macro['scene_coverage'] = {'pooled_coverage_for_context':
                    result[f'{split}/adverse pooled/{task}/{stratum}']['scene_coverage']}
                result[f'{split}/adverse equal-weight/{task}/{stratum}'] = macro
    return result
