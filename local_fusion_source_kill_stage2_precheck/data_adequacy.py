"""Exact available counts; absent train Oracle labels remain null, never zero."""
from collections import Counter
from .common import WEATHERS, expand_cells, scene_summary, merge_scenes


def h1_from_summary(k0):
    base = {}
    for split in ('train', 'validation'):
        for weather in WEATHERS:
            for task in ('cls', 'reg'):
                for stratum in ('associated', 'background'):
                    key = f'{split}/{weather}/{task}/{stratum}'
                    old = k0['conditions'][key]
                    coverage = old['beneficial_set_scene_coverage']
                    counts = {split + ':' + str(scene): value for scene, value in coverage['counts'].items()}
                    positives = coverage['group_count']
                    if sum(counts.values()) != positives:
                        raise ValueError('K0 scene counts inconsistent: ' + key)
                    if abs(positives / old['group_count'] - old['discrete_beneficial_set_rate']) > 1e-12:
                        raise ValueError('K0 positive count/rate mismatch: ' + key)
                    base[key] = dict(total_groups=old['group_count'], beneficial_discrete_groups=positives,
                        positive_fraction=positives / old['group_count'], scene_coverage=scene_summary(counts),
                        unique_source_beneficial_groups=None, multi_source_equivalent_beneficial_groups=None,
                        unique_physical_positive_groups=None)
    def merge(rows):
        total = sum(r['total_groups'] for r in rows)
        n = sum(r['beneficial_discrete_groups'] for r in rows)
        return dict(total_groups=total, beneficial_discrete_groups=n,
            positive_fraction=n / total if total else None,
            scene_coverage=merge_scenes(rows, 'scene_coverage'), unique_source_beneficial_groups=None,
            multi_source_equivalent_beneficial_groups=None, unique_physical_positive_groups=None)
    return expand_cells(base, merge)


def h2_from_summary(k10, variant):
    base = {}
    for weather in WEATHERS:
        same = k10['conditions'][f'{weather}_same_{variant}']['counts']
        task = k10['conditions'][f'{weather}_task_{variant}']['counts']
        for split in ('train', 'validation'):
            available = split == 'validation'
            base[f'{split}/{weather}'] = dict(
                available_splits=['validation'] if available else [],
                same_recovered=same['recovered'] if available else None,
                task_recovered=task['recovered'] if available else None,
                task_extra_recovered_targets=task['recovered'] - same['recovered'] if available else None,
                task_separated_selected_actions=task['task_separated'] if available else None,
                task_selected_A=task['selected_A'] if available else None,
                same_selected_A=same['selected_A'] if available else None,
                true_task_improvement_targets=None,
                # Same's selected A necessarily beats KEEP on discrete6:
                # original winner uses strict full9 lexicographic ordering.
                joint_same_positive_target_groups=same['selected_A'] if available else None,
                scene_coverage=None, max_scene_share=None,
                observed_validation_extra_recovered=task['recovered'] - same['recovered'] if available else None,
                status='summary_only' if available else 'unavailable_no_train_oracle')
    def merge(rows):
        result = {'available_splits': sorted({s for r in rows for s in r['available_splits']})}
        for field in base['validation/clean']:
            if field in ('status', 'available_splits', 'scene_coverage', 'max_scene_share'):
                continue
            values = [r[field] for r in rows]
            result[field] = sum(values) if all(v is not None for v in values) else None
        observed = [r['observed_validation_extra_recovered'] for r in rows
                    if r['observed_validation_extra_recovered'] is not None]
        result['observed_validation_extra_recovered'] = sum(observed) if observed else None
        result.update(scene_coverage=None, max_scene_share=None,
                      status='partial_missing_train' if any(not r['available_splits'] for r in rows) else 'summary_only')
        return result
    return expand_cells(base, merge)


def separation_proxy(cls, reg):
    """Set disagreement of independent single-task optima; no joint outcomes exist."""
    both = cls['beneficial'] and reg['beneficial']
    return dict(both_tasks_individually_beneficial=both,
        proxy_disjoint_best_action_sets=both and not set(cls['equivalent_best_actions']) & set(reg['equivalent_best_actions']),
        proxy_disjoint_best_cav_sets=both and not set(cls['best_source_ids']) & set(reg['best_source_ids']),
        is_true_oracle_positive=False)


def proxy_cells(pairs):
    base = {}
    for split in ('train', 'validation'):
        for weather in WEATHERS:
            rows = [r for r in pairs if (r['split'], r['weather'], r['stratum']) == (split, weather, 'associated')]
            selected = [r for r in rows if r['proxy_disjoint_best_cav_sets']]
            base[f'{split}/{weather}'] = dict(total_candidate_focal_groups=len(rows),
                both_individually_beneficial=sum(r['both_tasks_individually_beneficial'] for r in rows),
                proxy_disjoint_best_action_sets=sum(r['proxy_disjoint_best_action_sets'] for r in rows),
                proxy_disjoint_best_cav_sets=len(selected),
                proxy_scene_coverage=scene_summary(Counter(r['scene_id'] for r in selected)))
    def merge(rows):
        return {**{field: sum(r[field] for r in rows) for field in (
            'total_candidate_focal_groups', 'both_individually_beneficial',
            'proxy_disjoint_best_action_sets', 'proxy_disjoint_best_cav_sets')},
            'proxy_scene_coverage': merge_scenes(rows, 'proxy_scene_coverage')}
    return expand_cells(base, merge)
