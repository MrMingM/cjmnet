"""Audit saved Stage 1 keys in their original current state; never rerun Oracle."""
from collections import Counter
import math
from .common import KEEP, WEATHERS, ADVERSE, SPLITS, scene_summary
from .source_redundancy import action_identity
from local_fusion_source_kill_stage1.oracle_adapter import partition, eligible


def key_tuple(value):
    if len(value) != 9 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
        raise ValueError('Invalid saved nine-field Oracle key')
    return tuple(value)


def analyze_target(record, audit, mode, variant):
    if mode not in ('same', 'task') or variant not in ('FULL', 'A_ONLY'):
        raise ValueError('Unsupported precheck Oracle condition')
    if record['target_index'] != audit['target_index']:
        raise ValueError('Oracle record/audit target alignment changed')
    keep = key_tuple(audit['keep_key'])
    if keep[1:3] != (0, 0) or keep[4] != 0 or keep[-1] != 2:
        raise ValueError('Current-state KEEP key invalid')
    best = keep
    best_sources = (KEEP, KEEP)
    same = keep
    groups = Counter()
    for action in audit['actions']:
        cls, reg = action['cls_source'], action['reg_source']
        action_identity(cls)
        action_identity(reg)
        if cls == KEEP and reg == KEEP:
            raise ValueError('KEEP must be the baseline, not a shortlisted action')
        if mode == 'same' and cls != reg:
            raise ValueError('Same contains a task-separated action')
        key = key_tuple(action['key'])
        group = partition(key, keep)
        allowed = eligible(variant, group)
        if (action['group'], action['allowed']) != (group, allowed):
            raise ValueError('Saved current-state group/eligibility inconsistent')
        groups[group] += 1
        if allowed and key > best:
            best, best_sources = key, (cls, reg)
        if allowed and cls == reg and key > same:
            same = key
    if dict(groups) != {k: v for k, v in audit['shortlisted'].items() if v}:
        raise ValueError('Saved shortlist counters inconsistent')
    if best != key_tuple(audit['selected_key']) or best_sources != (record['cls_source'], record['reg_source']):
        raise ValueError('Saved selected action does not reproduce original strict key ordering')
    selected_group = 'KEEP' if best_sources == (KEEP, KEEP) else partition(best, keep)
    if selected_group != audit['selected_group'] or record['changed'] != (selected_group != 'KEEP'):
        raise ValueError('Saved winner group inconsistent')
    separated = best_sources[0] != best_sources[1]
    if record['task_separated'] != separated:
        raise ValueError('Saved task separation inconsistent')
    if (best[0], -best[5]) != (record['matched_after'], record['fp_after']):
        raise ValueError('Saved selected key/counts inconsistent')
    cavs = [action_identity(name)[1] for name in best_sources]
    return dict(target_index=record['target_index'],
        selected_discrete_positive=best[:6] > keep[:6],
        selected_group=selected_group, changed=record['changed'], task_separated=separated,
        task_separated_both_nonkeep=separated and None not in cavs,
        cross_cav_selected=separated and None not in cavs and cavs[0] != cavs[1],
        same_cav_cross_family_selected=separated and None not in cavs and cavs[0] == cavs[1],
        keep_vs_source_separated=separated and None in cavs,
        true_task_improvement=mode == 'task' and best[:6] > same[:6],
        best_same_discrete6=list(same[:6]), selected_discrete6=list(best[:6]),
        step_recovered=best[4], step_lost=-best[1], step_new_fp=-best[2])


def summarize_details(rows):
    result = {}
    for split in SPLITS:
        for weather in WEATHERS + ('adverse pooled',):
            selected = [r for r in rows if (r['split'] == split or split == 'train+validation') and
                        (r['weather'] in ADVERSE if weather == 'adverse pooled' else r['weather'] == weather)]
            # The current Stage 1 contract contains validation only. Pooled true
            # sample n remains unavailable when no train trajectory exists.
            complete = split == 'validation'
            same = [r for r in selected if r['mode'] == 'same' and r['selected_discrete_positive']]
            task = [r for r in selected if r['mode'] == 'task']
            positive = [r for r in task if r['true_task_improvement']]
            scenes = scene_summary(Counter(r['scene_id'] for r in positive))
            result[f'{split}/{weather}'] = dict(
                true_task_improvement_targets=len(positive) if complete else None,
                observed_validation_true_task_improvement_targets=len(positive) if split != 'train' else None,
                joint_same_positive_target_groups=len(same) if complete else None,
                observed_validation_joint_same_positive_target_groups=len(same) if split != 'train' else None,
                scene_coverage=scenes if complete else None,
                observed_validation_scene_coverage=scenes if split != 'train' else None,
                max_scene_share=scenes['max_scene_share'] if complete else None,
                same_scene_coverage=scene_summary(Counter(r['scene_id'] for r in same)) if complete else None,
                unique_physical_task_positive_targets=len({(r['split'], r['frame'], r['target_index']) for r in positive})
                    if complete else None,
                task_selected_separation_breakdown={name: sum(r[name] for r in task) for name in (
                    'task_separated_both_nonkeep', 'cross_cav_selected',
                    'same_cav_cross_family_selected', 'keep_vs_source_separated')} if complete else None,
                weather_distribution=dict(Counter(r['weather'] for r in positive)) if complete else None,
                same_state_comparator='KEEP + allowed same-action-name candidates in the saved Task shortlist',
                unit='GT-target ROI/current-state decision; not S0 Shared-proposal candidate')
    for split in SPLITS:
        rows_weather = [result[f'{split}/{w}'] for w in ADVERSE]
        result[f'{split}/adverse equal-weight'] = {
            'summary_type': 'weather_equal_weight', 'not_a_sample_count': True,
            **{field: sum(r[field] for r in rows_weather) / len(ADVERSE)
               if all(r[field] is not None for r in rows_weather) else None for field in (
                   'true_task_improvement_targets', 'joint_same_positive_target_groups',
                   'observed_validation_true_task_improvement_targets')},
            'scene_coverage': {'pooled_coverage_for_context': result[f'{split}/adverse pooled']['scene_coverage']}}
    return result
