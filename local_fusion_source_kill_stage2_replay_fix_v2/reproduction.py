"""Only outcome-homogeneous, wholly forbidden E candidates may permute."""
from collections import Counter
from copy import deepcopy
import re

from local_fusion_source_kill_stage2.common import KEEP
from local_fusion_source_kill_stage2_replay_fix.reproduction import compare_trajectory as original_compare

NAME_PATH = re.compile(r'^trajectory\.current_state_audit\[(\d+)\]\.actions\[\d+\]\.(?:cls_source|reg_source)$')


def source_pair(row):
    return row['cls_source'], row['reg_source']


def harmless_target(audit, selected):
    keep = audit['keep_key']
    return (len(keep) == 9 and keep[8] == 2 and audit['selected_group'] == 'KEEP' and
        audit['selected_key'] == keep and source_pair(selected) == (KEEP, KEEP) and
        selected['changed'] is False and selected['expansion'] is None and
        all(row['group'] == 'E' and row['allowed'] is False and len(row['key']) == 9 and
            row['key'][:8] == keep[:8] and row['key'][8] < keep[8] for row in audit['actions']))


def compare_trajectory(actual, historical):
    result = original_compare(actual, historical)
    if result['accepted']:
        return result
    rejected = result['rejected_differences']
    if not rejected:
        return result  # Numerical ranking failures remain failures.
    positions = set()
    for row in rejected:
        match = NAME_PATH.fullmatch(row['path'])
        if match is None:
            return result  # Not a candidate-name-only failure.
        positions.add(int(match[1]))
    comparison_copy = deepcopy(actual)
    proofs = []
    for position in sorted(positions):
        a, h = actual['current_state_audit'][position], historical['current_state_audit'][position]
        if not (harmless_target(a, actual['records'][position]) and
                harmless_target(h, historical['records'][position])):
            return result
        aa, hh = a['actions'], h['actions']
        names_a, names_h = list(map(source_pair, aa)), list(map(source_pair, hh))
        if Counter(names_a) != Counter(names_h):
            return result
        # This is a comparison copy only. Inference, labels and saved actual
        # trajectories keep their original candidate order and source names.
        for row, old in zip(comparison_copy['current_state_audit'][position]['actions'], hh):
            row['cls_source'], row['reg_source'] = source_pair(old)
        proofs.append(dict(target_position=position, target_index=a['target_index'],
            current_pair_order=names_a, historical_pair_order=names_h,
            candidate_count=len(aa), all_group_E=True, all_unselectable=True,
            selected_KEEP_in_both=True, each_outcome8_equals_own_KEEP=True,
            source_pair_multiset_unchanged=True,
            roi_pair_matching_required=False,
            reason='No candidate can update state; all candidates equal KEEP in outcome8'))
    residual = original_compare(comparison_copy, historical)
    if not residual['accepted']:
        return result
    result.update(status='complete', accepted=True, rejected_differences=[],
        full9_order_changes=residual['full9_order_changes'],
        a_only_legal_order_changes=residual['a_only_legal_order_changes'],
        permitted_unselectable_E_permutations=proofs,
        original_candidate_name_differences=rejected,
        comparison_copy_only=True, original_inference_and_labels_unchanged=True)
    return result
