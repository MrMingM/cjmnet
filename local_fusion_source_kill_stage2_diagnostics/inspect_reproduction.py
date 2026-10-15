"""Read a saved K2 replay-failure artifact. No GPU, writes, or changed gate."""
import argparse
from collections import Counter
from copy import deepcopy
import json

from local_fusion_source_kill_stage2.common import digest, read_json, safe_path
from .inspect import continuous_path, ordering_changes
from .differences import exact_differences


def permutation_summary(record, limit=5):
    """Source-pair alignment for diagnosis only; never assign duplicate pairs."""
    actual = record['actual_trajectory']
    historical = record['historical_trajectory']
    chosen = exact_differences(actual['records'], historical['records'], limit=1000000)
    if len(chosen['first_differences']) != chosen['total_differences']:
        raise ValueError('Incomplete selected-record comparison')
    chosen_other = [r for r in chosen['first_differences'] if not continuous_path(r['path'])]
    left, right = actual['current_state_audit'], historical['current_state_audit']
    result = dict(scope='source-pair permutation diagnosis only; no changed acceptance or verdict',
        frame=record.get('frame'), weather=record.get('weather'), mode=record.get('mode'),
        recorded_accepted=record['accepted'], selected_record_noncontinuous_difference_count=len(chosen_other),
        first_selected_record_noncontinuous_differences=chosen_other[:limit],
        target_count_equal=len(left)==len(right), changed_targets=[])
    if len(left) != len(right):
        return result
    def pair(row):
        return (row['cls_source'], row['reg_source'])
    for position, (a,h) in enumerate(zip(left,right)):
        if a.get('target_index') != h.get('target_index'):
            result['changed_targets'].append(dict(position=position, status='TARGET_ORDER_DIFFERS'))
            continue
        aa, hh = a['actions'], h['actions']
        an, hn = list(map(pair,aa)), list(map(pair,hh))
        if an == hn:
            continue
        detail = dict(position=position, target_index=a.get('target_index'),
            current_pair_order=an, historical_pair_order=hn,
            current_actions=aa, historical_actions=hh,
            same_source_pair_multiset=Counter(an)==Counter(hn),
            current_selected_group=a.get('selected_group'), historical_selected_group=h.get('selected_group'),
            current_selected_key=a.get('selected_key'), historical_selected_key=h.get('selected_key'))
        result['changed_targets'].append(detail)
        if Counter(an) != Counter(hn):
            detail['status']='SOURCE_PAIR_MULTISET_CHANGED'
            continue
        if len(set(an)) != len(an):
            detail['status']='AMBIGUOUS_DUPLICATE_SOURCE_PAIRS'
            detail['note']='Historical audits omit ROI expansion; duplicate pairs cannot be safely aligned.'
            continue
        mapping={pair(row):row for row in aa}
        aligned=deepcopy(a)
        aligned['actions']=[deepcopy(mapping[name]) for name in hn]
        difference=exact_differences(aligned,h,limit=1000000)
        if len(difference['first_differences']) != difference['total_differences']:
            raise ValueError('Incomplete aligned comparison')
        other=[r for r in difference['first_differences'] if not continuous_path(r['path'])]
        detail.update(status='UNIQUE_SOURCE_PAIRS_ALIGNED_FOR_DIAGNOSIS',
            note='Source-pair alignment alone does not prove equal ROI membership or authorize resume.',
            aligned_noncontinuous_difference_count=len(other), first_aligned_noncontinuous_differences=other[:limit],
            max_aligned_absolute_difference=max((r.get('absolute_difference',0.) for r in difference['first_differences']),default=0.))
        if not other:
            av, hv = dict(current_state_audit=[aligned]),dict(current_state_audit=[h])
            full=ordering_changes(av,hv)
            legal=ordering_changes(av,hv,legal_only=True)
            detail.update(aligned_full9_order_change_count=len(full),aligned_legal_order_change_count=len(legal),
                first_aligned_full9_order_changes=full[:limit],first_aligned_legal_order_changes=legal[:limit])
    result['changed_target_count']=len(result['changed_targets'])
    return result


def action_at(audit, position):
    if position == 0:
        return dict(cls_source='KEEP_SHARED', reg_source='KEEP_SHARED',
                    group='KEEP', allowed=True, key=audit['keep_key'])
    return audit['actions'][position-1]


def explain_order(change, actual, historical):
    result = dict(change)
    target = change.get('target_position')
    positions = change.get('action_positions')
    if target is not None and positions is not None:
        a = actual['current_state_audit'][target]
        h = historical['current_state_audit'][target]
        result.update(target_index=a.get('target_index'),
            current_actions=[action_at(a, p) for p in positions],
            historical_actions=[action_at(h, p) for p in positions],
            current_selected_group=a.get('selected_group'),
            historical_selected_group=h.get('selected_group'),
            current_selected_key=a.get('selected_key'),
            historical_selected_key=h.get('selected_key'))
    return result


def summarize(record, limit=5):
    if limit < 1:
        raise ValueError('limit must be positive')
    differences = record['differences']
    rows = differences['first_differences']
    if len(rows) != differences['total_differences']:
        raise ValueError('Saved difference list incomplete; cannot classify it safely')
    rejected = record['rejected_differences']
    other = [r for r in rows if not continuous_path(r['path'])]
    result = dict(scope='read-only saved evidence; no scientific verdict or new acceptance',
        frame=record.get('frame'), weather=record.get('weather'), mode=record.get('mode'),
        root_identity=record.get('root_identity'), recorded_accepted=record['accepted'],
        total_differences=differences['total_differences'], categories=differences['categories'],
        max_absolute_difference=record.get('max_absolute_difference'),
        rejected_field_count=len(rejected), noncontinuous_field_count=len(other),
        first_rejected_fields=rejected[:limit], first_noncontinuous_fields=other[:limit],
        first_differences=rows[:limit],
        # The original checker stops before computing order if a field fails.
        # In that case its stored [] must not be presented as proof of no flip.
        original_order_check_executed=not rejected,
        recorded_full9_order_change_count=len(record['full9_order_changes']) if not rejected else None,
        recorded_legal_order_change_count=len(record['a_only_legal_order_changes']) if not rejected else None)
    reasons = []
    if other:
        reasons.append('NONCONTINUOUS_FIELDS_CHANGED')
    if any(continuous_path(r['path']) for r in rejected):
        reasons.append('CONTINUOUS_VALUE_FAILED_LEGACY_CHECK')
    actual, historical = record.get('actual_trajectory'), record.get('historical_trajectory')
    if actual is not None and historical is not None and not other:
        full = ordering_changes(actual, historical)
        legal = ordering_changes(actual, historical, legal_only=True)
        result.update(diagnostic_order_check_executed=True,
            diagnostic_full9_order_change_count=len(full), diagnostic_legal_order_change_count=len(legal),
            first_full9_order_changes=[explain_order(r, actual, historical) for r in full[:limit]],
            first_legal_order_changes=[explain_order(r, actual, historical) for r in legal[:limit]])
        if full:
            reasons.append('FULL9_ORDER_CHANGED')
        if legal:
            reasons.append('A_ONLY_LEGAL_ORDER_CHANGED')
    else:
        result.update(diagnostic_order_check_executed=False,
            diagnostic_order_skip_reason='Noncontinuous differences or missing trajectories prevent aligned order analysis')
    result['observed_reasons'] = reasons
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', required=True, help='Existing *.reproduction.json')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--permutation-only', action='store_true', help='Inspect swapped candidate identities without changing the gate')
    args = parser.parse_args()
    path = safe_path(args.record)
    if args.limit < 1:
        parser.error('--limit must be positive')
    result = (permutation_summary if args.permutation_only else summarize)(read_json(path), args.limit)
    result.update(record_path=str(path), record_sha256=digest(path))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
