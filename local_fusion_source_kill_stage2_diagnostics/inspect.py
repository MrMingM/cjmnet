"""CPU-only inspection of existing diagnostic JSON; no replay or tolerance."""
import argparse
import math
import re
from pathlib import Path
from local_fusion_source_kill_stage2.common import atomic_json,digest,read_json,safe_path
from .differences import exact_differences


def continuous_path(path):
    # Only the actual original nine-key continuous coordinates and explicit
    # proxy/focal numeric coordinates. Never hide action IDs, discrete flags,
    # array order, candidate membership, counts or ROI metadata.
    return bool(re.search(r'\.(?:keep_key|selected_key|key)\[[67]\]$',path) or
                re.search(r'\.focal_(?:before|after)\.(?:score|iou)$',path) or
                re.search(r'\.(?:chosen_proxy|best_proxy)\[[123]\]$',path))


def ordering_changes(actual,expected,legal_only=False):
    left=actual['current_state_audit']; right=expected['current_state_audit']
    result=[]
    if len(left)!=len(right):
        return [dict(reason='target count differs')]
    def sign(a,b): return int(a>b)-int(a<b)
    for target,(a,b) in enumerate(zip(left,right)):
        aa=[a['keep_key']]+[r['key'] for r in a['actions']]
        bb=[b['keep_key']]+[r['key'] for r in b['actions']]
        a_allowed=[True]+[r.get('allowed',True) for r in a['actions']]
        b_allowed=[True]+[r.get('allowed',True) for r in b['actions']]
        if len(aa)!=len(bb):
            result.append(dict(target_position=target,reason='shortlist length differs'))
            continue
        if legal_only and a_allowed!=b_allowed:
            result.append(dict(target_position=target,reason='A_ONLY eligibility differs'))
            continue
        for i in range(len(aa)):
            for j in range(i+1,len(aa)):
                if legal_only and not (a_allowed[i] and a_allowed[j]):
                    continue
                if sign(tuple(aa[i]),tuple(aa[j]))!=sign(tuple(bb[i]),tuple(bb[j])):
                    result.append(dict(target_position=target,action_positions=[i,j],
                        actual_order=sign(tuple(aa[i]),tuple(aa[j])),expected_order=sign(tuple(bb[i]),tuple(bb[j]))))
    return result


def summarize_pair(actual,expected):
    difference=exact_differences(actual,expected,limit=1000000)
    if len(difference['first_differences'])!=difference['total_differences']:
        raise ValueError('Diagnostic too large; refuse an incomplete summary')
    other=[r for r in difference['first_differences'] if not continuous_path(r['path'])]
    numbers=[r for r in difference['first_differences'] if continuous_path(r['path'])]
    finite=all('absolute_difference' in r and math.isfinite(r['absolute_difference']) for r in numbers)
    return dict(total_differences=difference['total_differences'],categories=difference['categories'],
        only_named_continuous_fields_differ=not other,other_field_difference_count=len(other),
        first_other_field_differences=other[:10],continuous_values_finite=finite,
        max_continuous_absolute_difference=max((r.get('absolute_difference',float('inf')) for r in numbers),default=0.),
        full9_order_changes=ordering_changes(actual,expected),
        a_only_legal_order_changes=ordering_changes(actual,expected,legal_only=True),
        note='Description only; no changed gate, accepted tolerance, or K2 verdict')


def inspect(folder):
    folder=safe_path(folder)
    report=read_json(folder/'DIAGNOSIS.json')
    actual_path=folder/'original_stage1_adapter.json'
    expected_path=folder/'stage1_expected.json'
    summary=summarize_pair(read_json(actual_path),read_json(expected_path))
    summary.update(scope='CPU-only existing JSON analysis',frame=report['frame'],weather=report['weather'],mode=report['mode'],
        environment=report['environment'],s0_shared_reference=report.get('s0_shared_reference'),
        input_hashes={p.name:digest(p) for p in (actual_path,expected_path,folder/'DIAGNOSIS.json')},
        analysis_source_hash=digest(Path(__file__)))
    output=folder/'CPU_SUMMARY.json'
    atomic_json(output,summary)
    print('FRAME:',summary['weather'],summary['mode'],summary['frame'])
    for key in ('total_differences','categories','only_named_continuous_fields_differ','other_field_difference_count',
                'continuous_values_finite','max_continuous_absolute_difference'):
        print(key+' = '+str(summary[key]))
    print('full9_order_change_count = '+str(len(summary['full9_order_changes'])))
    print('a_only_legal_order_change_count = '+str(len(summary['a_only_legal_order_changes'])))
    print('environment = '+str(summary['environment']))
    print('s0_shared_reference = '+str(summary['s0_shared_reference']))
    if summary['first_other_field_differences']:
        print('FIRST_OTHER_DIFF = '+str(summary['first_other_field_differences'][0]))
    if summary['full9_order_changes']:
        print('FIRST_ORDER_CHANGE = '+str(summary['full9_order_changes'][0]))
    if summary['a_only_legal_order_changes']:
        print('FIRST_LEGAL_ORDER_CHANGE = '+str(summary['a_only_legal_order_changes'][0]))
    print('No gate or scientific verdict changed. OUTPUT='+str(output))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis',required=True,help='Existing diagnostic output directory')
    args=parser.parse_args()
    inspect(args.diagnosis)


if __name__=='__main__':
    main()
