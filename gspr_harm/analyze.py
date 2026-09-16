"""Read existing harm-probe records; no model, GPU or dataset access required."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def analyze_weather(summary, regions, frames):
    counts = Counter(row['class'] for row in regions)
    if any(counts[k] != summary['region_classes'][k] for k in ('harmful', 'beneficial', 'neutral')):
        raise ValueError('Region class counts disagree with summary')
    by_frame = {row['sample_index']: row for row in frames}
    if len(by_frame) != len(frames) or len(frames) != summary['frames']:
        raise ValueError('Frame count/uniqueness mismatch')
    for row in regions:
        if row['sample_index'] not in by_frame:
            raise ValueError('Region frame missing from frames.jsonl')
        delta = row['L_send']['total_loss']-row['L_drop']['total_loss']
        if not math.isclose(delta, row['delta_send_minus_drop']['total_loss'], abs_tol=1e-10):
            raise ValueError('Recorded loss difference inconsistent')
        label = 'harmful' if delta > row['epsilon'] else 'beneficial' if delta < -row['epsilon'] else 'neutral'
        if label != row['class']:
            raise ValueError('Region class inconsistent with epsilon')
    harmful = [r for r in regions if r['class'] == 'harmful']
    harmful_by_frame = defaultdict(list)
    for row in harmful:
        harmful_by_frame[row['sample_index']].append(row)
    conflict = [r for r in harmful if r['lost_after_drop_iou70'] > 0]
    recovery = [r for r in harmful if r['recovered_after_drop_iou70'] > 0]
    unchanged = [r for r in harmful if r['lost_after_drop_iou70'] == 0
                 and r['recovered_after_drop_iou70'] == 0 and r['fp_change_after_drop'] == 0]
    per_frame = []
    for index, frame in by_frame.items():
        selected = harmful_by_frame[index]
        base = frame['controls']['a0b0']
        joint = frame['controls']['gt_single_deletion_joint']
        if sum(r['removed_native_blocks'] for r in selected) != len(joint['removed_sender_blocks']):
            raise ValueError('Joint deletion count differs from harmful regions')
        single_sum = sum(r['delta_send_minus_drop']['total_loss'] for r in selected)
        joint_gain = base['loss']['total_loss']-joint['loss']['total_loss']
        per_frame.append({'sample_index': index, 'harmful_regions': len(selected),
            'single_deletions_with_lost_targets': sum(r['lost_after_drop_iou70'] > 0 for r in selected),
            'sum_single_loss_reductions': single_sum, 'actual_joint_loss_reduction': joint_gain,
            'joint_minus_sum_single': joint_gain-single_sum})
        if 'lost_gt_indices_iou70' in joint and all('lost_gt_indices_iou70' in r for r in selected):
            single_lost = {i for r in selected for i in r['lost_gt_indices_iou70']}
            joint_lost = set(joint['lost_gt_indices_iou70'])
            per_frame[-1].update(joint_lost_gt_indices=sorted(joint_lost),
                joint_lost_also_lost_in_single=sorted(joint_lost & single_lost),
                joint_lost_not_lost_in_any_single=sorted(joint_lost-single_lost))
    joint = summary['controls']['gt_single_deletion_joint']
    base = summary['controls']['a0b0']
    if conflict:
        conclusion = '已证实：部分损失标签与单独删除后的 IoU0.7 目标保留冲突；联合交互也可能存在，不能据此排除。'
    elif joint['lost_vs_a0b0'] > 0:
        conclusion = '所有损失有害区域单独删除均未丢失匹配目标，但联合删除丢失目标：存在联合删除才出现的目标保留问题。'
    else:
        conclusion = '未发现上述目标丢失冲突；计数不变仍不能证明 AP 不变，需考虑置信度排序和定位变化。'
    def example(r):
        return {k: r[k] for k in ('sample_index', 'sender_id', 'blocks_yx',
            'delta_send_minus_drop', 'recovered_after_drop_iou70', 'lost_after_drop_iou70', 'fp_change_after_drop')}
    return {'frames': summary['frames'], 'harmful_regions': len(harmful),
        'harmful_single_deletions_losing_targets': len(conflict),
        'single_loss_conflict_fraction': len(conflict)/len(harmful) if harmful else None,
        'harmful_single_deletions_recovering_targets': len(recovery),
        'harmful_single_deletions_unchanged_match_and_fp_counts': len(unchanged),
        'frames_with_single_loss_conflicts': sorted({r['sample_index'] for r in conflict}),
        'sum_single_lost_target_events_not_unique_objects': sum(r['lost_after_drop_iou70'] for r in harmful),
        'joint_lost_targets': joint['lost_vs_a0b0'], 'joint_recovered_targets': joint['recovered_vs_a0b0'],
        'joint_ap70_change': joint['ap70']-base['ap70'],
        'per_frame_loss_interactions': per_frame,
        'conflict_examples': [example(r) for r in sorted(conflict,
            key=lambda r: r['lost_after_drop_iou70'], reverse=True)[:5]],
        'recovery_examples': [example(r) for r in recovery[:3]], 'interpretation': conclusion}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'complete':
        raise ValueError('Requires a completed harm probe')
    report = {'run': str(run), 'weather': {}, 'limitations': [
        '原记录仅保存目标数量，未保存具体匹配目标 ID；不能将不同单块删除事件直接相加为独立目标数。',
        '旧版记录无法定位联合丢失目标；新版存在 lost_gt_indices 字段时，另报告每帧联合丢失与单独丢失的集合关系。',
        '联合损失收益与单块收益之和的偏差是损失非加性的描述，不单独证明 AP 下降的原因，也包含数值误差。',
        'IoU0.7 匹配数量不变不等于 AP 不变；本报告不是 AP 标签或预测头的有效性验证。']}
    for weather, result in summary['weather'].items():
        report['weather'][weather] = analyze_weather(result, read_rows(run/weather/'regions.jsonl'),
                                                     read_rows(run/weather/'frames.jsonl'))
    output = run/('label_analysis_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    if any(result.get('full_validation') for result in summary['weather'].values()):
        print(json.dumps({w: {k: v for k, v in r.items() if k != 'per_frame_loss_interactions'}
                          for w, r in report['weather'].items()}, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print('Send back: '+str(output))


if __name__ == '__main__':
    main()
