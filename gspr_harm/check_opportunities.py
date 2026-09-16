"""看看现有逐块结果里，ΔL 有没有漏掉检测改善的机会；不跑模型。"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path


def summarize(rows, expected):
    counts = Counter()
    classes = {name: Counter() for name in ('harmful', 'beneficial', 'neutral')}
    scenes = {}
    examples = []
    for row in rows:
        label = row['class']
        counts[label] += 1
        recovered = row['recovered_after_drop_iou70']
        lost = row['lost_after_drop_iou70']
        fp_change = row['fp_change_after_drop']
        # 这里看的是目标匹配和误检数量，不把它冒充 AP。
        target_gain = recovered > 0 and lost == 0 and fp_change <= 0
        fp_gain = recovered == 0 and lost == 0 and fp_change < 0
        improvement = target_gain or fp_gain
        conflict = lost > 0
        group = classes[label]
        group['regions'] += 1
        group['recover_targets_without_losing_targets_or_increasing_fp'] += int(target_gain)
        group['reduce_fp_without_changing_target_matches'] += int(fp_gain)
        group['improving_deletions'] += int(improvement)
        group['deletions_losing_targets'] += int(conflict)
        group['unchanged_match_and_fp_counts'] += int(recovered == lost == fp_change == 0)
        if improvement:
            scene = scenes.setdefault(row['scene'], Counter())
            scene['improving_deletions'] += 1
            scene['missed_by_delta_loss'] += int(label != 'harmful')
            if label != 'harmful' and len(examples) < 8:
                examples.append({k: row[k] for k in ('scene', 'sample_index', 'sender_id', 'blocks_yx',
                    'class', 'delta_send_minus_drop', 'recovered_after_drop_iou70',
                    'lost_after_drop_iou70', 'fp_change_after_drop')})
    if any(counts[k] != expected[k] for k in classes):
        raise ValueError('逐块记录数量与 summary 不一致，请确认来自同一轮完整实验')
    improvements = sum(x['improving_deletions'] for x in classes.values())
    missed = sum(classes[k]['improving_deletions'] for k in ('beneficial', 'neutral'))
    return {'by_delta_loss_class': classes,
        'improving_single_deletions': improvements,
        'improving_single_deletions_missed_by_delta_loss': missed,
        'missed_fraction': missed/improvements if improvements else None,
        'scenes_with_improving_deletions': len(scenes),
        'scenes_with_missed_improvements': sum(x['missed_by_delta_loss'] > 0 for x in scenes.values()),
        'per_scene': scenes, 'missed_examples': examples,
        'interpretation': ('已发现 ΔL 漏掉的目标匹配/误检改善机会；还需重跑联合筛选确认 AP。' if missed else
            '未发现满足本检查条件的漏选机会；不能据此认定过滤没有 AP 提升空间。')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'complete':
        raise ValueError('请使用已完成的实验目录')
    report = {'run': str(run), 'weather': {}, 'limits': [
        '只检查现有单块删除结果，不重新计算 AP，也不把目标计数改善等同于 AP 改善。',
        '同一目标可能在多个试删分支出现，区域数量不能当作独立目标数量。',
        '这个条件只取没有目标保留/误检数量代价的改善，未覆盖有得有失的方案或置信度排序变化。',
        '未搜索所有联合删除集合，因此不能给出过滤性能上限，也不能否定过滤方向。',
        'GT 只用于离线诊断，这不是可部署筛选规则。']}
    for weather, values in summary['weather'].items():
        with (run/weather/'regions.jsonl').open(encoding='utf-8') as stream:
            report['weather'][weather] = summarize((json.loads(line) for line in stream if line.strip()),
                                                    values['region_classes'])
    output = run/('opportunity_check_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    compact = {weather: {k: v for k, v in values.items() if k not in ('per_scene', 'missed_examples')}
               for weather, values in report['weather'].items()}
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    print('Send back: '+str(output))


if __name__ == '__main__':
    main()
