"""Stage-2.5 CPU stratification and strict completion-aware Stage-3 reports."""
import argparse
from pathlib import Path
from collections import Counter
from .stage3_analysis import (WEATHERS, FAILURES, read_json, read_rows, write_json,
                              stratify, rate, unique_rows, choose_frame_subset)
from .stage3_runtime import load_lineage, safe_output, validate_sources, sha


def table(groups):
    lines = ['| 分组 | 有效来源目标次数 | 全融合漏检次数 | 失败率 | 独立帧数 | 失败帧数 |',
             '|---|---:|---:|---:|---:|---:|']
    for key, r in groups.items():
        value = f"{r['failure_rate']:.2%}" if r['failure_rate'] is not None else 'NA'
        lines.append(f"| {key} | {r['source_valid_occurrences']} | {r['full_miss_occurrences']} | "
                     f"{value} | {r['unique_frames']} | {r['failure_unique_frames']} |")
    return lines


def offline(stage2_root, output):
    validate_sources()
    out = safe_output(output)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(stage='2.5', development_only=True, test_data_used=False,
                  stage2_root=str(Path(stage2_root).resolve()), unit='target-frame occurrence', weathers={})
    for weather in WEATHERS:
        lineage = load_lineage(stage2_root, weather)
        rows = [dict(r, distance=lineage['stats'][k]['distance']) for k, r in lineage['rows'].items()]
        result['weathers'][weather] = stratify(rows)
    write_json(out/'scene_distance_report.json', result)
    lines = ['# Stage-2.5：来源有效但全融合漏检的分布', '',
             '只用 development validation。单位为 target-frame occurrence，不是独立车辆。',
             '分母：Stage-2 审查子集中，至少一个邻车单独检出的目标次数。', '']
    for weather, r in result['weathers'].items():
        lines += ['## '+weather, '', '### 场景', '']+table(r['by_scene'])
        lines += ['', '### 距离', '']+table(r['by_distance'])+['']
    (out/'scene_distance_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def summarize(root, stage):
    out = safe_output(root)
    offline_result = read_json(out/'stage2_5/scene_distance_report.json')
    result = dict(stage=stage, development_only=True, test_data_used=False,
                  stage2_5=offline_result, weathers={})
    modes = set()
    protocols = {}
    for weather in WEATHERS:
        folder = out/weather
        summary = read_json(folder/'summary.json')
        protocol = read_json(folder/'protocol.json')
        if not summary['complete'] or protocol['stage'] != stage:
            raise ValueError('Incomplete or wrong-stage output')
        if not protocol['development_only'] or protocol['test_data_used']:
            raise ValueError('Non-development output')
        if protocol['stage2_root'] != offline_result['stage2_root']:
            raise ValueError('Offline/GPU Stage-2 lineage mismatch')
        modes.add(protocol['smoke']); protocols[weather] = protocol
        if stage == '3A':
            rows = read_rows(folder/'targets.jsonl')
            unique_rows(rows)
            counts = Counter(r['failure_stage'] for r in rows)
            if dict(counts) != summary['failure_counts'] or len(rows) != protocol['candidate_targets']:
                raise ValueError('Target log/summary coverage mismatch')
            if any(k not in FAILURES for k in counts):
                raise ValueError('Unknown failure stage')
            stats = offline_result['weathers'][weather]
            if not protocol['smoke'] and len(rows) != stats['total']['full_miss_occurrences']:
                raise ValueError('Full Stage-3A did not cover all Stage-2 candidates')
            summary.update(scenes=len({r['scene'] for r in rows}),
                fractions={k: rate(counts[k], len(rows)) for k in FAILURES},
                targets_sha256=sha(folder/'targets.jsonl'))
        else:
            rows = read_rows(folder/'frames.jsonl')
            if len(rows) != len({r['sample_index'] for r in rows}) or len(rows) != summary['frames']:
                raise ValueError('Duplicate or missing Stage-3B frames')
            if sorted(r['sample_index'] for r in rows) != protocol['selected_candidate_frames']:
                raise ValueError('Stage-3B frame coverage mismatch')
            expected = dict(occurrences=0, target_wise_recoverable=0, frame_wise_recovered=0,
                            frame_wise_lost_baseline_gt=0, frame_wise_fp_count_delta=0, frame_wise_new_fp=0)
            for r in rows:
                decision = choose_frame_subset(r['subsets'])
                if decision != r['frame_oracle']:
                    raise ValueError('Frame choice differs from frozen selection rule')
                chosen = decision['chosen']
                expected['occurrences'] += len(r['candidate_targets'])
                expected['target_wise_recoverable'] += len(decision['target_wise_recoverable'])
                expected['frame_wise_recovered'] += len(chosen['recovered_candidates'])
                expected['frame_wise_lost_baseline_gt'] += len(chosen['lost_baseline_gt'])
                expected['frame_wise_fp_count_delta'] += chosen['fp_count_delta']
                expected['frame_wise_new_fp'] += chosen['new_fp_count']
            if any(summary[k] != v for k, v in expected.items()):
                raise ValueError('Stage-3B log/summary mismatch')
            if expected['occurrences'] != protocol['candidate_targets']:
                raise ValueError('Stage-3B target coverage mismatch')
        result['weathers'][weather] = summary
    if len(modes) != 1:
        raise ValueError('Cannot mix smoke and full runs')
    result['smoke'] = modes.pop()
    name = 'stage3a' if stage == '3A' else 'stage3b'
    result['protocols'] = protocols
    write_json(out/(name+'_results.json'), result)
    write_json(out/'protocol.json', dict(stage=stage, development_only=True, test_data_used=False,
               smoke=result['smoke'], stage2_root=offline_result['stage2_root'],
               per_weather_protocols={w: sha(out/w/'protocol.json') for w in WEATHERS}))
    lines = ['# '+name+' 诊断报告', '',
        '**SMOKE：只检查连通和一致性，不能作科研结论。**' if result['smoke'] else
        'Development validation＋在线天气；不是 OPV2V-W 正式测试。', '',
        '样本单位为 target-frame occurrence，同一车辆可以跨帧重复出现。', '']
    if stage == '3A':
        lines += ['## 候选范围', '', '| 天气 | 目标出现次数 | 帧数 | 场景数 |', '|---|---:|---:|---:|']
        for weather, s in result['weathers'].items():
            lines.append(f"| {weather} | {s['occurrences']} | {s['frames']} | {s['scenes']} |")
        lines += ['', '## 最后可观察到的失败阶段', '', '| 阶段 | fog | rain | snow |', '|---|---:|---:|---:|']
        for k in FAILURES:
            cells = [f"{s['failure_counts'].get(k, 0)} ({s['fractions'][k]:.1%})" if s['occurrences'] else '0 (NA)'
                     for s in result['weathers'].values()]
            lines.append('| '+k+' | '+' | '.join(cells)+' |')
        lines += ['', '各天气的主要表现：']
        for weather, s in result['weathers'].items():
            leading = max(s['failure_counts'], key=s['failure_counts'].get) if s['failure_counts'] else '无候选'
            lines.append(f'- {weather}：最多出现在 {leading}。')
        lines += ['', '这些分类只定位正确框最后在哪一步消失，不证明 attention、query、分数或 NMS 是根因。',
                  'nms_topk_filtered 是原 NMS 最高分1000框截断；它没有 suppressor。',
                  '逐框 score、IoU、抑制来源、匹配竞争及几何变化见 targets.jsonl 与 proposals/*.npz。', '']
    else:
        lines += ['## 有限车辆组合干预', '',
                  '| 天气 | 候选次数 | 逐目标可恢复 | 统一帧级组合补回 | 丢失原TP | 新增FP | FP数净变化 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for w, s in result['weathers'].items():
            lines.append(f"| {w} | {s['occurrences']} | {s['target_wise_recoverable']} | "
                         f"{s['frame_wise_recovered']} | {s['frame_wise_lost_baseline_gt']} | {s['frame_wise_new_fp']} | {s['frame_wise_fp_count_delta']} |")
        lines += ['', '逐目标机会不能当作统一帧级输出。新增FP按最终FP框之间 IoU≥0.7 的一对一匹配定义，表示几何位置未延续；不是物理对象身份追踪。',
                  '本实验不报告 AP，不是部署方法，也不是融合理论上限；不证明 ego query 压制。', '']
    for w, s in offline_result['weathers'].items():
        lines += ['## '+w+'：Stage-2.5 全候选分布', '', '### 场景（有分母）', '']+table(s['by_scene'])
        lines += ['', '### 距离（有分母）', '']+table(s['by_distance'])+['']
    lines += ['## 复现边界', '', next(iter(protocols.values()))['replay_limit'], '',
              'Stage-3A 不会自动启动 Stage-3B。先人工阅读结果，再决定是否进行独立组合实验。']
    (out/(name+'_report.md')).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', out/(name+'_report.md'), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mode', choices=('offline', '3A', '3B'), required=True)
    p.add_argument('--stage2-root')
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    if a.mode == 'offline':
        if not a.stage2_root:
            p.error('--stage2-root required for offline')
        offline(a.stage2_root, a.output_dir)
    else:
        summarize(a.output_dir, a.mode)


if __name__ == '__main__':
    main()
