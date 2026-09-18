"""Streaming report: unique target opportunities separate from frame-wide costs."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .stage3_analysis import WEATHERS, read_json, write_json, choose_frame_subset
from .stage3_runtime import sha, safe_output
from qa_observation_diagnostic.metrics import distance_bin


def analyze(folder):
    protocol = read_json(folder/'protocol.json')
    summary = read_json(folder/'summary.json')
    path = folder/'diagnostics.jsonl'
    if not summary['complete'] or summary.get('diagnostic_schema') != 1:
        raise ValueError('Incomplete enhanced run')
    if sha(path) != summary['diagnostics_sha256']:
        raise ValueError('Diagnostic log changed after completion')
    seen, total = set(), 0
    categories, transitions, suppressions = Counter(), Counter(), Counter()
    families = defaultdict(lambda: Counter())
    strata = defaultdict(lambda: Counter())
    examples = defaultdict(list)
    margins, nms_gaps = [], []
    peer_transitions = Counter()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            frame = json.loads(line)
            index = frame['sample_index']
            if index in seen:
                raise ValueError('Duplicate frame')
            seen.add(index)
            targets = frame['candidate_targets']; total += len(targets)
            branches = frame['branches']
            by_family = defaultdict(list)
            for b in branches:
                if set(b['targets']) != set(map(str, targets)):
                    raise ValueError('Missing branch target')
                key = b['name'] if b['family'] == 'postprocess' else b['family']
                by_family[key].append(b)
            for family, bs in by_family.items():
                # Frame-local hindsight choice, never a deployable selector.
                adapted = [dict(b, subset=[i]) for i, b in enumerate(bs)]
                decision = choose_frame_subset(adapted)
                chosen = decision['chosen']
                c = families[family]
                c['eligible_frames'] += 1; c['eligible_occurrences'] += len(targets)
                c['target_wise_opportunity'] += len(decision['target_wise_recoverable'])
                c['frame_wise_recovered'] += len(chosen['recovered_candidates'])
                c['lost_original_tp'] += len(chosen['lost_baseline_gt'])
                c['new_fp'] += chosen['new_fp_count']; c['fp_net_change'] += chosen['fp_count_delta']
                safe = [b for b in bs if not b['lost_baseline_gt'] and b['new_fp_count'] == 0]
                c['zero_observed_cost_frame_recovered'] += max([len(b['recovered_candidates']) for b in safe] or [0])
            for j in targets:
                key = str(j)
                meta = frame['target_metadata'][key]
                baseline = frame['baseline_targets'][key]
                margin = baseline.get('qualifying_score_margin')
                if meta['failure_stage'] == 'score_filtered' and margin is not None:
                    margins.append(margin)
                gaps = [e['score_gap'] for e in baseline['qualifying_nms_suppressions']]
                if gaps:
                    nms_gaps.append(min(gaps))  # one minimum per target, not per proposal
                def hits(family):
                    return any(j in b['recovered_candidates'] for b in branches if b['family'] == family)
                patterns = dict(subset_recoverable=hits('subset'),
                    score_swap_recoverable=hits('peer_score_full_geometry'),
                    geometry_swap_recoverable=hits('full_score_peer_geometry'),
                    query_recoverable=hits('peer_query_scale_all'),
                    uniform_recoverable=hits('uniform'), zero_ego_value_recoverable=hits('zero_ego_value'),
                    postprocess_recoverable=hits('postprocess'))
                patterns['neither_head_swap_recovers'] = not (patterns['score_swap_recoverable'] or patterns['geometry_swap_recoverable'])
                for name, value in patterns.items():
                    if value:
                        categories[name] += 1
                        if len(examples[name]) < 12:
                            examples[name].append(dict(sample_index=index, target_index=j, **meta))
                kinds = {e['geometric_relation'] for e in frame['baseline_targets'][key]['qualifying_nms_suppressions']}
                suppressions.update(kinds)  # each target once per kind, overlapping categories
                for dimension, label in [('scene', str(meta['scene'])), ('distance', distance_bin(meta['distance'])),
                                         ('initial_failure', meta['failure_stage'])]:
                    c = strata[dimension+':'+label]; c['occurrences'] += 1
                    for name, value in patterns.items():
                        c[name] += int(value)
            for edge in frame['source_addition_edges']:
                for change in edge['targets'].values():
                    transitions[change['before']+' -> '+change['after']] += 1
            for pair in frame.get('peer_to_ego_peer', []):
                for change in pair['targets'].values():
                    if change['before'] == 'detected':
                        peer_transitions[change['before']+' -> '+change['after']] += 1
    if sorted(seen) != protocol['selected_candidate_frames'] or total != protocol['candidate_targets']:
        raise ValueError('Diagnostic frame/target coverage mismatch')
    def quantiles(values):
        return dict(n=len(values), p10_p50_p90=np.quantile(values, [.1, .5, .9]).tolist() if values else None)
    return dict(frames=len(seen), occurrences=total, patterns=dict(categories),
        score_filtered_margin=quantiles(margins), minimum_suppressor_score_gap=quantiles(nms_gaps),
        peer_to_ego_peer_transitions=dict(peer_transitions),
        families={k: dict(v) for k, v in families.items()}, strata={k: dict(v) for k, v in strata.items()},
        suppressor_relations=dict(suppressions), source_addition_transitions=dict(transitions), examples=dict(examples))


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--output-dir', required=True)
    args = p.parse_args(); root = safe_output(args.output_dir)
    results = read_json(root/'stage3b_results.json')
    data = {w: analyze(root/w) for w in WEATHERS}
    write_json(root/'mechanism_results.json', dict(smoke=results['smoke'], weathers=data))
    lines = ['# Stage-3B 增强机制诊断', '',
        '**SMOKE：不能作科学结论。**' if results['smoke'] else '开发验证集与在线天气；只审查既定失败候选帧，不是完整数据集 AP。', '',
        '各干预均复用固定来源特征。分类/回归交叉替换是人工反事实，不是部署方法。',
        '任何干预有效都只支持该干预能改变结果；不会自动证明 attention 或某辆车是唯一根因。', '']
    for w, r in data.items():
        lines += [f'## {w}：{r["occurrences"]} 次目标出现 / {r["frames"]} 帧', '',
            '| 干预族 | 逐目标机会 | 每帧统一干预补回 | 丢失原TP | 新增FP | FP净变化 | 无原TP损失且无新增FP时可补回 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for family, v in r['families'].items():
            lines.append('| '+family+' | '+' | '.join(str(v.get(k, 0)) for k in
                ('target_wise_opportunity', 'frame_wise_recovered', 'lost_original_tp', 'new_fp', 'fp_net_change',
                 'zero_observed_cost_frame_recovered'))+' |')
        lines += ['', '每帧选择规则沿用补回最多→损失原TP最少→新增FP最少。不同干预族不可相加。',
            'peer 编号只在当前帧内有效；peer-alone 是绕过融合的参考，不是保留 ego 的部署动作。', '',
            f'分数失败目标：最高合格框分数减门槛，P10/中位数/P90：{r["score_filtered_margin"]["p10_p50_p90"]}。',
            f'发生合格框抑制的目标：每目标最小抑制者分数差，P10/中位数/P90：{r["minimum_suppressor_score_gap"]["p10_p50_p90"]}。', '',
            '### 单源已检出 → 加入ego并经过原融合（peer × 目标次数）', '']
        lines += [f'- {k}：{v}' for k, v in r['peer_to_ego_peer_transitions'].items()]
        lines += ['',
            '### 原 full 的 NMS 抑制关系（目标次数，可重叠）', '']
        lines += [f'- {k}：{v}' for k, v in r['suppressor_relations'].items()]
        lines += ['', '这些是几何关系，不把 IoU<0.7 的框直接等同于背景，也不把别车正确框当误检。', '',
            '### 配对来源加入后的阶段转移（边 × 目标次数，不是独立样本）', '']
        lines += [f'- {k}：{v}' for k, v in sorted(r['source_addition_transitions'].items()) if k.split(' -> ')[0] != k.split(' -> ')[1]]
    lines += ['', '## 如何决定下一步', '',
        '- 分数替换补回、几何替换补不回：优先检查分类得分与定位质量的对应关系；查看同 anchor 的 logit/IoU 变化。',
        '- 几何替换补回：几何仍可能通过重叠关系影响 NMS；不要把失败都归于分类。',
        '- 两种替换都补不回：可能需要联合变化，也可能受竞争框影响；不是信息不存在的证明。',
        '- 仅放宽后处理就补回且代价小：先做独立开发验证的完整帧评估；当前候选帧不能用于宣称 AP 提升。',
        '- 车辆组合补回：查看 source_addition_edges，区分 ego+peer 已失败与额外邻车加入后才失败。',
        '- 固定 keys/values 只换 query 后补回：说明该权重干预具有恢复能力；结合 uniform 和单尺度对照，不能直接证明原 query 是唯一根因。',
        '- zero_ego_value 固定原权重，只清除 ego value 项；它也会改变输出幅度，不能单独解释为 ego 有害。',
        '- 对所有判断检查跨场景/距离一致性、原TP损失和新增FP。相关统计及最多12个复查例子见 mechanism_results.json。', '',
        '## 产物与边界', '',
        'diagnostics.jsonl 保存每种干预逐目标阶段、固定 anchor 分数/logit/IoU、真实NMS抑制者、所有最终框、配对来源边。',
        '分层恢复率的分母是本次失败候选，不是全部有效来源目标。重复车辆/帧/组合不独立，不做虚假的显著性声明。',
        '默认不保存每个干预的全部密集框；SAVE_DENSE=1 可保存，磁盘与压缩耗时会明显增加。',
        '旧A的核心重放/追踪/数学实现必须一致，新增编排单独记录SHA；输入哈希和旧结果重放检查不放宽。']
    (root/'mechanism_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', root/'mechanism_report.md', flush=True)


if __name__ == '__main__':
    main()
