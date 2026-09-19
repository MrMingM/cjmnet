"""Streaming CPU-only analysis of three questions; never imports Torch.

Reads completed diagnostic logs, writes a NEW directory. All choices use GT
hindsight and are diagnostic opportunities, never deployment/AP results.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def distance_bin(d):
    for lo, hi in zip((0, 20, 40, 60, 80, 100), (20, 40, 60, 80, 100, math.inf)):
        if lo <= d < hi:
            return f'{lo}-{int(hi) if math.isfinite(hi) else "inf"}'
    raise ValueError('Invalid distance')


def pattern(score, geometry):
    return 'both' if score and geometry else 'score_only' if score else 'geometry_only' if geometry else 'neither'


def overlap(a, b):
    from shapely.geometry import Polygon
    pa, pb = (Polygon([(p[0], p[1]) for p in box[:4]]) for box in (a, b))
    union = pa.union(pb).area
    if union <= 0:
        raise ValueError('Invalid polygon union')
    value = pa.intersection(pb).area/union
    if not math.isfinite(value):
        raise ValueError('Non-finite polygon IoU')
    # Original compute_iou returns float32.
    return struct.unpack('f', struct.pack('f', value))[0]


def identity(full):
    return dict(name='KEEP_FULL', recovered_candidates=[], lost_baseline_gt=[],
                new_fp_count=0, fp_count_delta=0, matched_gt=full['matched_gt'], frame_fp=full['frame_fp'])


def costs(b, full):
    before, after = set(full['matched_gt']), set(b['matched_gt'])
    gained = after-before
    lost = before-after
    if lost != set(b['lost_baseline_gt']):
        raise ValueError('Stored lost GT differs from actual matched GT sets')
    return dict(recovered=len(b['recovered_candidates']), lost=len(lost),
        new_fp=b['new_fp_count'], fp_delta=b['fp_count_delta'],
        all_gt_gained=len(gained), other_gt_gained=len(gained-set(b['recovered_candidates'])),
        net_matched_gt=len(after)-len(before))


POLICIES = {'zero_cost': (0, 0), 'no_lost_allow1fp': (0, 1),
            'allow1lost_no_newfp': (1, 0), 'allow1lost_allow5fp': (1, 5),
            'net_gt_minus_newfp': None}


def choose(actions, full, policy):
    options = [identity(full)] + actions
    values = [(b, costs(b, full)) for b in options]
    cap = POLICIES[policy]
    if cap is not None:
        values = [(b, c) for b, c in values if c['lost'] <= cap[0] and c['new_fp'] <= cap[1]]
        key = lambda bc: (-bc[1]['recovered'], -bc[1]['net_matched_gt'], bc[1]['lost'],
                         bc[1]['new_fp'], bc[0]['name'] != 'KEEP_FULL', bc[0]['name'])
    else:
        key = lambda bc: (-(bc[1]['net_matched_gt']-bc[1]['new_fp']), bc[1]['lost'],
                         bc[1]['new_fp'], bc[0]['name'] != 'KEEP_FULL', bc[0]['name'])
    return min(values, key=key)


def pareto(actions, full):
    options = [identity(full)]+actions
    values = [costs(b, full) for b in options]
    vectors = [(v['recovered'], v['net_matched_gt'], -v['lost'], -v['new_fp']) for v in values]
    return [dict(action=b['name'], **v) for b, v, vector in zip(options, values, vectors)
            if not any(all(x >= y for x, y in zip(other, vector)) and other != vector for other in vectors)]


def analyze_weather(folder, out):
    protocol, summary = read(folder/'protocol.json'), read(folder/'summary.json')
    path = folder/'diagnostics.jsonl'
    if (not summary['complete'] or summary.get('diagnostic_schema') != 1 or protocol['smoke']
            or protocol['stage'] != '3B' or not protocol['development_only'] or protocol['test_data_used']):
        raise ValueError('Require complete, non-smoke development diagnostic')
    before_hash = sha(path)
    if before_hash != summary['diagnostics_sha256']:
        raise ValueError('Diagnostic log SHA mismatch')
    default_nms = protocol['postprocessing']['nms_threshold']
    default_score = protocol['postprocessing']['score_threshold']
    q1 = defaultdict(Counter); q2 = defaultdict(Counter); q3 = defaultdict(Counter)
    seen, count = set(), 0
    with path.open(encoding='utf-8') as source, \
         (out/'recovery_overlap.jsonl').open('w', encoding='utf-8') as intersections, \
         (out/'box_changes.jsonl').open('w', encoding='utf-8') as events, \
         (out/'frame_choices.jsonl').open('w', encoding='utf-8') as choices:
        for line in source:
            frame = json.loads(line); index = frame['sample_index']
            if index in seen:
                raise ValueError('Duplicate frame')
            seen.add(index)
            branches = frame['branches']; targets = frame['candidate_targets']; count += len(targets)
            if len(targets) != len(set(targets)) or len(branches) != len({b['name'] for b in branches}):
                raise ValueError('Duplicate target/branch')
            subsets = [b for b in branches if b['family'] == 'subset']
            full = max(subsets, key=lambda b: len(b['subset']))
            if full['recovered_candidates'] or full['lost_baseline_gt'] or full['new_fp_count']:
                raise ValueError('Full subset must reproduce baseline')
            for b in branches:
                if set(b['targets']) != set(map(str, targets)):
                    raise ValueError('Missing target in branch')
                expected = {j for j in targets if b['targets'][str(j)]['matched']}
                if expected != set(b['recovered_candidates']):
                    raise ValueError('Recovery flag mismatch')
            for j in targets:
                target = str(j); meta = frame['target_metadata'][target]
                base = frame['baseline_targets'][target]
                valid_peers = {b['peer'] for b in branches if b['family'] == 'peer_alone' and b['targets'][target]['matched']}
                if not valid_peers:
                    raise ValueError('Candidate has no valid single peer')
                def recovered(family):
                    return {b['peer'] for b in branches if b['family'] == family
                            and b['peer'] in valid_peers and b['targets'][target]['matched']}
                score = recovered('peer_score_full_geometry'); geometry = recovered('full_score_peer_geometry')
                groups = ['all', 'failure:'+meta['failure_stage'], 'scene:'+str(meta['scene']),
                          'distance:'+distance_bin(meta['distance']),
                          'scene_distance:'+str(meta['scene'])+'/'+distance_bin(meta['distance'])]
                target_pattern = pattern(bool(score), bool(geometry))
                query_hit = any(b['family'] == 'peer_query_scale_all' and b['targets'][target]['matched'] for b in branches)
                subset_hit = any(b['family'] == 'subset' and b['targets'][target]['matched'] for b in branches)
                for group in groups:
                    c = q1[group]; c['target_occurrences'] += 1
                    c['target_'+target_pattern] += 1
                    c['both_same_peer_exists'] += int(bool(score & geometry))
                    c['both_only_different_peers'] += int(bool(score and geometry and not score & geometry))
                    c['query_subset_'+pattern(query_hit, subset_hit)] += 1
                    for peer in valid_peers:
                        c['valid_peer_target_pairs'] += 1
                        c['pair_'+pattern(peer in score, peer in geometry)] += 1
                dump(intersections, dict(sample_index=index, target_index=j, **meta,
                    valid_peers=sorted(valid_peers), score_recovery_peers=sorted(score),
                    geometry_recovery_peers=sorted(geometry), target_pattern=target_pattern,
                    same_peer_both=sorted(score & geometry), query_recovered=query_hit, subset_recovered=subset_hit))
                for b in branches:
                    current = b['targets'][target]
                    if not current['matched']:
                        continue
                    family = b['family']; key = family+'/'+meta['failure_stage']
                    c = q2[key]; c['recovered_branch_target_evaluations'] += 1
                    record = dict(sample_index=index, target_index=j, action=b['name'], family=family,
                        initial_failure=meta['failure_stage'], scene=meta['scene'], distance=meta['distance'],
                        matched_candidate_id=current['matched_candidate_id'], matched_proposal=current['matched_proposal'])
                    # Fixed baseline highest-scoring qualified anchor, not a moving best box.
                    cid = base['stages']['decoded']['highest_qualifying_score_candidate_id']
                    if cid is not None:
                        old, now = base['watched_anchors'][str(cid)], current['watched_anchors'][str(cid)]
                        stable = old['gt_iou'] >= .7 and now['gt_iou'] >= .7
                        threshold = b.get('score_threshold', default_score)
                        crossing = old['score'] <= default_score and now['score'] > default_score
                        record['fixed_qualified_anchor'] = dict(candidate_id=cid, before=old, after=now,
                            score_delta=now['score']-old['score'], iou_delta=now['gt_iou']-old['gt_iou'],
                            geometry_still_qualifies=stable, crosses_original_score_threshold=crossing,
                            passes_current_threshold=now['score'] > threshold,
                            final_match_uses_this_anchor=current['matched_candidate_id'] == cid)
                        c['fixed_anchor_still_qualifies'] += int(stable)
                        c['fixed_anchor_qualifies_and_crosses_original_threshold'] += int(stable and crossing)
                        c['final_match_uses_fixed_anchor'] += int(current['matched_candidate_id'] == cid)
                    flags = set(); pairs = []
                    for e in base['qualifying_nms_suppressions']:
                        good_id, other_id = e['suppressed']['candidate_id'], e['suppressor']['candidate_id']
                        good, other = current['watched_anchors'][str(good_id)], current['watched_anchors'][str(other_id)]
                        old_good, old_other = base['watched_anchors'][str(good_id)], base['watched_anchors'][str(other_id)]
                        initial_overlap = overlap(old_good['ego_corners'], old_other['ego_corners'])
                        if abs(initial_overlap-e['pair_iou']) > 1e-6:
                            raise ValueError('Saved/recomputed baseline polygon IoU disagree')
                        new_overlap = overlap(good['ego_corners'], other['ego_corners'])
                        nms = b.get('nms_threshold', default_nms)
                        pair_flags = dict(original_suppressor_now_qualifies=other['gt_iou'] >= .7,
                            original_suppressor_is_final_match=current['matched_candidate_id'] == other_id,
                            original_good_is_final_match=current['matched_candidate_id'] == good_id,
                            good_now_strictly_higher_score=good['score'] > other['score'],
                            overlap_no_longer_exceeds_original_threshold=new_overlap <= default_nms,
                            overlap_no_longer_exceeds_current_threshold=new_overlap <= nms,
                            suppressor_not_in_current_nms_pool=not other['survives']['nms_topk'],
                            same_suppression_relation_persists=good['suppressed_by'] == other_id)
                        flags.update(k for k, v in pair_flags.items() if v)
                        pairs.append(dict(good_candidate_id=good_id, suppressor_candidate_id=other_id,
                            before_pair_iou=initial_overlap, after_pair_iou=new_overlap,
                            before_score_gap=old_other['score']-old_good['score'],
                            after_score_gap=other['score']-good['score'],
                            good_before=old_good, good_after=good, suppressor_before=old_other,
                            suppressor_after=other, flags=pair_flags))
                    c.update(flags)  # once per recovered branch-target, not per pair
                    record['original_nms_pairs'] = pairs
                    dump(events, record)
            families = defaultdict(list)
            for b in branches:
                families[b['family']].append(b)
                if b['family'] != 'peer_alone':
                    families['ALL_EXPERIMENTAL_EXCEPT_PEER_ALONE'].append(b)
            for family, actions in families.items():
                selections = {}
                for policy in POLICIES:
                    chosen, metrics = choose(actions, full, policy)
                    c = q3[family+'/'+policy]; c['frames'] += 1
                    c['kept_full_frames'] += int(chosen['name'] == 'KEEP_FULL')
                    c.update(metrics)
                    selections[policy] = dict(action=chosen['name'], **metrics)
                dump(choices, dict(sample_index=index, family=family, selections=selections,
                                   pareto=pareto(actions, full)))
            if len(seen) % 25 == 0:
                print(folder.name, 'frames', len(seen), flush=True)
    if sorted(seen) != protocol['selected_candidate_frames'] or count != protocol['candidate_targets']:
        raise ValueError('Frame/target coverage mismatch')
    if sha(path) != before_hash:
        raise ValueError('Input changed during offline analysis')
    return dict(frames=len(seen), occurrences=count, input_sha256=before_hash,
                q1=dict(q1), q2=dict(q2), q3=dict(q3))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input-root', required=True); p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    source, out = Path(args.input_root).resolve(), Path(args.output_dir).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Use a separate sibling output directory, not the input tree')
    import shapely
    out.mkdir(parents=True, exist_ok=False)
    manifest = {str(source/w/name): sha(source/w/name) for w in ('fog', 'rain', 'snow')
                for name in ('summary.json', 'protocol.json')}
    result = dict(complete=False, development_only=True, script_sha256=sha(Path(__file__)),
                  shapely_version=shapely.__version__, input_metadata_sha256=manifest, weathers={})
    for weather in ('fog', 'rain', 'snow'):
        folder = out/weather; folder.mkdir()
        result['weathers'][weather] = analyze_weather(source/weather, folder)
    if any(sha(Path(name)) != value for name, value in manifest.items()):
        raise ValueError('Metadata changed during analysis')
    result['complete'] = True
    (out/'three_questions_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# Stage-3B 三问题离线分析', '',
        '只读取已完成的失败候选帧；不运行模型、不使用GPU，不报告AP。',
        '所有干预选择使用GT事后信息；各族不可相加，无代价也未约束原TP分数变化。', '']
    for weather, r in result['weathers'].items():
        lines += [f'## {weather}: {r["occurrences"]}目标出现 / {r["frames"]}帧', '',
            '### 1. 分数/几何替换恢复交集', '',
            '仅纳入单独能检出当前目标的peer。target按任意有效peer恢复统计，pair严格使用同一peer。', '',
            '| 分组 | 目标次数 | 仅分数 | 仅几何 | 两者均有机会 | 两者都无 | 存在同peer两者均恢复 | 仅不同peer分别恢复 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for group, c in r['q1'].items():
            if group.startswith('scene_distance:'):
                continue
            lines.append('| '+group+' | '+' | '.join(str(c.get(k, 0)) for k in
                ('target_occurrences','target_score_only','target_geometry_only','target_both','target_neither',
                 'both_same_peer_exists','both_only_different_peers'))+' |')
        all_c = r['q1']['all']
        lines += ['', 'query/subset逐目标交集（query可使用本帧任何已测试peer查询）：',
            f'- 仅query {all_c.get("query_subset_score_only",0)}；仅subset {all_c.get("query_subset_geometry_only",0)}；两者 {all_c.get("query_subset_both",0)}；均无 {all_c.get("query_subset_neither",0)}。', '',
            '### 2. 恢复分支中的固定框变化', '',
            '单位为成功分支×目标，可重叠、不独立；各项变化是描述，不是互斥根因。', '',
            '| 干预族/原失败阶段 | 成功次数 | 原合格anchor保持合格且跨原分数门槛 | 最终使用原最高分合格anchor | 原抑制者变合格 | 原抑制者成为最终匹配 | 原框变为严格高分 | 框间IoU不再超过原NMS门槛 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for group, c in r['q2'].items():
            lines.append('| '+group+' | '+' | '.join(str(c.get(k,0)) for k in
                ('recovered_branch_target_evaluations','fixed_anchor_qualifies_and_crosses_original_threshold',
                 'final_match_uses_fixed_anchor','original_suppressor_now_qualifies',
                 'original_suppressor_is_final_match','good_now_strictly_higher_score',
                 'overlap_no_longer_exceeds_original_threshold'))+' |')
        lines += ['', '逐对分数、几何、抑制关系见box_changes.jsonl；新最终框未必属于原固定anchor集合。', '',
            '### 3. 允许保留full后的恢复—代价折中', '',
            '约束为每帧预算，不是全数据集总预算。惩罚策略为总匹配GT净增−新增FP，系数1仅作描述，不等于AP。', '',
            '| 干预族/策略 | 保留full帧 | 补回候选 | 丢失原TP | 新增FP | FP净变化 | 其他GT新增检出 | 全部GT匹配净增 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for group, c in r['q3'].items():
            lines.append('| '+group+' | '+' | '.join(str(c.get(k,0)) for k in
                ('kept_full_frames','recovered','lost','new_fp','fp_delta','other_gt_gained','net_matched_gt'))+' |')
    lines += ['', '## 明细', '',
        '- recovery_overlap.jsonl：每目标有效peer及分数/几何恢复集合。',
        '- box_changes.jsonl：每个成功分支中的原合格框/抑制者变化及新最终匹配。',
        '- frame_choices.jsonl：保留full与各干预的实际选择、四维非劣解集合。',
        '- three_questions_results.json：包括同peer统计、初始失败/场景/距离/场景×距离分层。',
        '仅分析已保存干预，不推断未测试的局部操作。后处理门槛变化与原门槛分别记录。']
    (out/'three_questions_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('DONE:', out/'three_questions_report.md', flush=True)


if __name__ == '__main__':
    main()
