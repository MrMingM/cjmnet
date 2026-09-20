"""Three offline analyses of completed Stage-3C frames; NumPy only, no inference."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from gspr_evidence.stage3_analysis import read_json, write_json
from gspr_evidence.stage3_runtime import sha, source_sha, safe_output
from qa_local_intervention.report import summarize

FAMILIES = ('score', 'geometry', 'weights')


def safe(a):
    return not a['lost_gt'] and a['new_fp_count'] == 0


def outcome(a):
    if not safe(a):
        return 'harmful_detected' if a['focal_detected'] else 'harmful_missed'
    return 'safe_detected' if a['focal_detected'] else 'safe_missed'


def quantiles(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return dict(n=len(a), p10_median_p90=np.quantile(a,[.1,.5,.9]).tolist() if len(a) else None)


def overlap(actions, require_safe=False):
    """One focal target, no action-count inflation; optional same-peer caller."""
    return {family:any(a['family']==family and a['focal_detected'] and
                      (not require_safe or safe(a)) for a in actions) for family in FAMILIES}


def anchor_features(base, score_donor, geometry_donor):
    """Numeric pre-intervention contrasts at GT-selected anchors; NOT GT-free selection."""
    if base is None:
        return None
    result = dict(full_score=base['score'])
    if score_donor is not None:
        result['peer_score_minus_full'] = score_donor['score']-base['score']
        result['peer_logit_minus_full'] = score_donor['logit']-base['logit']
    if geometry_donor is not None:
        b, p = np.asarray(base['decoded_box']), np.asarray(geometry_donor['decoded_box'])
        result['peer_full_center_distance'] = float(np.linalg.norm(p[:3]-b[:3]))
        result['peer_full_size_l2'] = float(np.linalg.norm(p[3:6]-b[3:6]))
        result['peer_full_abs_yaw_difference'] = float(abs(np.arctan2(np.sin(p[6]-b[6]), np.cos(p[6]-b[6]))))
    return result


def inspect_frame(f):
    j = str(f['focal_target']); base = f['baseline']['targets'][j]
    spec = f['sampling']
    identity = dict(weather=f['weather'], sample_index=f['sample_index'], target_index=int(j),
                    cohort=f['cohort'], scene=spec['scene'], distance_bin=spec['distance_bin'],
                    initial_stage=base['failure_stage'])
    anchors = dict(best_iou=base['stages']['decoded']['best_iou_candidate_id'],
                   best_qualified=base['stages']['decoded']['highest_qualifying_score_candidate_id'])
    events = []
    global_map = {(a['peer'],a['family']):a for a in f['global_actions'] if a['family'] in ('score','geometry')}
    for a in f['local_actions']:
        donor_score = global_map[a['peer'],'score']['targets'][j]['watched_anchors']
        donor_geometry = global_map[a['peer'],'geometry']['targets'][j]['watched_anchors']
        features = {name:anchor_features(base['watched_anchors'].get(str(cid)),
                       donor_score.get(str(cid)),donor_geometry.get(str(cid))) for name,cid in anchors.items()}
        after = a['targets'][j]
        events.append(dict(identity, action=a['name'], peer=a['peer'], family=a['family'], scales=a['scales'],
            alpha=a['alpha'], radius=a['radius'], outcome=outcome(a), focal_detected=a['focal_detected'],
            lost_gt=a['lost_gt'], new_fp_count=a['new_fp_count'],
            score_drop_gt005=a['retained_gt_drop_over_005'], features=features,
            post_stage=after['failure_stage'], post_best_iou=after['stages']['decoded']['best_iou'],
            post_qualifying_score_margin=after['qualifying_score_margin'],
            gt_dependent_baseline=dict(qualifying_score_margin=base['qualifying_score_margin'],
                best_iou=base['stages']['decoded']['best_iou'],
                suppressor_score_gaps=[e['score_gap'] for e in base['qualifying_nms_suppressions']]),
            mask_stats=a['mask_stats']))
    complement = dict(identity, modes={})
    for policy in ('any','zero_cost'):
        complement['modes'][policy] = dict(
            target=overlap(f['local_actions'],policy=='zero_cost'),
            by_peer={str(p):overlap([a for a in f['local_actions'] if a['peer']==p],policy=='zero_cost')
                     for p in spec['valid_peers']})
    complement['joint_scale_cases'] = []
    weights = {(a['peer'],a['radius'],a['alpha'],tuple(a['scales'])):a for a in f['local_actions'] if a['family']=='weights'}
    for (peer,radius,alpha,scales),a in weights.items():
        if scales!=(0,1):
            continue
        singles = [weights[peer,radius,alpha,(s,)] for s in (0,1)]
        if a['focal_detected'] and not any(s['focal_detected'] for s in singles):
            complement['joint_scale_cases'].append(dict(peer=peer,radius=radius,alpha=alpha,zero_cost=safe(a)))
    unresolved = None
    if f['cohort']=='candidate' and not any(a['focal_detected'] for a in f['local_actions']):
        critical_ids = set(cid for cid in anchors.values() if cid is not None)
        critical_ids.update(e['suppressor']['candidate_id'] for e in base['qualifying_nms_suppressions'])
        unresolved = dict(identity, baseline=base, roi_stats=f['roi_stats'], actions=[],
            global_can_recover=any(a['focal_detected'] for a in f['global_actions']),
            critical_anchor_output_roi={radius:{str(cid):stats.get('focal_watched_anchor_inside_output_roi',{}).get(str(cid))
                                      for cid in sorted(critical_ids)} for radius,stats in f['roi_stats'].items()})
        for a in f['local_actions']:
            after=a['targets'][j]
            unresolved['actions'].append(dict(name=a['name'],peer=a['peer'],family=a['family'],radius=a['radius'],
                scales=a['scales'],alpha=a['alpha'],stage=after['failure_stage'],
                best_iou=after['stages']['decoded']['best_iou'], margin=after['qualifying_score_margin'],
                lost_gt=a['lost_gt'],new_fp_count=a['new_fp_count'],
                watched_anchors=after['watched_anchors'],suppressions=after['qualifying_nms_suppressions']))
    return events, complement, unresolved


def table(title, headers, rows):
    return [title,'','| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+[
        '| '+' | '.join(str(x) for x in row)+' |' for row in rows]+['']


def dump_lines(path, rows):
    with path.open('w',encoding='utf8') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--stage3c-root',required=True); p.add_argument('--output-dir',required=True)
    args=p.parse_args(); root=safe_output(args.stage3c_root); out=safe_output(args.output_dir)
    if out==root or root in out.parents or out in root.parents:
        raise ValueError('Output must be a separate directory, not inside/above the experiment')
    fingerprints={}; events=[]; complement=[]; unresolved=[]; plans=set()
    for w in ('fog','rain','snow'):
        folder=root/w
        for name in ('protocol.json','summary.json'):
            path=folder/name; fingerprints[str(path)]=sha(path)
        protocol=read_json(folder/'protocol.json')
        if protocol.get('smoke') or not protocol.get('development_only') or protocol.get('test_data_used',True):
            raise ValueError('Require complete development Stage-3C, not smoke/test data')
        plans.add(protocol['plan_sha256'])
        # Existing report validator checks coverage, checksums, action grids, and frame choices.
        summarize(folder)
        summary=read_json(folder/'summary.json')
        for index,digest in summary['frame_sha256'].items():
            path=folder/'frames'/f'{index}.json'; fingerprints[str(path)]=digest
            f=read_json(path)
            if f['weather']!=w or f['cohort']!=f['sampling']['cohort'] or f['focal_target']!=f['sampling']['target_index']:
                raise ValueError('Frame identity mismatch')
            e,c,u=inspect_frame(f); events+=e; complement.append(c)
            if u is not None: unresolved.append(u)
        print(f'{w}: read {len(summary["frame_sha256"])} frames',flush=True)
    if len(plans)!=1: raise ValueError('Mixed sampling plans')
    safe_output(out,create=True)
    boundary=['仅分析已保存的局部动作，不运行模型。候选/控制分别统计；事件重复，不是独立车辆或显著性检验。',
        'GT用于定位区域、挑选有效peer/anchor和评价。分数、框分歧虽来自模型，取样位置仍依赖GT；不能称为无GT选择器。',
        '当前未保存完整无GT候选与所有背景区域，也没有所有尺度的原始权重/特征；不能离线证明触发规则泛化。','']
    # Q1: keep cohort, family and action parameters separate to avoid mixture effects.
    groups=defaultdict(list)
    for e in events:
        key=(e['weather'],e['cohort'],e['family'],str(e['scales']),e['alpha'],e['radius'],e['outcome'])
        groups[key].append(e)
    risk=[]
    for key,rows in sorted(groups.items()):
        features={}
        for anchor in ('best_iou','best_qualified'):
            names=set(k for r in rows for k in (r['features'][anchor] or {}))
            for name in sorted(names):
                features[anchor+'/'+name]=quantiles([(r['features'][anchor] or {}).get(name) for r in rows])
        risk.append(dict(group=list(key),events=len(rows),frames=len({r['sample_index'] for r in rows}),
                         score_drop_events=sum(bool(r['score_drop_gt005']) for r in rows),features=features))
    lines=['# 1 安全恢复与误伤对照','']+boundary
    lines+=['safe_detected：检出且无原TP损失/新增FP；control中表示保持检出。harmful不要求焦点丢失，也包括伤及其他GT或新增FP。','']
    lines+=table('## 动作分组', ['天气/队列/动作/尺度/强度/区域/结果','事件','涉及帧','TP分数降>0.05事件'],
                 [('/'.join(map(str,r['group'])),r['events'],r['frames'],r['score_drop_events']) for r in risk])
    lines+=['## 修改前分歧（P10 / 中位数 / P90；不是阈值推荐）','']
    lines+=table('', ['分组','anchor/指标','有效事件','分位数'], [('/'.join(map(str,r['group'])),k,v['n'],v['p10_median_p90'])
                    for r in risk for k,v in r['features'].items()])
    lines+=['同一动作参数下比较safe_detected与harmful组，再按action_events.jsonl中的scene、distance_bin复查；不能混合天气拟合阈值。',
            '分数下降是额外风险信号，不改变本次zero_cost定义。没有某类事件时，不推断该动作普遍安全。']
    (out/'01_safety.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    # Q2: mutually exclusive family membership, plus same-peer pairs.
    overlap_rows=[]; pair_rows=[]
    for w in ('fog','rain','snow'):
        targets=[r for r in complement if r['weather']==w and r['cohort']=='candidate']
        for policy in ('any','zero_cost'):
            patterns=Counter('+'.join(k for k,v in r['modes'][policy]['target'].items() if v) or 'none' for r in targets)
            for pattern,n in sorted(patterns.items()): overlap_rows.append((w,policy,pattern,n,len(targets)))
            for a,b in combinations(FAMILIES,2):
                both=[r for r in targets if r['modes'][policy]['target'][a] and r['modes'][policy]['target'][b]]
                same=[r for r in both if any(p[a] and p[b] for p in r['modes'][policy]['by_peer'].values())]
                pair_rows.append((w,policy,a+' & '+b,len(both),len(same)))
    lines=['# 2 动作互补性','']+boundary
    lines+=table('## 互斥恢复组合：每个焦点只计一次',['天气','约束','可恢复族组合','目标数','分母'],overlap_rows)
    lines+=table('## 两族交集与同peer交集',['天气','约束','两族','可用不同peer','存在同peer'],pair_rows)
    lines+=table('## 联合尺度独有恢复（固定同peer/强度/区域）',['天气','帧','目标','peer','区域','强度','零代价'],
        [(r['weather'],r['sample_index'],r['target_index'],a['peer'],a['radius'],a['alpha'],a['zero_cost'])
         for r in complement if r['cohort']=='candidate' for a in r['joint_scale_cases']])
    lines+=['联合独有表示尺度0+1成功而同参数两个单尺度均失败。它不等于优于所有单尺度强度的并集。不同区域和动作可重复。']
    (out/'02_complementarity.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    # Q3: all failed focal targets, preserving ambiguity rather than assigning root cause.
    lines=['# 3 未恢复目标排查','']+boundary
    lines+=table('## 数量',['天气','候选焦点','未恢复'],[(w,sum(r['weather']==w and r['cohort']=='candidate' for r in complement),
                   sum(r['weather']==w for r in unresolved)) for w in ('fog','rain','snow')])
    for r in unresolved:
        lines+=[f'## {r["weather"]} frame={r["sample_index"]} target={r["target_index"]}',
                f'场景={r["scene"]}；距离={r["distance_bin"]}；初始={r["initial_stage"]}；全局存在恢复={r["global_can_recover"]}。','']
        lines+=table('### 各动作族最终停留阶段（动作次数）',['族','阶段','次数'],
            [(family,stage,n) for (family,stage),n in sorted(Counter((a['family'],a['stage']) for a in r['actions']).items())])
        lines+=table('### 关键anchor是否在输出修改区域（None表示旧产物未记录）',['区域','anchor','在区域内'],
            [(radius,cid,inside) for radius,ids in r['critical_anchor_output_roi'].items() for cid,inside in ids.items()])
        lines+=table('### 每个动作的最佳定位和分数余量',['动作','阶段','最佳IoU','最高合格分数减门槛','丢失TP','新增FP'],
            [(a['name'],a['stage'],a['best_iou'],a['margin'],len(a['lost_gt']),a['new_fp_count']) for a in r['actions']])
    lines+=['区域包含关系仅针对输出anchor；不能据此确定特征感受野。roi_stats中记录网格数/fallback，但未存全部mask时不能证明两种区域mask相同。',
            '继续score_filtered表示本动作未让合格框得分足够；继续NMS表示竞争仍未解决；不能把最后阶段当作唯一根因。',
            'unresolved.jsonl保留原始/干预后固定anchor与抑制关系，供逐框复查。未恢复不证明来源信息不存在。']
    (out/'03_unresolved.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    dump_lines(out/'action_events.jsonl',events); dump_lines(out/'target_overlap.jsonl',complement)
    dump_lines(out/'unresolved.jsonl',unresolved)
    for path,digest in fingerprints.items():
        if sha(path)!=digest: raise ValueError('Input changed while analyzing: '+path)
    write_json(out/'analysis_results.json',dict(complete=True,source=str(root),plan_sha256=next(iter(plans)),
        input_sha256=fingerprints,analysis_sha256=source_sha(__file__),risk_groups=risk,
        overlap_rows=overlap_rows,pair_rows=pair_rows,unresolved_count=len(unresolved),
        report_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('DONE:',out,flush=True)


if __name__=='__main__':
    main()
