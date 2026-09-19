"""Validate all Stage-3C frame shards and summarize matched local/global actions."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from gspr_evidence import stage3_runtime as sr
from gspr_evidence.stage3_analysis import read_json, write_json
from .common import action_grid, choose_action


def summarize(folder):
    p, s = read_json(folder/'protocol.json'), read_json(folder/'summary.json')
    if not s['complete'] or p['stage']!='3C' or s['smoke']!=p['smoke']:
        raise ValueError('Incomplete/wrong Stage-3C output')
    if s['plan_sha256']!=p['plan_sha256']:
        raise ValueError('Plan drift')
    rows = {r['sample_index']:r for r in p['sampled_rows']}
    if set(map(int,s['frame_sha256']))!=set(rows) or set(rows)!=set(p['selected_candidate_frames']):
        raise ValueError('Frame coverage mismatch')
    decisions, paired, controls, opportunities, mechanisms = (defaultdict(Counter) for _ in range(5))
    denominators = defaultdict(Counter)
    for index, spec in rows.items():
        path=folder/'frames'/f'{index}.json'
        if sr.sha(path)!=s['frame_sha256'][str(index)]:
            raise ValueError('Frame data changed')
        f=read_json(path)
        if f['sampling']!=spec or f['sample_index']!=index:
            raise ValueError('Sampling metadata changed')
        d=denominators[spec['cohort']]
        d['frames']+=1; d['baseline_matched_gt']+=len(f['baseline']['matched_gt'])
        d['baseline_fp']+=f['baseline'].get('frame_fp',0)
        d['valid_peer_events']+=len(spec['valid_peers'])
        locals_=f['local_actions']; globals_=f['global_actions']
        if len(locals_)!=len(spec['valid_peers'])*len(list(action_grid())):
            raise ValueError('Incomplete local action grid')
        if len(globals_)!=len(spec['valid_peers'])*8 or len(f['paired_actions'])!=len(locals_):
            raise ValueError('Incomplete global/paired action grid')
        actual={(r['peer'],r['family'],r['radius'],tuple(r['scales']),r['alpha']) for r in locals_}
        expected={(peer,r['family'],r['radius'],tuple(r['scales']),r['alpha'])
                  for peer in spec['valid_peers'] for r in action_grid()}
        if actual!=expected:
            raise ValueError('Local action parameters changed')
        expected_decisions={f'{scope}/{family}/{policy}' for scope in ('local','global')
                            for family in ('all','score','geometry','weights')
                            for policy in ('zero_cost','recovery_first')}
        if set(f['frame_decisions'])!=expected_decisions:
            raise ValueError('Incomplete decision grid')
        if {r['local'] for r in f['paired_actions']}!={r['name'] for r in locals_}:
            raise ValueError('Missing/duplicate local action pairs')
        by_name={r['name']:r for r in locals_+globals_}
        if len(by_name)!=len(locals_)+len(globals_):
            raise ValueError('Duplicate action names')
        for pair in f['paired_actions']:
            local, global_=by_name[pair['local']], by_name[pair['global_action']]
            for field in ('family','peer','scales','alpha'):
                if local[field]!=global_[field]:
                    raise ValueError('Unpaired intervention comparison')
            label=f'{spec["cohort"]}/{pair["family"]}/scales{pair["scales"]}/alpha{pair["alpha"]}/roi{pair["radius"]}'
            c=paired[label]; c['action_pairs']+=1
            c['global_focus_detected']+=int(global_['focal_detected'])
            c['local_focus_detected']+=int(local['focal_detected'])
            c['both_focus_detected']+=int(local['focal_detected'] and global_['focal_detected'])
            c['global_only']+=int(global_['focal_detected'] and not local['focal_detected'])
            c['local_only']+=int(local['focal_detected'] and not global_['focal_detected'])
            c['local_lost']+=len(local['lost_gt']); c['global_lost']+=len(global_['lost_gt'])
            c['local_new_fp']+=local['new_fp_count']; c['global_new_fp']+=global_['new_fp_count']
        for r in locals_:
            if spec['cohort']=='control':
                label=f'{r["family"]}/scales{r["scales"]}/alpha{r["alpha"]}/roi{r["radius"]}'
                c=controls[label]; c['action_evaluations']+=1
                c['focal_target_lost']+=int(not r['focal_detected'])
                c['any_original_tp_lost']+=int(bool(r['lost_gt']))
                c['any_new_fp']+=int(r['new_fp_count']>0)
                c['any_retained_score_drop_gt005']+=int(bool(r['retained_gt_drop_over_005']))
                c['lost_within10m']+=len(r['lost_within10m_gt']); c['lost_beyond10m']+=len(r['lost_beyond10m_gt'])
            if spec['cohort']=='candidate' and r['focal_detected']:
                label=f'{r["family"]}/{spec["failure_stage"]}'
                c=mechanisms[label]; c['recovered_action_target_events']+=1
                base=f['baseline']['targets'][str(spec['target_index'])]
                after=r['targets'][str(spec['target_index'])]
                fixed=base['stages']['decoded']['highest_qualifying_score_candidate_id']
                if fixed is not None:
                    old=base['watched_anchors'][str(fixed)]; now=after['watched_anchors'][str(fixed)]
                    c['fixed_good_stays_good_crosses_score']+=int(old['gt_iou']>=.7 and now['gt_iou']>=.7 and
                        old['score']<=p['postprocessing']['score_threshold']<now['score'])
                    c['fixed_good_final_match']+=int(after['matched_candidate_id']==fixed)
                suppressors={e['suppressor']['candidate_id'] for e in base['qualifying_nms_suppressions']}
                c['old_suppressor_is_final_match']+=int(after['matched_candidate_id'] in suppressors)
        for scope, actions in (('local',locals_),('global',globals_)):
            for family in ('all','score','geometry','weights'):
                pool=[r for r in actions if family=='all' or r['family']==family]
                c=opportunities[f'{spec["cohort"]}/{scope}/{family}']; c['focal_targets']+=1
                c['any_focus_detected']+=int(any(r['focal_detected'] for r in pool))
                c['safe_focus_detected']+=int(any(r['focal_detected'] and not r['lost_gt'] and r['new_fp_count']==0 for r in pool))
        for key,r in f['frame_decisions'].items():
            scope, family, policy = key.split('/')
            actions = locals_ if scope=='local' else globals_
            pool = [a for a in actions if family=='all' or a['family']==family]
            baseline = dict(name='KEEP_FULL', recovered_candidates=[], lost_gt=[], new_fp_count=0,
                            net_matched_gt=0, focal_detected=spec['target_index'] in f['baseline']['matched_gt'])
            expected = choose_action(pool, baseline, safe=policy=='zero_cost')
            if r != dict(action=expected['name'], **{k:expected[k] for k in
                    ('recovered_candidates','lost_gt','new_fp_count','net_matched_gt','focal_detected')}):
                raise ValueError('Frame decision does not match recorded actions')
            c=decisions[spec['cohort']+'/'+key]; c['frames']+=1
            c['kept_full']+=int(r['action']=='KEEP_FULL')
            c['recovered_candidates']+=len(r['recovered_candidates']); c['lost_gt']+=len(r['lost_gt'])
            c['new_fp']+=r['new_fp_count']; c['net_matched_gt']+=r['net_matched_gt']
            c['focal_detected']+=int(r['focal_detected'])
    return dict(smoke=p['smoke'], sampling=p['sampled_rows'], plan_sha256=p['plan_sha256'],
        decisions=dict(decisions), paired=dict(paired), control_action_harm=dict(controls),
        focal_opportunities=dict(opportunities), local_mechanisms=dict(mechanisms), denominators=dict(denominators))


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--output-dir',required=True)
    args=p.parse_args(); root=sr.safe_output(args.output_dir)
    results={w:summarize(root/w) for w in ('fog','rain','snow')}
    if len({r['plan_sha256'] for r in results.values()})!=1 or len({r['smoke'] for r in results.values()})!=1:
        raise ValueError('Mixed sampling plans/smoke modes')
    write_json(root/'stage3c_results.json',dict(complete=True,weathers=results))
    smoke=results['fog']['smoke']
    lines=['# Stage-3C 局部干预诊断','', '**SMOKE：仅验证代码路径，不作科研结论。**' if smoke else
        '64个候选帧＋24个独立控制帧的开发诊断；GT定义区域和来源有效性，不是部署方法或AP。', '',
        '所有指标仅代表固定分层样本；控制帧属于Stage-2的full成功恢复子集，不代表全部正常帧。',
        '局部动作只修改一个焦点目标区域；每帧只选一个动作或保持full。没有拼接逐目标最优输出。','']
    def table(title, groups, columns):
        lines.extend([title,'','| 分组 | '+' | '.join(k for k,label in columns)+' |',
                      '|---|'+'---:|'*len(columns)])
        for name,c in groups.items():
            lines.append('| '+name+' | '+' | '.join(str(c.get(label,0)) for k,label in columns)+' |')
        lines.append('')
    for w,r in results.items():
        lines+=['## '+w,'']
        table('### 样本与基线',r['denominators'], [('帧','frames'),('原TP','baseline_matched_gt'),('原FP','baseline_fp'),('有效peer×帧','valid_peer_events')])
        table('### 逐焦点目标恢复机会',r['focal_opportunities'], [('目标数','focal_targets'),('任意动作检出','any_focus_detected'),('无TP损失且无新增FP检出','safe_focus_detected')])
        table('### 统一帧级动作（含保持full）',r['decisions'], [('帧','frames'),('保持full','kept_full'),('补回候选','recovered_candidates'),('丢失原TP','lost_gt'),('新增FP','new_fp'),('全部GT净增','net_matched_gt')])
        table('### 同peer、同尺度、同强度的局部/全局配对（事件可重复）',r['paired'],
              [('动作对','action_pairs'),('全局检出','global_focus_detected'),('局部检出','local_focus_detected'),('全局独有','global_only'),('局部独有','local_only'),('全局丢失TP','global_lost'),('局部丢失TP','local_lost'),('全局新增FP','global_new_fp'),('局部新增FP','local_new_fp')])
        table('### 控制帧所有局部动作的误伤（未筛选成功动作）',r['control_action_harm'],
              [('动作数','action_evaluations'),('焦点丢失','focal_target_lost'),('任意原TP丢失','any_original_tp_lost'),('有新增FP','any_new_fp'),('原TP分数降超0.05','any_retained_score_drop_gt005'),('10米内丢失','lost_within10m'),('10米外丢失','lost_beyond10m')])
        table('### 局部成功动作的框变化',r['local_mechanisms'],
              [('成功事件','recovered_action_target_events'),('固定合格框跨门槛','fixed_good_stays_good_crosses_score'),('固定合格框最终匹配','fixed_good_final_match'),('原抑制者最终匹配','old_suppressor_is_final_match')])
    lines+=['## 解释边界','',
        '局部输出替换是人工上限；权重插值固定keys/values。局部特征变化仍可经deblock感受野或NMS影响区域外，结果完整保留。',
        '先比较同一代价约束下局部/全局恢复机会，再看未筛选控制动作误伤；局部有效不等于无GT选择器已可实现。',
        '小区域没有网格中心时采用最近中心，roi_stats明确记录；不是精确感受野边界。',
        '未定义事后AP收益或显著性。详细的每个原TP分数变化、所有最终框、固定anchor和抑制者轨迹见各天气frames/*.json。']
    (root/'stage3c_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('DONE:',root/'stage3c_report.md',flush=True)


if __name__=='__main__':
    main()
