"""Assemble predefined main arms and mechanism-specific comparisons."""
import numpy as np
from v2xr_griq_v2.dataset import OUT,V1,WEATHERS,bind,read,sha
from v2xr_griq_v2.evaluate import aps


def main():
    groups={}
    data=sha(OUT/'DATA_AUDIT.json')
    for g in ('E1','E2','E3','E4'):
        p=OUT/g
        marker=read(p/'COMPLETE.json')
        if marker['data']!=data or any(sha(p/k)!=h for k,h in marker['files'].items()):
            raise ValueError(f'Group integrity check failed {g}')
        groups[g]=read(p/'metrics.json')
    e1,e2,e3,e4=[groups[k]['arms'] for k in groups]
    reference=e1['F0']
    if any(x['arms']['F0']!=reference for x in groups.values()):
        raise ValueError('Baseline differs across groups')
    rows={'F0':reference,'V1 additive':e1['E1-0'],'E1-1':e1['E1-1'],'E1-2':e1['E1-2'],
          'E1-3':e1['E1-3'],'E3-4':e3['E3-4'],'E3-3':e3['E3-3'],'E3-5':e3['E3-5']}
    rows.update({f'E2-{s}':e2[f'{s}-additive-fixed'] for s in ('s32','s128','all')})
    stability=read(OUT/'E4/STABILITY.json')
    def mean(r,w):
        return r[w]['F']['macro']
    def fmt(x):
        return ' / '.join(f'{i*100:.3f}' for i in x)
    params={'F0':read(V1.parent/'v2xr-radar-ra-rb-v1/F0/FINAL_RESULTS.json')['parameters']}
    params.update({n:groups['E1']['training'][n]['parameters'] for n in ('E1-1','E1-2','E1-3')})
    params['V1 additive']=groups['E1']['training']['E1-0']['parameters']
    params.update({f'E2-{s}':groups['E2']['training'][f'{s}-additive-fixed']['parameters'] for s in ('s32','s128','all')})
    params.update({n:groups['E3']['training'][n]['parameters'] for n in ('E3-3','E3-4','E3-5')})
    lines=['# GRIQ v2 三天气正式比较','',
           '| 方法 | Clean AP30/50/70 | Fog AP30/50/70 | Snow AP30/50/70 | 三天气平均 MacroAP | 参数 |',
           '|---|---|---|---|---:|---:|']
    for n,r in rows.items():
        lines.append(f'| {n} | '+' | '.join(fmt(aps(r[w]['F'])) for w in WEATHERS)+
                     f' | {100*np.mean([mean(r,w) for w in WEATHERS]):.3f} | {params[n]} |')
    for n,r in stability.items():
        lines.append(f'| E4 {n} 三种子均值 | '+' | '.join(fmt(r[w]['mean'][:3]) for w in WEATHERS)+
                     f' | {100*np.mean([r[w]["mean"][3] for w in WEATHERS]):.3f} | {groups["E4"]["training"][n+"-20261017"]["parameters"]} |')
    pareto={}
    for w in WEATHERS:
        a={k:np.array(aps(v[w]['F'])) for k,v in rows.items()}
        pareto[w]=[k for k,x in a.items() if not any((y>=x).all() and (y>x).any() for y in a.values())]
    lines+=['','所有数值使用 F 协议。O、O*、匹配计数、逐场景及帧块配对区间见各组 metrics.json。',
            '正式结果仅用于报告。E1-3 的损失权重已在训练选模场景锁定。']
    verdict=['# 研究结论','','差值单位为百分点，天气顺序为 Clean、Fog、Snow。']
    def diff(a,b):
        return ' / '.join(f'{100*(mean(a,w)-mean(b,w)):+.3f}' for w in WEATHERS)
    verdict += ['',f'- E1-2 对历史单头 additive 的 MacroAP 差值为 {diff(e1["E1-2"],e1["E1-0"])}。']
    for i,t in enumerate((30,50,70)):
        delta=[100*(aps(e1['E1-2'][w]['F'])[i]-aps(reference[w]['F'])[i]) for w in WEATHERS]
        verdict.append(f'  对 F0 的 AP{t} 差值为 '+ ' / '.join(f'{d:+.3f}' for d in delta)+'。')
    verdict += [f'- E2 全量对 32 帧的固定更新 MacroAP 差值为 {diff(e2["all-additive-fixed"],e2["s32-additive-fixed"])}。',
                f'  早停方案差值为 {diff(e2["all-additive-converge"],e2["s32-additive-converge"])}。']
    evidence={}
    for n in ('E3-0','E3-3','E3-5'):
        evidence[n]={w:e3['E3-4'][w]['mechanism_paired'][n]['interval_pp'][-1] for w in WEATHERS}
        verdict.append(f'- E3-4 对 {n} 的 MacroAP 差值为 {diff(e3["E3-4"],e3[n])}。配对区间为 {evidence[n]}。')
    supported=[w for w in ('fog','snow') if all(evidence[k][w] and evidence[k][w][0]>0 for k in evidence)]
    verdict += [f'- 局部交互同时超过三项对照且配对区间高于零的恶劣天气为 {", ".join(supported) or "无"}。']
    for v in ('additive','independent','joint_local_capacity'):
        delta=[100*(stability['cross'][w]['mean'][3]-stability[v][w]['mean'][3]) for w in WEATHERS]
        verdict.append(f'- E4 局部交互对 {v} 的三种子平均 MacroAP 差值为 '+ ' / '.join(f'{d:+.3f}' for d in delta)+'。')
    cv=read(OUT/'E4/CV_RESULTS.json')
    for i in range(3):
        delta=[100*(mean(cv[f'cv{i}_cross']['weather'],w)-mean(cv[f'cv{i}_independent']['weather'],w)) for w in ('clean','fog')]
        verdict.append(f'- E4 训练场景交叉验证第 {i+1} 折中，局部交互对独立聚合的 Clean/Fog MacroAP 差值为 '+ ' / '.join(f'{d:+.3f}' for d in delta)+'。')
    for arm in ('ablate_no_gspr','ablate_no_radar','ablate_points','intervene_permute','intervene_flatten_radar',
                'intervene_one_cell','intervene_residual_off'):
        verdict.append(f'- E4 {arm} 对完整局部模型的 MacroAP 差值为 {diff(e4[arm],e4["cross-20261017"])}。')
    verdict += ['', '同一验证集的帧块区间只衡量本次评测样本波动。三种子和训练场景折结果在 STABILITY.json 和 CV_RESULTS.json。']
    bind(OUT/'FINAL_RESULTS.json',dict(groups=groups,pareto=pareto,mechanism=evidence,stability=stability,data=data))
    (OUT/'FINAL_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'FINAL_RESEARCH_DECISION.md').write_text('\n'.join(verdict)+'\n',encoding='utf-8')
    bind(OUT/'COMPLETE.json',dict(data=data,artifacts={p:sha(OUT/p) for p in
         ('FINAL_RESULTS.json','FINAL_COMPARISON.md','FINAL_RESEARCH_DECISION.md')}))


if __name__=='__main__':
    main()
