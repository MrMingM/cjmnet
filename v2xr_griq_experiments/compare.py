"""Compare locked independent mechanisms without selecting on the final set."""
from v2xr_griq_experiments.dataset import OUT, WEATHERS, bind, read, sha

CONTRASTS = {
    'M1': ('q1','q0','additive','logistic_q1','point_counts'),
    'M2': ('dual','single'),
    'M3': ('gated','hard','bce','random','clip'),
}


def main():
    signature = sha(OUT/'DATA_AUDIT.json')
    groups = {}
    for name in CONTRASTS:
        root = OUT/name
        marker = read(root/'COMPLETE.json')
        if marker['data'] != signature or any(sha(root/k)!=v for k,v in marker['artifacts'].items()):
            raise ValueError(f'Group outputs differ: {name}')
        groups[name] = read(root/'metrics.json')
    base = {w:groups['M1']['weather'][w]['F0'] for w in WEATHERS}
    for g in groups.values():
        if any(g['weather'][w]['F0'] != base[w] for w in WEATHERS):
            raise ValueError('Three independent evaluations do not share the same F0')
    lines = ['# 三机制独立比较','',
             '| 组别 | Clean F AP70 | Fog F AP70 | Snow F AP70 | 对 F0 差值（百分点） |',
             '|---|---:|---:|---:|---|']
    records = dict(F0=base)
    for name,g in groups.items():
        records[name] = {w:g['weather'][w][g['primary']] for w in WEATHERS}
    for name, rows in records.items():
        p = [rows[w]['F']['0.7']['ap']*100 for w in WEATHERS]
        delta = [i-base[w]['F']['0.7']['ap']*100 for i,w in zip(p,WEATHERS)]
        lines.append(f'| {name} | {p[0]:.3f}% | {p[1]:.3f}% | {p[2]:.3f}% | '
                     + ' / '.join(f'{v:+.3f}' for v in delta)+' |')
    lines += ['', '每种天气各 829 帧。以下括号内为相对 F0 的场景内配对块区间。']
    credible = []
    for name,g in groups.items():
        lines += ['',f'## {name}', '',
                  '| 天气 | 方法 | F AP70 | 对 F0 变化与区间（百分点） | 校准预算下恢复 / 丢失 / FP | 同 FP 上限离线恢复 / 丢失 / FP |',
                  '|---|---|---:|---|---|---|']
        for w in WEATHERS:
            for arm in ('score_threshold',*CONTRASTS[name]):
                r = g['weather'][w][arm]
                delta = (r['F']['0.7']['ap']-base[w]['F']['0.7']['ap'])*100
                ci = r.get('paired',{}).get('pp_95')
                s = f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else '—'
                q = r['budget']['targets'].get('all',{})
                b = r.get('equal_fp_offline',{})
                frontier = f'{b["recovered"]} / {b["lost"]} / {b["fp"]}' if b else '—'
                lines.append(f'| {w} | {arm} | {r["F"]["0.7"]["ap"]:.6f} | {delta:+.3f} {s} | '
                             f'{q.get("recovered",0)} / {q.get("lost",0)} / {r["budget"]["F"]["0.7"]["fp"]} | {frontier} |')
        arm, controls = CONTRASTS[name][0], ('score_threshold',*CONTRASTS[name][1:])
        favorable = []
        for w in ('fog','snow'):
            all_rows = g['weather'][w]
            r = all_rows[arm]
            ci = r.get('paired',{}).get('pp_95')
            p = r['F']['0.7']['ap']
            wins = p > max(all_rows[k]['F']['0.7']['ap'] for k in controls)
            if ci and ci[0] > 0 and wins:
                favorable.append(w)
        if favorable:
            credible.append([name,favorable])
    lines += ['', '预算阈值在训练场景固定。表中保留正式验证的实际 FP 读数。'
              '每个场景的变化、候选指标及其他消融见各组 metrics.json。']
    decision = ['# 研究判断','']
    if credible:
        for name,weather in credible:
            decision.append(f'{name} 在 {", ".join(weather)} 的配对区间高于零，并超过所列独立对照。'
                            '其余天气、实际误检预算和各场景变化仍需共同解读。')
    else:
        decision.append('三个主机制均未同时满足恶劣天气下相对 F0 的配对区间高于零，且超过同池机制对照。当前证据不足以支持联合训练。')
    decision += ['', '区间描述本次验证集的帧块波动。它不覆盖不同检测权重或不同训练种子的波动。']
    bind(OUT/'FINAL_RESULTS.json',dict(data=signature,groups=groups,comparison=records,positive_weather=credible))
    (OUT/'FINAL_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'FINAL_RESEARCH_DECISION.md').write_text('\n'.join(decision)+'\n',encoding='utf-8')
    bind(OUT/'COMPLETE.json',dict(data=signature, artifacts={name:sha(OUT/name) for name in
         ('FINAL_RESULTS.json','FINAL_COMPARISON.md','FINAL_RESEARCH_DECISION.md')}))


if __name__ == '__main__':
    main()
