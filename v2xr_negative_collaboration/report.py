"""Summarize paired measurements and scenario leave-one-out sensitivity."""
import json
from collections import Counter


def rate(row):
    n=row.get('negative',0)
    total=n+row.get('both',0)
    return n/total if total else None


def pct(value):
    return '—' if value is None else f'{value:.2%}'


def report(out):
    groups={g:json.loads((out/g/'RESULTS.json').read_text()) for g in ('E0','E1','E2')}
    if any(r['status']!='complete' for r in groups.values()):
        raise ValueError('Incomplete three-model comparison')
    lines=['# 恶劣天气下的负向协同','',
           '同一组冻结权重分别运行主车单车和完整协同。每种天气 829 帧，固定真值 6,235 个。框重叠度 0.7 使用官方一对一匹配。','',
           '| 组 | 天气 | 单车 AP30/50/70 | 协同 AP30/50/70 | AP70 差（百分点） | 两者检出 | 正向 | 负向 | 两者漏检 | 负向率 | 净恢复 |',
           '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for g,result in groups.items():
        for w,r in result['weather'].items():
            t=str(r['official'])
            ap=r['ap'][t]
            c=Counter(r['counts'][t+'/all'])
            fmt=lambda m:' / '.join(pct(ap[m][str(x)]) for x in (.3,.5,.7))
            delta=100*(ap['multi']['0.7']-ap['single']['0.7'])
            lines.append(f"| {g} | {w} | {fmt('single')} | {fmt('multi')} | {delta:+.3f} | {c['both']} | {c['positive']} | {c['negative']} | {c['neither']} | {pct(rate(c))} | {c['positive']-c['negative']} |")
    lines+=['','## 阈值敏感性、分层与逐场景检验','',
            '| 组 | 天气 | 置信度阈值 | 分层 | 单车检出 | 正向 | 负向 | 负向率 | 失败阶段 |',
            '|---|---|---:|---|---:|---:|---:|---:|---|']
    loo={}
    for g,result in groups.items():
        loo[g]={}
        for w,r in result['weather'].items():
            loo[g][w]={}
            for key,data in sorted(r['counts'].items()):
                t,label=key.split('/',1)
                row=Counter(data)
                stages=', '.join(f'{k[7:]}={v}' for k,v in row.items() if k.startswith('reason/'))
                lines.append(f"| {g} | {w} | {t} | {label} | {row['both']+row['negative']} | {row['positive']} | {row['negative']} | {pct(rate(row))} | {stages} |")
                if t==str(r['official']) and label.startswith('scene/'):
                    rest=Counter(r['counts'][t+'/all'])-row
                    loo[g][w][label]=rate(rest)
    lines+=['','## 逐场景留一','']
    for g,data in loo.items():
        for w,scenes in data.items():
            scores=[x for x in scenes.values() if x is not None]
            if scores:
                lines.append(f'- {g} {w}，逐一去掉 {len(scenes)} 个场景后，负向率范围 {pct(min(scores))}–{pct(max(scores))}。')
    lines+=['','## 研究判断','',
            '**暂缓研究**。本实验的失败阶段追踪能区分定位、分数筛选和去重，尚需直接控制融合中的具体变化，才能确定协同过程如何影响目标信息。三组独立训练的权重比较仅说明现象随模型变化。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'LEAVE_ONE_SCENE_OUT.json').write_text(json.dumps(loo,indent=2)+'\n',encoding='utf-8')
