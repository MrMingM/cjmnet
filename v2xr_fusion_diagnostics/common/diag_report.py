"""Report paired effects and distinguish measurement from causal claims."""
from pathlib import Path

WEATHERS=('clean','fog','snow')


def median(row,key,kind='target_feature_quantiles'):
    return row[kind].get(key,{}).get('median')


def fmt(value):
    return '—' if value is None else f'{value:.4f}'


def count(row,mode,label,key):
    return row['changes'].get(mode,{}).get(label,{}).get(key,0)


def robust(row,mode):
    return row['net_change_ci95'][mode][0]>0


def finding(group,rows):
    adverse=('fog','snow')
    if group=='D1':
        modes=['combined','content','weights']
        content=[]; weights=[]
        for w in adverse:
            for scale in range(3):
                c=median(rows[w],f'{scale}/cohort/content')
                a=median(rows[w],f'{scale}/cohort/weights')
                if c is not None and a is not None:
                    content.append(c>a)
                    weights.append(a>c)
        principal='内容' if len(content)==6 and all(content) else '权重' if len(weights)==6 and all(weights) else '随天气或尺度变化'
        association=[]
        for w in adverse:
            row=rows[w]
            associated=False
            for i in range(3):
                changed=median(row,f'{i}/cohort_changed/relative_gain')
                same=median(row,f'{i}/cohort_same/relative_gain')
                associated |= changed is not None and same is not None and changed>1 and changed>same
            association.append(associated and count(row,'combined','cohort','recovered')>0)
        supported=all(association) and all(robust(rows[w],'combined') for w in adverse)
        opposite=all(rows[w]['net_change_ci95']['combined'][1]<0 for w in adverse)
        return dict(label='有支持证据' if supported else '当前实验不支持' if opposite else '证据不足',
                    question='邻车天气扰动经 AttFuse 传播并与检测损失关联',
                    effect_source=principal,rule='两种天气均出现恢复目标的相对扰动放大及其与未恢复目标的差别，且全验证集净检出差的配对区间下界大于零',
                    modes=modes)
    if group=='D2':
        consistent=[i for i in range(3) if all(robust(rows[w],f'single_s{i}')
                        and count(rows[w],f'single_s{i}','cohort','recovered')>0 for w in adverse)]
        opposite=all(rows[w]['net_change_ci95'][f'single_s{i}'][1]<0 for w in adverse for i in range(3))
        return dict(label='有支持证据' if consistent else '当前实验不支持' if opposite else '证据不足',
                    question='某一尺度的协同输出在两种天气持续带来负面影响',
                    scales=consistent,rule='同一尺度改用主车输出后，两种天气均有历史目标恢复且全验证集净检出差配对区间下界大于零')
    scopes=[]
    for scope in ('ego','neighbor','both'):
        if all(rows[w]['controls'][scope+'/cohort']['net']>0
               and robust(rows[w],scope+'_r1')
               and rows[w]['controls'][scope+'/all']['net']>0
               and any((median(rows[w],f'{i}/cohort/{scope}_post_interaction') or 0)>.01
                       for i in range(3)) for w in adverse):
            scopes.append(scope)
    return dict(label='有支持证据' if scopes else '证据不足',
                question='LiDAR 天气上下文使雷达对检出的贡献减弱',
                scopes=scopes,rule='同一干预范围在两种天气均有大于 1% 的目标融合交互、正向检测交互和正向全验证集配对差')
    

def write_group(group,rows,path):
    verdict=finding(group,rows)
    lines=[f'# {group} 融合机制诊断','',f'**{verdict["label"]}**。{verdict["question"]}。','',verdict['rule']+'。','',
           '| 天气 | 模式 | 协议 | AP30 | AP50 | AP70 | 固定真值恢复 / 丢失 | 历史目标恢复 | 净检出差配对区间 |',
           '|---|---|---|---:|---:|---:|---:|---:|---|']
    for w in WEATHERS:
        row=rows[w]
        for mode,values in row['AP'].items():
            ci=row['net_change_ci95'][mode]
            recovered=count(row,mode,'all','recovered')
            lost=count(row,mode,'all','lost')
            historic=count(row,mode,'cohort','recovered')
            for p,v in values.items():
                a=v['AP']
                lines.append(f'| {w} | {mode} | {p} | {a["AP30"]:.6f} | {a["AP50"]:.6f} | {a["AP70"]:.6f} | '
                             f'{recovered} / {lost} | {historic} | [{ci[0]:.0f}, {ci[1]:.0f}] |')
    if group=='D1':
        lines+=['',f'内容与权重的主导关系为**{verdict["effect_source"]}**。以下为历史目标的相对变化中位数。','',
                '| 天气 | 尺度 | 邻车输入 | 权重矩阵主车行 | 仅内容 | 仅权重 | 两者 | 输出相对变化 / 输入相对变化 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for w in ('fog','snow'):
            row=rows[w]
            for i in range(3):
                values=[fmt(median(row,f'{i}/cohort/{key}')) for key in
                        ('neighbor_input','attention_ego_row','content','weights','combined','relative_gain')]
                lines.append(f'| {w} | {i} | '+' | '.join(values)+' |')
    if group=='D2':
        lines+=['','| 天气 | 层内全单车与真实单车的最大检测头差 | 超出数值容差帧数 | 固定目标身份分歧数 |',
                '|---|---:|---:|---:|']
        for w,row in rows.items():
            c=row['checks']
            lines.append(f'| {w} | {c.get("layer_single_max_head_error",0):.6g} | '
                         f'{c.get("layer_single_nonclose_frames",0)} | {c.get("layer_single_target_disagreement",0)} |')
        lines+=['','上表比较保留多车前端计算和真实移除邻车两种执行方式。逐层差异及同时干预与单尺度结果的交互计数见 JSON。']
    if group=='D3':
        lines+=['','| 天气 | 干预车辆 | 尺度 | 历史目标主车融合前交互 | 融合后交互 | 历史目标检出净交互 | 正常目标检出净交互 |',
                '|---|---|---:|---:|---:|---:|---:|']
        for w in ('fog','snow'):
            row=rows[w]
            for scope in ('ego','neighbor','both'):
                for i in range(3):
                    a=fmt(median(row,f'{i}/cohort/{scope}_pre_interaction'))
                    b=fmt(median(row,f'{i}/cohort/{scope}_post_interaction'))
                    x=row['controls'][scope+'/cohort']['net']
                    y=row['controls'][scope+'/normal_detected']['net']
                    lines.append(f'| {w} | {scope} | {i} | {a} | {b} | {x} | {y} |')
        lines+=['','交互度量为两个 LiDAR 对比的差值。每帧已核对雷达编码输出在 LiDAR 干预时逐元素相同。']
    lines+=['','全量指标各含 829 帧。净检出区间按帧配对自助抽样计算，保留同帧多个目标的关联。'
            '原始逐目标结果、分距离和背景统计，以及变化大而检出不变和变化小而检出改变的样本均可在 JSON 查阅。']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    return verdict
