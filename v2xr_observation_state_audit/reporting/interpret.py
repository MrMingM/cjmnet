"""Add measured answers and annotation checks to the completed report."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import os
import sys

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'

import numpy as np

WEATHERS=('clean','fog','snow')


def quality(root):
    result={}
    for file in sorted((root/'cache').glob('*.npz')):
        with np.load(file,allow_pickle=False) as data:
            scene=str(data['frame'][0])
            for w in WEATHERS:
                for kind in ('grid','target','prediction'):
                    prefix=f'{w}_{kind}_'
                    tag=f'{w}/{kind}'
                    counts=result.setdefault(tag,defaultdict(Counter))[scene]
                    a,b,c,d=data[prefix+'states'].T
                    l,r=data[prefix+'encoded'].T
                    counts.update(n=len(a),B_and_D=int(((b==1)&(d==1)).sum()),
                                  lidar_encoded=int(l.sum()),radar_encoded=int(r.sum()),
                                  both_zero_fill=int((~l&~r).sum()), A_unknown=int((a<0).sum()),
                                  C_unknown=int((c<0).sum()))
    return result


def num(x):
    return '—' if x is None else f'{x:.3f}'


def rate(n,k):
    return '—' if n==0 else f'{k}/{n} ({k/n:.1%})'


def empty_comparison(root):
    """Test A versus C only among LiDAR-empty regions; also hold Radar presence fixed."""
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault('OBS_OUT',str(root.resolve().parent))
    from analyze import held_out, load, metrics

    result={}
    for w in WEATHERS:
        data=load(root,w,'grid')
        a,_,c,d=data['states'].T
        eligible=(a==1)|(c==1)
        result[w]={}
        for name,mask in [('all',eligible),('radar_empty',eligible&(d==0)),('radar_present',eligible&(d==1))]:
            if not mask.any():
                continue
            y=(a[mask]==1).astype(int)
            scenes=data['scene'][mask]
            features={'distance_position':data['base'][:,1:],
                      **{k:data[k] for k in ('lidar','radar','concat','trunk','fused')}}
            result[w][name]=dict(n=len(y),A=int(y.sum()),C=int((1-y).sum()),scenes=len(set(scenes)),
                                  scores={k:metrics(y,held_out(x[mask],y,scenes,x[mask],scenes))
                                          for k,x in features.items()})
    return result


def run(root):
    source=root/'FINAL_REPORT.md'
    previous=source.read_text(encoding='utf-8')
    marker='## 实测要点'
    if marker in previous:
        raise ValueError('Report has already been annotated')
    a=json.loads((root/'D2-1.json').read_text())
    b=json.loads((root/'D2-2.json').read_text())
    q=quality(root)
    (root/'LABEL_QUALITY.json').write_text(json.dumps(q,indent=2)+'\n')
    empty=empty_comparison(root)
    (root/'EMPTY_A_VS_C.json').write_text(json.dumps(empty,indent=2)+'\n')
    lines=[marker,'',
           '### Q1 · 哪些状态在特征中保留','',
           '下表为各层留出预测的平均精度，括号内列出阳性数。数值受状态阳性比例影响，因此同时列出该比例与基础特征成绩。','',
           '| 天气 | 状态 | 阳性比例（数量） | 点数/距离 | LiDAR 柱 | Radar 柱 | 拼接 | 主干 | 融合后 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for w in WEATHERS:
        for state,out in a[w].items():
            vals=[num(out['scores'].get(k,{}).get('pooled',{}).get('pr_auc'))
                  for k in ('count_distance','lidar','radar','concat','trunk','fused')]
            lines.append(f"| {w} | {state} | {rate(out['labelled'],out['positives'])} | "+' | '.join(vals)+' |')
    lines+=['','A 与 C 额外比较仅保留 LiDAR 零点且具有 A 或 C 证据的区域，同时按 Radar 有点/无点分层。下表为 ROC-AUC，详细结果见 EMPTY_A_VS_C.json。','',
            '| 天气 | Radar 分层 | A/C 样本数 | 距离位置 | LiDAR 柱 | Radar 柱 | 拼接 | 主干 | 融合后 |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for w,subsets in empty.items():
        for group,row in subsets.items():
            values=[num(x['roc_auc']) for x in row['scores'].values()]
            lines.append(f"| {w} | {group} | {row['A']}/{row['C']} | "+' | '.join(values)+' |')
    lines+=['','### Q2 · 状态与检测错误的原始关联','',
            'A/B/C/D 可重叠，各行的有证据阳性和阴性是该状态自己的分母。Unknown 单列在 D2-2.json。','',
            '| 对象 | 天气 | 状态 | 阳性中的错误 | 阴性中的错误 | Unknown 中的错误 |',
            '|---|---|---|---:|---:|---:|']
    for kind, weather in b.items():
        for w in WEATHERS:
            for state,counts in weather[w]['coverage'].items():
                vals=[rate(counts[k]['n'],counts[k]['errors']) for k in ('1','0','-1')]
                lines.append(f'| {kind} | {w} | {state} | '+' | '.join(vals)+' |')
    lines+=['','### Q3/Q4 · 控制已有信息后的增量和跨天气','',
            '正差值代表加入观测状态后更能排序检测错误。下表按场景等权平均，括号给出 95% 场景区间。','',
            '| 对象 | 留出设置 | D − C | E −（C＋GSPR） | 强基线加状态 | 强基线另加 Clean 对照 |',
            '|---|---|---|---|---|---|']
    for kind,weather in b.items():
        for w,r in weather.items():
            vals=[]
            for k in ('D-C','E-C_GSPR','strong_live-strong','oracle_strong-strong_live'):
                d=r['delta'][k]
                text='—'
                if d['mean'] is not None:
                    text=f"{d['mean'][1]:+.4f}"
                    if d['ci95'] is not None:
                        lo,hi=d['ci95'][1]
                        text+=f' [{lo:+.4f}, {hi:+.4f}]'
                vals.append(text)
            lines.append(f'| {kind} | {w} | '+' | '.join(vals)+' |')
    lines+=['','### Q5 · 对融合缺陷的解释范围','',
            '本轮在主车固定窗口抽取激光雷达与雷达点数。A/C 射线只支持近水平截面，未知区域可能来自视场限制、无回波或未被观测。'
            'D 可从两类点数直接算出，加入状态后的差值可能来自对同一信息的重新表达。'
            'B 存在天气点替换和窗口边界误差，只用于已有 Clean 参考的离线分析。'
            '主干与融合层分辨率相同，均可接收邻域信息。两层探针差异还可能来自激活分布变化。'
            '此前 Radar 诊断改变了其他模型的雷达补偿，本轮从 F0 单次原始输出做观察，两者没有直接可比的因果量。','']
    where=previous.index('## 研究判断')
    source.write_text(previous[:where]+'\n'.join(lines)+'\n'+previous[where:],encoding='utf-8')
    p=root/'RESULT_HASHES.json'
    hashes=json.loads(p.read_text())
    for name in ('FINAL_REPORT.md','LABEL_QUALITY.json','EMPTY_A_VS_C.json'):
        hashes[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    hashes['interpret.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    p.write_text(json.dumps(hashes,indent=2)+'\n')


if __name__=='__main__':
    run(Path(sys.argv[1]))
