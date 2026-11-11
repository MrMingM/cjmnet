"""Combine three completed diagnostics with pinned artifact hashes."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/'common'))
from diag_bootstrap import OUT,atomic_json,identity,read_json,sha256


def main():
    contract=identity(read_json(OUT/'CONTRACT.json'))
    rows={}
    for group in ('D1','D2','D3'):
        root=OUT/group
        row=read_json(root/'FINAL_RESULTS.json')
        if (row['status']!='complete' or row['contract']!=contract
                or any(sha256(root/p)!=digest for p,digest in row['artifacts'].items())):
            raise ValueError(f'Incomplete or changed result for {group}')
        rows[group]=row
    lines=['# P0 固定权重融合机制诊断','',
           '| 组别 | 物理卡 | 判断 | 对应假设 |','|---|---:|---|---|']
    for group,row in rows.items():
        r=row['finding']
        lines.append(f'| {group} | {row["device"]["physical_gpu"]} | {r["label"]} | {r["question"]} |')
    lines+=['','[D1 逐组分析](D1/MECHANISM_ANALYSIS.md) · [D2 逐组分析](D2/MECHANISM_ANALYSIS.md) · '
            '[D3 逐组分析](D3/MECHANISM_ANALYSIS.md)','','## 后续研究问题','']
    suggestions={'D1':'检验邻车扰动的主导分量如何随天气、车辆数和距离变化，以及减少该分量能否保留正常协同收益。',
                 'D2':'复验明确受损尺度的贡献是否跨场景稳定，并区分该尺度的独立作用与多尺度交互。',
                 'D3':'进一步定位影响雷达检出贡献的非加性交互在哪一级形成，并用受控模态干预复验。'}
    supported=[key for key,row in rows.items() if row['finding']['label']=='有支持证据']
    lines += [f'- {key}。{suggestions[key]}' for key in supported] if supported else [
        '三组证据均未达到预设标准，本轮不选定新的融合方向。']
    summary=OUT/'SUMMARY.md'
    temp=summary.with_suffix('.tmp')
    temp.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    temp.replace(summary)
    atomic_json(OUT/'FINAL_RESULTS.json',dict(status='complete',contract=contract,groups=rows))
    print('ALL THREE DIAGNOSTICS COMPLETE',flush=True)


if __name__=='__main__':
    main()
