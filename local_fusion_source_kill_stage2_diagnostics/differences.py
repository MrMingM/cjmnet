"""Explain exact comparisons. No tolerance, rounding, or dropped fields."""
import json
import math
import re
from collections import Counter
from numbers import Real


def category(path):
    if 'mask_stats' in path:
        return 'ROI_MASK_METADATA'
    if re.search(r'(?:keep_key|selected_key|\.key)\[[0-5]\]',path):
        return 'DISCRETE_KEY'
    if re.search(r'(?:keep_key|selected_key|\.key)\[[67]\]',path) or re.search(r'\.focal_(?:before|after)\.(?:score|iou)$',path):
        return 'FOCAL_SCORE_OR_IOU'
    if re.search(r'\.focal_(?:before|after)\.detected$',path):
        return 'DETECTION_STATE'
    if '.cls_source' in path or '.reg_source' in path or '.expansion' in path or '.changed' in path:
        return 'SELECTED_ACTION'
    if '.proxy' in path or 'chosen_proxy' in path or 'best_proxy' in path:
        return 'PROXY'
    if '.group' in path or '.selected_group' in path or '.shortlisted' in path or '.allowed' in path:
        return 'PARTITION_OR_ELIGIBILITY'
    if '.matched_' in path or '.fp_' in path or '.was_shared_missed' in path:
        return 'DETECTION_STATE'
    return 'SCHEMA_OR_OTHER'


def exact_differences(actual, expected, limit=100):
    found=[]
    counts=Counter()
    total=0
    def add(path,left,right,reason):
        nonlocal total
        total+=1
        kind=category(path)
        counts[kind]+=1
        if len(found)>=limit:
            return
        row=dict(path=path,category=kind,reason=reason,actual=left,expected=right,
                 actual_type=type(left).__name__,expected_type=type(right).__name__)
        if isinstance(left,Real) and isinstance(right,Real) and all(math.isfinite(float(v)) for v in (left,right)):
            row['absolute_difference']=abs(float(left)-float(right))
        found.append(row)
    def walk(left,right,path):
        if isinstance(left,dict) and isinstance(right,dict):
            for key in sorted(set(left)|set(right),key=str):
                child=path+'.'+str(key)
                if key not in left:
                    add(child,None,right[key],'missing actual key')
                elif key not in right:
                    add(child,left[key],None,'extra actual key')
                else:
                    walk(left[key],right[key],child)
        elif isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
            if type(left)!=type(right):
                add(path,left,right,'sequence type differs')
            if len(left)!=len(right):
                add(path+'.length',len(left),len(right),'sequence length differs')
            for index,(a,b) in enumerate(zip(left,right)):
                walk(a,b,f'{path}[{index}]')
        elif left!=right:
            add(path,left,right,'exact value differs')
    walk(actual,expected,'trajectory')
    return dict(exact_equal=total==0,total_differences=total,categories=dict(counts),
                first_differences=found,comparison='exact; no tolerance or field omissions')


def render_summary(report):
    lines=['# K2 单帧复现诊断','','## 一句话结论','',report['conclusion'],'',
           f'Weather={report["weather"]}; mode={report["mode"]}; frame={report["frame"]}; '
           f'physical GPU={report["environment"]["ROCR_VISIBLE_DEVICES"]}。','',
           '这是故障诊断，未修改协议、复现容差、原结果或 worker 状态，不生成科学 PASS/FAIL。','']
    reference=report.get('s0_shared_reference')
    if reference:
        lines += [f'Shared prediction 与原 S0 快照的 SHA256 是否一致：{reference["exact_shared_hash_equal"]}。','']
    for label,comparison in report.get('comparisons',{}).items():
        lines += ['## '+label,'',f'完全一致：{comparison["exact_equal"]}；差异字段数：{comparison["total_differences"]}。','']
        for row in comparison['first_differences'][:12]:
            actual=json.dumps(row['actual'],ensure_ascii=False,default=str)
            expected=json.dumps(row['expected'],ensure_ascii=False,default=str)
            if len(actual)>300: actual=actual[:300]+'…'
            if len(expected)>300: expected=expected[:300]+'…'
            lines += ['- '+row['path']+' ('+row['category']+')',
                      '  actual: `'+actual+'`', '  expected: `'+expected+'`']
            if 'absolute_difference' in row:
                lines.append('  abs diff: `'+str(row['absolute_difference'])+'`')
        lines.append('')
    if report.get('interleaved_error'):
        lines += ['## 原 K2 调用的错误','',report['interleaved_error'],'']
    lines += ['完整精确差异见 DIAGNOSIS.json；三个路径的原始 records/audits 单独保存。',
              '本工具不把“小浮点差异”自动当成可接受，也不更改历史 gate。']
    return '\n'.join(lines)+'\n'
