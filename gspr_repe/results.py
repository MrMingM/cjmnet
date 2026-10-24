"""Computed reports; no invented measurements or automatic performance gates."""
import csv
import json
from auditlib import atomic_json, complete, read_json, sha256
VARIANTS = {k:k for k in ('B0','B1','B2','B3','B4')}


def write_evaluation(output, summary):
    lines=['# Official evaluation','', '| Weather | GT protocol | AP30 | AP50 | AP70 |','|---|---|---:|---:|---:|']
    for weather,row in summary['results'].items():
        for protocol,ap in row['AP'].items():
            lines.append('| '+weather+' | '+protocol+' | '+' | '.join(str(ap[k]) for k in ('ap30','ap50','ap70'))+' |')
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def stage1(root):
    dest=root/'stage1'; dest.mkdir(exist_ok=True)
    tests=complete(root/'self_test/summary.json')
    evidence={}
    for variant in VARIANTS:
        folder=root/'smoke'/variant
        checkpoint=complete(folder/'smoke_checkpoint.json')
        if sha256(checkpoint['checkpoint'])!=checkpoint['checkpoint_sha256']:
            raise ValueError('Smoke checkpoint changed')
        summary=complete(folder/'evaluation_smoke12/summary.json')
        if summary['frame_count']!=12 or summary['checkpoint_sha256']!=checkpoint['checkpoint_sha256']:
            raise ValueError('Smoke inference/checkpoint identity mismatch')
        history=read_json(folder/'history.json')['epochs']
        if not history or not history[-1]['repe']:
            raise ValueError('No real optimization/diagnostic records')
        evidence[variant]=dict(checkpoint=checkpoint,history=history,evaluation=summary)
    atomic_json(dest/'acceptance.json',dict(status='complete',tests=tests,smoke=evidence,stage2_allowed=True))
    lines=['# 第一阶段验收','', '服务器结构测试、真实数据小规模训练、验证和推理全部完成。',
        '参数更新、严格重载和关闭模块等价性见 self_test/summary.json。',
        '耗时为训练与验证合计；host forward time 尚未同步设备，不能当作设备推理延迟。','',
        '| 模型 | 参数量 | 小训练耗时/秒 | 峰值显存/字节 |','|---|---:|---:|---:|']
    for v,r in evidence.items():
        e=r['history'][-1]
        lines.append(f"| {v} | {e['parameter_counts']['total']} | {e['duration_seconds']} | {e['peak_memory_bytes']} |")
        lines+=['',v+' 最后一次模块统计：`'+json.dumps(e['repe'][-1],ensure_ascii=False)+'`']
    (dest/'STAGE1_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def final(root):
    report=root/'reports'; report.mkdir(exist_ok=True)
    found={}; rows=[]; histories={}; failures=[]
    for v in VARIANTS:
        from evaluation import selection
        chosen=selection(root/v)
        summary=complete(root/v/'final_evaluation/summary.json')
        if not summary['full_validation'] or summary['frame_count']!=829 or summary['checkpoint_sha256']!=chosen['checkpoint_sha256']:
            raise ValueError('Incomplete formal result: '+v)
        for record in read_json(root/v/'final_evaluation/prediction_manifest.json')['files']:
            if sha256(record['path'])!=record['sha256']:
                raise ValueError('Changed prediction cache')
        found[v]=summary
        histories[v]=read_json(root/v/'history.json')
        for weather,result in summary['results'].items():
            for protocol,ap in result['AP'].items():
                rows.append(dict(model=v,weather=weather,protocol=protocol,**ap))
        frame_file=root/v/'final_evaluation/per_frame_metrics.jsonl'
        for line in frame_file.read_text(encoding='utf-8').splitlines():
            frame=json.loads(line); metrics=frame['protocols']['F']['thresholds']['0.7']
            failures.append(dict(model=v,weather=frame['condition'],frame=frame['frame_key'],tp=metrics['tp'],fp=metrics['fp'],fn=metrics['fn']))
    if len({x['frozen_d1c_gt_identity'] for x in found.values()})!=1:
        raise ValueError('GT identity differs across ablations')
    # Audit actual sampling rather than assume matched draws from seeds.
    sampling={}
    for v in VARIANTS:
        sampling[v]=[r['trace_sha256'] for r in histories[v]['epochs']]
    paired=all(sampling[v]==sampling['B0'] for v in VARIANTS)
    contrasts={}
    for a,b in (('B1','B0'),('B2','B0'),('B3','B2'),('B4','B3'),('B4','B0')):
        contrasts[a+'-'+b]={w:100*(found[a]['results'][w]['AP']['F']['ap70']-found[b]['results'][w]['AP']['F']['ap70']) for w in ('clean','fog','snow')}
    from diagnostics import summarize
    stratified = summarize(root, VARIANTS)
    data=dict(status='complete',stratified_detection=stratified,results=found,history=histories,fixed_GT_AP70_delta_pp=contrasts,sampling_trace_equal=paired,
        limitations=['single seed; no statistical significance claim','B3-B2 combines A and B; cannot isolate B causal contribution',
        'Q uses detached GSPR proxy; S uses aligned training Clean support with distance-bin fallback; neither is scene truth',
        'C conservatively enhances occupied cells only; fully empty target completion is not claimed',
        'per-target Q/S causal attribution requires further intervention experiments'])
    atomic_json(report/'STAGE2_RESULTS.json',data)
    with (report/'ABLATION_TABLE.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines=['# 正式对照结果','', '全部实验串行完成；负收益保留。固定 Clean GT AP70 差值（百分点）：','',
        '| 对照 | Clean | Fog | Snow |','|---|---:|---:|---:|']
    for name,values in contrasts.items():
        lines.append('| '+name+' | '+' | '.join(f'{values[w]:+.6f}' for w in ('clean','fog','snow'))+' |')
    lines+=['','采样轨迹完全一致：'+str(paired),'', 'O/O*/F 全部 AP 与匹配数量见 STAGE2_RESULTS.json。',
        '单种子结果只描述本轮表现，不能判断统计显著。']
    (report/'STAGE2_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    worst=sorted(failures,key=lambda x:x['fn']+x['fp'],reverse=True)[:100]
    (report/'FAILURE_CASES.md').write_text('# 高漏检/误检帧\n\n```json\n'+json.dumps(worst,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    assessment=lines+['','A 的独立增益参考 B1−B0；完整约束效果参考 B4−B3。',
        'B3−B2 同时引入 A/B，不能据此认定 B 单独有效，也不能给三个模块贡献排名。',
        'Snow 是否改善按上表固定 GT 差值判断；正确匹配、漏检、误检的原始计数全部保留。',
        'PointASNL 已有自适应加权与残差；SAFDNet 已有特征扩散；本轮不能将这些作为原创。',
        'TransBridge 使用更强的密集补全监督，本轮 Q/S 代理不能声称复现其补全。',
        '下一轮优先：训练内 Clean/Fog 配对覆盖监督；新增 A+B 无增强对照以单独检验 B。',
        '', '尚未实现的分析与限制：']+data['limitations']
    (report/'FINAL_RESEARCH_ASSESSMENT.md').write_text('\n'.join(assessment)+'\n',encoding='utf-8')
