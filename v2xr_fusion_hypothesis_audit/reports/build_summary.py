"""Combine the three diagnostics without equating evidence with deployment gains."""
from shared.input_audit import OUT, WEATHERS, BASE, atomic_json, identity, read_json, sha256


def summarize():
    paths={'V1':'V1/CANDIDATE_COVERAGE.json','V2':'V2/REGION_MATCHING.json',
           'V3_box':'V3/BOX_ERROR_CORRELATIONS.json','V3_feature':'V3/FEATURE_CORRELATIONS.json',
           'V3_prediction':'V3/PREDICTABILITY.json'}
    data={k:read_json(OUT/v) for k,v in paths.items()}
    base=read_json(BASE/'summary.json')
    v1=data['V1']['metrics']
    v2=data['V2']['metrics']
    v3=data['V3_prediction']
    lines=['# 冻结 P0 的跨车融合研究判断','',
           'P0 仍使用原权重、原输入和原后处理。以下为零神经网络训练的诊断。','',
           '| 天气 | P0 F AP70 | P0 命中目标的单车最终框覆盖 | 阈值后 NMS 前 | Top200 | 最终框并集实际召回 |',
           '|---|---:|---:|---:|---:|---:|']
    for w in WEATHERS:
        ap=base['results'][w]['AP']['F']['ap70']
        vals=[v1[f'{w}/P0_success/{name}/0.7']['value'] for name in ('final','pre_nms','top200')]
        recall=v1[f'{w}/official_final_union/0.7']['value']
        lines.append(f"| {w} | {ap:.3%} | "+' | '.join(f'{x:.2%}' for x in (*vals,recall))+' |')
    lines+=['','| 假设 | 支持证据 | 反对证据或局限 | 证据可靠性 | 研究决策 |',
             '|---|---|---|---|---|']
    coverage=', '.join(f"{w} {v1[f'{w}/P0_success/top200/0.7']['value']:.1%}" for w in WEATHERS)
    lines.append(f'| C1 候选覆盖 | Top200 覆盖 {coverage} | 覆盖是理想上限，受候选数量和误检代价约束 | 场景重采样区间见 V1 | 有候选的目标可继续分析，缺候选目标存在结构性上限 |')
    geom=[]
    for w in ('fog','snow'):
        for scale in range(2):
            r=v2.get(f'{w}/{scale}/fixed_blocks/p0_missed',{}).get('scene_macro_auprc_delta',{})
            geom.append((f'{w}/{scale}',r))
    conditional=[(w,v3.get(w,{}).get('metrics',{})) for w in ('fog','snow')]
    improves=lambda row: row.get('interval95') is not None and row['interval95'][1]<0 and row['scenes']>=3
    usable=all(improves(m.get(f'conditional_minus_{base}_{measure}',{}))
               for _,m in conditional for base in ('fixed','independent') for measure in ('product_mse','oracle_xy_mse'))
    structured=all(r.get('interval95') is not None and r['interval95'][0]>0 and r['scenes']>=3 for _,r in geom)
    lines.append(f'| C1 条件相关误差 | 留出场景比较预测误差和理想关联定位误差 | 关联缺失和共同模型偏差须结合 V3 | 仅当前验证集 | {"继续核验候选代价" if usable else "证据不足"} |')
    lines.append(f'| C2 结构额外信息 | Fog/Snow 未检测区域的留出场景对照 | 需超过位置和局部特征，真值辅助区域单列 | 仅当前验证集 | {"继续核验可用对应" if structured else "证据不足"} |')
    lines+=['','## 判断',
             '两者都值得进一步验证。' if usable and structured else
             '优先验证 C1 在已有候选上的价值。' if usable else
             '优先验证 C2，保留坐标和局部相似度对照。' if structured else
             '当前证据不足以优先推进 C1 或 C2。',
             'C1 的误差估计即使有效，也只适用于已有候选。C2 需区分区域判别增益和实际检测收益。',
             '本验证集已用于历史研究筛选。场景留出降低本轮额外过拟合，最终结论需要独立数据。']
    report='\n'.join(lines)+'\n'
    (OUT/'FINAL_DECISION.md').write_text(report,encoding='utf-8')
    atomic_json(OUT/'FINAL_RESULTS.json',dict(status='complete',contract=identity(read_json(OUT/'CONTRACT.json')),
        reports={k:dict(path=v,sha256=sha256(OUT/v)) for k,v in paths.items()},
        c1_conditional_supported=usable,c2_structural_supported=structured))
    (OUT/'AI_CONTEXT_APPEND.md').write_text(report+f'\n服务器目录 `{OUT}`。\n',encoding='utf-8')
