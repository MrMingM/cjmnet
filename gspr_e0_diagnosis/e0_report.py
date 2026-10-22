"""Evidence-based E0 reports; metrics are read only from completed server artifacts."""
import numpy as np
from e0_common import HERE, WEATHERS, ARMS, atomic, read, sha
from e0_analysis import arm_folder, interventions, reliability_summary


def degradation(output):
    root = output / 'stages/paired'
    summary = read(root / 'E0_POINT_DEGRADATION.json')
    result = {}
    files = []
    for index in range(829):
        path = root / 'frames' / f'{index:06d}.json'
        frame = read(path)
        files.append(dict(path=str(path), sha256=sha(path)))
        for weather in WEATHERS:
            row = frame['conditions'][weather]
            value = result.setdefault(weather, dict(raw_points=0, retained_real_points=0,
                raw_range_counts=[0]*4, official_voxels=0, occupied_pillars=0,
                few_point_pillars=0, high_density_pillars=0, synthetic_fallback_cavs=0,
                author_color_zero_points=0, author_color_nonzero_points=0,
                native_missing_clean_pillars=0, native_clean_pillars=0,
                clean_occupied_cells=0, missing_clean_cells=0, additional_weather_cells=0,
                fixed_targets=0, zero_retained_point_targets=0, fewer_points_than_clean_targets=0,
                nearest_proxy={str(t): dict(clean_without_near_return=0, weather_without_near_clean=0,
                    clean_points=0, weather_points=0) for t in (.05, .1, .2)}))
            for agent in row['agents']:
                for name in ('raw_points', 'official_voxels', 'occupied_pillars', 'few_point_pillars', 'high_density_pillars', 'author_color_zero_points', 'author_color_nonzero_points'):
                    value[name] += agent[name]
                value['native_missing_clean_pillars'] += agent['native_voxel_coverage']['missing_clean_cells']
                value['native_clean_pillars'] += agent['native_voxel_coverage']['clean_occupied']
                value['raw_range_counts'] = [a+b for a, b in zip(value['raw_range_counts'], agent['raw_range_counts'])]
                value['synthetic_fallback_cavs'] += agent['synthetic_lidar_fallback']
                for t, stats in agent['retained_nearest_proxy'].items():
                    for field, count in stats.items():
                        value['nearest_proxy'][t][field] += count
            value['retained_real_points'] += row['total_retained_real_points']
            value['clean_occupied_cells'] += row['union_coverage']['clean_occupied']
            value['missing_clean_cells'] += row['union_coverage']['missing_clean_cells']
            value['additional_weather_cells'] += row['union_coverage']['additional_weather_cells']
            value['fixed_targets'] += len(row['targets'])
            value['zero_retained_point_targets'] += sum(t['points'] == 0 for t in row['targets'])
            value['fewer_points_than_clean_targets'] += sum(t['points'] < t['clean_points'] for t in row['targets'])
    return dict(**summary, aggregate=result, frame_evidence=files,
        aggregation='point returns sum over CAV observations; coverage sums per-frame common-ego XY unions; targets use fixed Clean identities',
        information_limit='Empty/low coverage is observed geometric loss; unmeasured appearance/occlusion/feature information prevents a total-information claim')


def draw_figures(output, folder, reliability, intervention, point):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = folder / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for weather in WEATHERS:
        for ax, variable in zip(axes, ('r', 'u')):
            stat = reliability['weather'][weather]['distributions']['all_real'][variable]
            h = np.asarray(stat['histogram'], dtype=float)
            x = (np.arange(len(h))+.5)/len(h)
            ax.plot(x, h/max(h.sum(), 1), label=weather)
            ax.set_xlabel('reliability' if variable == 'r' else 'uncertainty')
            ax.set_ylabel('fraction of real valid voxel slots')
            ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'reliability_uncertainty.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, weather in zip(axes, WEATHERS):
        for variable in ('r', 'u'):
            groups = reliability['weather'][weather]['distributions']
            y = [groups['sensor_range/'+str(i)][variable]['mean'] for i in range(4)]
            ax.plot(range(4), [np.nan if v is None else v for v in y], marker='o', label=variable)
        ax.set_xticks(range(4), ['0-20', '20-40', '40-70', '>=70'])
        ax.set_xlabel('sensor distance (m)')
        ax.set_title(weather)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'range_groups.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(3)
    for i, arm in enumerate(ARMS):
        ax.bar(x+(i-1.5)*.18, [intervention['results'][arm][w]['AP']['F']['ap70'] for w in WEATHERS], width=.18, label=arm)
    ax.set_xticks(x, WEATHERS)
    ax.set_ylabel('fixed Clean GT F AP70')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'interventions_F_AP70.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(3), [point['aggregate'][w]['missing_clean_cells']/max(point['aggregate'][w]['clean_occupied_cells'], 1) for w in WEATHERS])
    ax.set_xticks(range(3), WEATHERS)
    ax.set_ylabel('missing Clean ego XY cells (post-FOV, pre-voxel)')
    fig.tight_layout()
    fig.savefig(figures / 'paired_missing_coverage.png', dpi=180)
    plt.close(fig)
    inventory = read(output / 'stages/preflight/inventory.json')
    for index in inventory['sample_indices']:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
        for column, weather in enumerate(WEATHERS):
            observation = read(arm_folder(output, 'R0') / 'observations' / weather / f'{index:06d}.json')
            sample = observation['statistics']['sample']
            if sha(sample['path']) != sample['sha256']:
                raise ValueError('Spatial heat-map sample hash differs')
            with np.load(sample['path'], allow_pickle=False) as data:
                for row, variable in enumerate(('r', 'u')):
                    ax = axes[row, column]
                    scatter = ax.scatter(data['xyz'][:, 0], data['xyz'][:, 1], c=data[variable], s=.8, vmin=0, vmax=1, cmap='viridis', rasterized=True)
                    for box in data['fixed_corners']:
                        polygon = box[[0, 1, 2, 3, 0], :2]
                        ax.plot(polygon[:, 0], polygon[:, 1], color='red', linewidth=.5)
                    ax.set_title(weather+' '+variable)
                    ax.set_aspect('equal')
                    ax.set_xlabel('ego x (m)')
                    ax.set_ylabel('ego y (m)')
                    fig.colorbar(scatter, ax=ax)
        fig.suptitle('Paired sampled real model slots; fixed Clean GT red; frame '+str(index))
        fig.tight_layout()
        fig.savefig(figures / f'spatial_{index:06d}.png', dpi=180)
        plt.close(fig)
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(figures.glob('*.png'))}


def conclusions(intervention, reliability, point, errors, baseline):
    delta = intervention['delta_from_R0_pp']
    margin = read(HERE / 'protocol.json')['effect_description_margin_pp']
    fog = {a: delta[a]['fog']['F'] for a in ('R1', 'R2', 'R3')}
    snow = {a: delta[a]['snow']['F'] for a in ('R1', 'R2', 'R3')}
    spatial = all(fog[a]['ap70'] < -margin for a in ('R2', 'R3'))
    close = all(abs(fog[a]['ap70']) <= margin for a in ('R2', 'R3'))
    spatial_text = ('Fog 的 R2、R3 均较 R0 下降超过预设描述区间，支持当前空间对应关系参与收益。'
        if spatial else 'Fog 的 R2、R3 均处于 R0 的预设描述区间内，当前干预尚未建立空间对应关系的必要性。'
        if close else 'Fog 的 R2、R3 反应不同，空间对应关系的作用需要结合具体干预解释。')
    r = reliability['weather']['snow']['distributions']
    retained = r['near_clean_proxy']['r']
    suspicious = r['unmatched_clean_proxy']['r']
    author_zero, author_nonzero = r['author_color/zero']['r'], r['author_color/nonzero']['r']
    info = point['aggregate']['snow']
    stats = errors['weather']['snow']['counts']['0.7']
    c_ap = baseline['C']['results']['snow']['AP']['F']['ap70']
    d_ap = baseline['D']['results']['snow']['AP']['F']['ap70']
    f_delta = 100*(baseline['D']['results']['fog']['AP']['F']['ap70']-baseline['C']['results']['fog']['AP']['F']['ap70'])
    mean_text = lambda s: '无样本' if s['mean'] is None else f'{s["mean"]:.6f}（n={s["n"]}）'
    lines = ['# E0 研究决策报告', '', '以下结论只使用本轮完整服务器产物。相关性统计、单种子和一次固定置换均不构成因果证明。', '',
        '## 简单版', '',
        f'1. **是否利用空间可靠性差异？** {spatial_text} “接近”预先定义为 F AP70 差值在 ±{margin} pp，仅用于描述，不代表统计等价。',
        f'2. **Fog 为什么改善？** 原 C/D 的固定 GT F AP70 差值为 {f_delta:+.6f} pp。R1/R2/R3 的同权重差值分别为 '+
            '、'.join(f'{a} {fog[a]["ap70"]:+.6f} pp' for a in fog)+'。这些结果约束机制解释；物理噪声识别、均值缩放、特征适配的独立贡献仍无法完全确定。',
        '3. **Snow 为什么没有明显改善？** 见下方五种解释的证据表。'+
            f'当前保留下来的近邻代理点平均 r={mean_text(retained)}；未找到 Clean 近邻的代理点平均 r={mean_text(suspicious)}。二者没有真实点级标签保证。',
        '4. **是否做 E1 混合天气训练？** 当前训练已含 Clean/Fog。下一阶段可在独立训练开发场景做最小 C/D×有无 Snow 的匹配对照；先保留原 3/7 Fog 组作为锚点，再单独比较新增 Snow 的配比。不要同时修改结构、概率与损失。是否添加监督需先解决标签真实性。',
        '5. **是否做 E2 配对监督？** 建议先验证覆盖度与目标相关可靠性哪一个更能解释失败。'+
            f'Snow 有 {info["zero_retained_point_targets"]}/{info["fixed_targets"]} 个固定目标没有真实保留点；这为覆盖度监督提供可量化动机。当前近邻代理不足以决定真实噪声标签。',
        '6. **是否做 E3 引导融合？** '+('当前干预支持保留空间质量信息开展离线可用性验证。' if spatial else '当前证据不足以直接开展可靠性乘注意力的模型开发。')+
            '可靠性有变化或与代理相关，还需要证明它能预测来源/区域的检测收益；本轮没有融合干预或来源因果证据。',
        '7. **E4 研究什么？** 优先围绕已观察的覆盖缺失和目标定位/召回变化做最小验证。归一化卷积、置信度传播需要可信质量量；结构扩容需要表达不足证据；多任务梯度冲突优化需要另行测量训练梯度。本轮均未验证，不同时启动这些方案。', '',
        '## 进一步理解', '',
        '| Snow 候选解释 | 支持证据 | 反证或约束 | 尚不能确定 |',
        '|---|---|---|---|',
        '| 训练没有 Snow | 已核对训练配置、契约和实际固定采样源码，天气路径仅 Clean/Fog | Snow 上仍保留一定检测能力；域外并不必然无收益 | 需要匹配 Snow 训练对照才能衡量贡献 |',
        '| 有用点被压低 | 同 D 权重 R1 的 F AP70 差值 '+f'{snow["R1"]["ap70"]:+.6f} pp；R2/R3 '+f'{snow["R2"]["ap70"]:+.6f}/{snow["R3"]["ap70"]:+.6f} pp'+
            '，结合目标内 r 和挽救/丢失计数 | R1 同时改变幅值；近邻匹配不保证点真实性 | 无法将全部差值归因于错误抑制 |',
        '| 有效信息缺失 | '+f'缺失 Clean 占用格 {info["missing_clean_cells"]}/{info["clean_occupied_cells"]}；零保留点目标 {info["zero_retained_point_targets"]}'+
            '；逐目标记录链接检测状态 | 占用格与点数只是几何代理，仍可能有其他车辆或上下文信息 | 不量化全部语义信息，也不宣称不可恢复上界 |',
        '| 可靠性表达不足 | r/u 分布、分支与密度条件相关性见统计报告；需结合 R2/R3 判断 | 无逐点真值且没有校准证据 | 不能从弱代理相关单独推断结构不足 |',
        '| 后续融合/检测未利用改善 | 源码确认质量变量未显式进入融合；最终框分数/定位变化已记录 | 加权特征仍会影响融合；未显式传变量不等于未利用 | 需要固定前端的融合或候选观测实验 |', '',
        f'Snow C/D 的 F AP70={c_ap:.9f}/{d_ap:.9f}；固定 GT 一对一 TP={stats["C_tp"]}/{stats["D_tp"]}，FP={stats["C_fp"]}/{stats["D_fp"]}。',
        'AP 还依赖帧内分数排序与累计精度曲线，TP 总数与 AP 可以方向不同。逐目标匹配、分数和定位差值提供最终输出层证据；本轮未对完整 C/D 的 NMS 前候选建立对应关系，不能完成内部因果分解。', '',
        '| 诊断证据 | 支持的研究方向 | 不支持的方向 | 后续最小验证 |',
        '|---|---|---|---|',
        '| '+spatial_text+' | 保留均值的空间权重对照或质量可用性检查 | 直接宣称已识别物理噪声 | 独立开发场景、多置换种子复核 |',
        '| 实测配对占用与目标点数缺失 | 区域覆盖度、配对目标支持监督 | 单纯重加权能恢复所有缺失信息 | 覆盖统计预测同一目标漏检，控制距离/车辆数 |',
        '| 目标挽救/丢失与最终框定位/分数差异 | 针对有真实候选支持的检测瓶颈检查 | 把最终框数量当召回或 NMS 因果证据 | 同权重插入 NMS 前观察器，不调阈值 |',
        '| 无训练梯度测量与融合干预 | 暂停复杂优化器/融合结构决策 | 自动实施 E1–E4 或同时堆模块 | 冻结 E0 证据后另行确定一个最小对照 |', '',
        '事实：记录的配置、源码路径、缓存、匹配和数值。支持的解释：由指定干预和条件统计共同约束的有限机制解释。假设：训练域、表达能力、后续利用等尚未单独操纵的原因。',
        '所有分组样本量、空组和低样本标记保留；近邻阈值敏感性为 0.05/0.1/0.2m；单次 R3 置换和 ROCm 数值容差限制须随报告披露。']
    lines += ['', '官方加载器还用绿色通道等于0的点构造 de_lidar_np。当前 C/D 没有启用该分支，也没有把该通道用于 GSPR 点级监督。',
        f'Snow 中能唯一映射到该掩码零值组/非零组的平均 r 分别为 {mean_text(author_zero)} / {mean_text(author_nonzero)}。'+
        '映射歧义、未映射、样本量和控制密度/距离/强度后的关联见统计报告。它是已有作者筛选掩码的额外观测，生成语义尚未验证，不能直接当成物理噪声真值。']
    lines += ['E2 的最小前置验证应包含核实这条绿色掩码的生成语义；得到验证后再判断点真实性监督的价值。它与区域覆盖度、目标相关可靠性属于不同的监督对象，不应互相替代。']
    return '\n'.join(lines)+'\n'


def report(cli, folder, request_id):
    intervention = interventions(cli.output)
    reliability = reliability_summary(cli.output)
    point = degradation(cli.output)
    errors = read(cli.output / 'stages/error_analysis/E0_ERROR_ANALYSIS.json')
    baseline = {v: read(cli.output / 'stages/baseline' / v / 'summary.json') for v in ('C', 'D')}
    protocol = read(cli.output / 'stages/preflight/E0_PROTOCOL.json')
    inventory = read(cli.output / 'stages/preflight/inventory.json')
    # Check original inputs/checkpoint bytes again at the end; no silent historical mutation.
    from e0_runtime import setup, model_inventory, cache_audit
    _, evidence, frozen, audit, _, _ = setup(cli, gpu=False)
    actual = model_inventory(cli, audit)
    if actual != inventory['models']:
        raise ValueError('Original model/evaluation changed during E0')
    for model in actual.values():
        cache_audit(model, evidence, frozen)
    completed = folder / 'E0_FINAL_RESULTS.json'
    if completed.exists():
        final = read(completed)
        if final['status'] != 'complete' or final['request'] != request_id:
            raise ValueError('Existing final report identity differs')
        for name, digest in final['evidence'].items():
            if sha(folder / name) != digest:
                raise ValueError('Already published final evidence changed')
        for name, digest in final['figures'].items():
            if sha(folder / name) != digest:
                raise ValueError('Already published final figure changed')
        return
    figures = draw_figures(cli.output, folder, reliability, intervention, point)
    conclusions_text = conclusions(intervention, reliability, point, errors, baseline)
    targets = dict(E0_PROTOCOL=protocol, E0_RELIABILITY_STATS=reliability,
        E0_POINT_DEGRADATION=point, E0_INTERVENTIONS=intervention, E0_ERROR_ANALYSIS=errors)
    for name, value in targets.items():
        atomic(folder / (name+'.json'), value, immutable=True)
    lines = ['# GSPR E0 机制诊断结果', '', '所有数值来自完整服务器产物；只执行推理、分析及同权重干预。', '',
        '| Arm | Weather | Protocol | AP30 | AP50 | AP70 | ΔAP30 pp | ΔAP50 pp | ΔAP70 pp | GT | None |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for weather in WEATHERS:
            row = intervention['results'][arm][weather]
            for p in ('O', 'O*', 'F'):
                ap, delta = row['AP'][p], intervention['delta_from_R0_pp'][arm][weather][p]
                lines.append(f'| {arm} | {weather} | {p} | '+ ' | '.join(f'{ap[k]:.9f}' for k in ('ap30', 'ap50', 'ap70'))+' | '+
                    ' | '.join(f'{delta[k]:+.6f}' for k in ('ap30', 'ap50', 'ap70'))+f' | {row["counted_gt"][p]} | {row["counts"]["none_frames"]} |')
    lines += ['', 'TP/FP/FN 的各 IoU 阈值分母与计数见 E0_INTERVENTIONS.json；O 的 None 帧按冻结协议跳过，F 保留固定 Clean GT。',
        '逐帧 O/O*/F 匹配：各 arm/all_protocol_matches.jsonl.gz；C/D 逐目标分析：error_analysis/objects.jsonl.gz。', '',
        '## 可靠性统计', '', '| Weather | mean r | std r | mean u | std u | real slots |', '|---|---:|---:|---:|---:|---:|']
    for weather in WEATHERS:
        stats = reliability['weather'][weather]['distributions']['all_real']
        values = [stats[v][k] for v in ('r', 'u') for k in ('mean', 'std')]
        lines.append('| '+weather+' | '+' | '.join('NA' if x is None else f'{x:.6f}' for x in values)+f' | {stats["r"]["n"]} |')
    lines += ['', '可靠性与不确定性未经点级真实概率校准。分位数是有记录误差宽度的直方图近似，均值/方差使用全体有效槽位。',
        '距离、点数、CAV、框内外、近邻代理、分支相关与密度条件关联见 E0_RELIABILITY_STATS.json。',
        '空间图仅展示预设样本的确定性子采样；全量统计未使用该子采样。', '',
        '![Reliability and uncertainty](figures/reliability_uncertainty.png)',
        '![Interventions](figures/interventions_F_AP70.png)',
        '![Paired coverage](figures/paired_missing_coverage.png)', '',
        '限制：单训练种子、已反复使用的 validation、固定一次置换、空间近邻代理与 ROCm 数值误差。没有正式 test 或训练结果。']
    for arm in ('R1', 'R2', 'R3'):
        for weather in WEATHERS:
            d = intervention['delta_from_R0_pp'][arm][weather]['F']
            if d['ap70'] > 0 and (d['ap30'] < -.5 or d['ap50'] < -.5):
                lines.append(f'观察到权衡：{arm}/{weather} F AP70 上升，同时 AP30 或 AP50 下降超过 0.5 pp；不能概括为整体改善。')
    (folder / 'E0_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    (folder / 'E0_CONCLUSIONS.md').write_text(conclusions_text, encoding='utf-8', newline='\n')
    execution = dict(status='complete', request=request_id, no_test=True, no_training=True,
        original_models_unchanged=True, preflight=inventory, figures=figures,
        stage_receipts={str(p.relative_to(cli.output)): sha(p) for p in (cli.output / 'stages').rglob('complete.json')
                        if folder not in p.parents},
        source_request_sha256=sha(cli.output / 'request.json'),
        failure_history=[dict(path=str(p), sha256=sha(p), error=read(p)) for p in sorted((cli.output / 'failures').glob('*.json'))])
    audit_path = folder / 'E0_EXECUTION_AUDIT.json'
    if audit_path.exists():
        existing = read(audit_path)
        if {k: v for k, v in existing.items() if k != 'failure_history'} != {k: v for k, v in execution.items() if k != 'failure_history'}:
            raise ValueError('Partially published execution audit identity differs')
        # Retain the exact failure-history snapshot taken when this audit was first published.
    else:
        atomic(audit_path, execution, immutable=True)
    final = dict(status='complete', stage='E0', request=request_id, no_test=True, no_training=True,
        results=intervention['results'], delta_from_R0_pp=intervention['delta_from_R0_pp'],
        detection_counts=intervention['detection_counts'], original_baselines=baseline,
        checkpoint_sha256=intervention['checkpoint_sha256'],
        evidence={p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'E0_FINAL_RESULTS.json'},
        figures=figures, next_experiments_automatically_started=False)
    atomic(folder / 'E0_FINAL_RESULTS.json', final, immutable=True)
