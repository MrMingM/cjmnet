"""Automated section 42 verdicts with exact triggered thresholds and raw counts."""
from pathlib import Path
import numpy as np
from .common import (ADVERSE, FAMILIES, WEATHERS, atomic_json, atomic_text, jsonlines,
                     now, read_json, settings)
from .bootstrap import ap70, k1_bootstrap
from .verdicts import combine, final_verdict, k1_family


def condition_frames(folder, condition):
    from .k1_source_specificity import condition_path
    path = condition_path(folder, condition['weather'], condition['family'], condition['seed'], condition['mode'])
    return list(jsonlines(path.parent / condition['stats_file']))


def k1_report(folder, protocol, conditions, calculate_ap):
    paired_baselines = {}
    def table(family, seed):
        result = {}
        for weather in ADVERSE:
            rows = {}
            for mode in ('same', 'task'):
                detail = condition_frames(folder, conditions[f'{weather}/{family}/{seed}/{mode}'])
                fingerprints = [(r['shared'], r['shared_prediction_sha256'],r['model_input_sha256'],r['source_count']) for r in detail]
                if weather in paired_baselines and fingerprints != paired_baselines[weather]:
                    raise ValueError('Unpaired Shared prediction/input/source-count across families/seeds/modes')
                paired_baselines.setdefault(weather,fingerprints)
                rows[mode] = [r['frame'] for r in detail]
                shared = [r['shared'] for r in detail]
                if 'shared' in rows and rows['shared'] != shared:
                    raise ValueError('Different Shared stats across Same/Task')
                rows['shared'] = shared
            result[weather] = rows
        return result
    source = table('Source', 0)
    seeds = settings()['seeds']
    source_gain = {layer: float(np.mean([100 * (
        conditions[f'{w}/Source/0/same']['AP']['ap70'] - conditions[f'{w}/Source/0/same']['Shared']['ap70'])
        if layer == 'same' else 100 * (conditions[f'{w}/Source/0/task']['AP']['ap70'] -
        conditions[f'{w}/Source/0/same']['AP']['ap70']) for w in ADVERSE])) for layer in ('same', 'task')}
    details, bootstrap, family_verdicts = {}, {}, {}
    for family in FAMILIES:
        null = [table(family, seed) for seed in seeds]
        boot = k1_bootstrap(source, null, calculate_ap, settings()['bootstrap']['repetitions'], settings()['bootstrap']['seed'])
        bootstrap[family] = boot
        details[family] = {}
        family_verdicts[family] = {}
        for layer in ('same', 'task'):
            gains = []
            for seed in seeds:
                weather_gains = {}
                for weather in ADVERSE:
                    same = conditions[f'{weather}/{family}/{seed}/same']
                    task = conditions[f'{weather}/{family}/{seed}/task']
                    weather_gains[weather] = 100 * (same['AP']['ap70']-same['Shared']['ap70']) if layer == 'same' else (
                        100 * (task['AP']['ap70']-same['AP']['ap70']))
                gains.append(dict(seed=seed, adverse_mean_pp=float(np.mean(list(weather_gains.values()))), weather_pp=weather_gains))
            null_gain = float(np.mean([g['adverse_mean_pp'] for g in gains]))
            per_scene = {}
            for scene in sorted({r['scene_id'] for r in source['fog']['shared']}):
                scene_effects = []
                for mapping in [source]+null:
                    effects = []
                    for weather in ADVERSE:
                        ap = {method: ap70([r for r in mapping[weather][method] if r['scene_id'] == scene], calculate_ap)
                              for method in ('shared', 'same', 'task')}
                        effects.append(100 * (ap['same']-ap['shared']) if layer == 'same' else 100 * (ap['task']-ap['same']))
                    scene_effects.append(float(np.mean(effects)))
                per_scene[scene] = dict(source_gain_pp=scene_effects[0], seed_gains_pp=scene_effects[1:],
                                       null_seed_mean_pp=float(np.mean(scene_effects[1:])))
            details[family][layer] = dict(seeds=gains, mean_gain_pp=null_gain,
                best_seed=max(gains, key=lambda x: x['adverse_mean_pp']),
                worst_seed=min(gains, key=lambda x: x['adverse_mean_pp']), per_scene=per_scene)
            family_verdicts[family][layer] = k1_family(source_gain[layer], null_gain, boot[layer])
    verdict = {layer.upper(): combine([row[layer]['verdict'] for row in family_verdicts.values()]) for layer in ('same', 'task')}
    result = dict(status='complete', root_identity=protocol['identity'], verdict=verdict,
        source_gain_pp=source_gain, families=family_verdicts, seed_details=details, conditions=conditions,
        bootstrap={f: {l: {k: v for k, v in ci.items() if k != 'distribution'} for l, ci in b.items()}
                   for f, b in bootstrap.items()}, test_data_used=False, k3_frozen=True)
    atomic_json(folder / 'k1_seed_details.json', details)
    atomic_json(folder / 'k1_scene_bootstrap.json', bootstrap)
    atomic_json(folder / 'K1_SOURCE_SPECIFICITY.json', result)
    lines = ['# K1 Source Specificity', '', '## 一句话结论', '',
        f'固定三类对照比较得到 K1-Same={verdict["SAME"]}、K1-Task={verdict["TASK"]}；下表列出触发原因。', '',
        '正式比较只用 Fog/Rain/Snow 等权平均，null 先平均三个 seeds；Clean 仅 sanity。', '',
        '| Layer | Family | Source gain / pp | Null mean / pp | Source−Null / pp | Ratio | Verdict | Reason |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for family, layers in family_verdicts.items():
        for layer, row in layers.items():
            lines.append(f'| {layer} | {family} | {source_gain[layer]:.6f} | {details[family][layer]["mean_gain_pp"]:.6f} | '
                f'{row.get("difference_pp", float("nan")):.6f} | {row.get("ratio", float("nan")):.4f} | {row["verdict"]} | '
                + ('; '.join(row['reasons']) or '全部冻结条件通过') + ' |')
    lines += ['', '## 每天气、每 seed AP', '',
        '| Weather | Family | Seed | Shared AP70 | Same AP30/50/70 | Task AP30/50/70 | Same−Shared / pp | Task−Shared / pp | Task−Same / pp |',
        '|---|---|---:|---:|---|---|---:|---:|---:|']
    for weather in WEATHERS:
        for family in ('Source',)+FAMILIES:
            selected_seeds = [0] if family == 'Source' else seeds[:1] if weather == 'clean' else seeds
            for seed in selected_seeds:
                same = conditions[f'{weather}/{family}/{seed}/same']
                task = conditions[f'{weather}/{family}/{seed}/task']
                shared = same['Shared']['ap70']
                fmt = lambda row: '/'.join(f'{row["AP"][k]:.6f}' for k in ('ap30','ap50','ap70'))
                lines.append(f'| {weather} | {family} | {seed} | {shared:.6f} | {fmt(same)} | {fmt(task)} | '
                    f'{100*(same["AP"]["ap70"]-shared):.6f} | {100*(task["AP"]["ap70"]-shared):.6f} | '
                    f'{100*(task["AP"]["ap70"]-same["AP"]["ap70"]):.6f} |')
    lines += ['', '## 输出变化幅度（median / p90）', '',
        '候选 IoU 变化按 `1−IoU(candidate, Shared)` 计算，不读取 GT；全部量按 Shared top256 几何有效 anchor 统计。', '',
        '| Weather | Family | Seed | Mode | abs score | center / m | size / m | yaw / rad | candidate IoU change |',
        '|---|---|---:|---|---|---|---|---|---|']
    for row in conditions.values():
        v = row['output_magnitude']
        values = [f'{v[k]["median"]:.6g}/{v[k]["p90"]:.6g}' for k in (
            'abs_score','center_shift','size_change','yaw_change','candidate_iou_change')]
        lines.append(f'| {row["weather"]} | {row["family"]} | {row["seed"]} | {row["mode"]} | '+' | '.join(values)+' |')
    lines += ['', '## 解释边界', '',
        'PASS 只支持当前 development/validation 下相对匹配 null 的 source-specific headroom。'
        '不证明 source utility 可学习、可部署、正式 test 泛化或恶劣天气特有。K3 继续冻结。',
        '完整动作计数、每 seed/best/worst、逐场景与 5000 次 bootstrap 分布见 JSON。']
    atomic_text(folder / 'K1_SOURCE_SPECIFICITY.md', '\n'.join(lines)+'\n')
    return result


def k2_report(folder, protocol, rows, gates):
    from .stability import aggregate
    details, bootstrap = {}, {}
    for layer in ('same', 'task'):
        details[layer], bootstrap[layer] = aggregate([r for r in rows if r['layer'] == layer], layer,
                                                      settings()['bootstrap']['repetitions'])
    verdict = {layer.upper(): combine([r['verdict'] for r in families.values()]) for layer, families in details.items()}
    result = dict(status='complete', root_identity=protocol['identity'], verdict=verdict,
        sample_gates=gates, families=details, baseline_variant='A_ONLY', test_data_used=False, k3_frozen=True)
    result['weather_distribution'] = {layer:{weather:len({(r['frame'],r['target_index']) for r in rows
        if r['layer']==layer and r['weather']==weather}) for weather in ADVERSE} for layer in ('same','task')}
    atomic_json(folder / 'K2_LABEL_STABILITY.json', result)
    atomic_json(folder / 'k2_scene_bootstrap.json', bootstrap)
    lines = ['# K2 Label Stability', '', '## 一句话结论', '',
        f'小扰动后 K2-Same={verdict["SAME"]}、K2-Task={verdict["TASK"]}；center、yaw、ROI 分别判断。', '',
        f'基准 Same n={gates["same"]["n"]}，Task n={gates["task"]["n"]}。每个变体从原 A_ONLY 当前状态独立重算。', '',
        '等价最佳按 discrete6 定义，用 source/pair set 交集处理并列，exact top-1 单列。', '',
        '| Layer | Family | KEEP agreement | Source set retention | A/B/E agreement | Task positive retention | Pair retention (still positive) | Verdict | Reason |',
        '|---|---|---:|---:|---:|---:|---:|---|---|']
    for layer, families in details.items():
        for family, row in families.items():
            m = row['metrics']
            primary_same = [f'{m[k]:.2%}' for k in ('keep_agreement','source_retention','membership_agreement')] if layer=='same' else ['—']*3
            primary_task = [f'{m[k]:.2%}' for k in ('positive_retention','pair_retention')] if layer=='task' else ['—']*2
            lines.append(f'| {layer} | {family} | '+' | '.join(primary_same+primary_task)+f' | {row["verdict"]} | '
                +('; '.join(row['reasons']) or '全部冻结条件通过')+' |')
    lines += ['', '## 原始稳定/翻转计数', '',
        'Same 的 stable 指 source-set retention；Task 的 stable 指 true Task-positive 保留。', '',
        '| Layer | Variant | Stable / n | Flip / n | Exact action | Exact source | Top-2 Jaccard |',
        '|---|---|---|---|---:|---:|---:|']
    for layer, families in details.items():
        for family, row in families.items():
            for name, variant in row['per_variant'].items():
                m = variant['metrics']
                lines.append(f'| {layer} | {name} | {variant["stable_count"]}/{variant["n"]} | '
                    f'{variant["flip_count"]}/{variant["n"]} | {m["exact_action_stable"]:.2%} | '
                    f'{m["exact_source_stable"]:.2%} | {m["top2_jaccard"]:.4f} |')
    lines += ['', '## 逐场景计数', '', '| Layer | Variant | Scene | Stable | Flip | n |', '|---|---|---|---:|---:|---:|']
    for layer, families in details.items():
        for row in families.values():
            for name, variant in row['per_variant'].items():
                for scene, count in sorted(variant['per_scene'].items()):
                    lines.append(f'| {layer} | {name} | {scene} | {count["stable"]} | {count["flip"]} | {count["n"]} |')
    lines += ['', '## 解释边界', '',
        'family 指标是本类 variants 的 macro average，最差 variant 与场景 bootstrap 区间保存在 JSON。'
        '标签稳定不证明标签可学习；K3 继续冻结。']
    atomic_text(folder / 'K2_LABEL_STABILITY.md', '\n'.join(lines)+'\n')
    return result


def inconclusive_report(folder, component, protocol, error):
    name = 'K1_SOURCE_SPECIFICITY' if component == 'K1' else 'K2_LABEL_STABILITY'
    value = dict(status='failed', root_identity=protocol['identity'], verdict={'SAME':'INCONCLUSIVE','TASK':'INCONCLUSIVE'},
        reasons=[repr(error)], k3_frozen=True, test_data_used=False, completed_at=now())
    atomic_json(folder / (name+'.json'), value)
    atomic_text(folder / (name+'.md'), f'# {component}\n\n## 一句话结论\n\n'
        f'{component}-Same / Task = INCONCLUSIVE。原因：{error}\n\nK3 继续冻结，按 §42 停止逻辑处理。\n')


def summarize(run):
    from .common import load_contract, resume
    protocol = load_contract(run)
    for component in ('k1','k2'):
        if not (Path(run)/component/'manifest.json').exists():
            return None
        manifest = read_json(Path(run)/component/'manifest.json')
        if manifest.get('identity') != protocol['identity']:
            raise ValueError('Worker manifest protocol identity mismatch')
        if manifest['status'] == 'running':
            return None
        if manifest['status'] == 'complete' and resume(Path(run)/component/'result.json', protocol['identity']) is None:
            raise ValueError('Complete worker lacks hash-verified result commit')
    outputs = {}
    for component, name in (('k1','K1_SOURCE_SPECIFICITY'),('k2','K2_LABEL_STABILITY')):
        path = Path(run)/component/(name+'.json')
        outputs[component] = read_json(path) if path.exists() else {
            'status':'failed','verdict':{'SAME':'INCONCLUSIVE','TASK':'INCONCLUSIVE'}}
        if path.exists() and outputs[component].get('root_identity') != protocol['identity']:
            raise ValueError('Report protocol identity mismatch')
        for extension in ('.json','.md'):
            source = Path(run)/component/(name+extension)
            if source.exists():
                atomic_text(Path(run)/(name+extension), source.read_text(encoding='utf-8'))
    for component, filename in (('k1','k1_scene_bootstrap.json'),('k1','k1_seed_details.json'),
                                ('k2','k2_scene_bootstrap.json'),('k2','k2_flip_records.jsonl')):
        source=Path(run)/component/filename
        if source.exists():
            atomic_text(Path(run)/filename,source.read_text(encoding='utf-8'))
    result = final_verdict(outputs['k1']['verdict']['SAME'], outputs['k1']['verdict']['TASK'],
                           outputs['k2']['verdict']['SAME'], outputs['k2']['verdict']['TASK'])
    result.update(root_identity=protocol['identity'], K1=outputs['k1'], K2=outputs['k2'],
                  status='complete' if all(v['status']=='complete' for v in outputs.values()) else 'failed',
                  resource_stop_at=protocol['resource_stop_at'], k3_data_gate_record_only=settings()['k3_record_only'],
                  test_data_used=False)
    atomic_json(Path(run)/'STAGE2_FINAL.json', result)
    lines = ['# Source Utility Kill Protocol — Stage 2', '', '## 一句话结论', '', result['decision']+'。', '']
    for key in ('K1_SAME','K1_TASK','K2_SAME','K2_TASK','H1_SOURCE_SPECIFIC','H2_TASK_SPECIFIC'):
        lines.append(f'- {key} = {result[key]}')
    lines += ['', '## 触发门槛', '',
        '逐 null family、逐 perturbation family 的阈值与失败原因见 K1/K2 报告；运行不完整按 INCONCLUSIVE 处理。', '',
        '## 目前能得出什么结论', '',
        '结论限于当前 development/validation、匹配 null 和预注册的小扰动。', '',
        '## 还不能得出什么结论', '',
        '未证明 utility 可学习、模型可部署、正式 test 泛化或恶劣天气特有。', '',
        '## 后续边界', '',
        'K3 继续冻结。即使对应 K1/K2 PASS，也须人工审核 true train joint data gate；本程序不生成 K3 标签或启动训练。']
    atomic_text(Path(run)/'STAGE2_FINAL.md', '\n'.join(lines)+'\n')
    return result
