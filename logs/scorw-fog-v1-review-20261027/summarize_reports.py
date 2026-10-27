"""Summarize existing verified reports; no data/model/scientific experiment rerun."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
REPORTS=HERE/"verified_reports"


def read(name):return json.loads((REPORTS/name).read_text("utf-8"))


def main():
    manifest=read("REPORT_MANIFEST.json")
    for p in REPORTS.glob("*.json"):
        if p.name in manifest and hashlib.sha256(p.read_bytes()).hexdigest()!=manifest[p.name]["sha256"]:
            raise ValueError("Report copy changed: "+p.name)
    qa=read("DATASET_QA.json");gates=read("VALIDATION_GATES.json");det=read("VALIDATION_ATTFUSE_RESULTS.json")
    receipt=read("FINAL_COMMIT.json");protocol=read("DATASET_PROTOCOL.json");contract=read("CONTRACT.json")
    canonical=lambda x:json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
    if hashlib.sha256(canonical(protocol)).hexdigest()!=contract["identity"] or contract["identity"]!=gates["contract"]:
        raise ValueError("Protocol/contract/gates identity mismatch")
    if hashlib.sha256((REPORTS/"DATASET_QA.json").read_bytes()).hexdigest()!=receipt["dataset_QA_sha256"]:
        raise ValueError("D7 QA receipt mismatch")
    keys=("original","output","hard","weather","unknown","lost","high_tau","empty_clouds","unmodelled_near_field")
    totals={k:sum(s["totals"][k] for s in qa["splits"].values()) for k in keys}
    n=totals["original"];distance=[]
    for i,b in enumerate(qa["splits"]["train"]["per_distance"]):
        values={k:sum(s["per_distance"][i][k] for s in qa["splits"].values()) for k in ("input","output","hard","replaced","lost","input_signal","output_signal")}
        distance.append(dict(low_m=b["low"],high_m=b["high"],**values,
            replaced_fraction=values["replaced"]/values["input"] if values["input"] else None,
            signal_ratio=values["output_signal"]/values["input_signal"] if values["input_signal"] else None))
    detection={}
    for name in ("AP30","AP50","AP70"):
        a,b=det["metrics"]["clean"][name],det["metrics"]["fog"][name]
        detection[name]=dict(clean=a,fog=b,AP_drop_percentage_points=100*(a["AP"]-b["AP"]),
            net_additional_misses=b["FN"]-a["FN"],FP_change=b["FP"]-a["FP"],
            clean_recall=a["TP"]/a["gt"],fog_recall=b["TP"]/b["gt"],
            clean_precision=a["TP"]/(a["TP"]+a["FP"]),fog_precision=b["TP"]/(b["TP"]+b["FP"]))
    scene_ranges={}
    for split in ("train","validate","test"):
        rows=[r for r in qa["per_scene"] if r["split"]==split]
        ordered=sorted(rows,key=lambda r:r["totals"]["weather"]/r["totals"]["original"])
        scene_ranges[split]=dict(min_scene=ordered[0]["scene"],min_fraction=ordered[0]["totals"]["weather"]/ordered[0]["totals"]["original"],
            max_scene=ordered[-1]["scene"],max_fraction=ordered[-1]["totals"]["weather"]/ordered[-1]["totals"]["original"])
    val=sorted((r for r in qa["per_scene"] if r["split"]=="validate"),key=lambda r:r["frames"],reverse=True)
    near_original=sum(s["input_distance_hist"][0] for s in qa["splits"].values())
    near_fog=sum(s["output_distance_hist"][0] for s in qa["splits"].values())
    near_hard=distance[0]["hard"]
    weather_near_lower=max(0,near_fog-near_hard-totals["unknown"])
    # All selected hard returns keep the original XYZ in the frozen receiver and
    # float32 PCD codec. Subtracting ALL unknown returns gives a conservative bound.
    near=dict(original_0_3m=near_original,fog_0_3m=near_fog,increase=near_fog-near_original,
        weather_0_3m_count_lower_bound=weather_near_lower,
        weather_0_3m_fraction_lower_bound=weather_near_lower/totals["weather"],
        basis="count-conservation bound from reported distance marginals and all unknown returns; no pointwise sidecar rescan")
    metadata=json.loads((HERE/"SERVER_DIRECTORY_METADATA.json").read_text("utf-8"));files={r["name"]:r for r in metadata if "name" in r}
    tz=timezone(timedelta(hours=8));start=files["CONTRACT.json"]["mtime"];end=files["VALIDATION_GATES.json"]["mtime"]
    timing=dict(contract_file_time=datetime.fromtimestamp(start,tz).isoformat(),final_gates_file_time=datetime.fromtimestamp(end,tz).isoformat(),
        elapsed_hours=(end-start)/3600,scope="wall interval between file timestamps; includes audit, inference and any pauses/resumes; not CPU hours")
    usage=(HERE/"SERVER_DATASET_USAGE.txt").read_text("utf-8").splitlines()
    allocated=int(next(s.split()[0] for s in usage if s[:1].isdigit()))
    audit=json.loads((HERE/"REPORT_DOWNLOAD_AUDIT.json").read_text("utf-8"))
    verified=sum(r["status"]=="SHA256_VERIFIED" for r in audit["records"].values())
    result=dict(status=gates["status"],report_hashes_verified=verified,protocol_contract_and_QA_receipt_verified=True,
        splits=qa["splits"],totals=totals,source_fractions={k:totals[k]/n for k in ("hard","weather","unknown","lost","high_tau")},
        distance=distance,detection=detection,per_scene=qa["per_scene"],scene_replacement_ranges=scene_ranges,
        largest_two_validation_scenes=dict(scenes=[r["scene"] for r in val[:2]],frames=sum(r["frames"] for r in val[:2]),
            fraction=sum(r["frames"] for r in val[:2])/det["frames"]),near_field_bound=near,timing=timing,
        dataset_allocated_bytes=allocated,dataset_allocated_GiB=allocated/2**30,
        REAL_WORLD_VALIDATION_MISSING=True,experiments_rerun=False,server_files_modified=False,
        limitations=["No fresh rehash of the full 2.04 billion PCD returns; full-data validation is evidenced by the saved D0-D7 audits.",
            "Root detector report contains aggregate validation AP, not per-scene AP or ego-only/cooperation causal attribution.",
            "Existing forced reader cannot emit the Chinese Markdown files under its ASCII locale; all key JSON reports were read exactly and SHA256 verified."])
    (HERE/"ANALYSIS_SUMMARY.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# SCOR-W-Fog v1 完成结果复核（2026-10-27）","",
        "**结论：全量数据构建完成，数值与读取门槛通过；冻结AttFuse在合成雾下出现明显漏检增加。这版可以作为内部合成雾数据集使用，真实雾天标定仍缺失。**","",
        "本次通过已核验服务器指纹和既有受限SSH入口读取报告，只保存本地副本及分析文件。没有修改服务器文件，没有重新生成点云、运行检测或训练。",
        f"共核对{verified}份JSON报告的SHA256，并核对协议—合同—最终门槛身份及D7质量报告回执。另保存发布哈希清单。全量PCD未在本轮重新逐点读取，其完整性结论依据已完成D0–D7的保存审计。","",
        "## 1. 数据规模与完成条件","","| 划分 | scene | 协作帧 | 每车PCD文件 | 原始及Fog点数 | 天气回波替换率 |","|---|---:|---:|---:|---:|---:|"]
    for split in ("train","validate","test"):
        s=qa["splits"][split];lines.append(f"| {split} | {s['scenes']} | {s['frames']:,} | {s['pcds']:,} | {s['totals']['original']:,} | {100*s['totals']['weather']/s['totals']['original']:.2f}% |")
    lines += ["",f"合计68个scene、10,524个协作帧、36,038份Fog PCD、{n:,}个输入与输出回波。所有划分点数相同，空点云为0。",
        "所有PCD均通过实际OpenCOOD读入、体素预处理和批处理检查；Train/Validate各scene固定首末帧还执行了原dataset取样及frame collate。元数据与位姿保持原来源及哈希，scene之间划分隔离。Test只进行构建与兼容性检查，没有运行Test AP。",
        "每scene一个固定共享世界场，跨车、跨时间不重置。68个scene场身份互不重复。物理参数及主seed沿用冻结配置。","",
        "## 2. 回波改变了什么","",
        f"按模拟器来源标签统计：原始硬回波{totals['hard']:,}（{100*totals['hard']/n:.4f}%），天气回波{totals['weather']:,}（{100*totals['weather']/n:.4f}%），来源不确定{totals['unknown']:,}（{100*totals['unknown']/n:.4f}%），无返回0。",
        "这些标签描述接收器选中的原始/软散射/混合信号，不是逐点车辆语义真值，也不是实测光子来源。",
        "总点数保持不变，但约1.70亿个原位置回波被替换，原始几何信息已经改变。当前版本实际触发了回波替换与距离异常，超出了仅改变强度的实现；无返回机制没有在这组冻结参数中触发，无法覆盖真实雾天点数丢失这一类退化。该情况符合本次预注册完成规则。","",
        "| 原始光路距离 | 原始点数 | 被天气回波替换 | 输出/原始强度总和 |","|---|---:|---:|---:|"]
    for d in distance:
        if not d["input"]:continue
        lines.append(f"| {d['low_m']}–{d['high_m']}m | {d['input']:,} | {100*d['replaced_fraction']:.2f}% | {100*d['signal_ratio']:.2f}% |")
    lines += ["","上表按Clean原始光路距离分组；输出强度包含替换后的天气信号，不能直接解释为目标硬回波透射率。远距离组的强度比例回升，也不表示目标恢复了。120–200m只有4,012条光路，该组比例不能当作大量独立样本的统计结论。","",
        "## 3. 两个必须保留的物理局限","",
        f"**近场聚集很强。** 0–3m点数从{near_original:,}增加到{near_fog:,}，增加量{near['increase']:,}恰等于天气替换总量。在硬回波位置不变、来源计数守恒的规则下，即使把全部unknown都归入近场，仍可推得至少{100*near['weather_0_3m_fraction_lower_bound']:.2f}%的天气回波位于0–3m。这是汇总计数的保守下界，没有逐点重扫sidecar。该模式符合当前最强峰/重叠函数模型的输出，需要警惕模型依赖固定近场合成特征；其真实合理性尚未得到实测标定。",
        f"**高衰减光路不少。** {totals['high_tau']:,}条光路的光学厚度大于1，占{100*totals['high_tau']/n:.2f}%。光学厚度表示沿路累计雾遮挡强度。单次散射模型在这些路径上的数值有限，不保证真实物理误差小。",
        "接收器使用以晴天已有回波为条件的单次散射、有限脉冲与最强叠加峰近似。未恢复真实扫描束/逐束时间/未观测遮挡物体/实际波形，强度阈值也未做传感器标定。REAL_WORLD_VALIDATION_MISSING保持不变。","",
        "## 4. 固定AttFuse检测结果","",
        "同一冻结权重、同一批1,980个验证协作帧、同一54,621个按帧计数的标注目标实例。AP为综合检测精度；30/50/70表示预测框与标注框的重合要求。这里沿用R3的俯视平面框重合计算和全局分数排序，并固定Clean标注；原native入口可能按天气点数过滤可见标注，两者应分开标注。没有训练或按AP修改天气。","",
        "| 指标 | Clean | Fog | 下降 |","|---|---:|---:|---:|"]
    for name,m in detection.items():lines.append(f"| {name} | {100*m['clean']['AP']:.2f}% | {100*m['fog']['AP']:.2f}% | {m['AP_drop_percentage_points']:.2f}个百分点 |")
    a,b=detection["AP70"]["clean"],detection["AP70"]["fog"]
    lines += ["","在70%框重合要求下：","",
        "| 计数 | Clean | Fog |","|---|---:|---:|",
        f"| 正确检出 | {a['TP']:,} | {b['TP']:,} |",f"| 漏检 | {a['FN']:,} | {b['FN']:,} |",f"| 误检 | {a['FP']:,} | {b['FP']:,} |","",
        f"漏检净增加{b['FN']-a['FN']:,}个目标实例，召回率从{100*detection['AP70']['clean_recall']:.2f}%降至{100*detection['AP70']['fog_recall']:.2f}%。误检总数减少，但预测总数也从{a['TP']+a['FP']:,}降至{b['TP']+b['FP']:,}，预测中正确结果占比从{100*detection['AP70']['clean_precision']:.2f}%降至{100*detection['AP70']['fog_precision']:.2f}%。因此主要现象为检出减少、漏检上升。计数差是净变化，不能直接当成逐目标转移事件数。",
        "该结果支持合成雾确实影响固定检测器的行为。当前D6没有单车对照和逐目标协作反事实，无法从这些总体AP证明协同融合本身存在特有缺陷，也无法证明共享场优于独立场或GSPR优于AttFuse。模型结果标记UNCALIBRATED_EXPLORATORY。","",
        "## 5. 场景差异与汇总范围","",
        "| 划分 | 最低scene替换率 | 最高scene替换率 |","|---|---:|---:|"]
    for split,r in scene_ranges.items():lines.append(f"| {split} | {100*r['min_fraction']:.2f}% | {100*r['max_fraction']:.2f}% |")
    lines += ["","验证场景逐项如下。它们的替换率受原始距离分布、车辆布局和实际雾场共同影响，不能单独归因于场浓度。","",
        "| scene | 帧 | CAV数 | 替换率 |","|---|---:|---:|---:|"]
    for s in (r for r in qa["per_scene"] if r["split"]=="validate"):
        lines.append(f"| {s['scene']} | {s['frames']} | {len(s['cavs'])} | {100*s['totals']['weather']/s['totals']['original']:.2f}% |")
    lines += ["",f"帧数最多的两个验证scene占{100*result['largest_two_validation_scenes']['fraction']:.2f}%验证帧。根报告只有总体AP，没有逐scene AP，因此当前不能断言9个scene都下降相同幅度。54,621个目标实例和连续帧也不能作为独立scene样本来放大统计把握。","",
        "## 6. 工程与数值证据","",
        "9项工程验收通过。3组合成条件各36条、2份Train抽样各147条，共402条输入光路，分别以batch=1/8/16比较；记录的强度、光学厚度、软硬信号和天气XYZI最大误差全部为0，来源/有效性完全一致，最终PCD SHA256相同。这证明已测样本的批量实现与冻结逐点实现一致，未逐点重跑全量标量参考。",
        "Hahner硬回波ADC、查表软回波及替换/无返回重放通过，新均匀软响应与独立积分、波形面积和空间收敛检查通过。40条合成回归中，新接收器产生12个天气替换及15个数字零码无返回，证明无返回分支能够执行；全量OPV2V返回0仍是当前输入和冻结参数下的结果。原版对应19个替换及8个零码无返回；两种接收器选峰规则不同，最终数量仅作描述性比较。工程/数学通过不等于真实天气真实性已证明。",
        "IoU审计记录8,873,887次产生有限结果的几何相交警告，非法几何0、非有限/非法结果0；未修复框、未改变原计算值。警告数量较大，需要在复现记录中保留，当前没有因它跳过目标或放宽指标。","",
        "## 7. 实际时间、空间和使用建议","",
        f"合同文件时间：{timing['contract_file_time']}；最终状态文件时间：{timing['final_gates_file_time']}。文件时间跨度约{timing['elapsed_hours']:.2f}小时，即{timing['elapsed_hours']/24:.2f}天，包含清点、生成、审计、推理以及可能的暂停或恢复，不能当成纯CPU计算时间。",
        f"当前输出目录实占{allocated:,}字节，约{allocated/1e9:.2f}GB / {allocated/2**30:.2f}GiB。使用只读低优先级du统计，不跟随Clean符号链接，因此不包含原始数据和外部冻结权重。D0的20.67天及261.70GB是运行前保守预算，不是实际耗时/占用。小样本计时含进程启动、距离分层与2倍安全系数，不应用预算/实际时间之比宣称算法加速。",
        "建议冻结并使用这版作为内部合成雾训练/验证数据。现有结果足以支持它已产生可复现的几何退化和检测压力；零无返回、强近场聚集、高衰减路径和真实标定缺失继续写入说明。后续模型比较固定数据、权重/训练预算、标注及评测范围。当前仅评测Validate；Test继续留给冻结后评测，不按Test AP选择天气参数。",
        "Clean视图和Fog元数据包含符号链接，迁移或清理时需保留原始数据依赖；只复制Fog PCD不能得到完整可用的数据集。","",
        "## 8. 本轮复核文件","",
        "- verified_reports/：下载并核对的JSON报告。",
        "- REPORT_DOWNLOAD_AUDIT.json：逐报告复制与SHA256状态。",
        "- ANALYSIS_SUMMARY.json：实际指标、全部68个scene统计和本报告使用的派生比例。",
        "- SERVER_DIRECTORY_METADATA.json、SERVER_DATASET_USAGE.txt：完成时间及实际磁盘占用来源。",
        "- 现有SSH读取入口使用ASCII输出，三份中文Markdown未能下载；所有关键统计使用完整JSON副本，未依据截断文本猜测。",
        "","服务器结果根：/data/cjm/datasets/scorw-fog-v1-coveragefix。"]
    (HERE/"ANALYSIS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=result["status"],report_hashes_verified=verified,timing=timing,allocated_bytes=allocated,
        near_field_bound=near,largest_two_validation_scenes=result["largest_two_validation_scenes"]),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
