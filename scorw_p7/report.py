"""Human-readable results and compact machine-readable exports, without data clouds."""
from pathlib import Path
import io
import csv
import gzip
import tarfile

from scorw_p7.core import atomic_json, atomic_text, atomic_bytes, digest, read_json, identity
from scorw_p7.signal_metrics import METRICS


def write_reports(output, scenes, analysis, verdict, selection, freeze, matrices):
    stripped = [{k:v for k,v in s.items() if k not in {"pairs","targets"}} for s in scenes]
    atomic_json(output/"P7_PER_SCENE_RESULTS.json",dict(scenes=stripped,independent_unit="scene"))
    atomic_json(output/"P7_CROSS_CAV_CORRELATION.json",dict(primary=True,scenes=[dict(scene=s["scene"],
        partition=s["partition"],pairs=s["pairs"],summary=s["metrics"]["delta_correlation"]) for s in scenes]))
    atomic_json(output/"P7_COMPLEMENTARITY_RESULTS.json",dict(sampled_proxy=True,scenes=[dict(
        scene=s["scene"],partition=s["partition"],targets=s["targets"],subgroups=s["subgroups"]) for s in scenes]))
    atomic_json(output/"P7_STATISTICAL_ANALYSIS.json",analysis)
    atomic_json(output/"P7_BALANCE_CHECK.json",dict(status="passed",scenes={s["scene"]:s["balance"] for s in scenes},
        per_ray_checks=[m["balance"] for m in matrices],scope="per original sampled ray and fixed target/CAV, all20 fields; no single-repeat equality claim"))
    atomic_json(output/"P7_NEGATIVE_CONTROLS.json",dict(status="passed",scenes={s["scene"]:dict(
        shared_reorder=s["negative_control"],single_neighbor=s["single_neighbor_check"]) for s in scenes}))
    atomic_json(output/"P7_RAY_SAMPLING_AUDIT.json",dict(clouds=[m["sampling"] for m in matrices],
        full_exact=False,primary_ray_field_integrations=sum(m["primary_ray_field_pairs"] for m in matrices),
        primary_estimator=freeze["sampling"],pcds_written=0,weather_conditioned_sampling=False))
    atomic_json(output/"P7_DECISION.json",verdict)
    lines = ["# SCOR-W P7 结果", "", verdict["decision"], "",
        "只评估固定 Clean 返回点条件下的采样信号代理；没有运行检测器，没有生成天气 PCD，实雾真实性未获验证。", "",
        f"本次实际计算 {len(scenes)} 个 scene；新独立验证 {sum(s['partition']=='independent_validation' for s in scenes)} 个；每 scene 2 个连续索引帧，完整保留20个天气场。",
        "开发场景复用已分析的P6开发集；新验证仅指SCOR-W P5/P6未分析，不代表未用于历史检测训练。", "",
        "逐CAV、逐固定光路/目标的完整循环边缘分布严格平衡；每次重复的总体强度可以不同。E3是有限场池错位对照，不是独立重新抽场。", "",
        "| scene | partition | 有效目标实例 | 有效目标-CAV对 | ΔCorr | ΔC | Δstd(C) | ΔP(C>0) |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in scenes:
        c,m = s["counts"],s["metrics"]
        lines.append(f"| {s['scene']} | {s['partition']} | {c['valid_target_instances']} | {c['valid_target_cav_pairs']} | "
                     + " | ".join(str(m[k]) for k in METRICS) + " |")
    lines += ["", "各帧车辆身份与参与车辆：", ""]
    for f in freeze["active_frames"]:
        lines.append(f"- {f['scene']}/{f['timestamp']}: CAV={f['cavs']}; active={f['active_cavs']}; ego={f['ego']}")
    lines += ["", "独立验证的场景级汇总（E2−E3；区间为条件于固定场池和光路样本的探索性场景bootstrap）：", ""]
    for metric,item in analysis["partitions"]["independent_validation"]["metrics"].items():
        lines.append(f"- {metric}: mean={item['mean']}, CI95={item['ci95']}, valid scenes={item['valid_scenes']}, direction fraction={item['direction_consistency_fraction']}; per-scene={item['per_scene']}")
    lines += ["", "正、负和方向反转场景均保留，未筛选matched重复。场内目标/帧/车辆/20次重组合都未当成独立scene。",
        "单邻车子组的平均ΔC受平衡设计约束为零；应同时查看C的方差、分位数与正值频率。",
        "R采用固定分层样本的加权比值，是近似量；Clean全量点数、强度与样本数量见采样审计，不能据此称为全量精确回波统计。",
        "相关性先在每个目标/帧/CAV对内沿天气池中心化，再按目标和scene汇总，未混合不同目标的绝对距离关系。",
        "框内点关联没有逐点表面语义真值；扫描束、逐束时间、波形、接收阈值不可恢复；只模拟双程目标衰减，不模拟非均匀soft散射。",
        "20场仅给有限池条件估计，留一场敏感性不是新的独立实验，也不是自动增加seed。", "",
        "项目建议："+verdict.get("project_recommendation",verdict["next_action"]), ""]
    atomic_text(output/"RESULTS.md","\n".join(lines))
    atomic_text(output/"NEXT_STAGE_DECISION.md","# P7 下一阶段判断\n\n"+verdict["decision"]+"\n\n"+
        verdict.get("reason","")+"\n\n"+verdict.get("project_recommendation",verdict["next_action"])+
        "\n\n门槛在任何天气计算前冻结，是项目启发式，不是校准过的检测收益或统计功效标准。无显著性不等于小效应。\n"+
        "满足CONTINUE_TO_DETECTOR仅给予下一步固定权重AttFuse小样本诊断资格；本程序永不启动该实验。其余结果保留，不通过增加seed、场景或调参寻找正结果。\n")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream,fieldnames=["scene","partition",*METRICS],lineterminator="\n")
    writer.writeheader()
    writer.writerows(dict(scene=s["scene"],partition=s["partition"],**s["metrics"]) for s in scenes)
    atomic_text(output/"P7_SCENE_TABLE.csv",stream.getvalue())


def archive_reports(root, names, max_bytes=64*1024*1024):
    """Whitelist only small reports, source/protocol identities and completion gates."""
    rows = {name:dict(sha256=digest(root/name),bytes=(root/name).stat().st_size) for name in sorted(names)}
    if sum(r["bytes"] for r in rows.values()) > max_bytes:
        raise ValueError("Small report budget exceeded; caches/PCDs must never enter exports")
    manifest = dict(files=rows,pcds_included=False,arrays_included=False,report_identity=identity(rows))
    path = root/"P7_REPORT_MANIFEST.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Existing published report manifest changed")
    atomic_json(path,manifest)
    payload = io.BytesIO()
    with gzip.GzipFile(fileobj=payload,mode="wb",mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode="w") as archive:
            for name in sorted([*names,"P7_REPORT_MANIFEST.json"]):
                data = (root/name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size,info.mtime,info.mode = len(data),0,0o644
                archive.addfile(info,io.BytesIO(data))
    atomic_bytes(root/"P7_SMALL_RESULTS.tar.gz",payload.getvalue())
    return dict(path=str(root/"P7_SMALL_RESULTS.tar.gz"),sha256=digest(root/"P7_SMALL_RESULTS.tar.gz"),reports=manifest)
