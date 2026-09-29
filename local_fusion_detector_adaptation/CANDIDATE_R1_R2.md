# R1 / R2 候选可靠性验证：服务器运行说明

本实现复用已导出的官方 train 与旧 validation F/top256 候选、原 GT 标签和 400 维融合特征。它不训练 PointPillar、GSPR、F 或 F+D，不改原始候选框。**所有科研计算在远程服务器运行；当前本地只提供代码。**

2026-09-29 修复：首次运行在 `clean/0` 的 F 复现检查中停止。导出器现按原 train/validation 提取脚本分别恢复全局随机种子与 DataLoader 种子，并用旧候选保存的解码框和分数做严格核对。默认 overlay 改为 `_v2` 路径，避开首次失败留下的未完成目录；无须删除旧目录。

## 实际检验的输入

- **R1**：每车冻结单车头在全 anchor 网格独立产生候选，按各车自己的分数选择范围内、几何有效的 top256，再跨 anchor 与 F 融合候选作 BEV IoU 匹配。记录自车/邻车匹配、邻车不同 anchor 的匹配、来源候选在原 NMS 前后的状态，以及各车到候选的距离和投影点云在候选中心 3 m 内的点数。局部点数仅是“附近有采样”的代理，**不能证明可见、无遮挡或可靠地为空**。无匹配也只表示有界候选池内无匹配。
- **R2**：以 F 的全部 top256 anchor ID 为主键，读取冻结 F+D **同 anchor**分类与回归；即使该 ID 没进入 F+D 的 top256，也照样保留。计算分数差、F/F+D 框 IoU、中心、尺寸和朝向差。离线分别检验“只加分数差”“只加几何差”“二者合用”。
- 导出到单独的 overlay JSONL，所有字段在推理时可得；GT 不写进 overlay。旧 `candidate_rows.jsonl`、GT 和特征缓存不被修改。

离线 Cheap Gate 先做不训练的分层检查：按天气、原分数段、距离和车辆数列出单特征 AUC/PR-AUC，并检查真实 NMS 坏压好边及“没匹配但附近有点”的数量。每种天气还从匹配的邻车候选中均匀抽样最多 10000 条，与 GT 核对几何误匹配比例。然后用官方 train 场景做 5 折按场景隔离的信息检查，并在旧 validation 场景上用固定训练设置做一次探索性 AP 回放。对照包括候选自身、候选自身＋来源基础量、R1 的“只有观测机会代理”（距离和附近点数）、R1/R2 特征打乱、R1、R2 及二者组合。两种固定策略分开报告：仅修改原 NMS 抑制组顺序；对完整 top256 重评分。主 AP 沿用原 F 分数，只衡量框去留；输出重评分的 AP 单独记录。最终框数以每帧原 F `score>0.2` 的结果为预算。每种天气还记录 R1/R2 的额外计算秒数。

## 服务器命令

先把本目录新增的两个 Python 文件和 `run_candidate_r1_r2.sh` 同步到远程 `/home/cjm/OpenCOOD-main/cjmnet/local_fusion_detector_adaptation/`。默认路径沿用既有实验：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
CHECK_ONLY=1 sh local_fusion_detector_adaptation/run_candidate_r1_r2.sh
```

先跑不训练的 R1/R2 审计。默认 `MODE=audit-only` 只导出旧 validation overlay 并生成分层报告；导出可能耗时较长，后台运行：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
OUT=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227/candidate_r1_r2_audit_v2_20260929 \
  nohup sh local_fusion_detector_adaptation/run_candidate_r1_r2.sh \
  > /data/cjm/datasets/logs/candidate_r1_r2_v2_20260929.log 2>&1 < /dev/null &
echo $!
```

查看进度：

```sh
tail -f /data/cjm/datasets/logs/candidate_r1_r2_v2_20260929.log
```

确认不训练审计有希望后，再跑官方 train 导出与离线 Cheap Gate；它会复用已完成的 validation overlay：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
MODE=all OUT=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227/candidate_r1_r2_gate_v2_20260929 \
  nohup sh local_fusion_detector_adaptation/run_candidate_r1_r2.sh \
  > /data/cjm/datasets/logs/candidate_r1_r2_gate_v2_20260929.log 2>&1 < /dev/null &
echo $!
```

需要分开运行时设置 `MODE=export-only` 或 `MODE=gate-only`。已有**完整** overlay 会复用；发现只建立了一部分的 overlay 目录会报错，避免静默续用。可设置 `TRAIN_ROOT`、`EVAL_ROOT`、`TRAIN_OVERLAY`、`EVAL_OVERLAY`、`RUN`、`SOURCE_TOPK`、`SOURCE_PRESELECT`、`SEEDS` 和 `OUT`。训练/验证 overlay 必须来自同一冻结模型、来源池参数和候选数据；代码会核对文件与 checkpoint SHA256、场景、anchor ID、F 分数/框，并在旧 validation 上复现 F/F+D 原 AP。

每阶段结束后请取回各自 `OUT/candidate_r1_r2.md`、`OUT/candidate_r1_r2.json`；完整 Cheap Gate 还会生成 `OUT/candidate_r1_r2_frames.jsonl`。完整 JSON 包含不训练的分层诊断、额外计算时间、分数段 AUC/PR-AUC、真实坏压好 NMS 边排序、AP30/50/70、帧顺序和跨帧 AP、TP/FP、新增/丢失 GT、逐场景剔除与预先固定的投入门槛。

## 结论边界

旧 validation 的 9 个场景已多次用于开发。通过 Cheap Gate 只表示值得获取**新的独立场景**复验，不能作为论文最终效果。R1 的各车观测共享检测器权重，误差可能相关；R2 需要同时运行 F 与 F+D，额外计算必须和收益一起报告。若 R1/R2 没有越过候选自身与来源基础量对照，就停止这两条方法线。
