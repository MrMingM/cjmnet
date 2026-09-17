# Stage-2 Evidence Validity Audit

本目录用于 Q-A（恶劣天气下观测不足）第一轮诊断之后的**证据有效性审查**。

它不设计新网络、不训练新模型，也不使用 OPV2V-W test 做假设筛选。目标只有一个：

> 在进入 Oracle / 方法设计之前，先检查 Stage-1 中“evidence strong / weak”的经验尺子是否真的对应 task-usable evidence。

## 1. 审查内容

### A. Threshold sensitivity（纯离线）

使用上一轮 `qa_observation_diagnostic` 生成的 `targets.jsonl`，重新构造 Q20/Q25/Q30/Q40 clean-detected distance-bin thresholds，检查：

- `ego strong` 比例随阈值变化多少；
- `any peer strong` 比例随阈值变化多少；
- 与 Q25 候选集合的 Jaccard；
- 所谓 fusion candidate 是否严重依赖某一个任意阈值。

阈值稳定只说明 proxy 稳定，不证明 proxy 语义正确。

### B. Snow / N_eff 数值审查（纯离线）

上一轮 Snow 出现巨大负相对变化，是 clean `box_neff≈0` 的分母爆炸。

本阶段不再用 `max(clean,1e-6)` 掩盖零分母，而是同时报告：

- clean N_eff=0 的样本数和比例；
- 绝对差 `clean-weather`；
- bounded symmetric difference `(clean-weather)/(clean+weather+eps)`；
- `log1p(clean)-log1p(weather)`；
- 原相对损失只在 `clean>1e-6` 的样本上报告。

注意 bounded symmetric difference 是**新指标**，不能继续叫旧的“相对损失率”。

### C. Scene / distance 分层（纯离线）

检查 `ego strong`、metric fusion candidate 等现象是否集中在少数场景或距离段，避免把场景特例误判成统一机制。

### D. Ego -> full degradation（纯离线）

直接统计：

- ego 检出、full 反而漏检；
- ego 漏检、full 恢复；
- 场景集中度。

它只能说明 full fusion 可能有负作用 case，不能单独证明 attention 是原因。

### E. Peer-alone task-usable evidence audit（需要 GPU 推理）

这是本阶段最关键的新测试。

对于 Stage-1 中所有 `clean ego 检出 -> weather ego 漏检` 目标，对每个 CAV source 单独执行：

1. 使用该 source 已经投影到 ego 坐标系的 multi-scale pre-fusion feature；
2. **绕过 AttFuse**；
3. 使用冻结的原 deblocks + cls/reg heads；
4. 使用原 post-processing；
5. 记录该 peer 是否能独立产生 IoU>=0.7 的正确目标框，同时记录最终框 best IoU、score 和 frame FP。

因此：

- `metric peer strong` = N_eff/coverage proxy 判定；
- `peer-alone detected` = 更强的 task-usable evidence 证据；
- `peer-alone detected + full miss` = **source-valid / fusion-invalid candidate**。

最后一种仍然只是融合瓶颈候选，不代表 AttFuse attention 已被证明是唯一原因。

## 2. 为什么 peer-alone 不是 mask ablation

现有 `gspr_communication.masked_attfuse.fuse()` 强制 ego 永远可用，因此“只打开一个 peer mask”实际上仍是 `ego + peer`，不能叫 peer-alone。

本目录的 `peer_audit.py` 直接从 `encoded['levels'][peer]` 取单一来源 feature，独立走 deblocks/head；没有 ego feature，也没有 AttFuse。

由于当前协议是 `proj_first=True`，这些 feature 已经处于 ego 对齐坐标系，可直接与 ego GT 做同一检测评估。

## 3. 一键运行

默认复用已经完成的 Stage-1：

```bash
/data/cjm/datasets/logs/qa_observation_20260916_194240
```

服务器执行：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
git pull
ROCR_VISIBLE_DEVICES=0 bash qa_evidence_validity/run_all.sh
```

如果 Stage-1 路径不同：

```bash
STAGE1_ROOT=/data/cjm/datasets/logs/你的qa_observation目录 \
ROCR_VISIBLE_DEVICES=0 \
bash qa_evidence_validity/run_all.sh
```

GPU 规则与项目一致：只设置 `ROCR_VISIBLE_DEVICES=<physical_index>`；脚本主动 unset `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`。

## 4. 输出

输出目录：

```text
/data/cjm/datasets/logs/qa_evidence_validity_时间戳/
```

主要文件：

```text
evidence_validity_report.md
evidence_validity_results.json
offline/offline_results.json
peer/fog/peer_targets.jsonl
peer/rain/peer_targets.jsonl
peer/snow/peer_targets.jsonl
```

跑完优先回传：

```text
evidence_validity_report.md
evidence_validity_results.json
```

## 5. Stage-2 结束后的 Gate

这一步结束后不继续无限诊断。

- 如果 strong/weak 对 Q20-Q40 极不稳定，或 metric peer strong 对 peer-alone detection 的 precision 很差：先修 evidence ruler，暂不做 fusion Oracle。
- 如果存在跨场景、数量非偶然的 `peer-alone detected + full miss`：进入小规模 source-combination / local-fusion causal intervention。
- 如果 Snow 的 ego-strong 现象对阈值稳定，但 peer/ego 的 task-usable evidence 仍不支持当前 proxy：优先研究“proxy 为什么高估 target evidence”。
- local H-A5 后续若做 clean-information replacement，只能解释“某干预位置具有恢复机会”，不能直接宣称“错误最初发生在该层”。

所有 Gate 仅使用 development validation；OPV2V-W test 保持冻结，不能用于这一阶段调假设。
