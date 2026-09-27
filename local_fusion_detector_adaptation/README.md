# 融合后检测端适应：F 与 F＋D 小型对照

这次实验只检验一个问题：同一份 v3 residual 融合特征交给检测端后，让检测端跟着训练，是否比只继续训练融合有稳定净收益。

- **F**：继续训练 v3 融合；独立复制的融合后 `deblocks`、`cls_head`、`reg_head` 全部冻结。
- **F＋D**：从相同权重起步，继续训练相同融合，同时训练这套融合后检测端。所有 BatchNorm 均处于 eval，运行均值和方差固定；卷积权重与 BatchNorm 的可学习参数可以更新。
- 原基础模型在两组中始终冻结，仍负责 GSPR、各车编码和来源置信度。融合后检测端是独立副本，不会改变来源信号。

两组使用同一官方 train 场景与帧、同一 shuffle 种子、Clean/在线 mixed 天气配对、PointPillarLoss 与 v3 权重修改惩罚、同一融合学习率和固定轮数。单车帧两组都计算损失但跳过更新，使优化步数相等。F＋D 的额外计算和参数量会单独体现在运行耗时；同样步数并不等于同样 FLOPs。

默认选最多 32 个训练场景、24 个 validation 场景，每场景最多 10 帧，训练 2 轮。场景与帧只按固定种子随机抽取，不按目标或失败类型挑选。没有足够场景时使用现有场景，实际索引记入 `protocol.json`。每组只保存最后一轮，不按 validation 反复挑 checkpoint。

验证沿用 BEV 平面 IoU、帧顺序累积的 AP 协议，逐条件报告 AP30/AP50/AP70、相对原 baseline 与相对 F 的恢复、原 TP 损失和新增 FP 身份。`decision_results.json` 的扩大门槛预先固定：F＋D 相对 F 三天气平均 AP70 至少 +0.005，至少两天气为正；Clean AP70 和四条件 AP50 相对 F 与原 baseline 均不能低于 0.001；原 TP 损失、新增 FP 身份相对 F 与 baseline 各不能超过参照 TP 的 1%。这是投入门槛，不是统计显著性检验。

本轮只运行官方 validation 子集和在线模拟天气，不访问 OPV2V-W。validation 与历次方案曾共用，因此也不能称完全未接触的新数据。若小型实验过门槛，再换一个配对种子复验；方向一致后才考虑完整验证和一次正式评价。

## 服务器后台运行

在原服务器项目目录和 opencood 环境：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_$(date +%Y%m%d_%H%M%S)
nohup bash local_fusion_detector_adaptation/run_all.sh > "${RUN}.launcher.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$RUN"
tail -f "${RUN}.launcher.log"
```

可覆盖 `PY`、`FRONTEND_ROOT`、`FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT`、`V3_RUN`、`V3_CONFIG`、`PILOT_CONFIG`。`V3_RUN/residual/best.pth` 必须存在；`RUN` 必须是新目录。`protocol.json` 保存起点 checkpoint 哈希、训练索引、源码快照哈希和门槛；`train_history.json` 记录每轮帧数、优化步数和耗时；`decision_results.json` 保存两组原始指标与决定。
