# v3 局部动作选择：小型决策验证

本流程**不训练新的融合网络，不访问 OPV2V-W**。它使用已经训练好的 v3 residual checkpoint，检验能否仅凭预测时可见的信息，挑出值得在某个 BEV tile 启用的 v3 修改。

## 验证对象

- 原动作：整帧使用冻结 AttFuse。
- 候选动作：在单个 tile 的尺度 0、1 使用 v3 residual 输出，其余位置与尺度保持原 AttFuse。当前只验证这一种动作，**尚不验证选择哪个 peer**。
- 无 GT 候选生成：以原融合及逐来源分类响应组成活动图；按活跃、低分和背景三层抽取 tile。每帧最多六个 tile。这个分层并不保证覆盖所有目标。
- 训练标签：在官方 train 场景中，单独执行候选动作，比较 NMS 后的恢复目标、原 TP 损失和新增 FP 身份。GT 仅用于制标签。
- 选择器：用每个 tile 的来源特征摘要、原/候选预测摘要和 v3 gate，预测动作是有害、无变化还是有益。推理和动作排序不接触 GT。
- 对照：简单 peer-vs-full 置信度差、v3 gate 大小。在每帧相同动作数 `top-1/2/4` 下比较，并另列选择器允许 KEEP 的结果。
- 验证：官方 validation 中按场景取样，分别在 Clean、在线模拟 Fog/Rain/Snow 上重新生成候选，执行选出的多个 tile **整帧联合推理**，报告 AP30/50/70 与恢复／误伤。单动作标签与联合结果分别报告，不能互相替代。

默认从 train 随机抽至多 32 个场景、每场景 8 帧；validation 至多 12 个场景、每场景 8 帧。抽样只看场景边界与固定种子，不看 GT。train 和 validation 是原数据的不同场景目录。若训练候选中有益或有害动作少于 10 个，结果标记为证据不足；不应据此否定方案 A。小子集 AP 只作方向参考。

## 服务器后台运行

在代码已经同步、原 OPV2V 环境可用的服务器上：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export V3_RUN=/data/cjm/datasets/logs/local_fusion_v3_20260922_182832
export RUN=/data/cjm/datasets/logs/decision_pilot_$(date +%Y%m%d_%H%M%S)
nohup bash local_fusion_decision_pilot/run_all.sh > "${RUN}.launcher.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$RUN"
```

完整流程会先跑轻量单元测试，再生成 train/validation 单动作缓存、训练选择器、完成四条件验证。中途失败可在同一 `RUN` 上设置 `RESUME=1` 重启；缓存的配置、模型及源码哈希必须相同。结果在 `decision_pilot_results.json`，过程在 `progress.json`。没有使用测试集，也不会自动启动后续完整训练。

首次运行可能花费数小时，主要开销是每个训练／验证帧对若干 tile 重跑检测头和 NMS。若服务器前端路径不同，可设置 `FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT`；若 v3 路径不同，设置 `V3_RUN`。

## 判断方式

先检查候选有益／有害样本数量，再比较相同 top-K 下三种排序的有益率、误伤率与整帧结果。若选择器只靠减少动作数保护 Clean，或与简单排序相当，就没有证据支持扩大方案 A。OPV2V-W 已在多轮研究中使用，本流程不会把它当独立测试。
