# 方案 B：保护原正确目标的小型决策验证

这是**训练策略验证**，不是最终方法实验。它从同一份 v3 residual checkpoint 复制两套完全相同的权重：`continued_v3` 按原损失继续训练，`protected_v3` 额外约束冻结 baseline 已检对目标所对应的分类 logit 和框回归。两套模型使用同一批官方 train 场景、相同轮次与学习率；都只在最后一轮评价，不挑最好轮次。

训练时先执行冻结 baseline 的整帧 NMS，用 GT 确认哪些最终框是真正 TP。只在能够追溯到这些 TP 的正样本 anchor 上做保持约束；无法追溯的 TP 记录为未覆盖，不猜测其监督位置。GT 仅在训练制掩码和 validation 评分使用。推理时仍是原 v3 residual 网络，不增加 peer 通信或额外检测模块。

默认官方 train 随机取 16 个场景、每场景 6 帧，Clean/混合模拟天气成对训练 2 轮；官方 validation 独立取 8 个场景、每场景 6 帧，分别检查 Clean/模拟 Fog/Rain/Snow。抽样只依据场景边界和固定种子。**不会访问 OPV2V-W。** 小子集 AP 波动很大，只用于决定“这个保护策略是否值得做完整实验”，不能当最终性能。

这里的 validation 与本次小样本继续训练场景分开，但原 v3 起点曾用官方 validation 选 checkpoint；因此它也不是对整条研究流程从未接触过的独立测试。

## 服务器后台运行

将本目录同步到服务器后，沿用 v3 环境：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export V3_RUN=/data/cjm/datasets/logs/local_fusion_v3_20260922_182832
export RUN=/data/cjm/datasets/logs/preservation_pilot_$(date +%Y%m%d_%H%M%S)
nohup bash local_fusion_preservation_pilot/run_all.sh > "${RUN}.launcher.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$RUN"
```

如前端文件路径不同，可设置 `FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT`。流程会先跑轻量测试，再完成小样本训练和四条件 validation。结果在 `decision_results.json`，逐条件结果在 `validation/<weather>/summary.json`；`protocol.json` 保存抽样索引、源码和权重哈希。

判断时先看 `train_history.json` 里的 `teacher_tp` 与 `protected_anchors`：若保护覆盖率极低，实验不能说明 B 无效。再比较两个继续训练模型相对冻结 baseline 的 Clean 原 TP 损失、新增 FP、AP30/50/70，以及 Snow 恢复和 AP。若 Clean 有所保护但 Snow 增益几乎消失，或两模型没有稳定差别，就不应直接扩大为完整研究方案。正式结论仍需预先锁定设计后在未反复使用的独立场景上复验。

单车帧没有可调整的来源权重，因此不产生融合模块梯度。训练会照常计入该帧的损失与保护覆盖统计，并在 `train_history.json` 记录 `skipped_no_grad_branches`；只有本帧存在实际梯度时才执行优化器更新。
