# 方案 A 第二轮：候选位置与来源动作的针对性复验

第一轮训练动作 2772 个，其中 2742 个按预设后果统计为零；Snow validation 的收益真实存在，但学习选择器尚未稳定超过简单排序。本轮**保持第一轮的场景与帧抽样种子、v3 checkpoint、模拟天气及评估口径**，只改变两件事：

1. 在每帧仍选最多六个 tile，但以推理可见的**分类响应变化、来源与 full 的分歧**来优先排序；活跃、低分、背景分别保留名额。不看 GT、IoU 或事后动作后果。
2. 对每个 tile 比较 v3 residual，以及最多两个 peer-query AttFuse 动作：一个 peer 由分类分歧选出，另一个由回归分歧选出。两个规则若选中同一 peer，自动去重。peer-query 仍融合多车特征，**不是直接复制该 peer 的框或特征**。

所有动作先单独经过检测头与 NMS，用训练 GT 标记有益／有害／无变化。选择器只看动作前可获取的特征及该动作的**预 NMS**分类、回归响应。validation 上选择器、peer-vs-full 置信度差、动作响应变化使用**相同的候选池与每帧相同的不同 tile 数量**；同一 tile 不能同时选两个动作。另列选择器允许 KEEP 的结果。选中的动作最后一次联合执行并作整帧 NMS，报告 AP30/50/70、恢复、原 TP 损失和新增 FP 身份。

输出 `candidate_breakdown` 按 residual／peer-query 及活跃／低分／背景统计非零动作。它能帮助判断第一轮的稀疏信号主要来自候选位置还是动作类型，但由于两者同时改变，**不能单凭两个版本的差值做严格归因**。这仍不是完整方案 A，不使用 OPV2V-W，也不能把小子集 AP 当独立效果评价。

## 服务器后台运行

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export V3_RUN=/data/cjm/datasets/logs/local_fusion_v3_20260922_182832
export RUN=/data/cjm/datasets/logs/decision_pilot_v2_$(date +%Y%m%d_%H%M%S)
nohup bash local_fusion_decision_pilot_v2/run_all.sh > "${RUN}.launcher.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$RUN"
```

脚本先运行单元测试，再做 train/validation 单动作缓存、训练轻量选择器、四条件联合验证。运行中看 `$RUN/progress.json` 和 launcher log；结果在 `$RUN/decision_pilot_v2_results.json`。中断后可用原 `RUN` 加 `RESUME=1` 重启。前端路径变化可设置 `FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT`。

## 时间与判读

场景帧数与第一轮相同：实际 validation 为每条件 72 帧。本轮每帧最多 6×(1+2)=18 个单动作，第一轮最多 6 个；因此单动作制标签上界约为第一轮 **3 倍**。选择器训练很短，主要时间花在检测头、后处理与缓存。若硬件和数据加载条件相同，先按**第一轮实际耗时的约 2–3 倍**安排；没有第一轮 `progress.json`，无法给出可靠的小时数。运行 20 帧后，可用日志中的已耗时与动作数估算剩余时间。

先看动作池中有益与有害动作是否增多，尤其是 Fog/Rain 与 Clean；再看固定 top-1/2/4 下是否真正优于两种简单排序。若 `learned_keep` 仍主要丢掉有益动作而保留有害动作，或新动作只有误伤增多，则不应扩大方案 A。新增 FP 身份不是总 FP 净增量。
