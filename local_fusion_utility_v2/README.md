# 方案一修正版 v2

本目录独立于 `local_fusion_utility/`。旧代码、旧缓存和旧权重保持原样；v2 禁止混用旧缓存/权重。

## 修改与限制

1. 区域摘要和动作共享固定尺度0网格边界。100行划为13个区域时，最后区域只统计96–99行；在尺度1/2按相同物理位置统计，padding不参与平均/最大值。
2. 每帧每分支采样3个区域，优先覆盖活跃、低分、背景三类，各区域评价全部可用peer。无某一类时用其他类补足。采样不读取GT。源数不同导致标签生成成本不同：P个peer时，每分支3P个动作，两分支6P个动作；旧版每分支固定8个。记录缓存行数可审计实际成本。描述量用float32缓存，推理同精度。
3. checkpoint仍依据完整validation缓存回归损失选择，以保持训练设置可比；门槛则改为完整validation上的真实多区域联合执行、后处理和AP。所有peer参与每个位置的竞争。每种方法只有一个跨天气门槛。
4. 预先固定验证约束：Clean AP70最多下降0.001（0.1个百分点）；每个验证条件丢失TP不超过baseline TP的1%；新增FP不超过baseline TP的1%。在通过约束的候选中最大化四条件平均AP70，平分时选修改更少者。KEEP_FULL作为显式候选，总能入选；如果所有修改策略不合格，就保留原模型并仍自动完成测试。约束仅用于验证选择，不保证独立测试不退化。
5. 配置、阈值、checkpoint均在benchmark之前固定。测试自动核对校准使用的checkpoint哈希。resolved_experiment.yaml保存完整训练options，避免此前雾表相对/绝对路径导致的协议错误。

原22维描述、网络、三输出损失、冻结范围、8轮训练、随机种子、模拟天气设置及尺度0+1动作保持一致。此版首先修正对应关系与选择流程，未同时引入更强天气或新的融合结构。仍使用单动作训练标签；联合推理的风险通过完整validation选择约束，不声称联合训练已实现。

## 一次启动，全部完成

服务器命令（无需预先创建RUN目录）：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export ROCR_VISIBLE_DEVICES=0
export RUN=/data/cjm/datasets/logs/local_fusion_utility_v2_$(date +%Y%m%d_%H%M%S)
nohup bash local_fusion_utility_v2/run_all.sh > "${RUN}.log" 2>&1 &
echo "PID=$! RUN=$RUN"
```

脚本按以下顺序执行，任何阶段失败即停止，不会带着失败状态进入测试：

1. 必须全部通过的CPU单元测试（缺依赖会失败，不会跳过后继续）；
2. 完整OPV2V train：clean + 在线混合模拟天气生成标签；
3. 完整OPV2V validation：生成同协议标签；
4. utility及loss-gain分别训练，每个8轮；
5. 完整validation clean/fog/rain/snow：所有门槛候选的联合输出评价，确定统一门槛；
6. development：用固定门槛完整评价上述四条件，保留上一轮仅尺度0、仅尺度1及去KEEP的执行消融；这些消融复用主权重，不冒称分别重训；
7. benchmark：自动评价OPV2V clean test及固定OPV2V-W fog/rain/snow test，关闭在线天气；
8. 汇总 `all_results.json`。

第5步明显比旧的缓存门槛扫描慢，但多个候选共享同帧特征编码，无需逐候选重跑点云骨干。第6步是同一验证集上的选定方案复核，不是新的独立测试。Stage-3数据不进入这些步骤。正式测试沿用已使用过的benchmark，因此属于后续版本比较；不把它描述为从未观察过的新测试集。

默认前端：`/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/{config.yaml,net_best_validation.pth}`。
可用环境变量 `FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT`、`PY`、`CONFIG` 覆盖；`ROCR_VISIBLE_DEVICES`设置一张可用HCU。配置应在运行前固定，不根据OPV2V-W结果重选。

## 进度与结果

```bash
tail -f "${RUN}.log"
# 主日志显示当前阶段；每个阶段有独立日志，例如：
tail -f "$RUN/prepare_train.log"
tail -f "$RUN/calibration.log"
tail -f "$RUN/benchmark.log"
```

阶段/耗时/退出码：`pipeline_status.json`。
主方法与对照权重：`utility/best.pth`、`loss_gain/best.pth`。
候选门槛及验证约束：`calibration/calibration.json`。
模拟天气验证：`development/protocol.json`。
固定测试：`benchmark/protocol.json`。
最终汇总：`all_results.json`。

脚本拒绝覆盖已有RUN；不宣称支持整条流水线自动恢复。缓存生成器仍支持显式 `--resume`，训练器不支持断点续训。首次正常完成无需手动启动任何后续阶段。

现有时延字段只统计动作融合/检测头的一部分，不能当作端到端延迟；通信字段为raw特征载荷大小，不包含网络序列化头。正式性能/成本论文表应另按统一协议计时和记账。
