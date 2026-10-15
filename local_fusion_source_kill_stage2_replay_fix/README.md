# K2 数值复现检查修复

原 K2 把整段历史记录逐字段完全相等设成了工程门禁。已诊断的 fog/same/frame0 有 114 处连续数值变化，最大约 1.67e-6；离散字段、动作和 full9 排序一致。当前环境重复前向完全一致，Stage 1 与 Stage 2 adapter 的结果也完全一致。差异在扰动执行前已出现，尚不能确定由哪个底层运算造成。

本目录修复这项额外的工程检查。沿用原 S0 `shared_reproducibility` 数值约定（`reproducibility.POLICY`，atol=rtol=2e-5），并强制每帧：

- 只有 key 的第 7/8 项、focal score/IoU 和 proxy 的命名连续坐标允许数值变化；这些坐标必须有限且满足原 S0 的 numeric_difference 检查。
- 所有离散字段、来源、动作顺序、ROI 元数据、A/B/E 分组、合法性和所选动作完全一致。
- 所有 shortlisted actions 加 KEEP 的 full9 两两排序及并列关系完全一致，包含不合法动作；A_ONLY 合法动作另核对。
- 不舍入、不改 action key、不改 GT，不改 §42 的任何科学门槛。

S0 policy 原来用于数值复现，本修复明确把同一数值检查用于命名的连续轨迹坐标；这项工程适配及其字段范围写入新 RUN。每帧保存完整新旧轨迹和逐项差异；任何离散/排序变化都拒绝。本工具不把单帧诊断外推为其他帧已经通过。

## 保持旧任务运行

只同步这个新目录。不要同步或修改原 `local_fusion_source_kill_stage2/`、Stage 1/P0、AI_CONTEXT、冻结模型或旧 RUN。新目录不在原 Stage 2 的源码哈希集合中，不会改变正在运行的 K1。

新 RUN 继承旧 RUN 的全部科学协议与输入。新增修复代码、诊断文件、原 S0 policy 和环境的哈希加入新协议；必须使用诊断时同一软件和 GPU 环境。resume 校验代码、输入、环境和已完成产物。旧 K2 缓存不复用；失败记录保留。

## 运行流程

先在修复 GPU 上调用原 Source FULL evaluator，完成四天气 Same/Task 的 Shared 与 Oracle AP30/50/70 复现，容差仍是 1e-6。这里仅运行 Source，不运行 null/calibration。全部通过后调用原 K2 worker，保留原八项扰动、样本门槛、bootstrap、报告和判定。

通过 AST 仅替换原 evaluate_frame 中已识别的整记录相等检查。原文件和模块全局变量不改；如果原代码该位置发生变化，适配器拒绝执行。

首次启动（后台，可能耗时较长）：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
export ROCR_VISIBLE_DEVICES=3
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export PARENT_RUN=/data/cjm/datasets/logs/source_kill_stage2_20261014_143004
export DIAGNOSIS_DIR="$PARENT_RUN/k2_diagnostics/20261014_070743_547924_3927511"
unset RUN
sh local_fusion_source_kill_stage2_replay_fix/launch.sh
```

记录输出的新 RUN，查看进度或恢复：

```sh
tail -f <NEW_RUN>/driver.log
python -m local_fusion_source_kill_stage2_replay_fix.status --run <NEW_RUN>
RUN=<NEW_RUN> sh local_fusion_source_kill_stage2_replay_fix/launch.sh
```

恢复须保留相同 PARENT_RUN、DIAGNOSIS_DIR、GPU 和软件环境。启动和每个新计算单元均受原 resource stop 限制。

## 汇总原 K1 与新 K2

K2 完成时尝试读取原 RUN 的 K1；若 K1 仍运行，仅提示等待。两边完成后执行只读输入的 CPU 汇总：

```sh
python -m local_fusion_source_kill_stage2_replay_fix.driver --run <NEW_RUN> --summarize-only
```

它核验原 K1 的完成凭证及产物哈希，保留 K1 原 identity，不伪造为新 RUN 的结果。使用原 final_verdict 停止逻辑。输出 STAGE2_FINAL.json/md；K3 保持冻结。汇总不启动 GPU worker，也不重启原 K1。

新目录主要输出：PROTOCOL.json、K2_REPLAY_SOURCE_SNAPSHOT.json、source_reproduction/（原 AP 门禁）、k2/（原 K2 输出及逐帧 *.reproduction.json）、repair_manifest.json、driver.log、STAGE2_FINAL.json/md。完整结果保存在服务器外部日志目录。

本地只运行纯 NumPy/虚构记录的单元测试和静态检查；实际 Source/K2 必须在服务器执行。
