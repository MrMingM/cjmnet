# Source Utility Stage-2 Precheck

本轮只盘点已有数据：P0-A 检查离散最佳动作是否必须选唯一 CAV；P0-B 分开统计 Same/H1 和 Task/H2 正例与 scene 覆盖。K1-0 只记 **A-dominant descriptive decomposition**，无预注册 PASS/FAIL。没有 K1、K2 perturbation、训练、模型推理或正式 test 访问。K3 继续冻结。

## 输入与当前缺失

请同步本目录（含 source_snapshot.json）、原 Stage 1 代码、原 S0 `counterfactual.py/common.py`，以及根目录 `logs/k/` 的六个原 Stage 1 MD/JSON。历史文件保持原样；本目录只读取它们。root AI_CONTEXT.md 记录当前任务边界。

现有 K0 JSON 有 group_count、beneficial_set_scene_coverage 的精确 group/scene 计数，能完成 H1 样本盘点；没有 per-group 全部动作的 discrete6 与 source 集合，无法由 top outcome8 tie multiplicity 推回最佳 discrete6 source redundancy。现有 K1-0 JSON 有 Same/Task recovered 和 selected counts，缺少逐目标当前状态比较；其 frame_artifact/frame_sha256 指向服务器原记录。

因此完整 P0-A/P0-B 需要原 Stage 1 RUN 及其引用的 S0 cache。全部可以 CPU 完成，只用 Torch 的 `torch.load(map_location='cpu', weights_only=True)` 读取已有 .pt，不加载 checkpoint、不实例化模型、不构造数据集、不生成 labels。GPU 可见设备全部清空。

## 统计定义

- P0-A 的 beneficial 与 Stage 1/K0 完全一致：原 action_keys 的 non-KEEP `[:6] > KEEP[:6]`。
- 最佳集仅按 discrete6 比较，不让 score/IoU/KEEP simplicity 打破并列；保留所有 equivalent-best actions。
- source 为当前帧 CAV 索引。single:i 和 query:i 属于同 source 的不同 action family；query:i 本身仍使用所有来源。
- single only/query only/both 是互斥三类；same-source cross-family 可以与 cross-source redundancy 共存。两者不相加。
- `MODIFY_STABLE_SOURCE_AMBIGUOUS` 只表示当前缓存中明确离散优于 KEEP，且最佳集跨 CAV；扰动稳定性未检验。
- associated/background 分开，cls/reg 分开；每个 group 保留 frame/proposal/task/focal association，不能跨 focal 合并。
- 所有 split 和 adverse pooled 都带 scene 覆盖、每 scene 正例与最大占比；scene ID 为 train:N/validation:N，避免碰撞。
- adverse pooled 为天气出现次数之和，equal-weight 为三个天气的指标均值；后者不是新样本。另报去重 `(split, frame, proposal, focal)` 正例数。frame/target 在天气之间相关，pooled 不代表独立泛化。

## H1/H2 的真实证据边界

H1 主表是 S0 的 cls-only/reg-only beneficial groups。两者不能相加推导联合 Same。S0 没有联合替换 cls/reg 的 candidate outcome，本轮不生成；该 candidate-level 计数记 unavailable。原 Same Oracle 可给另一张真实联合 Same GT-target/current-state 正例表，统计单位明确不同。

H2 分两层：

1. `Task recovered − Same recovered` 是独立终态路径的恢复数量净差，自动从原 JSON 重算。它不提供 Task-only GT 身份，也不等于同状态正例数。
2. 读取原 Task `current_state_audit.actions`，按保存的当前 KEEP key 重新核对 A/B/E 和 winner。若 Task 最佳 discrete6 严格优于 **同一 Task state** 中 KEEP+所有合法 Same-action-name 候选的最佳 discrete6，则计为 true same-state task improvement。FULL/A_ONLY 分开报告，不把两条 trajectory 的动作当成独立额外样本。这是已保存 shortlist 中的真实 outcome 比较，未追加完整 NMS、未重做 shortlist。

Task separated 沿用原 `cls_source != reg_source`，并单列：双 non-KEEP 跨 CAV、同 CAV 跨 family、KEEP/来源分开。selected separated 数量不作为 H2 真正例。

没有 train Task Oracle，就保持 train true n=null；train+validation 的完整 true n 同样为 null，同时单列 observed_validation_*。不能把已知 validation 加未知 train 当成真实 pooled 总数。

辅助 proxy 定义为两任务都分别离散优于 KEEP且最佳 action/CAV 集合不相交。单独叫 task-separation proxy，不能推断联合动作的实际 outcome，也不进入真实正例总数。

## 本地允许的检查与部分报告

```sh
python -m unittest local_fusion_source_kill_stage2_precheck.test_core
python -m local_fusion_source_kill_stage2_precheck.test_core --smoke
python -m local_fusion_source_kill_stage2_precheck.driver --static
python -m local_fusion_source_kill_stage2_precheck.driver --summary-only
```

smoke 只用纯合成 fixtures，不读取科研 cache/数据/模型。summary-only 只读取仓库内六个原 Stage 1 报告，在 logs/k/ 生成明确标记 partial_requires_server_cache 的 P0 报告。P0-A 不可用的比例为 null；不存在臆造 source redundancy 或 Task 真正例。

## 服务器完整 CPU 运行

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
sh local_fusion_source_kill_stage2_precheck/launch.sh
```

默认后台运行，输出 RUN/PID/LOG。RUN 为 `/data/cjm/datasets/logs/source_kill_stage2_precheck_YYYYMMDD_HHMMSS`；大逐 group/target JSONL 留在 RUN，不进 Git。

程序在 `/data/cjm/datasets/logs/source_kill_stage1_*` 中按六个原结果文件的 SHA256 和 complete manifest 匹配 Stage 1 RUN，不猜最新路径。如存在多个相同副本或搬迁目录，请明确指定：

```sh
STAGE1_RUN=/data/cjm/datasets/logs/<原Stage1_RUN> \
sh local_fusion_source_kill_stage2_precheck/launch.sh
```

同一 RUN 防双启动，完成报告哈希通过才复用；源/输入身份变了就使用新输出目录。所有输入 realpath 先拒绝 test 及别名，cache/record 逐文件核对原 hash。不要求 .git。CPU 统计完成后到此停止，不自动进入任何下一阶段。

## 输出与人工决策

七个精简报告：P0_SOURCE_REDUNDANCY.{md,json}、P0_DATA_ADEQUACY.{md,json}、P0_STAGE2_PRECHECK_FINAL.{md,json}、DATA_ADEQUACY_RECOMMENDATION.md。JSON 中保留每 scene counts、weather 分布、去重数、同状态 Task 真正例和独立 proxy。还会记录 P0_MANIFEST.json。

建议列只做敏感性盘点：K2-Same n=50/100/200，K2-Task n=20/30/50，K3-Same train n=100/200/500，K3-Task train n=50/100/200；覆盖和集中度分别列多个候选。没有冻结、自动 PASS/FAIL、训练授权或统计功效承诺。不继承每天气 100。最终 minimum-n、scene 限制、pooled 单位、inconclusive 处理和 K1/K2 协议由用户人工决定。
