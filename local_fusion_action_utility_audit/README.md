# Direction B Action Utility Audit

这套诊断回答：同一个候选框换成不同来源的分类或回归输出，为什么会检得更好；这些选择能否仅用推理可见信息学会，并在完整后处理中带来 AP。

只在当前 `main` 添加独立代码。GSPR、detector、B0 Shared 和 v3 全冻结。训练模块只有每任务每消融一个 `torch.nn.Linear(d, 1)`，没有隐藏层。不要在本地运行科研数据、模型前向、训练或 AP 实验。

## 一次后台启动

将本目录同步到服务器上同版本仓库，保留既有依赖及 B0 权重：

本地开发仍要求在 `main`。服务器有 `.git` 时核验实际分支及 commit；只有源码、没有 `.git` 时，使用随目录同步的 `source_snapshot.json`，核验它来自本地 main 且本实验每份源码的 SHA256 一致。旧模块、冻结源和 checkpoint 仍接受原有校验。Git 存在但报权限或 HEAD 错误时停止，并显示原始 Git 报错。

以后本地修改本实验后，同步前更新源码记录（只读取 Git 和源码，不运行实验）：

```sh
python -B -c "from local_fusion_action_utility_audit.common import write_source_snapshot; print(write_source_snapshot())"
```

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
export ROCR_VISIBLE_DEVICES=2
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
sh local_fusion_action_utility_audit/launch.sh
```

打印 `RUN`、`PID`、`LOG`。默认全部输出位于 `/data/cjm/datasets/logs/action_utility_audit_YYYYMMDD_HHMMSS`。查看进度与完成状态：

```sh
tail -f <RUN>/driver.log
python -m local_fusion_action_utility_audit.status --run <RUN>
```

最后一条成功时打印 `S0 complete`、`S1 complete`、`S2 complete`、`S3 complete`、`FINAL complete`，返回码 0；还会检查最终完整性阶段。最终阅读 `<RUN>/FINAL_RESULTS.md`。

中断后复用同一个 RUN：

```sh
RUN=<RUN> sh local_fusion_action_utility_audit/launch.sh
```

完成的天气直接复用，未完成天气按帧继续。每个已完成帧的缓存哈希都必须一致；probe 在 epoch 末保存 optimizer 恢复点，重做中断的 epoch。loader 每次仍按原顺序遍历，包括已完成帧，以保持在线天气和随机点处理一致。配置、源代码、commit、checkpoint 任一身份改变拒绝续跑，需新 RUN。每次启动和阶段切换核验源码及配置，加载模型时核验权重；最终报告必须等待 baseline 和 S0–S3 完成。进程锁避免两个 driver 同时使用一个 RUN。启动脚本的 PID 检查只是辅助，最终以文件锁为准。

可通过环境变量覆盖 `PY`、`RUN`、`CONFIG`、`V3_CONFIG`、`FRONTEND_ROOT`、`FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT`、`V3_RUN`、`V3_CHECKPOINT`、`B0_RUN`。默认 B0 为 `/data/cjm/datasets/logs/task_split_pilot_20260927_120637`。任何覆盖都要通过原 B0 contract 校验。

## 自动顺序

1. preflight：检查 main、环境、冻结源、checkpoint/source contract、resolved 数据路径、场景映射；禁止任何 `test` 路径，包括文件链接。
2. mandatory `test_core`：合成单元测试及 `sh -n`。
3. 完整 B0 baseline reproduction：四天气、原 validation indices、原 B0 `seed+1`、原帧顺序 AP30/AP50/AP70；绝对差超过 `1e-6` 停止。新标签尚未生成。
4. S0：Shared top256，固定 1.5 倍 proposal ROI；每 GT 关联 IoU 至少 .1 的最佳 Shared proposal；每帧最多 32 个无关联 proposal 按分数高/中/低三层采样。对每组每个 single/query 来源，完整执行 classification-only / regression-only 后处理。每个动作都重新从原 Shared 开始，保存完整后果，完整枚举，无旧 Oracle shortlist。
5. S1：分类线性 pairwise probe 的 fit、train calibration、validation；同时报告三种不训练 baseline 和竞争特征消融。
6. S2：几何线性 probe；相同协议，另报最佳/学习 cls 与 reg 来源相同率。
7. S3-A：GT 只限定关联 proposal 集，学习器决定来源及 KEEP/MODIFY；报告 Assisted-Cls/Reg/Task。
8. S3-B：所有 Shared proposal 都由学习器决定。Greedy margin>0；Conservative 使用已经固定的 train threshold。报告 Cls/Reg/Task Greedy、Task Conservative。
9. FINAL 与完整性检查。任一步失败停止，manifest 记录明确 stage 和错误，非零退出。

标签生成可能非常耗时：每组需要 `2 × (2*CAV)` 次完整后处理/NMS，背景也完整执行。日志给天气、帧、候选组进度。没有为了节约计算删掉 source 或用 GT 代理 shortlist 代替真实后果。

## 数据与 GT 边界

没有访问正式 test。训练和开发评价完全复用 B0 保存的 train/validation indices/scenes；使用 B0 在线天气协议，不能改成 fixed physics 数据后与原 B0 AP 强行相减。

train scenes 用固定 seed 分成约 80% probe-fit / 20% probe-calibration，同场景的所有天气在同侧。只有 fit scenes 计算标准化和训练权重。固定训练 30 epochs，最后一轮权重；validation 不挑 epoch、特征版本或阈值。primary features 在 YAML 中预先固定为 with_competition_features。

GT 使用仅限：S0 关联和结构化标签、S1/S2 标签评价、S3-A proposal 集限制、最终 AP/TP/FP 评价。特征函数只接收预测 proxies、Shared proposal/mask，GT 输入字段为 0。自动 schema 白名单和禁用标签词检查遇未知字段立即报错。cache 的 `labels` / `evaluation_metadata` 与 `features` 分开；S3-B 会重新提取 features 并核对一致，不从标签选动作。

同一 proposal 可以关联多个 GT：S0 保留每个 target 的 label 组，GT coverage 按 GT 计数；S3-A 对 proposal 去重，仅执行一次学习动作。background 指未被 association 选中，可能仍与某 GT 有重叠；它没有 focal GT，focal 字段为 null，全帧 recovered/lost/newFP 照常统计。

## 排序、阈值和冲突的明确含义

后果排序依次为：action TP 更多、lost 更少、newFP 更少、focal 检出、recovered 更多、总 FP 更少、focal score 更高、focal IoU 更高，同效优先 KEEP。原后果始终保存。非 KEEP 同效 source 的显示顺序固定，不以 source ID 训练分类。

pairwise logistic loss 用同一组里有序 action 对，不同 source 同效 pair 忽略；同效 KEEP pair 用 KEEP 简化偏好。`mean_action_regret` 按不同的**后果等级**排序成 0–1，最佳减选中值；同效来源 regret 为 0。JSON 另报 TP 缺口和多出的 lost/newFP。这是可解释的等级距离，不是 AP 差，也不承诺 score/IoU 的具体量纲价值。

Conservative 的候选 threshold 来自 train-calibration 正 margin 的预注册分位数。选择满足有益精度≥.8、每动作 lost≤.01、newFP≤.05、至少20动作的最小 threshold；无可行 threshold 时 KEEP_ALL（null）。两任务分别校准，合并后可能存在相互影响，因此 S3 仍需完整评价。所有 baseline、消融均报告；不按 validation 选择最好消融部署。

候选专属常量会在两个 action 的线性分数差中抵消。为让竞争特征真正影响 source ranking，额外使用预先固定的推理证据乘积和 source box 相对其他 proposal 的竞争量。没有学习深层表示，没有绝对 CAV ID 特征；仅 KEEP/single/query、is_ego、source count 和相对证据。prediction–prediction IoU 允许使用。

S3 先按 `max(cls_margin, reg_margin)` 降序，再 Shared score 降序、anchor_id 升序。联合 proposal 动作占用共同输出 cell；后续只修改未占用 cell。partial overlap 算一次 overlap rejection，另外保存 fully-rejected 数；accepted 与 overlap-rejected 可能同时增加。冲突规则在 Assisted 和 Proposal 都相同，不使用 GT 解决。

S1/S2 gate 使用 adverse mean pairwise≥.60、相对逐天气最强简单 baseline 的平均 regret 降低≥20%、至少2/3天气 regret 更低。S3 gate 使用 Conservative adverse AP70≥+.30pp、至少2/3正收益、Clean下降≤.20pp、adverse recovered>lost。`experiment.yaml` 保存完整预注册值，结果后不自动修改。

自动解释 A/B/C/D/E，并保留结果不明确的情况。C/D 用相同 Greedy 策略比较 Assisted-Task 与 Proposal-Task-Greedy，避免混入保守阈值的影响；S3_PASS 仍按 Conservative 判定。E 提示联合动作交互以及标签/AP差异；不能由 Assisted-Task 无收益直接证明唯一根因。不同来源率也受同效来源影响，不能当作新增 TP 数。

## 文件用途

| 文件 | 用途 |
|---|---|
| `__init__.py` | 独立 Python 包 |
| `experiment.yaml` | 固定候选、抽样、训练、校准和判定门槛 |
| `source_snapshot.json` | 本地 main 导出的基础 commit 与本实验源码哈希，供无 .git 的服务器副本核验 |
| `common.py` | 原模型/loader复用、路径与冻结检查、原子IO、manifest |
| `features.py` | 推理特征、schema、防GT泄漏、预测几何与竞争量 |
| `counterfactual.py` | 同Shared单动作试验、完整后果、排序标签 |
| `linear_ranker.py` | 单层排序器、标准化、baseline、train校准、评价 |
| `s0_counterfactual.py` | baseline复现、S0缓存、S0汇总 |
| `s1_cls_rank.py` | 分类fit/evaluate入口 |
| `s2_reg_rank.py` | 回归fit/evaluate入口 |
| `s3_replay.py` | Assisted/Proposal回放、固定冲突处理、AP与误伤 |
| `summarize.py` | 最终表格和科研判断 |
| `pipeline.py` | 按阶段自动执行、恢复和完整性检查 |
| `driver.py` | 合并stdout/stderr到driver.log |
| `status.py` | 五项完成状态和返回码检查 |
| `test_core.py` | 协议、合成张量和shell测试 |
| `run_all.sh` | POSIX服务器流程入口 |
| `launch.sh` | POSIX后台启动、RUN/PID/LOG |
| `README.md` | 协议、用途、命令和判读 |

输出：`S0/S1/S2/S3_RESULTS.json/.md`、`final_results.json`、`FINAL_RESULTS.md`、`protocol.json`、`manifest.json`、`driver.log`、`linear_cls.pt`、`linear_reg.pt`、两个无竞争消融权重、`feature_normalization.json`、`feature_schema.json`、`baseline_reproduction.json`、逐帧 `cache/`。所有大缓存和 probe 权重在 RUN，仓库只保存源码/配置/文档/测试。

本地允许的检查：AST语法、配置解析、`test_core --static-only` 的纯合成协议检查和 `sh -n`；服务器流水线还必须通过 Torch/OpenCOOD 合成测试及真实 B0 baseline reproduction。静态通过不能写成远程实验或 AP 已验证。
