# 动作排序标签对照实验

只回答一个问题：用明确的检测后果训练来源排序，能否比包含 focal score/IoU 偏好的原标签更好地控制修改、误伤，并兑现为 AP？本实验没有增加网络、特征、候选、来源或冲突策略。

## 保护原实验与同步时机

所有新增内容只在本目录。不要同步整个仓库、更新旧模块、更新旧源码快照或 commit/pull 正在运行的服务器目录。仅同步 `local_fusion_action_label_control/`；它不在原实验记录的依赖哈希中。建议等原 S0–S3 全部结束后再同步并运行完整对照，避免原流程的提交/源码身份检查或 GPU/CPU 争用。

如果希望提前用 CPU 做 offline，可在原 S0、S1、S2 完成后仅同步新目录，保留所有既有源码原样；原 FINAL 无需完成。原 S0 未完成时只能先运行 preflight 检查准备状态，缺失阶段和文件会列明并非零退出，绝不重新生成 S0。完整回放还要求原 baseline reproduction 通过；原 S3 尚未完成允许生成本次回放，但最终完成前必须补上原 S3 核对。再次使用同一 RUN 的 `MODE=all` 自动复用本次已完成阶段。

服务器可以没有 `.git`。新入口只按内容哈希校验原依赖、配置、权重和缓存；提交号仅记录为可选元数据，不是运行门槛，不伪造旧 commit。原协议单独保存在新 protocol 的 `source_protocol`，新代码身份单独记录为 `control_source_hashes`。原依赖哈希不匹配时停止；请从原运行匹配的代码备份恢复那些文件，不能删除校验或重写旧 protocol。

## 两标签的精确差异

OriginalLabel 直接 import 原 `action_keys/pairwise_preferences/outcome_ranks`，原排序为：action TP、-lost、-newFP、focal 检出、recovered、-actionFP、focal score、focal IoU，最后同效 KEEP。权重直接加载原 `linear_cls.pt/linear_reg.pt` 主版本，未经重新训练。

DetectionOutcomeLabel 的检测后果键严格为：

```python
(outcome['action_tp'], -outcome['lost'], -outcome['new_fp'], -outcome['action_fp'])
```

字典序排序，不使用 focal score/IoU 或 CAV/source ID。训练偏好最后加 KEEP 简化位；非 KEEP 同效不生成训练对。检测后果 regret 使用不同后果键的归一化等级距离，排除 KEEP 简化位，因此同效来源 regret=0。它仍不是 AP 标签，不能保证单动作最优组合成整帧最优。

输入列固定为原 `with_competition_features`，无消融。原标准化参数直接复用，不重新计算。新组按原 seed、零权重/零bias初始化、单个 `torch.nn.Linear(d,1)`、AdamW、学习率、weight decay、epoch、分组缓冲批规则和最后epoch策略训练；绝不从原已训练 probe 微调。每轮报告 pair 数、更新次数和 loss；标签变化可能改变训练预算。原历史只保存 pair/loss，更新次数按已核验原样本分组和批规则恢复，并标明来源。没有可训练偏好时零权重、KEEP_ALL，并明确不能记为成功。

两组 Conservative 重新按共同检测后果键严格优于 KEEP 定义“有益”。只使用原 train-calibration scenes，沿用原分位数、最少动作、有益精度、lost/action、newFP/action 和最小可行 margin 规则。无可行 threshold 则 KEEP_ALL。原 threshold 保存在历史参考字段，不参与公平对比。Greedy 严格分数大于 KEEP 才改。

## 缓存、GT 与复现边界

SOURCE_RUN 必须显式指定，永远只读。RUN 必须是 `/data/cjm/datasets/logs/` 下独立目录；拒绝同路径、祖先/后代嵌套以及解析链接后的危险关系。新流程不构造原可写 Manifest，也不调用含 Git 校验及 Manifest 初始化的原 Runtime 构造函数。

离线阶段只读原缓存的推理特征、proposal、source_names 和结构化后果。原 frame 缓存必须在 manifest 中 complete 且哈希一致；首次 preflight 的 S0 哈希索引固定在新 protocol 身份中，即使原缓存及 manifest 一起变化也拒绝续跑。还核验协议、scene-level fit/calibration 划分、特征 schema、两任务标准化、权重身份、来源顺序以及配置和 checkpoint 身份。缺少带哈希的可靠原输入直接停止。

GT 仅用于标签/评价；Assisted 回放额外允许 GT 限定关联 proposal 集。推理特征、来源选择、KEEP/MODIFY、margin 排序和 Proposal 冲突决策中的 GT 字段为0。没有读取正式 test。路径和文件链接检查复用原接口，保留官方 train/validate 及原在线天气，不改为固定离线天气。

标签审计同时报告 train/validation、天气、cls/reg、关联/背景及全体。互斥动作分类按 TP 数变化→同数量下目标身份变化→FP变化→focal匹配变化→仅score/IoU变化→保存后果同效依次判断。score/IoU变化仍可能影响AP，不能写成救回GT或直接称为无效动作。缓存未保存全部匹配身份列表，身份变化由 recovered/lost 计数识别，不虚构具体身份。每组最佳检测键保存于 `cache/label_audit/`，JSON提供索引；统计也给多GT关联的 proposal/标签组数。训练保留原 candidate–target 权重；唯一 frame–proposal 组成统计对该 proposal 的多个 target 标签等权分摊，总权重为1。

正式回放才启动 GPU。适配层直接继承原 Runtime 的 `loader/predict`，构造时复用原模型加载、B0 contract、v3 load 和冻结检查；没有 monkey patch。每帧只前向构建一次 Shared/source pool，两组共享；原 source 前向本身包含 Shared、single/query 分支。重新提取特征并与 S0 逐值比较，同时核对 Shared tensor 哈希与原 S0、baseline；之后执行七种方法：

- Shared
- OriginalLabel / DetectionOutcomeLabel × Assisted-Task-Greedy
- OriginalLabel / DetectionOutcomeLabel × Proposal-Task-Greedy
- OriginalLabel / DetectionOutcomeLabel × Proposal-Task-CommonConservative

直接复用原 `predict_actions/conflict_resolver/execute`，ROI、`max(cls_margin,reg_margin)`、Shared分数/anchor tie、first-owner cell及部分重叠处理完全相同。每种输出独立做原 postprocess/rotated NMS 和 frame-order AP。报告所有 AP30/50/70、相对Shared及配对变化（pp）、后果、动作量、KEEP、cls/reg比例、cells和部分/完全重叠拒绝。保存逐帧提议策略、接受cell索引、后果及AP统计，不保存巨量密集来源输出。

Shared 对原 baseline、OriginalLabel 两种 Greedy 对原 S3 的对应 AP30/50/70 均要求绝对差≤1e-6，失败即停。原 S3 未完成时 `SOURCE_S3_CHECK.json=pending`，本次不能标为 FINAL/完整完成。可在原 S3 完成后续跑核对，无需重跑本次模型前向。

## 服务器入口

以下 `<SOURCE_RUN>` 必须替换为原运行的完整路径；不会自动挑选最新运行。长任务一律后台启动。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
SOURCE_RUN="<SOURCE_RUN>" MODE=offline sh local_fusion_action_label_control/launch.sh
```

完整流程（建议原实验结束后运行）：

```sh
SOURCE_RUN="<SOURCE_RUN>" MODE=all ROCR_VISIBLE_DEVICES=2 sh local_fusion_action_label_control/launch.sh
```

只检查准备状态：

```sh
SOURCE_RUN="<SOURCE_RUN>" MODE=preflight sh local_fusion_action_label_control/launch.sh
```

启动打印 RUN/PID/LOG/MODE；默认新输出为 `/data/cjm/datasets/logs/action_label_control_YYYYMMDD_HHMMSS`。可通过 `RUN/PY/MODE/ROCR_VISIBLE_DEVICES` 明确覆盖。只通过 ROCR 指定 GPU，清除 HIP/CUDA 可见变量；offline 不调用 GPU 检测模型。先 offline 后完整回放或中断续跑，复用打印出的新RUN：

```sh
SOURCE_RUN="<SOURCE_RUN>" RUN="<CONTROL_RUN>" MODE=all sh local_fusion_action_label_control/launch.sh
```

查看进度和检查状态：

```sh
tail -f "<CONTROL_RUN>/driver.log"
python -B -m local_fusion_action_label_control.status --run "<CONTROL_RUN>" --mode offline
python -B -m local_fusion_action_label_control.status --run "<CONTROL_RUN>" --mode all
```

offline 状态成功仅表示离线阶段完成。默认完整状态检查需 LABEL_AUDIT、TRAIN、CALIBRATION、PROBE、REPLAY、SOURCE_S3、FINAL 和 integrity 全部完成且输出哈希一致，返回码0并打印 CONTROL COMPLETE；否则非零。最终阅读 `FINAL_RESULTS.md`。

## 输出和文件用途

| 文件 | 用途 |
|---|---|
| `__init__.py` | 独立包 |
| `experiment.yaml` | 仅两标签、共同标准和报告描述线；训练设置继承原协议 |
| `common.py` | 路径隔离、内容身份、原子IO复用；commit仅可选信息 |
| `inputs.py` | 只读原协议、阶段、哈希、场景、schema、权重和缓存校验 |
| `labels.py` | 原标签直接复用、新检测后果键、平局和分类规则 |
| `audit.py` | 标签组成、重复关联及每组最佳键 |
| `probes.py` | 新组线性训练、原权重复用、共同检测标准校准 |
| `evaluate.py` | 双评价标准、分组指标和实际训练预算 |
| `runtime.py` | 原只读回放Runtime适配，无Git运行前提 |
| `replay.py` | 单次来源前向、七方法回放、Shared/原S3核对 |
| `summarize.py` | 六个科研问题及中文结论，区分保守回退与学习改善 |
| `pipeline.py` | preflight/offline/all、锁、阶段恢复、日志、完整性 |
| `status.py` | 离线与完整完成状态分别检查 |
| `test_core.py` | 纯合成协议/张量与shell检查 |
| `run_all.sh` | POSIX服务器入口 |
| `launch.sh` | POSIX nohup后台入口 |
| `README.md` | 复用事实、使用方式、边界及未验证部分 |

新RUN输出：`protocol.json`、`manifest.json`、`driver.log`、`PREFLIGHT.json`、`source_cache_index.json`、`feature_schema.json`、`feature_normalization.json`、`LABEL_AUDIT.json/.md`、`PROBE_RESULTS.json/.md`、`REPLAY_RESULTS.json/.md`、`common_calibration.json`、`SOURCE_S3_CHECK.json`、`FINAL_RESULTS.md`、`final_results.json`、`probes/{OriginalLabel,DetectionOutcomeLabel}/linear_{cls,reg}.pt` 和逐帧 `cache/`。

完成阶段与逐帧文件均保存哈希，不能仅凭文件存在跳过。新组按epoch保存小型optimizer恢复点，重做未完成epoch；回放恢复仍遍历原loader的所有帧，包括完成帧，保持天气/点处理随机序列。原源码、配置、权重或缓存内容变化拒绝恢复。manifest失败时记录stage并非零退出。offline可以成功结束但不产生虚假的完整完成标记。

本地仅允许语法、配置、shell和合成协议测试：`python -B -m local_fusion_action_label_control.test_core --static-only`。完整合成张量测试由服务器入口强制执行；训练和真实模型/AP复现必须留在服务器。新标签是否有效需实际运行后判断；未自动增加网络、消融、bootstrap、显著性检验或调参。

## 本次实现自检（2026-10-08）

- 19 项不依赖 Torch 的合成/协议检查通过；包含标签平局、KEEP 简化、共同校准、同效 regret、缓存只读/哈希固定、场景不符拒绝、原回放规则复用和 manifest 恢复。
- 13 个 Python 文件 AST 语法检查、experiment.yaml 校验通过；全部新增文件 LF、无 BOM、无行尾空格。两份脚本通过真实 `sh -n` 检查。
- 实现前后 118 个原实验及相关保护文件 SHA256 一致；冻结清单的 14 个源码文件全部通过校验。没有修改 AI_CONTEXT、原审计或冻结模块，没有同步服务器或执行 Git 写操作。
- 本地缺少 Torch；未执行依赖 Torch 的两个合成张量测试，也未运行科研训练、冻结模型前向或真实数据评价。服务器入口在正式阶段前强制执行包含张量检查及 shell 检查的完整测试；环境缺包将明确失败。
- 待服务器验证：实际原缓存及权重兼容、线性训练/校准、在线天气回放、Shared 哈希/AP 与原 S3 Greedy AP 复现。实现完成不能代替这些实验结果。
