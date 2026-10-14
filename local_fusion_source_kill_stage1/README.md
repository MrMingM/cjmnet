# Source Utility Kill Protocol — Stage 1

本目录只回答两个问题：最佳 source 标签是否有清楚的离散 margin；§33 大 Oracle 的 AP 收益有多少在 AOnly/BOnly 受限路径中保留。实现不训练、不修改冻结模型、不启动 K1/K2/K3。K3 保持冻结。预注册见 AI_CONTEXT.md §40。

## 运行边界

本地只执行纯单元测试和静态检查。真实 cache、checkpoint、数据以及 GPU 实验只在远程服务器读取。代码可整目录复制到没有 `.git` 的服务器，所有身份检查使用 SHA256。

```sh
python -m unittest local_fusion_source_kill_stage1.test_core
python -m local_fusion_source_kill_stage1.common --static
sh -n local_fusion_source_kill_stage1/run_stage1.sh
sh -n local_fusion_source_kill_stage1/launch.sh
```

`--static` 不读取服务器 cache/模型/数据，不等于服务器 preflight 已通过。交付包含 source_snapshot.json；必须整目录同步，不能为通过校验手改哈希。

## 证据与 preflight

默认 S0 run 从 AI_CONTEXT.md §38 的明确路径提取，并由其 protocol/manifest/S0 缓存校验确认；默认 Oracle run 从 §33 的成功路径提取。可通过 `S0_RUN` / `ORACLE_RUN` 明确指定已搬迁的原运行目录，必须仍通过相同的身份检查。不会通过“最新目录”猜测运行，不从 Markdown 反推 group outcome。

本地 `logs/sectionB/all/` 有 FINAL_RESULTS.md/LABEL_AUDIT.md 和诊断文件，未包含原 S0 protocol/manifest/cache；原运行路径也由 S4 的 DEFAULT_BASE 记录交叉支持。服务器 preflight 直接读取原运行：

- 原 S0 complete、逐帧 cache 哈希、源码/协议身份、已有浮点修复 provenance；保留修复迁移后的原 train labels。
- 同一 B0/Shared checkpoint、train/validation indices/scenes、fit/calibration 划分、原 Oracle 配置与源码。
- 前端/v3/B0 权重哈希，原 Oracle 四天气 frame-order AP 参考值。
- dataset 根目录和所有嵌套链接的 realpath，在 loader/数据读取前拒绝正式 test。四个硬禁止路径及任意 `test` 目录别名全部拒绝。

预检仅读数据契约、哈希及目录元数据，不运行 detector。K1-0 创建模型时再次核对原 frontend/v3 contract 和 B0 arm。旧 S0 的精确快照只用于 K0；K1-0 完整遵守 §33 的原在线天气推理契约，若浮点漂移导致 FULL AP 超过 1e-6 则 STOP，不引入新容差或替换旧参考。

## K0′

CPU 读取原 `cache/{train,validation}/{weather}/{index:08d}.pt` 的 `labels[*].outcomes.{cls,reg}`。每个 group 保留 frame、proposal anchor、task 和 focal association；associated/background 分开。完整 KEEP/single/query 池由 cache 的 source_names 对齐，严格调用原 action_keys。比较 full9 排序的前两名，用 [:6]/[:8] 区分离散、连续、完全并列。KEEP-only 计 exact tie 并单独计数；并列 source 集合完整保存。

另外统计相对 KEEP 的 discrete beneficial set、continuous-only benefit、harmful exists；报告 scene 覆盖和集中度。旧风险表的 48 个 split/weather/task/stratum 格子按原两位百分数逐项复现，不通过即失败。风险分母为 non-KEEP action；TP increase 仅表示能直接带来净 TP 增长的 action 比例。

K0_SIGNAL 是预注册的描述性标签，绝无自动 source STOP。adverse mean 为三天气等权，cls/reg 分别报告。train/associated 与 validation/associated 的完整表用于核对趋势；all 仅附加结果。

## K1-0

`oracle_adapter.py` 用原 `run_oracle.__code__` 创建独立函数实例，仅替换该实例的 `_action_key` 绑定。原模块全局变量/源码不改动。FULL 返回原 key；受限策略对被禁止动作返回当时 KEEP key，使其不能在原严格 `>` 比较中胜出。所有 shortlist action 仍执行原 compose→postprocess→state→key，A/B/E 全部在自身当前 greedy state 重新判断。

FULL/A_ONLY/B_ONLY 的每次调用均执行原函数中的 Shared clone；三条路径独立。Same shortlist 最多 6，Task 直接保留原 overall/separated/same 各 top-k 后去重。source 数量变化仍用原候选构造。GT 权限完全沿用 §33 target ROI、proxy、outcome 和评价契约；没有额外 GT 特征、训练或标签来源。

先完成四天气×Same/Task 的 FULL，与保存的 §33 Shared/Oracle-Same/Oracle-Task AP30/50/70 全部 ≤1e-6 后，写 FULL_REPRODUCTION_GATE.json，再执行 A_ONLY/B_ONLY。失败停止 Stage 1，不输出 A/B 解释。原 Oracle 的 `decide()` 不会被调用。

输出绝对 AP 和 pp，retention 仅辅助；分母绝对值 <.5pp 标记 UNSTABLE_RATIO。decomposition 是 restricted-policy attribution，不是线性可加分解。B/E 是原 key 定义下的等价，并不证明整帧所有非 focal 分数/框身份都不变。recovered/lost/new_fp 是终态相对 Shared；step_* 单列逐目标累积后果。

## 启动、并行与恢复

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
export ROCR_VISIBLE_DEVICES=2
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
sh local_fusion_source_kill_stage1/launch.sh
```

launch 创建 RUN，以 nohup 启动驱动，输出 RUN/PID/LOG。默认 RUN 为 `/data/cjm/datasets/logs/source_kill_stage1_YYYYMMDD_HHMMSS`。driver 使用文件锁，防止同一 RUN 双启动；PID 文件仅作便捷提示。preflight 与单元测试后并行启动 K0 CPU、K1-0 GPU，保存独立 PID 和日志。K0 失败会终止 K1-0；任一失败总状态非零，全部成功才 summary。手动终止 driver 会清理两个子进程。

```sh
RUN=<已有RUN> sh local_fusion_source_kill_stage1/launch.sh
tail -f <RUN>/driver.log
tail -f <RUN>/k1_0.log
python -m local_fusion_source_kill_stage1.status --run <RUN>
```

完成的 K0 报告校验后直接复用。K1-0 按 weather×mode×variant 独立原子提交；有完整 condition JSON、身份校验和 frame-record SHA256 才跳过。未完成 condition 从该 condition 起点重跑。临时文件不会被读取为完成结果。改变源码/协议/输入会拒绝复用；不能为了通过门禁改哈希/标签/参考 AP。

首次服务器 preflight 通过后，`/data/cjm/datasets/logs/source_kill_suite_clock.json` 保存跨 RUN 的统一 14 自然日时钟；截止后停止投入，含义仅为 resource stop。它不提供科学否定结论。

## 输出

- protocol.json / manifest.json：源码、输入、原运行、权重、项目时钟和独立组件状态。
- k0.log / K0_GROUP_MARGIN.json / K0_GROUP_MARGIN.md / k0_group_records.jsonl。
- k1_0.log / K1_0_ORACLE_DECOMPOSITION.json / K1_0_ORACLE_DECOMPOSITION.md。
- FULL_REPRODUCTION_GATE.json；conditions/ 下完整 condition 与逐帧动作/key 审计。
- STAGE1_FINAL.json / STAGE1_FINAL.md：阶段 1 结论和下一阶段待验证问题。
- driver.log / pid / k0.pid / k1_0.pid；独立 *_status.json 避免并发覆盖 manifest。

只有三个状态均 complete 才代表本阶段成功。结果报告自动停在人工判断处；不会运行 K1/K2/K3 或恢复任何 source 模型开发。
