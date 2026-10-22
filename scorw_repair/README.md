# SCOR-W R0–R2：无真实参考版本（R2预算修复）

当前修复补上沿途散射、硬软波形叠加、回波竞争、替换、错误距离与零输出码无返回。
使用已有 Hahner 源码、查表资源及冻结世界场，不访问 STF 下载地址，不读取私有 URL，不解压真实天气数据。
`NOT_VALIDATED_NO_REAL_REFERENCE` 始终保留。R1 门槛通过后自动执行已获授权的 `UNCALIBRATED_EXPLORATORY` 检测。

当前修订修复R2的累计预算检查。旧 `/data/cjm/datasets/scorw-repair-v2/` 保留，只在完整核验后导入R0/R1和已提交回波缓存。
详细说明及当前恢复命令见 [R2_BUDGET_FIX.md](R2_BUDGET_FIX.md)。

## 同步与版本

整套同步 `scorw_repair/`（包括 source_snapshot.json）到服务器同名目录。
历史 shared_weather_benchmark、scorw_p7、reference_weather、AttFuse/OpenCOOD 只读复用。
不创建分支/worktree，运行不需要 .git；源/配置/模型/输入/缓存 SHA256 严格校验。
不要在服务器重写 snapshot 来绕过不一致。

本修订使用独立 `/data/cjm/datasets/scorw-repair-v2-budgetfix/{preflight,smoke,full}`。
旧 scorw-repair-v1 合同、数据和结果保留；v1 缓存不能迁移成 v2 已验证结果。
`official_stf/` 为可选的旧纯文本资料，不参与活动合同。缺少它或 related_code 中非必要论文不会阻断。
`download.py`、`real_reference.py` 仅保留未来接口，活动入口和数值测试不导入或调用它们。
未来启用真实数据需要新协议，本次不提供隐式下载开关。`DOWNLOAD=1` 明确拒绝。

## 一个入口，按顺序完成

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
MODE=preflight DOWNLOAD=0 RESUME=1 sh scorw_repair/run_all.sh
```

预检查只检查本机源码、已有查表资源、依赖、原始 train 路径和冻结权重身份，网络请求为零。
不安装、升级依赖或修改 Torch/ROCm。

SMOKE 也包括一帧、两车、完整20场的最小检测诊断，可能耗时，使用后台：

```sh
mkdir -p /data/cjm/datasets/logs/scorw-repair-v2-budgetfix
nohup env MODE=smoke DOWNLOAD=0 RESUME=1 sh scorw_repair/run_all.sh >> /data/cjm/datasets/logs/scorw-repair-v2-budgetfix/smoke.log 2>&1 &
```

正式小规模一键流程：

```sh
nohup env MODE=full DOWNLOAD=0 RESUME=1 sh scorw_repair/run_all.sh >> /data/cjm/datasets/logs/scorw-repair-v2-budgetfix/full.log 2>&1 &
```

顺序：合同/源码 → 合成数值测试 → 原始 Clean 坐标及身份 → R0 源码/字段审计 →
Hahner 均匀雾回归 → R1 固定抽样光路与平衡20场 → R2 冻结 AttFuse → 报告归档。
任何数值、回归、坐标、GT、来源、平衡或模型身份失败，后续立即停止，保留有效结果。
缺少真实雾天参考被记录，不阻断 R1 或满足其余门槛的探索 R2。
SMOKE 与 FULL 使用不同合同，SMOKE 不替代正式小样本结果。

```sh
tail -n 100 -f /data/cjm/datasets/logs/scorw-repair-v2-budgetfix/full.log
```

恢复：确认旧进程退出，重用同一条后台命令（已有 `RESUME=1`）。
每个完成阶段先验证完整清单、合同和哈希再复用；未完成阶段从确定性缓存恢复。
源码、配置、运行环境或模型身份改变后拒绝复用旧合同。`RESUME=0` 只接受全新目录。
源/输入发生变化不会自动重抽 seed、放宽容差或调参数。
Linux 入口自检 `sh -n`，脚本采用 POSIX sh 与 LF。

## 冻结规模和解释边界

- R0：实际原始 PCD 字段、8-bit RGB 强度编码、位姿、GT、Hahner 源码/参数依据；没有真实雾天比较。
- R1：FULL 为字典序前两个已研究过的 P7 开发 scene，各原定两帧；SMOKE 为第一个 scene。
  保留旧分层取样，每目标/CAV×距离×正强度层1点、每云最多256个关联成员；不能覆盖则停止。
  报告完整/抽样点数，抽样统计不充当全量精确结果。
- B0=Clean；B1=原 Hahner（保留 noise=10）；B2=新均匀；B3=新共享非均匀；B4=同池错位。
  B2/B3/B4 使用同一引擎。场算法、参数和完整20场保持 P7 设置，不增加散射随机数。
  B3/B4 逐光路完整接收结果复用，逐车边缘分布严格平衡；单次重复不保证退化匹配。
- R2：每个选中 scene 的首个冻结帧、ego和首个有效邻车，完整原点云进入检测。
  每帧最多400万、全流程最多800万个新引擎光路×场计算；R2开始前一次性计算全部预算，超过停止，不截断场池。
  B0/B1/B2各一次，B3/B4各20次，全部分别运行 Ego-only 与协同；不训练、不按 AP 调参。
  固定 AttFuse epoch13/config，必须与原 reference-weather evaluation/protocol.json 哈希相同。
  AP30/50/70 为固定 Clean GT 的 BEV/VOC2010 全局排序；None 按零预测，不跳过。
  输入经相同原 RGB 编码器；只写该诊断子集的小 PCD，不生成完整天气集。
  空 CAV 保留零 BEV，非空 scatter 必须逐位相同。历史 dataset loader 禁止导入；GT_RANGE 只读 AST 常量。
  每目标同时记录物理回波及实际通过 ego/range 过滤的证据，后者仍可能受 voxel 上限截断。
  邻车证据但协同漏检是模型行为统计，少量保留点不保证能检测。
  最多8个满足固定规则的邻车救回目标做证据移除诊断，不能当作全面因果证明。

## 结果与判断

`full/` 或 `smoke/` 根目录：
R_PROTOCOL_FREEZE.json、SOURCE_AUDIT.json、COORDINATE_AUDIT.md、R0_SENSOR_AUDIT.json、R0_RESULTS.md、
R1_HAHNER_REGRESSION.json/.md、R1_FIELD_POOL_FREEZE.json、R1_PHYSICS_AUDIT.json、R1_RESULTS.md、
CACHE_REUSE_AUDIT.json、R2_COMPUTE_PLAN.json、R2_IOU_AUDIT.json、R2_GATE.json、R2_TARGET_DIAGNOSIS.json、R2_RESULTS.md、VALIDATION_GATES.json、EVIDENCE_LEVELS.json、
RESULTS.md、NEXT_STAGE_DECISION.md、REPORT_MANIFEST.json、SMALL_RESULTS.tar.gz。
失败阶段会记录不可执行原因；未运行的检测不会给出虚假 AP。
数组和诊断 PCD 留在 stages/；小结果归档只包含报告及身份，可同步 GitHub，原始数据/大数组/权重不提交。

四项证据分别记录：`ENGINEERING_PASS`、`PHYSICS_NUMERICAL_PASS`、
`REAL_WORLD_VALIDATION_MISSING`、`DETECTOR_EXPLORATORY_RESULT`。
工程/数值通过不意味着真实天气可靠。只有后续独立真实数据验证、接收模型校准及稳定可解释的诊断证据，
才讨论正式数据集；当前不会自动批准。无实际意义的结构差异时应停止扩大，不能调 seed/AP 寻找正结果。

物理依据及近似见 PHYSICAL_MODEL.md。所有真实 OPV2V/检测运行只在服务器。
本地仅允许静态检查与合成数值/流程单元测试，未自动启动服务器任务。
