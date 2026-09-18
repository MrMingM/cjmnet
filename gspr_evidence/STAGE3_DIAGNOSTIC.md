# Stage-3B 增强诊断：分数、几何、来源组合与融合权重

## 2026-09-18 历史版本恢复与启动前核对

用户已重建并验证：成功Stage-3A使用的runtime，仅将历史帧FP计数不一致由抛异常改为打印WARNING。
本地已精确恢复该版本，标准化换行后的SHA256为
`a8204c9ea2c98be7565cc26bc32e59bcd65f05e659773b806d2346637c0a8cdd`。
target匹配/IoU/score、sensing、原模型输出、输入哈希、tracer与原后处理一致性等检查没有放宽。

本次追加同步：`stage3_runtime.py`、`audit_stage3_lineage.py`、`run_stage3b_diagnostic.sh`、
`test_stage3_diagnostic.py`及本说明。保留服务器已恢复的`stage3_sources.json`，不要用本地清单覆盖。

启动器现先用CPU读取三种天气的历史protocol，逐项打印所有`stage3*.py`的历史/当前哈希，
检查依赖清单、历史报告protocol副本和targets哈希，并保存`lineage_audit.json`。
仅将4个明确新增的增强模块标记为NEW_DIAGNOSTIC_MODULE；任何历史文件差异或其他未知新增文件均停止，不覆盖文件。
本地无法访问服务器历史日志，因此“全历史清单一致”需由服务器该检查确认，不能由本地运行替代。

必须使用`run_stage3b_diagnostic.sh`，旧`run_stage3b.sh`仍会比较整个implementation字典，
新增诊断模块会使旧入口报`Stage-3A/B mismatch: implementation_sha256`。
`FP assignment differs from Stage-2`实际比较本次replay得到的FP与同帧另一条匹配实现的FP，
不是与历史日志FP比较；此断言保留。输入哈希和target日志检查也全部保留。

`SMOKE=0`执行顺序：CPU历史核对 → 单元测试 → 三天气各2个候选帧smoke → 全量。
任何一步失败均停止；不要反复重跑全量或修改历史哈希来绕过错误。

## 目的和当前状态

根据用户提供的已完成 Stage-3A 报告：Fog/Rain 主要最后消失在 NMS，Snow 主要消失在 score threshold。
本入口把下一步所需的主要冻结模型干预整合到一次任务，避免每发现一个缺口就重新启动推理。
**代码已实现，本地CPU验证通过；真实GPU结果尚未产生。不能保证一次实验就证明唯一根因。**

历史 Stage-3A 路径：`/data/cjm/datasets/logs/qa_stage3a_20260917_193015`。
新入口不会覆盖旧日志，不改冻结 GSPR、AttFuse、检测头、原后处理文件，也不修改旧 Stage-3A/B 入口。
候选和条件失败率的边界沿用 STAGE3.md：只用开发验证集、在线天气、clean ego detected/weather ego missed/peer-valid/full-miss 子集。

## 一次运行包含什么

| 实验 | 保持不变 | 改变什么 | 用途 |
|---|---|---|---|
| 全车辆组合 | 已编码特征、原融合、原检测头和后处理 | 保留ego的全部车辆子集，最多16个 | 区分 ego+peer 已失败，还是加入其他来源后失败 |
| 逐框追踪 | 每个分支原输出 | 无，纯观察 | 固定anchor分数/logit/IoU、所有阶段、真实抑制者 |
| peer分数 + full几何 | full回归输出 | 替换整帧分类输出 | 测得分变化的恢复机会及整帧代价 |
| full分数 + peer几何 | full分类输出 | 替换整帧回归输出 | 测定位/重叠几何的恢复机会 |
| peer query | 所有来源keys/values | 使用peer特征作为查询，分别改全部尺度或一个尺度 | 检查融合权重的敏感性及响应尺度 |
| uniform | 所有来源values | 所有来源均匀加权 | 检查原权重分配是否是可干预环节 |
| zero ego value | 原始full权重，其他values | 将ego的value贡献置零，不重新归一化 | 区分权重变化与ego内容贡献；含幅度变化，不能单独证明ego有害 |
| score × NMS | full模型输出 | score取原值/0.05/0.1，NMS取原值/0.3/0.5的组合 | 检查后处理敏感性和误检代价 |

peer干预仅对当前帧中至少单独检出一个候选的邻车执行，所有候选共享该分支。
分类/回归交叉替换是整帧人工反事实，不逐目标拼接正确答案。
每个分支都保留最终所有框及分配GT、补回候选、损失原TP、新增FP和FP净变化。
新增FP采用原Stage-3B的一对一框匹配定义，不能用净增量代替。

只换query时keys和values不动；没有训练或修改模型参数。独立实现的原权重计算先与full输出及最终框ID/GT匹配做恒等验证。
保存关注anchor对应位置的各尺度权重、点积和来源特征范数。位置由anchor中心归一映射到尺度网格，不代表检测头整个感受野。
权重高/低仅是观察；干预补回支持可改变该结果，不自动证明原query是唯一根因。

## 减少重复计算和耗时

- 每帧原始输入只生成一次，所有干预共享已编码来源特征；原模型前向和历史重放仍保留，用于一致性检查。
- 相同车辆组合只评估一次，目标共享结果；full trace复用。
- 后处理对照不重跑网络。干预不涉及训练。
- 默认不保存所有干预的密集候选框，而保存关注anchor、抑制关系和全部最终框。`SAVE_DENSE=1` 才额外保存全部decoded proposals，磁盘和压缩耗时显著增加。
- 每完成一个分支打印耗时和补回/误伤数量，每完成一帧打印进度与估计剩余时间；逐帧刷新JSONL。
- 这次比原Stage-3B更全面，也更耗时。最多5车且4个有效peer时，每帧16个subset、28个peer相关分支、2个权重对照、8个额外后处理组合；另有原full与恒等检查。不声称几分钟能完成。
- 中断后保留已完成帧日志，但当前不支持直接续写同目录；失败不会生成完整报告。不要将不完整日志当作全量结论。

## 服务器运行（后台）

将以下7个新文件复制到服务器 `/home/cjm/OpenCOOD-main/cjmnet/gspr_evidence/`：

- `stage3b_diagnostic.py`
- `stage3_diagnostic_runtime.py`
- `stage3_diagnostic_analysis.py`
- `stage3_diagnostic_report.py`
- `test_stage3_diagnostic.py`
- `run_stage3b_diagnostic.sh`
- `STAGE3_DIAGNOSTIC.md`

不需要重跑Stage-3A。保留服务器当前通过Stage-3A的依赖、权重和配置，不升级环境。
下面的全量命令会先运行单元测试，以及Fog/Rain/Snow各前两个候选帧的完整干预预检，全部通过后自动进入全量。

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN_NAME=qa_stage3b_diagnostic_$(date +%Y%m%d_%H%M%S)
nohup env ROCR_VISIBLE_DEVICES=3 \
  STAGE3A_ROOT=/data/cjm/datasets/logs/qa_stage3a_20260917_193015 \
  STAGE2_ROOT=/data/cjm/datasets/logs/qa_evidence_validity_20260917_112301 \
  RUN_NAME="$RUN_NAME" SMOKE=0 SAVE_DENSE=0 \
  bash gspr_evidence/run_stage3b_diagnostic.sh \
  > "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log" 2>&1 &
echo "PID=$!  RUN_NAME=$RUN_NAME"
tail -f "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log"
```

仅用 `ROCR_VISIBLE_DEVICES`，可改为当前空闲的0–6号卡；不用7号卡。`Ctrl+C`退出tail，不终止后台任务。
若只想单独预检，把 `SMOKE=0` 改为 `SMOKE=2`。SMOKE报告不能用于研究结论。

## 读哪些输出

1. `mechanism_report.md`：优先阅读。每干预族的逐目标机会、统一帧级结果、损失原TP、新增FP、无观测代价的恢复数；分数门槛距离、NMS分数差、来源加入前后阶段转移。
2. `mechanism_results.json`：场景/距离/初始失败类型分层、多个可重叠的恢复模式、每种模式最多12个复查例子。
3. `stage3b_report.md`：原协议的车辆组合上限，单独保持口径。
4. `<weather>/diagnostics.jsonl`：每帧所有分支、固定anchor变化、框竞争、局部权重观察、source_addition_edges、peer_to_ego_peer。
5. `<weather>/frames.jsonl`：原Stage-3B统一帧级组合结果。
6. `protocol.json`、各天气`protocol.json`和`summary.json`：源码/配置/输入哈希与完成状态。

只需要把两个MD报告及 `mechanism_results.json` 传回，就可以先决定研究方向；有歧义再读对应帧JSONL。

## 一致性与结论边界

新增编排文件必然改变源码集合，所以不要求整份implementation字典与历史A完全相等。
仍逐项要求历史 `stage3_runtime.py`、`stage3_trace.py`、`stage3_analysis.py` SHA相等，并要求audited sources、模型、配置、软件、后处理、每帧输入哈希及历史检测重放一致。
这些核心文件本次没有修改。新的源码集合单独记录并在运行结束复查。
如果出现历史核心SHA或输入不一致，应先核对服务器版本/重放原因；不能删除检查、伪造快照或直接要求重跑A来隐藏差异。

所有候选帧的其他GT参与误伤统计，但没有覆盖成功帧和整个验证集；“无观测代价”只指本次帧中未丢失原TP、未新增几何FP，不能外推为全数据集无害。
不同干预族、不同目标最优动作不能相加；相同车辆跨帧及多种组合都相关。报告不声称AP提升、不做独立样本显著性检验、不把某个来源编号跨帧视为同一辆车。
如果head替换、query改变、阈值改变都能补回，应保留多种解释，用代价和跨场景稳定性排序，而不是宣布唯一根因。
