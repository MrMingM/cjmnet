# B1 模块 A：一次只读诊断

只读取已完成 B0/B1 的选中权重、历史、阶段哈希、完整三天气829帧缓存、官方 F 匹配、统计和标注审计。Torch 只在服务器 CPU 上加载已校验的权重；不导入模型、OpenCOOD或原生算子，不访问点云，不训练、不推理、不重算AP或IoU，不执行第二阶段模型修改。

## 部署到正在运行其他 REPE 组的海光服务器

不要把目录实体直接上传到 `gspr_repe/`，也不要改 `source_snapshot.json`。旧流水线递归扫描包内文件，实体目录会改变身份。本工具必须先放在外部，再由 install.py 建立目录链接；安装器验证链接前后的 REPE 文件集合和哈希完全相同。Linux 的 Path.rglob 默认不遍历目录链接；若实际解释器行为不符，会撤销链接并拒绝安装。

将本次 `b1_diagnosis_tools.zip` 上传到 `/data/cjm/datasets/packages/`。外部目录应当尚不存在；不覆盖已有工具。执行：

```sh
set -eu
PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
TOOLS=/data/cjm/datasets/tools/repe_b1_diagnosis_v1
test ! -e "$TOOLS"
mkdir -p "$TOOLS"
"$PYTHON" -m zipfile -e /data/cjm/datasets/packages/b1_diagnosis_tools.zip "$TOOLS"
"$PYTHON" -B "$TOOLS/install.py"
```

运行时只需要总入口；下面为后台形式，推荐使用，以免权重/缓存校验因终端退出中断：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup sh gspr_repe/b1_diagnosis/run_all.sh \
  > /data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4.b1_diagnosis.launch.log \
  2>&1 < /dev/null &
echo "PID=$!"
```

所有报告只写入实验根目录的 `b1_diagnosis/`：`GATE_ANALYSIS.json`、`TARGET_CHANGES.json`、`B1_DIAGNOSIS.md`、`INPUT_AUDIT.json`、`COMPLETE.json`。失败生成 `FAILURE.json`，并在三个要求的报告中说明阻塞原因。已有诊断报告会被保留，重复启动拒绝覆盖。诊断使用独立锁，不抢占训练的锁。

## 证据和解释范围

- 核验两组20轮Clean选权重记录、选中权重SHA256、训练及评测契约、阶段完成哈希、源码、GT身份/几何、帧顺序，以及每个预测缓存的哈希和身份。必须有完整F匹配和统计，不对缺失缓存补算。
- 复用 `diagnostics.py` 的目标集合逻辑，直接读取 `protocols.F.matched_pairs['0.7']`；匹配来自 `v2xr_protocol_audit/evaluate.py` 中对官方匹配器的跟踪。本工具不另建匹配器或AP规则。另列缓存中已有0.3/0.5匹配作为定位证据。
- 精确读取选中权重的单标量gate。第1/10/20轮权重若存在且哈希匹配则读取；可选初始00状态仅按保存的哈希从可信项目文件以CPU加载，含优化器/随机状态但不恢复或执行它们。
- 点数审计文件缺失时继续目标计数和距离分析，明确点数/有点车辆覆盖信息缺失；若存在但身份错误则停止。覆盖代理只是有框内点的车辆数，不代表空间覆盖率。按B0正确目标作分母输出丢失率，避免只看丢失绝对数量。
- 保存的后处理预测无法说明被NMS抑制的候选框、未保存的低分框或中间特征变化；不把所有IoU0.7失配称为物理目标完全消失。TP/FP分数均值/中位数仅供描述，不证明排序根因。
- gate比例不代表FLOPs，也不代表实际特征改变量或独立贡献。B1同时改变原GSPR加权、有效点PFN、归一化和最大池化，现有B0/B1不能隔离原因。报告仅建议一个后续“保留B1路径但关闭聚合残差”的匹配训练对照，本次不实施。

本地只允许标准库模拟样例、语法/静态检查。真实权重与缓存只在服务器读取。单独目录不纳入旧REPE训练快照，部署方式如上；不要重新打包或覆盖正在运行的实验源码。
