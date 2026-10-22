# R2 总预算与恢复修复

原停止信息 `R2_COMPUTE_BUDGET_EXCEEDED_BEFORE_RESPONSE_GENERATION` 来自全流程400万条光路×场上限。
代码在每帧开始前累加检查，可能完成第一帧全部检测后才发现第二帧使总量超限。
已核对仓库P7归档及其哈希清单：相同帧/车辆组合预计分别需要2,444,946与2,439,360次，共4,884,306。
依据及来源哈希见 inputs/R2_COST_ESTIMATE_FROM_ARCHIVED_P7.json。当前实际点数仍以服务器已有 INPUT_MANIFEST.json 再次核验为准。

修订仅调整工程预算和检查时机：单帧最多400万、全部最多800万，仍限定两个开发场景、每场景一帧、两车、完整20场。
R2开始时从固定 Clean 点数一次性列出全部开销，先写 R2_COMPUTE_PLAN.json，超限即停。
没有减少天气场、抽样检测点云、按AP挑帧，也没有改变天气、物理积分、接收模型或检测权重。
21次响应指20个池场加B2常数场；B3/B4复用结果，B0与B1分别记录。

新输出 `/data/cjm/datasets/scorw-repair-v2-budgetfix/{smoke,full}` 使用新合同。
旧 `/data/cjm/datasets/scorw-repair-v2/` 只读保留。程序自动核验旧合同、已完成阶段完整清单、原始输入、模型及运行环境，
核验旧版与当前所有物理代码/设置相同后，导入已完成的 coordinates、R0、Hahner回归和R1报告。
旧R2中已有完整COMMIT的逐场波形NPZ会逐项核验参数、原始点身份和位姿，再复制进新合同缓存。
不复用旧检测预测。旧缓存缺少COMMIT则重新计算；存在COMMIT却损坏则停止。
任意原始数据、物理参数、模型或不兼容运行环境改变都会拒绝导入。
旧进程仍运行时拒绝恢复；旧合同及结果不会被改写，不在服务器刷新源码哈希。

Shapely警告单独处理：保持原IoU算法与返回数值，核验框几何有效性及所有IoU有限且在[0,1]。
`invalid value encountered in intersection` 若伴随有效有限结果则累计到 R2_IOU_AUDIT.json，避免刷屏。
若出现NaN、非法框或非法IoU则立即阻断；不修补框、不删框、不填零、不改变AP门槛。
此前警告本身不足以证明AP有效或无效，本版重新检测和审计。

整套同步 scorw_repair/（含compat/和source_snapshot.json）后，用同一个POSIX入口后台恢复：

```sh
mkdir -p /data/cjm/datasets/logs/scorw-repair-v2-budgetfix
nohup env MODE=full DOWNLOAD=0 RESUME=1 sh /home/cjm/OpenCOOD-main/cjmnet/scorw_repair/run_all.sh >> /data/cjm/datasets/logs/scorw-repair-v2-budgetfix/full.log 2>&1 &
```

```sh
tail -n 100 -f /data/cjm/datasets/logs/scorw-repair-v2-budgetfix/full.log
```

后续中断后重用同一命令。新合同会校验新缓存，不重复完成的光路计算；检测阶段仍会重算该固定子集的预测。
输出根目录新增 CACHE_REUSE_AUDIT.json、R2_COMPUTE_PLAN.json、R2_IOU_AUDIT.json。
看到 REUSE completed 和 REUSE response 表示核验后复用。最终统计仍标为 UNCALIBRATED_EXPLORATORY。
