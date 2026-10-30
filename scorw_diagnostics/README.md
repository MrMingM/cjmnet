# SCOR-W 评测协议与逐点输入诊断

只读取已完成的 SCOR-W Validate、原始来源记录、D6预测及固定模型配置。无训练、无天气生成、无参数搜索，无GPU推理。旧目录不写入，缓存和数据每次恢复均重新核验哈希。不依赖Git。

`sh scorw_diagnostics/run_all.sh` 顺序执行：输入/源码/权重/缓存冻结→服务器单元测试→全部1980帧缓存AP复现→GT及两种匹配规则、两种排序交叉审计→固定首两帧精确体素追踪→FULL全部1980帧10478份点云追踪→守恒与报告。任何技术失败停止，保留已完成阶段。

## 运行

在 `/home/cjm/OpenCOOD-main/cjmnet`：

```sh
MODE=preflight RESUME=0 sh scorw_diagnostics/run_all.sh
MODE=smoke RESUME=0 sh scorw_diagnostics/run_all.sh
```

预检查也读取全部Validate文件并核验哈希，可能需要较长时间。三种模式分开合同、分开输出。SMOKE仅原生GT及点流取冻结前两帧，A1仍复现全部缓存。FULL不得复用SMOKE作为全量证据。

正式后台执行：

```sh
mkdir -p /data/cjm/datasets/logs/scorw-diagnostics-v1.2
nohup env MODE=full RESUME=0 sh /home/cjm/OpenCOOD-main/cjmnet/scorw_diagnostics/run_all.sh > /data/cjm/datasets/logs/scorw-diagnostics-v1.2/full.log 2>&1 < /dev/null &
```

```sh
tail -f /data/cjm/datasets/logs/scorw-diagnostics-v1.2/full.log
```

仅在旧进程退出后恢复：

```sh
nohup env MODE=full RESUME=1 sh /home/cjm/OpenCOOD-main/cjmnet/scorw_diagnostics/run_all.sh >> /data/cjm/datasets/logs/scorw-diagnostics-v1.2/full.log 2>&1 < /dev/null &
```

## 诊断定义

- A1首先精确复现原匹配缓存AP，再直接调用原生AP函数交叉检查；绝不重新推理。
- A3执行真实原生 `getitem→collate→generate_gt_bbx`，通过同一去重/范围操作核对ID；D6框从Clean元数据重建并验证原缓存GT哈希。ID无法可靠恢复就停止，不强行几何配对。
- A4固定缓存预测，交叉使用两套GT、两套实际匹配器、两种排序。报告按排序→GT→匹配的条件贡献及完整交叉表；分解顺序会影响贡献。历史预测不同的影响保留未解释。
- B中D6从不shuffle；严格使用同样的float32输入、局部自车过滤、世界到ego转换、范围裁剪和CAV选择。逐帧哈希必须与D6 `input_hash`一致。
- 官方4维体素结果与独立5维携带ID的体素结果逐文件严格比较：坐标、有效点数、全部4维特征、实际collate结果；再独立核对同体素原始顺序。ID不参与模型输入。重复XYZI仍有各自索引。任何不一致立即阻断网络数量结论。
- S4表示地址属于已创建的有效体素，S5表示该点真的占用送入PillarVFE的有效槽位。已占满体素的候选点可通过S4而不通过S5。这里不做网络重推理；通过D6输入哈希、同一原生预处理/collate和直接转交PillarVFE的冻结forward源码确认输入槽位。不声称点必然影响最大池化后的特征。
- 原始点→Fog点通过 `original_index[emitted]` 精确关联。无返回、未激活CAV、自车删除、范围删除、体素边界、体素总数、单体素容量和有效槽位互斥计数，严格守恒。
- 目标归属使用原始Clean坐标及固定D6三维框，重叠框点单列，不归入唯一目标。原始点距离按各CAV传感器分组；目标距离按ego分组，二者不混用。
- 案例固定取遍历中最先出现的6个40–60米活跃目标，有唯一归属点即可，不依据替换率或检测结果选择。逐点案例每例最多12条，其余只保留必要统计。
- 逐scene给结果；目标/连续帧层面的相关性只作描述，不作为独立场景统计或因果证明。

## 输出

`/data/cjm/datasets/logs/scorw-diagnostics-v1.2/{preflight,smoke,full}`：
`INPUT_FREEZE.json`、`SOURCE_AUDIT.json`、`AP_PROTOCOL_AUDIT.json`、`AP_PROTOCOL_COMPARISON.md`、`GT_DIFFERENCE.json`、`MATCHING_DIFFERENCE.json`、`FACTOR_CONTRIBUTIONS.json`、`POINT_FLOW_SUMMARY.json`、`POINT_FLOW_BY_DISTANCE.json`、`OBJECT_POINT_FLOW.json`、`NEAR_FIELD_WEATHER_FLOW.json`、`EXAMPLE_TARGETS.md`、`EXAMPLE_POINT_CASES.json`、`OBSERVATION_DETECTION_ASSOCIATION.json`、`FINAL_DIAGNOSIS.md`、`VALIDATION_GATES.json`、`REPORT_MANIFEST.json`。

逐帧GT和匹配在 `A/frames/`，逐PCD和逐目标统计分片在 `B/frames/`（SMOKE为 `B_smoke/frames/`）。技术失败完整堆栈保存在 `FAILURE_TRACEBACK.txt`。全部输出在数据盘，不写回数据集。
