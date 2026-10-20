# SCOR-W P6 一键顺序执行

将本目录完整同步到服务器仓库根目录，与 `shared_weather_benchmark_p6/` 并列。
无需修改旧包、源码快照或实验合同，无需 Git。

一次启动会自动顺序执行：

1. checks：现有 P6 源码、配置与合成数值检查。
2. P6-A：核验第一版证据。
3. P6-B：重新计算观测指标。
4. P6-C SMOKE：2 scene × 2 帧 × 1 seed，E1/E2/E3。
5. P6-C 正式：12 scene × 2 帧 × 3 seed，E1/E2/E3。

启动前请同步完整 P6 包。本脚本不重跑历史 P0–P5，不运行检测模型、训练或 AP 评估。
全部真实数据操作只在 Linux 服务器执行，沿用 opencood 解释器和原有 CPU 环境。

## 一条后台启动命令

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs/scorw-p6-v1
nohup env RESUME=1 sh scorw_p6_pipeline/run_all.sh \
  >> /data/cjm/datasets/logs/scorw-p6-v1/p6_pipeline.log 2>&1 &
```

默认 `RESUME=1`，首次运行和中断恢复均使用上面同一条命令。
已有阶段会重新核验合同、输入与产物哈希，校验通过后复用；中断阶段沿用原有恢复机制。
重复启动会被总流程锁拒绝。`RESUME=0` 仅允许无历史 P6 状态/产物时启动，绝不删除结果。
不要在此总流程运行期间另起单阶段命令。

## 查看进度

```sh
tail -n 100 -f /data/cjm/datasets/logs/scorw-p6-v1/p6_pipeline.log
cat /data/cjm/datasets/logs/scorw-p6-v1/pipeline/PIPELINE_STATUS.json
```

日志有 `START`、`PASS`、`STOP` 标记。总状态为 `complete` 且五步均 `passed` 才表示正式规模执行完毕。
`PIPELINE_STATUS.json` 原子写入，包含活动阶段、PID、尝试次数、门禁路径和哈希、错误堆栈。
重新启动时会开始新一轮核验，状态保留当前尝试；总日志追加保留全部尝试。
进程被 SIGKILL 或机器掉电时状态可能暂留 `running`，恢复仍必须重走原入口核验。

任一步退出非零、门禁缺失/失败、源码或配置改变时立即停止；失败后的阶段不启动。
启动前的源码、路径、合同或重复运行检查失败时，原因写入总日志，原有状态文件保持原样。
SMOKE 成功后自动启动正式规模，正式结束还核对两次冻结的完整 scene/seed 计划一致性。
匹配不足、负结果和 `INCONCLUSIVE` 等科学判断会原样保留，不据此重抽天气、调参或跳过结果。
总脚本没有修改物理模型；计算完成不代表物理真实性得到验证。

## 最终文件

- A：`/data/cjm/datasets/logs/scorw-p6-v1/audit/`
- B：`/data/cjm/datasets/logs/scorw-p6-v1/metrics/`
- SMOKE：`/data/cjm/datasets/shared-weather-benchmark-p6-v1-smoke/`
- 正式：`/data/cjm/datasets/shared-weather-benchmark-p6-v1/`
- 总流程状态与独立源码/配置合同：`/data/cjm/datasets/logs/scorw-p6-v1/pipeline/`

本目录的 `tests.py` 只使用临时文件与模拟子进程，验证执行顺序、失败停止和恢复重新核验。
它不执行研究入口、不导入研究代码、不读取真实数据。
