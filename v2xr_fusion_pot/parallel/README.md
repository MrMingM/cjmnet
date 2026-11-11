# 三卡并行续跑

P1 使用物理 GPU 4，P2 使用 GPU 5，P3 使用 GPU 6。训练参数、数据、权重目录和轮次恢复规则复用原正式流程。服务器每个进程先核对卡空闲，再用总线地址确认 PyTorch 实际用卡。卡 5、卡 6 的设备身份独立写入各组 `PARALLEL_DEVICE.json`，原运行合同继续用于校验旧检查点。

原先卡 4 上的顺序任务由 `handoff.py` 在 P1 当前轮次完整提交后切换。交接程序将 P1 从同一输出目录续跑，并在三组正式评测成功后用卡 4 生成机制诊断和总报告。

日志位于 `/data/cjm/datasets/logs/v2xr-fusion-pot-v1/`，文件为 `handoff.log`、`P1.parallel.log`、`P2.parallel.log`、`P3.parallel.log` 和 `report.parallel.log`。

若某组退出，确认对应卡空闲后，在海光用对应组名恢复。以下以 P2 为例。

```sh
r=/data/cjm/datasets/logs/v2xr-fusion-pot-v1
nohup sh /home/cjm/OpenCOOD-main/cjmnet/v2xr_fusion_pot/parallel/run_parallel.sh P2 >> "$r/P2.parallel.log" 2>&1 < /dev/null &
```

三组完成后如需单独补跑汇总，将组名换成 `report`，日志换成 `report.parallel.log`。
