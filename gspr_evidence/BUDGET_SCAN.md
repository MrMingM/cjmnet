# A0B0 预算扫描

这轮只把通信预算逐渐加大。GSPR、主干、AttFuse、检测头、A0B0 排序和平均邻车配额都不训练、不改动。无需通信头 checkpoint 或教师缓存。

预算是**每场景所有邻车合计**：256 KiB、1、2、4、8、16、32 MiB，以及不受预算约束的 full。1 MiB = 1048576 字节；actual mean bytes 才是实际平均传输量，不能把预算上限当实际载荷。请求、索引、头部、三尺度特征全部沿用旧 A0B0 核算，不包括 IP/MAC 等无线协议开销。每帧各预算共用编码结果，仍实际序列化和解码。

## 第一步：后台跑完整开发验证

同步新增文件 `budget_scan.py`、`budget_report.py`、`test_budget_scan.py`、`run_budget_scan.sh`、本说明到服务器同名目录。等待当前 GPU 任务结束或改用空闲卡。

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export FRONTEND_CONFIG=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml
export FRONTEND_CHECKPOINT=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth
export PHASE=development
LOG="/data/cjm/datasets/logs/gspr_budget_development_$(date +%Y%m%d_%H%M%S).log"
nohup bash gspr_evidence/run_budget_scan.sh > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "PID=$PID"
echo "LOG=$LOG"
tail -n 60 -f "$LOG"
```

退出 tail 用 Ctrl+C，不会停止后台实验。日志打印 RESULT_DIR、全部帧数和 ETA。结果为 `RESULT_DIR/budget_results.md`，选择结果为 `selection.json`。

开发数据固定完整 OPV2V validate（不抽场景、不跳帧）；天气是在线模拟 fog/rain/snow，**不是正式 OPV2V-W test**。本预算实验的开发和正式阶段统一使用**非全局排序 AP**，避免选择预算时又变更计算方式；因此开发数值也不能直接与以前 global-sort 的开发表相比。

按未经显示舍入的 AP30/AP50/AP70 比较，四类天气共12项全部不低于同输入 full 才通过。自动选择**已测档位中**最小的通过预算，不宣称是连续预算空间的最小值。若全部不通过，选择 full 回退，不偷偷放宽允许下降量。

`original_full` 在每帧调用原始 AttFuse＋GSPR，与通信 full 核对输出并分别计算 AP；其字节数沿用 full 的逻辑载荷作为参考。`saturated_fraction` 是预算足够传完全部块的帧比例，单车帧也计入。通过指标不代表一定节省很多通信，需一起看实际字节。

## 第二步：固定选择后后台跑正式 test

先看第一步完整报告，再设置 DEV 为第一步日志打印的实际 RESULT_DIR。正式脚本只读取该目录的选择结果，**不会扫描 test 来挑预算**。

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export FRONTEND_CONFIG=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml
export FRONTEND_CHECKPOINT=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth
export DEV=/data/cjm/datasets/logs/gspr_a0b0_budget_development_替换为第一步时间戳
export PHASE=benchmark
LOG="/data/cjm/datasets/logs/gspr_budget_test_$(date +%Y%m%d_%H%M%S).log"
nohup bash gspr_evidence/run_budget_scan.sh > "$LOG" 2>&1 < /dev/null &
echo "PID=$!"
echo "LOG=$LOG"
tail -n 60 -f "$LOG"
```

正式数据固定 OPV2V clean test、已有 OPV2V-W fog/rain/snow test，全部帧；关闭在线增强，沿用历史非全局排序 AP。只测选好的预算、full、original_full。若开发阶段选择 full，正式阶段只核对两条全通信路径。

正式报告若有一项下降，就不满足当前零下降要求；不能根据 test 再自动换一个预算。测得没有下降，也不是对未来任意数据的保证。

程序新建目录，保存每帧 AP 原始统计与字节数、源码/权重/配置指纹。没有实现帧级断点恢复；中断的目录不能作为完整结果。可用 `budget_scan --help` 单独重跑某个天气到新目录，汇总需四类天气一致、完整的 summary。
