# F2、F3、F4 细尺度跨车交互

三个实验分别在海光物理卡 2、3、4 训练，只更新两个细尺度上的 18,432 个投影参数。三组同时从同一份 P0 第 18 轮权重和相同随机初始化起步。历史 R-A/R-B 试验也使用 F2 命名，本目录的 F2 指本次低秩交互。

唯一正式入口如下。输出目录须由管理员预先提供给 `cjm` 写入。

```sh
r=/data/new_disk/cjm_logs/v2xr-fusion-interaction-v1
nohup sh /home/cjm/OpenCOOD-main/cjmnet/v2xr_fusion_interaction/run_all.sh >> "$r/launch.log" 2>&1 < /dev/null &
```

入口先核对 P0、数据及设备，再并行运行 GPU 单元测试和每组两个真实 batch 更新。三组初始权重哈希、冻结状态哈希、两个真实 batch 输入哈希一致后，统一启动训练。各组在选出最佳权重后立即进行正式评测和机制诊断。F2 完成后在卡 2 用 P0 权重进行统一基线复测。中断后用同一命令恢复，保留已提交轮次及已核对的天气预测缓存。

运行日志包括 `audit.log`、`launch_status.log`、各组的 `smoke.log`、`full.log`、`STATUS.json` 和失败详情。全量完成后生成 `SUMMARY.md`、`METHOD_ASSESSMENT.md`、`FINAL_RESULTS.json` 及各组机制报告和逐目标记录。汇总还生成 `AI_CONTEXT_APPEND.md`，并把同一结果追加至服务器已有的 `AI_CONTEXT.md`。

全量评测包括每种天气 829 帧的 O、O*、F 协议和三个重叠阈值。历史 327 个协同漏检只用于结果分析。该验证集已多次用于研究筛选，独立验证留待后续。
