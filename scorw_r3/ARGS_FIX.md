# R3 v1.1：兼容 AttFuse 的空参数列表

报错发生在检测器初始化前的配置检查：R3 v1 对 `fusion.args` 连续调用 `.get()`，而仓库的 `attfuse_config.yaml` 和原始 `point_pillar_intermediate_fusion.yaml` 均使用 `args: []`。原加载器默认 `proj_first=True`；旧R2也支持这种空列表。本次修复保留这一默认语义，明确的 `proj_first=False` 或不明格式仍停止。不修改模型YAML、权重、预处理、天气或评测参数。

预检查现在先用同一YAML解析器核验fusion参数，兼容问题会在天气生成前暴露。模型审计记录原参数类型和实际采用的默认含义。

## 已完成缓存的恢复

源码变化需要新合同。新输出根为 `/data/cjm/datasets/scorw-r3-v1-argsfix`，日志为 `/data/cjm/datasets/logs/scorw-r3-v1-argsfix`。旧 `/data/cjm/datasets/scorw-r3-v1` 保持原字节和失败记录。

只接受旧v1精确交付快照和配置，原始字节保存在 `compat/r3_v1_snapshot.json`、`compat/r3_v1_config.json`。复用必须满足：

1. 旧运行确实在 `Runtime.__init__` 因空列表 `.get()` 失败，且检测器尚未成功初始化或提交。
2. 原始合同、报告清单、已通过阶段提交和全部缓存字节校验通过。
3. 天气、接收器、场景选择、采样、预算、统计规则、模型、依赖版本、线程及外部输入身份相同。当前源码差异仅允许审核过的配置兼容、入口、路径及文档/测试修改。
4. R3-A、selection、waves 只读导入新合同。坐标和小报告复制，点云/数组仍读取原waves目录，每次阶段切换和结束再次检查身份。
5. 当前版本全部42项验收重新执行，检测与分析重新运行；旧检测结果不进入新结果。

复用成功日志出现 `REUSE completed R3 R3A`、`REUSE completed R3 selection`、`REUSE completed R3 waves`。没有旧同模式运行时正常从头执行。拒绝用不匹配或部分失败的缓存自动凑出通过结果。

必须保留旧v1的waves目录以及旧R0–R2所需缓存。新根层 `R3_CACHE_REUSE_AUDIT.json` 和各阶段 `LEGACY_IMPORT.json` 记录来源和提交哈希。数据与真实物理验证状态均不因修复升级。

## 一次启动

同步完整 `scorw_r3/`，包括compat、新测试和新源码快照。保持旧进程已结束。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs/scorw-r3-v1-argsfix
nohup env MODE=smoke RESUME=0 sh scorw_r3/run_all.sh > /data/cjm/datasets/logs/scorw-r3-v1-argsfix/smoke.log 2>&1 < /dev/null &
```

首次进入argsfix目录使用 `RESUME=0`；之后中断恢复使用 `RESUME=1`：

```sh
nohup env MODE=smoke RESUME=1 sh scorw_r3/run_all.sh >> /data/cjm/datasets/logs/scorw-r3-v1-argsfix/smoke.log 2>&1 < /dev/null &
tail -f /data/cjm/datasets/logs/scorw-r3-v1-argsfix/smoke.log
```

正式完整流程：

```sh
nohup env MODE=full RESUME=0 sh scorw_r3/run_all.sh > /data/cjm/datasets/logs/scorw-r3-v1-argsfix/full.log 2>&1 < /dev/null &
```

本次仅进行本地静态检查，没有在本地跑科研数值或真实数据实验，也没有自动启动服务器。新增8项格式/缓存采用验收随服务器smoke/full的numerical阶段执行；实际GPU推理仍待服务器验证。
