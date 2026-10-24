# NVIDIA 远端运行说明

去图片像素的精简数据模式见 `v2xr_pointcloud_only/README.md`。先在海光保留原数据生成精简副本并验证抽样输入一致，再用 `POINTCLOUD_ONLY_ROOT` 显式启用；不直接删除图片、不关闭相机视野筛选。该模式使用新的预检与输出路径，旧全数据模式仍保留。以下默认路径表描述未启用精简模式时的配置。

本说明依据 AI_REMOTE.md（2026-10-24）。默认入口已切换为独立 CUDA 模式，原海光模式通过 BACKEND=rocm 保留。

## 路径与依赖

| 用途 | 默认值 |
|---|---|
| Python | `/workspace/cjmnet_cuda_env/envs/cjmnet-cu124/bin/python` |
| 已编译官方运行副本 | `/workspace/cjmnet_cuda_env/official-runtime-cu124` |
| 提取训练集 | `/data/cjm/datasets/v2xr-official-train-v2/train` |
| 官方Clean验证视图和辅助数据 | `/data/cjm/datasets/v2xr-official-validate-v1/validate` |
| Clean/Fog/Snow配对验证数据 | `/data/cjm/datasets/v2xr-lidar-weather-validate-v1` |
| 新建天气读取视图和GT审计 | `/data/cjm/datasets/gspr-repe-cu124-inputs-v1` |
| 新实验输出 | `/data/cjm/datasets/logs/gspr-repe-cu124-v1` |
| 启动日志 | `/data/cjm/datasets/logs/gspr_repe_cu124.launch.log` |

Python3.10、Torch2.5.1+cu124、torchvision0.20.1+cu124、NumPy1.26.4、SciPy1.15.3、Shapely2.0.0、Open3D0.18.0、Cython0.29.37、spconv2.3.6、cumm0.4.11沿用已安装环境并核对。pypcd沿用安装脚本指定klintan源码包，实际分发版本记录到本次环境报告；ASCII/binary真实点云读取由官方加载流程检验。脚本不安装包、不重新编译扩展、不使用平台py312环境。

不需要激活脚本：入口直接使用指定Python，设置运行库路径，拒绝其他Torch/CUDA版本。CUDA模式清除ROCm可见性变量，保留平台已有CUDA_VISIBLE_DEVICES。默认不传GPU；程序要求恰好一张可见卡。如果环境没有CUDA_VISIBLE_DEVICES且确需选卡，可显式GPU=0；已有变量时同时传GPU会拒绝运行，避免覆盖平台分配。

## 上传与启动

旧GSPR专用包不含本次后端和天气混合依赖。请上传新 `cjmnet_repe_cuda_flat.zip` 到 `/workspace/`。环境和已编译runtime继续复用。确认没有任务使用旧源码后，先保留旧目录，再解压回你指定的位置：

```sh
test ! -e /workspace/gspr_source/cjmnet_before_repe_20261024 && mv /workspace/gspr_source/cjmnet /workspace/gspr_source/cjmnet_before_repe_20261024
/workspace/cjmnet_cuda_env/envs/cjmnet-cu124/bin/python -m zipfile -e /workspace/cjmnet_repe_cuda_flat.zip /workspace/gspr_source/cjmnet
/workspace/cjmnet_cuda_env/envs/cjmnet-cu124/bin/python /workspace/gspr_source/cjmnet/verify_source_bundle.py
```

旧源码保留在 `/workspace/gspr_source/cjmnet_before_repe_20261024`；若备份路径已存在，上述第一条命令会停止，请另选备份名称。新压缩包直接包含 `gspr_repe/` 等目录，没有 `cjmnet/` 外层，解压目标写 `/workspace/gspr_source/cjmnet`。请勿把这个flat包解压到 `/workspace/gspr_source`，否则文件会散落到父目录。单次后台启动完整实验：

```sh
cd /workspace/gspr_source/cjmnet && mkdir -p /data/cjm/datasets/logs && nohup env BACKEND=cuda MODE=all RESUME=1 sh gspr_repe/run_all.sh > /data/cjm/datasets/logs/gspr_repe_cu124.launch.log 2>&1 < /dev/null &
```

启动后可记录 `echo "PID=$!"`。进度查看：

```sh
tail -n 50 /data/cjm/datasets/logs/gspr_repe_cu124.launch.log
```

MODE=preflight只执行完整数据和环境预检；MODE=smoke执行第一阶段；MODE=all在第一阶段验收后串行执行全部正式对照。恢复使用同一命令、同一源码和输入，不重复启动仍在运行的任务；锁会拒绝并发写入。

## 新CUDA预检做什么

1. 检查实际可见GPU、Python/包版本、官方运行副本全部源码哈希及本次扩展路径/哈希。
2. 实际执行CUDA框内点算子、CPU框内点算子和框重叠算子。仅编译成功不会被登记为运行通过。
3. 核对8083帧训练验收身份、提取内容SHA256、场景/车辆划分和原提取CRC记录。原`.7z`不作为训练输入，无需再次传输；原清单中归档身份保留，不冒充本机重新做过归档CRC。
4. 检查官方验证视图原manifest和配对提取records。rsync -L得到的普通文件按内容核验，不要求继续是原软链接。天气视图中的复制文件也与payload哈希核对。
5. 新建独立只读链接视图，包含天气PCD、Clean/YAML、雷达、四张相机图。不会修改原数据；不会猜测原海光overlay路径。
6. 在本次CUDA环境重新通过官方加载器审计Clean/Fog/Snow全部829帧GT，核对帧身份、几何及6224/3673/5854目标数量。训练/验证不重叠。此过程只读取数据，可能较慢。
7. 原生算子身份、环境、数据和源码均写入新记录，后续使用同一锁定身份。旧D−1A/B/C、旧纯AttFuse权重及旧模型目录不再是CUDA运行前提。

此后仍保持原训练配方：batch8、20轮、seed20261017，所有B0–B4都匹配训练；只按Clean O AP70选轮次。新CUDA评测调用官方 `inference_intermediate_fusion`、官方后处理及AP函数，并复用已审查的官方匹配跟踪器。包装入口有变化；算法定义不变；当前没有CUDA与旧HIP端到端逐位一致性的实测结论。O/O*/F和目标变化全部保留。

RTX4090 24GB能否容纳batch8仍待远端实测；内存不足时记录技术失败并停止，不自动减少批次或改变训练协议。数据传输未完成、缺相机/雷达文件、GPU未分配或驱动不工作时也会明确失败。

## 结果与恢复

原有日志和阶段顺序保持。`current_stage.json`和`STOP.json`记录进度/技术失败；阶段`failure.json`保留原因。`stage1/STAGE1_REPORT.md`仅在真实验收通过后生成；正式报告在`reports/`。

输入审计在`CUDA_INPUT_ROOT/summary.json`和`per_frame.jsonl`。同步源码后需要重新开始一轮时，同时设置新的 `OUTPUT_ROOT` 和 `CUDA_INPUT_ROOT`；既有输入/输出不覆盖，不手改旧哈希。

路径可以通过TRAIN_ROOT、VIEW_ROOT、PAIRED_ROOT、RUNTIME_ROOT、CUDA_INPUT_ROOT、OUTPUT_ROOT、CUDA_ENV_ROOT覆盖；默认值已经对应AI_REMOTE.md。新增CUDA记录必须与历史ROCm结果分开。

当前本地只做静态和模拟目录解析测试，未连接远端、未验证传输完成、未运行任何真实数据/GPU/训练。


## 为什么项目根目录没有opencood

本次源码与编译运行目录分开。GSPR实验代码位于`/workspace/gspr_source/cjmnet/gspr_repe`。官方源码保存于`/workspace/gspr_source/cjmnet/v2xr_official_repro/upstream/opencood`，实际加载已编译的`/workspace/cjmnet_cuda_env/official-runtime-cu124/opencood`。

`cuda_runtime.environment`调用原`observe.bootstrap`，将官方runtime放到Python搜索路径首位。`training.config`读取runtime内的`opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml`，将数据路径改为本次目录，保存到各实验输出的`config.yaml`。CUDA评测同样从runtime读取原配置并设置本次天气视图。

因此，项目根目录没有`opencood/`是预期布局，不需要把配置或opencood移动过来。也不要移动/覆盖已经编译的runtime；源码中的修改模块通过独立适配器接入它。实际导入路径和源码/扩展哈希会在服务器预检中核对。
