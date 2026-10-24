# 无卡模式准备 cjmnet CUDA 环境

只需先上传 `setup_cjmnet.sh`，无需上传数据集或整个项目。脚本仅在 Linux x86_64 服务器执行，本地不安装任何环境、不运行模型或实验。

目标是单张 RTX 4090：Python 3.10、PyTorch 2.5.1+cu124、torchvision 0.20.1+cu124，以及 CUDA 12.4 的编译器、运行时开发头文件。安装到 `/workspace/cjmnet_cuda_env`，保留平台原有的 Python 3.12 / PyTorch 2.13 / CUDA 13.2，不安装或更换宿主机驱动。

修订2：优先检测并使用平台已有的 Conda，无需额外下载 Micromamba；没有 Conda 时才使用 Micromamba。下载遇到连接意外断开也会重试，保留证书校验。

修订3：PyTorch/torchvision 的 CUDA 12.4 安装包从官方索引单独下载，NVIDIA 依赖从 `PYPI_INDEX` 下载，避免官方 CUDA 索引把依赖下载指向当前超时的 `pypi.nvidia.com`。使用 pip 25.2、600秒读取超时和下载恢复重试；已下载完整的 PyTorch/依赖安装包保存在 `downloads/torch-wheels/`，供脚本重跑复用。当前调用结束后未完成的片段不保证跨调用续传。

修订4：修正安装脚本的一点 ASCII 自检样例。固定版 `pypcd` 的 ASCII 读取直接调用 `np.loadtxt`，单点会变成没有长度的标量，导致库内部 `len(pc_data)` 报错。自检改用两个点，同时核对实际字段/坐标及二进制读取，保留上游读取器，不修改已安装库或科研代码。该改动仅修正环境检查；上游单点 ASCII 限制仍保留。

## 第一步：后台安装

把脚本上传到 `/workspace/setup_cjmnet.sh`，然后在当前服务器终端执行：

```bash
cd /workspace
nohup bash /workspace/setup_cjmnet.sh > /workspace/cjmnet_env_setup.log 2>&1 &
echo "PID=$!"
```

查看日志：

```bash
tail -n 80 -f /workspace/cjmnet_env_setup.log
```

出现 `SUCCESS: install` 才表示安装和环境检查成功。无卡模式下 `gpu_visible: false` 正常。安装需要联网，首次至少留出 15 GiB 空间；下载缓存和实际安装会占用磁盘。请确认 `/workspace` 在平台停止/切换有卡模式后保留；无法从当前截图判断平台的持久化规则。

脚本会安装 Ubuntu 的编译器和 Open3D 所需系统库。当前 root 终端可直接执行。其余文件、环境和缓存写在专用目录。下载失败可用同一条命令重试，已有依赖会复用；不要在安装仍运行时重复启动。详细日志在 `/workspace/cjmnet_cuda_env/logs/`。

若旧版日志停在下载 Micromamba 的 `curl: (35) ... unexpected eof`，环境尚未创建，先上传修订2脚本替换 `/workspace/setup_cjmnet.sh`，再运行：

```bash
nohup env INSTALL_SYSTEM_DEPS=0 bash /workspace/setup_cjmnet.sh \
  > /workspace/cjmnet_env_setup_retry.log 2>&1 &
echo "PID=$!"
```

这条恢复命令适用于系统依赖已成功安装、随后下载失败的当前日志。查看 `/workspace/cjmnet_env_setup_retry.log`，应首先看到 `Environment manager: conda (...)`（平台确有 Conda 时）。若自动检测不到，可通过 `CONDA_EXE=/实际路径/bin/conda ENV_MANAGER=conda` 指定现有解释器管理器；不需要更换平台原环境。日志中的空 NVIDIA 库提示先保留，本次明确退出点是下载连接失败。

如果已进入 Python 3.10 环境、停在 `nvidia-cusolver-cu12` 等依赖的 `ReadTimeoutError`，上传修订3覆盖服务器脚本，然后用已创建的环境继续：

```bash
nohup env INSTALL_SYSTEM_DEPS=0 REUSE_PREFIX=1 \
  PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  bash /workspace/setup_cjmnet.sh > /workspace/cjmnet_env_setup_retry3.log 2>&1 &
echo "PID=$!"
```

此命令保留原环境和缓存，跳过系统依赖与 Conda 创建步骤，重新检查已有 CUDA 12.4 编译器后继续 Python 包安装。PyTorch/torchvision 仍固定官方 `+cu124` 版本。清华镜像可能同步不全或受网络影响，出现下载错误需看新日志，不通过卸载环境或关闭证书校验处理。

如果日志已经显示 `Successfully installed pypcd-0.1.1`，随后在自检的 `PointCloud.from_path` 报 `TypeError: len() of unsized object`，上传修订4覆盖服务器脚本，只重新检查即可，无需重装依赖：

```bash
MODE=check bash /workspace/setup_cjmnet.sh
```

该检查通常较短，不下载包、不重建 Conda 环境、不运行模型。完成标志是 `SUCCESS: check`，随后再激活环境。该单点错误出现在脚本自检环节，不能据此判定真实数据读取已经通过或失败；真实数据和 GPU 仍待服务器后续阶段验证。

激活环境：

```bash
source /workspace/cjmnet_cuda_env/activate.sh
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda)"
```

预期解释器为 `/workspace/cjmnet_cuda_env/envs/cjmnet-cu124/bin/python`，版本输出 `2.5.1+cu124 12.4`。新开终端后重新 `source`。环境脚本不写入 `.bashrc`，也不激活项目训练脚本。

## 第二步：上传源码后编译项目算子，可继续无卡模式

完整上传 cjmnet 源码到 `/workspace/cjmnet`，包括原有 `v2xr_official_repro` 目录和哈希清单；无需 `.git`。运行：

```bash
nohup env MODE=build-ops PROJECT_ROOT=/workspace/cjmnet \
  bash /workspace/setup_cjmnet.sh > /workspace/cjmnet_build_ops.log 2>&1 &
echo "PID=$!"
```

脚本核对官方源码快照，使用已有的构建选择器，只编译实际 AttFuse 链需要的两个扩展：

- `box_overlaps`：CPU 框重叠计算。
- `roiaware_pool3d_cuda`：包含官方框内点检查的 CPU/CUDA 扩展。

编译副本放在 `/workspace/cjmnet_cuda_env/official-runtime-cu124`，不修改上传的冻结源码，不复用海光 `.so` 文件。显式指定 `TORCH_CUDA_ARCH_LIST=8.9`，无卡时也能为 4090 编译。未使用的旧 PointNet2 算子不编译，避免其 `THC/THC.h` 问题。结束标志为 `SUCCESS: build-ops`；不代表训练入口已可运行。

## 第三步：有卡模式检查

在已分配 GPU 的实例上执行，仅检查极小矩阵运算，不训练模型：

```bash
MODE=check-gpu bash /workspace/setup_cjmnet.sh
```

应看到 `gpu_runtime_tested: true`。这仅验证 PyTorch 对 GPU 的访问，不验证完整模型显存、训练速度或 CUDA ROI 路径。普通重复环境检查用 `MODE=check`。

## 与仓库实际代码的对应关系

- NumPy 1.26.4、SciPy 1.15.3、Shapely 2.0.0、Open3D 0.18.0 沿用已归档海光训练环境记录。
- `spconv==2.3.6` / `cumm==0.4.11` 使用 Linux CPU 预编译包：官方 `SpVoxelPreprocessor` 实际调用 `Point2VoxelCPU3d`，网络仍由 CUDA PyTorch 在 GPU 训练。海光记录未固定这两个包版本，后续严谨速度对比仍需核对两端体素化版本。
- `pypcd` 使用已核对的 klintan 分支源码版本，按固定源码标识下载归档安装，无需服务器 Git。
- Cython 0.29.37 用于旧 `box_overlaps.pyx`。不执行上游旧 `requirements.txt`，防止 Python 3.7 时代的 NumPy/Numba/SciPy 约束破坏新环境。当前纯 LiDAR AttFuse 导入链无需 MMCV，不安装它。
- 输出包括 `environment_check.json`、`pip-freeze.txt`、`conda-explicit.txt`、下载归档哈希与安装脚本哈希。核心包固定；完整传递依赖以首次实际解析结果为准，两个文件用于后续复建，不依赖 Git。

**仍待完成：** `v2xr_weather_l012` 和继承模块强制要求 HIP、海光设备变量及旧环境/算子哈希，且包含旧服务器绝对路径。这些训练入口需要独立 CUDA 迁移，旧哈希清单不能直接手改为通过。新环境可准备，旧训练恢复目录不能直接作为 CUDA 续训目录。此安装脚本不处理这些模型迁移、不改原实验配置、不读取数据、不启动训练。

可选参数：`BASE_DIR` 更换独立安装目录；`ENV_MANAGER=auto|conda|micromamba` 选择环境管理器，默认优先已有 Conda，并在 `manager.txt` 记录实际路径；`PYPI_INDEX` 更换普通 Python 包下载源；`TORCH_INDEX` 必须保留能提供 `2.5.1+cu124` 的源；`MAX_JOBS` 默认 2；`INSTALL_SYSTEM_DEPS=0` 仅适用于系统依赖已齐全时；`REUSE_PREFIX=1` 仅适用于 Python/CUDA 已创建、需要继续下载依赖时。报错请保留日志和 JSON，不通过删除身份检查继续运行。

## 本地核查边界

交付前只做 shell 语法、嵌入 Python 语法、LF 换行和静态安装/构建流程检查。未在本地安装环境、导入 Torch/OpenCOOD、编译算子或运行实验；真实安装、包下载、编译及 GPU 检查必须以服务器日志为准。

参考：[PyTorch 官方版本](https://pytorch.org/get-started/previous-versions/)、[Micromamba 安装](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)、[NVIDIA CUDA 12.4 安装说明](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-installation-guide-linux/index.html)、[spconv](https://github.com/traveller59/spconv)。
