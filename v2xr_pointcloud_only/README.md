# V2X-R 去图片像素的精简数据与读取适配

目标：GSPR/REPE 仍只接收点云，加载器不再读取图片像素，保留原相机视野筛选。
每张 PNG 的真实宽高从其文件头提取，保存原图片 SHA256 和字节数，用来对应原验收记录。
不使用假图片、不统一假设分辨率、不关闭 fov，不删除雷达，不改点云、标注、模型或训练配方。

## 1. 先在海光服务器生成精简目录

按 AI_CONTEXT.md 的路径约定：代码放原项目父目录下的独立副本，压缩包、数据、日志和临时文件放数据盘；不使用 ~/ 下的零散目录。

| 用途 | 海光路径 |
|---|---|
| 上传压缩包 | /data/cjm/datasets/packages/v2xr-pointcloud-only-v1/cjmnet_repe_cuda_flat.zip |
| 独立代码副本 | /home/cjm/OpenCOOD-main/cjmnet_pointcloud_delivery |
| 新精简数据 | /data/cjm/datasets/v2xr-pointcloud-only-v1 |
| 日志、PID和退出码 | /data/cjm/datasets/logs/v2xr-pointcloud-only-v1 |
| 临时目录 TMPDIR | /data/cjm/datasets/tmp/v2xr_pointcloud_only_v1 |

以上是本次新约定，尚未声明目录已在服务器创建。原项目 /home/cjm/OpenCOOD-main/cjmnet 保留。
先创建包目录、日志目录和临时目录，将新压缩包上传到表中位置，再解压：

```sh
mkdir -p /data/cjm/datasets/packages/v2xr-pointcloud-only-v1 \
  /data/cjm/datasets/logs/v2xr-pointcloud-only-v1 \
  /data/cjm/datasets/tmp/v2xr_pointcloud_only_v1
/home/cjm/miniconda3/envs/opencood/bin/python -m zipfile -e \
  /data/cjm/datasets/packages/v2xr-pointcloud-only-v1/cjmnet_repe_cuda_flat.zip \
  /home/cjm/OpenCOOD-main/cjmnet_pointcloud_delivery
```

新包没有 cjmnet 外层。原服务器项目不覆盖，原数据不删除。下列命令只在海光服务器执行：

```sh
nohup sh /home/cjm/OpenCOOD-main/cjmnet_pointcloud_delivery/v2xr_pointcloud_only/run_hygon_prepare.sh \
  > /data/cjm/datasets/logs/v2xr-pointcloud-only-v1/launch.log 2>&1 < /dev/null &
echo "PID=$!"
```

过程扫描三个源目录并计算哈希，可能耗时数小时，期间不要改变原数据或同时重新传输。
输出目录必须不存在；任务中断后保留不完整目录供检查，另选新的输出名，不覆盖或自动删除。
默认使用硬链接：精简目录与原目录共享非图片文件的磁盘内容，避免再复制百余GB。
两份目录都按只读数据使用；硬链接下任意一边修改文件内容都会影响另一边。
跨文件系统无法硬链接时明确报错，不自动复制。仅在空间充足时另选输出加 --copy。

```sh
tail -n 20 /data/cjm/datasets/logs/v2xr-pointcloud-only-v1/prepare.log
```

完成标志 SLIM_PREPARED，报告图片省略量、非图片逻辑大小及去重估算大小。
只保留非图片实际文件；所有原始 manifest/records 原样保留，在根部另存 POINTCLOUD_ONLY.json。
原 manifest 仍描述完整原数据，不能再用旧全文件验收器校验精简目录；新入口明确核对图片省略记录。

## 2. 海光上检查优化前后输入是否一致

上述启动器在生成精简数据后自动串行执行本步骤，失败时停止，不要另外启动第二个检查。
不启动网络训练。使用原海光 opencood 环境和已编译官方运行副本，在真实数据上比较少量
Clean/Fog 混合训练样本及 Clean 验证样本的完整批次、标注和随机状态。仅在服务器运行：

```sh
tail -n 20 /data/cjm/datasets/logs/v2xr-pointcloud-only-v1/equivalence.log
cat /data/cjm/datasets/logs/v2xr-pointcloud-only-v1/exit_code
```

全部成功时 exit_code 为0、launch.log 有 SLIM_PIPELINE_COMPLETE，等价检查输出 SAMPLED_INPUTS_EQUAL 并保存 EQUIVALENCE.json。运行中 exit_code 可能尚不存在。报错时保留原数据和日志，
不要继续传输或删除图片。抽样一致不能代替全部样本等价证明，CUDA 预检仍会检查三天气829帧GT。

## 3. 再传输精简数据与代码

仅传新根目录 v2xr-pointcloud-only-v1，使用 rsync -rltH --partial --append-verify；
H 保留硬链接关系，避免 payload/读取视图共享点云在目标端重复占空间。新目录无图片和软链接。
不要对旧完整目录使用 --delete，不删除原海光数据。原始点云、雷达、YAML、记录均保留。

目标最新只剩68G，不能保证同时容纳旧完整数据与新精简数据。先回传 SLIM_PREPARED 的大小，
核实目标空间后再决定扩容或处理目标旧副本；本文不授权删除任何远端目录。
代码在新目录验收后替换工作副本，旧源码先备份，保留已安装环境和已编译runtime。

## 4. 租用服务器校验与运行入口

传输完成后，先后台校验精简目录全部保留文件的 SHA256：

```sh
nohup /workspace/cjmnet_cuda_env/envs/cjmnet-cu124/bin/python -u \
  /workspace/gspr_source/cjmnet/v2xr_pointcloud_only/slim_data.py verify \
  --root /data/cjm/datasets/v2xr-pointcloud-only-v1 \
  > /workspace/v2xr_slim_verify.log 2>&1 < /dev/null &
echo "PID=$!"
```

完成标志 SLIM_CONTENT_PASSED。新版 REPE CUDA 入口通过显式变量启用：

```sh
nohup env POINTCLOUD_ONLY_ROOT=/data/cjm/datasets/v2xr-pointcloud-only-v1 \
  MODE=preflight RESUME=0 \
  sh /workspace/gspr_source/cjmnet/gspr_repe/run_all.sh \
  > /workspace/gspr_repe_pointcloud_preflight.log 2>&1 < /dev/null &
echo "PID=$!"
```

预检会真实使用GPU和官方算子，需分配有卡实例。默认新建独立的
/data/cjm/datasets/gspr-repe-cu124-pointcloud-inputs-v1 和
/data/cjm/datasets/logs/gspr-repe-cu124-pointcloud-v1；不沿用旧阶段提交或权重。
该命令仅预检，不启动正式训练。预检及后续试跑成功后再给正式训练命令。

适配器只替换官方basedataset模块的图片读取函数，返回具有真实.size的对象。
官方源码及其冻结哈希、GPU算子副本均不改动。原雷达分支保留，避免随机数消耗和数据处理变化。
精简路径、尺寸清单身份和适配器源码哈希纳入新实验记录，不能与旧数据模式混用恢复。
本地只允许语法、哈希和人工文本样例测试；真实数据等价、显存及训练效果待服务器实测。
