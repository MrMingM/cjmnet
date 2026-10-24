# 新 CUDA 服务器的数据准备

当前服务器尚无数据。环境和两个算子已准备，CUDA 训练入口尚待交付，不运行旧海光启动脚本。

数据建议放 `/workspace/datasets`，与 `/workspace/cjmnet` 源码分开。先确认平台停止和切换有卡模式后此目录仍保留。

保留 V2X-R 的 `split/scene/CAV/timestamp` 目录布局，逐帧需要 YAML、Clean 点云、Fog/Snow 点云、Radar 点云和四张相机图片。虽然模型只用 LiDAR，当前官方数据加载器仍读取雷达和相机文件。不要只复制 Clean PCD，也不要打散目录。

已提取 train 在原服务器的 `/data/cjm/datasets/v2xr-official-train-v2`。迁移时保留 manifest、plan、data_acceptance、records 等已有身份文件，避免重新解压。验证数据需要同时保留官方读取视图与 Fog/Snow 对应文件。旧视图存在软链接，复制后必须保证其目标在新服务器可访问；仅复制链接文字会导致断链。

不需要为准备数据上传正式 test；此次只准备 train 和 validate。旧权重、原始训练日志继续在海光服务器保留。

`check_data.py` 只检查目录与文件存在性，不导入 Torch，不读取点云内容，不运行模型。数据齐备后执行：

```bash
nohup python /workspace/cjmnet/v2xr_cuda_delivery/check_data.py \
  --train /workspace/datasets/v2xr/train \
  --validate /workspace/datasets/v2xr/validate \
  > /workspace/v2xr_data_check.log 2>&1 &
echo "PID=$!"
```

把路径换成实际上传位置。本检查不替代原有内容哈希、8083/829协作帧验收或官方加载器检查；统计的单车帧也不等于协作帧数。
