# 稳定计算 v4：代码交付，服务器验证待运行

用户在 2026-10-20 明确选择“修复计算，保留 v3 验收”。新版修复计算重复性，保留全部既有证据。
本地只进行静态、shell 语法和合成检查；没有导入 Torch/OpenCOOD、加载真实权重或运行实验。

## 已有证据与修改

原设置的检测路径有微小梯度变化；优先使用确定性算法后，所选微批的检测梯度完全一致。
辅助网络给裁块的梯度一致，经过裁块组合后回传到鸟瞰特征的梯度有微小变化，独立重放也复现。
这些单批证据不等于已经完整解释原 E2 多次更新后的 Adam 超差，也不构成放行依据。

- E0/E1/E2/E3 和推理均启用 `torch.use_deterministic_algorithms(True, warn_only=False)`，
  `cudnn.benchmark=False`、`cudnn.deterministic=True`。不支持时明确失败，不回退到只警告。
- 连续学生特征仍按相同窗口、四邻点双线性权重、零补齐、align_corners=False 提取。
  新实现用基础 Torch `index_select`，由严格确定性设置约束重复索引的梯度累加。
  不插值 O/E/U 标签，不 detach 条件，不增加容量或改变 E1—E3 的输入/损失定义。
- `resume_compare.py` 整文件及 `smoke.resume_check` 的 v3 验收逻辑保持原样。
  rtol=1e-4/atol=1e-5、模型逐元素检查、Adam RMS 检查、精确加载和 RNG/LR/step 检查均保留。
- 新服务器阶段 `stable_calculation_checks` 检查原裁块前向/梯度等价性、重复窗口和边界补零，
  以及真实 E2 单帧三次全参数梯度精确重复；不执行优化器更新。
  原定义的参考反向仅在该对照步骤临时允许非确定性，随后立即恢复严格设置；训练没有此回退。
- 等价性检查使用原前向检查容差 rtol=1e-5/atol=1e-6；重复结果要求精确相等。
  新增检查通过后仍须执行四组短训练恢复及三天气 O/O*/F；实际服务器检查尚待执行。
- 同一组先完成短训练、重放、精确加载和 v3 比较，再评测两轮权重，减少失败前的评测耗时。

依据：[PyTorch 2.5.1 确定性操作说明](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/__init__.py)
及 [GridSampler 坐标定义](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/GridSampler.h)。
实际 HIP 能力以服务器检查为准；没有跨平台或完整训练逐位相等的预先保证。

## 新目录与缓存继承

训练计算实现变化，旧 smoke 权重和通过记录不能充当新版验证。新输出建立新契约，旧目录保留。
`STABLE_PARENT.json` 固定此前交付的 v3 源码，用于审计旧来源链与数据构造代码。
不修改旧 contract 或旧哈希以接受新版。

`REUSE_DATA_FROM` 只继承数据：

1. 排除旧任务仍在运行，核对旧来源链、预检完成、Python/包版本、输入身份和实际 GPU。
2. 核对 config、标签几何、data/cache/geometry/loss 代码与只读依赖没有改变。
3. 复用 train manifest、冻结评测证据、输入元数据和已完成天气解码记录。
4. 逐项验证已准备 NPZ 的 SHA256；新目录链接不可变原缓存，文件系统不支持时复制。
   不修改旧 payload、journal、训练或验证文件。原缓存所在旧目录必须继续保留。
5. 新 journal 登记新源码身份、原 journal 路径/哈希及相同 payload 哈希；
   窗口身份使用新 base journal 哈希重建，标签数组不重算、不改动。
6. 不导入旧训练/验证状态。GPU 容量重新测量，固定旧合同的 microbatch，失败后不自动缩小。

导入证据为 `data_reuse.json`、`data_reuse_inventory.json`，计算策略为 `compute_policy.json`。
最终报告包含上述来源与新增检查。没有 Git 仓库或分支的运行前提。
初始化中断可重复同一条命令继续；初始化完成后不会重置新版已完成阶段。

## 上传和后台启动

将完整 v4 包及根目录 `launch_v2xr_e0e3.sh` 同步到原代码目录。
交付 zip 包含这两项及上一版 v3 的源码备份 zip，不含数据或权重。
zip 上传到以下目录后解压：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
unzip -o v2xr_raygen_e0e3_stable_v4_20261020.zip
```

物理卡继续使用 2。第一次创建新运行，后台执行完整流程：

```sh
OLD_RUN=/data/cjm/datasets/logs/v2xr-raygen-e0e3-20261020T010502Z-1870574
NEW_RUN=/data/cjm/datasets/logs/v2xr-raygen-e0e3-stable-v4-$(date -u +%Y%m%dT%H%M%SZ)
GPU=2 MODE=all BACKGROUND=1 RESUME=0 \
  REUSE_DATA_FROM="$OLD_RUN" OUTPUT_ROOT="$NEW_RUN" \
  sh /home/cjm/OpenCOOD-main/cjmnet/launch_v2xr_e0e3.sh
printf 'NEW_RUN=%s\n' "$NEW_RUN"
tail -f "$NEW_RUN/launcher.log"
```

日志先显示缓存继承与 `stable_calculation_checks`；新增阶段失败就停止，没有后续训练。
通过后重新检查容量及 E0—E3；smoke 全部通过后 MODE=all 自动继续完整标签与正式训练。
只检查时用 MODE=smoke。后续恢复填实际打印的新目录，不再次生成时间戳：

```sh
GPU=2 MODE=all BACKGROUND=1 RESUME=1 \
  OUTPUT_ROOT=/data/cjm/datasets/logs/此处填写实际的新目录 \
  sh /home/cjm/OpenCOOD-main/cjmnet/launch_v2xr_e0e3.sh
```

先前独立诊断脚本针对旧源码，勿在其原命令中混用新版源码与旧权重。
新版检查已集成到主流程；出现失败请回传新目录 failure.json 和对应日志。
