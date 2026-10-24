# UECP 共享源码隔离恢复

2026-10-22。适用于 `/data/cjm/datasets/logs/uecp_gspr_v2xr_v1`：已登记resume-v2修复、恢复门禁已通过、正式E0第1轮训练已提交，验证进程因共享源码变化停在 `E0_dev_01`。

## 原因

这次验证还没有开始读取模型，就被原v2运行器的源码身份检查拦住。服务器三个共享文件已被另一个Raygen实验更新。已逐字节对照本地历史交付ZIP：

| 文件 | UECP原登记SHA256开头 | 服务器当前SHA256开头 | 变化 |
|---|---|---|---|
| `preflight.py` | `82be7404` | `27489af2` | Raygen预检新增计算规则登记 |
| `runtime.py` | `de6db71d` | `0e48bcf2` | bootstrap新增`stable_crop.configure(torch)`，会改变计算设置 |
| `training.py` | 原`b6577e83`，v2已审查`cc5f523e` | `1952a740` | Raygen新增恢复检查和计算规则检查 |

不能仅放行新哈希：UECP确实调用共享runtime的bootstrap，直接运行当前版本会改变这次实验的计算环境。

## 此包做什么

一个新ZIP包含：原resume-v2运行器原字节副本、原`runtime.py`/`preflight.py`、已登记的`cc5f...training.py`及独立恢复入口。后三者只复制到本次输出的私有目录，并以原Python模块名装载到**当前UECP进程**。每个训练/验证子进程均走同一入口；当前项目中的Raygen文件不会被覆盖。

原v2源码检查复核进程实际执行的私有源码视图。其返回的审查结果仍须与旧v2登记逐字段相同；其他依赖和完整UECP包继续严格校验真实项目文件。共享目录的三个当前文件另行记录实际SHA，未知版本仍拒绝。

- 模型、损失、数据划分、增强、优化器、预算、选模、阈值和原生AP不变。
- v2已登记的精确加载、Adam比较与重放检查规则不变。
- 保留原request、contract、PINS、源码快照、旧ZIP、已有checkpoint和旧错误历史。
- 原E0第1轮训练提交及其所有产物重新验SHA后复用；从失败的验证阶段继续。若提交缺失/损坏则停止，不伪装完成。
- 已成功的smoke与训练按原控制器收据复用；验证失败的该轮验证会重跑，尚未完成的后续训练继续按原规则执行。
- 原旧源码仍按原字节执行。没有把当前Raygen的裁块/确定性设置加入UECP。

## 上传与唯一后台继续命令

上传 `uecp_gspr_dependency_snapshot_v1.zip` 到 `/home/cjm/`，**无需解压，也不需要替换旧v2 ZIP或项目源码**。执行：

```sh
ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1
ZIP=/home/cjm/uecp_gspr_dependency_snapshot_v1.zip
LOG=$(mktemp "$ROOT/diagnostics/dependency_snapshot_launcher.XXXXXX")
nohup /home/cjm/miniconda3/envs/opencood/bin/python -B "$ZIP" run \
  --output "$ROOT" --mode all > "$LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

该入口先完成私有源码/原提交/修复谱系检查，再调用原v2总控制器。GPU、数据路径、随机种子从原request和smoke提交恢复；不要另开原总脚本或旧v2运行器。原控制器/worker尚存活时会被锁拒绝。

后续中断恢复也执行同一段命令和同一ZIP，不从头启动，不需要重复旧受控诊断的`--evidence`。

查看当前阶段、失败阶段详细日志：

```sh
cat /data/cjm/datasets/logs/uecp_gspr_v2xr_v1/current_stage.json
tail -n 100 /data/cjm/datasets/logs/uecp_gspr_v2xr_v1/logs/E0_dev_01.log
```

如只想检查源码与提交，可在后台启动命令的`--mode all`后加`--check-only`。它只读已有产物的SHA和元数据，保存私有源码及审查记录，不加载模型、不前向/反向、不更新优化器。完整入口已自动包含这些检查，无需操作者先执行第二条命令。

## 新记录

- `source_repair/uecp_dependency_snapshot_v1.json`：本ZIP身份、原v2修复记录/契约哈希、私有源码完整SHA及旧审查视图。
- `source_repair/uecp_dependency_snapshot_v1/modules/`：三个原版本源码；禁止手工修改。
- `source_repair/uecp_dependency_snapshot_v1/original_resume_v2.zip`：原v2 ZIP，SHA必须为`eef5dbb83f03b3112c582a458040761d708495df1fb309f302383dc0e46e72ca`。
- `observations/`：每次启动的实际共享源码SHA、模块绑定、种子和审查记录。
- 新阶段提交及新最终/失败报告增加`execution_dependency_snapshot`。已经完成的旧报告不被重写。

本地只执行静态、源码/ZIP字节核验和合成工程测试，不运行任何真实科研实验。真实验证、后续训练和最终报告是否成功，以服务器实际日志与提交为准。该补丁处理此次已核对的源码版本冲突；其他错误仍如实停止。

交付检查：21项新增源码隔离/ZIP/模块绑定/元数据测试和8项原v2修复回归测试，共29项通过；Pyflakes和Python 3.10语法检查通过。未在本地导入Torch/OpenCOOD，也没有训练或推理。
