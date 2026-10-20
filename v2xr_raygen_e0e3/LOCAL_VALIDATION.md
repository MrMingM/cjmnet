# 本地检查记录

日期：2026-10-19。范围限于静态及纯 Python/NumPy 合成检查。没有读取真实PCD、导入Torch/OpenCOOD、使用GPU或运行科研训练/推理。没有实验AP结果。

- 16项合成检查通过：不同传感器原点；端点/空域/后方未知；多车冲突与相同x0；重复点不变；最近回波及角度环绕边界；体素边界；ego/仅邻车来源；同步几何变换；进入窗口的外部射线；全未知/空标签；类与来源分母；窗口随机流独立；掩码packbits往返；受控选模与最早并列；原生真实AST经适配后的零GT/空CAV路径（纯NumPy替身，无原生runtime导入）。
- 新增Python源码AST解析、JSON解析、配置约束、原生适配分支匹配、LF与源码SHA256快照校验通过。
- launch_v2xr_e0e3.sh、run_all.sh 均通过 `sh -n`。第一次Git sh在Windows沙箱内创建内核对象失败；随后在沙箱外仅执行语法检查成功，没有执行启动脚本。
- v2xr_train_attfuse、v2xr_protocol_audit、v2xr_official_repro 原source_snapshot逐项核对：缺失0，变化0。326个实际只读依赖源码/配置/文档文件记录在PINS.json。

复查命令：

```sh
python -B v2xr_raygen_e0e3/check_static.py
python -B -m v2xr_raygen_e0e3.tests_local
sh -n launch_v2xr_e0e3.sh
sh -n v2xr_raygen_e0e3/run_all.sh
```

尚待服务器：真实数据完整性/标签类别、原生输出等价、真实辅助梯度、单HIP显卡容量、真实四组smoke、Adam/scheduler/RNG恢复重放、评测中断恢复、完整训练与829帧三天气O/O*/F以及最终报告。自动检查已接入MODE=all。未声称这些服务器检查已经通过。
