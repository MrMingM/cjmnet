# 本地交付检查记录

日期：2026-10-19。环境：Windows；没有运行真实数据、Torch/OpenCOOD 模型或任何科研实验。

已实际完成：

- `python -B v2xr_weather_l012/check_static.py`：新增 Python AST、JSON、LF、独立源码快照、冻结历史包及依赖 SHA256、官方采样代码和调度器调用检查。
- `python -B -m pyflakes v2xr_weather_l012`：无未定义名称或 lint 报错。
- `python -B v2xr_weather_l012/tests_local.py`：9 项纯标准库合成测试通过。覆盖 Clean/Fog 配对及非法后缀、直接执行冻结采样方法的 AST（合成 IO）、3/7 来源比例与逐 CAV 混合、随机状态重放、实际读取适配器对缺失Fog/字节变化的拒绝及原始IO参数类型保持、门槛精确边界、L0 停止/L1 未过仍完整进入 L2 的总调用顺序、阶段中断恢复/跳过/篡改拒绝、完整 epoch 提交检查和冻结评测调用链。
- 已实际通过 `sh -n` 检查 `run_all.sh` 与 `test_shell.sh`。通过假 Python 的 shell 合成测试：带空格路径、RESUME 参数、仅 ROCR_VISIBLE_DEVICES 的 GPU 过滤、下层退出码37向上传递。Git sh 在 Windows 沙箱内初始化失败，随后在沙箱外完成相同的只读语法及临时夹具测试；未调用科研解释器。

上述合成测试只验证调度、数据匹配、身份与恢复提交规则，不把它们描述成真实网络测试通过。

服务器自动执行、目前尚未运行：

- 真实官方模型 C 等价；GSPR 的共享参数/RNG/单位权重、掩码、检测梯度、正则及损失 buffer 重载。
- C/D 实际网络全部 BN 层的发现、仅目标 buffer 更新及前后输出变化。
- C/D FiLM 实际 Scatter shape、初始恒等、检测梯度、主干/BN 冻结和文件保存重载。
- 完整 Adam、scheduler、criterion buffer/累积状态及 Python/NumPy/Torch CPU+CUDA RNG 的下一步恢复一致性。
- 真实官方多 worker 数据观察器不改变样本、增强和 GT 的检查；实际 Clean/Fog 读取与完整训练输入哈希核对。
- 冻结官方 AP 的 None、固定 GT、一对一匹配测试；各 smoke 的原版 O 输出一致性检查。
- L0/L1 20轮、BN 全8083训练Fog帧校准、FiLM固定3轮及全部829帧三天气评测。

MODE=all 会自动执行服务器前置测试和独立 smoke；失败则不启动正式训练。最终性能及真实运行闭环仍以服务器产物为准，目前所有实验结果均为“待服务器运行”。
