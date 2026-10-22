# 本地检查记录（2026-10-21）

已完成：

- 8 个 Python 文件静态编译检查，通过；未生成 bytecode。
- `python -B -m v2xr_author_fog_eval.tests`：25 项检查全部通过。
- Git for Windows 的 `sh -n v2xr_author_fog_eval/run_server.sh` 返回0；脚本为 LF，无 Bash 专用语法。
- 424 份冻结依赖源码/元数据的 SHA256 均与本次只读依赖清单一致；冻结评测包自身的既有来源校验通过。
- 冻结评测模块的接口导入检查通过，全程未导入 Torch/OpenCOOD。
- `reference/FINAL_RESULTS.json` 与原 `logs/train/FINAL_RESULTS.json` 逐字节相同，SHA256 为 `13f6ac5de347144cde676270bac651a0e2af8b3a90d89f1a580bcc4f5f78f848`。
- 当前分支为 main；没有创建 branch/worktree，没有修改历史代码或结果。

合成检查覆盖：安全读取请求、标准包装/统一 module. 映射、全部 key/shape/dtype、严格加载与逐 tensor 值、缺失检测头/不匹配/非有限值/加载损坏拒绝；冻结 AP 累计和一对一匹配、None/空预测、禁止 global sort；真实缓存分支的有效孤立帧接纳、完整缓存复用、损坏/缺失/跨合同拒绝；C/D 完整精度对照、分母和 GT 身份、协议分解、目标/帧增益、历史原始记录缺失时不编造 O 匹配计数；控制器失败停止及完成标记。

测试只使用 stdlib、NumPy、PyYAML 和临时的合成内容；没有读取实际点云或 checkpoint，没有执行模型、训练、GPU 或正式科研实验。评分单测使用冻结 AP 函数及玩具 IoU 对应关系；服务器仍须执行原七项真实 tensor/多边形合成检查。

尚未完成且只能在服务器验证：作者文件真实 SHA256 与安全反序列化、官方结构/参数兼容性、HIP/native 环境、冻结数据视图完整性、12帧 smoke、Clean/Fog/Snow 各829帧推理及最终 AP/逐帧结果。此文件不构成服务器评测通过证明，没有作者同集实测成绩。
