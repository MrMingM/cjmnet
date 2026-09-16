# 源码基础与改造边界

本地检查过 CoSDH、How2comm 的通信实现。第一版选择 CoSDH 中沿用的 Where2comm 通信选择代码作为规则基础，并复用本项目已有 GSPR/AttFuse 的实际模块与权重。

上游文件：`related_code/CoSDH-main/opencood/models/comm_modules/where2comm.py`

SHA256：`e0fc6940e5f8f91a118aa8a0360ccfe716a989d4cd10c736a91b571478808384`

文件原署名：Yue Hu <phyllis1sjtu@outlook.com>。原许可标记：TDG-Attribution-NonCommercial-NoDistrib。完整仓库许可保留于 `related_code/CoSDH-main/LICENSE`，本目录的 `UPSTREAM_LICENSE.txt` 为原文副本。研究使用与传播须继续遵守上游许可。

`selection.py` 提取并改造了 sigmoid 后取最大前景置信度、Top-K、scatter 二值 mask 的逻辑。改造包括：请求与响应分数先匹配再排序、固定字节预算、稳定同分排序、训练硬选择的软反向近似。原代码训练随机 K、推理另一套阈值/比例；这里不沿用这个训练推理差异。

CoSDH 原文件的请求 mask 在选择后相乘；当前内部 A0B0 使用 `(1-ego_confidence)*sender_confidence` 排序。A0B0 **不是原始 CoSDH，也不是完整 Where2comm 复现**。未引入 CoSDH 的 intermediate-late hybridization；后续论文对比仍需要单独复现原模型。

How2comm 的请求条件响应实现作为结构参考；未复制其时序、互信息或通道通信模块。没有把多个仓库加入 sys.path，也没有导入另一个同名 opencood 包。

新编写部分：质量/支持统计、A/B 预测头、序列化协议、预算分配、独立训练与评测、有效消息掩码适配。新代码是否可靠仍需服务器 smoke test，不能因使用成熟源码就视为已验证。
