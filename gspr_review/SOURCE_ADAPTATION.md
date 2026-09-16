# 源码移植记录：参考差异与校正增益（2026-09-10）

## 来源与移植范围

已读取用户解压的三个仓库。新实现位于 `counterfactual.py`；没有运行作者的数据下载脚本，也没有加载作者权重。

| 来源 | 已读取源码 | 采用部分 | 适配与未采用部分 |
| --- | --- | --- | --- |
| [CF-VQA / CVPR 2021](https://github.com/yuleiniu/cfvqa) | `related_code/cfvqa-master/cfvqa/models/networks/cfvqa.py`，`forward`、`fusion`、`transform` | 事实输出与参考输出相减的结构 | 原代码为 `z_qkv-z_q`；这里是同一 MLP 对真实消息和中性消息的评分差。不移植 VQA 编码器、分类损失、固定 `.cuda()`、可学习常数替代全部输入。不是原文完整因果估计器。 |
| [KalmanNet / IEEE TSP 2022](https://github.com/KalmanNet/KalmanNet_TSP) | `related_code/KalmanNet_TSP-main/KNet/KalmanNet_nn.py`，`KNet_step`、`step_KGain_est` | 增益作用于差异，再用于更新原状态的分工 | 原代码为 `KGain × dy`；这里以有界小网络生成标量增益，作用于参考评分差，再做 tanh 有界化。不移植 GRU、时序状态、协方差模型；不相减不同物理点的可靠概率。 |
| [InterSHAP / AAAI 2025](https://github.com/LauraWenderoth/InterSHAP) | `related_code/InterSHAP-main/synergy_evaluation/interaction_values.py`，`shaply_values`、`shaply_interaction_values` | 两来源 Shapley 交互矩阵的计算规则 | 独立实现两来源特例：每个非对角元是混合差分的一半，对角元为 Shapley 值减去非对角元。不移植 pandas/SHAP 通用枚举框架，不冒称完整 InterSHAP 数据集指标。 |

CF-VQA 原仓库附 Apache-2.0 LICENSE，原始文件保留在下载目录。其余两个解压目录根部未发现 LICENSE。本轮对公开方法的基本数学运算进行独立适配，没有复制其 GRU/解释器的大段源码或重新发布原仓库。今后如需直接搬入更多源码，须先核对对应文件许可。原始文件哈希见 `source_adaptation_hashes.json`。

## 第一版模型

输入继续为原来的 11 维：7 个本地点描述、4 个已解码邻车证据 `[b_r,b_n,u,support]`。GSPR、BEV 主干、检测头、AttFuse 参数与 BN 冻结。请求、回包、邻车选取及真实字节预算维持原协议。

中性参考保留自车 7 维输入、消息的 u、support 和 `b_r+b_n`，只把可靠/噪声证据改为各占原总量一半。量化后证据不一定严格和为 1，因此使用实际收到的证据总量，不强行重归一化。

```
d = score(local, actual_peer) - score(local, neutral_peer)
k = sigmoid(gain(local_p, local_u, actual_peer))
delta = max_adjustment * tanh(k * d)
new_weight = clamp(old_weight + delta, reliability_floor, 1)
```

`contrast` 对照固定 k=1；`contrast_gain` 学习 k。两者最后评分层零初始化，使初始点权重严格不变。增益初始为 0.5，避免两部分都零初始化造成无梯度。第一步增益梯度为零是预期行为：评分差学出后，增益才开始获得梯度；测试覆盖了这个过程。

这只保证真实消息与中性参考一致时不校正、纯本地共同偏移会相消。它不保证消息一定有用，也没有赋予 GSPR 的 u 高斯协方差或目标漏检概率的含义。该中性参考是实验假设，不等于物理上真实存在的观测；保留未知和支持也不能保证其完全处于训练分布内。

沿用原支持条件：请求内、原有效点、收到的可靠证据和支持大于零才允许改变。没有支持不是噪声证明。由于通信预算与准入掩码维持原版，本实验不能解决覆盖不足或原准入条件排除了有效反证的问题；这是后续可以单独验证的环节。

## 实验与诊断

运行 `bash gspr_review/run_counterfactual.sh`：

1. 全部协议/张量测试；分别验证 legacy、contrast、contrast_gain 的真实数据连接。
2. 使用同一基础配置、种子、训练数据、天气协议和训练步数，分别重新训练三个小网络。模型初始化后重置数据 RNG，避免不同参数量改变采样顺序。
3. 每个架构分别评测 Clean/Weather 下的 normal、neutral、permuted、protocol、a0b0，总计 30 组。老 checkpoint 保留，本次不自动复用或覆盖。
4. 每个训练结果运行固定掩码诊断，输出已有权重/PFN 指标及两来源参考交互统计。
5. 汇总精确 AP、通信字节和检测变化到 `comparison.json`，并检查消息对照的帧数和字节一致。

neutral/permuted 是**离线内容干预评测**，仍按原始实际包计费、保持原准入掩码，不是另一个真实通信协议。permuted 按已访问几何格循环置换收到的完整四维证据，同一格内的点使用相同置换向量。contrast 系列的 neutral 应匹配同协议不复核；legacy 的 neutral 未必不修改，这是其行为对照。

两来源交互诊断只分析 bounded pre-clamp update，默认每帧最多等距抽取 2048 个可修改点。自车参考取该帧可修改点的本地输入均值，邻车参考为上述中性证据。统计是相对于该参考的交互量，不是因果证明、不是 AP，也不是全体点上的精确数据集 InterSHAP 分数；增益和评分差的均值来自同一抽样。

## 验证与服务器命令

本地在项目临时虚拟环境、CPU PyTorch 2.5.1 中通过 28 项新旧测试（无跳过），包括实际自动求导、两步增益梯度、掩码与内容干预、消息编解码和交互矩阵完整性。Python 编译、Bash `-n` 检查通过，原 14 个冻结源文件哈希一致。尚未执行服务器真实多车前向、训练或 AP 评测；这些由下述脚本完成。测试环境位于 `tmp/review_test_env`，不需要同步到服务器。

将更新后的整个 `gspr_review` 目录同步到服务器原项目后运行：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
bash gspr_review/run_counterfactual.sh
```

默认使用此前固定 GSPR 配置/权重目录 `gspr_joint_full_v1_seed20260907_20260907_161155`。默认新输出为 `/data/cjm/datasets/logs/gspr_cf_seed20260909_<时间戳>`。种子仍用 20260909，以保持本轮架构对照一致。可通过既有 `FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT`、`FOG_LOOKUP_DIR` 环境变量覆盖资源位置。

源码仓库只用于阅读和移植，服务器运行不需要三个原论文仓库、VQA 数据集、Kalman 仿真数据或 SHAP 依赖。

关注顺序：先确认 normal 优于同协议 protocol 和内容置换；再比较是否超过全预算 A0B0；最后比较 contrast_gain 是否优于 contrast。若只增大交互数值而没有检测收益，不判定成功。默认依然是小样本机制实验，不能据此宣称最优或跨数据集有效。
