# OPV2V GSPR＋A1

海光入口为 `sh opv2v_gspr_a1_v2/run_all.sh`。在项目根目录运行时设置 `ROCR_VISIBLE_DEVICES` 为一张空闲物理卡。默认完成核验、真实 Clean 批次小验收、三种子的训练和评测。设置 `A1_MODE=smoke` 可在验收结束后退出。

```sh
ROCR_VISIBLE_DEVICES=2 nohup sh opv2v_gspr_a1_v2/run_all.sh \
  >> /data/cjm/datasets/logs/opv2v-gspr-a1-v2.launch.log 2>&1 </dev/null &
```

结果位于 `/data/cjm/datasets/logs/opv2v-gspr-a1-v2`。同一入口从上次完整轮次恢复模型、优化器、历史和随机状态。评测按天气提交，中断时重做当前天气。运行需保持源码和数据身份。失败信息写入 `FAILURE.txt`，分阶段日志位于同目录。

`point.py`、`joint.py` 和 `evaluate.py` 分别来自项目中的历史预训练、联合训练和四天气评测入口。联合训练使用 Clean 检测样本和独立的原始模拟天气逐点监督批次。各阶段五轮，参数和种子由 `pipeline.py` 定义。原检测权重为 Detection-only epoch 11。联合选模以逐点 macro AUROC ≥ 0.995 为门槛，在合格轮次中取最低 Clean 验证损失。正式评测在选模完成后执行。

OPV2V 的 PFN 保持原有 GSPR 加权、线性层、归一化、ReLU 和最大池化。A1 使用同一次归一化输出，经逐点 ReLU 后构造柱内关系，从真实投影矩阵取得传感器原点。A1 保留 GGCA 的四近邻和可靠性先验以及有界残差。该适配让残差幅度对基准柱特征保持梯度。零投影使初始残差为零。逐点监督文件按内容核验，检测数据身份按路径、字节数和修改时间清单记录。

最终报告给出三个种子的均值、样本标准差及对历史 GSPR 均值的差。历史 20260906 pilot 种子设置存在差异，表中增量是描述性历史比较。AttFuse 的 Clean 指标保留原日志精度。
