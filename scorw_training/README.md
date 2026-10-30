# SCOR-W S0/S1/S2

S0全Clean原始AttFuse；S1每轮精确3187帧Clean+3187帧Fog原始AttFuse；S2相同混合样本与原始GSPR-AttFuse。三组独立随机初始化，共享全部AttFuse主干、PFN、检测头初始参数；不加载历史模型。每轮遍历6374训练帧一次，末批2帧保留。混合粒度为整协作帧，全部活跃CAV使用同一天气；逐帧场景/CAV/天气/输入哈希记录逐轮比较。

训练参数：seed20261030、20轮、batch4、Adam lr0.002 eps1e-10 decay1e-4，MultiStepLR在完整10/15轮后衰减0.1。三组共享翻转、旋转和缩放增强，整帧所有点和框共用同一增强，随后范围裁剪；单车点顺序也按同一身份确定。最大CAV5、70m通信、D6筛选顺序和固定Clean训练GT。训练使用原生32000体素上限，评测70000。

S2严格复用attfuse_gspr/reliability.py、pillar_vfe.py和point_pillar_gspr_attfuse.py。检测损失复用既有GsprPointPillarLoss，point_weight=0，不传逐点target；保留原有keep=.02、evidence=.0001、target_keep=.9正则。S2与S1差异包括GSPR及其原有正则，不能单独归因于网络结构。每批检查全部已产生梯度、全部状态有限值、GSPR非零梯度及参数实际更新，失败停止。没有GGCA/REPE、没有在线天气。

每轮完整1980帧Clean/Fog F评测，以两项AP70平均值选择，平局更早轮次。主要F协议保持D6 Clean GT、通信后截断、原始点序、按帧累计AP；Validate输入及GT哈希逐帧与D6匹配。最终额外执行真正原生数据集getitem/collate/GT及模型推理，保留原生CAV截断顺序、shuffle及GT过滤。每个协议三组GT身份必须相同。输出AP30/50/70、TP/FP/FN以及按ego三维距离0–20/20–40/40–60/60–80/80–120/120以上分桶召回。

三组全部训练和选模后才进入最终Validate，全部Validate完成后锁定三组权重并统一执行2170帧Test；Test GT不会在smoke读取。Preflight仅哈希核验Test原文件字节，不解释目标标注。完整Test结果只针对已锁定权重，不能反向修改协议。真实天气校验缺失，单seed实验。

统一POSIX入口：

```sh
GPU=5 MODE=preflight RESUME=0 sh scorw_training/run_all.sh
GPU=5 MODE=smoke RESUME=0 sh scorw_training/run_all.sh
```

SMOKE独立目录：3组各2次真实训练更新，2帧Clean/Fog F选模及2帧F/原生评测，不执行Test，不作为科学结果。FULL为3组20轮、全部Validate和锁定后的全部Test。预检查会哈希读取所有split的PCD/元数据，耗时以实际日志为准。

```sh
mkdir -p /data/cjm/datasets/logs/scorw-training-v1.1
nohup env GPU=5 MODE=full RESUME=0 sh /home/cjm/OpenCOOD-main/cjmnet/scorw_training/run_all.sh > /data/cjm/datasets/logs/scorw-training-v1.1/full.log 2>&1 < /dev/null &
tail -f /data/cjm/datasets/logs/scorw-training-v1.1/full.log
```

仅旧进程退出后恢复：

```sh
nohup env GPU=5 MODE=full RESUME=1 sh /home/cjm/OpenCOOD-main/cjmnet/scorw_training/run_all.sh >> /data/cjm/datasets/logs/scorw-training-v1.1/full.log 2>&1 < /dev/null &
```

输出根目录按preflight/smoke/full隔离：保存源码配置合同、初始主干、逐组HISTORY/SELECTED、逐轮model/SAMPLING/COMPLETE、训练优化器和调度器及CPU/设备随机状态、逐帧评测缓存和最终FINAL_RESULTS.json/FINAL_REPORT.md。训练每50批和每轮末保存完整状态；中断从最近提交处重放未提交的训练，不跳样本。原子内容哈希文件+指针更新，阶段锁及已完成产物哈希校验；源码配置改变拒绝复用旧合同。不得修改历史数据、模型或旧实验。

接收框后处理的IoU仅对包围盒严格不相交的框直接返回精确零；其余调用原始compute_iou，分数、top1000、NMS阈值及匹配顺序不变。服务器测试要求包含相交/重合/接触/旋转框的IoU逐位相同。
