# E0复用与E1/E2独立并行实验

唯一入口run_all.sh，POSIX sh、LF。默认ROCm及现有opencood解释器。设计见METHOD_DESIGN.md；旧文件只读，无Git/分支/worktree前提。

```sh
nohup env BACKEND=rocm GPUS=2,4 MODE=all RESUME=1 \
  sh /home/cjm/OpenCOOD-main/cjmnet/gspr_e1e2_v5/run_all.sh \
  > /home/cjm/datasets/logs/gspr_e1e2_v5.launch.log 2>&1 < /dev/null &
```

默认输出/home/cjm/datasets/logs/gspr-e1e2-20261101-v5。E1、E2各自存储权重、完整状态、采样、Clean选模、final_evaluation和mechanism。lanes/E1与lanes/E2记录独立阶段及日志。父current_stage记录整体阶段。后续恢复用同一命令和RESUME=1，勿在原控制器仍存活时重复启动。

MODE=preflight只全量输入及E0审计；MODE=smoke只结构和真实小训练/推理；MODE=all验收后正式训练。GPUS必须两张不同的物理卡。每个新计算通道开始前检查显存≤16MiB；不空闲则停止，不终止外部任务。不自动降batch或更改GPU映射来绕过合同。

默认E0=/home/cjm/datasets/logs/gspr-ggca-v3/A1；B2原初始状态=/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B2/resume_states/00.pth。所有初始状态及选模路径从记录取得并验哈希。不重训E0，不加载其训练后权重进行E1/E2微调。

正式结果由reports/FINAL_RESULTS.json/md自动生成，包括三天气三协议、匹配计数、目标距离/点数召回、采样轨迹一致性和机制控制。smoke数值仅验工程，不写为正式成绩。
