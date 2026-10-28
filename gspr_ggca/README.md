# GGCA A0—A3独立实验

唯一启动入口run_all.sh。默认ROCm，解释器和GPU可见性约定来自原gspr_repe/run_all.sh；不安装环境、不编译新算子。A0唯一来源是/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0，禁止重新训练。A1/A2/A3各独立从A0的初始00.pth对齐共享模型及RNG，20轮、batch8、seed20261017、逐CAV Clean4/7和Fog3/7，Snow不进训练。

把完整gspr_ggca/同步到/home/cjm/OpenCOOD-main/cjmnet/。旧gspr_repe/和其他冻结依赖保持原样；PINS.json核对实际调用的旧引擎文件和依赖，不要求旧包中无关文档或测试文件升级。服务器不依赖Git，不创建branch/worktree。

日志和结果总根按本轮要求使用/home/cjm/datasets/logs。默认完整结果目录/home/cjm/datasets/logs/gspr-ggca-v3；临时目录/home/cjm/datasets/tmp/gspr_ggca。与旧/data日志、权重及数据独立。启动前确保所选GPU空闲、home文件系统有足够空间保存完整训练恢复状态；不隐式减少批次或删旧文件。

```sh
mkdir -p /home/cjm/datasets/logs
nohup env BACKEND=rocm GPU=0 MODE=all RESUME=1 \
  sh /home/cjm/OpenCOOD-main/cjmnet/gspr_ggca/run_all.sh \
  > /home/cjm/datasets/logs/gspr_ggca_v3.launch.log 2>&1 < /dev/null &
echo "PID=$!"
```

GPU=0沿用旧入口默认值，使用其他空闲物理卡可改GPU。后台进程自动顺序完成全部阶段，不需逐个启动。MODE=preflight完成输入与A0审计，并在必要时补A0评测；MODE=smoke再完成结构测试及三组真实小训练/推理；MODE=all继续正式三组训练、完整三天气评测、机制诊断与汇总。补A0评测及全量哈希可耗时，均应后台执行。

同一源码、输入、配置和GPU下RESUME=1复用经哈希检查的完整阶段，从最近完整轮次恢复模型/优化器/调度器/损失累计状态/Python、NumPy、Torch CPU和设备RNG；未完成轮次重跑。不要改旧contract、完成标记或哈希绕过失败。技术错误记录STOP.json/current_stage.json和logs/；性能下降不会停止后续组。

产物：baseline_reference.json，preflight/A0_REUSE_AUDIT.{md,json}，self_test/{summary.json,B1_FACTORIAL_DIAGNOSTIC.json}，smoke/A1—A3，stage1/acceptance.json，A1—A3完整训练记录和final_evaluation，mechanism/逐目标诊断，reports/{FINAL_RESULTS.md,FINAL_RESULTS.json,ABLATION_TABLE.csv,FAILURE_CASES.md}。

模型诊断不读取GT。diagnose.py在前向完成后才用冻结Clean框关联输入点，额外固定首12帧/天气做A0—A3的模块目标关联；它不重新计算AP、不优化A0，也不替代完整829帧逐目标匹配。正式A1—A3推理过程另保存所有实际未缓存前向的可靠性/不确定性、几何支持、保护和开销统计。缓存恢复覆盖范围必须按记录说明解释。

本地只允许check_static.py、test_static.py、test_baseline.py。server_tests.py的Torch、完整模型、真实数据及ROCm数值检查通过服务器入口执行。本次交付不宣称已完成服务器A0审计、训练、数值兼容或AP验收。
