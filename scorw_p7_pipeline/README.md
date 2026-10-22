# P7 一次启动：检查 → SMOKE → 正式 → 结果归档

将整个 `scorw_p7_pipeline/` 放到服务器仓库根目录，与已交付的 `scorw_p7/` 并列。保持P7包和snapshot完整；无需修改原P7研究代码、配置或源码快照，不依赖Git。

唯一启动入口：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs/scorw-p7-v1
nohup env RESUME=1 sh scorw_p7_pipeline/run_all.sh \
  >> /data/cjm/datasets/logs/scorw-p7-v1/pipeline.log 2>&1 &
```

自动依次执行MODE=checks、MODE=smoke、MODE=full。full内部串行执行P7-0至P7-4，最后由原P7入口输出报告和小型结果包。SMOKE通过后无需人工再次启动。

首次运行与中断恢复均使用同一命令。默认RESUME=1，在没有旧合同时正常创建；已有结果重新进入原入口核验源码、配置、原始输入及产物哈希，再复用已完成计算。总控状态不会用来跳过原入口检查。RESUME=0仅允许尚无P7结果或总控状态时启动，绝不删除结果。

任一步退出非零、完成门禁不符、源码/配置改变、报告/归档哈希失败，就停止后续步骤。全流程有独立锁，重复启动会被拒绝；运行期间不要另起单阶段命令。SIGTERM会停止子进程后释放锁；SIGKILL/断电可能留下running状态，恢复仍须重新核验。

这只根据软件正确性门禁决定是否进入下一步，不根据正负效果、匹配率或科学结论选择天气、场景或seed。不会训练检测器、跑AP或生成天气PCD。正式结果为INCONCLUSIVE或STOP_OR_REDESIGN时原样保留。新独立验证不足时，最终状态为complete_development_only，决策INCONCLUSIVE_VALIDATION_UNAVAILABLE，P7-3保持not_executable。

## 进度与结果

```sh
tail -n 100 -f /data/cjm/datasets/logs/scorw-p7-v1/pipeline.log
cat /data/cjm/datasets/logs/scorw-p7-v1/pipeline/PIPELINE_STATUS.json
```

日志有START、PASS、STOP。独立状态、尝试次数、PID、阶段及错误在PIPELINE_STATUS.json。流程完成后同目录生成PIPELINE_RESULT.json和便于一次回传的RESULT_SUMMARY.txt。

```sh
cat /data/cjm/datasets/logs/scorw-p7-v1/pipeline/RESULT_SUMMARY.txt
```

正式报告仍在 `/data/cjm/datasets/scorw-p7-v1/`；SMOKE在独立的`-smoke`目录。最终回传包：

```text
/data/cjm/datasets/scorw-p7-v1/P7_SMALL_RESULTS.tar.gz
```

总控还会核对SMOKE与正式运行的场景、seed、CAV偏移和重叠场景天气场身份一致；采样预算按各自预注册SMOKE/正式配置执行。所有报告清单、最终包和LATEST完成记录均核验哈希。

不自动登录服务器、不自动上传或提交GitHub。源/配置变化后不能继续复用同一总控合同。启动前校验失败会写到总日志，并保持原有状态文件；阶段运行失败则记录stopped。

`tests.py`只用合成文件与模拟子进程检查顺序、停止、恢复及完成状态，不读取真实数据。
