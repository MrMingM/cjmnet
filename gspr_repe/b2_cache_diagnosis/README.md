# B0/B2 Fog、Snow只读缓存对照

一条入口完成两模型身份、完整829帧评测和固定Clean GT核验；读取原官方F协议的IoU0.7匹配，按frame_key+object_id对齐恢复/丢失；统计距离、当前天气框内点数、有点车辆数及观测代理。Clean作为简要参照。只使用标准库，不导入Torch/OpenCOOD，不加载点云，不训练、推理、重算AP/IoU或修改权重/原结果。

`cache_core.py`从既有B1只读诊断抽取未改变的缓存校验、匹配读取和统计函数；内部比较标签别名B1仅用于复用，实际加载的是B2的合同、权重和缓存，公开报告全部为B0/B2。GT清单自身哈希绑定模型合同，因此比较共同冻结GT身份和逐帧目标、几何、顺序，不要求不同权重的清单自身哈希相同。

## 邻域观测的限制

预测缓存没有B2的俯视网格点占用、邻居特征和实际增强权重。报告将实际八邻域证据标为NOT_CACHED。额外提供的2/5/10米附近其他Clean GT目标的当前天气点数/最近距离，是目标级代理，不能代替实际八邻域，不能证明传播来源，且不含背景/路面点。该半径仅用于描述，不改变模型或选择权重。

框内点数来自原已保存GT审计，是多车计数之和，可能重复观测同一表面。有点车辆数是覆盖代理，不是空间覆盖率。缺失审计时仍统计目标变化和距离，点数/观测分层标missing；审计存在但哈希或GT不一致则停止。缺失预测或匹配缓存不补算。

## 部署与单一入口

上传本次`b2_cache_diagnosis_tools.zip`至`/data/cjm/datasets/packages/`。与B1工具一样，实体必须放在gspr_repe包外，安装器仅添加目录链接，并校验旧REPE递归快照不变。不要将实体目录解压进gspr_repe，不修改旧source_snapshot.json，不覆盖B1诊断工具或结果。

```sh
(
set -eu
PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
TOOLS=/data/cjm/datasets/tools/repe_b2_cache_diagnosis_v1
test ! -e "$TOOLS"
mkdir -p "$TOOLS"
"$PYTHON" -m zipfile -e /data/cjm/datasets/packages/b2_cache_diagnosis_tools.zip "$TOOLS"
"$PYTHON" -B "$TOOLS/install.py"
cd /home/cjm/OpenCOOD-main/cjmnet
nohup sh gspr_repe/b2_cache_diagnosis/run_all.sh \
  > /data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4.b2_cache_diagnosis.launch.log \
  2>&1 < /dev/null &
echo "PID=$!"
)
```

输出至`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/b2_cache_diagnosis/`：

- `B0_B2_DIAGNOSIS.md`：中文汇总、分层恢复率/丢失率、典型案例及限制。
- `TARGET_CHANGES.csv`：Fog/Snow逐目标恢复/丢失清单，含距离、点数、观测代理及两模型匹配IoU和分数。
- `TARGET_CHANGES.json`：三天气统计、Fog/Snow完整变化目标、各阈值匹配证据及预测框、各车点数、分层和逐帧变化。
- `INPUT_AUDIT.json`：输入哈希和一致性证据；`COMPLETE.json`：完成输出哈希；失败为`FAILURE.json`，不会补算。

恢复率以B0漏检目标作分母，丢失率以B0正确目标作分母；另列距离×点数联合分层。AP受排序及正确检测分布影响，目标数量和误检变化不能直接解释全部AP变化。分层与邻近代理只构成描述性关联，不是模块机制的因果证明。报告重复启动拒绝覆盖。脚本兼容POSIX sh/LF。
