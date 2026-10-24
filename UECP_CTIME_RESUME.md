# UECP 训练点云状态时间变化的恢复入口

适用输出：`/data/cjm/datasets/logs/uecp_gspr_v2xr_v1`。2026-10-24。

服务器已完成 E0 第3轮训练，在 `E0_dev_03` 开始前被输入身份检查拦截。已检查的
`2024_06_24_19_56_04/182/000060.pcd` 大小、修改时间、实际路径不变，只有 ctime 变化；
实际 SHA256 与原提取清单、原解码记录都为
`599fdb0e646d70e47cd4d9d8b24b0bc7b1ccfaa19debdfa639fdadb08aabc09f`。
ctime 是文件状态变更时间；本次没有足够证据判定由哪项系统操作引起。

## 恢复规则

新 ZIP 嵌入之前的 `uecp_gspr_dependency_snapshot_v1.zip` **原始完整字节**，
SHA256 必须为 `2d40d997bb8228fd78c2639041c78a37f3b8fe6af4e8489a9f50aac19a956377`。
原始 resume-v2、私有源码、源码检查、计算设置和已登记契约继续执行。

1. 原来的源码、契约、诊断证据、预检及训练提交检查先通过。
2. 扫描原始 `preflight/input_stamps.json` 的**全部**文件。
3. 只有已验收训练清单内、路径/大小/修改时间不变、仅 ctime 变化的文件可以进一步核验。
4. 对这些文件重新计算完整 SHA256，与原始训练清单比较；普通天气点云还须保留原成功解码记录。
   哈希前后文件状态必须稳定。未知文件、内容变化、路径/大小/修改时间变化均停止。
5. 通过后写入独立不可覆盖的审计收据。原始 stamp、manifest、内容收据、历史权重、历史结果完全不改。
6. 各训练/验证子进程均通过新 ZIP 启动，验证收据哈希及原始锚点。
   只对收据明确列出的文件接受此次新 ctime；文件再次变化立即停止。
   原始运行时源码检查和其他文件检查保留。
7. 复用原控制器验证通过的已完成阶段。E0 第3轮训练提交和其全部产物再次校验，失败的验证阶段重跑。

不会重新训练已经通过校验并提交的前三轮；不会降低 AP 门槛、放宽模型数值容差、修改数据或实验设置。
如果存在其他变化，失败审计会记录正在检查的路径和错误，流程停止，不宣布完成。

**证据限制：当前字节一致无法证明过去从未发生短暂内容更改。** 本次恢复建立在当前内容与
原始验收/解码内容一致的证据上；不凭此声称已经独立证明所有历史训练输入从未变化。
原实验训练轨迹和提交仍被保留，便于后续检查。

## 上传及唯一后台启动命令

上传 `uecp_gspr_ctime_resume_v1.zip` 到 `/home/cjm/`。无需解压，不替换旧 ZIP 或项目源码。

```sh
ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1
ZIP=/home/cjm/uecp_gspr_ctime_resume_v1.zip
LOG=$(mktemp "$ROOT/diagnostics/ctime_resume_launcher.XXXXXX")
nohup /home/cjm/miniconda3/envs/opencood/bin/python -B "$ZIP" run \
  --output "$ROOT" --mode all > "$LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

同一条命令也用于后续中断恢复。GPU、随机种子和数据路径从原请求恢复。
每次控制器启动都会重新扫描登记输入并核验发生变化的训练文件；扫描时间取决于文件数量和存储速度。
已经有完整阶段结果时，原控制器按哈希收据复用。
不要同时启动旧入口。锁冲突会报错，不会杀死其他进程。

查看失败的验证日志：

```sh
tail -n 100 /data/cjm/datasets/logs/uecp_gspr_v2xr_v1/logs/E0_dev_03.log
```

新增收据在 `source_repair/uecp_ctime_resume_v1/audits/*.json`。
新阶段提交和新报告增加 `input_ctime_audit`，包含收据位置、收据 SHA 和新启动包 SHA。
已经完成的历史报告不重写。

## 验证边界

本地仅静态检查和使用临时文本文件的合成工程测试；不读取服务器点云，不加载模型，不训练或推理。
真实数据全扫描及第3轮验证能否继续通过，以服务器实际审计与日志为准。

交付检查：25项新增合成测试和8项原resume-v2回归测试，共33项通过；
Pyflakes、Python 3.10语法及ZIP完整性检查通过。原依赖快照ZIP字节身份保留。
