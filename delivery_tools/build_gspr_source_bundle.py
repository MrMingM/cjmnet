"""Package the frozen V2X-R GSPR dependency chain without scientific imports."""
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from build_v2xr_source_bundle import digest, write_json

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ('v2xr_gspr_clean', 'v2xr_train_attfuse',
            'v2xr_protocol_audit', 'v2xr_official_repro')


def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    destination = ROOT / 'delivery' / ('gspr_source_' + stamp)
    bundle = destination / 'cjmnet'
    bundle.mkdir(parents=True, exist_ok=False)
    selected, reasons = {}, {}

    def add(name, reason, expected=None):
        source = ROOT / name
        if ROOT not in source.resolve(strict=True).parents or source.is_symlink():
            raise ValueError('Source outside workspace: ' + name)
        sha = digest(source)
        if expected is not None and sha != expected:
            raise ValueError('Frozen source differs: ' + name)
        selected[name] = sha
        reasons.setdefault(name, []).append(reason)

    for package in PACKAGES:
        manifest = json.loads((ROOT / package / 'source_snapshot.json').read_text(encoding='utf-8'))
        add(package + '/source_snapshot.json', 'original source identity')
        for name, sha in manifest['files'].items():
            add(package + '/' + name, 'complete dependency package: ' + package, sha)
    for package, field in (('v2xr_gspr_clean', 'files'), ('v2xr_train_attfuse', 'dependencies')):
        pins = json.loads((ROOT / package / 'PINS.json').read_text(encoding='utf-8'))[field]
        for name, sha in pins.items():
            add(name, 'pinned dependency: ' + package, sha)
    frozen = json.loads((ROOT / 'v2xr_official_repro/frozen_baseline.json').read_text(encoding='utf-8'))
    for name, sha in frozen.items():
        add(name, 'required historical source audit; not a runnable experiment selection', sha)
    for name in ('AI_CONTEXT.md', 'cuda_env_setup/setup_cjmnet.sh', 'cuda_env_setup/README.md',
                 'v2xr_cuda_delivery/check_data.py', 'v2xr_cuda_delivery/DATA_TRANSFER.md'):
        add(name, 'context/environment/data migration support')
    for name in sorted(selected):
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copyfile(ROOT / 'delivery_tools/verify_source_bundle.py', bundle / 'verify_source_bundle.py')
    (bundle / 'DELIVERY_README.md').write_text('''# V2X-R GSPR-AttFuse 源码包

此包保留 GSPR 模型、官方 V2X-R 骨干、共享训练/评测引擎和冻结校验依赖。
仅制作交付副本；原模型、训练配置及原始 snapshot/PINS 未修改。没有运行科研实验。

## 上传和解压

上传 cjmnet_gspr_source.zip 到 /workspace，然后解压到新的目录，避免覆盖现有 cjmnet：

```bash
python -m zipfile -e /workspace/cjmnet_gspr_source.zip /workspace/gspr_source
source /workspace/cjmnet_cuda_env/activate.sh
python /workspace/gspr_source/cjmnet/verify_source_bundle.py
```

若 /workspace/gspr_source 已存在，请另选空目录。校验应输出 source_integrity_passed。
这是独立源码位置，后续 CUDA 启动入口须显式指向这个目录；现有已编译 runtime 不会因解压而更新。

## 保留范围

- v2xr_gspr_clean：当前 V2X-R Clean GSPR 接入、训练、选模和评测。
- attfuse_gspr/reliability.py 等 PINS 指定文件：冻结网络及配置来源。
- v2xr_train_attfuse：GSPR 实际继承的共享训练引擎、数据验收、评测。
- v2xr_protocol_audit、v2xr_official_repro：官方源码、算子及原协议检查。
- cuda_env_setup、v2xr_cuda_delivery：环境准备及只读数据文件检查。

原官方源码归档及 frozen_baseline.json 指定的部分历史文件属于现有哈希检查依赖，必须保留。
它们不表示本包重新启用旧实验。各文件保留原因见 SOURCE_BUNDLE_AUDIT.json。
不含 v2xr_weather_l012 的天气混合实验入口、独立 AttFuse 选中模型评测包、数据、权重、日志、
论文、临时文件、Git 元数据或根目录旧 opencood。官方 opencood 保留在官方源码快照内。

## 运行边界

本包按当前 v2xr_gspr_clean 的 Clean-only 配方交付，含 GSPR 模型及必要共享依赖。
它是源码包，完整 CUDA 训练入口尚未交付。旧 run_server.sh 仍要求海光环境、旧数据/证据
与基线记录，不能直接在 NVIDIA 服务器启动。不要修改 snapshot 或历史记录绕过检查。
训练需要的外部数据/历史验收记录不在源码包内。下一步完成数据检查、独立 CUDA 入口适配，
并在租用服务器试跑。环境已安装且算子已编译，无需因上传源码重复安装环境。
本地仅做源码哈希、语法和压缩包检查，不导入模型，不读取数据，不运行实验。
''', encoding='utf-8', newline='\n')
    write_json(bundle / 'SOURCE_BUNDLE_AUDIT.json', dict(
        scope='V2X-R Clean GSPR-AttFuse and required shared dependencies',
        original_files_modified=False, original_files_deleted=False,
        cuda_migration_complete=False, retained_file_reasons=reasons))
    files = {p.relative_to(bundle).as_posix(): dict(size=p.stat().st_size, sha256=digest(p))
             for p in sorted(bundle.rglob('*')) if p.is_file()}
    write_json(bundle / 'SOURCE_BUNDLE_MANIFEST.json', dict(
        schema=1, snapshot_packages=list(PACKAGES), files=files,
        scope='V2X-R Clean GSPR-AttFuse and required dependencies'))
    checks = []
    for script in [bundle / 'verify_source_bundle.py'] + [bundle / p / 'check_static.py' for p in PACKAGES]:
        result = subprocess.run([sys.executable, '-B', str(script)], cwd=bundle,
                                capture_output=True, text=True)
        checks.append(dict(script=str(script.relative_to(bundle)), returncode=result.returncode,
                           stdout=result.stdout, stderr=result.stderr))
        write_json(destination / 'STATIC_CHECKS.json', checks)
        if result.returncode:
            raise RuntimeError(str(checks[-1]))
    for p in bundle.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    archive = destination / 'cjmnet_gspr_source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                output.write(p, 'cjmnet/' + p.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == {'cjmnet/' + n for n in files} | {'cjmnet/SOURCE_BUNDLE_MANIFEST.json'}
        for name, row in files.items():
            assert hashlib.sha256(z.read('cjmnet/' + name)).hexdigest() == row['sha256']
    for name, sha in selected.items():
        assert digest(ROOT / name) == sha and digest(bundle / name) == sha
    sha = digest(archive)
    (destination / 'SHA256SUMS.txt').write_text(sha + '  ' + archive.name + '\n', encoding='utf-8')
    summary = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha,
                   files=len(files) + 1, static_checks=len(checks),
                   original_sources_unchanged=True, training_executed=False,
                   cuda_migration_complete=False)
    write_json(destination / 'DELIVERY_SUMMARY.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
