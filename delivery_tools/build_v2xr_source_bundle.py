"""Create a clean delivery copy; never remove or edit original scientific code."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile


PACKAGES = ('v2xr_weather_l012', 'v2xr_gspr_clean', 'v2xr_train_attfuse',
            'v2xr_protocol_audit', 'v2xr_official_repro', 'v2xr_official_selected_eval')
TOOLS = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def build(root, destination):
    root = root.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('Use a new output directory: ' + str(destination))
    if root not in destination.parents:
        raise ValueError('Delivery output must be within this workspace.')
    selected, reasons = {}, {}

    def add(name, reason, expected=None):
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Unsafe relative source path: ' + name)
        source = root.joinpath(*relative.parts)
        if root not in source.resolve(strict=True).parents or not source.is_file() or source.is_symlink():
            raise ValueError('Expected an ordinary workspace source file: ' + name)
        actual = digest(source)
        if expected is not None and actual != expected:
            raise ValueError('Frozen source changed: ' + name)
        selected[name] = dict(size=source.stat().st_size, sha256=actual)
        reasons.setdefault(name, []).append(reason)

    for package in PACKAGES:
        source_manifest = root / package / 'source_snapshot.json'
        entries = json.loads(source_manifest.read_text(encoding='utf-8'))['files']
        add(package + '/source_snapshot.json', 'original package identity')
        for name, sha in entries.items():
            add(package + '/' + name, 'complete original package: ' + package, sha)
    # Source identity is checked in the existing official/protocol static tools.
    # These legacy files are required by that audit, despite not being entry points.
    frozen = json.loads((root / 'v2xr_official_repro/frozen_baseline.json').read_text(encoding='utf-8'))
    for name, sha in frozen.items():
        add(name, 'required frozen historical audit dependency', sha)
    for package, field in (('v2xr_weather_l012', 'files'), ('v2xr_gspr_clean', 'files'),
                           ('v2xr_train_attfuse', 'dependencies')):
        pins = json.loads((root / package / 'PINS.json').read_text(encoding='utf-8'))[field]
        for name, sha in pins.items():
            add(name, 'pinned source dependency: ' + package, sha)
    add('AI_CONTEXT.md', 'research context and execution restrictions')
    for path in sorted((root / 'cuda_env_setup').iterdir()):
        if path.is_file() and path.suffix in ('.sh', '.md'):
            add(path.relative_to(root).as_posix(), 'current CUDA environment preparation')

    original_metrics = Counter()
    omitted_metrics = Counter()
    omitted_counts = Counter()
    # Do not walk Git metadata, private agent settings, or this delivery tree.
    excluded_metadata = {'.git', '.agents', '.codex', '.idea', '.pytest_cache', 'delivery'}
    for child in root.iterdir():
        if child.name in excluded_metadata or child.resolve() == destination:
            continue
        files = child.rglob('*') if child.is_dir() else [child]
        for path in files:
            if path.is_file() and not path.is_symlink():
                name = path.relative_to(root).as_posix()
                size = path.stat().st_size
                original_metrics[child.name] += size
                if name not in selected:
                    omitted_metrics[child.name] += size
                    omitted_counts[child.name] += 1

    destination.mkdir(parents=True)
    bundle = destination / 'cjmnet'
    bundle.mkdir()
    for name, row in sorted(selected.items()):
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        if digest(target) != row['sha256']:
            raise ValueError('Copy changed source bytes: ' + name)
    shutil.copyfile(TOOLS / 'verify_source_bundle.py', bundle / 'verify_source_bundle.py')
    readme = '''# cjmnet：V2X-R 源码精简交付包

本包按已确认范围制作：V2X-R AttFuse、GSPR-AttFuse、v2xr_weather_l012。
原工作目录保留，清理只发生在交付副本中。没有修改训练、模型、配置或原有哈希清单，没有运行实验。

## 解压及核对

压缩包只有一个顶层目录 cjmnet，可解压到 /workspace：

```bash
unzip /workspace/cjmnet_v2xr_source.zip -d /workspace
source /workspace/cjmnet_cuda_env/activate.sh
python /workspace/cjmnet/verify_source_bundle.py
```

请将上传的压缩包改名为示例中的 cjmnet_v2xr_source.zip。若 /workspace/cjmnet 已存在，先解压到一个新的空目录，避免混入旧文件。
应输出 source_integrity_passed。本检查只读源码及哈希，无需 Git，不读取数据或运行模型。

## 保留内容

- v2xr_weather_l012：当前天气混合训练及比较。
- v2xr_gspr_clean：官方 V2X-R 模型中的 GSPR 接入。
- v2xr_train_attfuse：官方 AttFuse 训练、数据验收和评测桥接。
- v2xr_protocol_audit、v2xr_official_repro：冻结官方数据、算子和评测实现。
- v2xr_official_selected_eval：已选模型的官方协议评测入口。
- cuda_env_setup：已经用于当前服务器的独立 CUDA 环境脚本。
- AI_CONTEXT.md：研究约束与历史证据。

旧目录中的少量文件按 v2xr_official_repro/frozen_baseline.json 原样保留，仅作为冻结审计依赖；本包不授权恢复那些旧实验。
官方 upstream_archive.zip、官方源码中暂不使用的模型/配置、部分测试及说明文件也在原校验清单内，必须一起保留。
当前 V2X-R 模型加载 v2xr_official_repro/upstream/opencood 的独立副本，因此未打包根目录旧 opencood 及其中训练权重。

## 从交付包移除的内容

旧权重、训练日志与结果、大量临时文件、历史交付压缩包、论文、无关参考工程、编辑器/缓存/Git 元数据，以及当前依赖链之外的实验。
逐目录排除数量和大小见 SOURCE_BUNDLE_AUDIT.json；每个保留文件的大小和 SHA256 见 SOURCE_BUNDLE_MANIFEST.json。

## 当前运行边界

这是源码完整性通过的精简包，CUDA 训练迁移尚未完成。原入口仍有 HIP/ROCm 强制检查、旧服务器绝对路径，以及历史环境/算子身份约束。
不能仅因已经安装 CUDA PyTorch 就直接用旧 run_all.sh 开训；不要手改原 snapshot 或证据 JSON 绕过检查。
本包不包含数据集、训练权重、原始归档、数据验收产物、历史 D-1A/B/C 运行目录及基线权重。
这些内容属于外部运行输入；在新服务器训练前，仍需迁移相应数据/证据并完成独立 CUDA 入口适配。

源码上传后可继续无卡编译所需两个算子（不训练）：

```bash
nohup env MODE=build-ops PROJECT_ROOT=/workspace/cjmnet \\
  bash /workspace/cjmnet/cuda_env_setup/setup_cjmnet.sh \\
  > /workspace/cjmnet_build_ops.log 2>&1 &
echo "PID=$!"
```

实际模型运行、GPU 验证、显存及速度检查必须在远程服务器进行。本地仅完成文件选择、源码校验和压缩包验证。
'''
    (bundle / 'DELIVERY_README.md').write_text(readme, encoding='utf-8', newline='\n')
    audit = dict(scope=list(PACKAGES[:3]), original_scientific_files_modified=False,
        original_files_deleted=False, model_experiments_executed=False,
        original_bytes_excluding_metadata_and_delivery=sum(original_metrics.values()),
        selected_original_files=len(selected), selected_original_bytes=sum(row['size'] for row in selected.values()),
        historical_audit_dependency_count=len(frozen),
        original_required_archive='v2xr_official_repro/upstream_archive.zip',
        omitted_from_delivery={name: dict(files=omitted_counts[name], bytes=omitted_metrics[name]) for name in sorted(omitted_metrics)},
        excluded_metadata_directories=sorted(excluded_metadata),
        retained_file_reasons=reasons, cuda_training_ready=False)
    write_json(bundle / 'SOURCE_BUNDLE_AUDIT.json', audit)
    files = {p.relative_to(bundle).as_posix(): dict(size=p.stat().st_size, sha256=digest(p))
             for p in sorted(bundle.rglob('*')) if p.is_file()}
    manifest = dict(schema=1, snapshot_packages=list(PACKAGES), files=files,
                    scope='V2X-R AttFuse, GSPR-AttFuse, v2xr_weather_l012; original bytes retained')
    write_json(bundle / 'SOURCE_BUNDLE_MANIFEST.json', manifest)
    checks = [('bundle_integrity', bundle / 'verify_source_bundle.py')]
    checks += [(package, bundle / package / 'check_static.py') for package in PACKAGES]
    results = []
    for name, script in checks:
        result = subprocess.run([sys.executable, '-B', str(script)], cwd=bundle, capture_output=True, text=True)
        row = dict(check=name, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        results.append(row)
        if result.returncode:
            write_json(destination / 'STATIC_CHECKS.json', results)
            raise RuntimeError('Static check failed: ' + name + '\n' + result.stdout + result.stderr)
    write_json(destination / 'STATIC_CHECKS.json', results)
    # Verify the copied Python files parse, without importing scientific packages.
    for path in bundle.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    archive = destination / 'cjmnet_v2xr_source.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(bundle.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                output.write(path, 'cjmnet/' + path.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None:
            raise ValueError('ZIP CRC validation failed')
        names = check.namelist()
        expected_names = {'cjmnet/' + name for name in files} | {'cjmnet/SOURCE_BUNDLE_MANIFEST.json'}
        if set(names) != expected_names or len(names) != len(expected_names):
            raise ValueError('ZIP member inventory differs')
        for name, row in files.items():
            if hashlib.sha256(check.read('cjmnet/' + name)).hexdigest() != row['sha256']:
                raise ValueError('ZIP source hash differs: ' + name)
    # A final comparison confirms packaging did not alter any selected original.
    for name, row in selected.items():
        if digest(root / name) != row['sha256']:
            raise ValueError('Original source changed during packaging: ' + name)
    archive_sha = digest(archive)
    (destination / 'SHA256SUMS.txt').write_text(archive_sha + '  ' + archive.name + '\n', encoding='utf-8')
    summary = dict(status='complete', archive=str(archive), archive_bytes=archive.stat().st_size,
        archive_sha256=archive_sha, source_file_count=len(files)+1,
        selected_original_bytes=audit['selected_original_bytes'],
        original_bytes_excluding_metadata_and_delivery=audit['original_bytes_excluding_metadata_and_delivery'],
        static_checks=[r['check'] for r in results], original_files_deleted=False,
        cuda_migration_complete=False, model_experiments_executed=False)
    write_json(destination / 'DELIVERY_SUMMARY.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=TOOLS.parent)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    build(args.root, args.destination or args.root / 'delivery' / ('v2xr_source_' + stamp))
