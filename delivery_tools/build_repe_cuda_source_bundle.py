"""Build a self-contained REPE CUDA source archive; never include data or model weights."""
import ast
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent.parent
PACKAGES=('gspr_repe','v2xr_weather_l012','v2xr_gspr_clean','v2xr_train_attfuse','v2xr_protocol_audit','v2xr_official_repro')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output=ROOT/'delivery'/('repe_cuda_'+stamp)
    folder=output/'cjmnet'
    folder.mkdir(parents=True,exist_ok=False)
    selected={}
    def add(name,expected=None):
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name:
            raise ValueError('Unsafe source name')
        src=ROOT/name
        if src.is_symlink() or ROOT not in src.resolve(strict=True).parents:
            raise ValueError('Source escapes workspace')
        sha=digest(src)
        if expected is not None and sha!=expected:
            raise ValueError('Frozen source differs: '+name)
        selected[name]=sha
    for package in PACKAGES:
        add(package+'/source_snapshot.json')
        for name,sha in json.loads((ROOT/package/'source_snapshot.json').read_text(encoding='utf-8'))['files'].items():
            add(package+'/'+name,sha)
    for package,field in (('gspr_repe','files'),('v2xr_weather_l012','files'),('v2xr_gspr_clean','files'),('v2xr_train_attfuse','dependencies')):
        for name,sha in json.loads((ROOT/package/'PINS.json').read_text(encoding='utf-8'))[field].items():
            add(name,sha)
    for name,sha in json.loads((ROOT/'v2xr_official_repro/frozen_baseline.json').read_text(encoding='utf-8')).items():
        add(name,sha)
    for name in ('AI_CONTEXT.md','AI_REMOTE.md','cuda_env_setup/setup_cjmnet.sh','cuda_env_setup/README.md'):
        add(name)
    for path in sorted((ROOT/'v2xr_pointcloud_only').iterdir()):
        if path.is_file() and path.suffix in ('.py','.md','.sh'):
            add(path.relative_to(ROOT).as_posix())
    for name in selected:
        dst=folder/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dst)
    shutil.copyfile(ROOT/'delivery_tools/verify_source_bundle.py',folder/'verify_source_bundle.py')
    (folder/'DELIVERY_README.md').write_text('''# GSPR REPE CUDA 源码交付

这是GSPR后端A/B/C及Clean/Fog混合训练的独立CUDA入口，不是旧Clean-only GSPR源码包。
上传zip到/workspace后，确认旧目录没有运行任务，先保留旧源码，再解压回指定位置：

```sh
test ! -e /workspace/gspr_source/cjmnet_before_repe_20261024 && mv /workspace/gspr_source/cjmnet /workspace/gspr_source/cjmnet_before_repe_20261024
/workspace/cjmnet_cuda_env/envs/cjmnet-cu124/bin/python -m zipfile -e /workspace/cjmnet_repe_cuda_flat.zip /workspace/gspr_source/cjmnet
/workspace/cjmnet_cuda_env/envs/cjmnet-cu124/bin/python /workspace/gspr_source/cjmnet/verify_source_bundle.py
```

旧源码保留在/workspace/gspr_source/cjmnet_before_repe_20261024；备份目录已存在时先另选名称。压缩包直接包含gspr_repe等目录，没有cjmnet顶层，因此解压目标是/workspace/gspr_source/cjmnet。环境和算子已安装，不需要重新安装或编译。
后台启动命令见gspr_repe/CUDA_REMOTE.md。默认BACKEND=cuda，使用已有CUDA环境、三个已传输数据目录。
首次预检重新核对提取数据哈希和三天气官方GT，旧.7z/历史权重/D-1A/B/C输出不作为新CUDA运行输入。
源码中的旧包是冻结依赖，不能改其snapshot或把历史ROCm结果冒充CUDA结果。
本交付仅通过本地静态和源码检查；GPU、真实数据、显存及训练均待远端验收。
''',encoding='utf-8',newline='\n')
    files={f.relative_to(folder).as_posix():dict(size=f.stat().st_size,sha256=digest(f)) for f in folder.rglob('*') if f.is_file()}
    dump(folder/'SOURCE_BUNDLE_MANIFEST.json',dict(schema=1,snapshot_packages=PACKAGES,files=files,
        scope='GSPR REPE CUDA backend; matched Clean/Fog B0-B4; no data/weights/history'))
    checks=[]
    scripts=[folder/'verify_source_bundle.py']+[folder/package/'check_static.py' for package in PACKAGES]+[folder/'gspr_repe/test_migration_static.py',folder/'v2xr_pointcloud_only/test_static.py']
    for script in scripts:
        result=subprocess.run([sys.executable,'-B',str(script)],cwd=folder,capture_output=True,text=True,encoding='utf-8')
        checks.append(dict(script=str(script.relative_to(folder)),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        dump(output/'STATIC_CHECKS.json',checks)
        if result.returncode:
            raise RuntimeError(repr(checks[-1]))
    for file in folder.rglob('*.py'):
        ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file))
    archive=output/'cjmnet_repe_cuda_flat.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for file in sorted(folder.rglob('*')):
            if file.is_file():
                z.write(file,file.relative_to(folder).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(files)|{'SOURCE_BUNDLE_MANIFEST.json'}
        assert not any(name.startswith('cjmnet/') for name in z.namelist())
        for name,row in files.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==row['sha256']
    for name,sha in selected.items():
        assert digest(ROOT/name)==sha
    sha=digest(archive)
    (output/'SHA256SUMS.txt').write_text(sha+'  '+archive.name+'\n',encoding='utf-8')
    summary=dict(status='complete',archive=str(archive),sha256=sha,bytes=archive.stat().st_size,
        static_checks=len(checks),files=len(files)+1,archive_layout='flat; extract directly into project root',scientific_experiments_executed=False,cuda_server_acceptance='pending')
    dump(output/'SUMMARY.json',summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
