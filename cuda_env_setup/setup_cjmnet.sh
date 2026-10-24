#!/usr/bin/env bash
# Server only. Install a separate environment; never start training or read datasets.
set -Eeuo pipefail
SCRIPT_PATH=$(realpath -e -- "${BASH_SOURCE[0]}")

MODE=${MODE:-install}
BASE_DIR=${BASE_DIR:-/workspace/cjmnet_cuda_env}
PROJECT_ROOT=${PROJECT_ROOT:-}
INSTALL_SYSTEM_DEPS=${INSTALL_SYSTEM_DEPS:-1}
PYPI_INDEX=${PYPI_INDEX:-https://pypi.org/simple}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}
MAX_JOBS=${MAX_JOBS:-2}
ENV_MANAGER=${ENV_MANAGER:-auto}
REUSE_PREFIX=${REUSE_PREFIX:-0}
case "$MODE" in install|check|build-ops|check-gpu) ;; *) echo 'MODE must be install, check, build-ops or check-gpu'; exit 2 ;; esac
case "$ENV_MANAGER" in auto|conda|micromamba) ;; *) echo 'ENV_MANAGER must be auto, conda or micromamba'; exit 2 ;; esac
case "$REUSE_PREFIX" in 0|1) ;; *) echo 'REUSE_PREFIX must be 0 or 1'; exit 2 ;; esac
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Requires a Linux x86_64 server.'; exit 2; }
BASE_DIR=$(realpath -m -- "$BASE_DIR")
[[ "$BASE_DIR" != / && "$BASE_DIR" != /workspace ]] || { echo 'Use a dedicated BASE_DIR subdirectory.'; exit 2; }
mkdir -p "$BASE_DIR/logs" "$BASE_DIR/downloads" "$BASE_DIR/tmp"
LOG="$BASE_DIR/logs/${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; printf "FAILED: mode=%s line=%s exit=%s\nLog: %s\n" "$MODE" "$LINENO" "$rc" "$LOG"; exit "$rc"' ERR
export MAMBA_ROOT_PREFIX="$BASE_DIR/mamba"
ENV_PREFIX="$BASE_DIR/envs/cjmnet-cu124"
MICROMAMBA="$BASE_DIR/tools/bin/micromamba"
export TMPDIR="$BASE_DIR/tmp" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
printf 'Mode: %s\nEnvironment: %s\nLog: %s\n' "$MODE" "$ENV_PREFIX" "$LOG"

if [[ "$MODE" == install ]]; then
    if [[ -d "$ENV_PREFIX" && ! -f "$BASE_DIR/.cjmnet-managed-env" ]]; then
        echo 'Refusing to change an existing environment without this installer ownership marker.'
        exit 2
    fi
    available_kib=$(df -Pk "$BASE_DIR" | awk 'NR==2 {print $4}')
    if [[ ! -x "$ENV_PREFIX/bin/python" && "$available_kib" -lt 15728640 ]]; then
        echo 'First installation needs at least 15 GiB free for packages and download caches.'
        exit 2
    fi
    if [[ "$INSTALL_SYSTEM_DEPS" == 1 ]]; then
        [[ $(id -u) == 0 ]] || { echo 'Run as root, or preinstall system dependencies and set INSTALL_SYSTEM_DEPS=0.'; exit 2; }
        command -v apt-get >/dev/null || { echo 'System dependency installation requires Ubuntu/Debian apt-get.'; exit 2; }
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -o Acquire::Retries=3
        apt-get install -y --no-install-recommends ca-certificates curl bzip2 tar util-linux \
            build-essential gcc-11 g++-11 libgl1 libglib2.0-0 libgomp1 libgflags-dev libgoogle-glog-dev
    fi
fi
for tool in curl tar bzip2 flock sha256sum gcc-11 g++-11; do
    command -v "$tool" >/dev/null || { echo "Missing system dependency: $tool"; exit 2; }
done
exec 9>"$BASE_DIR/.setup.lock"
flock -n 9 || { echo 'Another setup/check/build is using this BASE_DIR.'; exit 2; }

fetch() {
    local url=$1 destination=$2
    curl --fail --location --http1.1 --retry 5 --retry-all-errors --retry-delay 3 --connect-timeout 30 \
        --output "$destination.partial" "$url"
    mv -- "$destination.partial" "$destination"
}

# Use the platform's existing Conda executable without changing its base/py312
# environment. A py312 prompt alone is not evidence that Conda is installed.
conda_executable=${CONDA_EXE:-}
if [[ ! -x "$conda_executable" ]]; then
    conda_executable=$(type -P conda || true)
fi
manager_kind=$ENV_MANAGER
manager_path=
if [[ "$ENV_MANAGER" == auto && -f "$BASE_DIR/manager.txt" ]]; then
    { IFS= read -r manager_kind; IFS= read -r manager_path; } < "$BASE_DIR/manager.txt"
elif [[ "$ENV_MANAGER" == conda || ( "$ENV_MANAGER" == auto && -x "$conda_executable" ) ]]; then
    manager_kind=conda
    manager_path=$conda_executable
else
    manager_kind=micromamba
    manager_path=$MICROMAMBA
fi
if [[ "$manager_kind" == conda ]]; then
    [[ -x "$manager_path" ]] || { echo 'Conda executable not found. Set CONDA_EXE to its full path.'; exit 2; }
elif [[ "$manager_kind" == micromamba ]]; then
    if [[ "$MODE" != install && ! -x "$MICROMAMBA" ]]; then
        echo 'Recorded Micromamba executable is missing. Run install or select ENV_MANAGER=conda.'
        exit 2
    fi
    manager_path=$MICROMAMBA
    if [[ ! -x "$MICROMAMBA" ]]; then
        mkdir -p "$BASE_DIR/tools"
        fetch https://micro.mamba.pm/api/micromamba/linux-64/latest "$BASE_DIR/downloads/micromamba.tar.bz2"
        tar -xjf "$BASE_DIR/downloads/micromamba.tar.bz2" -C "$BASE_DIR/tools" bin/micromamba
        sha256sum "$BASE_DIR/downloads/micromamba.tar.bz2" > "$BASE_DIR/downloads/micromamba.sha256"
    fi
else
    echo 'Invalid environment manager record.'
    exit 2
fi
printf 'Environment manager: %s (%s)\n' "$manager_kind" "$manager_path"
printf '%s\n%s\n' "$manager_kind" "$manager_path" > "$BASE_DIR/manager.txt"
run_manager() {
    if [[ "$manager_kind" == conda ]]; then
        CONDA_PKGS_DIRS="$BASE_DIR/conda-cache" "$manager_path" "$@"
    else
        "$manager_path" --no-rc "$@"
    fi
}

if [[ "$MODE" == install ]]; then
    printf 'cjmnet CUDA environment setup revision 4\n' > "$BASE_DIR/.cjmnet-managed-env"
    # Use explicit official channels. Only the new prefix is changed.
    # These are the development components needed by this project's ROI CUDA op.
    # PyTorch installs its own CUDA math libraries; no host driver is installed.
    if [[ "$REUSE_PREFIX" == 1 ]]; then
        [[ -f "$ENV_PREFIX/conda-meta/history" && -x "$ENV_PREFIX/bin/python" ]] || {
            echo 'REUSE_PREFIX=1 requires the already-created Python/CUDA environment.'; exit 2;
        }
        echo 'Reusing Python/CUDA environment; skipping the Conda package transaction.'
    else
        action=create
        [[ ! -f "$ENV_PREFIX/conda-meta/history" ]] || action=install
        run_manager "$action" --yes --prefix "$ENV_PREFIX" \
            --override-channels --channel https://conda.anaconda.org/nvidia/label/cuda-12.4.1 \
            --channel https://conda.anaconda.org/conda-forge \
            'python=3.10' pip tk 'cuda-nvcc=12.4.*' 'cuda-cudart=12.4.*' \
            'cuda-cudart-dev=12.4.*' 'cuda-driver-dev=12.4.*' 'cuda-cccl=12.4.*'
    fi
fi
[[ -x "$ENV_PREFIX/bin/python" ]] || { echo 'Environment does not exist. Run MODE=install first.'; exit 2; }
export ENV_PREFIX BASE_DIR MAX_JOBS
export PATH="$ENV_PREFIX/bin:$PATH"
export CC=$(command -v gcc-11) CXX=$(command -v g++-11)
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST=8.9
unset ROCM_HOME ROCM_PATH HIP_HOME HIP_PATH ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

# Present one complete CUDA development root to torch.utils.cpp_extension,
# supporting both NVIDIA conda's flat and targets/x86_64-linux layouts.
"$ENV_PREFIX/bin/python" - <<'PY'
import os
from pathlib import Path
prefix = Path(os.environ['ENV_PREFIX'])
root = Path(os.environ['BASE_DIR']) / 'cuda-12.4'
root.mkdir(exist_ok=True)
def find(candidates, marker):
    for p in candidates:
        if (p / marker).is_file():
            return p
    raise RuntimeError('CUDA development component missing: ' + marker)
links = dict(bin=find([prefix / 'bin'], 'nvcc'),
    include=find([prefix / 'include', prefix / 'targets/x86_64-linux/include'], 'cuda_runtime.h'),
    lib64=find([prefix / 'lib', prefix / 'lib64', prefix / 'targets/x86_64-linux/lib'], 'libcudart.so'))
if (prefix / 'nvvm').is_dir():
    links['nvvm'] = prefix / 'nvvm'
for name, target in links.items():
    link = root / name
    if link.is_symlink() and link.resolve() == target.resolve():
        continue
    if link.exists() or link.is_symlink():
        raise RuntimeError('Refusing to replace an existing CUDA path: ' + str(link))
    link.symlink_to(target, target_is_directory=True)
PY
export CUDA_HOME="$BASE_DIR/cuda-12.4" CUDA_PATH="$BASE_DIR/cuda-12.4"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$CUDA_HOME/include:$ENV_PREFIX/include:$ENV_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
"$CUDA_HOME/bin/nvcc" --version | tee "$BASE_DIR/nvcc-version.txt"
grep -q 'release 12.4' "$BASE_DIR/nvcc-version.txt" || { echo 'Expected CUDA compiler 12.4.'; exit 2; }
cat > "$BASE_DIR/tmp/toolchain_probe.cu" <<'CUDA'
#include <cuda_runtime.h>
__global__ void cjmnet_install_probe(float* x) { x[0] = 1.0f; }
CUDA
# Compile only: no GPU access, executable, dataset or model.
"$CUDA_HOME/bin/nvcc" -std=c++17 -ccbin "$CXX" -arch=sm_89 -c \
    "$BASE_DIR/tmp/toolchain_probe.cu" -o "$BASE_DIR/tmp/toolchain_probe.o"

if [[ "$MODE" == install ]]; then
    cat > "$BASE_DIR/requirements.txt" <<'REQ'
numpy==1.26.4
scipy==1.15.3
shapely==2.0.0
open3d==0.18.0
Cython==0.29.37
opencv-python-headless==4.10.0.84
matplotlib==3.9.2
pandas==2.2.3
scikit-learn==1.5.2
scikit-image==0.24.0
numba==0.60.0
Pillow==10.4.0
PyYAML==6.0.2
tensorboardX==2.6.2.2
protobuf==4.25.5
easydict==1.13
einops==0.8.0
tqdm==4.66.5
pyquaternion==0.9.9
pyparsing==3.1.4
timm==0.6.7
efficientnet-pytorch==0.7.1
ninja==1.11.1.3
pybind11==2.13.6
python-lzf==0.2.6
spconv==2.3.6
cumm==0.4.11
REQ
    {
        printf 'torch==2.5.1+cu124\ntorchvision==0.20.1+cu124\n'
        cat "$BASE_DIR/requirements.txt"
    } > "$BASE_DIR/constraints.txt"
    # Ignore platform pip configuration and keep all writes in the new prefix.
    export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_CACHE_DIR="$BASE_DIR/downloads/pip-cache"
    export PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=10
    unset PIP_EXTRA_INDEX_URL PIP_TARGET PIP_PREFIX PIP_USER
    python -m pip install --index-url "$PYPI_INDEX" 'pip==25.2' 'setuptools==75.1.0' 'wheel==0.44.0'
    # pip >=25.2 resumes interrupted downloads within the same invocation.
    export PIP_RESUME_RETRIES=20
    wheelhouse="$BASE_DIR/downloads/torch-wheels"
    mkdir -p "$wheelhouse"
    # Download just the exact torch/vision builds from the official CUDA index.
    # Its dependency links can point at pypi.nvidia.com, so resolve those separately.
    python -m pip download --no-deps --only-binary=:all: --dest "$wheelhouse" --index-url "$TORCH_INDEX" \
        --constraint "$BASE_DIR/constraints.txt" 'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124'
    # Save every completed dependency wheel for retries across script invocations.
    # All dependencies, including NVIDIA libraries, use the selected ordinary PyPI index.
    python -m pip download --only-binary=:all: --dest "$wheelhouse" --find-links "$wheelhouse" \
        --index-url "$PYPI_INDEX" --constraint "$BASE_DIR/constraints.txt" \
        'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124'
    python -m pip install --no-index --find-links "$wheelhouse" \
        --constraint "$BASE_DIR/constraints.txt" 'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124'
    # Prebuilt CPU voxelization is intentional: the actual loader calls Point2VoxelCPU3d.
    python -m pip install --index-url "$PYPI_INDEX" --only-binary=:all: \
        --constraint "$BASE_DIR/constraints.txt" 'spconv==2.3.6' 'cumm==0.4.11'
    python -m pip install --index-url "$PYPI_INDEX" \
        --constraint "$BASE_DIR/constraints.txt" --requirement "$BASE_DIR/requirements.txt"
    pypcd_commit=796e545df2ee5e7836df6fe2f274aae3428d1ac2
    pypcd_archive="$BASE_DIR/downloads/pypcd-$pypcd_commit.tar.gz"
    [[ -f "$pypcd_archive" ]] || fetch "https://codeload.github.com/klintan/pypcd/tar.gz/$pypcd_commit" "$pypcd_archive"
    sha256sum "$pypcd_archive" > "$BASE_DIR/downloads/pypcd.sha256"
    # Use the same reviewed pypcd fork as the successful server environment, without Git.
    python -m pip install --index-url "$PYPI_INDEX" --no-deps --no-build-isolation "$pypcd_archive"
fi

cat > "$BASE_DIR/activate.sh" <<'ACTIVATE'
# Source this file from bash; the platform's original environment remains installed.
_cjmnet_base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
{ IFS= read -r _cjmnet_kind; IFS= read -r _cjmnet_manager; } < "$_cjmnet_base/manager.txt"
if [[ "$_cjmnet_kind" == conda ]]; then
    _cjmnet_hook=$("$_cjmnet_manager" shell.bash hook) || return 1
    eval "$_cjmnet_hook" || return 1
    conda activate "$_cjmnet_base/envs/cjmnet-cu124" || return 1
elif [[ "$_cjmnet_kind" == micromamba ]]; then
    export MAMBA_ROOT_PREFIX="$_cjmnet_base/mamba"
    _cjmnet_hook=$("$_cjmnet_manager" shell hook --shell bash) || return 1
    eval "$_cjmnet_hook" || return 1
    micromamba activate "$_cjmnet_base/envs/cjmnet-cu124" || return 1
else
    echo 'Invalid environment manager record.'
    return 1
fi
export CUDA_HOME="$_cjmnet_base/cuda-12.4" CUDA_PATH="$_cjmnet_base/cuda-12.4"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$CUDA_HOME/include:$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST=8.9 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$_cjmnet_base/tmp"
unset PYTHONPATH PYTHONHOME ROCM_HOME ROCM_PATH HIP_HOME HIP_PATH ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
printf 'Active Python: %s\nCUDA compiler: %s\n' "$(command -v python)" "$CUDA_HOME/bin/nvcc"
unset _cjmnet_base _cjmnet_kind _cjmnet_manager _cjmnet_hook
ACTIVATE

# Environment check only: synthetic CPU data, no model, real point cloud or training.
cd "$BASE_DIR"
export CJMNET_SETUP_MODE="$MODE"
python - <<'PY'
import importlib
from importlib import metadata
import json, os, platform, sys
from pathlib import Path
import numpy as np
import torch, torchvision
expected = {'numpy':'1.26.4', 'scipy':'1.15.3', 'shapely':'2.0.0', 'open3d':'0.18.0',
    'spconv':'2.3.6', 'cumm':'0.4.11', 'Cython':'0.29.37'}
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__ == '2.5.1+cu124', torch.__version__
assert torchvision.__version__ == '0.20.1+cu124', torchvision.__version__
assert torch.version.cuda == '12.4' and torch.version.hip is None
for name, version in expected.items():
    assert metadata.version(name) == version, (name, metadata.version(name))
for name in ('cv2','scipy','shapely','open3d','yaml','tensorboardX','easydict','einops',
             'timm','efficientnet_pytorch','pyquaternion','numba','tkinter'):
    importlib.import_module(name)
from cumm import tensorview as tv
from spconv.utils import Point2VoxelCPU3d
v = Point2VoxelCPU3d(vsize_xyz=[.4,.4,4.], coors_range_xyz=[-2.,-2.,-3.,2.,2.,1.],
    num_point_features=4, max_num_voxels=16, max_num_points_per_voxel=32)
voxels, coords, counts = v.point_to_voxel(tv.from_numpy(np.array([[.1,.1,0.,.5]], dtype=np.float32)))
assert voxels.numpy().shape == (1,32,4) and int(counts.numpy()[0]) == 1
from pypcd import pypcd
sample = Path(os.environ['BASE_DIR']) / 'tmp/synthetic_setup.pcd'
# The pinned upstream ASCII reader uses np.loadtxt without ndmin=1. One
# structured row becomes a scalar, which its len(pc_data) check cannot handle.
# Use a normal multi-point fixture; do not patch the installed data reader.
sample.write_text('VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
    'WIDTH 2\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA ascii\n0 0 0 0\n1 2 3 0\n')
cloud = pypcd.PointCloud.from_path(str(sample))
assert cloud.points == 2 and cloud.pc_data.shape == (2,)
assert cloud.pc_data.dtype.names == ('x', 'y', 'z', 'rgb')
assert np.array_equal(cloud.pc_data['x'], np.array([0., 1.], dtype=np.float32))
assert np.array_equal(cloud.pc_data['z'], np.array([0., 3.], dtype=np.float32))
binary_sample = sample.with_name('synthetic_setup_binary.pcd')
header = sample.read_text().split('DATA ascii\n', 1)[0] + 'DATA binary\n'
binary_sample.write_bytes(header.encode('ascii') + cloud.pc_data.tobytes())
binary_cloud = pypcd.PointCloud.from_path(str(binary_sample))
assert binary_cloud.points == 2 and np.array_equal(binary_cloud.pc_data, cloud.pc_data)
available = torch.cuda.is_available()
report = dict(status='environment_checked', python=sys.version, executable=sys.executable,
    platform=platform.platform(), torch=torch.__version__, torchvision=torchvision.__version__,
    cuda=torch.version.cuda, hip=torch.version.hip, versions=expected,
    gpu_visible=available, gpu_runtime_tested=False, voxelization='Point2VoxelCPU3d passed',
    pypcd='two-point ASCII and binary fixtures passed; original single-point ASCII limitation retained',
    cuda_compiler_probe='sm_89 object compiled without GPU execution',
    project_ops_built=False, training_ready=False,
    note='No GPU is expected in no-card mode. Historical ROCm launchers still require a separate CUDA adaptation.')
if os.environ['CJMNET_SETUP_MODE'] == 'check-gpu':
    if not available:
        raise RuntimeError('No NVIDIA GPU visible. Run check-gpu after switching to GPU mode.')
    # Tiny installation check, not a scientific experiment or model training.
    x = torch.ones((8,8), device='cuda:0')
    assert torch.equal((x @ x).cpu(), torch.full((8,8), 8.))
    torch.cuda.synchronize()
    report.update(gpu_runtime_tested=True, gpu=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0))
destination = Path(os.environ['BASE_DIR']) / ('gpu_check.json' if report['gpu_runtime_tested'] else 'environment_check.json')
destination.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
python -m pip check
python -m pip freeze > "$BASE_DIR/pip-freeze.txt"
run_manager list --prefix "$ENV_PREFIX" --explicit > "$BASE_DIR/conda-explicit.txt"

if [[ "$MODE" == build-ops ]]; then
    [[ -n "$PROJECT_ROOT" ]] || { echo 'Set PROJECT_ROOT to the uploaded cjmnet directory.'; exit 2; }
    PROJECT_ROOT=$(realpath -e -- "$PROJECT_ROOT")
    export PROJECT_ROOT
    # No datasets or old server artifacts are needed for this build.
    python - <<'PY'
import os, sys
from pathlib import Path
project = Path(os.environ['PROJECT_ROOT'])
repro = project / 'v2xr_official_repro'
sys.path.insert(0, str(repro))
from common import install_runtime, verify_package
verify_package()
runtime = Path(os.environ['BASE_DIR']) / 'official-runtime-cu124'
if runtime == project or project in runtime.parents or runtime in project.parents:
    raise RuntimeError('The CUDA runtime must be outside the uploaded source repository.')
install_runtime(runtime)
PY
    runtime="$BASE_DIR/official-runtime-cu124"
    (
        cd "$runtime"
        python "$PROJECT_ROOT/v2xr_official_repro/build_preflight.py" --output "$BASE_DIR/build_environment.json"
        python opencood/utils/setup.py build_ext --inplace
        python "$PROJECT_ROOT/v2xr_official_repro/build_official_ops.py" --runtime "$runtime"
    )
    export CJMNET_CUDA_RUNTIME="$runtime"
    python - <<'PY'
import hashlib, importlib, json, os, sys
from pathlib import Path
runtime = Path(os.environ['CJMNET_CUDA_RUNTIME'])
sys.path.insert(0, str(runtime))
import torch
import opencood.data_utils.datasets
import opencood.models.point_pillar_intermediate
import opencood.loss.point_pillar_loss_opencood
import opencood.tools.train_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
import numpy as np
assert bbox_overlaps(np.array([[0,0,1,1]], np.float32), np.array([[0,0,1,1]], np.float32))[0,0] == 1.
assert int(points_in_boxes_cpu(torch.zeros((1,3)), torch.tensor([[0.,0.,0.,2.,2.,2.,0.]]))[0,0]) == 1
rows = {}
for name in ('opencood.utils.box_overlaps', 'opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_cuda'):
    file = Path(importlib.import_module(name).__file__).resolve()
    assert runtime in file.parents, str(file)
    rows[name] = dict(path=str(file), sha256=hashlib.sha256(file.read_bytes()).hexdigest())
report = dict(status='native_build_and_CPU_import_checks_passed', runtime=str(runtime), extensions=rows,
    gpu_ROI_tested=False, training_ready=False, note='Original ROCm contracts/launchers still need CUDA adaptation.')
(Path(os.environ['BASE_DIR']) / 'native_check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
fi

sha256sum "$SCRIPT_PATH" > "$BASE_DIR/setup-script.sha256"
printf '\nSUCCESS: %s\nActivate with: source %q/activate.sh\nLog: %s\n' "$MODE" "$BASE_DIR" "$LOG"
if [[ "$MODE" == install ]]; then
    echo 'Environment installation and CPU checks finished. GPU checks and model migration are separate steps.'
fi
