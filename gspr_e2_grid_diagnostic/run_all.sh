#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
GPU=${GPU:-6}
MODE=${MODE:-all}
OUTPUT=${OUTPUT:-/home/cjm/datasets/logs/e2-grid-diagnostic-v1}
case "$MODE" in smoke|all) ;; *) echo 'MODE must be smoke or all' >&2; exit 2 ;; esac
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES PYTHONPATH PYTHONHOME
ROCR_VISIBLE_DEVICES=$GPU
BACKEND=rocm
PYTHONHASHSEED=20261016
PYTHONDONTWRITEBYTECODE=1
PYTHONNOUSERSITE=1
PYTHONUNBUFFERED=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
TMPDIR=${TMPDIR:-/home/cjm/datasets/tmp/e2-grid-diagnostic}
export ROCR_VISIBLE_DEVICES BACKEND PYTHONHASHSEED PYTHONDONTWRITEBYTECODE PYTHONNOUSERSITE PYTHONUNBUFFERED OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS TMPDIR
mkdir -p "$TMPDIR"
# Refuse to share the selected device with another job; never terminate it.
"$PYTHON" -c 'import subprocess,re,sys; r=subprocess.run(["rocm-smi","--showmeminfo","vram","-d",sys.argv[1]],capture_output=True,text=True,check=True); a=re.findall(r"Total Used Memory \(MiB\):\s*(\d+)",r.stdout); assert len(a)==1 and int(a[0])<=16,"GPU is occupied; choose a free card"' "$GPU"
"$PYTHON" -B "$SCRIPT_DIR/run.py" --smoke --output "${OUTPUT}_smoke"
if [ "$MODE" = all ]; then
 exec "$PYTHON" -B "$SCRIPT_DIR/run.py" --output "$OUTPUT"
fi
