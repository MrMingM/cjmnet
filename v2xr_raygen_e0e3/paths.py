"""Validate destination before even the launcher creates a log file."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2xr_raygen_e0e3.common import ROOT, read, safe


def check(output):
    p = safe(output).resolve()
    if Path('/data') not in p.parents:
        raise ValueError('OUTPUT_ROOT must be absolute on /data')
    sources = [ROOT, Path(os.environ.get('PROJECT_ROOT', str(ROOT))),
        Path(os.environ.get('TRAIN_ROOT', '/data/cjm/datasets/v2xr-official-train-v2')),
        Path(os.environ.get('VALIDATE_ROOT', '/data/cjm/datasets/v2xr-official-validate-v1/validate')).parent,
        Path(os.environ.get('PAIRED_VALIDATE_ROOT', '/data/cjm/datasets/v2xr-lidar-weather-validate-v1')),
        Path(os.environ.get('RUNTIME_ROOT', '/data/cjm/datasets/v2xr-official-runtime-v1')),
        Path(os.environ.get('MODEL_DIR', '/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1')),
        Path(os.environ.get('D1A_ROOT', '/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1')),
        Path(os.environ.get('GT_AUDIT_ROOT', '/data/cjm/datasets/logs/v2xr-official-gt-audit-v1')),
        Path(os.environ.get('D1C_ROOT', '/data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1')),
        Path('/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1'),
        Path('/data/cjm/datasets/V2X-R/train')]
    for source in sources:
        source = safe(source).resolve()
        if p == source or p in source.parents or source in p.parents:
            raise ValueError('Output overlaps a read-only input: ' + str(source))
    if (p / 'request.json').exists() and read(p / 'request.json').get('family') != 'v2xr_raygen_e0e3':
        raise ValueError('Existing request belongs to another experiment; no launcher log was written')
    if p.exists() and not (p / 'request.json').exists():
        if any(x.name not in {'.controller.lock', 'launcher.log', 'launcher.pid'} for x in p.iterdir()):
            raise ValueError('Nonempty unrelated output refused before launcher log creation')
    return p


if __name__ == '__main__':
    check(sys.argv[1])
