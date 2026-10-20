"""Reject overlap with inputs/historical outputs before creating even a launcher log."""
import os
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uecp_gspr_v2xr.common import ROOT, read, safe, settings


def check(output):
    p = safe(output).resolve()
    logs = Path('/data/cjm/datasets/logs').resolve()
    if logs not in p.parents:
        raise ValueError('OUTPUT_ROOT must be a new separate child under /data/cjm/datasets/logs')
    protected = [ROOT, Path(settings()['gspr']['checkpoint']).parent,
        Path('/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1'),
        Path('/data/cjm/datasets/logs/v2xr-gspr-clean-train-v1')]
    defaults = dict(PROJECT_ROOT=str(ROOT), TRAIN_ROOT='/data/cjm/datasets/v2xr-official-train-v2',
        VALIDATE_ROOT='/data/cjm/datasets/v2xr-official-validate-v1',
        PAIRED_VALIDATE_ROOT='/data/cjm/datasets/v2xr-lidar-weather-validate-v1',
        RUNTIME_ROOT='/data/cjm/datasets/v2xr-official-runtime-v1',
        MODEL_DIR='/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1',
        D1A_ROOT='/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1',
        GT_AUDIT_ROOT='/data/cjm/datasets/logs/v2xr-official-gt-audit-v1',
        D1C_ROOT='/data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1')
    protected += [Path(os.environ.get(k, v)) for k, v in defaults.items()]
    if os.environ.get('QUALITY_LABEL_MANIFEST'):
        protected.append(Path(os.environ['QUALITY_LABEL_MANIFEST']).parent)
    for source in protected:
        s = safe(source).resolve()
        if p == s or s in p.parents or p in s.parents:
            raise ValueError('Output overlaps a read-only source: '+str(s))
    if (p / 'request.json').exists():
        if read(p / 'request.json').get('family') != 'uecp_gspr_v2xr':
            raise ValueError('Output belongs to a different experiment')
    elif p.exists() and any(x.name not in ('.controller.lock', 'launcher.log', 'launcher.pid') for x in p.iterdir()):
        raise ValueError('Unowned nonempty output refused')
    return p


if __name__ == '__main__':
    check(sys.argv[1])
