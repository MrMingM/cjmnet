"""Import-free local checks; safe on Windows without Torch/OpenCOOD."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
files = sorted(ROOT.glob('*.py'))
if not files:
    raise SystemExit('No Python sources')
for file in files:
    data = file.read_bytes()
    if b'\r\n' in data:
        raise SystemExit(f'CRLF is forbidden: {file}')
    ast.parse(data.decode('utf-8'), filename=str(file))
for file in ROOT.glob('*.sh'):
    content = file.read_bytes()
    if b'\r' in content or b'pipefail' in content or b'[[' in content:
        raise SystemExit(f'Non-POSIX shell or CRLF detected: {file}')
    result = subprocess.run(['sh', '-n', str(file)], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f'sh -n failed: {file}\n{result.stderr}')
model = (ROOT / 'lr1_model.py').read_text(encoding='utf-8')
run = (ROOT / 'run.py').read_text(encoding='utf-8')
for token in ('PointPillarIntermediateLRF', 'GeometryFirstPillarReliability',
              'from gspr_ggca.modules import GGCA', 'self.lidar_pillar_vfe.register_forward_pre_hook',
              "self.ggca(", 'LR1Loss', "self.native(prediction, target)"):
    if token not in model:
        raise SystemExit(f'Expected actual frozen-method hook missing: {token}')
for token in ("VARIANTS = ('LRG', 'LR1')", 'native_engine.smoke', 'native_engine.train',
              'native_engine.finalize', 'verify_dual_gt', "'MATCHED_GT.json'",
              "'Clean4/7 Fog3/7", 'seed(20261017)'):
    if token not in run:
        raise SystemExit(f'Expected experiment stage missing: {token}')
print('LR1_STATIC_CHECKS_PASSED')
