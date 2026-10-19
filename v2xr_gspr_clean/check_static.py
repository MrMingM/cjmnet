"""Local-safe AST/LF/JSON/dependency-hash checks. No experiment imports."""
import ast
import hashlib
import json
from pathlib import Path
import re

here = Path(__file__).resolve().parent
root = here.parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected = json.loads((here / 'source_snapshot.json').read_text(encoding='utf-8'))['files']
actual = {}
for path in here.rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts or path.name == 'source_snapshot.json':
        continue
    data = path.read_bytes()
    if b'\r' in data:
        raise ValueError('Expected LF: ' + str(path))
    if path.suffix == '.py':
        ast.parse(data.decode('utf-8'), filename=str(path))
    if path.suffix == '.json':
        json.loads(data)
    actual[path.relative_to(here).as_posix()] = sha(path)
if actual != expected:
    raise ValueError('New package snapshot differs')
pins = json.loads((here / 'PINS.json').read_text(encoding='utf-8'))
for name, digest in pins['files'].items():
    if sha(root / name) != digest:
        raise ValueError('Frozen dependency changed: ' + name)
official_yaml = root / 'v2xr_official_repro/upstream/opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'
match = re.search(r'^loss:\s*\n\s+core_method:\s*([^\s#]+)', official_yaml.read_text(encoding='utf-8'), re.MULTILINE)
if not match:
    raise ValueError('Official YAML detection loss was not found')
official_method = match.group(1).strip('\'"')
tree = ast.parse((here / 'integration.py').read_text(encoding='utf-8'))
declared = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'OFFICIAL_LOSS_METHOD' for t in node.targets))
loss = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GsprOfficialLoss')
base = loss.bases[0].id
imports = [(node.module, alias.name) for node in tree.body if isinstance(node, ast.ImportFrom)
           for alias in node.names if alias.name == base]
if (declared != official_method or imports != [('opencood.loss.' + official_method, base)]
        or base.lower() != official_method.replace('_', '').lower()):
    raise ValueError('GSPR loss guard/base class differs from the actual official YAML')
settings = json.loads((here / 'settings.json').read_text(encoding='utf-8'))
if settings['loss']['base_detection_loss'] != imports[0][0] + '.' + base:
    raise ValueError('Registered detection loss metadata differs')
print('Official YAML -> loss guard -> imported base class -> settings checked without runtime imports.')
print('AST/LF/JSON and new/frozen source SHA256 passed. No Torch/OpenCOOD/real-data execution.')
