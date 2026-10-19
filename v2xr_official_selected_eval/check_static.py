"""Local-safe AST/LF/hash checks only; never imports runtime/evaluation code."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
expected = json.loads((root / 'source_snapshot.json').read_text(encoding='utf-8'))['files']
actual = {}
for path in root.rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts or path.name == 'source_snapshot.json':
        continue
    data = path.read_bytes()
    if b'\r' in data:
        raise ValueError('Expected LF: ' + str(path))
    if path.suffix == '.py':
        ast.parse(data.decode('utf-8'), filename=str(path))
    actual[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
if actual != expected:
    raise ValueError('Snapshot mismatch')
print('AST/LF/source hashes passed; no Torch/OpenCOOD/data execution.')
