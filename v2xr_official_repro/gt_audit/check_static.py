"""Local-safe AST/LF/hash audit only; no scientific code or data is executed."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(1, str(PACKAGE))


def main():
    from package_check import verify_package
    verify_package()
    spec = importlib.util.spec_from_file_location('d1a_static', PACKAGE / 'check_static.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    old = json.loads((HERE / 'd1a_source_snapshot.json').read_text(encoding='utf-8'))['files']
    for name, digest in old.items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == digest, name
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for path in HERE.iterdir():
        if path.suffix in ('.py', '.sh', '.md', '.json'):
            assert b'\r' not in path.read_bytes(), f'Expected LF: {path}'
    shell = (HERE / 'run_server.sh').read_text()
    assert 'pipefail' not in shell and '[[' not in shell
    print('D-1B AST/LF checks OK; every pre-existing D-1A package file is byte-identical. '
          'No Torch/OpenCOOD/real-data execution.')


if __name__ == '__main__':
    main()
