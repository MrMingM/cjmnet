"""No research execution: grammar, LF, frozen hashes and synthetic unit tests."""
import ast
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e01_common import sources


def main():
    for path in HERE.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        raw = path.read_bytes()
        raw.decode('utf-8')
        if b'\r' in raw:
            raise ValueError('LF required: '+str(path))
        if path.suffix == '.py':
            ast.parse(raw, filename=str(path), feature_version=(3, 10))
        if path.suffix == '.json':
            json.loads(raw)
    sources()
    import tests_local
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests_local))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if 'torch' in sys.modules or 'opencood' in sys.modules:
        raise RuntimeError('Local checks must not import research runtime')
    print('PASS: Python 3.10 grammar, LF/UTF-8/JSON, source hashes and synthetic tests; no scientific experiment run')


if __name__ == '__main__':
    main()
