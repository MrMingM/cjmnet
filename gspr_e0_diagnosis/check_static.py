"""Static and synthetic checks. Safe locally; never imports research libraries."""
import ast
import json
import sys
import unittest
from e0_common import HERE, sources


def main():
    for path in HERE.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.suffix in ('.py', '.sh', '.json', '.md'):
            raw = path.read_bytes()
            if b'\r' in raw or raw.startswith(b'\xef\xbb\xbf'):
                raise ValueError('Expected UTF-8 without BOM and LF: '+str(path))
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
    sources()
    import tests_local
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests_local))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if any(n == 'torch' or n == 'opencood' or n.startswith('opencood.') for n in sys.modules):
        raise AssertionError('Local checks imported a research library')
    print('STATIC_AND_SYNTHETIC_CHECKS_PASSED; no Torch/OpenCOOD/data/GPU execution')


if __name__ == '__main__':
    main()
