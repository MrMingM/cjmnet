"""Local engineering checks only; all real scientific execution is server-only."""
import ast
import json
from pathlib import Path


def main():
    for path in Path(__file__).resolve().parent.iterdir():
        if not path.is_file():
            continue
        raw = path.read_bytes()
        raw.decode('utf-8')
        if b'\r' in raw:
            raise ValueError('LF required: ' + str(path))
        if path.suffix == '.py':
            tree = ast.parse(raw, filename=str(path), feature_version=(3, 10))
            compile(tree, str(path), 'exec')
        if path.suffix == '.json':
            json.loads(raw)
    from tests_local import main as synthetic
    synthetic()
    print('PASS: Python 3.10 grammar, compilation, LF/UTF-8/JSON and frozen source identities')


if __name__ == '__main__':
    main()
