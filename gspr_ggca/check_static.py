"""Standard-library-only delivery/source/shell checks; never load Torch."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    for file in HERE.glob('*.py'):
        ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(file))
    shell = (HERE / 'run_all.sh').read_bytes()
    if b'\r' in shell or shell.startswith(b'\xef\xbb\xbf') or any(x in shell for x in (b'pipefail', b'[[', b'#!/bin/bash')):
        raise ValueError('Shell must use POSIX sh and LF')
    expected = json.loads((HERE / 'source_snapshot.json').read_text(encoding='utf-8'))['files']
    actual = {p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if expected != actual:
        raise ValueError('GGCA source snapshot differs')
    for name, digest in json.loads((HERE / 'PINS.json').read_text(encoding='utf-8'))['files'].items():
        if hashlib.sha256((HERE.parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned dependency changed: ' + name)
    print('PASS: GGCA syntax, POSIX/LF, source hashes and frozen dependencies; no model/data execution')


if __name__ == '__main__':
    main()
