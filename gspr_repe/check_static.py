"""Local static validation only. No Torch or dataset execution."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    for file in root.glob('*.py'):
        ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file))
    script=(root/'run_all.sh').read_bytes()
    assert b'\r' not in script and not script.startswith(b'\xef\xbb\xbf')
    for token in (b'pipefail',b'[[',b'#!/bin/bash'):
        assert token not in script
    expected=json.loads((root/'source_snapshot.json').read_text())['files']
    actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='source_snapshot.json'}
    assert expected==actual,'Source hash mismatch; synchronize complete package'
    assert json.loads((root/'settings.json').read_text())['variants']==['B0','B1','B2','B3','B4']
    print('PASS: Python syntax, shell LF/POSIX forbidden constructs, configuration and source hashes; no experiment executed')


if __name__=='__main__':
    main()
