"""Local-safe source, AST, LF and historical snapshot verification. No data/Torch imports."""
import ast
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    here = Path(__file__).resolve().parent
    root = here.parent
    expected = json.loads((here / 'source_snapshot.json').read_text(encoding='utf-8'))['files']
    actual = {p.relative_to(here).as_posix(): digest(p) for p in here.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if actual != expected:
        raise ValueError('D-1D package snapshot mismatch')
    pins = json.loads((here / 'PINS.json').read_text(encoding='utf-8'))
    for name, sha in pins['dependencies'].items():
        if digest(root / name) != sha:
            raise ValueError('Frozen dependency changed: ' + name)
    for package in ('v2xr_official_repro', 'v2xr_protocol_audit'):
        folder = root / package
        snap = json.loads((folder / 'source_snapshot.json').read_text(encoding='utf-8'))['files']
        found = {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
        if found != snap:
            raise ValueError('Historical package changed: ' + package)
    count = 0
    for p in here.iterdir():
        if p.suffix in ('.py', '.sh', '.md', '.json'):
            if b'\r' in p.read_bytes():
                raise ValueError('CR/CRLF refused: ' + p.name)
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
            count += 1
    shell = (here / 'run_server.sh').read_text(encoding='utf-8')
    if 'pipefail' in shell or '[[' in shell:
        raise ValueError('Non-POSIX shell construct')
    upstream = json.loads((root / 'v2xr_official_repro/UPSTREAM.json').read_text(encoding='utf-8'))['files']
    tree = ast.parse((here / 'inventory.py').read_text(encoding='utf-8'))
    references = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'OFFICIAL_REFERENCE_FILES' for t in n.targets))
    for name in references:
        if name not in upstream:
            raise ValueError('Official reference case/spelling mismatch: ' + name)
        if digest(root / 'v2xr_official_repro/upstream' / name) != upstream[name]:
            raise ValueError('Official reference hash mismatch: ' + name)
    acceptance = json.loads((here / 'TRAIN_DATA_ACCEPTANCE.json').read_text(encoding='utf-8'))
    if (acceptance['status'] != 'user_accepted' or acceptance['accepted_cooperative_frames'] != 8083
            or acceptance['official_reference_cooperative_frames'] != 8084
            or acceptance['difference_from_reference'] != -1
            or set(acceptance['approved_artifact_sha256']) != {
                'inventory.json', 'frame_keys.jsonl', 'scene_counts.tsv', 'archive_counts.tsv'}):
        raise ValueError('Explicit train data acceptance schema/counts changed')
    for value in [acceptance['approved_inventory_identity'], *acceptance['approved_artifact_sha256'].values()]:
        if not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('Accepted inventory SHA256 is malformed')
    print(f'D-1D static checks passed: {count} Python ASTs; LF; own snapshot; both historical snapshots; dependency pins.')
    print('Official reference names checked case-sensitively against UPSTREAM.json, independent of host filesystem.')
    print('Train data approval: 8083 actual / 8084 published; inventory identity and four evidence hashes pinned.')
    print('No data, model, training or scientific experiment executed. POSIX sh -n runs on the server launcher.')


if __name__ == '__main__':
    run()
