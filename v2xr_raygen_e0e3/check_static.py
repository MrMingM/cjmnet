"""Local AST/config/LF/source integrity check; never imports research runtime."""
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2xr_raygen_e0e3.common import HERE, ROOT, read, verify_sources


def main():
    count = 0
    for p in HERE.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        if p.suffix in ('.py', '.json', '.sh', '.md'):
            if b'\r' in p.read_bytes():
                raise ValueError('Expected LF: ' + str(p))
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
            count += 1
        if p.suffix == '.json':
            json.loads(p.read_text(encoding='utf-8'))
    if b'\r' in (ROOT / 'launch_v2xr_e0e3.sh').read_bytes():
        raise ValueError('Launcher must use LF')
    c = read(HERE / 'config.json')
    assert c['groups'] == ['E0', 'E1', 'E2', 'E3'] and c['effective_batch'] == 8
    assert c['window']['shape_xyz'] == [32, 32, 8] and c['diffusion']['steps'] == 1000
    assert c['lambda_gen'] == .1 and c['epochs'] == 20 and not c['amp']
    sources = verify_sources()
    from v2xr_raygen_e0e3.data import adapt_native
    # Count the exact native branches expected by the adapter without importing native code.
    p = ROOT / 'v2xr_official_repro/upstream/opencood/data_utils/datasets/opv2v/intermediate_fusion_dataset.py'
    tree = ast.parse(p.read_text(encoding='utf-8'))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'get_item_single_car')
    assert len([n for n in ast.walk(method) if isinstance(n, ast.While)]) == 3
    assert any(isinstance(n, ast.If) and ast.unparse(n.test) == 'self.training and len(object_stack) == 0'
               for n in ast.walk(tree))
    assert 'torch' not in sys.modules and 'opencood' not in sys.modules
    print(f'AST/LF/config/native-adapter/source checks passed: {count} Python files, '
          f'{len(sources["dependencies"])} pinned read-only files')


if __name__ == '__main__':
    main()
