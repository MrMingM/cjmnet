"""Parse/compile without executing research modules; validate frozen source manifests."""
import ast
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uecp_gspr_v2xr.common import HERE, ROOT, read, sources


def run():
    count = 0
    for p in HERE.rglob('*.py'):
        text = p.read_text(encoding='utf-8'); ast.parse(text, filename=str(p)); compile(text, str(p), 'exec')
        count += 1
    for p in HERE.rglob('*.json'):
        json.loads(p.read_text(encoding='utf-8'))
    for p in HERE.rglob('*.sh'):
        text = p.read_bytes()
        assert b'\r' not in text and text.endswith(b'\n'), p
        assert all(x not in text for x in (b'pipefail', b'[[', b'BASH_SOURCE'))
    c = read(HERE / 'settings.json')
    assert c['effective_batch'] == 8 and c['epochs'] == 20 and c['seed'] == 20261017
    assert len(c['variants']) == 9 and not c['amp'] and c['gspr']['frozen']
    assert c['range'] == [-140.8, -40, -3, 140.8, 40, 1] and c['voxel'] == [.4, .4, 4]
    assert set(c['weather']) == {'clean', 'fog', 'snow'}
    # Confirm the single upstream detach that the controlled E2 AST edit relies on.
    upstream = ROOT / 'related_code/UECP-main/opencood/models/utils/pyramid_fusion_modulev1.py'
    tree = ast.parse(upstream.read_text(encoding='utf-8'))
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'detach'
               and isinstance(n.func.value, ast.Name) and n.func.value.id == 'u_map' for n in ast.walk(tree)) == 1
    snapshot = sources()
    print('PASS compiled', count, 'Python sources; JSON, LF, protocol, detach AST and', len(snapshot['dependencies']), 'read-only dependency hashes')


if __name__ == '__main__':
    run()
