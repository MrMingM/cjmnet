"""Only standard-library source/shell checks; no local scientific execution."""
import ast
import hashlib
import json
from pathlib import Path
import re

def main():
    here=Path(__file__).resolve().parent
    assert (here/'train_acceptance.py').is_file(), 'Explicit inherited train-acceptance adapter required'
    audit=ast.parse((here/'auditlib.py').read_text(encoding='utf-8'))
    assert 'evaluation_path' in {n.name for n in audit.body if isinstance(n,ast.FunctionDef)}
    result_tree=ast.parse((here/'results.py').read_text(encoding='utf-8'))
    assert any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='write_evaluation' for t in n.targets) for n in result_tree.body)
    for p in here.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    for p in here.rglob('*.sh'):
        b=p.read_bytes()
        if b'\r' in b:raise ValueError('Shell requires LF')
        if re.search(r'pipefail|\[\[|\]\]|\bfunction\b',b.decode()):raise ValueError('Non-POSIX shell construct')
    settings=json.loads((here/'settings.json').read_text())
    assert settings['S']['alpha_cap']==.5 and settings['extra_loss']==0
    assert not settings['C']['constrained'] and not settings['C']['state_gate']
    # Ensure exactly one A1 message insertion can be identified before deploying.
    old=ast.parse((here.parent/'gspr_ggca/modules.py').read_text(encoding='utf-8'))
    cls=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='GGCA')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='chunk')
    assert ast.unparse(fn.body[-1].value)=='(result, diag)'
    assert len([n for n in fn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id=='messages' and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.where'])==1
    for n,h in json.loads((here/'PINS.json').read_text())['files'].items():
        assert hashlib.sha256((here.parent/n).read_bytes()).hexdigest()==h,n
    actual={p.relative_to(here).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in here.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='source_snapshot.json'}
    assert actual==json.loads((here/'source_snapshot.json').read_text())['files']
    print('PASS static AST, message insertion, POSIX/LF and complete source/dependency hashes; no experiment executed')

if __name__=='__main__':main()
