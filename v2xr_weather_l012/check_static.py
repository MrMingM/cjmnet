"""No scientific runtime imports: AST/LF/hash/protocol/call-chain checks only."""
import ast
import json
from pathlib import Path
import subprocess
import sys


def run():
    here = Path(__file__).resolve().parent
    from auditlib import sources
    sources()
    files = [p for p in here.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for path in files:
        if path.suffix in ('.py', '.sh', '.json', '.md') and b'\r' in path.read_bytes():
            raise ValueError('CR/CRLF refused: ' + str(path))
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
    # Real official locations, not README example names.
    path = here.parent / 'v2xr_official_repro/upstream/opencood/data_utils/datasets/opv2v/basedataset.py'
    text = path.read_text(encoding='utf-8')
    if 'aug = random.randint(0, 6)' not in text or 'if  aug <= 2 :' not in text:
        raise ValueError('Reviewed official per-CAV sampling source differs')
    training = (here / 'training.py').read_text(encoding='utf-8')
    if 'scheduler.step(epoch)' not in training or 'train_utils.setup_lr_schedular' not in training:
        raise ValueError('Official explicit-epoch scheduler contract missing')
    if sys.platform == 'linux':
        subprocess.run(['sh', '-n', str(here / 'run_all.sh')], check=True)
    print(f'STATIC PASS: {len(files)} files; AST/JSON/LF; own/dependency snapshots; official sampler/scheduler. No model experiment executed.')


if __name__ == '__main__':
    run()
