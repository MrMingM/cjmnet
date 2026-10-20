"""Full-state atomic epoch commits. No weights-only resume path."""
import os
from pathlib import Path
import random
from auditlib import complete, sha256


def seed(value=20261017):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


def save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    with temp.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)


ACCUMULATORS = ('total_loss', 'reg_loss', 'conf_loss', 'diff_loss', 'loss_cnt', 'loss_dict')


def state(model, optimizer, scheduler, criterion, epoch, contract):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), criterion=criterion.state_dict(),
        accumulators={k: getattr(criterion, k) for k in ACCUMULATORS if hasattr(criterion, k)},
        epoch=epoch, contract=contract, rng=rng())


def restore(path, model, optimizer, scheduler, criterion, contract, epoch):
    import torch
    row = torch.load(path, map_location='cpu', weights_only=False)
    if row['contract'] != contract or row['epoch'] != epoch:
        raise ValueError('Resume identity/epoch differs')
    model.load_state_dict(row['model'], strict=True)
    optimizer.load_state_dict(row['optimizer'])
    scheduler.load_state_dict(row['scheduler'])
    criterion.load_state_dict(row['criterion'], strict=True)
    for key, value in row['accumulators'].items():
        setattr(criterion, key, value)
    restore_rng(row['rng'])


def committed(root, count, contract):
    rows = []
    gap = False
    for epoch in range(1, count + 1):
        path = root / 'epochs' / f'{epoch:02d}' / 'complete.json'
        if not path.exists():
            gap = True
            continue
        if gap:
            raise ValueError('Noncontiguous epoch commit')
        row = complete(path)
        if row['epoch'] != epoch or row['contract'] != contract:
            raise ValueError('Committed epoch identity changed')
        for key in ('checkpoint', 'resume_state', 'clean_summary', 'trace'):
            if sha256(row[key]) != row[key + '_sha256']:
                raise ValueError('Committed artifact changed: ' + key)
        rows.append(row)
    return rows
