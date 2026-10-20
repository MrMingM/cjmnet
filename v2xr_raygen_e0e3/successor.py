"""Create a new stable-compute run, importing verified data artifacts only; old run stays read-only."""
import argparse
import contextlib
import errno
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v2xr_raygen_e0e3.common import (HERE, ROOT, atomic_bytes, bound, identity, lock, read,
                                    safe, server, settings, sha, verify_sources, write)


def transfer(source, target, digest):
    if sha(source) != digest:
        raise ValueError('Reusable artifact changed: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != digest:
            raise ValueError('Successor artifact differs: ' + str(target))
        return
    try:
        # A symlink preserves even the parent's inode ctime/link count; readers still verify payload SHA256.
        os.symlink(source.resolve(), target)
    except OSError as error:
        if (error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP)
                and getattr(error, 'winerror', None) != 1314):
            raise
        # Source is immutable; copying uses a temporary destination, never moves the source.
        temporary = target.with_name(target.name + '.importing')
        shutil.copyfile(source, temporary)
        if sha(temporary) != digest:
            raise ValueError('Copied artifact hash differs')
        os.replace(temporary, target)


def import_caches(source, target, old_contract, new_contract, config, geometry):
    old_id = identity(dict(input=old_contract['inputs'], config=config, sources=old_contract['sources'], geometry=geometry))
    if read(source / 'labels/contract.json')['identity'] != old_id:
        raise ValueError('Parent ray-cache contract differs')
    new_id = identity(dict(input=new_contract['inputs'], config=config, sources=new_contract['sources'], geometry=geometry))
    bound(target / 'labels/contract.json', dict(identity=new_id, scheme='per sample raw finite rays + candidates'))
    manifest = read(source / 'preflight/train_manifest.json')
    uids = [[s['scene'], t, s['ego']] for s in manifest['scenes'] for t in s['timestamps']]
    inventory = []
    start = time.monotonic()
    for journal in sorted((source / 'labels/samples').glob('*.json')):
        if not journal.stem.isdecimal() or len(journal.stem) != 6:
            raise ValueError('Unexpected parent sample cache name')
        index, row = int(journal.stem), read(journal)
        if (index >= len(uids) or row['status'] != 'complete' or row['uid'] != uids[index]
                or row['cache_identity'] != old_id):
            raise ValueError('Parent sample cache identity differs')
        dest = target / journal.relative_to(source)
        transfer(journal.with_suffix('.npz'), dest.with_suffix('.npz'), row['sha256'])
        updated = dict(row, cache_identity=new_id,
                       reused_from=dict(path=str(journal), journal_sha256=sha(journal)))
        bound(dest, updated)
        inventory.append(dict(path=dest.relative_to(target).as_posix(), journal_sha256=sha(dest), payload_sha256=row['sha256']))
        if len(inventory) % 20 == 0:
            print(f'Imported ray-cache records {len(inventory)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    for folder in sorted((source / 'labels/windows').glob('*')):
        if not folder.is_dir() or not folder.name.isdecimal():
            continue
        epoch = int(folder.name)
        if not 1 <= epoch <= config['epochs']:
            raise ValueError('Parent window epoch differs')
        for journal in sorted(folder.glob('*.json')):
            if not journal.stem.isdecimal() or len(journal.stem) != 6:
                raise ValueError('Unexpected parent window cache name')
            index, row = int(journal.stem), read(journal)
            base = source / 'labels/samples' / journal.name
            expected = identity(dict(base=sha(base), config=config, uid=uids[index], epoch=epoch))
            if row['status'] != 'complete' or row['identity'] != expected:
                raise ValueError('Parent window identity differs')
            dest = target / journal.relative_to(source)
            transfer(journal.with_suffix('.npz'), dest.with_suffix('.npz'), row['sha256'])
            new_base = target / 'labels/samples' / journal.name
            updated = dict(row, identity=identity(dict(base=sha(new_base), config=config, uid=uids[index], epoch=epoch)),
                           reused_from=dict(path=str(journal), journal_sha256=sha(journal)))
            bound(dest, updated)
            inventory.append(dict(path=dest.relative_to(target).as_posix(), journal_sha256=sha(dest), payload_sha256=row['sha256']))
            if len(inventory) % 100 == 0:
                print(f'Imported cache records {len(inventory)} elapsed={time.monotonic()-start:.1f}s', flush=True)
        print('Imported verified window epoch ' + folder.name, flush=True)
    if not inventory:
        raise ValueError('No verified parent caches were found')
    return inventory


def initialize(source, target, gpu):
    from v2xr_raygen_e0e3.controller import make_request
    from v2xr_raygen_e0e3.resume_compat import effective_execution, verify_active_sources
    from v2xr_raygen_e0e3.runtime import input_stamps
    from v2xr_raygen_e0e3.stable_crop import POLICY, POLICY_ID
    own, config = verify_sources(), settings()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Parent and successor outputs must be separate sibling directories')
    original, old_protocol = read(source / 'contract.json'), read(source / 'protocol.json')
    old_request = read(source / 'request.json')
    parent = read(HERE / 'STABLE_PARENT.json')
    if (old_request['family'] != 'v2xr_raygen_e0e3' or original['config'] != config
            or original['sources'] != old_request['sources']
            or original['identity'] != identity({k: v for k, v in original.items() if k != 'identity'})):
        raise ValueError('Parent request/config/contract differs')
    verify_active_sources(source, parent['parent_sources'])
    protocol = read(HERE / 'protocol.json')
    if protocol['geometry'] != old_protocol['geometry']:
        raise ValueError('Geometry changed; prepared labels cannot be reused')
    for name in parent['cache_code_unchanged']:
        key = 'v2xr_raygen_e0e3/' + name
        if own['files'][key] != original['sources']['files'][key]:
            raise ValueError('Data/cache algorithm changed; reuse refused: ' + key)
    if own['dependencies'] != original['sources']['dependencies']:
        raise ValueError('Read-only dependencies changed')
    for key, value in old_request['paths'].items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError('Parent input environment differs: ' + key)
        os.environ[key] = value
    if os.environ.get('PYTHONHASHSEED') != str(config['seed']):
        raise ValueError('Set PYTHONHASHSEED before starting the successor initializer')
    execution = effective_execution(source)
    if gpu != execution['physical_gpu']:
        raise ValueError('Reuse requires the previously accepted physical GPU ' + execution['physical_gpu'])
    if (sys.version != original['environment']['python'] or
            {k: importlib.metadata.version(k) for k in original['environment']['versions']} != original['environment']['versions']):
        raise ValueError('Parent Python/package environment changed')
    status = read(source / 'stage_status.json')
    if any(status.get(stage, {}).get('status') != 'complete' for stage in ('preflight', 'smoke_prepare')):
        raise ValueError('Parent preflight and prepared smoke cache must be committed')
    if read(source / 'preflight/complete.json')['contract_identity'] != original['identity']:
        raise ValueError('Parent preflight completion differs')
    input_stamps(source)
    provenance = dict(source=str(source), source_contract_sha256=sha(source / 'contract.json'),
        source_request_sha256=sha(source / 'request.json'), parent_sources=parent['parent_sources'],
        execution=execution, compute_policy_identity=POLICY_ID)
    request = make_request(config, own)
    bound(target / 'request.json', request)
    bound(target / 'config.json', config)
    bound(target / 'protocol.json', protocol)
    contract = dict(original, sources=own,
        environment=dict(original['environment'], physical_gpu=gpu),
        compute_policy=POLICY, compute_policy_identity=POLICY_ID, data_reuse=provenance)
    contract.pop('identity')
    contract['identity'] = identity(contract)
    bound(target / 'contract.json', contract)
    completed = target / 'data_reuse.json'
    if completed.exists():
        previous = read(completed)
        if (previous['identity'] != identity({k: v for k, v in previous.items() if k != 'identity'})
                or previous['provenance'] != provenance or previous['target_contract_identity'] != contract['identity']
                or sha(target / 'data_reuse_inventory.json') != previous['inventory_sha256']):
            raise ValueError('Successor reuse record changed')
        print('Reuse verified successor data import; keep all newly completed stages', flush=True)
        return
    for name in ('train_manifest.json', 'evidence.json', 'input_stamps.json', 'smoke_indices.json'):
        dest = target / 'preflight' / name
        data = (source / 'preflight' / name).read_bytes()
        if dest.exists() and dest.read_bytes() != data:
            raise ValueError('Successor preflight artifact differs')
        atomic_bytes(dest, data)
    for journal in (source / 'preflight/content').rglob('*.json'):
        dest = target / journal.relative_to(source)
        if dest.exists() and dest.read_bytes() != journal.read_bytes():
            raise ValueError('Successor decoded-content record differs')
        atomic_bytes(dest, journal.read_bytes())
    inventory = import_caches(source, target, original, contract, config, protocol['geometry'])
    write(target / 'data_reuse_inventory.json', inventory)
    bound(target / 'source_snapshot.json', own)
    bound(target / 'compute_policy.json', dict(policy=POLICY, identity=POLICY_ID))
    bound(target / 'preflight/complete.json', dict(status='complete', contract_identity=contract['identity']))
    bound(target / 'data_check.json', dict(read(source / 'data_check.json'), inherited_from=str(source)))
    record = dict(status='complete', provenance=provenance, target_contract_identity=contract['identity'],
                  inventory_sha256=sha(target / 'data_reuse_inventory.json'), imported_cache_files=len(inventory),
                  training_states_imported=False, evaluation_results_imported=False,
                  note='Payload bytes reused; new metadata records new source identity; old artifacts never rewritten')
    record['identity'] = identity(record)
    bound(completed, record)
    stages = read(target / 'stage_status.json') if (target / 'stage_status.json').exists() else {}
    stages['preflight'] = dict(status='complete', inherited_data_only=True, source=str(source),
                               target_contract_identity=contract['identity'])
    write(target / 'stage_status.json', stages)
    print('Successor data import complete: ' + str(target) + '; training/evaluation stages start fresh', flush=True)


def main():
    server()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    source, target = safe(args.source).resolve(strict=True), safe(args.output).resolve()
    from v2xr_raygen_e0e3.paths import check
    check(target)
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Parent and successor outputs must be separate sibling directories')
    if Path('/data') not in source.parents or source == ROOT or ROOT in source.parents:
        raise ValueError('Parent must be a separate accepted run on /data')
    target.mkdir(parents=True, exist_ok=True)
    import fcntl
    with contextlib.ExitStack() as stack:
        for name, kind in (('.controller.lock', fcntl.LOCK_SH), ('.worker.lock', fcntl.LOCK_EX)):
            stream = stack.enter_context((source / name).open('rb'))
            fcntl.flock(stream, kind | fcntl.LOCK_NB)
        stack.enter_context(lock(target / '.controller.lock'))
        initialize(source, target, args.gpu)


if __name__ == '__main__':
    main()
