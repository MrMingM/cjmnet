"""One compressed transaction per paired frame."""
import json
import os
from functools import lru_cache

import numpy as np
from .input_audit import OUT, identity, read_json, sha256


def path(index, smoke=False):
    return OUT/('smoke_cache_metadata' if smoke else 'cache')/f'{index:06d}.npz'


@lru_cache(maxsize=1)
def inherited(contract):
    current = read_json(OUT/'CONTRACT.json')
    if identity(current)!=contract:
        return {}
    records={}
    for field, name in (('inherited_cache_identity','CACHE_INHERITANCE.json'),
                        ('native_metadata_cache_identity','CACHE_BEFORE_NATIVE_METADATA.json')):
        if field in current:
            path=OUT/name
            if sha256(path)!=current[field]:
                raise ValueError('Inherited cache manifest changed')
            row=read_json(path)
            records[row['contract']]=row['files']
    return records


def save(file, metadata, arrays):
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata, allow_nan=False)), **arrays)
    os.replace(temp, file)


def load(file, contract):
    with np.load(file, allow_pickle=False) as archive:
        meta = json.loads(archive['metadata'].item())
        if meta['contract'] != contract:
            legacy = inherited(contract).get(meta['contract'], {})
            if file.parent.name!='cache' or legacy.get(file.name)!=sha256(file):
                raise ValueError(f'Cache contract mismatch: {file}')
        return meta, {key:archive[key] for key in archive.files if key != 'metadata'}


def frames(smoke=False):
    contract = identity(read_json(OUT/'CONTRACT.json'))
    for index in range(3 if smoke else 829):
        meta, arrays = load(path(index, smoke), contract)
        if meta['index'] != index:
            raise ValueError('Cache frame order differs')
        yield meta, arrays
