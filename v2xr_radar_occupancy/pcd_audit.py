"""Inventory archive members and inspect every source Radar PCD header."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile

from paths import DATA, OUT, write

ZIP = '/data/cjm/tools/7zip-26.04/bin/7zz'


def header(path):
    fields = {}
    with path.open('rb') as stream:
        for _ in range(64):
            row = stream.readline(8192).decode('ascii').strip().split()
            if not row or row[0].startswith('#'):
                continue
            fields[row[0].upper()] = row[1:]
            if row[0].upper() == 'DATA':
                if not all(k in fields for k in ('FIELDS','TYPE','SIZE','POINTS')):
                    raise ValueError(f'Incomplete PCD header: {path}')
                return fields
    raise ValueError(f'Missing DATA line: {path}')


def audit_archive(archive):
    listing = subprocess.run([ZIP,'l','-slt',str(archive)],check=True,
                             capture_output=True,text=True).stdout
    members = [r.removeprefix('Path = ') for r in listing.splitlines()
               if r.startswith('Path = ') and r.lower().endswith('_radar.pcd')]
    if len(set(members)) != len(members):
        raise ValueError(f'Duplicate archive path: {archive}')
    counts = Counter()
    if members:
        # Unique scratch directory per archive. 7-Zip validates decompressed member CRC.
        with tempfile.TemporaryDirectory(prefix='radar-',dir=OUT) as temp:
            subprocess.run([ZIP,'x','-y',f'-o{temp}',str(archive),'*_radar.pcd','-r'],
                           check=True,stdout=subprocess.DEVNULL)
            actual = {p.relative_to(temp).as_posix():p for p in Path(temp).rglob('*.pcd')}
            if set(members)!=actual.keys():
                raise ValueError(f'Archive Radar list differs from extraction: {archive}')
            for member in members:
                h=header(actual[member])
                signature = json.dumps({k:h.get(k) for k in ('FIELDS','TYPE','SIZE','COUNT','DATA')},sort_keys=True)
                counts[signature]+=1
    return dict(archive=str(archive), members=len(members), layouts=counts)


def run():
    root = DATA/'V2X-R'
    scenes = {}
    archives = sorted(p for split in ('train','validate','test') for p in (root/split).glob('*/*.7z'))
    if not archives:
        raise ValueError('No source archives')
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,row in enumerate(pool.map(audit_archive,archives),1):
            arc=Path(row['archive'])
            scene=str(arc.parent.relative_to(root))
            scenes.setdefault(scene,[]).append(row)
            if i%10==0 or i==len(archives):
                print(f'ARCHIVES {i}/{len(archives)}',flush=True)
    split_totals = {}
    for scene,rows in scenes.items():
        counts = split_totals.setdefault(scene.split('/')[0],Counter())
        for row in rows:
            counts.update(row['layouts'])
    write(OUT/'PCD_FIELDS.json',dict(archives=len(archives),scenes=scenes,split_layouts=split_totals))

