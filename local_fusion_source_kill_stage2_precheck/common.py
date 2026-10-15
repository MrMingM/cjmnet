"""Read-only inputs, atomic reports and aggregation without a research runtime."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
WEATHERS = ('clean', 'fog', 'rain', 'snow')
ADVERSE = WEATHERS[1:]
SPLITS = ('train', 'validation', 'train+validation')
KEEP = 'KEEP_SHARED'
HISTORICAL = tuple(f'{name}.{suffix}' for name in (
    'K0_GROUP_MARGIN', 'K1_0_ORACLE_DECOMPOSITION', 'STAGE1_FINAL') for suffix in ('json', 'md'))


def safe_path(value):
    path = Path(value).resolve()
    if any(part.lower() == 'test' for part in path.parts):
        raise ValueError('Formal test access prohibited: ' + str(path))
    return path


def read_json(path):
    return json.loads(safe_path(path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with safe_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_text(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def scene_summary(counts):
    counts = {str(k): int(v) for k, v in sorted(counts.items()) if v > 0}
    total = sum(counts.values())
    return dict(scene_count=len(counts), per_scene=counts, n=total,
                max_scene_share=max(counts.values()) / total if total else None)


def merge_scenes(rows, field):
    counts = Counter()
    for row in rows:
        counts.update(row[field]['per_scene'])
    return scene_summary(counts)


def expand_cells(base, merge):
    """Base keys split/weather[/task/stratum]; pooled IDs retain split identity."""
    result = dict(base)
    tails = sorted({tuple(key.split('/')[2:]) for key in base})
    for tail in tails:
        suffix = '/' + '/'.join(tail) if tail else ''
        for weather in WEATHERS:
            rows = [base[split + '/' + weather + suffix] for split in ('train', 'validation')]
            result['train+validation/' + weather + suffix] = merge(rows)
        for split in SPLITS:
            rows = [result[split + '/' + weather + suffix] for weather in ADVERSE]
            pooled = merge(rows)
            result[split + '/adverse pooled' + suffix] = pooled
            # Macro counts are weather means, not additional samples.
            macro = {'summary_type': 'weather_equal_weight',
                     'not_a_sample_count': True, 'weather_count': len(ADVERSE)}
            for key in pooled:
                values = [row.get(key) for row in rows]
                if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                    macro[key] = mean(values)
                elif all(v is None for v in values):
                    macro[key] = None
            for field, value in pooled.items():
                if isinstance(value, dict) and 'per_scene' in value:
                    macro[field] = {'pooled_coverage_for_context': value,
                                    'mean_scene_count': mean(r[field]['scene_count'] for r in rows),
                                    'mean_max_scene_share': mean(r[field]['max_scene_share'] for r in rows)
                                    if all(r[field]['max_scene_share'] is not None for r in rows) else None}
            result[split + '/adverse equal-weight' + suffix] = macro
    return result


def source_hashes():
    paths = [p for p in PACKAGE.iterdir() if p.suffix in ('.py', '.sh', '.md')]
    paths += [ROOT / 'local_fusion_action_utility_audit/counterfactual.py',
              ROOT / 'local_fusion_action_utility_audit/common.py',
              ROOT / 'local_fusion_source_kill_stage1/k0_group_margin.py',
              ROOT / 'local_fusion_source_kill_stage1/common.py',
              ROOT / 'local_fusion_source_kill_stage1/oracle_adapter.py',
              ROOT / 'local_fusion_task_source_oracle/oracle.py']
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def verify_snapshot():
    if read_json(PACKAGE / 'source_snapshot.json')['source_hashes'] != source_hashes():
        raise ValueError('Precheck source differs from snapshot; synchronize the entire directory')


def output_path(value, server=False):
    path = safe_path(value)
    if server and Path('/data/cjm/datasets/logs').resolve() not in path.parents:
        raise ValueError('Full audit output must be below /data/cjm/datasets/logs')
    if not server and path != (ROOT / 'logs/k').resolve() and (ROOT / '.tmp').resolve() not in path.parents:
        raise ValueError('Local summary output must be logs/k or a fixture folder under .tmp')
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp():
    return datetime.now(timezone.utc).isoformat()
