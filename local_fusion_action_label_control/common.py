"""Output isolation and content identity; Git is optional metadata only."""
from pathlib import Path
import argparse
import re
from local_fusion_action_utility_audit.common import (
    ROOT, KEEP, WEATHERS, Manifest, atomic_json, atomic_text, atomic_torch,
    digest, load_cache, object_hash, read_json,
)

GROUPS = ('OriginalLabel', 'DetectionOutcomeLabel')
FEATURES = 'with_competition_features'
HERE = Path(__file__).resolve().parent


def isolated_paths(source, run, create=False):
    source, run = Path(source).resolve(), Path(run).resolve()
    if source == run or source in run.parents or run in source.parents:
        raise ValueError('SOURCE_RUN and RUN must be separate, non-nested directories')
    logs = Path('/data/cjm/datasets/logs').resolve()
    if logs not in run.parents or ROOT.resolve() in run.parents:
        raise ValueError('RUN must be below /data/cjm/datasets/logs and outside the repository')
    if not source.is_dir():
        raise ValueError('SOURCE_RUN does not exist: ' + str(source))
    if create:
        run.mkdir(parents=True, exist_ok=True)
    return source, run


def safe_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if root not in path.parents:
        raise ValueError('Unsafe artifact path: ' + str(name))
    return path


def optional_commit():
    # No subprocess, branch assertion or dependency on .git. Never part of identity.
    try:
        git = ROOT / '.git'
        if git.is_file():
            git = (ROOT / git.read_text().strip().split('gitdir: ', 1)[1]).resolve()
        head = (git / 'HEAD').read_text().strip()
        if head.startswith('ref: '):
            head = (git / head[5:]).read_text().strip()
        return head if re.fullmatch(r'[a-f0-9]{40,64}', head) else None
    except (OSError, ValueError, IndexError):
        return None


def settings(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    expected = {'schema': 1, 'groups': list(GROUPS), 'features': FEATURES,
                'normalization': 'reuse_source_probe_fit', 'training': 'inherit_source_protocol',
                'calibration_benefit': 'detection_outcome_strictly_better_than_keep'}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Only the preregistered two-label control is supported')
    if value.get('ap_absolute_tolerance') != 1e-6:
        raise ValueError('AP reproduction tolerance must be 1e-6')
    for key in ('large_quality_tiebreak_fraction', 'almost_all_keep_modify_fraction'):
        if not 0 <= value[key] <= 1:
            raise ValueError('Invalid descriptive flag: ' + key)
    return value


def code_hashes():
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(HERE.iterdir())
            if p.suffix in ('.py', '.yaml', '.sh', '.md')}


def protocol(run):
    value = read_json(Path(run) / 'protocol.json')
    isolated_paths(value['source_run'], run)
    if code_hashes() != value['control_source_hashes']:
        raise ValueError('Control source changed; resume requires the same source hashes')
    if Manifest(run).value['identity'] != value['identity']:
        raise ValueError('Control manifest identity differs')
    return value


def cli(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--run', required=True)
    return parser


def report(run, name, value, markdown):
    atomic_json(Path(run) / (name + '.json'), value)
    atomic_text(Path(run) / (name + '.md'), markdown)
