"""Small reproducibility helpers shared by training and evaluation scripts."""

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone


def sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repo_root):
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def scenario_names(dataset_root):
    if not dataset_root or not os.path.isdir(dataset_root):
        return []
    return sorted([
        name for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name))
    ])


def checkpoint_path(model_dir, epoch):
    if epoch and int(epoch) > 0:
        return os.path.join(model_dir, 'net_epoch%d.pth' % int(epoch))
    latest = os.path.join(model_dir, 'latest.pth')
    if os.path.isfile(latest):
        return latest
    candidates = []
    for name in os.listdir(model_dir) if os.path.isdir(model_dir) else []:
        match = re.fullmatch(r'net_epoch(\d+)\.pth', name)
        if match:
            candidates.append((int(match.group(1)), name))
    if not candidates:
        return None
    return os.path.join(model_dir, max(candidates)[1])


def write_manifest(path, payload):
    payload = dict(payload)
    payload.setdefault(
        'created_utc', datetime.now(timezone.utc).isoformat())
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    return path
