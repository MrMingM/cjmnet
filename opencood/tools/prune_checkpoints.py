#!/usr/bin/env python3
"""Safely preview and prune obsolete OpenCOOD checkpoint files.

The default behaviour is a dry run.  Only ``*.pth`` regular files under the
given logs root are considered, and explicitly protected run directories are
never touched.
"""

import argparse
import os
from pathlib import Path


CONFIRM_TEXT = "DELETE_UNPROTECTED_PTH"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview or remove obsolete OpenCOOD .pth files.")
    parser.add_argument(
        "--logs_root", default="opencood/logs",
        help="Root containing OpenCOOD experiment directories.")
    parser.add_argument(
        "--keep_dir", action="append", default=[],
        help="Run directory to protect. Repeat this option as needed. "
             "A bare directory name is resolved below --logs_root.")
    parser.add_argument(
        "--mode", choices=["intermediate", "unprotected"],
        default="intermediate",
        help="'intermediate' removes old net_step*.pth files while keeping "
             "the newest steps in every unprotected run; 'unprotected' "
             "removes every .pth outside --keep_dir.")
    parser.add_argument(
        "--keep_latest_steps", type=int, default=1,
        help="With --mode intermediate, retain this many newest net_step "
             "checkpoints per unprotected run.")
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually delete the listed files. Without it this is a dry run.")
    parser.add_argument(
        "--confirm", default="",
        help="Required confirmation text for --mode unprotected --execute.")
    return parser.parse_args()


def is_within(path, root):
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def resolve_keep_dir(raw, logs_root):
    if raw is None or not str(raw).strip():
        raise ValueError(
            "Received an empty --keep_dir value. If a shell variable was "
            "used, define and verify it in the current terminal first.")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        cwd_candidate = (Path.cwd() / candidate).resolve()
        root_candidate = (logs_root / candidate).resolve()
        candidate = cwd_candidate if cwd_candidate.exists() else root_candidate
    else:
        candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError("Protected directory not found: %s" % candidate)
    if not is_within(candidate, logs_root):
        raise ValueError(
            "Protected directory is outside logs root: %s" % candidate)
    return candidate


def step_number(path):
    stem = path.stem
    try:
        return int(stem.rsplit("net_step", 1)[1])
    except (IndexError, ValueError):
        return -1


def protected(path, keep_dirs):
    return any(is_within(path, keep_dir) for keep_dir in keep_dirs)


def collect_candidates(logs_root, keep_dirs, mode, keep_latest_steps):
    all_checkpoints = sorted(
        path.resolve() for path in logs_root.rglob("*.pth")
        if path.is_file() and not path.is_symlink())
    unprotected = [
        path for path in all_checkpoints if not protected(path, keep_dirs)]
    if mode == "unprotected":
        return all_checkpoints, unprotected

    by_parent = {}
    for path in unprotected:
        if path.name.startswith("net_step"):
            by_parent.setdefault(path.parent, []).append(path)
    candidates = []
    for paths in by_parent.values():
        paths.sort(key=lambda path: (step_number(path), path.stat().st_mtime))
        keep_count = max(int(keep_latest_steps), 0)
        if keep_count:
            candidates.extend(paths[:-keep_count])
        else:
            candidates.extend(paths)
    return all_checkpoints, sorted(candidates)


def format_size(size):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return "%.2f %s" % (value, unit)
        value /= 1024.0


def main():
    opt = parse_args()
    logs_root = Path(opt.logs_root).expanduser().resolve()
    if not logs_root.is_dir():
        raise FileNotFoundError("Logs root not found: %s" % logs_root)
    if opt.keep_latest_steps < 0:
        raise ValueError("--keep_latest_steps must be non-negative")
    keep_dirs = [
        resolve_keep_dir(raw, logs_root) for raw in opt.keep_dir]
    if opt.mode == "unprotected" and not keep_dirs:
        raise ValueError(
            "--mode unprotected requires at least one valid --keep_dir")
    if opt.execute and opt.mode == "unprotected" and \
            opt.confirm != CONFIRM_TEXT:
        raise ValueError(
            "--mode unprotected --execute requires "
            "--confirm %s" % CONFIRM_TEXT)

    all_checkpoints, candidates = collect_candidates(
        logs_root, keep_dirs, opt.mode, opt.keep_latest_steps)
    total_bytes = sum(path.stat().st_size for path in candidates)

    print("Logs root: %s" % logs_root)
    print("Mode: %s" % opt.mode)
    print("Protected directories:")
    for keep_dir in keep_dirs:
        print("  KEEP %s" % keep_dir)
    print("Checkpoint files found: %d" % len(all_checkpoints))
    print("Deletion candidates: %d (%s)" %
          (len(candidates), format_size(total_bytes)))
    for path in candidates:
        print("  DELETE %s (%s)" %
              (path, format_size(path.stat().st_size)))

    if not opt.execute:
        print("\nDry run only: no files were deleted.")
        return

    deleted = 0
    freed = 0
    for path in candidates:
        size = path.stat().st_size
        path.unlink()
        deleted += 1
        freed += size
    print("\nDeleted %d checkpoint files; freed %s." %
          (deleted, format_size(freed)))


if __name__ == "__main__":
    main()
