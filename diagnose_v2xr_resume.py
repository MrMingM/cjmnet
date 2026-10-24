"""Read existing smoke checkpoints on the server CPU; never train or change the resume gate.

Standalone so uploading this file cannot change the running package's source identity.
Local verification uses synthetic NumPy states only; torch is imported only by main on Linux.
"""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


RTOL, ATOL = 1e-4, 1e-5
KEYS = ('model', 'aux', 'optimizer', 'scheduler', 'rng', 'sampler', 'global_step',
        'accumulation_position', 'criterion', 'criterion_accumulators', 'config',
        'input_identity', 'microbatch', 'epoch', 'contract_identity')
ARTIFACTS = {'detector.pth', 'resume.pth', 'sampling.json', 'actual_samples.json'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_view(value):
    if isinstance(value, np.ndarray):
        return value
    # CPU-only torch tensors are converted by the caller; no torch import here.
    return None


class Comparison:
    """Collect every failure, including exact RNG/metadata checks after float failures."""
    def __init__(self, as_array=array_view):
        self.as_array = as_array
        self.tensors = []
        self.exact_errors = []

    def error(self, path, reason):
        self.exact_errors.append(dict(path=path, reason=reason))

    def compare(self, left, right, path):
        a, b = self.as_array(left), self.as_array(right)
        if a is not None or b is not None:
            if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype:
                self.error(path, 'array type/shape/dtype differs')
                return
            if not np.issubdtype(a.dtype, np.floating):
                if not np.array_equal(a, b):
                    self.error(path, 'integer/nonfloating array differs')
                return
            count = int(a.size)
            invalid_a, invalid_b = int((~np.isfinite(a)).sum()), int((~np.isfinite(b)).sum())
            row = dict(path=path, dtype=str(a.dtype), shape=list(a.shape), elements=count,
                       nonfinite_left=invalid_a, nonfinite_right=invalid_b)
            if invalid_a or invalid_b:
                row.update(passed=False, violation_count=count)
                self.error(path, 'nonfinite floating state')
            elif count:
                # Float64 diagnostics avoid overflow/underflow while preserving the original
                # float32 allclose decision used by smoke.py below.
                wide_a, wide_b = a.astype(np.float64), b.astype(np.float64)
                delta = np.abs(wide_a - wide_b)
                allowed = ATOL + RTOL * np.abs(wide_b)
                with np.errstate(over='ignore', invalid='ignore'):
                    passed_mask = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=False)
                where = int(np.argmax(delta))
                reference_norm = float(np.linalg.norm(wide_b.ravel()))
                row.update(passed=bool(passed_mask.all()), violation_count=int((~passed_mask).sum()),
                           changed_elements=int(np.count_nonzero(a != b)),
                           max_absolute_delta=float(delta.max()),
                           max_tolerance_ratio=float((delta / allowed).max()),
                           relative_l2=float(np.linalg.norm(delta.ravel()) / max(reference_norm, ATOL)),
                           left_max_abs=float(np.abs(wide_a).max()), right_max_abs=float(np.abs(wide_b).max()),
                           worst_flat_index=where, worst_left=float(wide_a.ravel()[where]),
                           worst_right=float(wide_b.ravel()[where]))
            else:
                row.update(passed=True, violation_count=0, max_absolute_delta=0.,
                           max_tolerance_ratio=0., relative_l2=0., changed_elements=0)
            self.tensors.append(row)
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                self.error(path, 'dictionary keys/type differs')
                if not isinstance(right, dict):
                    return
            for key in left.keys() & right.keys():
                self.compare(left[key], right[key], path + '/' + str(key))
        elif isinstance(left, (tuple, list)):
            if type(left) is not type(right) or len(left) != len(right):
                self.error(path, 'sequence type/length differs')
                return
            for index, (a, b) in enumerate(zip(left, right)):
                self.compare(a, b, path + '/' + str(index))
        elif isinstance(right, (dict, tuple, list)) or left != right:
            self.error(path, 'scalar/type differs')
        elif isinstance(left, (float, np.floating)) and not math.isfinite(float(left)):
            self.error(path, 'nonfinite scalar')

    def summary(self):
        sections = {}
        for key in KEYS:
            belongs = lambda path: path == key or path.startswith(key + '/')
            rows = [row for row in self.tensors if belongs(row['path'])]
            errors = [row for row in self.exact_errors if belongs(row['path'])]
            sections[key] = dict(passed=not errors and all(row['passed'] for row in rows),
                floating_arrays=len(rows), failed_floating_arrays=sum(not row['passed'] for row in rows),
                violating_elements=sum(row['violation_count'] for row in rows), exact_errors=len(errors),
                maximum_absolute_delta=max((row.get('max_absolute_delta', 0.) for row in rows), default=0.))
        failed = [row for row in self.tensors if not row['passed']]
        failed.sort(key=lambda row: (-row.get('max_tolerance_ratio', float('inf')), row['path']))
        return dict(sections=sections, exact_errors=self.exact_errors,
                    failed_floating_arrays=failed, all_floating_arrays=self.tensors,
                    registered_gate_passed=not self.exact_errors and not failed,
                    tolerance=dict(rtol=RTOL, atol=ATOL))


def verify_epoch(root, folder, group, epoch):
    commit = read(folder / 'train_complete.json')
    if (commit['status'] != 'complete' or not commit['smoke_only'] or commit['group'] != group
            or commit['epoch'] != epoch or commit['contract_identity'] != read(root / 'contract.json')['identity']
            or set(commit['output_hashes']) != ARTIFACTS):
        raise ValueError('Invalid completed smoke checkpoint: ' + str(folder))
    for name, expected in commit['output_hashes'].items():
        if sha(folder / name) != expected:
            raise ValueError('Committed artifact hash differs: ' + str(folder / name))
    return commit


def inspect(root, group, load_checkpoint, as_array=array_view):
    root = Path(root)
    original = root / 'smoke' / group / 'epochs'
    repeated = root / 'smoke_replay' / group / 'epochs'
    folders = (original / '01', repeated / '01', original / '02', repeated / '02')
    commits = [verify_epoch(root, folder, group, int(folder.name)) for folder in folders]
    start_identical = all(sha(folders[0] / name) == sha(folders[1] / name) for name in ARTIFACTS)
    device_same = all((row['physical_gpu'], row['gpu_name']) ==
                      (commits[0]['physical_gpu'], commits[0]['gpu_name']) for row in commits)
    schedules = {name: read(folders[2] / name) == read(folders[3] / name)
                 for name in ('sampling.json', 'actual_samples.json')}
    a = load_checkpoint(folders[2] / 'resume.pth')
    b = load_checkpoint(folders[3] / 'resume.pth')
    comparison = Comparison(as_array)
    for key in KEYS:
        if key not in a or key not in b:
            comparison.error(key, 'required checkpoint field missing')
        else:
            comparison.compare(a[key], b[key], key)
    if a.get('group') != group or b.get('group') != group:
        comparison.error('checkpoint_group', 'checkpoint group differs from requested group')
    if a.get('completed_training') is not True or b.get('completed_training') is not True:
        comparison.error('completed_training', 'checkpoint is not marked completed')
    if a.get('amp_scaler') is not None or b.get('amp_scaler') is not None:
        comparison.error('amp_scaler', 'FP32 checkpoint unexpectedly contains an AMP scaler')
    report = comparison.summary()
    if not start_identical or not device_same or not all(schedules.values()):
        conclusion = 'checkpoint_start_device_or_data_mismatch'
    elif report['exact_errors']:
        conclusion = 'exact_state_or_nonfinite_mismatch'
    elif not report['registered_gate_passed']:
        conclusion = 'numeric_tolerance_exceeded_cause_not_yet_verified'
    else:
        conclusion = 'within_existing_registered_tolerance'
    report.update(group=group, artifact_hashes_verified=True, epoch1_start_identical=start_identical,
                  same_device=device_same, epoch2_schedules_identical=schedules, conclusion=conclusion,
                  checkpoint_paths=[str(folder / 'resume.pth') for folder in folders],
                  checkpoint_commits=commits, failure=read(root / 'failure.json') if (root / 'failure.json').exists() else None,
                  limitation='Both epoch2 branches already load epoch1. This compares repeated resumed runs; '
                             'it does not independently prove continuous-versus-resumed equivalence or a HIP root cause.',
                  run_status_changed=False, tolerance_changed=False, retrained=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Existing remote run directory; no new training')
    parser.add_argument('--group', choices=['E0', 'E1', 'E2', 'E3'], default='E0')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Checkpoint inspection is server-only; local checks must use synthetic NumPy states')
    root = Path(args.output).resolve(strict=True)
    if Path('/data') not in root.parents or any(part.lower() in ('test', 'testing') for part in root.parts):
        parser.error('Use the existing accepted run directory on /data')
    import fcntl
    with contextlib.ExitStack() as stack:
        for name in ('.controller.lock', '.worker.lock'):
            stream = stack.enter_context((root / name).open('rb'))
            try:
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Run still active; wait for its controller/worker to exit before diagnosis') from None
        import torch
        def as_array(value):
            if isinstance(value, torch.Tensor):
                if value.device.type != 'cpu':
                    raise ValueError('Diagnostic checkpoint tensor must stay on CPU')
                return value.detach().numpy()
            return array_view(value)
        report = inspect(root, args.group,
                         lambda path: torch.load(path, map_location='cpu', weights_only=False), as_array)
        report.update(diagnostic_source_sha256=sha(Path(__file__)),
                      torch_version=torch.__version__, python=sys.version, time=time.time())
        target = root / 'diagnostics' / ('resume_' + args.group + '_' + str(time.time_ns()) + '.json')
        target.parent.mkdir(exist_ok=True)
        with target.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        print('REPORT=' + str(target), flush=True)
        print('文件哈希核验：通过；第1轮起点完全相同：' + str(report['epoch1_start_identical']), flush=True)
        print('同一显卡：' + str(report['same_device']) + '；第2轮数据与标签：' + str(report['epoch2_schedules_identical']), flush=True)
        for key in ('model', 'aux', 'optimizer', 'scheduler', 'rng', 'criterion_accumulators'):
            print(key + ': ' + json.dumps(report['sections'][key], ensure_ascii=False), flush=True)
        for row in report['exact_errors'][:10]:
            print('EXACT_MISMATCH ' + json.dumps(row, ensure_ascii=False), flush=True)
        for row in report['failed_floating_arrays'][:10]:
            print('FLOAT_MISMATCH ' + json.dumps(row, ensure_ascii=False), flush=True)
        print('CONCLUSION=' + report['conclusion'], flush=True)
        print('仅诊断已有文件；没有重训、改容差或修改任务状态。', flush=True)


if __name__ == '__main__':
    main()
