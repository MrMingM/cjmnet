"""Server-only official-O report/export and optional frozen selected-model inference."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONDITIONS = ('clean', 'fog', 'snow')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe(path):
    path = Path(path).absolute()
    for candidate in (path, path.resolve()):
        if any(p.lower() in ('test', 'testing') or 'opv2v-w' in p.lower() for p in candidate.parts):
            raise ValueError('This tool only evaluates validation: ' + str(candidate))
    return path.resolve()


def member(root, name):
    path = safe(root / name)
    if path == root or root not in path.parents:
        raise ValueError('Artifact path escapes its output directory: ' + name)
    return path


def atomic(path, text):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8', newline='\n')
    os.replace(temporary, path)


@contextmanager
def lock(path):
    import fcntl
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_check():
    expected = read(HERE / 'source_snapshot.json')['files']
    actual = {p.relative_to(HERE).as_posix(): sha(p) for p in HERE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if actual != expected:
        raise ValueError('Synchronize the complete v2xr_official_selected_eval directory')
    package = ROOT / 'v2xr_train_attfuse'
    for name, digest in read(package / 'source_snapshot.json')['files'].items():
        if sha(package / name) != digest:
            raise ValueError('Frozen training/evaluation source changed: ' + name)
    for name, digest in read(package / 'PINS.json')['dependencies'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Historical dependency changed: ' + name)
    return actual


def selected_model(training):
    summary = read(training / 'training_summary.json')
    chosen = read(training / 'selected.json')
    if (summary['status'] != 'complete' or summary['epochs'] != 20
            or chosen['status'] != 'complete' or chosen['candidate_count'] != 20
            or summary['selected_epoch'] != chosen['epoch']
            or summary['weather_used_for_selection'] or chosen['weather_used']):
        raise ValueError('Need completed 20-epoch Clean-only selection')
    for name in ('selected.json', 'selection_policy.json', 'history.json', 'config.yaml', 'protocol.json', 'contract.json'):
        if sha(member(training, name)) != summary['output_hashes'][name]:
            raise ValueError('Training artifact changed: ' + name)
    best = min(chosen['candidate_metrics'], key=lambda r: (-r['AP70'], r['epoch']))
    if best['epoch'] != chosen['epoch'] or best['AP70'] != chosen['clean_F_AP70']:
        raise ValueError('Clean selection differs from recorded candidates')
    checkpoint = safe(chosen['checkpoint'])
    if training not in checkpoint.parents or sha(checkpoint) != chosen['checkpoint_sha256']:
        raise ValueError('Selected checkpoint bytes/path changed')
    return chosen


def official_report(evaluation, chosen, expected_frames):
    summary = read(evaluation / 'summary.json')
    if (summary['status'] != 'complete' or summary['conditions'] != list(CONDITIONS)
            or summary['frame_count'] != expected_frames
            or summary['full_validation'] != (expected_frames == 829)
            or summary['checkpoint_sha256'] != chosen['checkpoint_sha256']):
        raise ValueError('Incomplete evaluation, different frame count, or wrong selected checkpoint')
    for name, digest in summary['output_hashes'].items():
        if sha(member(evaluation, name)) != digest:
            raise ValueError('Evaluation artifact changed: ' + name)
    results = {}
    for condition in CONDITIONS:
        row = summary['results'][condition]
        counts, ap = row['counts'], row['AP']['O']
        # The frozen evaluator already compares O to inference.main(). Check its saved statistics again.
        replay = read(evaluation / ('stat_' + condition + '_O.json'))
        original = read(evaluation / 'official_runs' / condition / 'official_result_stat.json')
        if replay != original:
            raise ValueError('O replay differs from original official inference statistics: ' + condition)
        counted = counts['official_gt'] - counts['official_gt_in_none_frames']
        if row['counted_gt']['O'] != counted or any(v['gt'] != counted for v in replay.values()):
            raise ValueError('Official None-frame GT denominator differs')
        if any(not math.isfinite(ap[k]) or not 0 <= ap[k] <= 1 for k in ('ap30', 'ap50', 'ap70')):
            raise ValueError('Invalid AP value')
        results[condition] = dict(AP=ap, official_gt=counts['official_gt'], counted_gt=counted,
            skipped_none_frames=counts['none_frames'], skipped_gt=counts['official_gt_in_none_frames'],
            prediction_count=counts['prediction_count'])
    return summary, results


def main():
    if sys.platform != 'linux':
        raise RuntimeError('Server only. Do not run evaluation or real-data checks locally.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('export', 'rerun'), default='export')
    parser.add_argument('--training-root', default='/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1')
    parser.add_argument('--output-root', default='/data/cjm/datasets/logs/v2xr-attfuse-official-protocol-v1')
    parser.add_argument('--evaluation-root', help='New inference output; default OUTPUT_ROOT/inference_full or inference_smoke12')
    parser.add_argument('--limit-frames', type=int, choices=(0, 12), default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    own = source_check()
    training, output = safe(args.training_root), safe(args.output_root)
    for path in (training, output):
        if Path('/data') not in path.parents:
            raise ValueError('Training and report directories must be on /data')
    protected = [training, ROOT]
    protected += [safe(os.environ.get(k, v)) for k, v in dict(
        DATA_ROOT='/data/cjm/datasets/V2X-R/train', TRAIN_ROOT='/data/cjm/datasets/v2xr-official-train-v2',
        RUNTIME_ROOT='/data/cjm/datasets/v2xr-official-runtime-v1',
        MODEL_DIR='/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1',
        D1A_ROOT='/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1',
        GT_AUDIT_ROOT='/data/cjm/datasets/logs/v2xr-official-gt-audit-v1',
        D1C_ROOT='/data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1',
        VIEW_ROOT='/data/cjm/datasets/v2xr-official-validate-v1').items()]
    for source in protected:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('Use an independent official report directory')
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / '.run.lock'):
        chosen = selected_model(training)
        frames = args.limit_frames or 829
        if args.action == 'export':
            if args.limit_frames or args.evaluation_root:
                raise ValueError('Export reads the existing full final_evaluation only')
            evaluation = training / 'final_evaluation'
        else:
            evaluation = safe(args.evaluation_root or output / ('inference_smoke12' if args.limit_frames else 'inference_full'))
            if output not in evaluation.parents:
                raise ValueError('New inference must be inside the independent report directory')
            environment = os.environ.copy()
            environment.update(MODE='evaluate-checkpoint', OUTPUT_ROOT=str(training),
                CHECKPOINT=chosen['checkpoint'], EVALUATION_ROOT=str(evaluation),
                LIMIT_FRAMES=str(args.limit_frames), SMOKE='0', RESUME='1' if args.resume else '0')
            print('Running frozen official inference; checkpoint:', chosen['checkpoint'], flush=True)
            subprocess.run(['sh', str(ROOT / 'v2xr_train_attfuse/run_server.sh')],
                cwd=ROOT, env=environment, check=True)
        summary, results = official_report(evaluation, chosen, frames)
        title = 'SMOKE ONLY' if frames != 829 else 'FULL VALIDATION'
        report = dict(status='complete', protocol='O / OFFICIAL_EXACT', scope='validation',
            action=args.action, frame_count=frames, full_validation=frames == 829,
            selected_epoch=chosen['epoch'], checkpoint=chosen['checkpoint'], checkpoint_sha256=chosen['checkpoint_sha256'],
            source_summary=str(evaluation / 'summary.json'), source_summary_sha256=sha(evaluation / 'summary.json'),
            source_contract_identity=summary['contract_identity'], source_files=own,
            explanation='Official weather-dependent GT; prediction=None skips the whole frame; unchanged BEV VOC2010 global_sort=False.',
            inference_reused=args.action == 'export', results=results)
        lines = ['# AttFuse: official O protocol', '', title,
            f'Validation frames per condition: {frames}/829; selected epoch: {chosen["epoch"]}.',
            '', 'Official weather GT and None-frame skipping; original dataset/model/postprocess/AP.',
            'This reports the selected V2X-R Clean-trained model. No training is performed.',
            '', '| Condition | AP30 | AP50 | AP70 | Official GT | Counted GT | Skipped None frames |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for condition, row in results.items():
            ap = row['AP']
            lines.append(f'| {condition} | {ap["ap30"]:.8f} | {ap["ap50"]:.8f} | {ap["ap70"]:.8f} | '
                f'{row["official_gt"]} | {row["counted_gt"]} | {row["skipped_none_frames"]} |')
        lines += ['', 'Source summary: ' + report['source_summary'],
            'Checkpoint SHA256: `' + chosen['checkpoint_sha256'] + '`.',
            'Export verifies existing artifacts; rerun executes the frozen inference entry point.',
            'The underlying evaluator also stores O*/F audit outputs; this report selects O only.']
        stem = 'OFFICIAL_O_SMOKE12' if args.limit_frames else 'OFFICIAL_O'
        atomic(output / (stem + '.md'), '\n'.join(lines) + '\n')
        report['report_sha256'] = sha(output / (stem + '.md'))
        atomic(output / (stem + '.json'), json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
        print('\n'.join(lines), flush=True)
        print('Saved:', output / (stem + '.md'), flush=True)


if __name__ == '__main__':
    main()
