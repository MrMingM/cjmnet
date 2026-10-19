"""O is the main reported protocol; inherited O*/F remain transparent audit results."""
from pathlib import Path
from auditlib import HERE, SETTINGS, atomic_json, complete, load_frozen, read_json, sha256

inherited = load_frozen('results', 'results.py')


def write_evaluation(output, summary):
    inherited.write_evaluation(output, summary)
    original = (output / 'RESULTS.md').read_text(encoding='utf-8')
    lines = ['# V2X-R Clean-trained GSPR-AttFuse: official O', '',
        'Random initialization; V2X-R Clean only; no OPV2V weights or point/noise labels.',
        'Main metric: O / OFFICIAL_EXACT. Model adapter changes point features, never GT or AP.',
        f'Frames per condition: {summary["frame_count"]}/829; ' + ('FULL VALIDATION' if summary['full_validation'] else 'SMOKE ONLY'),
        '', '| Condition | O AP30 | O AP50 | O AP70 | Official GT | Counted GT | None frames |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for condition, row in summary['results'].items():
        ap, counts = row['AP']['O'], row['counts']
        lines.append(f'| {condition} | {ap["ap30"]:.8f} | {ap["ap50"]:.8f} | {ap["ap70"]:.8f} | '
                     f'{counts["official_gt"]} | {row["counted_gt"]["O"]} | {counts["none_frames"]} |')
    lines += ['', 'O*/F audit details follow. These audits do not affect O selection or official matching.', '', original]
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def baseline(args):
    pins = read_json(HERE / 'PINS.json')['baseline']
    root = Path(args.baseline_root)
    for name, digest in pins['files'].items():
        if sha256(root / name) != digest:
            raise ValueError('Completed plain AttFuse reference changed: ' + name)
    summary = read_json(root / 'final_evaluation/summary.json')
    chosen = read_json(root / 'selected.json')
    if (summary['status'] != 'complete' or not summary['full_validation'] or summary['frame_count'] != 829
            or summary['checkpoint_sha256'] != pins['checkpoint_sha256']
            or chosen['checkpoint_sha256'] != pins['checkpoint_sha256']):
        raise ValueError('Plain AttFuse baseline is not the completed fixed reference')
    # Its original F selection also selects epoch20 if ranked by O; no baseline retraining needed.
    history = read_json(root / 'history.json')['epochs']
    o_best = min(history, key=lambda r: (-r['clean_AP']['O']['ap70'], r['epoch']))
    if o_best['epoch'] != chosen['epoch']:
        raise ValueError('Baseline selection is not the same under Clean O; disclose before comparison')
    return summary


def comparison(args, current):
    old = baseline(args)
    if old['frozen_d1c_gt_identity'] != current['frozen_d1c_gt_identity']:
        raise ValueError('GSPR and AttFuse have different frozen GT/frame identities')
    rows, lines = [], ['# GSPR vs plain AttFuse: same V2X-R Clean training and official O', '',
        'Both random-init, seed20261017, 20 epochs, Clean-only selection; full829-frame validation.',
        'Baseline epoch20 is highest under both its original F selection and O.',
        'GSPR adds soft point weighting plus fixed keep/evidence regularization; no point labels.',
        'Single seed; validation-selected results do not establish statistical significance or test generalization.',
        '', '| Condition | Metric | Plain AttFuse | GSPR-AttFuse | GSPR - plain / pp |',
        '|---|---|---:|---:|---:|']
    for condition in ('clean', 'fog', 'snow'):
        for metric in ('ap30', 'ap50', 'ap70'):
            a, b = old['results'][condition]['AP']['O'][metric], current['results'][condition]['AP']['O'][metric]
            rows.append(dict(condition=condition, metric=metric, baseline=a, gspr=b, delta_pp=100 * (b - a)))
            lines.append(f'| {condition} | {metric} | {a:.8f} | {b:.8f} | {100*(b-a):+.4f} |')
    output = Path(args.output_root)
    (output / 'COMPARISON_O.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    atomic_json(output / 'comparison_O.json', dict(status='complete', main_protocol='O', rows=rows,
        baseline_summary_sha256=sha256(Path(args.baseline_root) / 'final_evaluation/summary.json'),
        gspr_summary_sha256=sha256(output / 'final_evaluation/summary.json'),
        report_sha256=sha256(output / 'COMPARISON_O.md')))
