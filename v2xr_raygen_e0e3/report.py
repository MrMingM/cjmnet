"""Data/GPU-independent selection and reports; no placeholder scientific scores."""
import csv
import io
from pathlib import Path

from .common import atomic_bytes, bound, identity, read, sha, write


def choose(rows, tolerance=.005):
    if not rows:
        raise ValueError('No completed candidates')
    maximum = max(r['results']['clean']['AP']['O']['ap70'] for r in rows)
    eligible = [r for r in rows if maximum - r['results']['clean']['AP']['O']['ap70'] <= tolerance]
    best = min(eligible, key=lambda r: (-(r['results']['fog']['AP']['O']['ap70']
                                       + r['results']['snow']['AP']['O']['ap70']) / 2, r['epoch']))
    return best, maximum, [r['epoch'] for r in eligible]


def evaluation_markdown(summary):
    lines = ['# V2X-R BEV AP', '', 'Smoke only: ' + str(summary['smoke_only']), '',
             '|Weather|Protocol|AP30|AP50|AP70|Counted GT|Boxes|None|Empty|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for w, r in summary['results'].items():
        for p, ap in r['AP'].items():
            q = r['counts']
            lines.append(f'|{w}|{p}|{ap["ap30"]:.9f}|{ap["ap50"]:.9f}|{ap["ap70"]:.9f}|'
                         f'{r["counted_gt"][p]}|{q["prediction_count"]}|{q["none_frames"]}|{q["empty_non_none_frames"]}|')
    return '\n'.join(lines) + '\n'


def group_report(output, group, smoke=False):
    root = Path(output)
    c = read(root / 'config.json')
    branch = root / ('smoke' if smoke else 'groups') / group
    n = 2 if smoke else c['epochs']
    history, rows = [], []
    for epoch in range(1, n + 1):
        ep = branch / 'epochs' / f'{epoch:02d}'
        train = read(ep / 'train_complete.json')
        for name, digest in train['output_hashes'].items():
            if sha(ep / name) != digest:
                raise ValueError('Training evidence changed during report')
        val = read(ep / 'evaluation/summary.json')
        if (val['status'] != 'complete' or val['smoke_only'] != smoke
                or (not smoke and not val['full_validation'])
                or val['checkpoint_sha256'] != sha(ep / 'detector.pth')):
            raise ValueError('Incomplete/mismatched epoch validation')
        for name, digest in val['output_hashes'].items():
            if sha(ep / 'evaluation' / name) != digest:
                raise ValueError('Evaluation evidence changed during report')
        rows.append(val)
        history.append(dict(epoch=epoch, training=train, evaluation=val))
    best, maximum, eligible = choose(rows, c['selection_clean_tolerance'])
    selected = dict(status='complete', group=group, epoch=best['epoch'], smoke_only=smoke,
        checkpoint=str(branch / 'epochs' / f'{best["epoch"]:02d}' / 'detector.pth'),
        checkpoint_sha256=best['checkpoint_sha256'], highest_clean_O_AP70=maximum,
        eligible_epochs=eligible, tolerance=c['selection_clean_tolerance'],
        policy='maximum Fog/Snow mean O AP70 within .005 of highest Clean O AP70; exact ties earliest',
        candidate_count=n, candidate_summary_hashes={str(e): sha(branch / 'epochs' / f'{e:02d}' / 'evaluation/summary.json')
                                                   for e in range(1, n + 1)})
    bound(branch / 'selected.json', selected)
    write(branch / 'history.json', history)
    summary = dict(status='complete', group=group, smoke_only=smoke, selected=selected, evaluation=best,
                   parameters=read(branch / 'parameters.json'),
                   train_seconds=sum(r['training']['seconds'] for r in history),
                   training_exposure=sum(r['training']['totals']['frames'] for r in history),
                   updates=sum(r['training']['totals']['updates'] for r in history),
                   auxiliary_label_counts={k: sum(r['training']['label_counts'].get(k, 0) for r in history)
                                           for k in ('windows', 'valid_windows', 'voxels', 'O', 'E', 'U', 'ego', 'peer')},
                   gradients=[dict(epoch=r['epoch'], diagnostics=r['training']['gradients']) for r in history])
    counts = summary['auxiliary_label_counts']
    summary['valid_auxiliary_window_fraction'] = counts['valid_windows'] / counts['windows'] if counts['windows'] else 0.
    summary['ego_surface_fraction'] = counts['ego'] / counts['O'] if counts['O'] else 0.
    summary['peer_only_surface_fraction'] = counts['peer'] / counts['O'] if counts['O'] else 0.
    summary['disk_bytes'] = sum(p.stat().st_size for p in branch.rglob('*') if p.is_file() and not p.is_symlink())
    write(branch / 'summary.json', summary)
    atomic_bytes(branch / 'RESULTS.md', (evaluation_markdown(best) + '\nSelected epoch: ' + str(best['epoch'])
                 + '\n\nSingle-seed controlled validation result. No SOTA or convergence claim.\n').encode())
    return summary


def run(output, smoke=False):
    root = Path(output)
    target = root / 'smoke' if smoke else root
    summaries, failures = {}, []
    for g in ('E0', 'E1', 'E2', 'E3'):
        try:
            summaries[g] = group_report(root, g, smoke)
        except (FileNotFoundError, ValueError) as e:
            failures.append(dict(group=g, reason=str(e)))
    rows, differences = [], []
    for g, s in summaries.items():
        for w, result in s['evaluation']['results'].items():
            for p, aps in result['AP'].items():
                rows.append(dict(group=g, epoch=s['selected']['epoch'], weather=w, protocol=p, **aps,
                    counted_gt=result['counted_gt'][p], prediction_count=result['counts']['prediction_count'],
                    none_frames=result['counts']['none_frames'], empty_frames=result['counts']['empty_non_none_frames']))
    for a, b in (('E1', 'E0'), ('E2', 'E1'), ('E3', 'E2'), ('E3', 'E0')):
        if a not in summaries or b not in summaries:
            continue
        for w in ('clean', 'fog', 'snow'):
            for p in ('O', 'O*', 'F'):
                sa, sb = summaries[a]['evaluation']['results'][w]['AP'][p], summaries[b]['evaluation']['results'][w]['AP'][p]
                differences.append(dict(comparison=a + '-' + b, weather=w, protocol=p,
                                        **{k + '_pp': 100 * (sa[k] - sb[k]) for k in sa}))
        for p in ('O', 'O*', 'F'):
            differences.append(dict(comparison=a + '-' + b, weather='fog_snow_mean', protocol=p,
                **{k + '_pp': sum(100 * (summaries[a]['evaluation']['results'][w]['AP'][p][k]
                                        - summaries[b]['evaluation']['results'][w]['AP'][p][k]) for w in ('fog', 'snow')) / 2
                   for k in ('ap30', 'ap50', 'ap70')}))
    summary = dict(status='complete' if len(summaries) == 4 and not failures else 'incomplete',
                   smoke_only=smoke, groups=summaries, rows=rows, differences=differences, failures=failures,
                   interpretation='Only O gain without F gain does not support improved fixed-target detection; '
                                  'single seed / 20-epoch budget is not a SOTA or convergence claim.')
    stream = io.StringIO(newline='')
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_bytes(target / 'results.csv', stream.getvalue().encode())
    text = ['# V2X-R E0-E3', '', 'Status: ' + summary['status'], '', summary['interpretation'], '']
    for g, s in summaries.items():
        text += [f'## {g}: epoch {s["selected"]["epoch"]}', '', evaluation_markdown(s['evaluation']),
                 f'Train seconds: {s["train_seconds"]:.1f}; frames: {s["training_exposure"]}; updates: {s["updates"]}.', '']
    text += ['## Differences (percentage points)', '', '|Comparison|Weather|Protocol|AP30|AP50|AP70|', '|---|---|---|---:|---:|---:|']
    for r in differences:
        text.append(f'|{r["comparison"]}|{r["weather"]}|{r["protocol"]}|{r["ap30_pp"]:+.6f}|{r["ap50_pp"]:+.6f}|{r["ap70_pp"]:+.6f}|')
    if failures:
        text += ['', 'Incomplete evidence: ' + repr(failures)]
    if (root / 'smoke/server_checks.json').exists():
        summary['server_checks'] = read(root / 'smoke/server_checks.json')
    if smoke:
        summary['resume_checks'] = {g: read(root / 'smoke' / g / 'resume_check.json') for g in summaries}
        recovery = root / 'smoke/E0/epochs/01/evaluation/recovery_check.json'
        if not recovery.exists():
            raise ValueError('Smoke evaluation interruption/recovery check did not complete')
        summary['evaluation_recovery_check'] = read(recovery)
    if (root / 'execution.json').exists():
        from .resume_compat import GPU_CAPACITY, effective_execution
        summary['execution'] = effective_execution(root)
        summary['original_execution'] = read(root / 'execution.json')
        if (root / GPU_CAPACITY).exists():
            summary['gpu_capacity_recheck'] = read(root / GPU_CAPACITY)
    if (root / 'stage_status.json').exists():
        summary['stage_status'] = read(root / 'stage_status.json')
    summary['failure_history'] = [read(p) for p in sorted((root / 'failure_history').glob('*.json'))]
    for filename, key in (('compute_policy.json', 'compute_policy'), ('data_reuse.json', 'data_reuse'),
                          ('smoke/stable_calculation_checks.json', 'stable_calculation_checks')):
        if (root / filename).exists():
            summary[key] = read(root / filename)
    repair = root / 'source_repair/smoke_unknown_U_v1.json'
    if repair.exists():
        provenance = read(repair)
        summary['source_repair'] = dict(identity=provenance['identity'], record=str(repair),
            original_sources_identity=provenance['original_sources_identity'],
            active_sources_identity=identity(provenance['active_sources']),
            interpretation=provenance['interpretation'])
    from .resume_compat import NUMERIC_MARKER
    if (root / NUMERIC_MARKER).exists():
        numeric_repair = read(root / NUMERIC_MARKER)
        summary['resume_check_repair'] = numeric_repair
        summary['source_repair']['active_sources_identity'] = identity(numeric_repair['active_sources'])
    from .resume_compat import RMS_MARKER
    if (root / RMS_MARKER).exists():
        rms_repair = read(root / RMS_MARKER)
        summary['resume_rms_repair'] = rms_repair
        summary['source_repair']['active_sources_identity'] = identity(rms_repair['active_sources'])
    summary['total_disk_bytes'] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    if (root / 'labels/manifest.json').exists():
        labels = read(root / 'labels/manifest.json')
        summary['shared_labels'] = {k: v for k, v in labels.items() if k != 'entries'}
    write(target / 'summary.json', summary)
    atomic_bytes(target / 'RESULTS_E0_E3.md', ('\n'.join(text) + '\n').encode())
    return summary
