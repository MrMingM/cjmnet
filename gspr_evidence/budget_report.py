"""四种天气收齐后再比较，不拿 test 挑预算。"""
import argparse
import json
from pathlib import Path
from .budget_scan import choose_budget, METRICS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(args.run)
    summaries = {w: json.loads((root/w/'summary.json').read_text()) for w in ('clean', 'fog', 'rain', 'snow')}
    phases = {x['phase'] for x in summaries.values()}
    if len(phases) != 1 or len({json.dumps(x['comparison_contract'], sort_keys=True) for x in summaries.values()}) != 1:
        raise ValueError('不能混用不同协议或模型的结果')
    phase = next(iter(phases))
    lines = [f'# A0B0 budget scan: {phase}', '',
             'Non-global AP. Development uses validation + online weather; benchmark uses historical test files.', '',
             '| Weather | Budget | Frames | Actual MiB/frame | AP30 | AP50 | AP70 | Δ30 pp | Δ50 pp | Δ70 pp | All pass |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for weather, summary in summaries.items():
        full = summary['results']['full']
        for budget, row in summary['results'].items():
            if row['frames'] != full['frames']:
                raise ValueError('同一条件下帧数不一致')
            label = f'{int(budget)/2**20:g} MiB' if budget.isdigit() else budget
            delta = [100*(row[k]-full[k]) for k in METRICS]
            lines.append(f'| {weather} | {label} | {row["frames"]} | {row["mean_bytes"]/2**20:.4f} | '
                f'{row["ap30"]:.8f} | {row["ap50"]:.8f} | {row["ap70"]:.8f} | '
                f'{delta[0]:+.6f} | {delta[1]:+.6f} | {delta[2]:+.6f} | {row["passes_full"]} |')
    if phase == 'development':
        selection = choose_budget(summaries)
        (root/'experiment.yaml').write_bytes((root/'clean'/'resolved_experiment.yaml').read_bytes())
        (root/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
        lines += ['', f'Development choice: {selection["selected_mode"]}; budget bytes: {selection["selected_budget_bytes"]}.',
                  'This is only a development choice; formal test must confirm it.']
    else:
        accepted = all(row['passes_full'] for s in summaries.values() for k, row in s['results'].items() if k.isdigit())
        sparse = any(k.isdigit() for k in summaries['clean']['results'])
        lines += ['', f'Formal sparse budget passes all 12 metrics: {accepted if sparse else "N/A: full fallback"}.',
                  'Do not choose a different budget from test results. No measured drop is not a guarantee on future data.']
    (root/'budget_results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(root/'budget_results.md')


if __name__ == '__main__':
    main()
