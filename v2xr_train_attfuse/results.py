"""Readable tables from frozen O/O*/F evaluation; no tuning decisions."""
from pathlib import Path
from auditlib import atomic_json, complete, sha256


def gaps(results, protocol, weather):
    return {k: dict(AP=results['clean']['AP'][protocol][k] - results[weather]['AP'][protocol][k],
                    pp=100 * (results['clean']['AP'][protocol][k] - results[weather]['AP'][protocol][k]))
            for k in ('ap30', 'ap50', 'ap70')}


def write_evaluation(output, summary):
    r = summary['results']
    associated = summary.get('associated_d1d_training')
    lines = ['# D-1D validation protocol results', '',
        f"Frames per condition: {summary['frame_count']}/829. " + ('FULL VALIDATION' if summary['full_validation'] else 'SMOKE ONLY'),
        '', 'O: condition-specific official GT; None skips frame. O*: same GT, None counts as zero predictions.',
        'F: frozen D-1C Official Clean-selected paired fixed GT; None counts as zero predictions.',
        'All three reuse one prediction cache and the unchanged official BEV/VOC2010 evaluator, global_sort=False.',
        '', '| Condition | Official GT | Fixed GT | Removed | None frames | Official GT in None frames |',
        '|---|---:|---:|---:|---:|---:|']
    if associated:
        a = associated['train_data_acceptance']
        lines[4:4] = [f"Associated D-1D Clean training inventory: {a['actual_selected_cooperative_frames']} actual frames; "
            f"published reference {a['official_reference_cooperative_frames']}; deviation -1 explicitly accepted before training.",
            'This train-inventory statement applies to the D-1D run, not to OPV2V checkpoint pretraining.', '']
    for c, row in r.items():
        n = row['counts']
        lines.append(f"| {c} | {n['official_gt']} | {n['fixed_gt']} | {n['removed_gt']} | {n['none_frames']} | {n['official_gt_in_none_frames']} |")
    lines += ['', '| Condition | Protocol | AP30 | AP50 | AP70 |', '|---|---|---:|---:|---:|']
    for c, row in r.items():
        for p, ap in row['AP'].items():
            lines.append(f"| {c} | {p} | {ap['ap30']:.8f} | {ap['ap50']:.8f} | {ap['ap70']:.8f} |")
    if set(r) == {'clean', 'fog', 'snow'}:
        summary['weather_gaps'] = {p: {w: gaps(r, p, w) for w in ('fog', 'snow')} for p in ('O', 'O*', 'F')}
        lines += ['', '| Protocol | Clean AP70 | Fog AP70 | Snow AP70 | Clean-Fog pp | Clean-Snow pp |',
                  '|---|---:|---:|---:|---:|---:|']
        for p in ('O', 'O*', 'F'):
            lines.append(f"| {p} | {r['clean']['AP'][p]['ap70']:.8f} | {r['fog']['AP'][p]['ap70']:.8f} | {r['snow']['AP'][p]['ap70']:.8f} | "
                         f"{summary['weather_gaps'][p]['fog']['ap70']['pp']:.4f} | {summary['weather_gaps'][p]['snow']['ap70']['pp']:.4f} |")
        summary['effects'] = {}
        lines += ['', '| Weather | AP70 NoneSkipEffect pp | AP70 GTFilterEffect pp | AP70 GapMasking pp |', '|---|---:|---:|---:|']
        for w in ('fog', 'snow'):
            effects = {}
            for k in ('ap30', 'ap50', 'ap70'):
                a = r[w]['AP']
                values = dict(NoneSkipEffect=a['O'][k] - a['O*'][k], GTFilterEffect=a['O*'][k] - a['F'][k],
                    GapMasking=summary['weather_gaps']['F'][w][k]['AP'] - summary['weather_gaps']['O'][w][k]['AP'])
                effects[k] = {name: dict(AP=v, pp=v * 100) for name, v in values.items()}
            summary['effects'][w] = effects
            lines.append(f"| {w} | {effects['ap70']['NoneSkipEffect']['pp']:.4f} | {effects['ap70']['GTFilterEffect']['pp']:.4f} | {effects['ap70']['GapMasking']['pp']:.4f} |")
        lines += ['', 'These are descriptive protocol differences. O supports benchmark comparison; F is the robustness main result; O* audits None frames.',
            'A large positive GTFilterEffect/GapMasking indicates that official scoring hides degradation on fixed Clean targets.',
            'Near-zero effects mean small AP impact for these predictions, even when many GT were removed. Removed-target recall alone cannot establish small AP impact.',
            'Single-seed validation differences do not establish statistical significance.']
    lines += ['', 'Checkpoint SHA256: `' + summary['checkpoint_sha256'] + '`.',
              'Frozen D-1C GT identity: `' + summary['frozen_d1c_gt_identity'] + '`.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


def write_comparison(output):
    new = complete(output / 'final_evaluation' / 'summary.json')
    old = complete(output / 'opv2v_matched_evaluation' / 'summary.json')
    if (old['frozen_d1c_gt_identity'] != new['frozen_d1c_gt_identity']
            or not old['full_validation'] or not new['full_validation']):
        raise ValueError('Comparison GT/protocol/frame identity mismatch')
    rows, lines = [], ['# D-1D matched domain comparison', '',
        'Both checkpoints use the same frozen official architecture, max_cav=7, input pipeline, postprocess and D-1C O/O*/F GT.',
        'The OPV2V checkpoint is evaluation-only. Legacy v2xr_attfuse AP is excluded because its GT/FOV/max_cav protocol differs.',
        '', '| Training domain | Protocol | Clean AP70 | Fog AP70 | Snow AP70 | Clean-Fog pp | Clean-Snow pp |',
        '|---|---|---:|---:|---:|---:|---:|']
    for label, s in (('OPV2V', old), ('V2X-R Clean', new)):
        r = s['results']
        for p in ('O', 'O*', 'F'):
            row = dict(training_domain=label, protocol=p,
                AP70={c: r[c]['AP'][p]['ap70'] for c in ('clean', 'fog', 'snow')},
                gap_pp={w: gaps(r, p, w)['ap70']['pp'] for w in ('fog', 'snow')})
            rows.append(row)
            lines.append(f"| {label} | {p} | {row['AP70']['clean']:.8f} | {row['AP70']['fog']:.8f} | {row['AP70']['snow']:.8f} | {row['gap_pp']['fog']:.4f} | {row['gap_pp']['snow']:.4f} |")
    reduction = {w: old['weather_gaps']['F'][w]['ap70']['pp'] - new['weather_gaps']['F'][w]['ap70']['pp'] for w in ('fog', 'snow')}
    lines += ['', f'Fixed-GT weather-gap reduction (OPV2V gap minus V2X-R gap), pp: {reduction}.',
        'Interpret the gap together with absolute Clean/weather AP. A smaller gap with collapsed Clean AP is not evidence of improved weather robustness.',
        'Large positive reductions support a cross-dataset domain-shift contribution; substantial residual F gaps support further weather mechanism work.',
        'This is a single-seed descriptive comparison; no statistical significance claim is made.']
    (output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    atomic_json(output / 'comparison.json', dict(status='complete', rows=rows, fixed_gap_reduction_pp=reduction,
        source_summary_sha256={name: sha256(output / name / 'summary.json') for name in ('final_evaluation', 'opv2v_matched_evaluation')}))
