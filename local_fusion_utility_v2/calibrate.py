"""Select one global threshold per method using full validation joint outputs."""
import argparse
from pathlib import Path


def choose_threshold(summaries, method, candidates, gates):
    """Use common Clean retention/harm gates; KEEP_FULL is always available.

    No test metrics are accepted by this function. AP values use [0,1] units.
    """
    rows = []
    for index, threshold in enumerate([None, *candidates]):
        name = f'{method}@{index}'
        feasible = True
        scores, actions, details = [], [], {}
        for weather, summary in summaries.items():
            ap = summary['results'][name]['ap70']
            baseline = summary['results']['baseline']['ap70']
            d = summary['diagnostics_vs_baseline'][name]
            denom = max(summary['baseline_tp'], 1)
            passed = (d['lost'] <= gates['max_lost_tp_fraction'] * denom and
                      d['new_fp'] <= gates['max_new_fp_per_baseline_tp'] * denom)
            if weather == 'clean':
                passed = passed and ap >= baseline - gates['clean_ap70_tolerance']
            feasible = feasible and passed
            scores.append(ap)
            actions.append(d['selected_tiles_per_frame'])
            details[weather] = dict(ap70=ap, passed=bool(passed), **d)
        rows.append(dict(threshold=threshold, policy=name, feasible=bool(feasible),
                         mean_ap70=sum(scores)/len(scores),
                         mean_selected_tiles=sum(actions)/len(actions), conditions=details))
    feasible_rows = [row for row in rows if row['feasible']]
    if not feasible_rows:
        raise RuntimeError('KEEP_FULL should always satisfy the validation gates')
    best = max(feasible_rows, key=lambda row: (row['mean_ap70'], -row['mean_selected_tiles']))
    return dict(threshold=best['threshold'], selected_policy=best['policy'], candidates=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--frontend-config', required=True)
    p.add_argument('--frontend-checkpoint', required=True)
    p.add_argument('--utility-checkpoint', required=True)
    p.add_argument('--loss-gain-checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    from gspr_communication.runtime import new_output, device, seed_all, sha256, write_json, verify_frozen
    from gspr_evidence import runtime as er
    from . import runtime as rt
    from .evaluate import evaluate_condition

    verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    settings = rt.settings(options)
    gates = options['calibration']
    for key in ('clean_ap70_tolerance', 'max_lost_tp_fraction', 'max_new_fp_per_baseline_tp'):
        if not 0 <= float(gates[key]) <= 1:
            raise ValueError('Invalid calibration gate: '+key)
    target = device()
    seed_all(int(options['seed']))
    model, digest = er.load_model(hypes, options, args.frontend_checkpoint, target)
    specification = rt.contract(options, args.frontend_config, digest)
    predictors = {}
    checkpoints = {}
    for name, path in [('utility', args.utility_checkpoint), ('loss_gain', args.loss_gain_checkpoint)]:
        predictors[name], _ = rt.load_predictor(path, specification, target, name)
        checkpoints[name] = dict(path=str(Path(path).resolve()), sha256=sha256(path))
    choices = {method: settings[method+'_thresholds'] for method in ('confidence', 'utility', 'loss_gain')}
    policies = {f'{method}@{index}': (method, threshold)
                for method, values in choices.items()
                for index, threshold in enumerate([None, *values])}
    output = new_output(args.output_dir)
    summaries = {}
    for weather in ('clean', 'fog', 'rain', 'snow'):
        seed_all(int(options['seed']))
        dataset, loader, indices = er.make_loader(hypes, options, train=False, weather=weather)
        destination = output/weather
        destination.mkdir()
        print('Full joint-action validation calibration:', weather, flush=True)
        summaries[weather] = evaluate_condition(
            model, predictors, {}, dataset, loader, indices,
            'clean' if weather == 'clean' else 'weather', settings, target, destination,
            policies=policies)
        write_json(destination/'summary.json', summaries[weather])
    methods = {method: choose_threshold(summaries, method, values, gates)
               for method, values in choices.items()}
    result = dict(schema=2, selection_data='full_validation_joint_actions', test_data_used=False,
                  cache_contract=specification, checkpoints=checkpoints, gates=gates,
                  methods=methods, validation_root=hypes['validate_dir'])
    write_json(output/'calibration.json', result)
    for name, decision in methods.items():
        print(name, 'threshold:', decision['threshold'], '(None = KEEP_FULL)', flush=True)
    verify_frozen()


if __name__ == '__main__':
    main()
