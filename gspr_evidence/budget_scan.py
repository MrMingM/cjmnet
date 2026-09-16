"""只改 A0B0 的预算，同一帧的各档位共用编码结果。"""
import argparse
import copy
import json
import time
from pathlib import Path

BUDGETS = [262144, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432]
METRICS = ('ap30', 'ap50', 'ap70')


def passes(row, full):
    # 比原始精度，不靠四舍五入抹掉下降。
    return all(row[key] >= full[key] for key in METRICS)


def choose_budget(summaries):
    if set(summaries) != {'clean', 'fog', 'rain', 'snow'}:
        raise ValueError('四种天气都完成后才能选预算')
    fingerprints = {json.dumps(x['comparison_contract'], sort_keys=True) for x in summaries.values()}
    if len(fingerprints) != 1:
        raise ValueError('四种天气的模型、配置或评测协议不同')
    if any(x['phase'] != 'development' for x in summaries.values()):
        raise ValueError('不能拿 test 结果自动挑预算')
    sets = [{int(k) for k in s['results'] if k.isdigit()} for s in summaries.values()]
    if any(x != sets[0] for x in sets):
        raise ValueError('四种天气的预算档位不同')
    accepted = [b for b in sorted(sets[0]) if all(passes(s['results'][str(b)], s['results']['full']) for s in summaries.values())]
    return dict(selected_budget_bytes=accepted[0] if accepted else None,
                selected_mode='a0b0' if accepted else 'full', passing_budgets=accepted,
                criterion='All 12 AP values >= same-weather full, without rounding or allowed drop',
                comparison_contract=next(iter(summaries.values()))['comparison_contract'],
                reason='Smallest tested passing budget' if accepted else 'No tested sparse budget passed; retain full communication')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gspr_evidence/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--phase', choices=('development', 'benchmark'), required=True)
    parser.add_argument('--weather', choices=('clean', 'fog', 'rain', 'snow'), required=True)
    parser.add_argument('--selection', help='正式测试只读开发阶段选好的预算')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    import torch
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from . import runtime as rt
    from .benchmark import load_config as benchmark_config
    rt.verify_frozen()
    formal = args.phase == 'benchmark'
    options, hypes = (benchmark_config(args.config, args.frontend_config, args.weather) if formal
                      else rt.load_config(args.config, args.frontend_config))
    if not formal and Path(hypes['validate_dir']).resolve() != Path('/data/scd/datasets/opv2v_official_data_dumping/validate').resolve():
        raise ValueError('开发扫描只能使用完整 validation')
    rt.seed_all(options['seed'])
    device = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, device)
    contract = rt.contract(options, args.frontend_config, digest)
    # 预算扫描文件也留指纹，避免拿不同版本结果拼表。
    contract['budget_scan_source'] = rt.sha256(__file__)
    budgets = BUDGETS
    selection = None
    if formal:
        if not args.selection:
            parser.error('benchmark 必须提供 development 的 selection.json')
        selection = json.loads(Path(args.selection).read_text())
        if selection['comparison_contract'] != contract:
            raise ValueError('开发选择与正式测试的前端/源码/训练配置不一致')
        b = selection['selected_budget_bytes']
        if (selection['selected_mode'] == 'full') != (b is None) or (b is not None and b not in BUDGETS):
            raise ValueError('选择文件中的预算无效')
        budgets = [] if b is None else [b]
    elif args.selection:
        parser.error('development 不读取 selection')
    branch = 'clean' if formal else args.weather
    rt.seed_all(options['seed'])
    ds, loader, indices = rt.make_loader(hypes, options, weather=branch)
    out = rt.new_output(args.output_dir)
    import yaml
    (out/'resolved_experiment.yaml').write_text(yaml.safe_dump(options, sort_keys=False), encoding='utf-8')
    modes = ['full', 'original_full']+[str(b) for b in budgets]
    protocol = dict(phase=args.phase, weather=args.weather, data_root=hypes['validate_dir'],
                    online_weather=not formal and args.weather != 'clean', global_sort=False,
                    sample_indices=indices, scene_ends=ds.len_record, budget_bytes=budgets,
                    comparison_contract=contract, selection=selection, full_frames=True,
                    note='Non-global AP in BOTH phases; validation simulation is NOT OPV2V-W test',
                    timing='Shared encoding plus per-budget packet/fusion; excludes loading and postprocessing')
    rt.write_json(out/'protocol.json', protocol)
    stats = {m: {t: dict(tp=[], fp=[], gt=0, score=[]) for t in (.3, .5, .7)} for m in modes}
    totals = {m: dict(bytes=0, peak_bytes=0, request_bytes=0, feature_bytes=0, selected_blocks=0, saturated_frames=0) for m in modes}
    start = time.perf_counter()
    print(f'{args.phase}/{args.weather}: {hypes["validate_dir"]}; ALL {len(indices)} frames; budgets={budgets}; global_sort=False', flush=True)
    with torch.no_grad(), (out/'frames.jsonl').open('w') as stream:
        for number, batch in enumerate(loader, 1):
            batch = to_device(batch, device)
            ego = batch['ego']
            inp = rt.input_branch(ego, branch)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            encoded = model.encode(inp)
            torch.cuda.synchronize()
            encode_ms = (time.perf_counter()-tick)*1000
            for mode in modes:
                torch.cuda.synchronize()
                tick = time.perf_counter()
                if mode == 'original_full':
                    raw = model.engine.base(inp)
                    prediction = {key: raw[key] for key in ('psm', 'rm')}
                    del raw
                    for key in prediction:
                        torch.testing.assert_close(prediction[key], full_prediction[key], atol=2e-4, rtol=2e-4)
                    diag = copy.deepcopy(full_diag)
                else:
                    if mode != 'full':
                        model.engine.budget = int(mode)
                    prediction, diag = model.run(encoded, 'full' if mode == 'full' else 'a0b0')
                    if mode == 'full':
                        full_prediction, full_diag = prediction, copy.deepcopy(diag)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter()-tick)*1000
                boxes, scores, gt = ds.post_process(batch, {'ego': prediction})
                if mode == 'full':
                    full_gt = gt
                else:
                    torch.testing.assert_close(gt, full_gt)
                per_frame = {t: dict(tp=[], fp=[], gt=0, score=[]) for t in (.3, .5, .7)}
                for threshold in per_frame:
                    eval_utils.caluclate_tp_fp(boxes, scores, gt, per_frame, threshold)
                    stats[mode][threshold]['gt'] += per_frame[threshold]['gt']
                    for key in ('tp', 'fp', 'score'):
                        stats[mode][threshold][key].extend(per_frame[threshold][key])
                total = totals[mode]
                total['bytes'] += diag['total_bytes']
                total['peak_bytes'] = max(total['peak_bytes'], diag['total_bytes'])
                total['request_bytes'] += diag['request_bytes']
                total['feature_bytes'] += diag['feature_bytes']
                total['selected_blocks'] += diag['selected_blocks']
                total['saturated_frames'] += int(diag['selected_blocks'] == full_diag['selected_blocks'])
                stream.write(json.dumps(dict(sample_index=int(ego['communication_sample_index'][0]), mode=mode,
                    bytes=diag['total_bytes'], selected_blocks=diag['selected_blocks'], encode_ms=encode_ms,
                    packet_fusion_ms=elapsed_ms, ap_inputs=per_frame))+'\n')
            if number == 1 or number % 20 == 0:
                elapsed = time.perf_counter()-start
                print(f'{args.phase}/{args.weather} {number}/{len(indices)}; {elapsed/number:.2f}s/frame; '
                      f'ETA {(len(indices)-number)*elapsed/number/3600:.2f}h', flush=True)
                stream.flush()
    if number != len(indices) or not stats['full'][.7]['gt']:
        raise RuntimeError('评测未完整完成或没有 GT')
    results = {}
    for mode in modes:
        folder = out/mode
        folder.mkdir()
        # 非全局 AP 会就地改列表，所以给它副本。
        eval_utils.eval_final_results(copy.deepcopy(stats[mode]), str(folder), False)
        ap = [float(eval_utils.calculate_ap(copy.deepcopy(stats[mode]), t, False)[0]) for t in (.3, .5, .7)]
        results[mode] = dict(zip(METRICS, ap))
        results[mode].update(frames=number, mean_bytes=totals[mode]['bytes']/number,
            peak_bytes=totals[mode]['peak_bytes'], mean_feature_bytes=totals[mode]['feature_bytes']/number,
            mean_request_bytes=totals[mode]['request_bytes']/number,
            saturated_fraction=totals[mode]['saturated_frames']/number)
        results[mode]['passes_full'] = passes(results[mode], results.get('full', results[mode]))
    rt.write_json(out/'summary.json', dict(phase=args.phase, weather=args.weather, comparison_contract=contract, results=results))
    rt.verify_frozen()
    print(f'Complete: {out}/summary.json', flush=True)


if __name__ == '__main__':
    main()
