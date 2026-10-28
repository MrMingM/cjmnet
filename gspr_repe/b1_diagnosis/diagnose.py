"""CPU checkpoint inspection and saved F traces only; never import a detector."""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

WEATHERS = ('clean', 'fog', 'snow')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def check_identity(row, field='identity'):
    require(field in row and identity({k: v for k, v in row.items() if k != field}) == row[field],
            'JSON identity mismatch: ' + field)


class Reader:
    def __init__(self):
        self.files = {}

    def track(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        require(not any(p.lower() in ('test', 'testing') for p in path.parts), 'Test access refused')
        value = digest(path)
        require(expected is None or value == expected, 'SHA256 mismatch: ' + str(path))
        require(str(path) not in self.files or self.files[str(path)] == value, 'Input changed during read')
        self.files[str(path)] = value
        return path

    def json(self, path, expected=None):
        return json.loads(self.track(path, expected).read_text(encoding='utf-8'))

    def lines(self, path, expected=None):
        return [json.loads(line) for line in self.track(path, expected).read_text(encoding='utf-8').splitlines() if line.strip()]

    def unchanged(self):
        for path, value in self.files.items():
            require(digest(path) == value, 'Input modified during diagnosis: ' + path)


def save(output, name, value):
    target = output / name
    temporary = target.with_suffix(target.suffix + '.tmp')
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary.write_text(text, encoding='utf-8', newline='\n')
    os.replace(temporary, target)


def load_model_evidence(root, variant, reader):
    folder = root / variant
    selected = reader.json(folder / 'selected.json')
    check_identity(selected)
    require(selected['status'] == 'complete' and selected['weather_used'] is False, 'Invalid selection: ' + variant)
    contract = reader.json(folder / 'contract.json', selected['training_contract_sha256'])
    check_identity(contract)
    require(contract['variant'] == variant and not contract['smoke_only'] and contract['epochs'] == 20,
            'Wrong model/training contract: ' + variant)
    reader.track(folder / 'config.yaml', contract['config_sha256'])
    rows = reader.json(folder / 'history.json')['epochs']
    require([r['epoch'] for r in rows] == list(range(1, 21)), 'Need 20 contiguous epochs: ' + variant)
    for row in rows:
        commit = reader.json(folder / 'epochs' / ('%02d' % row['epoch']) / 'complete.json')
        require(row == commit and row['status'] == 'complete' and row['contract'] == contract['identity'],
                'History/epoch commit mismatch: ' + variant)
        ap = reader.json(row['clean_summary'], row['clean_summary_sha256'])
        require(ap['status'] == 'complete' and ap['full_validation'] and ap['frame_count'] == 829
                and ap['conditions'] == ['clean'] and ap['checkpoint_sha256'] == row['checkpoint_sha256']
                and ap['results']['clean']['AP']['O']['ap70'] == row['AP70'], 'Clean selection evidence mismatch')
    best = min(rows, key=lambda r: (-r['AP70'], r['epoch']))
    require(selected['epoch'] == best['epoch'] and selected['checkpoint_sha256'] == best['checkpoint_sha256']
            and selected['clean_O_AP70'] == best['AP70'], 'Selected weight/epoch mismatch')
    reader.track(selected['checkpoint'], selected['checkpoint_sha256'])
    evaluation = folder / 'final_evaluation'
    summary = reader.json(evaluation / 'summary.json')
    # Anchor completed evaluations and selections to the pipeline's saved hashes.
    for stage, paths in ((variant+'_train', [folder/'selected.json', folder/'contract.json', folder/'history.json']),
                         (variant+'_evaluate', [evaluation/'summary.json'])):
        journal = reader.json(root/'stages'/stage/'complete.json')
        require(journal['status'] == 'complete' and journal['stage'] == stage, 'Pipeline stage incomplete: '+stage)
        for path in paths:
            require(str(path) in journal['artifacts'], 'Stage artifact binding missing: '+str(path))
            reader.track(path,journal['artifacts'][str(path)])
    require(summary['status'] == 'complete' and summary['full_validation'] and summary['frame_count'] == 829
            and summary['conditions'] == list(WEATHERS)
            and summary['checkpoint_sha256'] == selected['checkpoint_sha256'], 'Incomplete/wrong evaluation: ' + variant)
    require(summary.get('model_variant') == variant, 'Evaluation model identity missing/wrong: ' + variant)
    for name, value in summary['output_hashes'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe evaluation manifest path')
        reader.track(evaluation / name, value)
    ec = reader.json(evaluation / 'contract.json')
    check_identity(ec)
    require(ec['identity'] == summary['contract_identity'] and ec['checkpoint_sha256'] == selected['checkpoint_sha256']
            and ec['source_hashes']['variant'] == variant
            and ec['source_hashes']['sources'] == contract['sources'], 'Evaluation source/weight mismatch')
    require(ec['associated_d1d_training']['training_contract_sha256'] == selected['training_contract_sha256'],
            'Evaluation associated training mismatch')
    require(contract['frozen_gt'] == ec['frozen_d1c_gt_identity'] == summary['frozen_d1c_gt_identity']
            and contract['input_identity'] == ec['input_manifest_identity'], 'Training/evaluation GT or input identity mismatch')
    protocol = reader.json(evaluation / 'protocol.json')
    require(protocol['identity'] == ec['identity'] and protocol['checkpoint_sha256'] == ec['checkpoint_sha256']
            and protocol['protocols'] == ec['protocols'], 'Protocol completion mismatch')
    gt = reader.json(evaluation / 'gt_manifest.json')
    check_identity(gt)
    require(gt['identity'] == summary['gt_identity'] and gt['contract_identity'] == ec['identity']
            and gt['definition'] == ec['protocols'], 'GT manifest identity mismatch')
    prediction = reader.json(evaluation / 'prediction_manifest.json')
    check_identity(prediction)
    require(prediction['identity'] == summary['prediction_cache_identity']
            and prediction['contract_identity'] == ec['identity'], 'Prediction manifest identity mismatch')
    frames = reader.lines(evaluation / 'per_frame_metrics.jsonl')
    keys = ec['frame_keys']
    require(len(keys) == 829 and len({tuple(k) for k in keys}) == 829, 'Invalid frame order/duplicates')
    expected = [(w, i) for w in WEATHERS for i in range(829)]
    require([(r['condition'], r['index']) for r in frames] == expected, 'Incomplete/reordered per-frame traces')
    require([(r['condition'], r['index']) for r in gt['frames']] == expected, 'Incomplete/reordered GT manifest')
    require([(r['condition'], r['index']) for r in prediction['files']] == expected, 'Incomplete/reordered cache manifest')
    cached = {}
    for entry in prediction['files']:
        w, i = entry['condition'], entry['index']
        require(Path(entry['path']).resolve() == (evaluation / 'predictions' / w / ('%06d.json' % i)).resolve(),
                'Unexpected prediction cache location')
        row = reader.json(entry['path'], entry['sha256'])
        check_identity(row, 'record_identity')
        require(row['contract_identity'] == ec['identity'] and row['condition'] == w
                and row['index'] == i and row['frame_key'] == keys[i], 'Cache frame/source identity mismatch')
        scores = row['scores'] or []
        require(len(scores) == row['prediction_count'] and all(math.isfinite(x) for x in scores), 'Invalid scores')
        require(len(row['prediction_boxes'] or []) == len(scores), 'Prediction boxes/count mismatch')
        cached[(w, i)] = row
    for frame, gr in zip(frames, gt['frames']):
        w, i = frame['condition'], frame['index']
        require(frame['frame_key'] == gr['frame_key'] == keys[i], 'Trace frame identity mismatch')
        require(frame['prediction_cache_sha256'] == frame['protocols']['F']['prediction_cache_sha256']
                == reader.files[str((evaluation / 'predictions' / w / ('%06d.json' % i)).resolve())], 'Trace/cache hash mismatch')
        fixed = cached[('clean', i)]['official_gt']
        require(identity(fixed) == gr['fixed_gt_identity'] and [x['object_id'] for x in fixed] == gr['fixed_ids'], 'Fixed Clean GT mismatch')
        require(identity(cached[(w, i)]['official_gt']) == gr['official_gt_identity'], 'Weather GT mismatch')
        require(frame['protocols']['F']['skipped'] is False, 'F must count None frames')
        require(len({x['object_id'] for x in fixed}) == len(fixed), 'Duplicate GT object ID')
        for threshold in ('0.3', '0.5', '0.7'):
            trace_matches(frame, threshold, cached[(w, i)], {x['object_id'] for x in fixed})
    stats = {}
    for w in WEATHERS:
        stat = reader.json(evaluation / ('stat_' + w + '_F.json'))['0.7']
        fs = [f for f in frames if f['condition'] == w]
        for name in ('tp', 'fp'):
            require(stat[name] == [x for f in fs for x in f['protocols']['F']['thresholds']['0.7'][name + '_flags']], 'Saved F statistic/trace mismatch')
        require(len(stat['score']) == len(stat['tp']) == len(stat['fp'])
                and all(math.isfinite(x) for x in stat['score']), 'Invalid saved score statistics')
        require(stat['gt'] == sum(f['fixed_gt_count'] for f in fs) == gt['fixed_total_per_condition'], 'F GT denominator mismatch')
        stats[w] = stat
    return dict(selected=selected, contract=contract, summary=summary, evaluation_contract=ec,
                protocol=protocol, gt=gt, cached=cached, frames=frames, rows=rows, stats=stats)


def trace_matches(frame, threshold, cache, ids):
    """Use exactly diagnostics.py's saved F matched_pairs, not a new IoU matcher."""
    f = frame['protocols']['F']
    require(threshold in f.get('matched_pairs', {}), 'Missing saved F matches: ' + str((frame['condition'], frame['index'], threshold)))
    pairs = f['matched_pairs'][threshold]
    counts = f['thresholds'][threshold]
    matched = {p['object_id']: p for p in pairs}
    require(len(matched) == len(pairs) == counts['tp'] and set(matched) <= ids, 'Invalid saved target matches')
    indices = [p['prediction_index'] for p in pairs]
    require(len(set(indices)) == len(indices) and all(0 <= i < cache['prediction_count'] for i in indices), 'Non-one-to-one prediction indices')
    require(all(math.isfinite(p['iou']) and p['iou'] >= float(threshold) for p in pairs), 'Invalid saved match IoU')
    require(counts['fn'] == len(ids) - len(pairs) and counts['tp'] + counts['fp'] == cache['prediction_count'], 'Frame count mismatch')
    for name in ('tp', 'fp'):
        require(name + '_flags' in counts and len(counts[name + '_flags']) == cache['prediction_count']
                and all(x in (0, 1) for x in counts[name + '_flags'])
                and sum(counts[name + '_flags']) == counts[name], 'Missing/inconsistent saved flags')
    require(all(t+f == 1 for t,f in zip(counts['tp_flags'],counts['fp_flags'])), 'Invalid TP/FP flag pairs')
    return matched


def verify_pair(models):
    a, b = models['B0'], models['B1']
    for key in ('settings', 'policy', 'train_manifest_sha256', 'train_manifest_identity',
                'pair_identity', 'config_sha256', 'input_identity', 'frozen_gt', 'sources'):
        require(a['contract'][key] == b['contract'][key], 'Training comparability mismatch: ' + key)
    for key in ('frame_keys', 'conditions', 'protocols', 'inference_seed', 'official_config_sha256',
                'official_source_hashes', 'frozen_d1c_gt_identity', 'input_manifest_identity', 'd1b'):
        require(a['evaluation_contract'][key] == b['evaluation_contract'][key], 'Evaluation comparability mismatch: ' + key)
    # A manifest's own identity includes its checkpoint-specific contract. Compare
    # frozen GT identity and actual geometry/order, not checkpoint-bound hashes.
    for key in ('frozen_d1c_gt_identity',):
        require(a['summary'][key] == b['summary'][key], 'GT identity mismatch: ' + key)
    require(a['gt']['frames'] == b['gt']['frames'], 'Per-frame GT/order/geometry mismatch')
    require(a['gt']['definition']['F']['name'] == 'OFFICIAL_CLEAN_FIXED_GT', 'Need reviewed F protocol')
    return dict(status='passed',models=['B0','B1'],protocol='F',threshold=0.7,frames_per_weather=829,
                frozen_gt_identity=a['summary']['frozen_d1c_gt_identity'],
                fixed_gt_frames_identity=identity(a['gt']['frames']),
                frame_order='identical',source_identity='identical',
                matching='Existing official F matched_pairs from diagnostics.py; no new matching/AP/inference')


def check_source(project, models, reader):
    recorded = models['B1']['contract']['sources']
    for rel, expected in recorded['files'].items():
        reader.track(project / 'gspr_repe' / rel, expected)
    for rel, expected in recorded['dependencies'].items():
        reader.track(project / rel, expected)
    for rel, expected in recorded['pointcloud_tools'].items():
        reader.track(project / rel, expected)
    # No import: extracting the initializer cannot construct/run a model.
    path = project / 'gspr_repe/modules.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Aggregation')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    node = next(n.value for n in ast.walk(init) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == 'gate' for t in n.targets))
    value = ast.literal_eval(node.args[0].args[0])
    require(value == -5., 'Aggregation gate initializer changed; review required')
    return value


def sigmoid(x):
    require(math.isfinite(x), 'Nonfinite scalar gate')
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def gate_from_state(path, torch, full=False):
    state = torch.load(path, map_location='cpu', weights_only=not full)
    if full:
        state = state['model']
    require(isinstance(state, dict), 'Expected checkpoint state dictionary')
    require(all(not hasattr(t, 'device') or t.device.type == 'cpu' for t in state.values()), 'CPU load required')
    keys = [k for k in state if k in ('repe_a.gate', 'module.repe_a.gate')]
    require(len(keys) == 1 and state[keys[0]].numel() == 1, 'Missing/ambiguous scalar repe_a.gate')
    require(not any(k.startswith(('repe_b.', 'repe_c.', 'module.repe_b.', 'module.repe_c.')) for k in state), 'B1 checkpoint contains B/C modules')
    value = float(state[keys[0]].item())
    return dict(key=keys[0],raw_gate=value,mixing_ratio=sigmoid(value),weighted_branch_percent=100*sigmoid(value))


def analyze_gate(root, model, reader, initial, baseline=None):
    import torch  # Only here, on Linux server; no model/native/GPU imports.
    torch.set_num_threads(1)
    selected = model['selected']
    if baseline is not None:
        b0 = torch.load(baseline['selected']['checkpoint'], map_location='cpu', weights_only=True)
        require(isinstance(b0, dict) and not any(k.startswith(('repe_a.','repe_b.','repe_c.','module.repe_a.','module.repe_b.','module.repe_c.')) for k in b0),
                'B0 selected checkpoint contains REPE modules')
        del b0
    final = gate_from_state(selected['checkpoint'], torch)
    trajectory, missing = [], []
    for epoch in (1, 10, 20):
        row = model['rows'][epoch - 1]
        path = Path(row['checkpoint'])
        if not path.exists():
            missing.append(str(path)); continue
        reader.track(path, row['checkpoint_sha256'])
        trajectory.append(dict(epoch=epoch,path=str(path),sha256=row['checkpoint_sha256'],**gate_from_state(path, torch)))
    observed_initial = None
    path = root / 'B1/resume_states/00.pth'
    marker = path.with_suffix('.json')
    if path.exists() and marker.exists():
        metadata = reader.json(marker)
        reader.track(path, metadata['sha256'])
        observed_initial = gate_from_state(path, torch, full=True)
        require(observed_initial['raw_gate'] == initial, 'Initial saved gate disagrees with source')
    return dict(status='complete',selected_epoch=selected['epoch'],checkpoint=selected['checkpoint'],
                checkpoint_sha256=selected['checkpoint_sha256'],source_initial_gate=initial,
                initial_mixing_ratio=sigmoid(initial),saved_initial=observed_initial,final=final,
                trajectory=trajectory,optional_missing=missing,device='cpu',
                formula='h=(1-alpha)*h_masked_max+alpha*h_adaptive, alpha=sigmoid(gate)',
                limitations=['alpha is a scalar interpolation coefficient, not FLOPs, activation-norm contribution or causal gain.',
                             'Both branches are computed regardless of gate; cached detections do not store their activation norms.',
                             'B1 also bypasses B0 decorated-feature r multiplication and changes valid-only PFN/normalization/max pooling.',
                             'Small alpha does not imply B1 equals B0; no standalone A causal effect is identified.'])


def optional_geometry_evidence(root, models, reader):
    run = reader.json(root / 'run_identity.json')
    path = Path(run['roots']['gt_audit_root']) / 'per_frame.jsonl'
    expected = models['B0']['evaluation_contract']['d1b'].get('per_frame.jsonl')
    if not path.exists():
        return None, '未找到原标注审计per_frame.jsonl；点数和车辆覆盖无法分析：' + str(path)
    require(expected is not None, 'No recorded hash for GT point-count evidence')
    evidence = reader.lines(path, expected)
    require([x['frame_key'] for x in evidence] == models['B0']['evaluation_contract']['frame_keys'], 'Point evidence frame order mismatch')
    for i, frame in enumerate(evidence):
        # Audit's pre-filter observations and cached live GT can differ in small
        # floating point rounding. Preserve the existing support.compare_gt tolerance.
        actual={x['object_id']:x for x in frame['conditions']['clean']['final_gt']}
        cached={x['object_id']:x for x in models['B0']['cached'][('clean',i)]['official_gt']}
        require(len(actual)==len(frame['conditions']['clean']['final_gt']) and actual.keys()==cached.keys(), 'Point evidence GT IDs mismatch')
        for oid in actual:
            a,b=actual[oid]['corners'],cached[oid]['corners']
            require(len(a)==len(b)==8 and all(len(x)==3 for x in a+b), 'Invalid point-evidence GT corners')
            require(all(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-4
                        for ac,bc in zip(a,b) for x,y in zip(ac,bc)), 'Point evidence GT geometry mismatch')
    return evidence, None


def target_metadata(gt, current):
    corners = gt['corners']
    distance = math.hypot(*(sum(c[k] for c in corners)/len(corners) for k in (0,1)))
    require(math.isfinite(distance), 'Invalid GT geometry')
    counts = None
    if current is not None and 'cavs' in current:
        counts = []
        for cav in current['cavs']:
            if 'before_point_filter_ids' not in cav or 'official_points_in_box' not in cav:
                counts = None; break
            ids, values = cav['before_point_filter_ids'], cav['official_points_in_box']
            require(len(ids) == len(values), 'GT point-count lengths mismatch')
            if gt['object_id'] in ids:
                value = values[ids.index(gt['object_id'])]
                require(value >= 0, 'Negative GT point count')
                counts.append(value)
    count = sum(counts) if counts is not None else None
    return dict(distance_m=distance,points_sum_across_cavs=count,
                observing_cavs=sum(x>0 for x in counts) if counts is not None else None,
                distance_bin='0-30' if distance<30 else '30-60' if distance<60 else '60+',
                point_bin='missing' if count is None else '0' if count==0 else '1-5' if count<=5 else '6-20' if count<=20 else '21+',
                coverage_bin='missing' if counts is None else '0' if not any(counts) else '1' if sum(x>0 for x in counts)==1 else '2+')


def summary_scores(stat):
    result = {}
    for name, flags in (('tp',stat['tp']),('fp',stat['fp'])):
        scores = [s for s, flag in zip(stat['score'], flags) if flag]
        result[name] = dict(count=len(scores),mean=statistics.mean(scores) if scores else None,
                            median=statistics.median(scores) if scores else None)
    return result


def analyze_targets(models, evidence):
    a, b = models['B0'], models['B1']
    frame_maps = {v:{(r['condition'],r['index']):r for r in models[v]['frames']} for v in models}
    result = {}
    for weather in WEATHERS:
        targets, changes = [], []
        bins = defaultdict(lambda: dict(gt=0,b0_tp=0,b1_tp=0,lost=0,recovered=0))
        total = dict(gt=0,b0_tp=0,b1_tp=0,b0_fp=0,b1_fp=0,lost=0,recovered=0)
        for i, key in enumerate(a['evaluation_contract']['frame_keys']):
            fs = [frame_maps[v][(weather,i)] for v in ('B0','B1')]
            caches = [models[v]['cached'][(weather,i)] for v in ('B0','B1')]
            fixed = a['cached'][('clean',i)]['official_gt']
            ids = {g['object_id'] for g in fixed}
            matches = [trace_matches(f,'0.7',c,ids) for f,c in zip(fs,caches)]
            lower = [{t:trace_matches(f,t,c,ids) for t in ('0.3','0.5')} for f,c in zip(fs,caches)]
            lost, recovered = set(matches[0])-set(matches[1]),set(matches[1])-set(matches[0])
            counts = [f['protocols']['F']['thresholds']['0.7'] for f in fs]
            total['gt']+=len(ids)
            for j,v in enumerate(('b0','b1')):
                total[v+'_tp']+=counts[j]['tp'];total[v+'_fp']+=counts[j]['fp']
            total['lost']+=len(lost);total['recovered']+=len(recovered)
            changes.append(dict(frame_key=key,index=i,lost=sorted(lost),recovered=sorted(recovered),
                                tp_delta=counts[1]['tp']-counts[0]['tp'],fp_delta=counts[1]['fp']-counts[0]['fp']))
            current = evidence[i]['conditions'][weather] if evidence is not None else None
            for gt in fixed:
                oid=gt['object_id'];meta=target_metadata(gt,current)
                status='lost' if oid in lost else 'recovered' if oid in recovered else 'both_correct' if oid in matches[0] else 'both_missed'
                for group in ('distance_bin','point_bin','coverage_bin'):
                    row=bins[(group,meta[group])];row['gt']+=1;row['b0_tp']+=oid in matches[0];row['b1_tp']+=oid in matches[1]
                    row['lost']+=oid in lost;row['recovered']+=oid in recovered
                if status not in ('lost','recovered'):
                    continue
                details={}
                for j,v in enumerate(('B0','B1')):
                    item={}
                    for t in ('0.3','0.5','0.7'):
                        pair=(matches[j] if t=='0.7' else lower[j][t]).get(oid)
                        item[t]=None if pair is None else dict(**pair,score=caches[j]['scores'][pair['prediction_index']])
                    details[v]=item
                targets.append(dict(frame_key=key,index=i,object_id=oid,status=status,**meta,matches=details,
                                    fixed_gt=gt,cache_reference=dict(condition=weather,index=i,
                                    filename='predictions/'+weather+('/%06d.json'%i))))
        total.update(tp_net=total['b1_tp']-total['b0_tp'],fp_net=total['b1_fp']-total['b0_fp'],
                     b0_fn=total['gt']-total['b0_tp'],b1_fn=total['gt']-total['b1_tp'])
        require(total['tp_net']==total['recovered']-total['lost'],'Target net/count mismatch')
        for v, name in (('B0','b0'),('B1','b1')):
            stat=models[v]['stats'][weather]
            require(sum(stat['tp'])==total[name+'_tp'] and sum(stat['fp'])==total[name+'_fp'] and stat['gt']==total['gt'], 'Aggregate F statistics mismatch')
        strata=[]
        for (group,label),row in sorted(bins.items()):
            strata.append(dict(group=group,bin=label,**row,
                loss_rate_among_b0_correct=row['lost']/row['b0_tp'] if row['b0_tp'] else None,
                recovery_rate_among_b0_missed=row['recovered']/(row['gt']-row['b0_tp']) if row['gt']>row['b0_tp'] else None,
                b0_recall=row['b0_tp']/row['gt'],b1_recall=row['b1_tp']/row['gt']))
        ap={v:models[v]['summary']['results'][weather]['AP']['F'] for v in models}
        result[weather]=dict(counts=total,AP=ap,ap70_delta_pp=100*(ap['B1']['ap70']-ap['B0']['ap70']),
            strata=strata,target_changes=targets,per_frame=changes,
            examples=sorted([t for t in targets if t['status']=='lost'],key=lambda t:(-(t['matches']['B0']['0.7']['iou']),str(t['frame_key']),str(t['object_id'])))[:8],
            score_distributions={v:summary_scores(models[v]['stats'][weather]) for v in models})
    return dict(status='complete',protocol='F',iou_threshold=0.7,results=result,
        metadata_definitions='Distance and bins follow diagnostics.py. Point counts sum native per-CAV box counts (not unique points); coverage is number of CAVs with >0 box points, not spatial coverage.',
        limits=['Only cached post-NMS detections; cannot inspect discarded pre-NMS boxes or feature activations.',
                'Unmatched at .7 does not prove object absent; .3/.5 matches provide limited localization evidence.',
                'No new AP or IoU calculation. Official AP uses per-frame score ordering/concatenation, not cross-frame global sort.',
                'TP/FP counts, score distributions and strata are descriptive; none uniquely attributes the AP change.'])


def report(gate, targets, warnings):
    alpha=gate['final']['mixing_ratio']
    lines=['# B1 模块 A 只读诊断','', '本报告来自已保存权重和评测缓存；未训练、未推理、未修改原产物。',
           '', '## 1. 聚合门实际是多少', '',
           f"选中第{gate['selected_epoch']}轮。初始gate={gate['source_initial_gate']}，混合比例={gate['initial_mixing_ratio']:.8f}；最终gate={gate['final']['raw_gate']:.8f}，混合比例={alpha:.8f}（{100*alpha:.6f}%）。",
           '这表示 h=(1−α)×有效点最大池化特征+α×自适应聚合特征。两个分支都会计算，不能说只执行了这么多百分比的计算。实际特征改变量还取决于两分支数值，缓存没有激活值，无法计算。',
           'B1还改变了GSPR装饰特征加权、PFN有效点归一化和最大池化；gate小不能推出B1接近B0，也不能把gate当作A的独立贡献。',
           '', '| 轮次 | gate | 混合比例 |','|---|---:|---:|']
    for row in gate['trajectory']:
        lines.append(f"| {row['epoch']} | {row['raw_gate']:.8f} | {row['mixing_ratio']:.8f} |")
    lines+=['','## 2. 固定Clean标注下的正确目标变化','',
            '以下为F协议、官方平面框交并比≥0.7，目标用frame_key+object_id对齐。',
            '| 天气 | B0正确 | B1正确 | 丢失 | 恢复 | 正确净变化 | B0误检 | B1误检 | AP70差/百分点 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for w,row in targets['results'].items():
        c=row['counts']
        lines.append(f"| {w} | {c['b0_tp']} | {c['b1_tp']} | {c['lost']} | {c['recovered']} | {c['tp_net']:+} | {c['b0_fp']} | {c['b1_fp']} | {row['ap70_delta_pp']:+.6f} |")
    fog=targets['results']['fog'];c=fog['counts']
    if c['tp_net']<0 and c['fp_net']>0:
        pattern='正确目标净丢失和误检增加同时存在，无法只凭数量判断哪项主导AP下降。'
    elif c['tp_net']<0:
        pattern='从数量看主要出现正确目标净丢失，误检没有净增加；排序影响仍未排除。'
    elif c['fp_net']>0:
        pattern='正确目标没有净减少，但误检增加；误检和排序应继续检查，不能仅凭计数分摊AP影响。'
    else:
        pattern='总正确/误检数量未朝更差方向变化；若AP仍下降，数量不足以解释，应关注正确检测分布及原评测序列中的排序，尚未证明具体原因。'
    observations=[]
    if c['tp_net']<0:observations.append(f"正确检测净减少{-c['tp_net']}个目标帧")
    elif c['tp_net']>0:observations.append(f"正确检测净增加{c['tp_net']}个目标帧")
    else:observations.append('正确检测净数量不变')
    observations.append(f"误检净变化{c['fp_net']:+}个预测框")
    lines+=['', '已证实的Fog表现：'+'；'.join(observations)+'。'+pattern+'这两类数量不宜用比例直接比较其AP贡献。',
            'AP还取决于保存预测的分数顺序及正确检测在序列中的位置。即便数量变差，也不能排除排序同时影响；当前只给出已匹配正确/误检预测的分数分布，不把它解释成排序原因已证实。',
            '', '### Fog分层（必须看分母和丢失率）','',
            '| 分组 | 区间 | 总目标 | B0正确 | 丢失 | 恢复 | B0正确中的丢失率 |','|---|---|---:|---:|---:|---:|---:|']
    for r in fog['strata']:
        rate='无分母' if r['loss_rate_among_b0_correct'] is None else f"{r['loss_rate_among_b0_correct']:.2%}"
        lines.append(f"| {r['group']} | {r['bin']} | {r['gt']} | {r['b0_tp']} | {r['lost']} | {r['recovered']} | {rate} |")
    lines.append('')
    for group,label in (('distance_bin','距离'),('point_bin','点数'),('coverage_bin','有点车辆数')):
        rows=[r for r in fog['strata'] if r['group']==group and r['bin']!='missing' and r['loss_rate_among_b0_correct'] is not None]
        if rows:
            high=max(rows,key=lambda r:r['loss_rate_among_b0_correct'])
            lines.append(f"{label}分层中，B0正确目标的丢失率最高区间为{high['bin']}：{high['lost']}/{high['b0_tp']}={high['loss_rate_among_b0_correct']:.2%}。这只是当前样本的描述，不构成显著性或原因判断；分母小时尤其不能推广。")
        if any(r['group']==group and r['bin']=='missing' for r in fog['strata']):
            lines.append(label+'存在缺失记录，不能对全部受损目标声称完整覆盖。')
    lines+=['', '距离为Clean框中心到原点的水平距离；点数为各车原生框内点数之和，可能重复观测同一表面。coverage仅为有点车辆数代理，未测量网格空间覆盖率。分层是描述性对照，未控制距离、点数等混杂因素。',
            '', '### 少量受损目标（按B0匹配IoU降序选取，非随机代表样本）','']
    for t in fog['examples']:
        lines.append(f"- 场景/帧={t['frame_key']}，目标={t['object_id']}，距离={t['distance_m']:.2f}米，点数={t['points_sum_across_cavs']}，有点车辆={t['observing_cavs']}；匹配证据：`{json.dumps(t['matches'],ensure_ascii=False)}`。")
    lines+=['', '完整目标列表、两模型0.3/0.5/0.7匹配及对应分数见TARGET_CHANGES.json。',
            '', '## 3. 能否判断是A无效或替换路径有问题','',
            '已证实：两模型权重、GT身份、帧序及协议已核验；以上只描述选中模型的差异。',
            '合理怀疑：B1同时改变聚合和原GSPR/PFN/归一化路径，路径替换是必须排查的混杂因素。',
            '尚未验证：现有B0/B1对照无法区分“自适应聚合本身无效”和“替换路径造成退化”，也不能确认单一根因；本诊断不偏向未经隔离的因果结论。',
            '', '## 4. 下一步唯一建议的最小对照（本次不执行）','',
            '增加一组“B1路径保留、聚合残差恒为0”：保留B1的GSPR调用、有效点PFN、归一化和最大池化，只令输出h=h_masked_max，按同一训练配方从匹配随机初始化训练。与B0比较可检查路径替换影响；与现有B1比较可检查在相同替换路径上加入自适应分支的影响。不能直接把B1训练后权重gate清零的推理当作等价训练对照。本次只提出，不修改模型、不启动该实验。',
            '', '## 缺失信息和解释限制','']
    lines += ['- '+w for w in warnings+gate['limitations']+targets['limits']]
    return '\n'.join(lines)+'\n'


def run(args):
    require(sys.platform == 'linux', 'Real diagnosis must run on the server, never locally')
    root=Path(args.root).resolve(strict=True);project=Path(args.project).resolve(strict=True)
    output=root/'b1_diagnosis'
    require(not output.is_symlink(), 'Output symlink refused')
    output.mkdir(exist_ok=True)
    require(not (output/'B1_DIAGNOSIS.md').exists(), 'Existing report preserved; do not overwrite a previous diagnosis')
    for p in output.rglob('*'):
        require(not p.is_symlink(), 'Output symlink refused')
    import fcntl
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reader=Reader()
        try:
            print('VERIFY B0/B1 weights, contracts, GT and full caches',flush=True)
            own=Path(__file__).resolve().parent
            manifest=reader.json(own/'TOOL_MANIFEST.json')
            for name,sha in manifest['files'].items():
                require(Path(name).name==name, 'Unsafe tool manifest member')
                reader.track(own/name,sha)
            models={v:load_model_evidence(root,v,reader) for v in ('B0','B1')}
            integrity=verify_pair(models)
            initial=check_source(project,models,reader)
            evidence,missing=optional_geometry_evidence(root,models,reader)
            print('CPU gate inspection; no model construction',flush=True)
            gate=analyze_gate(root,models['B1'],reader,initial,baseline=models['B0'])
            print('Analyze saved official F matches; no inference/AP recomputation',flush=True)
            targets=analyze_targets(models,evidence)
            warnings=[missing] if missing else []
            warnings += ['Optional gate checkpoint missing: '+p for p in gate['optional_missing']]
            reader.unchanged()
            save(output,'GATE_ANALYSIS.json',gate)
            save(output,'TARGET_CHANGES.json',targets)
            save(output,'INPUT_AUDIT.json',dict(**integrity,files=reader.files,unchanged_after_read=True))
            save(output,'B1_DIAGNOSIS.md',report(gate,targets,warnings))
            save(output,'COMPLETE.json',dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),
                outputs={n:digest(output/n) for n in ('GATE_ANALYSIS.json','TARGET_CHANGES.json','INPUT_AUDIT.json','B1_DIAGNOSIS.md')}))
            print('COMPLETE',str(output),flush=True)
        except Exception as error:
            message=str(error)
            save(output,'GATE_ANALYSIS.json',dict(status='blocked',reason=message))
            save(output,'TARGET_CHANGES.json',dict(status='blocked',reason=message))
            save(output,'FAILURE.json',dict(status='blocked',reason=message,traceback=traceback.format_exc()))
            save(output,'B1_DIAGNOSIS.md','# B1诊断未完成\n\n身份核验、缓存或CPU权重读取未通过：'+message+
                 '\n\n没有启动训练、推理或补算。无法给出gate、目标变化或因果判断；建议的路径隔离对照尚未实施。\n')
            raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4')
    parser.add_argument('--project',default='/home/cjm/OpenCOOD-main/cjmnet')
    run(parser.parse_args())
