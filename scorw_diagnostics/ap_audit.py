"""Identical cached predictions; factorial GT/matching/sorting audit on CPU."""
import copy
from pathlib import Path
import random
import numpy as np
from .core import read, write, text, identity, stage
from .data import catalogue, fixed_gt, cache_frame

THRESHOLDS = (.3, .5, .7)


def ap(rows, global_sort):
    gt = sum(r['gt'] for r in rows)
    if not gt:
        return None
    tp = np.asarray([v for r in rows for v in r['tp']], dtype=float)
    fp = np.asarray([v for r in rows for v in r['fp']], dtype=float)
    if global_sort:
        order = np.argsort(-np.asarray([v for r in rows for v in r['score']]))
        tp, fp = tp[order], fp[order]
    rec = (np.cumsum(tp) / gt).tolist()
    prec = (np.cumsum(tp) / np.maximum(1, np.cumsum(tp + fp))).tolist()
    r = [0., *rec, 1.]; p = [0., *prec, 0.]
    for k in range(len(p)-2,-1,-1):
        p[k] = max(p[k],p[k+1])
    return float(sum((r[k]-r[k-1])*p[k] for k in range(1,len(r)) if r[k]!=r[k-1]))


def table(frames, get):
    return {mode: {str(t): {str(g): ap([get(r, mode, t) for r in frames], g)
                           for g in (True, False)} for t in THRESHOLDS} for mode in ('clean', 'fog')}


def replay(c, out):
    root, d, keys, _ = catalogue(c)
    frames = [cache_frame(root, *key) for key in keys]
    values = table(frames, lambda r, m, t: r['predictions'][m]['matches'][str(t)])
    from scorw_r3.metrics import ap as old_ap
    from opencood.utils.eval_utils import calculate_ap
    for mode in values:
        for i, t in enumerate(THRESHOLDS):
            records = [r['predictions'][mode]['matches'][str(t)] for r in frames]
            for g in (True, False):
                stat = {t: dict(tp=[v for r in records for v in r['tp']],
                                fp=[v for r in records for v in r['fp']],
                                score=[v for r in records for v in r['score']], gt=sum(r['gt'] for r in records))}
                native = calculate_ap(copy.deepcopy(stat), t, g)[0]
                if abs(native - values[mode][str(t)][str(g)]) > c['ap_absolute_tolerance']:
                    raise ValueError('AP implementation regression')
            result = values[mode][str(t)]['True']
            if abs(result - c['reference_ap'][mode][i]) > c['ap_absolute_tolerance'] or result != old_ap(records):
                raise ValueError('A1 cached AP does not reproduce D6')
    write(out / 'AP_PROTOCOL_AUDIT.json', dict(status='passed', frames=len(frames),
          cache_replay=values, predictions_recomputed=False, only_sorting_changed=True))


def make_native(c):
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    root = Path(c['dataset']); cfg = read(root / 'DATASET_PROTOCOL.json')['config']
    h = load_yaml(cfg['model']['config'])
    for k in ('weather_augmentation', 'prediction_distillation', 'DG_params'):
        h.pop(k, None)
    h['data_augment'] = []; h['root_dir'] = str(root / 'clean/train'); h['validate_dir'] = str(root / 'clean/validate')
    if 'wild_setting' in h:
        h['wild_setting']['async'] = False; h['wild_setting']['loc_err'] = False
    ds = build_dataset(h, visualize=False, train=False)
    if len(ds) != c['expected_frames']:
        raise ValueError('Native frame scope mismatch')
    return ds, h


def native_gt(ds, index):
    """Actual getitem/collate/generate_gt_bbx; reconstruct IDs with exact same ops."""
    import torch
    from opencood.utils import box_utils
    sample = ds[index]
    if sample is None:
        raise ValueError('Native loader returned None')
    batch = ds.collate_batch_test([sample])
    gt = ds.post_processor.generate_gt_bbx(batch).cpu().numpy()
    b = batch['ego']; centers = b['object_bbx_center'][b['object_bbx_mask'] == 1]
    corners = box_utils.boxes_to_corners_3d(centers, ds.post_processor.params['order'])
    corners = box_utils.project_box3d(corners.float(), b['transformation_matrix'])
    ids = b['object_ids']; indices = [ids.index(x) for x in set(ids)]
    candidates = corners[indices]
    mask = box_utils.get_mask_for_boxes_within_range_torch(candidates)
    expected = candidates[mask].cpu().numpy()
    if not np.array_equal(gt, expected):
        raise ValueError('Native GT ID mapping ambiguous; stop')
    kept = [str(ids[k]) for k, yes in zip(indices, mask.tolist()) if yes]
    if len(set(kept)) != len(kept):
        raise ValueError('Native numeric/string IDs collide')
    return gt, kept, [str(x) for x in sample['ego'].get('cav_ids', [])], sample


def geometry(a, ids_a, b, ids_b):
    common = sorted(set(ids_a) & set(ids_b)); details = []
    def shape(x):
        return dict(center=x.mean(axis=0).tolist(),
                    edge_lengths=[float(np.linalg.norm(x[i] - x[j])) for i, j in ((0, 1), (1, 2), (0, 4))],
                    yaw_rad=float(np.arctan2(*(x[1, :2] - x[2, :2])[::-1])))
    for key in common:
        x = a[ids_a.index(key)]; y = b[ids_b.index(key)]
        dist = np.linalg.norm(x[:, None] - y[None, :], axis=2)
        details.append(dict(object_id=key, indexed_corner_max_m=float(abs(x-y).max()),
                            corner_set_error_m=float(max(dist.min(0).max(), dist.min(1).max())),
                            d6=shape(x), native=shape(y)))
    return dict(d6_count=len(a), native_count=len(b), common_ids=len(common),
                d6_only=sorted(set(ids_a)-set(ids_b)), native_only=sorted(set(ids_b)-set(ids_a)), common=details,
                matching='stable string object_id; no geometric forced pairing')


def match_pair(boxes, scores, gt, audit):
    import torch
    from opencood.utils import common_utils, eval_utils
    from scorw_r3.metrics import iou_matrix, match_matrix
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 8, 3); scores = np.asarray(scores, dtype=np.float32)
    matrix = iou_matrix(boxes, gt, audit)
    d6 = {str(t): match_matrix(matrix, scores, t) for t in THRESHOLDS}
    native = {}
    for t in THRESHOLDS:
        stat = {t: dict(tp=[], fp=[], score=[], gt=0)}
        # Actual frozen native matcher, including polygon removal/tie handling.
        with audit.guard(common_utils):
            eval_utils.caluclate_tp_fp(torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(gt), stat, t)
        native[str(t)] = stat[t]
    return d6, native


def run(c, out, parent, smoke):
    from scorw_repair.iou_audit import IoUAudit
    root, data, keys, rows = catalogue(c); ds, h = make_native(c)
    native_keys = []
    for scene in ds.scenario_database.values():
        first = next(iter(scene.values()))
        native_keys.extend((Path(f['lidar']).parent.parent.name, t) for t, f in first.items() if t != 'ego')
    if native_keys != keys:
        raise ValueError('Native scene/timestamp order differs')
    layouts = {s['scene']: s for s in data['layout']}; outputs = []
    n = c['smoke_frames'] if smoke else len(keys)
    historical = read(root / 'DATASET_PROTOCOL.json')['config']['model']['reference_protocol']
    protocol = read(historical); seed = protocol['protocol']['seed']
    from reference_weather.common import stable_seed
    for index, (s, t) in enumerate(keys[:n]):
        folder = out / 'frames' / s / t
        def work():
            frame_seed = stable_seed(seed, 'eval-frame', index)
            np.random.seed(frame_seed); random.seed(frame_seed)
            gt, ids, active, pose = fixed_gt(c, layouts[s], t, rows)
            cache = cache_frame(root, s, t)
            if cache['GT_ids'] != ids or cache['clean_GT_sha256'] != identity(gt.tolist()) or cache['cavs'] != active:
                raise ValueError('Reconstructed D6 GT/CAV identity mismatch')
            native, ni, nc, sample = native_gt(ds, index)
            audit = IoUAudit(None)
            matched = {}
            try:
                for mode in ('clean', 'fog'):
                    pred = cache['predictions'][mode]; matched[mode] = {}
                    for name, g in (('d6', gt), ('native', native)):
                        x, y = match_pair(pred['boxes'], pred['scores'], g, audit)
                        if name == 'd6' and x != pred['matches']:
                            raise ValueError('Recomputed D6 matching differs from frozen cache')
                        matched[mode][name] = dict(d6=x, native=y)
            finally:
                write(folder / 'IOU_AUDIT.json', audit.report)
            diff = geometry(gt, ids, native, ni)
            before_ids = [str(x) for x in sample['ego']['object_ids']]
            diff.update(native_before_final_xy_filter_ids=before_ids,
                        native_final_xy_filter_removed=sorted(set(before_ids)-set(ni)),
                        geometric_differences_above_tolerance=sum(x['corner_set_error_m'] > c['gt_geometry_tolerance_m']
                                                                 for x in diff['common']))
            diff.update(scene=s, timestamp=t, d6_CAVs=active, native_CAVs=nc,
                        d6_ego=layouts[s]['cavs'][0], native_ego=str(sample['ego'].get('cav_ids', [layouts[s]['cavs'][0]])[0]),
                        native_point_visibility='actual native generate_object_center/getitem executed; source audit supplied separately')
            write(folder / 'GT.json', dict(d6=gt.tolist(), d6_ids=ids, native=native.tolist(), native_ids=ni))
            write(folder / 'RESULT.json', dict(scene=s, timestamp=t, difference=diff, matches=matched))
        stage(folder, dict(parent=parent, experiment='A', frame=[s, t]), work)
        outputs.append(read(folder / 'RESULT.json'))
        if (index+1) % 20 == 0 or index+1 == n:
            print('A GT/matching %d/%d' % (index+1, n), flush=True)
    values = {g: {m: table(outputs, lambda r, mode, t: r['matches'][mode][g][m][str(t)])
                  for m in ('d6', 'native')} for g in ('d6', 'native')}
    differences = [r['difference'] for r in outputs]
    import inspect
    source = inspect.getsource(ds.post_processor.generate_object_center)
    evidence = dict(native_generate_object_center_source=source,
                    native_single_car_source=inspect.getsource(ds.get_item_single_car),
                    filter_range='native first projects and reconstructs yaw box, filters all 8 corners in XYZ; final GT uses XY',
                    fixed_gt='canonical original corners; 120m of any active CAV then XY all-corner GT_RANGE',
                    point_visibility='inspect recorded actual source and actual native GT; no assumed point-count threshold')
    write(out / 'GT_DIFFERENCE.json', dict(frames=n, per_frame=differences, native_configuration=h, source_evidence=evidence,
          native_IDs='exact output order verified against actual postprocessor',
          comparison_scope='same ego coordinates; D6 and native GT differ only as actually observed'))
    write(out / 'MATCHING_DIFFERENCE.json', dict(frames=n, factorial_AP=values,
          mismatched_frame_thresholds=sum(r['matches'][mode][g]['d6'][str(t)]['tp'] !=
                                         r['matches'][mode][g]['native'][str(t)]['tp']
              for r in outputs for mode in ('clean', 'fog') for g in ('d6', 'native') for t in THRESHOLDS),
          matcher_note='native result preserves actual TP/FP/score; D6 also stores assignments'))
    historical_result = Path(historical).parent / 'clean/result.json'
    old = read(historical_result) if historical_result.exists() else None
    lines = ['# AP评测协议对照', '', 'A1使用全部1980帧缓存复现D6；本表GT重匹配范围为%d帧。' % n,
             '所有对照固定D6预测框和置信度；本次没有重新推理。', '',
             '| 天气 | GT | 匹配规则 | 指标 | 全局排序 | 按帧累计 | 差值 |',
             '|---|---|---|---|---:|---:|---:|']
    for g in values:
        for m in values[g]:
            for mode, thresholds in values[g][m].items():
                for t, v in thresholds.items():
                    lines.append('|%s|%s|%s|AP%d|%.10f|%.10f|%+.10f|' %
                                 (mode, g, m, round(float(t)*100), v['True'], v['False'], v['True']-v['False']))
    contributions = {}
    for mode in ('clean', 'fog'):
        contributions[mode] = {}
        for t in map(str, THRESHOLDS):
            base = values['d6']['d6'][mode][t]['True']
            sorted_change = values['d6']['d6'][mode][t]['False'] - base
            gt_change = values['native']['d6'][mode][t]['False'] - values['d6']['d6'][mode][t]['False']
            matcher_change = values['native']['native'][mode][t]['False'] - values['native']['d6'][mode][t]['False']
            contributions[mode][t] = dict(order='global D6 -> frame-order D6 -> native GT -> native matcher',
                sorting=sorted_change, GT_given_frame_order=gt_change, matcher_given_native_GT=matcher_change,
                total=sorted_change+gt_change+matcher_change,
                interaction_warning='sequential decomposition depends on order; full factorial table retained')
    historical_residual = {}
    if not smoke:
        for i, t in enumerate(map(str, THRESHOLDS)):
            historical_value = old['primary_ap']['ap'+str(round(float(t)*100))] if old else c['historical_native_ap'][i]
            historical_residual[t] = dict(historical_native_AP=historical_value,
                D6_prediction_native_GT_native_matcher_frame_order_AP=values['native']['native']['clean'][t]['False'],
                unexplained_AP_difference=values['native']['native']['clean'][t]['False']-historical_value,
                historical_precision='full published result' if old else 'context rounded six decimals')
    write(out / 'FACTOR_CONTRIBUTIONS.json', dict(values=contributions, historical_native_result=old,
          historical_residual=historical_residual,
          historical_predictions_equal='UNVERIFIED_NO_NATIVE_PREDICTION_CACHE',
          unexplained='prediction/input order, CAV truncation-before/after communication, native postprocess and GT order ties may remain'))
    lines += ['', '因素贡献采用固定顺序分解：排序→GT→匹配；完整交叉表保留，贡献依赖分解顺序。',
              '历史第44节为原生推理，本次使用D6缓存。历史预测框是否相同尚未验证，剩余差异不能全部归因于上述三项。',
              '原生流程先截断CAV再做70m通信筛选，D6先通信筛选再截断；原生还打乱点顺序。实际影响需结合逐帧记录。']
    text(out / 'AP_PROTOCOL_COMPARISON.md', '\n'.join(lines)+'\n')
