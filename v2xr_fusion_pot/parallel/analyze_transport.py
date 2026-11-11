"""Read completed transport and fixed-target evidence on the server."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from common import atomic_json, read_json, sha256
from diagnostics import fixed_metrics, verify_gt

ROOT = run.OUT
OUT = ROOT / 'analysis-20261111'
WEATHERS = ('clean', 'fog', 'snow')
SOURCES = ('lidar_only', 'radar_only', 'both', 'no_points')


def modality(lidar, radar):
    return 'both' if lidar and radar else 'lidar_only' if lidar else 'radar_only' if radar else 'no_points'


def rates(row):
    n = row.get('positions', 0)
    return {**row, 'mean_mass': row.get('mass_sum', 0)/n if n else None,
            'mean_remaining': row.get('remaining_mass_sum', 0)/n if n else None,
            'matched_fraction': row.get('matched', 0)/n if n else None,
            'low_quality_fraction': row.get('low_quality', 0)/n if n else None,
            'rejected_fraction': row.get('rejected_or_unmatched', 0)/n if n else None}


def main():
    reports = {g: read_json(ROOT / g / 'TRANSPORT.json') for g in ('P1','P2','P3')}
    paths = {'P0': run.LR0/'final_evaluation', 'P3': ROOT/'P3/final_evaluation'}
    paths.update({name: ROOT/'ablations/P3'/name for name in ('module_off','matched_only','unmatched_only','radar_off')})
    paths['P0_gpu4'] = ROOT/'ablations/P0/same_weights_gpu4'
    for path in paths.values():
        verify_gt(path)
    metrics = {name: fixed_metrics(path) for name, path in paths.items()}
    transport = {}
    for group, report in reports.items():
        transport[group] = {}
        for weather in WEATHERS:
            data = report['stats'][weather]
            raw = data['totals']
            sources = defaultdict(Counter)
            for key, value in data['by_distance_and_source'].items():
                side, distance, sensor = key.split('/')
                for label in (side, side+'/'+sensor, side+'/distance/'+distance):
                    sources[label].update({k: v for k,v in value.items() if not k.endswith('_fraction')})
            residual = {}
            for path in ('matched_rms','unmatched_rms'):
                n, active = raw.get(path+'_cells',0), raw.get(path+'_nonzero_cells',0)
                residual[path] = dict(cells=n, active_cells=active,
                    coverage=active/n if n else None,
                    mean_all_cells=raw.get(path+'_sum',0)/n if n else None,
                    mean_active_cells=raw.get(path+'_sum',0)/active if active else None)
            transport[group][weather] = dict(sources={k:rates(v) for k,v in sources.items()},
                                            residual=residual, totals=raw)

    transitions = {}
    lost_rows = []
    for weather in WEATHERS:
        by_support, by_modality, by_neighbor, causal = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), Counter()
        for frame, objects in reports['P3']['targets'].items():
            if not frame.startswith(weather+'/'):
                continue
            hits = {label:{str(p['object_id']) for p in m[frame]['matched_pairs']['0.7']}
                    for label,m in metrics.items()}
            for oid, obj in objects.items():
                a, b = oid in hits['P0'], oid in hits['P3']
                event = 'lost' if a and not b else 'recovered' if b and not a else 'kept' if a else 'missed_both'
                sensor = modality(obj['ego_lidar']+obj['neighbor_lidar'], obj['ego_radar']+obj['neighbor_radar'])
                neighbor = modality(obj['neighbor_lidar'],obj['neighbor_radar'])
                for row in (by_support[obj['source']], by_modality[sensor],by_neighbor[neighbor]):
                    row['targets'] += 1
                    row[event] += 1
                    row['P0_detected'] += a
                    row['P3_detected'] += b
                    for label in ('module_off','matched_only','unmatched_only','radar_off','P0_gpu4'):
                        row[label+'_detected'] += oid in hits[label]
                    row['unmatched_recovered_same_weights'] += b and oid not in hits['matched_only']
                    row['unmatched_lost_same_weights'] += not b and oid in hits['matched_only']
                    row['module_recovered_same_weights'] += b and oid not in hits['module_off']
                    row['module_lost_same_weights'] += not b and oid in hits['module_off']
                if event == 'lost':
                    causal['lost'] += 1
                    for label in ('module_off','matched_only','unmatched_only','radar_off'):
                        causal['detected_with_'+label] += oid in hits[label]
                    causal['center_unmatched_nonzero'] += obj['center_unmatched_rms'] > 0
                    causal['P0_gpu4_also_detected'] += oid in hits['P0_gpu4']
                    lost_rows.append(dict(frame=frame,object_id=oid,modality=sensor,neighbor_modality=neighbor,
                                          hits={k:oid in v for k,v in hits.items()},**obj))
        transitions[weather] = dict(observation=by_support, modality=by_modality,
                                    neighbor_modality=by_neighbor, lost_causal=causal)
    final=read_json(ROOT/'FINAL_RESULTS.json')
    ap = {g:{w:final['groups'][g]['evaluation']['results'][w]['AP']['F'] for w in WEATHERS} for g in ('P0','P1','P2','P3')}
    result = dict(status='complete', transport=transport, targets=transitions, lost_targets=lost_rows, F_AP=ap,
        source_sha256={str(ROOT/g/'TRANSPORT.json'):sha256(ROOT/g/'TRANSPORT.json') for g in reports},
        definitions=dict(match='mass >= 0.5',low_quality='0.1 <= mass < 0.5',rejected='mass < 0.1',
            target_match='official F protocol IoU 0.7',region_denominator='occupied positions repeated per ego-neighbor pair',
            target_support='real per-vehicle points inside fixed Clean GT box'))
    atomic_json(OUT/'ANALYSIS.json',result)
    compact=dict(AP70={g:{w:v['ap70'] for w,v in rows.items()} for g,rows in ap.items()},
                 transport={g:{w:dict(ego=v['sources']['ego'],neighbor=v['sources']['neighbor'],residual=v['residual'],
                    solver_residual=v['totals']['max_solver_residual']) for w,v in rows.items()} for g,rows in transport.items()},
                 targets=transitions)
    print(json.dumps(compact,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
