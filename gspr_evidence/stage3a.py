"""Stage-3A: last-observed processing failure, never a root-cause claim."""
from collections import Counter
import json
import time
from . import stage3_runtime as sr
from .stage3_analysis import write_json


def main():
    args = sr.parser(__doc__).parse_args()
    import torch
    from opencood.tools.train_utils import to_device
    from .stage3_trace import trace_branch, describe_target, save_proposals, geometry_comparison
    rt, lineage, model, device, ds, loader, candidates, out, protocol = sr.prepare(args)
    protocol['stage'] = '3A'
    write_json(out/'protocol.json', protocol)
    artifacts = out/'proposals'
    artifacts.mkdir()
    counts, frames, occurrences = Counter(), 0, 0
    start = time.monotonic()
    with torch.no_grad(), (out/'targets.jsonl').open('w', encoding='utf-8') as stream:
        for batch in loader:  # Preserve complete queue; skip inference, not data generation.
            index = int(batch['ego']['communication_sample_index'][0])
            if index not in candidates:
                continue
            batch = to_device(batch, device)
            inp = rt.input_branch(batch['ego'], args.weather)
            encoded = model.encode(inp)
            predictions, _, gt = sr.replay_frame(model, ds, batch, encoded, inp, index, lineage)
            input_hash = sr.tensor_digest(dict(inp=inp, gt=gt))
            selected = candidates[index]
            full = trace_branch(ds, batch, predictions['full'])
            full_path = f'proposals/{index}_full.npz'
            save_proposals(out/full_path, full, selected)
            peers_needed = sorted({p for r in selected.values() for p in r['detected_peer_indices']})
            rows = {}
            for j, old in selected.items():
                desc = describe_target(full, j)
                if desc['matched']:
                    raise AssertionError('Stage-2 full-miss candidate unexpectedly recovered')
                stats = lineage['stats'][(index, j)]
                rows[j] = dict(sample_index=index, target_index=j, scene=stats['scene'],
                    distance=stats['distance'], weather=args.weather, input_sha256=input_hash,
                    unit='target-frame occurrence', full=desc, full_proposals=full_path,
                    peers=[], failure_stage=desc['failure_stage'])
            for a in peers_needed:
                peer = trace_branch(ds, batch, predictions['peer_'+str(a)])
                target_ids = [j for j, r in selected.items() if a in r['detected_peer_indices']]
                peer_path = f'proposals/{index}_peer_{a}.npz'
                save_proposals(out/peer_path, peer, target_ids)
                for j in target_ids:
                    desc = describe_target(peer, j)
                    if not desc['matched']:
                        raise AssertionError('Previously valid peer no longer detects target')
                    rows[j]['peers'].append(dict(peer_index=a, trace=desc, proposals=peer_path,
                        comparison=geometry_comparison(peer, full, j)))
                del peer
            for j in sorted(rows):
                stream.write(json.dumps(rows[j], ensure_ascii=False, allow_nan=False)+'\n')
                counts[rows[j]['failure_stage']] += 1
                occurrences += 1
            stream.flush()
            frames += 1
            elapsed = time.monotonic()-start
            print(f'{args.weather} candidate frames {frames}/{len(candidates)}, '
                  f'targets={occurrences}, elapsed={elapsed/60:.1f}min, '
                  f'candidate ETA={elapsed/frames*(len(candidates)-frames)/60:.1f}min', flush=True)
            del full, predictions, encoded
            if args.smoke and frames == len(candidates):
                break
    if frames != len(candidates) or occurrences != protocol['candidate_targets']:
        raise RuntimeError('Incomplete Stage-3A; no summary accepted')
    sr.final_guards(lineage)
    write_json(out/'summary.json', dict(complete=True, smoke=args.smoke, weather=args.weather,
        frames=frames, occurrences=occurrences, failure_counts=dict(counts),
        interpretation='Last-observed failure stages, NOT root-cause labels.'))


if __name__ == '__main__':
    main()
