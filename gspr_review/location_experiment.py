"""Train one BEV-location head, reuse a completed contrast point checkpoint."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import yaml
from . import runtime as rt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--point-run', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--reuse-bev', help='Reuse a completed BEV training directory; rerun only paired evaluation in a NEW output directory')
    parser.add_argument('--resume', action='store_true', help='Reuse completed training/evaluations in --output-dir; strict checks remain enabled')
    args = parser.parse_args()
    if args.resume and args.reuse_bev:
        parser.error('Use either --resume or --reuse-bev')
    point = Path(args.point_run).resolve()
    options = yaml.safe_load((point/'experiment.yaml').read_text(encoding='utf-8'))
    if options['review'].get('architecture') != 'contrast':
        parser.error('--point-run must be a completed contrast run')
    manifest = json.loads((point/'manifest.json').read_text(encoding='utf-8'))
    frontend = Path(manifest['frontend_checkpoint'])
    watched = [point/'experiment.yaml', point/'reviewer_best.pth', point/'frontend_config.yaml', frontend]
    hashes = {str(p): rt.sha256(p) for p in watched}
    if hashes[str(frontend)] != manifest['frontend_sha256']:
        raise ValueError('Frozen frontend differs from point run')
    if args.resume:
        if not args.output_dir:
            parser.error('--resume requires --output-dir')
        out = Path(args.output_dir).resolve()
        saved = json.loads((out/'location_protocol.json').read_text(encoding='utf-8'))
        if saved['point_run'] != str(point) or saved['input_sha256'] != hashes:
            raise ValueError('Resume inputs differ from the original location run')
    else:
        out = rt.new_output(args.output_dir or str(point)+'_location_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        rt.write_json(out/'location_protocol.json', {'point_run': str(point), 'input_sha256': hashes,
        'comparison': 'Same 1472 parameter head and evidence inputs; scalar applied to point weights vs ego BEV. Spatial pooling and relative BEV modulation differ; not physically identical perturbations.'})
    options['review']['architecture'] = 'bev_contrast'
    config = out/'bev_config.yaml'
    if args.resume:
        if yaml.safe_load(config.read_text(encoding='utf-8')) != options:
            raise ValueError('Saved BEV config differs from expected configuration')
    else:
        config.write_text(yaml.safe_dump(options, sort_keys=False), encoding='utf-8')
    def run(module, *arguments):
        print(datetime.now().isoformat(timespec='seconds'), module, *map(str, arguments), flush=True)
        subprocess.run([sys.executable, '-u', '-m', 'gspr_review.'+module, *map(str, arguments)], check=True, cwd=rt.ROOT)
    common = ['--frontend-config', point/'frontend_config.yaml', '--frontend-checkpoint', frontend]
    bev = Path(args.reuse_bev).resolve() if args.reuse_bev else out/'bev'
    if args.resume or args.reuse_bev:
        if yaml.safe_load((bev/'experiment.yaml').read_text(encoding='utf-8')) != options:
            raise ValueError('Reused BEV training configuration differs')
        bev_manifest = json.loads((bev/'manifest.json').read_text(encoding='utf-8'))
        if bev_manifest['frontend_sha256'] != hashes[str(frontend)] or rt.sha256(bev/'frontend_config.yaml') != hashes[str(point/'frontend_config.yaml')]:
            raise ValueError('Reused BEV frontend differs')
        history = [json.loads(s) for s in (bev/'history.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        if not history or history[-1]['epoch'] != int(options['epochs']) or not (bev/'reviewer_best.pth').is_file():
            raise ValueError('Training incomplete; this resume mode reuses only completed training')
        print('Reusing completed BEV training:', bev, flush=True)
    else:
        run('verify', '--config', config, *common)
        run('train', '--config', config, *common, '--run-dir', bev)
    bev_checkpoint_hash = rt.sha256(bev/'reviewer_best.pth')
    rt.write_json(out/'paired_evaluation.json', {'bev_run': str(bev), 'bev_checkpoint_sha256': bev_checkpoint_hash,
        'messages': 'Protocol records exact wire packets; other controls replay them after raw-input fingerprint verification. Candidate packet differences are diagnostics, not silently accepted as numerical error.'})
    rows = []
    step = 0
    for branch in ('processed_lidar', 'processed_lidar_weather'):
        signature = None
        for control in ('protocol', 'a0b0', 'point_normal', 'point_permuted', 'bev_normal', 'bev_permuted'):
            step += 1
            print(f'LOCATION EVALUATION {step}/12: {branch} {control}', flush=True)
            source = point if control.startswith('point') else bev
            mode = control if control in ('protocol', 'a0b0') else 'review'
            extra = [] if mode != 'review' else ['--reviewer', source/'reviewer_best.pth',
                '--evidence-intervention', 'permuted' if control.endswith('permuted') else 'normal']
            if mode != 'a0b0':
                extra += ['--packet-cache', out/f'{branch}_packets', '--packet-cache-mode', 'record' if mode == 'protocol' else 'replay']
            folder = out/f'{branch}_{control}'
            if args.resume and folder.exists():
                if not all((folder/f).is_file() for f in ('frames.jsonl', 'eval.yaml', 'communication.json', 'protocol.json')):
                    raise RuntimeError('Incomplete evaluation directory; preserve it and resolve before resuming: '+str(folder))
                metadata = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
                expected_reviewer = rt.sha256(source/'reviewer_best.pth') if mode == 'review' else None
                if (metadata['frontend_sha256'] != hashes[str(frontend)] or metadata['reviewer_sha256'] != expected_reviewer
                    or metadata['mode'] != mode or metadata['lidar_key'] != branch
                    or metadata['options'] != yaml.safe_load((source/'experiment.yaml').read_text(encoding='utf-8'))
                    or metadata.get('evidence_intervention', 'normal') != ('permuted' if control.endswith('permuted') else 'normal')):
                    raise RuntimeError('Saved evaluation is not compatible: '+str(folder))
                if mode != 'a0b0' and metadata.get('packet_cache') != str(out/f'{branch}_packets'):
                    raise RuntimeError('Old evaluation has no canonical packet cache; use --reuse-bev in a fresh output directory')
                print('Reusing evaluation:', folder, flush=True)
            else:
                run('evaluate', '--config', source/'experiment.yaml', '--frontend-config', source/'frontend_config.yaml',
                '--frontend-checkpoint', frontend, '--lidar-key', branch, '--mode', mode,
                *extra, '--compare-ego', '--output-dir', folder)
            frames = [json.loads(line) for line in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            if mode != 'a0b0':
                current = [(r['sample_index'], r['total_bytes'], r['eligible_points'], r['message_sha256']) for r in frames]
                if signature is None:
                    signature = current
                elif current != signature:
                    from .audit_location import audit
                    report = audit(out)
                    rt.write_json(out/'location_mismatch.json', report)
                    print(json.dumps(report, indent=2), flush=True)
                    raise RuntimeError('Control mismatch (including message hash); see '+str(out/'location_mismatch.json')+'. Training/results preserved; do not retrain.')
            metrics = yaml.safe_load((folder/'eval.yaml').read_text(encoding='utf-8'))
            communication = json.loads((folder/'communication.json').read_text(encoding='utf-8'))
            rows.append(dict(branch=branch, control=control,
                             **{k: metrics[k] for k in ('ap30', 'ap_50', 'ap_70')}, **communication))
            rows[-1]['candidate_packet_difference_counts'] = {kind: sum(r.get('candidate_packet_differences', {}).get(kind, 0) for r in frames)
                for kind in ('review_query', 'review_evidence', 'bev_query', 'bev_features')}
            rt.write_json(out/'location_partial.json', rows)
    for path, digest in hashes.items():
        if rt.sha256(path) != digest:
            raise RuntimeError('Original experiment input changed: ' + path)
    rt.verify_frozen()
    if rt.sha256(bev/'reviewer_best.pth') != bev_checkpoint_hash:
        raise RuntimeError('BEV checkpoint changed during evaluation')
    rt.write_json(out/'location_comparison.json', rows)
    print('Send back:', out/'location_comparison.json', flush=True)


if __name__ == '__main__':
    main()
