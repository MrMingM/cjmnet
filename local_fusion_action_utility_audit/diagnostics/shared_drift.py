"""Diagnose exact Shared-hash failures without changing or resuming the audit.

The diagnostic leaves the run's caches, source snapshot and protocol untouched.
Torch inference is run only by the user's remote-server command.
"""
import argparse
from datetime import datetime
from pathlib import Path
import time


from ..common import Runtime, atomic_json, load_cache, read_json
from ..s0_counterfactual import frame_ap_stats, prediction_hash
from ..reproducibility import numeric_difference, stats_difference, tensor_tree_hash


def choose_frame(protocol, manifest, weather=None, frame=None):
    failures = sorted((row for row in manifest['entries'].values()
                       if row['stage'] == 'S0-frame' and row['split'] == 'validation'
                       and row['status'] == 'failed'),
                      key=lambda row: row['updated_utc_epoch'], reverse=True)
    weather = weather or (failures[0]['weather'] if failures else 'clean')
    frame = frame if frame is not None else (int(failures[0]['frame']) if failures
                                            and failures[0]['weather'] == weather
                                            else protocol['validation_indices'][0])
    if frame not in protocol['validation_indices']:
        raise ValueError('Diagnostic frame must belong to the recorded B0 validation indices')
    return weather, frame


def get_batch(runtime, weather, frame):
    from opencood.tools.train_utils import to_device
    dataset, loader = runtime.loader('validation', weather)
    for ordinal, batch in enumerate(loader):
        index = int(batch['ego']['communication_sample_index'][0])
        if index == frame:
            batch = to_device(batch, runtime.target)
            del loader
            return dataset, batch, ordinal
    raise RuntimeError('Diagnostic frame was not produced by the recorded loader')


def diagnose(run, weather=None, frame=None):
    import torch
    from local_fusion_task_source_oracle.oracle import build_candidate_pool
    run = Path(run)
    protocol, manifest = read_json(run / 'protocol.json'), read_json(run / 'manifest.json')
    weather, frame = choose_frame(protocol, manifest, weather, frame)
    runtime = Runtime(run)
    if not runtime.manifest.complete('baseline-frame', 'validation', weather, frame):
        raise RuntimeError('A complete hash-verified baseline frame is required')
    baseline = load_cache(run / 'cache' / 'baseline' / weather / f'{frame:08d}.pt')
    print(f'DIAGNOSTIC validation/{weather} frame={frame}', flush=True)
    print('Existing experiment caches, protocol and manifest are read-only.', flush=True)
    dataset, batch, ordinal = get_batch(runtime, weather, frame)
    records, arrays, posts = [], [], []
    selected_input = ('processed_lidar' if weather == 'clean' else 'processed_lidar_weather')

    def infer(name, with_pool=False):
        inputs = {key: batch['ego'][key] for key in
                  (selected_input, 'record_len', 'communication_transforms',
                   'anchor_box', 'transformation_matrix')}
        start = time.monotonic()
        before_input = tensor_tree_hash(inputs)
        before_buffers = tensor_tree_hash({'model': dict(runtime.model.named_buffers()),
                                          'shared': dict(runtime.shared_arm.named_buffers())})
        hooks, traces = [], {}
        # Locate where repeatability first fails without replacing any operator.
        for label, module in (('spectral', runtime.model.engine.base.gspr.spectral),
                              ('gspr', runtime.model.engine.base.gspr),
                              ('pillar_vfe', runtime.model.engine.base.pillar_vfe),
                              ('backbone_first', runtime.model.engine.base.backbone.blocks[0])):
            hooks.append(module.register_forward_hook(
                lambda module, args, output, key=label: traces.__setitem__(key, tensor_tree_hash(output))))
        try:
            context, shared, _ = runtime.predict(batch, weather, verify=ordinal == 0, sources=False)
        finally:
            for hook in hooks:
                hook.remove()
        before_pool = prediction_hash(shared)
        if with_pool:
            build_candidate_pool(runtime.model.engine.base, context['levels'], shared,
                                 runtime.spec['candidate_families'])
        after_pool = prediction_hash(shared)
        arrays.append({key: value.detach().cpu().numpy().copy() for key, value in shared.items()})
        post = dataset.post_process(batch, {'ego': shared})
        stats = frame_ap_stats(post)
        posts.append([value.detach().cpu().numpy().copy() if value is not None else None for value in post])
        record = {'case': name, 'input_hash': before_input,
                  'input_unchanged': before_input == tensor_tree_hash(inputs),
                  'buffers_unchanged': before_buffers == tensor_tree_hash({
                      'model': dict(runtime.model.named_buffers()), 'shared': dict(runtime.shared_arm.named_buffers())}),
                  'raw_hash': before_pool, 'matches_saved_baseline_hash': before_pool == baseline['prediction_hash'],
                  'pool_did_not_mutate_shared': before_pool == after_pool,
                  'postprocess_did_not_mutate_shared': after_pool == prediction_hash(shared),
                  'intermediate_hashes': traces, 'stats_vs_saved_baseline': stats_difference(baseline['stats'], stats),
                  'seconds': time.monotonic() - start}
        if len(arrays) > 1:
            record['raw_vs_first'] = {key: numeric_difference(arrays[0][key], arrays[-1][key])
                                      for key in ('psm', 'rm')}
            record['post_vs_first'] = {
                key: numeric_difference(posts[0][i], posts[-1][i])
                if posts[0][i] is not None and posts[-1][i] is not None
                else {'same_none_status': (posts[0][i] is None) == (posts[-1][i] is None)}
                for i, key in enumerate(('boxes', 'scores', 'gt'))}
        records.append(record)
        print(f"{name}: input={before_input[:12]} raw={before_pool[:12]} "
              f"baseline_hash_match={record['matches_saved_baseline_hash']} "
              f"buffers_unchanged={record['buffers_unchanged']}", flush=True)
        if len(arrays) > 1:
            print('  raw_vs_first=' + str(record['raw_vs_first']), flush=True)
        print('  stats_vs_saved_baseline=' + str(record['stats_vs_saved_baseline']), flush=True)
        del shared, context

    with torch.no_grad():
        infer('same_batch_first')
        infer('same_batch_repeat')
        infer('same_batch_with_source_pool', with_pool=True)
        del dataset, batch
        dataset, batch, ordinal = get_batch(runtime, weather, frame)
        infer('fresh_loader_with_source_pool', with_pool=True)
    raw_close = all(value.get('allclose_2e5', False) for row in records[1:]
                    for value in row['raw_vs_first'].values())
    matching_equal = all(value['same_gt_count'] and value['same_tp_sequence'] and value['same_fp_sequence']
                         for row in records for value in row['stats_vs_saved_baseline'].values())
    summary = {'same_input_in_all_cases': len({row['input_hash'] for row in records}) == 1,
               'raw_hash_varies_between_repeats': len({row['raw_hash'] for row in records}) > 1,
               'repeat_raw_differences_within_existing_2e5_tolerance': raw_close,
               'saved_baseline_tp_fp_sequences_unchanged': matching_equal,
               'all_inputs_and_buffers_unchanged': all(row['input_unchanged'] and row['buffers_unchanged']
                                                       for row in records),
               'first_variable_intermediate_on_same_batch': next((key for key in
                   ('spectral', 'gspr', 'pillar_vfe', 'backbone_first')
                   if any(row['intermediate_hashes'][key] != records[0]['intermediate_hashes'][key]
                          for row in records[1:3])), None)}
    report = {'weather': weather, 'frame': frame, 'cases': records, 'summary': summary,
              'environment': {'torch': torch.__version__, 'hip': torch.version.hip,
                              'device': torch.cuda.get_device_name(runtime.target),
                              'cudnn_deterministic': torch.backends.cudnn.deterministic},
              'training_frames_complete': sum(row['stage'] == 'S0-frame' and row['split'] == 'train'
                                              and row['status'] == 'complete' for row in manifest['entries'].values()),
              'scope': 'diagnosis only; no experiment checks relaxed and no cache identity changed'}
    output = run / 'diagnostics' / ('shared_drift_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    atomic_json(output, report)
    print('SUMMARY:', flush=True)
    for key, value in summary.items():
        print(f'  {key}={value}', flush=True)
    print('REPORT=' + str(output), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', required=True)
    cli.add_argument('--weather', choices=('clean', 'fog', 'rain', 'snow'))
    cli.add_argument('--frame', type=int)
    args = cli.parse_args()
    diagnose(args.run, args.weather, args.frame)


if __name__ == '__main__':
    main()
