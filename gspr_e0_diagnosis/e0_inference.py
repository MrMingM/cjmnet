"""Original inference.main execution with transactional, additive observations."""
from contextlib import ExitStack
from pathlib import Path
import argparse
import numpy as np
from e0_common import (HERE, WEATHERS, atomic, frame_commit, frame_done, historical,
                       identity, read, sha)
from e0_numeric import array_hash
from e0_observe import Observer, summarize_points
from e0_runtime import npz_write, state_hash, tensor_numpy


def compare_predictions(boxes, scores, record, protocol):
    if (boxes is None) != record['prediction_is_none']:
        raise ValueError('R0 frozen prediction None state differs')
    if boxes is not None:
        b, s = tensor_numpy(boxes), tensor_numpy(scores)
        old_b = np.asarray(record['prediction_boxes'], dtype=record['box_dtype']).reshape(-1, 8, 3)
        old_s = np.asarray(record['scores'], dtype=record['score_dtype'])
        if b.shape != old_b.shape or s.shape != old_s.shape or b.dtype != old_b.dtype or s.dtype != old_s.dtype:
            raise ValueError('R0 frozen final prediction shape/dtype differs')
        np.testing.assert_allclose(b, old_b, atol=protocol['box_atol_m'], rtol=0)
        np.testing.assert_allclose(s, old_s, atol=protocol['score_atol'], rtol=0)


def finalize_observations(destination, request_id, count):
    for weather in WEATHERS:
        for index in range(count):
            row_path = destination / 'predictions' / weather / f'{index:06d}.json'
            out = destination / 'observations' / weather / f'{index:06d}.json'
            if frame_done(out, request_id):
                record = read(out)
            else:
                pending = destination / 'pending' / weather / f'{index:06d}.json'
                record = read(pending)
                row = read(row_path)
                if record['final_prediction_identity'] != identity([row['prediction_boxes'], row['scores'], row['prediction_is_none']]):
                    raise ValueError('Pending observation does not match committed prediction')
                record['prediction_sha256'] = sha(row_path)
                frame_commit(out, record, request_id)
            if sha(row_path) != record['prediction_sha256']:
                raise ValueError('Observation/prediction hash mismatch')
            sample = record.get('statistics', {}).get('sample')
            if sample and sha(sample['path']) != sample['sha256']:
                raise ValueError('Reliability spatial sample changed')
            head = record.get('head_arrays')
            if head and sha(head['path']) != head['sha256']:
                raise ValueError('Diagnostic head evidence changed')


def all_protocol_matches(destination, count):
    import gzip
    import json
    import torch
    from evaluate import official_match_trace, statistics
    from support import corner_array, prediction_tensors
    clean = [read(destination / 'predictions/clean' / f'{i:06d}.json')['official_gt'] for i in range(count)]
    target = destination / 'all_protocol_matches.jsonl.gz'
    temp = Path(str(target)+'.tmp')
    with temp.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for weather in WEATHERS:
                for index in range(count):
                    row = read(destination / 'predictions' / weather / f'{index:06d}.json')
                    boxes, scores = prediction_tensors(row)
                    record = dict(index=index, weather=weather, frame_key=row['frame_key'], protocols={})
                    for protocol in ('O', 'O*', 'F'):
                        objects = clean[index] if protocol == 'F' else row['official_gt']
                        stat = statistics()
                        values = {}
                        for threshold in (.3, .5, .7):
                            skipped = protocol == 'O' and boxes is None
                            pairs, ties = ([], []) if skipped else official_match_trace(boxes, scores,
                                torch.from_numpy(corner_array(objects)), [o['object_id'] for o in objects], stat, threshold)
                            values[str(threshold)] = dict(skipped=skipped, pairs=pairs, ties=ties,
                                gt_count=0 if skipped else len(objects), tp=len(pairs),
                                fp=0 if boxes is None else len(boxes)-len(pairs),
                                fn=0 if skipped else len(objects)-len(pairs))
                        record['protocols'][protocol] = values
                    stream.write((json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n').encode())
    temp.replace(target)


def run_arm(cli, destination, arm, request_id, preflight, smoke=False):
    # Do not call setup() here: the frozen evaluate() bootstraps source isolation once.
    import torch
    audit, evaluation, models = historical()
    audit.gpu()
    protocol = read(HERE / 'protocol.json')
    evidence = read(cli.output / 'stages/preflight/evidence.json')
    count = protocol['smoke_prefix_frames'] if smoke else 829
    args = argparse.Namespace(**audit.DEFAULTS)
    args.output_root, args.variant, args.resume = str(cli.output), 'D', True
    selected = preflight['models']['D']['selected']
    ctx = dict(request_id=request_id, sample_indices=preflight['sample_indices'])
    observers = []
    original_create = models.create
    import predict
    original_generate = predict.generate
    from support import patched
    native_loader = torch.utils.data.DataLoader

    class Loader:
        def __init__(self, dataset, *args_, **kwargs):
            self.inner = native_loader(dataset, *args_, **kwargs)

        def __len__(self):
            return len(self.inner)

        def __iter__(self):
            for index, batch in enumerate(self.inner):
                ctx['index'] = index
                ctx['frame_key'] = evidence['b']['selected_frame_keys'][index]
                ctx['cavs'] = evidence['frames'][index]['conditions'][ctx['weather']]['context']['participating_cavs']
                voxels = batch['ego']['processed_lidar']
                ctx['input_hashes'] = {k: array_hash(tensor_numpy(v)) for k, v in voxels.items()}
                ctx['input_hashes']['record_len'] = array_hash(tensor_numpy(batch['ego']['record_len']))
                ctx['input_hashes']['lidar_pose'] = array_hash(tensor_numpy(batch['ego']['lidar_pose']))
                if arm != 'R0':
                    reference = read(cli.output / ('stages/smoke/R0' if smoke else 'stages/reliability/R0') /
                        'observations' / ctx['weather'] / f'{index:06d}.json')
                    if ctx['input_hashes'] != reference['input_hashes']:
                        raise ValueError('Intervention input tensors differ from R0')
                yield batch

    def create(hypes, variant):
        model = original_create(hypes, variant)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        observer = Observer(model, arm, ctx)
        model._e0_observer = observer
        observers.append(observer)
        return model

    def generate(weather, *args_, **kwargs):
        ctx['weather'] = weather
        # Imported only after the original evaluate() has isolated official OpenCOOD.
        from opencood.tools import inference_utils
        native = inference_utils.inference_intermediate_fusion

        def infer(batch, model, dataset):
            observer = model._e0_observer
            ctx['state_before_forward'] = state_hash(model) if observer.before_state is None else observer.before_state
            if ctx['state_before_forward'] != preflight['loading']['D']['state_sha256']:
                raise ValueError('Live D state differs from selected checkpoint loaded during preflight')
            boxes, scores, gt = native(batch, model, dataset)
            index = ctx['index']
            if arm == 'R0':
                old = read(Path(preflight['models']['D']['root']) / 'final_evaluation/predictions' / weather / f'{index:06d}.json')
                compare_predictions(boxes, scores, old, protocol)
            heads = ctx['heads']
            b = None if boxes is None else tensor_numpy(boxes).tolist()
            s = None if scores is None else tensor_numpy(scores).tolist()
            record = dict(index=index, frame_key=ctx['frame_key'], weather=weather, arm=arm,
                input_hashes=ctx['input_hashes'], head_hashes={k: array_hash(v) for k, v in heads.items()},
                pfn_input_sha256=observer.last['pfn_input_sha256'], pfn_weight_verified=observer.last['pfn_weight_verified'],
                intervention=observer.last['intervention'], state_before=observer.before_state,
                final_prediction_identity=identity([b, s, boxes is None]))
            if arm == 'R0':
                record['statistics'] = summarize_points(observer, batch,
                    cli.output / ('stages/smoke/paired' if smoke else 'stages/paired'), evidence, index, destination)
            if smoke:
                head_path = destination / 'heads' / weather / f'{index:06d}.npz'
                npz_write(head_path, **heads)
                record['head_arrays'] = dict(path=str(head_path), sha256=sha(head_path))
            atomic(destination / 'pending' / weather / f'{index:06d}.json', record)
            observer.last = None
            ctx.pop('heads', None)
            return boxes, scores, gt

        with patched(inference_utils, 'inference_intermediate_fusion', infer):
            result = original_generate(weather, *args_, **kwargs)
        for observer in observers:
            digest = observer.verify_unchanged()
            if observer.before_state is not None and digest != observer.before_state:
                raise ValueError('Frozen D state changed')
        return result

    own = dict(family='GSPR_E0_DIAGNOSIS', revision=1, variant='D', arm=arm,
        request=request_id, checkpoint_sha256=selected['checkpoint_sha256'])
    try:
        with ExitStack() as stack:
            stack.enter_context(patched(models, 'create', create))
            stack.enter_context(patched(predict, 'generate', generate))
            stack.enter_context(patched(torch.utils.data, 'DataLoader', Loader))
            summary = evaluation.inherited.evaluate(args, own, selected['checkpoint'], destination, limit=count if smoke else 0)
    finally:
        for observer in observers:
            observer.close()
    finalize_observations(destination, request_id, count)
    all_protocol_matches(destination, count)
    if arm == 'R0' and not smoke:
        for weather in WEATHERS:
            for p in ('O', 'O*', 'F'):
                for metric in ('ap30', 'ap50', 'ap70'):
                    if abs(summary['results'][weather]['AP'][p][metric]-preflight['models']['D']['summary']['results'][weather]['AP'][p][metric]) > protocol['ap_atol']:
                        raise ValueError('Full R0 AP does not reproduce historical D within fixed tolerance')
    atomic(destination / 'E0_INFERENCE_AUDIT.json', dict(status='complete', arm=arm,
        frame_count=count, checkpoint_sha256=selected['checkpoint_sha256'],
        frozen_state_sha256=preflight['loading']['D']['state_sha256'],
        frozen_state_unchanged=True, pfn_weights_checked=True,
        observations=3*count, no_train=True, no_test=True,
        original_generator='v2xr_protocol_audit.predict.generate',
        original_evaluator='v2xr_protocol_audit.evaluate.replay'), immutable=True)
    return summary
