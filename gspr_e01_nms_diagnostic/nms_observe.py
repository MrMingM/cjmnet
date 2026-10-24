"""First-frame observers plus an explicitly requested, unscored BN counterfactual."""
from contextlib import ExitStack
from importlib import metadata
import os
from pathlib import Path
import traceback

from nms_common import atomic, read, sha, swap
from nms_numeric import array_hash, box_findings, footprint_area, json_numbers, map_indices, summary


class FirstFrameObserved(Exception):
    """Stop before AP/cache acceptance when the original first frame did not fail."""


class Observer:
    def __init__(self, output, trace_layers=False, bn_batch_stats=False):
        self.output = Path(output)
        self.arrays = {}
        self.context = dict(frame_index=-1, phase=None)
        self.details = dict(no_training=True, eligible_for_AP_or_selection=False,
            protocol_changes=bn_batch_stats, boxes_removed_by_observer=0, numerical_values_modified=bn_batch_stats,
            BN_current_batch_counterfactual=bn_batch_stats, score_GT_and_AP_functions_changed=False)
        self.bn_batch_stats = bn_batch_stats
        if bn_batch_stats and not trace_layers:
            raise ValueError('BN counterfactual requires full layer/state observations')
        self.polygon_indices = {}
        self.failure = None
        self.layer_trace = None
        if trace_layers:
            from nms_layers import LayerTrace
            self.layer_trace = LayerTrace(batch_statistics=bn_batch_stats)

    def capture(self, name, value):
        import numpy as np
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.dtype.kind == 'O':
            raise ValueError('Object arrays are not diagnostic tensor evidence')
        self.arrays[name] = value.copy()
        count = len(self.arrays.get(name + '_calls', [])) + 1
        self.arrays[name + '_calls'] = np.arange(count, dtype=np.int64)

    def install(self, stack, args, checkpoint):
        import numpy as np
        import torch
        from opencood.tools import inference_utils, train_utils
        from opencood.utils import box_utils, common_utils
        from opencood.data_utils.post_processor import voxel_postprocessor as voxel
        from observe import module_inventory
        cls = voxel.VoxelPostprocessor
        self.details['runtime'] = dict(hip=torch.version.hip, gpu=torch.cuda.get_device_name(0),
            physical_gpu=os.environ['ROCR_VISIBLE_DEVICES'], shapely=metadata.version('shapely'),
            imported_modules=module_inventory(Path(args.runtime_root)))
        original_load = torch.load
        state = original_load(checkpoint, map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
            raise ValueError('The failed local-training checkpoint must be a plain parameter dictionary')
        rows = {key: summary(value.detach().cpu().numpy()) for key, value in state.items()}
        self.details['checkpoint_tensors'] = dict(tensor_count=len(rows), tensors=rows,
            nonfinite_keys=[k for k, v in rows.items() if v['nan'] or v['positive_infinity'] or v['negative_infinity']],
            BN_running_variances={k: v for k, v in rows.items() if k.endswith('running_var')})
        del state

        def safe_load(path, *a, **kw):
            if hasattr(path, 'read') or Path(path).resolve() != Path(checkpoint).resolve() or sha(path) != self.details['checkpoint_sha256']:
                raise ValueError('Unexpected checkpoint read during diagnostic inference')
            kw['weights_only'] = True
            return original_load(path, *a, **kw)

        original_create = train_utils.create_model
        def create(*a, **kw):
            model = original_create(*a, **kw)
            self.details['model_class'] = type(model).__module__ + '.' + type(model).__name__
            def pre(module, inputs):
                if module.training or torch.is_grad_enabled():
                    raise ValueError('Diagnostic must preserve native eval/no_grad inference')
                actual = {key: array_hash(value.detach().cpu().numpy()) for key, value in module.state_dict().items()}
                if actual != {key: value['sha256'] for key, value in rows.items()}:
                    raise ValueError('Actual first-forward model differs from every original checkpoint tensor')
                self.details['native_model_all_tensors_loaded_exactly'] = True
            def forward(module, inputs, result):
                for name in ('psm', 'rm'):
                    self.capture(name, result[name])
            model.register_forward_pre_hook(pre)
            model.register_forward_hook(forward, prepend=True)
            if self.bn_batch_stats:
                from nms_bn import install_batch_statistics
                self.details['BN_counterfactual_modules'] = install_batch_statistics(model, stack)
            if self.layer_trace is not None:
                self.layer_trace.install(model, stack)
            return model

        original_infer = inference_utils.inference_intermediate_fusion
        def infer(batch, model, dataset):
            self.context['frame_index'] += 1
            if self.context['frame_index'] != 0:
                raise RuntimeError('Diagnostic is restricted to the first native inference frame')
            for name, value in batch['ego']['processed_lidar'].items():
                self.capture('input_' + name, value)
            for name in ('record_len', 'transformation_matrix'):
                self.capture('input_' + name, batch['ego'][name])
            self.details['frame'] = dict(index=0, frame_key=self.context['frame_keys'][0],
                condition=self.context['condition'], original_object_ids=list(map(str, batch['ego']['object_ids'])))
            result = original_infer(batch, model, dataset)
            self.details['native_inference_returned'] = True
            self.details['native_prediction_is_none'] = result[0] is None
            raise FirstFrameObserved('First frame returned without reproducing the original failure; no AP accepted')

        original_post = cls.post_process
        def post(instance, data_dict, output_dict):
            self.context['phase'] = 'postprocess'
            self.details['postprocess'] = dict(order=instance.params['order'],
                score_threshold=instance.params['target_args']['score_threshold'],
                nms_threshold=instance.params['nms_thresh'], cav_ids=list(data_dict))
            if len(data_dict) != 1 or instance.params['order'] != 'hwl':
                raise ValueError('Diagnostic mapping expects the frozen single-ego intermediate hwl path')
            try:
                return original_post(instance, data_dict, output_dict)
            finally:
                self.context['phase'] = None

        original_decode = cls.delta_to_boxes3d
        def decode(deltas, anchors):
            result = original_decode(deltas, anchors)
            if self.context['phase'] == 'postprocess':
                self.capture('anchors', anchors)
                self.capture('decoded', result)
            return result

        original_sigmoid = voxel.F.sigmoid
        def sigmoid(*a, **kw):
            result = original_sigmoid(*a, **kw)
            if self.context['phase'] == 'postprocess':
                self.capture('native_prob', result)
            return result

        original_corners = box_utils.boxes_to_corners_3d
        def corners(boxes, *a, **kw):
            result = original_corners(boxes, *a, **kw)
            if self.context['phase'] == 'postprocess':
                self.capture('selected', boxes)
                self.capture('native_corners', result)
            return result

        original_projection = box_utils.project_box3d
        def project(boxes, matrix, *a, **kw):
            result = original_projection(boxes, matrix, *a, **kw)
            if self.context['phase'] == 'postprocess':
                self.capture('transform', matrix)
                self.capture('projected', result)
            return result

        def mask_wrapper(original, name):
            def observe(*a, **kw):
                result = original(*a, **kw)
                if self.context['phase'] == 'postprocess':
                    self.capture(name, result)
                return result
            return observe

        original_nms = box_utils.nms_rotated
        def nms(boxes, scores, threshold):
            previous = self.context['phase']
            self.context['phase'] = 'nms'
            self.capture('nms_boxes', boxes)
            self.capture('nms_scores', scores)
            self.details['native_nms_called'] = True
            try:
                result = original_nms(boxes, scores, threshold)
                self.capture('native_nms_kept', result)
                self.details['native_nms_completed'] = True
                return result
            finally:
                self.context['phase'] = previous

        original_convert = common_utils.convert_format
        def convert(value):
            result = original_convert(value)
            if self.context['phase'] == 'nms':
                self.polygon_indices = {id(polygon): i for i, polygon in enumerate(result)}
                areas = np.asarray([p.area for p in result], dtype=np.float64)
                self.capture('native_polygon_areas', areas)
                self.details['NMS_geometry'] = dict(polygons=len(result),
                    zero_area=int(np.count_nonzero(areas == 0)), nonfinite_area=int(np.count_nonzero(~np.isfinite(areas))),
                    empty=sum(bool(p.is_empty) for p in result), invalid=sum(not p.is_valid for p in result))
            return result

        original_iou = common_utils.compute_iou
        def iou(polygon, candidates):
            try:
                return original_iou(polygon, candidates)
            except Exception as error:
                if self.context['phase'] == 'nms':
                    try:
                        self.capture_pair_failure(polygon, candidates, error)
                    except Exception as observation_error:
                        self.details['pair_observer_error'] = repr(observation_error)
                raise  # Preserve the original exception. No epsilon, filtering or replacement IoU.

        for obj, name, replacement in (
            (torch, 'load', safe_load), (train_utils, 'create_model', create),
            (inference_utils, 'inference_intermediate_fusion', infer),
            (cls, 'post_process', post), (cls, 'delta_to_boxes3d', staticmethod(decode)),
            (voxel.F, 'sigmoid', sigmoid), (box_utils, 'boxes_to_corners_3d', corners),
            (box_utils, 'project_box3d', project),
            (box_utils, 'remove_large_pred_bbx', mask_wrapper(box_utils.remove_large_pred_bbx, 'large_mask')),
            (box_utils, 'remove_bbx_abnormal_z', mask_wrapper(box_utils.remove_bbx_abnormal_z, 'z_mask')),
            (box_utils, 'nms_rotated', nms), (common_utils, 'convert_format', convert),
            (common_utils, 'compute_iou', iou)):
            stack.enter_context(swap(obj, name, replacement))

    def capture_pair_failure(self, polygon, candidates, error):
        import math
        problems = []
        for position, candidate in enumerate(candidates):
            try:
                union = float(polygon.union(candidate).area)
                if union != 0 and math.isfinite(union):
                    continue
                problems.append(dict(candidate_position=position,
                    picked_index=self.polygon_indices.get(id(polygon)),
                    candidate_index=self.polygon_indices.get(id(candidate)), union_area=union,
                    picked_area=float(polygon.area), candidate_area=float(candidate.area),
                    picked_empty=bool(polygon.is_empty), candidate_empty=bool(candidate.is_empty),
                    picked_valid=bool(polygon.is_valid), candidate_valid=bool(candidate.is_valid),
                    picked_WKT=polygon.wkt, candidate_WKT=candidate.wkt))
                if len(problems) == 12:
                    break
            except Exception as geometry_error:
                problems.append(dict(candidate_position=position, geometry_error=repr(geometry_error)))
                break
        self.failure = dict(exception_type=type(error).__name__, message=str(error),
                            candidate_count=len(candidates), denominator_problems=problems,
                            scope='actual failed original NMS IoU call; additional measurements only after the native exception')

    def finish(self, outcome, error=None):
        import numpy as np
        if self.layer_trace is not None:
            layer_report = self.layer_trace.report()
            atomic(self.output / 'layer_trace.json', json_numbers(layer_report))
            self.details['layer_trace_sha256'] = sha(self.output / 'layer_trace.json')
            print('LAYER_SUMMARY=' + __import__('json').dumps(json_numbers(layer_report['summary']),
                  ensure_ascii=False, allow_nan=False), flush=True)
        mapping = map_indices(self.arrays, self.details.get('postprocess', {}).get('score_threshold', .2))
        self.details['anchor_mapping'] = {k: v for k, v in mapping.items() if not isinstance(v, np.ndarray)}
        boxes = []
        if self.failure:
            indices = sorted({row[k] for row in self.failure['denominator_problems']
                              for k in ('picked_index', 'candidate_index') if row.get(k) is not None})
            for index in indices:
                item = dict(nms_index=index, score=float(self.arrays['nms_scores'][index]),
                    native_NMS_corners=self.arrays['nms_boxes'][index],
                    native_polygon_area=float(self.arrays['native_polygon_areas'][index]))
                if mapping['status'] == 'exact':
                    selected = int(mapping['nms_to_selected'][index])
                    anchor = int(mapping['selected_to_anchor'][selected])
                    decoded = self.arrays['selected'][selected]
                    raw_corners = self.arrays['native_corners'][selected]
                    projected = self.arrays['projected'][selected]
                    item.update(selected_index=selected, anchor_index=anchor, decoded_xyzhwlyaw=decoded,
                        raw_regression=self.arrays['rm'].transpose(0, 2, 3, 1).reshape(-1, 7)[anchor],
                        anchor_xyzhwlyaw=self.arrays['anchors'].reshape(-1, 7)[anchor],
                        native_corner_area=footprint_area(raw_corners), projected_area=footprint_area(projected),
                        findings=box_findings(decoded, raw_corners, projected))
                boxes.append(item)
        archive = self.output / 'first_frame_arrays.npz'
        temporary = archive.with_name(archive.name + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **self.arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, archive)
        report = dict(status='diagnostic_complete' if outcome in ('zero_union_reproduced', 'not_reproduced_first_frame') else 'diagnostic_failed',
            outcome=outcome, **self.details, native_failure=self.failure, implicated_boxes=boxes,
            arrays={k: summary(v) for k, v in self.arrays.items() if not k.endswith('_calls')},
            array_archive_sha256=sha(archive), error=error,
            limitation='Single first-frame checkpoint replay. Passive CPU observations may affect scheduling; a non-reproduction does not establish a fix or later-frame safety.')
        atomic(self.output / 'report.json', json_numbers(report))
        line = dict(outcome=outcome, BN_current_batch_counterfactual=self.bn_batch_stats,
            frame=self.details.get('frame'),
            NMS_geometry=self.details.get('NMS_geometry'),
            failing_boxes=[{k: b[k] for k in ('nms_index', 'score', 'decoded_xyzhwlyaw', 'raw_regression', 'findings') if k in b} for b in boxes[:4]],
            report=str(self.output / 'report.json'))
        print('SUMMARY=' + __import__('json').dumps(json_numbers(line), ensure_ascii=False, allow_nan=False), flush=True)
        return report


def execute(cli, request):
    """Use exactly the original E01 install/evaluate path; stop before scoring one frame."""
    from nms_common import original_e01
    original_e01()
    import e1_runtime
    args, audit, _, _, evaluation = e1_runtime.install(cli)
    audit.gpu()
    if audit.sources() != read(Path(cli.model_root) / 'contract.json')['sources']:
        raise ValueError('Exact original training/evaluation source identity differs')
    destination = Path(cli.destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Diagnostic inference destination must be fresh; original prediction caches are never reused/written')
    observer = Observer(Path(cli.report_root), trace_layers=getattr(cli, 'trace_layers', False),
                        bn_batch_stats=getattr(cli, 'bn_batch_stats', False))
    observer.details['checkpoint_sha256'] = request['checkpoint_sha256']
    observer.details['checkpoint'] = str(cli.checkpoint)
    import predict
    original_generate = predict.generate
    original_runtime = evaluation.inherited.runtime
    with ExitStack() as stack:
        def runtime(runtime_args, evidence):
            result = original_runtime(runtime_args, evidence)
            observer.install(stack, runtime_args, cli.checkpoint)
            return result
        def generate(condition, output, run_args, evidence, contract):
            observer.context.update(condition=condition, frame_keys=contract['frame_keys'])
            observer.details['evaluation_contract_identity'] = contract['identity']
            return original_generate(condition, output, run_args, evidence, contract)
        stack.enter_context(swap(evaluation.inherited, 'runtime', runtime))
        stack.enter_context(swap(predict, 'generate', generate))
        try:
            evaluation.evaluate(args, cli.checkpoint, destination, clean_only=True, smoke=True)
        except FirstFrameObserved:
            observer.finish('not_reproduced_first_frame')
            return
        except Exception as error:
            trace = traceback.format_exc()
            (Path(cli.report_root) / 'original_traceback.txt').write_text(trace, encoding='utf-8', newline='\n')
            reproduced = (isinstance(error, ZeroDivisionError) and observer.failure is not None
                and any(row.get('union_area') == 0 for row in observer.failure['denominator_problems']))
            observer.finish('zero_union_reproduced' if reproduced else 'other_failure', error=repr(error))
            if not reproduced:
                raise
            return
    raise RuntimeError('Unexpected evaluator completion; no first-frame inference observed')
