"""Authorized BN counterfactual on disposable diagnostic models; no gradients, saves or AP."""
from contextlib import contextmanager
from types import MethodType
import math


@contextmanager
def instance_forward(module, replacement):
    """Restore even the absence of an instance override, including after native failure."""
    existed = 'forward' in module.__dict__
    previous = module.__dict__.get('forward')
    module.forward = MethodType(replacement, module)
    try:
        yield
    finally:
        if existed:
            module.forward = previous
        else:
            del module.forward


def current_batch_forward(module, value):
    import torch
    from torch.nn import functional
    if torch.is_grad_enabled() or module.training or not module.track_running_stats:
        raise ValueError('BN counterfactual requires tracked eval modules under original no_grad')
    # None prevents any running-mean/variance updates; bypassing module.forward also preserves the counter.
    return functional.batch_norm(value, running_mean=None, running_var=None,
        weight=module.weight, bias=module.bias, training=True, momentum=0., eps=module.eps)


def install_batch_statistics(model, stack):
    from torch.nn.modules.batchnorm import _BatchNorm
    names = []
    for name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            if not module.track_running_stats or module.running_mean is None or module.running_var is None:
                raise ValueError('Expected original tracked BatchNorm: ' + name)
            stack.enter_context(instance_forward(module, current_batch_forward))
            names.append(name)
    if not names:
        raise ValueError('No tracked BN modules for counterfactual')
    return names


def compare_pair(native, batch, native_request, batch_request, native_trace, batch_trace):
    """All same-input/state prerequisites are exact. Output changes are descriptive, never a gate."""
    fields = ('checkpoint_sha256', 'training_contract_sha256', 'original_request_identity',
              'original_request_sha256', 'original_E1_gpu', 'original_threads', 'sources',
              'model_root', 'train_root', 'original_run_root')
    for field in fields:
        if native_request[field] != batch_request[field]:
            raise ValueError('BN comparison original identity differs: ' + field)
    if native_request.get('bn_batch_stats') or batch_request.get('bn_batch_stats') is not True:
        raise ValueError('BN comparison requires native then current-batch modes')
    for report, changed in ((native, False), (batch, True)):
        if report['eligible_for_AP_or_selection'] or report['protocol_changes'] != changed:
            raise ValueError('Counterfactual must remain explicitly unscored and labeled')
        if report.get('BN_current_batch_counterfactual') != changed:
            raise ValueError('Report BN mode does not match requested intervention')
        if not report.get('no_training') or not report.get('native_model_all_tensors_loaded_exactly'):
            raise ValueError('No-training and strict loaded-model evidence required')
    if native['frame'] != batch['frame'] or native['postprocess'] != batch['postprocess']:
        raise ValueError('BN comparison frame/order/postprocess differs')
    keys = {key for key in native['arrays'] if key.startswith('input_')}
    required = {'input_record_len', 'input_transformation_matrix', 'input_voxel_features',
                'input_voxel_coords', 'input_voxel_num_points'}
    if not required <= keys or keys != {key for key in batch['arrays'] if key.startswith('input_')}:
        raise ValueError('Missing or different first-frame processed inputs')
    for key in sorted(keys):
        if native['arrays'][key] != batch['arrays'][key]:
            raise ValueError('BN comparison first-frame tensor differs: ' + key)
    before = None
    for trace in (native_trace, batch_trace):
        row = trace['summary']
        if row.get('all_model_state_exact_after_forward') is not True or row.get('BN_buffers_exact_after_forward') is not True:
            raise ValueError('BN comparison must prove model parameters and buffers stayed unchanged')
        current = row['model_state_before_sha256']
        if not isinstance(current, str) or len(current) != 64 or before is not None and before != current:
            raise ValueError('BN comparison did not start from identical parameters/buffers')
        before = current
        if row.get('observation_errors') or row.get('state_observation_errors') or row.get('BN_sample_formula_failures'):
            raise ValueError('Incomplete or failed layer/BN formula evidence; inspect layer_trace.json')
        if not row.get('BN_observed_calls'):
            raise ValueError('No observed BN calls')
    if native_trace['summary']['BN_observed_calls'] != batch_trace['summary']['BN_observed_calls']:
        raise ValueError('Observed BN call count differs')
    if not batch.get('BN_counterfactual_modules') or set(batch['BN_counterfactual_modules']) != set(
            row['module'] for row in batch_trace['rows'] if 'BN' in row):
        raise ValueError('Not every overridden BN was observed')
    for trace, expected in ((native_trace, 'stored_running'), (batch_trace, 'current_batch_population')):
        if any(row['BN']['sampled_normalization_formula']['basis'] != expected
               for row in trace['rows'] if 'BN' in row):
            raise ValueError('BN sampled formula used a different statistics basis')

    def metrics(report):
        arrays = report['arrays']
        return dict(outcome=report['outcome'],
            psm_abs_max=arrays['psm']['finite_abs_max'], rm_abs_max=arrays['rm']['finite_abs_max'],
            raw_output_nonfinite={key: sum(arrays[key][field] for field in
                ('nan', 'positive_infinity', 'negative_infinity')) for key in ('psm', 'rm')},
            native_nms_called=bool(report.get('native_nms_called')),
            NMS_geometry=report.get('NMS_geometry'),
            native_prediction_is_none=report.get('native_prediction_is_none'),
            native_inference_returned=bool(report.get('native_inference_returned')))
    left, right = metrics(native), metrics(batch)
    ratios = {}
    for key in ('psm_abs_max', 'rm_abs_max'):
        a, b = left[key], right[key]
        ratios[key] = (a / b if isinstance(a, (int, float)) and isinstance(b, (int, float))
                       and math.isfinite(a) and math.isfinite(b) and b > 0 else None)
        if ratios[key] is not None and not math.isfinite(ratios[key]):
            ratios[key] = None
    return dict(status='paired_diagnostic_complete', same_input_exact=True,
        same_start_model_state_exact=True, model_parameters_and_buffers_unchanged=True,
        checkpoint_sha256=native_request['checkpoint_sha256'], frame=native['frame'],
        native=left, current_batch=right, native_divided_by_current_batch_output_scale=ratios,
        input_tensor_hashes={key: native['arrays'][key]['sha256'] for key in sorted(keys)},
        no_training=True, eligible_for_AP_or_selection=False, scientific_evaluation_complete=False,
        interpretation='Paired first-frame intervention only. Output scale and native NMS changes test the contribution of stored BN statistics on this frame; they do not establish a formal repair, future-frame safety, AP gain or a unique cause. Missing NMS geometry is not a measured zero-area count of zero.')
