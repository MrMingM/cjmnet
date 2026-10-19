"""Server-only synthetic model/gradient/state checks, never real-data training."""
import math
from pathlib import Path

from auditlib import SETTINGS, atomic_json, read_json, runtime


def run(args, output, own):
    runtime(args, read_json(Path(args.d1a_root) / 'protocol.json'))
    import torch
    from opencood.hypes_yaml import yaml_utils
    from opencood.models.point_pillar_intermediate import PointPillarIntermediate
    from opencood.loss.point_pillar_loss_opencood import PointPillarLossOpenCOOD
    from opencood.tools import train_utils
    from integration import PointPillarGsprOfficial, GsprOfficialLoss
    from results import baseline
    if not torch.cuda.is_available() or not torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('Synthetic checks require one server HIP GPU')
    baseline(args)  # Source reference and equivalent Clean O selection are fixed before new training.
    parameters = yaml_utils.load_yaml(str(Path(args.runtime_root) / 'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'))
    # Exercise the exact factory called by train.main(), rather than bypassing its guard.
    criterion = train_utils.create_loss(parameters)
    if not isinstance(criterion, GsprOfficialLoss) or not isinstance(criterion, PointPillarLossOpenCOOD):
        raise ValueError('Training loss factory does not use the configured official OpenCOOD loss')
    cfg = parameters['model']['args']
    torch.manual_seed(20261017)
    original = PointPillarIntermediate(cfg)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(20261017)
    model = PointPillarGsprOfficial(cfg)
    if not torch.equal(torch.get_rng_state(), expected_rng):
        raise ValueError('GSPR initialization perturbed shared RNG progression')
    reference = original.state_dict()
    if any(not torch.equal(value, model.state_dict()[k]) for k, value in reference.items()):
        raise ValueError('Shared official model initialization differs')
    original, model = original.cuda().eval(), model.cuda().eval()
    coords = torch.tensor([[0, 0, 80, 250], [0, 0, 81, 251],
                           [0, 0, 82, 252], [0, 0, 83, 253]], dtype=torch.int32, device='cuda:0')
    voxels = torch.zeros((4, 32, 4), device='cuda:0')
    for i in range(4):
        voxels[i, :4, 0] = coords[i, 3] * .4 - 140.8 + .2 + torch.linspace(-.1, .1, 4, device='cuda:0')
        voxels[i, :4, 1] = coords[i, 2] * .4 - 40 + .2
        voxels[i, :4, 2] = -1.0
        voxels[i, :4, 3] = .5
    batch = dict(processed_lidar=dict(voxel_features=voxels, voxel_coords=coords,
        voxel_num_points=torch.full((4,), 4, device='cuda:0', dtype=torch.int32)),
        record_len=torch.tensor([1], device='cuda:0'))
    model.force_unit_weights = True
    with torch.no_grad():
        a, b = original(batch), model(batch)
        for name in ('psm', 'rm'):
            torch.testing.assert_close(a[name], b[name], rtol=1e-5, atol=1e-5)
    del original, reference, a, b
    model.force_unit_weights = False
    model.train()
    result = model(batch)
    valid = result['point_valid_mask']
    if result['point_reliability'][~valid].abs().sum() != 0:
        raise ValueError('Padding points received nonzero reliability')
    if not ((result['point_reliability'][valid] >= .05) & (result['point_reliability'][valid] <= 1)).all():
        raise ValueError('Reliability outside registered soft-weight bounds')
    count, anchors, height, width = result['psm'].shape
    positive = torch.zeros((count, height, width, anchors), device='cuda:0')
    positive[:, height // 2, width // 2, 0] = 1
    target = dict(pos_equal_one=positive, neg_equal_one=1 - positive,
                  targets=torch.zeros((count, height, width, 7 * anchors), device='cuda:0'))
    detection = PointPillarLossOpenCOOD(parameters['loss']['args'])(result, target)
    detection_grads = torch.autograd.grad(detection, tuple(model.gspr.parameters()),
                                         retain_graph=True, allow_unused=True)
    active = [g for g in detection_grads if g is not None]
    if not active or not all(torch.isfinite(g).all() for g in active) or not any(g.abs().sum() > 0 for g in active):
        raise ValueError('Detection loss alone has no finite nonzero gradient to GSPR')
    loss = criterion(result, target)
    if not torch.isfinite(loss):
        raise ValueError('Synthetic training loss is nonfinite')
    if not math.isclose(criterion.loss_dict['total_loss'], loss.detach().item(), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('GSPR total loss logging differs from the actual first-batch loss')
    loss.backward()
    grads = [p.grad for p in model.gspr.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
        raise ValueError('GSPR does not receive finite nonzero gradients')
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    optimizer.step()
    output.mkdir(parents=True, exist_ok=True)
    file = output / 'synthetic_state.pth'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), criterion=criterion.state_dict()), file)
    saved = torch.load(file, map_location='cpu', weights_only=False)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    restored_criterion = train_utils.create_loss(parameters)
    restored_criterion.load_state_dict(saved['criterion'], strict=True)
    if any(not torch.equal(value.cpu(), saved['criterion'][k]) for k, value in restored_criterion.state_dict().items()):
        raise ValueError('Loss state failed strict restoration')
    if any(not torch.equal(v.detach().cpu(), saved['model'][k]) for k, v in model.state_dict().items()):
        raise ValueError('GSPR checkpoint failed exact full-state restoration')
    atomic_json(output / 'summary.json', dict(status='complete', synthetic_only=True, settings=SETTINGS,
        source_hashes=own, shared_initialization_equal=True, global_CPU_rng_equal=True,
        unit_weight_forward_matches_official=True, padded_weights_zero=True,
        gspr_gradients_finite_nonzero=True, detection_only_gradient_reaches_gspr=True,
        strict_model_optimizer_reload=True, official_training_loss_factory_checked=True,
        actual_detection_loss='opencood.loss.point_pillar_loss_opencood.PointPillarLossOpenCOOD',
        strict_criterion_buffer_reload=True,
        baseline_O_selection_matches_original_epoch20=True))
    print('GSPR SERVER SELF-TEST PASSED: official unit-weight equivalence, gradients and strict state reload.', flush=True)
