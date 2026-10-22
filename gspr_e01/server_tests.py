"""Small real-data checks; called only by the remote total launcher."""
import copy
from pathlib import Path
from e01_common import atomic


def check(cli):
    from e1_runtime import install
    import torch
    args, audit, training, checkpoint, _ = install(cli)
    audit.gpu()
    manifest, _ = training.prepare(args)
    hypes = training.config(args, cli.model_root)
    import models
    import data
    from opencood.tools import train_utils
    results = {}
    shared = None
    for variant in ('C', 'D'):
        checkpoint.seed()
        dataset = data.build_train(hypes, manifest)
        items = [dataset[i] for i in range(8)]
        batch = dataset.collate_batch_train(items)
        audits = batch.pop('_audit')
        # Re-seed model creation independently so the shared initialization can be inspected.
        checkpoint.seed()
        model = models.create(hypes, variant)
        if variant == 'C':
            shared = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            from e1_runtime import exact
            exact(shared, {k: v for k, v in model.state_dict().items() if not k.startswith('gspr.')})
        state = copy.deepcopy(model.state_dict())
        model.load_state_dict(state, strict=True)
        if any(not torch.equal(model.state_dict()[k], state[k]) for k in state):
            raise ValueError('Factory state strict load differs')
        model = model.to('cuda:0').train()
        criterion = models.criterion(hypes, variant, False).to('cuda:0')
        batch = train_utils.to_device(batch, torch.device('cuda:0'))
        batch['ego']['epoch'] = 0
        output = model(batch['ego'])
        loss = criterion(output, batch['ego']['label_dict'])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite real-data self-test loss')
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise ValueError('Factory backward failed')
        gspr_grads = [p.grad for n, p in model.named_parameters() if n.startswith('gspr.') and p.grad is not None]
        if variant == 'D' and (not gspr_grads or not any(bool(g.abs().sum() > 0) for g in gspr_grads)):
            raise ValueError('No GSPR gradient on actual training data')
        results[variant] = dict(loss=float(loss.detach()), psm=list(output['psm'].shape),
            rm=list(output['rm'].shape), device=str(output['psm'].device), gradients_finite=True,
            GSPR_gradient_present=bool(gspr_grads), actual_weather_reads=[r['reads'] for r in audits],
            strict_factory_state_load=True)
        del model, criterion, batch, output, loss, grads, gspr_grads, items
        torch.cuda.empty_cache()
    atomic(Path(cli.model_root) / 'self_test.json', dict(status='complete', results=results,
        shared_initialization_exact=True, optimizer_updates=0, scientific_result=False))
