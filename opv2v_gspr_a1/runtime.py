"""OPV2V integration and epoch-boundary recovery."""
import copy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT.parent)]


def install():
    from attfuse_gspr import point_pillar_gspr_attfuse as original
    from opencood.data_utils import datasets
    from opv2v_adapter import Model, Dataset

    sys.modules["opencood.models.point_pillar_gspr_attfuse"] = original
    original.PointPillarGsprAttfuse = Model
    datasets.__all__["IntermediateFusionDataset"] = Dataset


def commit(directory, model, optimizer, history):
    path = Path(directory) / "resume.pth"
    temp = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    history=history, python=random.getstate(), numpy=np.random.get_state(),
                    cpu=torch.get_rng_state(), gpu=torch.cuda.get_rng_state_all()), temp)
    os.replace(temp, path)


def restore(directory, model, optimizer):
    path = Path(directory) / "resume.pth"
    if not path.exists():
        return 0, []
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["gpu"])
    return len(state["history"]), state["history"]


def audit_step(model, optimizer):
    groups = {"gspr": model.gspr, "a1": model.a1, "backbone": model.backbone}
    active = {key: [p for p in module.parameters() if p.requires_grad]
              for key, module in groups.items()}
    report = {}
    before = {}
    for key, params in active.items():
        grads = [p.grad for p in params if p.grad is not None]
        if params and (not grads or any(not torch.isfinite(g).all() for g in grads)):
            raise ValueError("Missing/nonfinite gradient in " + key)
        report[key] = {"gradient_norm": sum(float(g.norm()) ** 2 for g in grads) ** .5}
        before[key] = [p.detach().clone() for p in params]
    optimizer.step()
    for key, params in active.items():
        changed = sum(not torch.equal(p, old) for p, old in zip(params, before[key]))
        if params and (not changed or not report[key]["gradient_norm"]):
            raise ValueError("No update in " + key)
        report[key]["updated_tensors"] = changed
    return report


def smoke(model, batch, optimizer, directory):
    from opencood.loss.point_pillar_loss import PointPillarLoss
    from opencood.hypes_yaml.yaml_utils import load_yaml

    model.eval()
    model.enabled = False
    with torch.inference_mode():
        baseline = model(batch)
        repeat = model(batch)
        model.enabled = True
        model.check_zero_residual = True
        zero = model(batch)
        model.check_zero_residual = False
    parity = {}
    for key in ("psm", "rm"):
        torch.testing.assert_close(repeat[key], baseline[key], rtol=1e-5, atol=5e-5)
        torch.testing.assert_close(zero[key], baseline[key], rtol=1e-5, atol=5e-5)
        parity[key] = dict(original_repeat_max_abs=float((repeat[key]-baseline[key]).abs().max()),
                           a1_zero_max_abs=float((zero[key]-baseline[key]).abs().max()))
    print("SMOKE PARITY", json.dumps(parity), flush=True)
    del baseline, repeat, zero
    model.train()
    loss_fn = PointPillarLoss(load_yaml(str(Path(directory) / "config.yaml"))["loss"]["args"])
    updates = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = loss_fn(output, batch["label_dict"])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite detection-only smoke loss")
        loss.backward()
        updates.append(audit_step(model, optimizer))
    xyz = torch.randn(2, 5, 3, device=loss.device)
    valid = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]], device=loss.device)
    features = torch.randn(2, 5, 64, device=loss.device)
    base = torch.randn(2, 64, device=loss.device)
    reliability = torch.full_like(valid, .8, dtype=features.dtype)
    uncertainty = 1 - reliability
    ranges = xyz.norm(dim=-1)
    with torch.inference_mode():
        model.a1.eval()
        clean = model.a1(features, base, xyz, valid, reliability, uncertainty, ranges)
        dirty = model.a1(features.masked_fill(~valid[..., None], 1000), base,
                         xyz.masked_fill(~valid[..., None], 1000), valid,
                         reliability.masked_fill(~valid, 1000), uncertainty.masked_fill(~valid, 1000),
                         ranges.masked_fill(~valid, 1000))
        torch.testing.assert_close(clean, dirty, rtol=0, atol=0)
    commit(directory, model, optimizer, updates)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    start, records = restore(directory, model, optimizer)
    assert start == 3 and records == updates
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    (Path(directory) / "SMOKE.json").write_text(json.dumps(dict(updates=updates, parity=parity, exact_pillar_identity=True), indent=2))
