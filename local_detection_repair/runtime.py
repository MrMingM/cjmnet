"""Dataset, freezing, checkpoint and reproducibility utilities."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

TRAIN_ROOT = Path("/data/scd/datasets/opv2v_official_data_dumping/train")
VALIDATION_ROOT = Path("/data/scd/datasets/opv2v_official_data_dumping/validate")
TEST_ROOTS = {
    "clean": Path("/data/scd/datasets/opv2v_official_data_dumping/test"),
    "fog": Path("/data/cjm/datasets/opv2v-w/fog/test"),
    "rain": Path("/data/cjm/datasets/opv2v-w/rain/test"),
    "snow": Path("/data/cjm/datasets/opv2v-w/snow/test"),
}
LOG_ROOT = Path("/data/cjm/datasets/logs")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def safe_output(path, resume=False):
    target = Path(path).resolve()
    allowed = LOG_ROOT.resolve()
    if target == allowed or allowed not in target.parents:
        raise ValueError("output must be a subdirectory of /data/cjm/datasets/logs")
    if resume:
        if not target.is_dir():
            raise FileNotFoundError("resume output directory does not exist")
    else:
        target.mkdir(parents=True, exist_ok=False)
    return target


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except Exception:
        return "unavailable"


def load_config(experiment, frontend_config):
    import yaml
    from gspr_review.runtime import load_config as load_review_config

    options = yaml.safe_load(Path(experiment).read_text(encoding="utf-8"))
    if int(options.get("workers", -1)) != 0:
        raise ValueError("v1 requires workers=0 so weather RNG order is explicit")
    if int(options.get("batch_size", -1)) != 1:
        raise ValueError("v1 requires batch_size=1 for candidate/action accounting")
    if tuple(options.get("train_weathers", ())) != ("clean", "fog", "rain", "snow"):
        raise ValueError("v1 expects explicit clean/fog/rain/snow training")
    _, hypes = load_review_config(experiment, frontend_config)
    if Path(hypes["root_dir"]).resolve() != TRAIN_ROOT.resolve():
        raise ValueError("training root must be official OPV2V train")
    if Path(hypes["validate_dir"]).resolve() != VALIDATION_ROOT.resolve():
        raise ValueError("validation root must be official OPV2V validate")
    return options, hypes


def weather_hypes(hypes, weather):
    local = copy.deepcopy(hypes)
    if weather == "clean":
        local.pop("weather_augmentation", None)
    elif weather in ("fog", "rain", "snow"):
        if "weather_augmentation" not in local:
            raise ValueError("weather augmentation configuration is missing")
        local["weather_augmentation"]["mode"] = "physics_" + weather
    else:
        raise ValueError(weather)
    return local


def benchmark_hypes(hypes, weather):
    local = copy.deepcopy(hypes)
    local["validate_dir"] = str(TEST_ROOTS[weather])
    local.pop("weather_augmentation", None)
    local["data_augment"] = []
    if not TEST_ROOTS[weather].is_dir():
        raise FileNotFoundError("official/fixed test root missing: " + str(TEST_ROOTS[weather]))
    return local


def _scene_frames(len_record, scenes):
    if scenes is None:
        return list(range(int(len_record[-1]))) if len_record else []
    frames = []
    for scene in scenes:
        if scene < 0 or scene >= len(len_record):
            raise ValueError("scene index outside dataset")
        start = int(len_record[scene - 1]) if scene else 0
        stop = int(len_record[scene])
        frames.extend(range(start, stop))
    return frames


def make_loader(hypes, options, split, weather, scenes=None, shuffle=False,
                seed=None, smoke=0, benchmark=False):
    import torch
    from torch.utils.data import DataLoader, Subset
    from gspr_communication.dataset_adapter import CommunicationDataset
    from gspr_communication.runtime import seed_worker

    if split not in ("train", "validation", "test"):
        raise ValueError(split)
    local = benchmark_hypes(hypes, weather) if benchmark else weather_hypes(hypes, weather)
    train = split == "train"
    ds = CommunicationDataset(local, train=train)
    indices = _scene_frames(ds.len_record, scenes)
    if smoke:
        indices = indices[:int(smoke)]
    if not indices:
        raise ValueError("empty dataset selection")
    generator = torch.Generator().manual_seed(
        int(options["seed"] if seed is None else seed)
    )
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=1,
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=0,
        collate_fn=ds.collate_batch_train if train else ds.collate_batch_test,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return ds, loader, indices


def scene_split(scene_count, fraction, seed):
    if scene_count < 2:
        raise ValueError("need at least two training scenes")
    fraction = float(fraction)
    if not 0 < fraction < 1:
        raise ValueError("repair_scene_fraction must be between 0 and 1")
    order = list(range(scene_count))
    random.Random(int(seed)).shuffle(order)
    cut = min(max(int(round(scene_count * fraction)), 1), scene_count - 1)
    repair = sorted(order[:cut])
    selector = sorted(order[cut:])
    if set(repair) & set(selector) or sorted(repair + selector) != list(range(scene_count)):
        raise AssertionError("scene split is not disjoint/exhaustive")
    return repair, selector


def input_branch(batch, weather, benchmark=False):
    ego = batch["ego"]
    key = "processed_lidar" if benchmark or weather == "clean" else "processed_lidar_weather"
    if key not in ego:
        raise ValueError("requested LiDAR branch unavailable: " + key)
    return {"processed_lidar": ego[key], "record_len": ego["record_len"]}


def seed_all(seed):
    from gspr_communication.runtime import seed_all as project_seed_all
    project_seed_all(int(seed))


def device():
    from gspr_communication.runtime import device as project_device
    return project_device()


def load_frontend(hypes, options, checkpoint, target):
    import torch
    from gspr_communication.model import CommunicationModel

    digest = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = CommunicationModel(
        hypes["model"]["args"], dict(options["communication"], variant="full")
    )
    model.load_frontend(state)
    model.requires_grad_(False)
    model.eval()
    if sha256(checkpoint) != digest:
        raise RuntimeError("frontend checkpoint changed while loading")
    assert_frozen(model)
    return model.to(target), digest


def assert_frozen(frontend):
    bad_grad = [name for name, p in frontend.named_parameters() if p.requires_grad]
    bad_mode = [name for name, module in frontend.named_modules()
                if module.training and name]
    if bad_grad:
        raise AssertionError("frontend parameters require grad: " + ",".join(bad_grad[:5]))
    if frontend.training or bad_mode:
        raise AssertionError("frontend or BatchNorm-bearing submodules left in train mode")
    for p in frontend.parameters():
        if p.grad is not None:
            raise AssertionError("frozen frontend accumulated gradients")


def optimizer_only(optimizer, modules):
    expected = {id(p) for module in modules for p in module.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if actual != expected:
        raise AssertionError("optimizer parameter set includes missing/foreign parameters")


def rng_state():
    import numpy as np
    import torch
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(value, partial)
    partial.replace(path)


def contract(experiment, frontend_config, frontend_checkpoint, options,
             frontend_sha, repair_scenes, selector_scenes):
    return {
        "schema": 1,
        "experiment_sha256": sha256(experiment),
        "frontend_config_sha256": sha256(frontend_config),
        "frontend_checkpoint_sha256": frontend_sha,
        "frontend_checkpoint_path": str(Path(frontend_checkpoint).resolve()),
        "git_sha": git_sha(),
        "seed": int(options["seed"]),
        "workers": int(options["workers"]),
        "train_root": str(TRAIN_ROOT),
        "validation_root": str(VALIDATION_ROOT),
        "repair_scenes": list(repair_scenes),
        "selector_scenes": list(selector_scenes),
        "weather_augmentation": copy.deepcopy(options.get("weather_augmentation")),
        "test_roots": {key: str(value) for key, value in TEST_ROOTS.items()},
    }


def ensure_contract(saved, current):
    keys = (
        "schema", "experiment_sha256", "frontend_config_sha256",
        "frontend_checkpoint_sha256", "seed", "workers", "train_root",
        "validation_root", "repair_scenes", "selector_scenes",
        "weather_augmentation",
    )
    mismatch = [key for key in keys if saved.get(key) != current.get(key)]
    if mismatch:
        raise ValueError("checkpoint contract mismatch: " + ", ".join(mismatch))
