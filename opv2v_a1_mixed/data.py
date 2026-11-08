"""Aligned weather datasets and an exact, seeded frame-level sampling plan."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

WEATHERS = ("clean", "fog", "rain", "snow")
WEIGHTS = (6000, 1333, 1333, 1334)
SIZES = {"train": 6374, "validate": 1980}


def scan(roots, weather_root):
    manifest = json.loads((weather_root/"manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["source_roots"] != {k:str(v.resolve()) for k,v in roots.items()}:
        raise ValueError("Weather provenance does not match the Clean splits")
    scenes = {split:{p.name for p in root.iterdir() if p.is_dir()} for split,root in roots.items()}
    if scenes["train"] & scenes["validate"]:
        raise ValueError("Train/validation scenes overlap")
    test = roots["train"].parent/"test"
    if (scenes["train"] | scenes["validate"]) & {p.name for p in test.iterdir() if p.is_dir()}:
        raise ValueError("Test scene overlaps development splits")
    counts = {}
    digest = hashlib.sha256()
    for split, root in roots.items():
        counts[split] = dict.fromkeys(WEATHERS, 0)
        for weather in WEATHERS:
            base = root if weather == "clean" else weather_root/weather/split
            if {p.name for p in base.iterdir() if p.is_dir()} != scenes[split]:
                raise ValueError("Scene mismatch: " + str(base))
            expected = {p.relative_to(root) for p in root.glob("*/*/*.pcd")}
            actual = {p.relative_to(base) for p in base.glob("*/*/*.pcd")}
            if expected != actual:
                raise ValueError(f"PCD coverage mismatch {weather}/{split}: missing={len(expected-actual)}, extra={len(actual-expected)}")
            clean_labels = {p.relative_to(root) for p in root.glob("*/*/*.yaml") if "additional" not in p.name}
            labels = {p.relative_to(base) for p in base.glob("*/*/*.yaml") if "additional" not in p.name}
            if clean_labels != labels:
                raise ValueError("Label coverage mismatch: " + str(base))
            for rel in sorted(actual):
                p = base/rel
                if base.resolve() not in p.resolve(strict=True).parents or p.stat().st_size == 0:
                    raise ValueError("Invalid weather PCD " + str(p))
                st = p.stat()
                digest.update(json.dumps([split,weather,str(rel),st.st_size,st.st_mtime_ns]).encode())
            for rel in sorted(labels):
                p, original = base/rel, root/rel
                if p.resolve(strict=True) != original.resolve(strict=True) and p.read_bytes()!=original.read_bytes():
                    raise ValueError("Weather label does not match Clean: " + str(p))
                digest.update(hashlib.sha256(original.read_bytes()).digest())
            counts[split][weather] = len(actual)
            if weather != "clean" and len(actual) != manifest["conditions"][split]["pcd_files"][weather]:
                raise ValueError("PCD count differs from generator manifest")
            print("DATA_OK",split,weather,len(actual),"PCDs",len(labels),"labels",flush=True)
    return dict(pcd_counts=counts,inventory_sha256=digest.hexdigest(),
                generator_manifest_sha256=hashlib.sha256((weather_root/"manifest.json").read_bytes()).hexdigest())


def datasets(hypes, roots, weather_root, split):
    from opencood.data_utils.datasets import build_dataset
    result = {}
    reference = None
    for weather in WEATHERS:
        config = copy.deepcopy(hypes)
        path = roots[split] if weather == "clean" else weather_root/weather/split
        config["root_dir" if split == "train" else "validate_dir"] = str(path)
        data = build_dataset(config, visualize=False, train=(split == "train"))
        keys = [[(cav, tuple((t,str(Path(row["lidar"]).relative_to(path))) for t,row in rows.items() if t!="ego"))
                 for cav,rows in scenario.items()] for scenario in data.scenario_database.values()]
        if len(data) != SIZES[split] or (reference is not None and keys != reference):
            raise ValueError("Dataset frame/CAV order mismatch " + str(path))
        reference = keys
        result[weather] = data
    return result


def plan(size, seed, epoch):
    counts = [size*w//10000 for w in WEIGHTS]
    order = sorted(range(4),key=lambda i: (-(size*WEIGHTS[i]%10000),i))
    for index in order[:size-sum(counts)]:
        counts[index] += 1
    tags = [weather for weather,count in zip(WEATHERS,counts) for _ in range(count)]
    rng = random.Random(seed+epoch*100003)
    rng.shuffle(tags)
    indices = list(range(size))
    rng.shuffle(indices)
    return [(i,tags[i]) for i in indices],dict(zip(WEATHERS,counts))


class Mixed(Dataset):
    def __init__(self, data, plan, seed):
        self.data, self.plan, self.seed = data, plan, seed

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, position):
        index, weather = self.plan[position]
        # Replay the same point shuffles and augmentation in both arms.
        py, np_state = random.getstate(), np.random.get_state()
        try:
            seed = (self.seed+position) % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                return self.data[weather][index],weather
        finally:
            random.setstate(py)
            np.random.set_state(np_state)

    def collate(self, items):
        samples, tags = zip(*items)
        batch = self.data["clean"].collate_batch_train(samples)
        if batch is None:
            raise RuntimeError("A scheduled frame failed to collate")
        return batch,Counter(tags)
