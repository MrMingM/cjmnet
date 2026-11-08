"""Use the official model, loader, loss and optimizer with an explicit input guard."""
from contextlib import contextmanager
import os
from pathlib import Path

from common import read_json
from inputs import DATA, TRAIN, VALIDATE, WEATHER, check_dataset

RUNTIME = DATA / 'v2xr-official-runtime-v1'
CONFIG = RUNTIME / 'opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_AttFuse.yaml'
SEED = 20261017
BATCH = 8
WORKERS = 16
EPOCHS = 20
EMPTY_POLICY = os.environ.get('LR0_EMPTY_POLICY', 'zero')
if EMPTY_POLICY not in ('reject', 'zero'):
    raise ValueError('LR0_EMPTY_POLICY must be reject or zero')


def config():
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())
    for key in ('eps', 'weight_decay'):
        cfg['optimizer']['args'][key] = float(cfg['optimizer']['args'][key])
    if (cfg['model']['core_method'] != 'point_pillar_intermediate_lrf'
            or cfg['train_params'] != dict(batch_size=BATCH, epoches=30, eval_freq=1, save_freq=1)
            or cfg['loss']['core_method'] != 'point_pillar_loss'
            or cfg['optimizer'] != dict(core_method='Adam', lr=.002,
                                        args=dict(eps=1e-10, weight_decay=1e-4))
            or cfg['lr_scheduler'] != dict(core_method='multistep', gamma=.1, step_size=[10, 15])):
        raise ValueError('Official dual-modal configuration differs')
    cfg.update(name='V2XR_LR0', root_dir=str(TRAIN), validate_dir=str(VALIDATE), train_sim=True)
    cfg['train_params']['epoches'] = EPOCHS
    return cfg


@contextmanager
def no_synthetic_points():
    """Abort if the official per-car empty-voxel loop tries to draw random points."""
    import random
    from opencood.data_utils.datasets.opv2v import intermediate_fusion_dataset as native

    original = native.random.choices

    def choices(*args, **kwargs):
        import sys
        frame = sys._getframe(1)
        if frame.f_code is native.IntermediateFusionDataset.get_item_single_car.__code__:
            loc = frame.f_locals
            empty = [sensor for sensor in ('lidar', 'radar')
                     if loc[f'processed_{sensor}']['voxel_features'].shape[0] == 0]
            car = loc['selected_cav_base']['id']
            raise RuntimeError(f'Official loader tried to fabricate points: car={car}, empty={empty}')
        return original(*args, **kwargs)

    native.random.choices = choices
    try:
        yield
    finally:
        native.random.choices = original


class GuardedDataset:
    def __init__(self, native, keys):
        self.native = native
        self.keys = keys

    def __len__(self):
        return len(self.native)

    def __getitem__(self, index):
        try:
            with no_synthetic_points():
                return self.native[index]
        except Exception as exc:
            raise RuntimeError(f'Data read failed at requested frame {self.keys[index]}') from exc


def build(config_path, split='validate', condition='clean', visualize=False, lidar_only=False):
    from opencood.hypes_yaml import yaml_utils
    from opencood.data_utils.datasets import build_dataset

    cfg = yaml_utils.load_yaml(str(config_path))
    cfg.update(eval_sim=condition != 'clean', sim_weather=WEATHER[condition] or WEATHER['fog'])
    native = build_dataset(cfg, visualize=visualize, train=split == 'train')
    keys = check_dataset(native, split, condition)
    if EMPTY_POLICY == 'zero':
        from empty_input import allow_empty_observations
        allow_empty_observations(native)
    if lidar_only:
        native.use_Radar = False
    return GuardedDataset(native, keys), cfg


def loader(dataset, training=False, evaluation=False, limit=None):
    from torch.utils.data import DataLoader, Subset

    native = dataset.native
    opts = dict(batch_size=1, num_workers=WORKERS, shuffle=False, drop_last=False,
                pin_memory=False, collate_fn=native.collate_batch_test)
    if not evaluation:
        opts.update(batch_size=BATCH if training else 1, shuffle=training, drop_last=training,
                    pin_memory=True, prefetch_factor=4, collate_fn=native.collate_batch_train)
    if limit is not None:
        dataset = Subset(dataset, range(limit))
    return DataLoader(dataset, **opts)


def create_model(cfg):
    from opencood.tools import train_utils

    model = train_utils.create_model(cfg).to('cuda:0')
    if EMPTY_POLICY == 'zero':
        from empty_input import preserve_empty_vehicle_slots
        preserve_empty_vehicle_slots(model)
    return model


def components(cfg, batches):
    from opencood.tools import train_utils

    model = create_model(cfg)
    loss = train_utils.create_loss(cfg).to('cuda:0')
    optimizer = train_utils.setup_optimizer(cfg, model)
    scheduler = train_utils.setup_lr_schedular(cfg, optimizer, n_iter_per_epoch=batches)
    return model, loss, optimizer, scheduler

