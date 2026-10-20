"""Keep native V2X-R GT path; add independently packed local-CAV inputs for UECP."""
from pathlib import Path

import numpy as np

from .common import read, stream_seed
from .runtime import loader_config
from .uncertainty_targets import density, pairwise


def padding_mask(processed):
    return (processed['voxel_features'] == 0).all(axis=(1, 2)) & (processed['voxel_coords'] == 0).all(axis=1)


class Dataset:
    def __init__(self, output, split='train_core', epoch=0, native=None):
        self.root, self.split, self.epoch = Path(output), split, epoch
        self.config = read(self.root / 'config.json')
        self.manifest = read(self.root / 'data_manifest.json')
        self.base = None
        if native is None:
            from v2xr_raygen_e0e3.data import Dataset as NativeTraining
            self.base = NativeTraining(output, epoch=epoch, auxiliary=False)
            self.native = self.base.native
            self.keys = self.base.keys
            self.indices = self.manifest[split]
        else:
            self.native = native
            self.indices = list(range(len(native)))
            self.keys = []
            for db in native.scenario_database.values():
                ego = next(k for k, r in db.items() if r['ego'])
                rows = db[ego]
                first = next(v for k, v in rows.items() if k != 'ego')
                scene = Path(first['yaml']).parent.parent.name
                self.keys += [[scene, t, ego] for t in rows if t != 'ego']
        self.native.post_processor.params['dir_args'] = loader_config(output)['postprocess']['dir_args']
        self.pending = []
        single = self.native.get_item_single_car
        def capture(row, ego_pose, ego_params):
            result = single(row, ego_pose, ego_params)
            self.pending.append(self.local(row, ego_params))
            return result
        self.native.get_item_single_car = capture

    def __getattr__(self, key):
        return getattr(self.native, key)

    def __len__(self):
        return len(self.indices)

    def local(self, row, ego_params):
        from opencood.utils import pcd_utils, box_utils
        from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
        raw = row['lidar_np'].copy()
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite current weather XYZI')
        raw = pcd_utils.mask_points_by_range(raw, self.config['range'])
        raw = pcd_utils.mask_ego_points(raw)
        matrix = np.asarray(row['params']['transformation_matrix'])
        projected = raw.copy()
        projected[:, :3] = box_utils.project_points_by_matrix_torch(raw[:, :3], matrix)
        # Use the native FOV decision on an added index column, then take exact original local rows.
        indexed = np.concatenate((projected, np.arange(len(raw))[:, None]), axis=1)
        visible = self.native.mask_ego_fov_flag(row, indexed, ego_params)
        raw = raw[visible[:, 4].astype(int)]
        rng = np.random.RandomState(stream_seed(self.config['seed'], self.epoch,
            [self.current_uid, str(row['id'])], 'local_point_order'))
        raw = raw[rng.permutation(len(raw))]
        packed = self.native.pre_processor.preprocess(raw.astype(np.float32)) if len(raw) else None
        if packed is None or not len(packed['voxel_features']):
            packed = dict(voxel_features=np.zeros((1, 32, 4), np.float32),
                          voxel_coords=np.zeros((1, 3), np.int32), voxel_num_points=np.ones(1, np.int32))
        count_map, target = density(raw, self.config['range'], cap=self.config['density_cap'])
        centers, mask, ids = self.native.post_processor.generate_object_center([row], row['params']['lidar_pose'])
        boxes = centers[mask == 1]
        hits = points_in_boxes_cpu(raw[:, :3], boxes[:, [0, 1, 2, 5, 4, 3, 6]]) if len(boxes) and len(raw) else np.zeros((len(boxes), len(raw)))
        valid_boxes = boxes[np.asarray(hits).sum(1) > 0]
        single_centers = np.zeros_like(centers); single_mask = np.zeros_like(mask)
        single_centers[:len(valid_boxes)] = valid_boxes; single_mask[:len(valid_boxes)] = 1
        anchors = self.native.post_processor.generate_anchor_box()
        if len(valid_boxes):
            single_labels = self.native.post_processor.generate_label(gt_box_center=single_centers, anchors=anchors, mask=single_mask)
        else:
            shape = anchors.shape[:3]
            single_labels = dict(pos_equal_one=np.zeros(shape), neg_equal_one=np.ones(shape), targets=np.zeros((*shape[:2], shape[2] * 7)))
        return dict(packed=packed, density=count_map, target=target, transform=matrix, single=single_labels,
                    cav=str(row['id']), raw=raw, raw_original=row['lidar_np'], uid=self.current_uid,
                    relative='/'.join([self.current_uid[0], str(row['id']), self.current_uid[1] + self.config['weather'][self.condition] + '.pcd']))

    def get_global(self, index, condition=None):
        self.pending = []
        self.current_uid = self.keys[index]
        self.condition = condition or (self.base.schedule(index) if self.base else 'clean')
        if self.base:
            item = self.base.get(index, force=condition)
        else:
            self.condition = 'clean' if not self.native.eval_sim else {'_fog_0.060': 'fog', '_snow_2.5_2.0': 'snow'}[self.native.sim_weather]
            item = self.native[index]
        if len(self.pending) != item['ego']['cav_num']:
            raise ValueError('UECP and official GT paths selected different CAV lists')
        item['_local'] = self.pending
        return item

    def __getitem__(self, index):
        actual = self.indices[index]
        try:
            return self.get_global(actual)
        except Exception as exc:
            raise RuntimeError('Data UID=' + repr(self.keys[actual])) from exc

    def model_inputs(self, items, training=False):
        import torch
        locals_ = [row for item in items for row in item['_local']]
        packed = self.native.merge_features_to_dict([row['packed'] for row in locals_])
        voxels = self.native.pre_processor.collate_batch(packed)
        empty = (voxels['voxel_features'] == 0).flatten(1).all(1) & (voxels['voxel_coords'][:, 1:] == 0).all(1)
        counts = np.stack([r['density'] for r in locals_])[:, None]
        inputs = dict(processed_lidar=voxels, record_len=torch.tensor([len(i['_local']) for i in items]),
            pairwise_t_matrix=torch.tensor(np.stack([pairwise([r['transform'] for r in i['_local']]) for i in items]), dtype=torch.float32),
            point_counts=torch.from_numpy(counts), empty_voxels=empty,
            quality_seeds=[stream_seed(self.config['seed'], 0, [r['uid'], r['cav']], 'shuffle_quality') for r in locals_])
        if training:
            inputs['un_maps'] = torch.from_numpy(np.stack([r['target'] for r in locals_]))
        return inputs

    def collate_batch_train(self, items):
        native = self.native.collate_batch_train(items)
        if native is None:
            raise ValueError('Cannot drop a background frame')
        native['ego']['uecp_input'] = self.model_inputs(items, training=self.base is not None)
        native['_single'] = self.native.post_processor.collate_batch([r['single'] for i in items for r in i['_local']])
        native['_audit'] = [i.get('_audit', dict(uid=self.current_uid)) for i in items]
        return native

    def collate_batch_test(self, items):
        native = self.native.collate_batch_test(items)
        native['ego']['uecp_input'] = self.model_inputs(items, training=False)
        return native

    def order(self, epoch):
        r = np.random.RandomState(stream_seed(self.config['seed'], epoch, self.split, 'order'))
        return r.permutation(len(self)).tolist()
