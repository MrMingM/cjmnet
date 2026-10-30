"""D6 geometry, fixed clean labels and deterministic per-frame joint augmentation."""
import numpy as np
from pathlib import Path
from scorw_diagnostics.core import read, identity, digest
from scorw_diagnostics.data import fixed_gt
from scorw_diagnostics.voxel_trace import project


def seed(c,*keys):
    return int(identity([c['seed'],*keys])[:8],16)


def schedule(c,epoch,n):
    rng=np.random.default_rng(seed(c,'order',epoch));order=rng.permutation(n).tolist()
    weather=np.zeros(n,dtype=np.int8)
    weather[np.random.default_rng(seed(c,'weather',epoch)).permutation(n)[:n//2]]=1
    return order,weather.tolist()


class Data:
    def __init__(self,c,h,split,train=False):
        from opencood.data_utils.pre_processor import build_preprocessor
        from opencood.data_utils.post_processor import build_postprocessor
        from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
        self.c=c;self.h=h;self.split=split;self.train=train;self.root=Path(c['dataset'])
        self.manifest=read(self.root/'manifests'/(split.upper()+'_SELECTION.json'))
        self.rows={(r['scene'],r['timestamp'],r['cav']):r for r in self.manifest['rows']}
        self.keys=[(s,t) for s in self.manifest['layout'] for t in s['timestamps']]
        self.pre=build_preprocessor(h['preprocess'],train);self.post=build_postprocessor(h['postprocess'],False)
        self.aug=DataAugmentor(h['data_augment'],train);self.anchors=self.post.generate_anchor_box()

    def frame(self,index,weather,epoch=0):
        from opencood.utils import pcd_utils,box_utils
        from scorw_dataset.io import read_points
        s,t=self.keys[index];name=s['scene'];gt,ids,active,pose=fixed_gt(self.c,s,t,self.rows)
        clouds=[];inputs=[]
        for cav in active:
            row=self.rows[(name,t,cav)]
            path=self.root/weather/row['relative']
            receipt=read(self.root/'records'/self.split/name/'clouds'/cav/t/'RESULT.json')
            if digest(path)!=(row['pcd_sha256'] if weather=='clean' else receipt['output_sha256']):raise ValueError('PCD changed during run')
            if digest(row['yaml'])!=row['yaml_sha256']:raise ValueError('Metadata changed during run')
            pc=read_points(path)
            if self.train:
                pc,_,_=project(pc,row['pose'],pose,pcd_utils,[-1e9]*3+[1e9]*3)
            else:pc,_,_=project(pc,row['pose'],pose,pcd_utils,self.h['preprocess']['cav_lidar_range'])
            clouds.append(pc)
        if self.train:
            centers=box_utils.corner_to_center(gt,order=self.post.params['order'])
            tagged=np.concatenate([np.column_stack((p,np.full(len(p),k,dtype=np.float32))) for k,p in enumerate(clouds)])
            np.random.seed(seed(self.c,'augmentation',epoch,index))
            value=self.aug.forward(dict(lidar_np=tagged,object_bbx_center=centers,object_bbx_mask=np.ones(len(centers))))
            centers=value['object_bbx_center'];tagged=value['lidar_np']
            clouds=[pcd_utils.mask_points_by_range(tagged[tagged[:,4]==k,:4],self.h['preprocess']['cav_lidar_range']) for k in range(len(active))]
            centers=box_utils.mask_boxes_outside_range_numpy(centers,self.h['preprocess']['cav_lidar_range'],self.post.params['order'])
            gt=box_utils.boxes_to_corners_3d(centers,self.post.params['order'])
            mask=np.ones(len(centers))
            labels=self.post.generate_label(gt_box_center=centers,anchors=self.anchors,mask=mask)
        else:labels=None
        features=[]
        for k,p in enumerate(clouds):
            p=np.ascontiguousarray(p,dtype=np.float32)
            if self.train:
                p=p[np.random.default_rng(seed(self.c,'point_order',epoch,index,k)).permutation(len(p))]
            from scorw_r3.core import array_hash
            inputs.append(array_hash(p))
            features.append({k:v.copy() for k,v in self.pre.preprocess(p).items()})
        return dict(features=features,labels=labels,gt=np.asarray(gt,dtype=np.float32),ids=ids,active=active,
                    scene=name,timestamp=t,input_hash=identity(inputs),weather=weather,index=index)

    def batch(self,frames):
        import torch
        from opencood.tools.train_utils import to_device
        features=[v for f in frames for v in f['features']]
        data=dict(processed_lidar=self.pre.collate_batch(features),record_len=torch.tensor([len(f['features']) for f in frames]),
                  anchor_box=torch.from_numpy(self.anchors),transformation_matrix=torch.eye(4))
        labels=self.post.collate_batch([f['labels'] for f in frames]) if frames[0]['labels'] is not None else None
        return to_device(data,torch.device('cuda:0')),to_device(labels,torch.device('cuda:0')) if labels else None
