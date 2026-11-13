"""Extract frozen outputs, independent car candidates and bounded feature summaries."""
from collections import Counter
import json

import numpy as np
import torch
import torch.nn.functional as F
from .input_audit import BASE, CLEAN, SINGLE, OUT, WEATHERS, atomic_json, identity, lr, mapped, read_json, sha256
from . import cache
from .metrics import fixed_metrics, hits
from diagnostics import object_support
from geometry_warnings import checked_intersections
from metrics import valid_nms_candidates
from .native_trace import trace
from runtime import build, create_model, loader
from checkpoint import seed
from support import patched
from opencood.tools import train_utils
from opencood.utils import box_utils
from v2xr_fusion_interaction.common.support import support_maps

SHAPES = ((128,100,352),(256,50,176),(512,25,88))


def one_car(ego, car):
    result = dict(ego, record_len=torch.ones_like(ego['record_len']))
    for sensor in ('processed_lidar','processed_radar'):
        data = ego[sensor]
        keep = data['voxel_coords'][:,0]==car
        result[sensor] = {key:value[keep] for key,value in data.items()}
        result[sensor]['voxel_coords'][:,0] = 0
    return result


class NativeVehicleMetadata:
    """Observe the actual per-car preprocessing calls within each loader worker."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.native = self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        native = self.dataset.native
        original = native.get_item_single_car
        cars = []

        def capture(car, *args, **kwargs):
            params = car['params']
            cars.append(dict(id=str(car['id']), is_ego=bool(car['ego']),
                             pose=np.asarray(params['lidar_pose'], dtype=float).tolist(),
                             to_ego=np.asarray(params['transformation_matrix'], dtype=float).tolist()))
            return original(car, *args, **kwargs)

        with patched(native, 'get_item_single_car', capture):
            result = self.dataset[index]
        result['ego']['audit_native_cars'] = cars
        return result

    def collate_batch_test(self, batch):
        output = self.dataset.native.collate_batch_test(batch)
        output['ego']['audit_native_cars'] = batch[0]['ego']['audit_native_cars']
        return output


def vehicle_records(ds, index, ego, manifest):
    scene, stamp = ds.keys[index]
    row = manifest[scene]
    poses = np.asarray(ego['lidar_pose'][0])
    cars = ego['audit_native_cars']
    ids = [car['id'] for car in cars]
    if (not ids or ids[0] != row['ego'] or [car['is_ego'] for car in cars] != [True]+[False]*(len(cars)-1)
            or ids != [car for car in row['agents'] if car in ids]
            or len(cars) != int(ego['record_len'].sum()) or len(cars) != len(poses)
            or not np.allclose([car['pose'] for car in cars], poses, atol=1e-5, rtol=0)):
        raise ValueError(f'Native CAV order or poses differ: {scene}/{stamp}')
    if not np.allclose(ego['transformation_matrix'], np.eye(4)):
        raise ValueError('Native postprocess transform is not identity on projected input')
    result=[]
    for i,car in enumerate(cars):
        transform = np.asarray(car['to_ego'])
        if transform.shape != (4,4) or not np.isfinite(transform).all():
            raise ValueError(f'Invalid native CAV transform: {scene}/{stamp}/{car["id"]}')
        empty={}
        for modality in ('lidar','radar'):
            data=ego['processed_'+modality]
            empty[modality]=not bool(((data['voxel_coords'][:,0]==i)&(data['voxel_num_points']>0)).any())
        result.append(dict(id=car['id'], pose=car['pose'], to_ego=car['to_ego'], empty=empty))
    return result


def summaries(model, ego, arrays, prefix):
    handles=[]
    for i,layer in enumerate(model.backbone.fuse_modules):
        def capture(_, args, i=i):
            x,length=args
            cars=int(length.sum())
            c,h,w=SHAPES[i]
            if x.shape != (cars,c,h,w) or len(length)!=1:
                raise ValueError(f'Scale or record_len differs: {i}/{x.shape}/{length}')
            signs=torch.as_tensor(np.random.default_rng(418+i).choice([-1.,1.],(8,c))/np.sqrt(c),
                                  dtype=x.dtype,device=x.device)
            compact=torch.einsum('kc,nchw->nkhw', signs, x)
            counts=support_maps(ego,cars,(h,w),2**(i+1))
            arrays[f'{prefix}_feature{i}']=compact[:,:,::4,::4].cpu().numpy().astype(np.float16)
            arrays[f'{prefix}_support{i}']=counts[:,::4,::4].cpu().numpy()
            if i<2:
                norm=x.square().mean(1,keepdim=True)
                _,ids=F.max_pool2d(norm,8,8,return_indices=True,ceil_mode=True)
                ix=ids.reshape(cars,-1)
                feat=x.flatten(2).gather(2,ix[:,None].expand(-1,c,-1)).transpose(1,2)
                arrays[f'{prefix}_peaks{i}']=ix.cpu().numpy().astype(np.int32)
                arrays[f'{prefix}_descriptors{i}']=F.normalize(feat,dim=-1).cpu().numpy().astype(np.float16)
            arrays[f'{prefix}_shape{i}']=np.array((cars,c,h,w))
        handles.append(layer.register_forward_pre_hook(capture))
    return handles


def candidates(arrays, prefix, result, processor):
    corners, boxes, _, scores, flags=result
    tensor=torch.as_tensor(corners)
    finite=np.isfinite(boxes).all(1)&np.isfinite(scores)&(boxes[:,3:6]>0).all(1)
    valid=(box_utils.remove_large_pred_bbx(tensor,processor.dataset)&box_utils.remove_bbx_abnormal_z(tensor)
           &box_utils.get_mask_for_boxes_within_range_torch(tensor,processor.params['anchor_args']['cav_lidar_range'])).numpy()
    indices=np.flatnonzero(finite & valid)
    ranking=indices[np.argsort(-scores[indices],kind='stable')[:200]]
    keep=np.union1d(np.flatnonzero(flags>=2),ranking)
    for key,value in dict(box=boxes, corners=corners, score=scores, stage=flags).items():
        arrays[f'{prefix}_{key}']=value[keep]
    arrays[f'{prefix}_index']=keep
    arrays[f'{prefix}_top']=np.searchsorted(keep,ranking)
    return dict(anchors=len(boxes), valid=int((finite&valid).sum()), final=int((flags==4).sum()), pre_nms=int((flags>=2).sum()))


def extract(smoke=False, probe=None):
    from inputs import VAL_MANIFEST
    from opencood.hypes_yaml import yaml_utils
    device=mapped('V2')
    seed(20261016)
    config=yaml_utils.load_yaml(str(lr.CONFIG))
    model=create_model(config).eval()
    model.load_state_dict(torch.load(lr.WEIGHT,map_location='cpu',weights_only=True),strict=True)
    generator=torch.Generator()
    generator.set_state(torch.get_rng_state())
    manifest={row['scene']:row for row in read_json(VAL_MANIFEST)['scenes']}
    datasets=[build(lr.CONFIG,condition=w,visualize=True)[0] for w in WEATHERS]
    if any(ds.keys!=datasets[0].keys or len(ds)!=829 for ds in datasets):
        raise ValueError('Three weather frame identities differ')
    previous={name:fixed_metrics(p) for name,p in [('multi',BASE),('single',SINGLE)]}
    contract=identity(read_json(OUT/'CONTRACT.json'))
    file_hashes={}
    limit=3 if smoke else 829
    counts=Counter()
    loaders=[loader(NativeVehicleMetadata(ds),evaluation=True,limit=829) for ds in datasets]
    for batches in loaders:
        batches.generator=generator
    for i,batches in enumerate(zip(*loaders,strict=True)):
        if i>=limit:
            break
        if probe is not None and i != probe:
            continue
        file=cache.path(i,smoke)
        frame=datasets[0].keys[i]
        if file.exists():
            meta,_=cache.load(file,contract)
            if meta['index']!=i or meta['frame']!=frame:
                raise ValueError('Resumed frame identity differs')
        else:
            ref=read_json(CLEAN/f'{i:06d}.json')
            if ref['frame_key']!=frame:
                raise ValueError('Fixed ground truth frame key differs')
            targets=ref['official_gt']
            meta=dict(contract=contract,index=i,frame=frame,targets=targets,weather={})
            payload={}
            for weather,ds,batch in zip(WEATHERS,datasets,batches):
                ego=batch['ego']
                processor=ds.native.post_processor
                cars=vehicle_records(ds,i,ego,manifest)
                gpu=train_utils.to_device(ego,torch.device('cuda:0'))
                details=dict(cars=cars, candidates={}, hits={})
                with torch.inference_mode(), checked_intersections(), valid_nms_candidates(Counter()):
                    hooks=summaries(model,gpu,payload,weather)
                    try:
                        results,detected=trace(model,gpu,processor,targets)
                    finally:
                        for hook in hooks: hook.remove()
                    details['candidates']['multi']=candidates(payload,f'{weather}_multi',results,processor)
                    details['hits']['multi']=sorted(detected)
                    for j,car in enumerate(cars):
                        try:
                            results,detected=trace(model,one_car(gpu,j),processor,targets)
                        except Exception as exc:
                            raise RuntimeError(f'Candidate trace failed: {weather}/{i}, vehicle {car["id"]}') from exc
                        details['candidates'][str(j)]=candidates(payload,f'{weather}_{j}',results,processor)
                        details['hits'][str(j)]=sorted(detected)
                for mode,k in [('multi','multi'),('single','0')]:
                    archived=hits(previous[mode],weather,i)
                    current=set(details['hits'][k])
                    details[mode+'_replay_disagreement']=len(archived^current)
                    details[mode+'_reference_hits']=sorted(archived)
                    details[mode+'_replay_only']=sorted(current-archived)
                    details[mode+'_archive_only']=sorted(archived-current)
                low=config['model']['args']['lidar_range']
                details['support']=object_support(ego,targets,None,(low[0],low[1],1.6,1.6,50,176),
                                                  dict(distance_edges_m=[30,60]))
                details['origin']=[low[0]+.2,low[1]+.2]
                meta['weather'][weather]=details
            if any([x['id'] for x in d['cars']]!=[x['id'] for x in meta['weather']['clean']['cars']]
                   for d in meta['weather'].values()):
                raise ValueError('Weather vehicle order differs')
            previous_smoke=OUT/'smoke_cache_native'/file.name
            if smoke and previous_smoke.exists():
                with np.load(previous_smoke, allow_pickle=False) as old:
                    old_weather=json.loads(old['metadata'].item())['weather']
                for weather,d in meta['weather'].items():
                    prior=old_weather[weather]['cars']
                    if len(prior)!=len(d['cars']) or any(
                        a['id']!=b['id'] or a['empty']!=b['empty']
                        or any(not np.allclose(a[k],b[k],rtol=0,atol=1e-5) for k in ('pose','to_ego'))
                        for a,b in zip(prior,d['cars'])
                    ):
                        raise ValueError(f'Native vehicle metadata differs from cached smoke: {weather}/{i}')
            cache.save(file,meta,payload)
        counts['fixed_targets']+=len(meta['targets'])
        for w,row in meta['weather'].items():
            for mode in ('multi','single'):
                counts[f'{w}/{mode}/disagreement']+=row[mode+'_replay_disagreement']
                counts[f'{w}/{mode}/archive_hits']+=len(row[mode+'_reference_hits'])
        file_hashes[file.name]=sha256(file)
        if probe is not None:
            print('TARGETED REPLAY PASSED', i, frame, flush=True)
            return
        if i==0 or (i+1)%25==0:
            print('SHARED CACHE',i+1,'/',limit,flush=True)
    if not smoke and (len(file_hashes)!=829 or counts['fixed_targets']!=6235):
        raise ValueError(f'Incomplete shared cache: {counts}')
    atomic_json(OUT/('SMOKE.json' if smoke else 'SHARED_CACHE_MANIFEST.json'),
                dict(status='complete',device=device,contract=contract,frames=len(file_hashes),
                     targets=counts['fixed_targets'],replay=counts,files=file_hashes))
