"""Probe actual source-supported cells for failed and radar-neighbor targets."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run
from common import atomic_json, read_json, sha256
from device import pre_device, map_device
from diagnostics import distance_bin
from fusion import support_maps
from transport import Grid
import numpy as np
import torch
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu


def main():
    pre_device()
    mapped = map_device()
    out = run.OUT/'analysis-20261111'
    analysis=read_json(out/'ANALYSIS.json')
    targets=read_json(run.OUT/'P3/TRANSPORT.json')['targets']
    selected={row['frame']+'/'+row['object_id']:row for row in analysis['lost_targets']
              if row['frame'].split('/')[0] in ('fog','snow')}
    for frame,objects in targets.items():
        if frame.split('/')[0] not in ('fog','snow'):
            continue
        for oid,obj in objects.items():
            if obj['neighbor_radar'] and not obj['neighbor_lidar']:
                selected.setdefault(frame+'/'+oid,dict(frame=frame,object_id=oid,**obj))
    cfg=yaml_utils.load_yaml(str(run.CONFIG))
    model=run.make_model(cfg,'P3').eval()
    weight=Path(read_json(run.OUT/'P3/selected.json')['checkpoint'])
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    model.fusion.observe=True
    height,width=model.coarse_shape
    grid=Grid(height,width,run.SETTINGS['window'],torch.device('cuda:0'))
    rows,overall={},defaultdict(Counter)
    native={}
    def capture(_,args,result):
        native['rms']=result.detach().float().square().mean(1).sqrt().flatten(1)
    hook=model.fusion.native.register_forward_hook(capture)
    for weather in ('fog','snow'):
        dataset,_=run.build(run.CONFIG,condition=weather,visualize=True) if hasattr(run,'build') else (None,None)
        if dataset is None:
            from runtime import build
            dataset,_=build(run.CONFIG,condition=weather,visualize=True)
        from runtime import loader
        with torch.no_grad():
            for i,batch in enumerate(loader(dataset,evaluation=True,limit=829)):
                frame=f'{weather}/{i}'
                objects={key.rsplit('/',1)[1]:obj for key,obj in selected.items() if obj['frame']==frame}
                if not objects:
                    continue
                ego=train_utils.to_device(batch['ego'],torch.device('cuda:0'))
                model(ego)
                report=model.fusion.report
                clean=read_json(run.LR0/'final_evaluation/predictions/clean'/f'{i:06d}.json')
                reference={str(row['object_id']):row for row in clean['official_gt']}
                oids=list(objects)
                boxes=np.asarray([reference[oid]['box'] for oid in oids],dtype=np.float32)[:,[0,1,2,5,4,3,6]]
                per_target=[defaultdict(set) for _ in oids]
                for sensor in ('lidar','radar'):
                    for vehicle,points in enumerate(batch['ego'][f'projected_{sensor}'][0]):
                        if vehicle==0 or not len(points):
                            continue
                        xyz=points[:,:3].cpu().numpy() if torch.is_tensor(points) else points[:,:3]
                        inside=np.asarray(points_in_boxes_cpu(xyz.astype(np.float32,copy=False),boxes))
                        for t in range(len(oids)):
                            p=xyz[inside[t]>0]
                            if not len(p):
                                continue
                            xy=np.floor((p[:,:2]-np.array(cfg['model']['args']['lidar_range'][:2])) /
                                        np.array(cfg['model']['args']['voxel_size'][:2])).astype(int)
                            valid=(xy[:,0]>=0)&(xy[:,0]<704)&(xy[:,1]>=0)&(xy[:,1]<200)
                            coarse=(xy[valid]+4)//8
                            cell=np.minimum(coarse[:,1],height-1)*width+np.minimum(coarse[:,0],width-1)
                            per_target[t][vehicle-1].update(cell.tolist())
                for t,oid in enumerate(oids):
                    row=dict(frame=frame,object_id=oid,source=objects[oid]['source'],
                             neighbor_radar_only=bool(objects[oid]['neighbor_radar'] and not objects[oid]['neighbor_lidar']),
                             P0_lost=(frame+'/'+oid in {r['frame']+'/'+r['object_id'] for r in analysis['lost_targets']}),
                             supported_cells=0,no_local_ego_cells=0,below_two_pillar_cells=0,
                             matched_mass=0.,remaining_mass=0.,unmatched_weight=0.,
                             matched_rms_sum=0.,unmatched_rms_sum=0.,native_rms_sum=0.)
                    spatial=set()
                    if report is not None:
                        for vehicle,cells in per_target[t].items():
                            ids=torch.tensor(sorted(cells),device='cuda:0',dtype=torch.long)
                            valid=report['neighbor'][vehicle,ids].sum(-1)>0
                            ids=ids[valid]
                            spatial.update(ids.tolist())
                            row['supported_cells']+=len(ids)
                            row['no_local_ego_cells']+=int((~report['near_ego'][vehicle,ids]).sum())
                            row['below_two_pillar_cells']+=int((~report['candidate'][vehicle,ids]).sum())
                            row['matched_mass']+=float(report['col'][vehicle,ids].sum())
                            row['remaining_mass']+=float(report['col_remaining'][vehicle,ids].sum())
                            row['unmatched_weight']+=float(report['unmatched_weight'][vehicle,ids].sum())
                        if spatial:
                            ids=torch.tensor(sorted(spatial),device='cuda:0',dtype=torch.long)
                            for path in ('matched_rms','unmatched_rms'):
                                row[path+'_sum']=float(report[path][0,ids].sum())
                            row['native_rms_sum']=float(native['rms'][0,ids].sum())
                    rows[frame+'/'+oid]=row
                    for label in (('lost' if row['P0_lost'] else 'other'),
                                  'radar_neighbor_only' if row['neighbor_radar_only'] else 'other_neighbor'):
                        counts=overall[weather+'/'+label]
                        counts['targets']+=1
                        counts['targets_with_cells']+=row['supported_cells']>0
                        counts['all_cells_without_local_ego']+=row['supported_cells']>0 and row['no_local_ego_cells']==row['supported_cells']
                        counts['all_cells_below_two_pillars']+=row['supported_cells']>0 and row['below_two_pillar_cells']==row['supported_cells']
                        counts['positive_unmatched_weight']+=row['unmatched_weight']>0
                        for key in ('supported_cells','no_local_ego_cells','below_two_pillar_cells',
                                    'matched_mass','remaining_mass','unmatched_weight','matched_rms_sum','unmatched_rms_sum','native_rms_sum'):
                            counts[key]+=row[key]
                if i%100==0:
                    print('PROBE',weather,i,flush=True)
    hook.remove()
    atomic_json(out/'FAILURE_PROBE.json',dict(status='complete',device=mapped,checkpoint_sha256=sha256(weight),
        rows=rows,summary=overall,
        definition='Neighbor coarse cells that contain real target points and actual nonempty pillar support'))
    print(json.dumps(dict(summary=overall)),flush=True)


if __name__=='__main__':
    main()
