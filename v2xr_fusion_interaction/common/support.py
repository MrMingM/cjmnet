"""Counts of nonempty input pillars projected to official convolution centers."""
import torch


def support_maps(ego, cars, shape, stride):
    height,width=shape
    maps=[]
    for sensor in ('lidar','radar'):
        data=ego[f'processed_{sensor}']
        ids,_,y,x=data['voxel_coords'].long().unbind(1)
        y=((y+stride//2)//stride).clamp_max(height-1)
        x=((x+stride//2)//stride).clamp_max(width-1)
        index=(ids*height+y)*width+x
        counts=torch.zeros(cars*height*width,dtype=torch.int32,device=ids.device)
        counts.scatter_add_(0,index,(data['voxel_num_points']>0).int())
        maps.append(counts.reshape(cars,height,width))
    return torch.stack(maps,dim=-1)
