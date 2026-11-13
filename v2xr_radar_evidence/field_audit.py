"""Check published PCD layouts and compare raw packed RGB with the actual reader."""
from collections import Counter
from pathlib import Path

import numpy as np

from context import DATA, OUT, REPO, read, sha, write


def main():
    from context import setup
    run=setup()
    import open3d as o3d
    from opencood.utils import pcd_utils
    index=DATA/'logs/v2xr-radar-occupancy-v1/PCD_FIELDS.json'
    fields=read(index)
    roots={'train':DATA/'v2xr-official-train-v2/train',
           'validate':DATA/'logs/v2xr-official-gt-audit-v1/data_view/validate'}
    rows=[]
    pooled=[]
    for split,root in roots.items():
        for scene in sorted(root.iterdir()):
            if not scene.is_dir():
                continue
            for car in sorted(scene.iterdir()):
                files=sorted(car.glob('*_radar.pcd'))
                if not files:
                    continue
                for idx in sorted({0,len(files)//2,len(files)-1}):
                    p=files[idx]
                    with p.open() as stream:
                        header={}
                        for line in stream:
                            parts=line.split()
                            if parts and not parts[0].startswith('#'):
                                header[parts[0]]=parts[1:]
                            if parts and parts[0]=='DATA':
                                break
                        if header['FIELDS']!=['x','y','z','rgb'] or header['DATA']!=['ascii']:
                            raise ValueError(f'Unexpected layout {p}')
                        raw=np.loadtxt(stream,dtype=np.float32,ndmin=2)
                    rgb=raw[:,3].copy().view(np.uint32)
                    channels=np.column_stack([(rgb>>offset)&255 for offset in (16,8,0)])
                    decoded=np.asarray(o3d.io.read_point_cloud(str(p)).colors)
                    reader=pcd_utils.pcd_to_np(str(p))
                    if len(raw)!=len(reader) or len(raw)!=len(decoded):
                        raise ValueError(f'Reader point count differs {p}')
                    if len(raw)==0:
                        continue
                    pooled.append(channels)
                    rows.append(dict(split=split,scene=scene.name,vehicle=car.name,timestamp=p.name,
                        path=str(p),sha256=sha(p),points=len(raw),header=header,
                        nonfinite=np.sum(~np.isfinite(raw),axis=0).tolist(),
                        packed_rgb_float_range=[float(raw[:,3].min()),float(raw[:,3].max())],
                        xyz_range=[raw[:,:3].min(0).tolist(),raw[:,:3].max(0).tolist()],
                        decode_error=float(np.max(abs(decoded-channels/255))),
                        loader_red_error=float(np.max(abs(reader[:,3]-decoded[:,0]))),
                        loader_xyz_error=float(np.max(abs(reader[:,:3]-raw[:,:3])))))
    values=np.concatenate(pooled)
    reader=run.RUNTIME/'opencood/utils/pcd_utils.py'
    candidates=[]
    upstream=REPO/'v2xr_official_repro/upstream'
    for p in upstream.rglob('*.py'):
        text=p.read_text(encoding='utf-8',errors='replace')
        if 'radar' in text.lower() and ('write_point_cloud' in text or 'save_pcd' in text):
            candidates.append(str(p.relative_to(REPO)))
    result=dict(archive_index=str(index),archive_sha256=sha(index),layout=fields['split_layouts'],
        loader=str(reader),loader_sha256=sha(reader),samples=rows,
        sampled_scenes=dict(Counter(k for k,_ in {(r['split'],r['scene']) for r in rows})),
        rgb_quantiles=np.quantile(values,[0,.01,.5,.99,1],axis=0).tolist(),
        rgb_unique=[int(len(np.unique(values[:,i]))) for i in range(3)],
        rgb_same_fraction=float(np.mean(np.all(values==values[:,:1],axis=1))),
        decoder_max=max(r['decode_error'] for r in rows),loader_max=max(r['loader_red_error'] for r in rows),
        export_candidates=candidates,physical_meaning='unknown',
        supplementary='https://openaccess.thecvf.com/content/CVPR2025/supplemental/Huang_V2X-R_Cooperative_LiDAR-4D_CVPR_2025_supplemental.pdf')
    write(OUT/'D0.json',result)
    print('COMPLETE D0',len(rows),result['rgb_quantiles'],flush=True)


if __name__=='__main__':
    main()

