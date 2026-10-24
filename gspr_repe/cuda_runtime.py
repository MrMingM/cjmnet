"""NVIDIA migration. Fresh data/runtime provenance, never fabricated historical evidence."""
import importlib.metadata as metadata
import hashlib
import json
import os
from pathlib import Path
import sys
from auditlib import ROOT, atomic_json, bound_json, complete, identity, no_test, read_json, sha256

VERSIONS = {'torch':'2.5.1+cu124','numpy':'1.26.4','scipy':'1.15.3','shapely':'2.0.0',
            'open3d':'0.18.0','spconv':'2.3.6','cumm':'0.4.11','Cython':'0.29.37','torchvision':'0.20.1+cu124'}


def extraction_identity(value):
    """The paired extractor uses compact JSON, unlike the other audit identities."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def enabled():
    return os.environ.get('BACKEND','rocm') == 'cuda'


def input_root():
    return no_test(os.environ['CUDA_INPUT_ROOT']).resolve()


def gpu():
    if os.environ.get('ROCR_VISIBLE_DEVICES') is not None or os.environ.get('HIP_VISIBLE_DEVICES') is not None:
        raise ValueError('CUDA entry requires unset ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES')
    if os.environ.get('PYTHONHASHSEED') != '20261016':
        raise ValueError('Use run_all.sh with PYTHONHASHSEED=20261016')
    if any(os.environ.get(k) for k in ('RANK','WORLD_SIZE','LOCAL_RANK','SLURM_PROCID')):
        raise ValueError('Distributed execution is outside this single-GPU protocol')


def environment(args):
    import torch
    actual={name:metadata.version(name) for name in VERSIONS}
    if actual != VERSIONS or sys.version_info[:2] != (3,10):
        raise ValueError('Use the existing cjmnet-cu124 Python 3.10 environment: '+repr(actual))
    if torch.version.hip or torch.version.cuda!='12.4' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Require CUDA 12.4 and exactly one visible NVIDIA GPU; allocation/driver must be active')
    from observe import bootstrap, module_inventory
    runtime=Path(args.runtime_root).resolve(strict=True)
    if 'opencood' not in sys.modules:
        bootstrap(runtime)
    expected=read_json(ROOT/'v2xr_official_repro/UPSTREAM.json')['files']
    for name,digest in expected.items():
        if sha256(runtime/name)!=digest:
            raise ValueError('CUDA runtime source differs from bundled upstream: '+name)
    from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu, points_in_boxes_gpu
    from opencood.utils.box_overlaps import bbox_overlaps
    import numpy as np
    points=torch.tensor([[0.,0.,0.],[5.,5.,5.]])
    boxes=torch.tensor([[0.,0.,0.,2.,2.,2.,0.]])
    if points_in_boxes_cpu(points,boxes).tolist()!=[[1,0]]:
        raise ValueError('CPU ROI execution failed')
    if points_in_boxes_gpu(points[None].cuda(),boxes[None].cuda()).cpu().tolist()!=[[0,-1]]:
        raise ValueError('CUDA ROI execution failed')
    if bbox_overlaps(np.array([[0,0,1,1]],np.float32),np.array([[0,0,1,1]],np.float32))[0,0]!=1:
        raise ValueError('Box overlap extension execution failed')
    torch.cuda.synchronize()
    from models import install
    install(args.variant)
    from camera_metadata import install as install_camera_metadata
    camera_input,_=install_camera_metadata(args)
    modules=module_inventory(runtime)
    native={k:v for k,v in modules.items() if v['path'].endswith('.so')}
    return dict(versions=actual,cuda=torch.version.cuda,hip_version=None,camera_input=camera_input,
        upstream_commit=read_json(ROOT/'v2xr_official_repro/UPSTREAM.json').get('commit'),
        upstream_source_hashes=expected, imported_modules=modules,native_modules=native,
        gpu=torch.cuda.get_device_name(0),device_capability=list(torch.cuda.get_device_capability(0)),
        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','platform default'),
        python_executable=sys.executable,
        installed_packages={name:metadata.version(name) for name in ('pypcd','tensorboardX','Pillow','PyYAML')})


def paths(args):
    from auditlib import DEFAULTS
    for key in DEFAULTS:
        setattr(args,key,str(no_test(getattr(args,key)).resolve()))
    output=Path(args.output_root)
    inputs=input_root()
    protected=[ROOT,Path(args.train_root),Path(args.view_root),Path(args.paired_root),Path(args.runtime_root)]
    for dest in (output,inputs):
        for src in protected:
            if dest==src or dest in src.parents or src in dest.parents:
                raise ValueError('CUDA output/input-view overlaps source data/code: '+str(src))
    if output==inputs or output in inputs.parents or inputs in output.parents:
        raise ValueError('CUDA_INPUT_ROOT must be independent of OUTPUT_ROOT')
    if Path('/data') not in output.parents or Path('/data') not in inputs.parents:
        raise ValueError('Store CUDA experiment output and input views under /data')


def verify_extracted(args):
    from camera_metadata import removed_image,proof as slim_proof,root as slim_root
    root=Path(args.train_root)
    manifest=complete(root/'manifest.json')
    if identity({k:v for k,v in manifest.items() if k!='identity'})!=manifest['identity']:
        raise ValueError('Original extraction manifest identity changed')
    from auditlib import legacy
    legacy.check_scene_partition(manifest)
    acceptance=read_json(root/'data_acceptance.json')
    policy=read_json(ROOT/'v2xr_train_attfuse/TRAIN_DATA_ACCEPTANCE.json')
    if (manifest['train_data_acceptance']!=acceptance or
        acceptance['accepted_cooperative_frames']!=8083 or
        acceptance['approved_inventory_identity']!=policy['approved_inventory_identity'] or
        acceptance['approved_artifact_sha256']!=policy['approved_artifact_sha256'] or
        acceptance['acceptance_policy_sha256']!=sha256(ROOT/'v2xr_train_attfuse/TRAIN_DATA_ACCEPTANCE.json') or
        identity({k:v for k,v in acceptance.items() if k!='identity'})!=acceptance['identity'] or
        not manifest['full_train'] or not manifest['integrity_complete']):
        raise ValueError('Missing original accepted 8083-frame extracted-data contract')
    expected={row['relative']:row for row in manifest['files']}
    metadata=slim_proof()
    removed={name for name,row in expected.items()
             if removed_image(root/'train'/name,row,metadata)}
    actual={p.relative_to(root/'train').as_posix() for p in (root/'train').rglob('*') if p.is_file()}
    if actual!=set(expected)-removed:
        raise ValueError('Missing or unexpected extracted train files; finish transfer before running')
    directories={s['scene'] for s in manifest['scenes']} | {
        s['scene']+'/'+c for s in manifest['scenes'] for c in s['agents']}
    if {p.relative_to(root/'train').as_posix() for p in (root/'train').rglob('*') if p.is_dir()}!=directories:
        raise ValueError('Transferred train directory inventory differs; empty CAV placeholders must stay excluded')
    refs={}
    if metadata is not None:
        refs[str(slim_root()/'POINTCLOUD_ONLY.json')]=sha256(slim_root()/'POINTCLOUD_ONLY.json')
        refs[str(slim_root()/'EQUIVALENCE.json')]=sha256(slim_root()/'EQUIVALENCE.json')
    for i,(name,row) in enumerate(expected.items()):
        if name in removed:
            continue
        path=no_test(root/'train'/name)
        if path.stat().st_size!=row['size'] or sha256(path)!=row['sha256']:
            raise ValueError('Transferred train content mismatch: '+str(path))
        refs[str(path)]=row['sha256']
        if i%5000==0:
            print('CUDA extracted data hashes',i,'/',len(expected),flush=True)
    for name in ('manifest.json','plan.json','data_acceptance.json'):
        refs[str(root/name)]=sha256(root/name)
    for row in manifest['archives']:
        path=root/'records'/row['scene']/(row['cav']+'.json')
        record=complete(path)
        if not record['crc_test_passed'] or record['archive_sha256']!=row['sha256']:
            raise ValueError('Original extraction CRC record differs: '+str(path))
        refs[str(path)]=sha256(path)
    # Source .7z archives are historical extraction provenance, not training inputs.
    # They remain listed in the original manifest; no historical bytes/hashes are rewritten.
    return manifest,refs


def make_overlay(args, refs):
    from camera_metadata import remap_original,removed_image,proof as slim_proof
    metadata=slim_proof()
    base=Path(args.view_root)/'validate'
    pair=Path(args.paired_root)
    target=input_root()/'validate'
    view=read_json(Path(args.view_root)/'manifest.json')
    if view['status']!='complete' or identity({k:v for k,v in view.items() if k!='identity'})!=view['identity']:
        raise ValueError('Original official validation-view manifest identity differs')
    keys=view['frame_keys']
    if len(keys)!=829 or len({tuple(k) for k in keys})!=829:
        raise ValueError('Expected original 829 unique validation frame keys')
    refs[str(Path(args.view_root)/'manifest.json')]=sha256(Path(args.view_root)/'manifest.json')
    pair_manifest=read_json(pair/'manifest.json')
    refs[str(pair/'manifest.json')]=sha256(pair/'manifest.json')
    # Bind copied payload bytes to the transferred original extraction hashes.
    plan=read_json(pair/'plan.json')
    if pair_manifest['status']!='complete' or pair_manifest['identity']!=plan['identity']:
        raise ValueError('Paired validation extraction incomplete')
    refs[str(pair/'plan.json')]=sha256(pair/'plan.json')
    if extraction_identity(plan['protocol'])!=plan['identity'] or plan['protocol']['frame_keys']!=keys:
        raise ValueError('Original paired validation plan identity/frame list differs')
    payload_hashes={}
    records=sorted((pair/'records/validate').glob('*/*.json'))
    if len(records)!=20:
        raise ValueError('Expected 20 paired validation extraction records')
    for record_path in records:
        record=read_json(record_path)
        if record['identity']!=plan['identity']:
            raise ValueError('Paired validation extraction identity changed')
        refs[str(record_path)]=sha256(record_path)
        for name,digest in record['files'].items():
            path=no_test(pair/'payload/validate'/record_path.parent.name/record_path.stem/name)
            if sha256(path)!=digest:
                raise ValueError('Paired payload differs: '+str(path))
            refs[str(path)]=digest
            payload_hashes[(record_path.parent.name,record_path.stem,name)]=digest
    for row in view['files']:
        path=no_test(remap_original(row['view']))
        if removed_image(path,row,metadata):
            continue
        if path.stat().st_size!=row['size'] or sha256(path)!=row['sha256']:
            raise ValueError('Transferred official view differs: '+str(path))
        refs[str(path)]=row['sha256']
    expected_paths=set()
    for yaml in sorted(base.glob('*/*/*.yaml')):
        if yaml.stem=='data_protocol' or 'additional' in yaml.stem:
            continue
        scene,cav=yaml.parent.parent.name,yaml.parent.name
        stamp=yaml.stem
        names=[stamp+'.yaml',stamp+'.pcd',stamp+'_radar.pcd']+[stamp+'_camera'+str(i)+'.png' for i in range(4)]
        if metadata is not None:
            for i in range(4):
                image=yaml.parent/(stamp+'_camera'+str(i)+'.png')
                key=image.relative_to(Path(os.environ['POINTCLOUD_ONLY_ROOT']).resolve()).as_posix()
                if key not in metadata['images']:
                    raise ValueError('Missing camera dimensions in slim validation: '+key)
            names=[name for name in names if not name.endswith('.png')]
        links={name:yaml.parent/name for name in names}
        for weather,suffix in (('fog','_fog_0.060'),('snow','_snow_2.5_2.0')):
            name=stamp+suffix+'.pcd'
            links[name]=pair/weather/'validate'/scene/cav/(stamp+'.pcd')
        for name,source in links.items():
            source=no_test(source).resolve(strict=True)
            if not source.is_file() or not source.stat().st_size:
                raise ValueError('Missing validation auxiliary/weather file: '+str(source))
            digest=sha256(source)
            if name.endswith(('_fog_0.060.pcd','_snow_2.5_2.0.pcd')) and digest!=payload_hashes.get((scene,cav,name)):
                raise ValueError('Weather view differs from original payload: '+str(source))
            refs[str(source)]=digest
            destination=target/scene/cav/name
            expected_paths.add(destination.relative_to(target).as_posix())
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.is_symlink():
                if destination.resolve(strict=True)!=source:
                    raise ValueError('Existing CUDA view points elsewhere: '+str(destination))
            elif destination.exists():
                raise ValueError('Refuse to overwrite existing CUDA view content: '+str(destination))
            else:
                destination.symlink_to(source)
    if len(list(target.glob('*/*/*.yaml')))!=3041:
        raise ValueError('Expected 3041 complete single-CAV validation frames')
    if {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}!=expected_paths:
        raise ValueError('Unexpected CUDA validation-view files')
    return keys


def prepare(args):
    a=environment(args)
    if args.command=='preflight':
        return None,a
    root=input_root()
    proof=complete(root/'summary.json')
    for name in ('versions','cuda','native_modules','gpu','visible_devices','camera_input'):
        if proof['environment'][name]!=a[name]:
            raise ValueError('CUDA preflight environment changed: '+name)
    manifest=read_json(Path(args.train_root)/'manifest.json')
    if sha256(Path(args.train_root)/'manifest.json')!=proof['train_manifest_sha256']:
        raise ValueError('Training manifest changed after preflight')
    if proof['sources']!=__import__('auditlib').sources():
        raise ValueError('CUDA preflight source identity changed; use a new CUDA_INPUT_ROOT and OUTPUT_ROOT')
    frames=[__import__('json').loads(s) for s in (root/'per_frame.jsonl').read_text().splitlines()]
    evidence=dict(a=a,b=dict(selected_frame_keys=proof['frame_keys']),
        input_manifest=dict(identity=proof['identity'],overlay=str(root/'validate')),frames=frames)
    return manifest,evidence


def frozen_gt(args):
    row=complete(input_root()/'summary.json')
    return dict(gt=dict(identity=row['gt_identity']))


def _preflight(args):
    import copy
    import random
    import numpy as np
    import torch
    from opencood.data_utils.datasets import build_dataset
    from checkpoint import seed
    from training import config
    from support import compare_gt,atomic_jsonl
    from auditlib import load_frozen
    output=Path(args.output_root);output.mkdir(parents=True,exist_ok=True)
    manifest,refs=verify_extracted(args)
    keys=make_overlay(args,refs)
    root=input_root()
    root.mkdir(parents=True,exist_ok=True)
    h=config(args,output)
    train_keys={(s['scene'],t) for s in manifest['scenes'] for t in s['timestamps']}
    if train_keys & {tuple(k) for k in keys}:
        raise ValueError('Train/validation frame overlap')
    observer=load_frozen('cuda_gt_observer','observe_gt.py',ROOT/'v2xr_official_repro/gt_audit')
    frames=[dict(index=i,frame_key=k,conditions={}) for i,k in enumerate(keys)]
    for weather,suffix in (('clean',''),('fog','_fog_0.060'),('snow','_snow_2.5_2.0')):
        cfg=copy.deepcopy(h); cfg.update(validate_dir=str(root/'validate'),train_sim=False,eval_sim=weather!='clean',sim_weather=suffix)
        dataset=build_dataset(cfg,visualize=True,train=False)
        from cuda_evaluation import dataset_keys
        if dataset_keys(dataset)!=keys:
            raise ValueError('Official validation scene/frame order differs from original view manifest')
        seed(20261016)
        binding=dict(input_identity=identity(refs),weather=weather,sources=__import__('auditlib').sources())
        bound_json(root/'gt_records'/weather/'contract.json',binding)
        for i,key in enumerate(keys):
            path=root/'gt_records'/weather/(f'{i:06d}.json')
            if path.exists():
                row=read_json(path)
                # Advance original dataset/RNG even when reusing a cached observer record.
                observed=observer.observe(dataset,i,key)
                if identity(observed)!=identity(row):
                    raise ValueError('Resumed official GT observation differs: '+str(path))
            else:
                row=observer.observe(dataset,i,key);atomic_json(path,row)
            if weather!='clean':
                clean=frames[i]['conditions']['clean']['final_gt']
                selected={x['object_id'] for x in row['final_gt']}
                if not selected <= {x['object_id'] for x in clean}:
                    raise ValueError('Weather GT is not a subset of Clean')
                compare_gt(row['final_gt'],[x for x in clean if x['object_id'] in selected],'CUDA live weather pairing')
            frames[i]['conditions'][weather]=row
            if i%50==0:
                print('CUDA fresh GT audit',weather,i,'/',len(keys),flush=True)
    totals={w:sum(f['conditions'][w]['final_gt_count'] for f in frames) for w in ('clean','fog','snow')}
    if totals!=dict(clean=6224,fog=3673,snow=5854):
        raise ValueError('Transferred official GT totals differ from accepted dataset: '+repr(totals))
    atomic_jsonl(root/'per_frame.jsonl',frames)
    a=args.prepared[1]
    for name in ('environment_check.json','native_check.json','native_build_selection.json'):
        path=(Path(args.runtime_root)/name) if name=='native_build_selection.json' else Path(args.runtime_root).parent/name
        if path.is_file():
            refs[str(path)]=sha256(path)
    proof=dict(status='complete',backend='cuda',frame_keys=keys,totals=totals,
        gt_identity=identity([{w:f['conditions'][w]['final_gt'] for w in totals} for f in frames]),
        environment=a,train_manifest_sha256=sha256(Path(args.train_root)/'manifest.json'),
        extracted_files_identity=identity(refs),original_archives_required=False,
        archive_scope='Original extraction/CRC records retained; source archive files not copied/revalidated on NVIDIA',
        historical_d1abc_reused=False,sources=__import__('auditlib').sources())
    proof['identity']=identity(proof)
    proof['output_hashes']={'per_frame.jsonl':sha256(root/'per_frame.jsonl')}
    bound_json(root/'summary.json',proof)
    refs[str(root/'summary.json')]=sha256(root/'summary.json')
    refs[str(root/'per_frame.jsonl')]=sha256(root/'per_frame.jsonl')
    atomic_json(output/'external_inputs.json',refs)
    atomic_json(output/'input_trees.json',{str(root/'validate'):dict(
        files=sorted(p.relative_to(root/'validate').as_posix() for p in (root/'validate').rglob('*') if p.is_file()),
        directories=sorted(p.relative_to(root/'validate').as_posix() for p in (root/'validate').rglob('*') if p.is_dir()))})
    atomic_json(output/'summary.json',dict(status='complete',backend='cuda',input_evidence_sha256=sha256(root/'summary.json'),
        sources=__import__('auditlib').sources(),environment=a,totals=totals))


def preflight(args):
    from auditlib import lock
    root=input_root()
    root.mkdir(parents=True,exist_ok=True)
    with lock(root/'.preflight.lock'):
        return _preflight(args)
