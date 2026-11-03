"""Run full inherited data/native/environment checks and audit existing E0."""
from pathlib import Path
from auditlib import atomic_json,sources,sha256

def run(args):
    from baseline import audit
    ref=audit(args,*args.prepared)
    manifest,evidence=args.prepared
    refs=dict(ref['artifacts'])
    for row in manifest['files']:
        refs[str(Path(args.train_root)/'train'/row['relative'])]=row['sha256']
    for name in ('manifest.json','data_acceptance.json','plan.json'):
        p=Path(args.train_root)/name;refs[str(p)]=sha256(p)
    for row in evidence['input_manifest']['files']:
        refs[row['source']]=row['sha256']
        refs[str(Path(evidence['input_manifest']['overlay'])/row['relative'])]=row['sha256']
    root=Path(args.output_root)
    atomic_json(root/'external_inputs.json',refs)
    trees={}
    for folder in (Path(args.train_root)/'train',Path(evidence['input_manifest']['overlay']),Path(args.view_root)/'validate'):
        trees[str(folder)]=dict(files=sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()),
            directories=sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_dir()))
    atomic_json(root/'input_trees.json',trees)
    atomic_json(root/'summary.json',dict(status='complete',sources=sources(),
        data_identity=manifest['identity'],input_identity=evidence['input_manifest']['identity'],
        E0_retrained=False,E0_evaluation_reused=True))
