"""Derive data-path views; preserve model/training parameters and original files."""
import argparse
from pathlib import Path
from copy import deepcopy
from scorw_dataset.core import atomic_text,atomic_json,ensure_link,digest


def derive(template,root,train_condition="fog",eval_condition="fog",eval_split="validate"):
    import yaml
    if train_condition not in {"clean","fog"} or eval_condition not in {"clean","fog"} or eval_split not in {"validate","test"}:raise ValueError("Invalid dataset view")
    with Path(template).open(encoding="utf-8") as f:h=yaml.load(f,Loader=yaml.Loader)
    h=deepcopy(h);h["root_dir"]=str(Path(root)/train_condition/"train")
    h["validate_dir"]=str(Path(root)/eval_condition/eval_split);h["test_dir"]=str(Path(root)/eval_condition/"test")
    # Already materialized weather; avoid applying an additional random weather pass.
    h.pop("weather_augmentation",None)
    return h


def write_views(c,root):
    import yaml
    root=Path(root);folder=root/"configs";folder.mkdir(exist_ok=True)
    paths={v:{s:str(root/v/s) for s in ("train","validate","test")} for v in ("clean","fog")}
    atomic_text(folder/"dataset_paths.yaml",yaml.safe_dump(paths,sort_keys=False))
    for condition in ("clean","fog"):
        h=derive(c["model"]["config"],root,condition,condition)
        atomic_text(folder/("attfuse_"+condition+".yaml"),yaml.dump(h,sort_keys=False))
        h["data_augment"]=[]
        for key in ("prediction_distillation","DG_params"):h.pop(key,None)
        view=folder/("attfuse_eval_"+condition)
        atomic_text(view/"config.yaml",yaml.dump(h,sort_keys=False))
        ensure_link(c["model"]["checkpoint"],view/Path(c["model"]["checkpoint"]).name)
    h=derive(c["model"]["config"],root,"fog","fog","test")
    atomic_text(folder/"attfuse_fog_test.yaml",yaml.dump(h,sort_keys=False))
    atomic_json(folder/"CONFIG_IDENTITIES.json",dict(template_sha256=digest(c["model"]["config"]),
        generated={p.relative_to(root).as_posix():digest(p) for p in sorted(folder.rglob("*.yaml"))},
        inference_weights_are_symlinks_to_original_frozen_checkpoint=True,
        GSPR="merge dataset_paths.yaml or use scorw_dataset.configure with the existing GSPR template; model settings retained"))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--template",required=True);p.add_argument("--dataset-root",required=True)
    p.add_argument("--train-condition",choices=("clean","fog"),default="fog");p.add_argument("--eval-condition",choices=("clean","fog"),default="fog")
    p.add_argument("--eval-split",choices=("validate","test"),default="validate");p.add_argument("--output",required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError("Refusing to overwrite an existing model config")
    import yaml
    atomic_text(a.output,yaml.dump(derive(a.template,a.dataset_root,a.train_condition,a.eval_condition,a.eval_split),sort_keys=False))
