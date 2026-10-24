"""Fresh GPU-visible process for D6 only; no inherited CPU-stage Torch state."""
import argparse
from pathlib import Path
from scorw_dataset.core import server_only,read_json,source_identity


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--gpu",required=True,type=int);p.add_argument("--parent",required=True);a=p.parse_args()
    server_only();root=Path(a.root);protocol=read_json(root/"DATASET_PROTOCOL.json")
    if source_identity()["identity"]!=protocol["sources"]["identity"]:raise ValueError("D6 source changed")
    from scorw_repair.detection import model_identity
    from scorw_dataset.evaluation import run
    if model_identity(protocol["config"])!=protocol["model"]:raise ValueError("D6 model identity changed")
    run(protocol["config"],root,Path(a.out),a.gpu,protocol["model"],a.parent)
