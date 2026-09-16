"""Launch GSPR-AttFuse with an exact AttFuse initialization checkpoint."""

import argparse
import os
import shutil
import sys

import torch


def _opencood_root():
    here = os.path.abspath(os.path.dirname(__file__))
    if os.path.isdir(os.path.join(here, "opencood")):
        return here
    candidate = os.path.abspath(os.path.join(here, "..", "OpenCOOD-main"))
    if os.path.isdir(os.path.join(candidate, "opencood")):
        return candidate
    raise RuntimeError("Cannot locate an OpenCOOD root containing opencood/")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "gspr_attfuse_config.yaml"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--gspr-checkpoint", default="",
                        help="Optional point-supervised gspr_best.pth override")
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    root = _opencood_root()
    local_root = os.path.abspath(os.path.dirname(__file__))
    sys.path[:0] = [root, local_root]
    run_dir = os.path.abspath(args.run_dir)
    dataset_root = os.path.abspath("/data/cjm/datasets")
    if os.path.commonpath((run_dir, dataset_root)) != dataset_root:
        raise ValueError("--run-dir must be inside /data/cjm/datasets")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(os.path.abspath(args.config),
                 os.path.join(run_dir, "config.yaml"))

    import attfuse_gspr.loss as local_loss
    import attfuse_gspr.point_pillar_gspr_attfuse as local_model
    sys.modules["opencood.models.point_pillar_gspr_attfuse"] = local_model
    sys.modules["opencood.loss.gspr_point_pillar_loss"] = local_loss

    from opencood.tools import train, train_utils
    original_create_model = train_utils.create_model

    def create_initialized_model(hypes):
        model = original_create_model(hypes)
        if args.baseline_checkpoint:
            checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            incompatible = model.load_state_dict(checkpoint, strict=False)
            print("Initialized from exact AttFuse checkpoint:",
                  args.baseline_checkpoint)
            print("New GSPR keys:", len(incompatible.missing_keys),
                  "unexpected keys:", len(incompatible.unexpected_keys))
        if args.gspr_checkpoint:
            gspr_checkpoint = torch.load(
                args.gspr_checkpoint, map_location="cpu")
            if isinstance(gspr_checkpoint, dict) and \
                    "state_dict" in gspr_checkpoint:
                gspr_checkpoint = gspr_checkpoint["state_dict"]
            gspr_checkpoint = {
                (key[len("gspr."):] if key.startswith("gspr.") else key): value
                for key, value in gspr_checkpoint.items()}
            gspr_incompatible = model.gspr.load_state_dict(
                gspr_checkpoint, strict=False)
            if gspr_incompatible.unexpected_keys:
                raise ValueError("Unexpected supervised GSPR keys: %s" %
                                 gspr_incompatible.unexpected_keys)
            print("Overrode GSPR from point-supervised checkpoint:",
                  args.gspr_checkpoint)
        return model

    train_utils.create_model = create_initialized_model
    train_utils.setup_train = lambda _hypes: run_dir
    sys.argv = ["train.py", "--hypes_yaml", os.path.abspath(args.config)]
    if args.half:
        sys.argv.append("--half")
    train.main()


if __name__ == "__main__":
    main()
