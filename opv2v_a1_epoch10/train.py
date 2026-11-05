"""Use the deployed joint loop with an explicit A1 or original-model factory."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "opv2v_gspr_a1_v2"
sys.path.insert(0, str(SOURCE))

import joint
import runtime
import torch


class SmokeComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("a1", "gspr"), required=True)
    parser.add_argument("--smoke", action="store_true")
    options, arguments = parser.parse_known_args()
    from attfuse_gspr import point_pillar_gspr_attfuse as original
    original_class = original.PointPillarGsprAttfuse

    def install():
        runtime.install()
        if options.model == "gspr":
            original.PointPillarGsprAttfuse = original_class

    def audit(model, optimizer):
        groups = model if options.model == "a1" else SimpleNamespace(
            gspr=model.gspr, backbone=model.backbone, a1=torch.nn.Identity())
        stats = runtime.audit_step(groups, optimizer)
        if options.model == "gspr":
            del stats["a1"]
        print("FIRST_STEP", json.dumps(stats), flush=True)
        if options.smoke:
            raise SmokeComplete()
        return stats

    joint.install = install
    joint.audit_step = audit
    sys.argv = [sys.argv[0], *arguments]
    try:
        joint.main()
    except SmokeComplete:
        if not options.smoke:
            raise
        print("SMOKE_PASS", options.model, flush=True)
    else:
        if options.smoke:
            raise RuntimeError("Smoke never performed an optimizer step")


if __name__ == "__main__":
    main()
