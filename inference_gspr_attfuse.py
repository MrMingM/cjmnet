"""Register the local GSPR model, then delegate to OpenCOOD inference."""

import os
import sys


def main():
    here = os.path.abspath(os.path.dirname(__file__))
    root = here if os.path.isdir(os.path.join(here, "opencood")) else \
        os.path.abspath(os.path.join(here, "..", "OpenCOOD-main"))
    sys.path[:0] = [root, here]
    import attfuse_gspr.point_pillar_gspr_attfuse as local_model
    sys.modules["opencood.models.point_pillar_gspr_attfuse"] = local_model
    from opencood.tools import inference
    inference.main()


if __name__ == "__main__":
    main()
