from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor


POSTPROCESSORS = {
    "VoxelPostprocessor": VoxelPostprocessor,
}


def build_postprocessor(anchor_cfg, train):
    process_method_name = anchor_cfg["core_method"]
    if process_method_name not in POSTPROCESSORS:
        raise ValueError(f"Unsupported postprocessor for UECP release: {process_method_name}")

    return POSTPROCESSORS[process_method_name](
        anchor_params=anchor_cfg,
        train=train,
    )
