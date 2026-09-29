from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor


PREPROCESSORS = {
    "SpVoxelPreprocessor": SpVoxelPreprocessor,
}


def build_preprocessor(preprocess_cfg, train):
    process_method_name = preprocess_cfg["core_method"]
    if process_method_name not in PREPROCESSORS:
        raise ValueError(f"Unsupported preprocessor for UECP release: {process_method_name}")

    return PREPROCESSORS[process_method_name](
        preprocess_params=preprocess_cfg,
        train=train,
    )
