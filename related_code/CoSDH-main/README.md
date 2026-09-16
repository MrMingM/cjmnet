# CoSDH (CVPR 2025)

CoSDH: Communication-Efficient Collaborative Perception via Supply-Demand Awareness and Intermediate-Late Hybridization

[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_CoSDH_Communication-Efficient_Collaborative_Perception_via_Supply-Demand_Awareness_and_Intermediate-Late_Hybridization_CVPR_2025_paper.pdf)

## Installation

This project is based on the [Coalign](https://github.com/yifanlu0227/CoAlign) project. Please refer to the original project’s installation guide.

Please visit the feishu docs [CoAlign Installation Guide](https://udtkdfu8mk.feishu.cn/docx/LlMpdu3pNoCS94xxhjMcOWIynie) for details!

Or you can refer to [OpenCOOD data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html)
and [OpenCOOD installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare
data and install CoAlign. The installation is totally the same as OpenCOOD, except some dependent packages required by CoAlign.

## Dataset

First, use `mkdir` to create a `dataset` folder in this project directory, then download or link the dataset using `ln`, organizing the dataset according to the format below:

```text
dataset
├── my_dair_v2x 
│   └── v2x_c
├── OPV2V
│   ├── test
│   ├── train
│   └── validate
├── V2X-Sim-2.0
│   ├── v2.0
│   ├── maps
│   └── sweeps
└── v2xsim2_info
    ├── v2xsim_infos_test.pkl
    ├── v2xsim_infos_train.pkl
    └── v2xsim_infos_val.pkl
```

Note that

1. `*.pkl` file in `v2xsim2_info` can be found in [Google Drive](https://drive.google.com/drive/folders/16_KkyjV9gVFxvj2YDCzQm1s9bVTwI0Fw?usp=sharing)
2. use  complemented annotation for DAIR-V2X in `my_dair_v2x` [Google Drive](https://drive.google.com/file/d/13g3APNeHBVjPcF-nTuUoNOSGyTzdfnUK/view?usp=sharing), see [Complemented Annotations for DAIR-V2X-C](https://github.com/yifanlu0227/CoAlign?tab=readme-ov-file#complemented-annotations-for-dair-v2x-c-) for more details.

## Checkpoints

Download: [Google Drive](https://drive.google.com/drive/folders/1T3LLCn257Gynoqmm_HeXJu3Q8PBYGHfL?usp=sharing)

## Training and Inference

**Train**

```bash
torchrun --master_port 22334 --nproc_per_node=4 \
opencood/tools/train_ddp.py \
-y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_cosdh.yaml
```

**Inference**

```bash
# intermediate–late hybrid fusion 
python opencood/tools/inference.py --model_dir opencood/logs/<ckpt-dir> --fusion_method intermediatelate
# intermediate fusion only
python opencood/tools/inference.py --model_dir opencood/logs/<ckpt-dir> --fusion_method intermediate
```

## Citation

Please cite our work if you find it useful.
```bibtex
@InProceedings{Xu_2025_CVPR,
    author    = {Xu, Junhao and Zhang, Yanan and Cai, Zhi and Huang, Di},
    title     = {CoSDH: Communication-Efficient Collaborative Perception via Supply-Demand Awareness and Intermediate-Late Hybridization},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6834-6843}
}
```

## Acknowlege

This project based on the code of [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [CoAlign](https://github.com/yifanlu0227/CoAlign), thanks to their great code framework.

