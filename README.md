# MASR-Pose: Explicit Geometry Enhancement via Manifold Aggregation and Structural Relation for Category-Level Object Pose Estimation

Official implementation of **MASR-Pose**, a category-level 6D object pose estimation method that strengthens sparse keypoint representations with explicit geometric modeling at three complementary stages: feature fusion, local keypoint-neighbor aggregation, and global keypoint-relation modeling.

> Paper: submitted to Image and Vision Computing (Elsevier).

<p align="center">
  <i>Architecture overview — see the paper for the full figure.</i>
</p>

## Overview

MASR-Pose builds on the [AG-Pose](https://github.com/Leeiieeo/AG-Pose) sparse keypoint pipeline and introduces three geometry-aware modules:

| Module | Stage | Role |
|---|---|---|
| **SGLR** (Semantic-Guided Local Rectification) | Feature fusion | Channel-preserving semantic-geometric fusion before keypoint extraction |
| **MAGA** (Manifold-Aware Geometric Aggregation) | Local aggregation | Normal-guided radial and tangential cues in keypoint-neighbor attention |
| **SRT** (Structural Relation Transformer) | Global reasoning | Gated pairwise geometric bias in self-attention across sparse keypoints |

The IAKD (Instance-Adaptive Keypoint Detector) and the NOCS/pose/size prediction heads are retained from the AG-Pose baseline.

## Environment Setup

Tested with **Python 3.9**, **PyTorch 1.12**, **CUDA 11.3**.

```bash
# Install dependencies
pip install gorilla-core==0.2.5.3
pip install opencv-python

# Compile PointNet++ ops
cd model/pointnet2
python setup.py install
cd ../..
```

### DINOv2 backbone

MASR-Pose uses a frozen DINOv2 ViT-S/14 as the RGB backbone. Download the [DINOv2 repository](https://github.com/facebookresearch/dinov2) and place it under the project root as `./dinov2`, or update the path in `model/Net.py` to point to your local clone.

## Data Preparation

### NOCS (CAMERA25 / REAL275)

1. Download and preprocess the NOCS dataset following [DPDN](https://github.com/JiehongLin/Self-DPDN).
2. Download segmentation results from [here](http://home.ustc.edu.cn/~llinxiao/segmentation_results.zip).

Expected structure under `./data`:

```
data
├── camera
│   ├── train
│   ├── val
│   ├── train_list_all.txt
│   ├── train_list.txt
│   └── val_list_all.txt
├── real
│   ├── train
│   ├── test
│   ├── train_list.txt
│   ├── train_list_all.txt
│   └── test_list_all.txt
├── segmentation_results
│   ├── CAMERA25
│   └── REAL275
├── camera_full_depths
├── gts
└── obj_models
```

### HouseCat6D

Download from [HouseCat6D](https://sites.google.com/view/housecat6d). Update `dataset_dir` in `config/HouseCat6D/housecat6d.yaml` to your data path.

```
HOUSECAT6D_DIR
├── scene**
├── val_scene*
├── test_scene*
└── obj_models_small_size_final
```

## Training

```bash
# NOCS (CAMERA25 → REAL275)
python train.py --config config/REAL/camera_real.yaml

# HouseCat6D
python train_housecat6d.py --config config/HouseCat6D/housecat6d.yaml
```

## Evaluation

```bash
# REAL275
python test.py --config config/REAL/camera_real.yaml --test_epoch 30

# HouseCat6D
python test_housecat6d.py --config config/HouseCat6D/housecat6d.yaml --test_epoch 150
```

## Visualization

```bash
python visualize.py --config config/REAL/camera_real.yaml --test_epoch 30
```

## Results

All results use the DINOv2 ViT-S/14 backbone.

### REAL275

| Method | 3D IoU75 | 5°2 cm | 5°5 cm | 10°2 cm | 10°5 cm |
|---|---|---|---|---|---|
| AG-Pose (repr.) | 80.3 | 55.4 | 63.4 | 74.9 | 84.7 |
| **MASR-Pose** | **79.9** | **59.9** | **66.7** | **77.5** | **86.4** |

### CAMERA25

| Method | 3D IoU75 | 5°2 cm | 5°5 cm | 10°2 cm | 10°5 cm |
|---|---|---|---|---|---|
| AG-Pose (repr.) | 91.3 | 77.8 | 82.8 | 85.5 | 91.6 |
| **MASR-Pose** | **92.5** | **79.9** | **84.0** | **87.5** | **93.0** |

### HouseCat6D

| Method | 3D IoU25 | 3D IoU50 | 5°2 cm | 5°5 cm | 10°2 cm | 10°5 cm |
|---|---|---|---|---|---|---|
| AG-Pose (repr.) | 81.8 | 62.5 | 11.5 | 12.0 | 32.7 | 35.8 |
| **MASR-Pose** | **89.4** | **78.9** | **23.3** | **24.4** | **52.9** | **56.4** |

### Ablation on REAL275 (5°2 cm)

| Variant | 5°2 cm |
|---|---|
| AG-Pose baseline | 55.4 |
| + MAGA | 56.5 |
| + MAGA + SRT | 59.6 |
| **Full MASR-Pose** (+ SGLR) | **59.9** |

### Parameter efficiency

| Method | Params | Extra | 5°2 cm |
|---|---|---|---|
| AG-Pose baseline | 223.19M | — | 55.4 |
| **MASR-Pose** | 232.45M | +9.26M | 59.9 |

## Citation

If you use this work, please cite:

```bibtex
@article{masrpose,
  title={MASR-Pose: Explicit Geometry Enhancement via Manifold Aggregation
         and Structural Relation for Category-Level Object Pose Estimation},
  author={},  % to be updated upon publication
  journal={Image and Vision Computing},
  year={2026},
  note={submitted}
}
```

## Acknowledgements

Our code builds on the [AG-Pose](https://github.com/Leeiieeo/AG-Pose) extended implementation. We also leverage code from:

- [NOCS](https://github.com/hughw19/NOCS_CVPR2019)
- [SPD](https://github.com/mentian/object-deformnet)
- [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet)
- [DPDN](https://github.com/JiehongLin/Self-DPDN)
- [VI-Net](https://github.com/JiehongLin/VI-Net)
- [HouseCat6D Toolbox](https://github.com/Junggy/HouseCat6D)
- [DINOv2](https://github.com/facebookresearch/dinov2)

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
