# Figure Integrity Verification

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The objective of this study is to address the issue of integrity verification of scientific papers through the comprehension of scientific figures. This issue is decomposed into two sub-problems: first, how to achieve fine-grained alignment between text and figure modules, and, second, how to identify content within figures that has not been adequately described. 

- [Overview](#overview)
- [Dataset](#dataset)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [License](#license)

# Overview
![Framework](https://github.com/shixiang1a/figure_understanding/blob/main/framework.png)

# Dataset
### Figure-seg
Dataset can be found in [Google Drive](https://drive.google.com/file/d/16yYWa66RbFkqfFJpI9XdX4-UPs4J9_Qz/view?usp=sharing)

# System Requirements
## Hardware requirements
A GPU capable of running 13B parameters is required to execute the training and inference processes of the 'EPM' framework. The framework has been tested on a single A100 80GB GPU.

## Software requirements
### OS Requirements
This package is supported for *Linux*. The package has been tested on the following systems:
+ Linux: Ubuntu 22.04.1

### Python Dependencies
Refer to the requirements.txt file for the software's Python dependencies.
```
pip install -r requirements.txt
```
# Installation Guide:

### Install from Github
```
git clone git@github.com:shixiang1a/figure_understanding.git
```

### Train Module Segmentation Model
```
python train.py --device <DEVICE_ID> --base_model <BASE_MODEL_PATH> --dataset figure_seg --figure_seg_data <FIGURE_SEG_DATASET> --neg_sample_rate <NEG_RATE> --combine_sample_rate <COM_RATE> --pretrain_mm_mlp_adapter <PROJECTOR_PATH> --vision-tower <CLIP_PATH> --vision_pretrained <SAM_PATH>
```

### Train Attribute VQA Model
```
python train.py --device <DEVICE_ID> --base_model <BASE_MODEL_PATH> --dataset atrr_vqa --figure_seg_data <FIGURE_SEG_DATASET> --pretrain_mm_mlp_adapter <PROJECTOR_PATH> --vision-tower <CLIP_PATH> --vision_pretrained <SAM_PATH>
```

# License
This project is covered under the **Apache 2.0 License**.
