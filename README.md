# Figure Integrity Verification

- [Overview](#overview)
- [Dataset](#dataset)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [License](#license)

# Overview

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

### Train segmentation model
```
python train.py --device <DEVICE_ID> --base_model <BASE_MODEL_PATH> --dataset <DATASET_PATH> --figure_seg_data <FIGURE_SEG_DATASET> --neg_sample_rate <NEG_RATE> --combine_sample_rate <COM_RATE> --pretrain_mm_mlp_adapter <PROJECTOR_PATH> --vision-tower <CLIP_PATH> --vision_pretrained <SAM_PATH>
```

# License
This project is covered under the **Apache 2.0 License**.
