﻿## SemiCTrans: Semi-supervised brain tumor MRI segmentation via a dual-uncertainty guided CNN-Transformer 2.5D model with voxel-wise contrastive learning
by Yudan Zhou, Jianfeng Bao, Congbo Cai, Zhong Chen, Shuhui Cai.


<img src="https://github.com/zydlsd/SemiCTrans/assets/136596978/6c58d094-0a8d-4ad1-ad9e-a58a6a3dc877" width="700px"> 



## Introduction
This is the officail code for "SemiCTrans: emi-supervised brain tumor MRI segmentation via a dual-uncertainty guided CNN-Transformer 2.5D model with pixel-level contrastive learning"

## Installation

This repository is based on PyTorch 0.4.1.

## Usage

1. Clone the repository: https://github.com/zydlsd/SemiCTrans.git
2. Put the data in data/brats2020.
3. Train the model.
4. Test the model.

## Acknowledgement
This code is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS.git). We thank all the authors for their contribution.
## Note for data
We provided the processed h5 data in the data folder. You can refer the code in `code/data/brats2020/data_processing.py` to process your own data.
## Questions
Please contact: ydzhou@stu.xmu.edu.cn
