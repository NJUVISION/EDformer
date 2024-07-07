# EDformer
This is the source code for the ECCV 2024 - EDformer: Transformer-Based Event Denoising Across Varied Noise Levels.

## Environment
Python 3.10 \
Pytorch 1.11.0 \
CUDA 11.3 \
cudnn 8 

## Installation
```
pip install -r requirements.txt
```
Require additional install [sparseconvnet](https://github.com/facebookresearch/SparseConvNet) and [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

## Datasets preparation
All datasets mentioned in the paper can be downloaded [here](https://pan.baidu.com/s/1F-JTP6j-CXwgE2DOqtO56A?pwd=3905).

## Train 
```
python train.py [ED24 datasets root path] 4096
```

## MESR Test
```
python eval_mesr.py
```

## AUC Test
```
python eval_auc.py
```
## zebra_fish Test
```
python eval_zebrafish.py
```