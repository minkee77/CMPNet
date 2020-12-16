# Few-Shot Meta-Baseline

This repository contains the code modified from [A New Meta-Baseline for Few-Shot Learning](https://github.com/yinboc/few-shot-meta-baseline).

## Main Results

*The models on *miniImageNet*  use ResNet-12 as backbone, the channels in each block are **64-128-256-512**. We introduce second-order convariance pooling([iSQRT-COV](https://github.com/jiangtaoxie/fast-MPN-COV)) into our few-shot model and test our model under semi-supervised setting.*

#### 5-way accuracy (%) on *miniImageNet*

method|1-shot|5-shot
:-:|:-:|:-:
Classifier-Baseline |58.91|77.76|
Meta-Baseline |63.17|79.26|
Classifier-Baseline |-|-|
Our |61.39|79.90|

#### semi-supervised 5-way accuracy (%) on *miniImageNet*

method|1-shot|1-shot w/D|5-shot|5-shot w/D
:-:|:-:|:-:|:-:|:-:
[LST](https://arxiv.org/abs/1906.00562) |70.1|64.1|78.7|77.4
[TransMatch](https://arxiv.org/abs/1912.09033) |63.02|59.32|81.19|79.29
Our-semi 68.22|64.02|84.54|81.22

####

Experiments on Meta-Dataset are in [meta-dataset](https://github.com/cyvius96/few-shot-meta-baseline/tree/master/meta-dataset) folder.

## Running the code

### Preliminaries

**Environment**
- Python 3.7.3
- Pytorch 1.2.0
- tensorboardX

**Datasets**
- [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))
- [tieredImageNet](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG) (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))
- [ImageNet-800](http://image-net.org/challenges/LSVRC/2012/)

Download the datasets and link the folders into `materials/` with names `mini-imagenet`, `tiered-imagenet` and `imagenet`.
Note `imagenet` refers to ILSVRC-2012 1K dataset with two directories `train` and `val` with class folders.

When running python programs, use `--gpu` to specify the GPUs for running the code (e.g. `--gpu 0,1`).
For Classifier-Baseline, we train with 4 GPUs on miniImageNet and tieredImageNet and with 8 GPUs on ImageNet-800. Meta-Baseline uses half of the GPUs correspondingly.

In following we take miniImageNet as an example. For other datasets, replace `mini` with `tiered` or `im800`.
By default it is 1-shot, modify `shot` in config file for other shots. Models are saved in `save/`.

### 1. Training Classifier-Baseline
```
python train_classifier.py --config configs/train_classifier_mini.yaml --gpu 0,1
```

### 2. Training Meta-Baseline
```
python train_meta.py --config configs/train_meta_mini.yaml --gpu 0,1
```

### 3. Test
To test the performance, modify `configs/test_few_shot.yaml` by setting `load_encoder` to the saving file of Classifier-Baseline, or setting `load` to the saving file of Meta-Baseline.

E.g., `load: ./save/meta_mini-imagenet-5shot_meta-baseline-resnet12/max-va.pth`

Then run
```
python test_few_shot.py --shot 5 --gpu 0,1
```

Test under semi-supervised setting
```
python test_few_shot_semi.py --shot 5 --gpu 0,1
```
