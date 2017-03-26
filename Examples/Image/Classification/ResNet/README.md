# CNTK Examples: Image/Classification/ResNet

## Overview

|Data:     |The CIFAR-10 dataset (http://www.cs.toronto.edu/~kriz/cifar.html) and the ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) for image classification.
|:---------|:---
|Purpose   |This folder contains a number of examples that demonstrate how to use CNTK to define residual network (http://arxiv.org/abs/1512.03385) for image classification.
|Network   |Deep convolutional residual networks (ResNet).
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Running the example

### Getting the data
We use the CIFAR-10 and ILSVRC2012 datasets to demonstrate how to train a deep convolutional residual network (ResNet). ResNet was invented by Researchers at [Microsoft Research](https://www.microsoft.com/en-us/research/), and it won first place in both [ILSVRC](http://www.image-net.org/challenges/LSVRC/) and [MS COCO](http://mscoco.org/) competitions in the year of 2015. The original implementation of ResNet was on Caffe (https://github.com/KaimingHe/deep-residual-networks).

CIFAR-10 and ILSVRC2012 datasets are not included in the CNTK distribution. The CIFAR-10 datasets can be downloaded and converted by following the instructions in [DataSets/CIFAR-10](../../DataSets/CIFAR-10). We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files in this folder assumes that by default. For ILSVRC2012 datasets, you may obtain it through http://image-net.org.

## Details

We offer multiple ResNet examples, including ResNet20 and ResNet110 for CIFAR-10 dataset, and ResNet50, ResNet101 and ResNet152 for the ILSVRC2012 dataset (BrainScript only at this moment). For details, please click the respective links below.

### [Python](./Python)

### [BrainScript](./BrainScript)

## Pre-trained Models

### Caffe-Converted

#### ResNet50
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/ResNet50_ImageNet.model
|:---------|:---
|Source Caffe model website | https://github.com/KaimingHe/deep-residual-networks
|Single crop top 5 error | 7.75%

#### ResNet101
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/ResNet101_ImageNet.model
|:---------|:---
|Source Caffe model website | https://github.com/KaimingHe/deep-residual-networks
|Single crop top 5 error | 7.12%

#### ResNet152
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/ResNet152_ImageNet.model
|:---------|:---
|Source Caffe model website | https://github.com/KaimingHe/deep-residual-networks
|Single crop top 5 error | 6.71%
