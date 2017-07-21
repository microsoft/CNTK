# CNTK Examples: Image/Classification/GoogLeNet

## Overview

|Data:     |The ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) for image classification.
|:---------|:---
|Purpose   |This folder contains examples that demonstrate how to use CNTK to define GoogLeNet (https://arxiv.org/abs/1409.4842) and its derivations for image classification.
|Network   |Deep convolutional neural networks codenamed "Inception" (GoogLeNet).
|Training  |See the details.
|Comments  |See below.

## Running the example

### Getting the data
We use the ILSVRC2012 datasets to demonstrate how to train a GoogLeNet. GoogLeNet was initially published by Researchers at Google Inc., and it is fine-tuned to have excellent classification accuracy and low computation cost. It won first place in the [ILSVRC](http://www.image-net.org/challenges/LSVRC/) 2014 detection challenge.


ILSVRC2012 datasets are not included in the CNTK distribution. You may obtain it through http://image-net.org.

## Details

We currently offer the BN-Inception (https://arxiv.org/abs/1502.03167) and Inception V3 (https://arxiv.org/abs/1512.00567), Inception-ResNet-V1 (https://arxiv.org/abs/1602.07261) models.

### [BN-Inception](./BN-Inception)

### [Inception V3](./InceptionV3)

### [Inception-ResNet-V1](./Inception-ResNet-V1)

## Pre-trained Models

### CNTK Pre-trained
Models pre-trained with CNTK scripts.

|CNTK model download path | https://www.cntk.ai/Models/CNTK_Pretrained/InceptionV3_ImageNet_CNTK.model
|:---------|:---
|Training script | [InceptionV3.cntk](./InceptionV3/BrainScript/InceptionV3.cntk)
|Single crop top 1 / top 5 error | 21.620% / 5.810%

### Caffe-Converted

#### BN-Inception
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/BNInception_ImageNet_Caffe.model
|:---------|:---
|Last updated | April, 28th, 2017
|Source Caffe model website | https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
|Single crop top 5 error | 11.50%
