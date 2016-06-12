# CNTK example: ImageNet ResNet 

## Overview

|Data:     |The ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) of images.
|:---------|:---
|Purpose   |This example demonstrates usage of the NDL (Network Description Language) to define networks similar to ResNet.
|Network   |NDLNetworkBuilder, deep convolutional residual networks (ResNet).
|Training  |Stochastic gradient descent with momentum.

## Details
The network configurations and experiment settings in this this folder resemble the ones in the original [ResNet paper](http://arxiv.org/abs/1512.03385) with few minor changes inspired by [this work](https://github.com/facebook/fb.resnet.torch).
The following table contains results as well as links to pre-trained models that can be used in various applications.

| Network       | Top-1 error | Top-5 error | Model
| ------------- | ----------- | ----------- | ----------
| ResNet-18     | 29.57       | 10.41       | [Download](https://www.cntk.ai/resnet/ResNet_18.model)
| ResNet-34     | 27.31       | 8.97        | [Download](https://www.cntk.ai/resnet/ResNet_34.model)
| ResNet-50     | 24.74       | 7.56        | [Download](https://www.cntk.ai/resnet/ResNet_50.model)
| ResNet-152    | 22.57       | 6.44        | [Download](https://www.cntk.ai/resnet/ResNet_152.model)

## Notes
This work is an implementation of ResNets in CNTK. If you are interested in the original implementation of ResNet, follow [this link](https://github.com/KaimingHe/deep-residual-networks).