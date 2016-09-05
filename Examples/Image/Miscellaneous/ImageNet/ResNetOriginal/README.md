# CNTK example: ImageNet ResNet 

## Overview

|Data:     |The ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) of images.
|:---------|:---
|Purpose   |This example demonstrates usage of the NDL (Network Description Language) to define networks similar to ResNet.
|Network   |NDLNetworkBuilder, deep convolutional residual networks (ResNet).
|Training  |Stochastic gradient descent with momentum.

## Details
The network configurations and experiment settings in this this folder resemble the ones in the original [ResNet paper](http://arxiv.org/abs/1512.03385) strictly without any extra optimization.
The following table contains results.

| Network       | Top-1 error | Top-5 error | Model
| ------------- | ----------- | ----------- | ----------
| ResNet-50     | 24.58       | 7.43        |

## Notes
This work is an implementation of ResNets in CNTK. If you are interested in the original implementation of ResNet, follow [this link](https://github.com/KaimingHe/deep-residual-networks).