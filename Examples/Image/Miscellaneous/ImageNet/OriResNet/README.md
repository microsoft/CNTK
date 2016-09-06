# CNTK example: Original ImageNet ResNet 

## Overview
This work is an implementation of ResNet in CNTK. The work is strictly based on the original [ResNet paper](http://arxiv.org/abs/1512.03385). If you are interested in the original implementation of ResNet, follow [this link](https://github.com/KaimingHe/deep-residual-networks). 

## Dataset
|Data:     |The ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) of images.
|:---------|:---
|Purpose   |This example demonstrates usage of the NDL (Network Description Language) to define networks similar to ResNet.
|Network   |NDLNetworkBuilder, deep convolutional residual networks (ResNet).
|Training  |Stochastic gradient descent with momentum.

## Details
* `Weight Decay in Batch Normalization`: Disable the weight decay in batch normalization. In our experiment, apply weight decay to all nodes will slow down the training curve convergence.
* `Post Batch Normalization`: After training and before evaluating, using post batch normalization command to evaluate the mean and variance of batch normalization nodes instead of running statistics mean and variance. From the experiment, the statistics results of post batch normalization are more robust.

## Results
The following table contains results.

| Network       | Top-1 error | Top-5 error | Model
| ------------- | ----------- | ----------- | ----------
| ResNet-50     | 24.58       | 7.43        |

## Notes