# CNTK Examples: Image/Classification/AlexNet

## Overview

|Data:     |The ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) for image classification.
|:---------|:---
|Purpose   |This folder contains examples that demonstrate how to use CNTK to define AlexNet (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) for image classification.
|Network   |AlexNet.
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Running the example

### Getting the data
We use the ILSVRC2012 datasets to demonstrate how to train the AlexNet which won the [ILSVRC](http://www.image-net.org/challenges/LSVRC/) 2012 challenge. AlexNet is an important milestone, as for the first time it was shown that deep convolutional neural networks can outperform traditional manual feature design for vision tasks by a significant margin.

ILSVRC2012 datasets are not included in the CNTK distribution. You may obtain it through http://image-net.org.

## Details

We give examples for both Python and BrainScript. Compared to the original AlexNet, and the Caffe implementation of AlexNet (https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet), our model differs slightly in that we no longer split the convolution layers into two groups (model parallelism). As a result our model has very slightly more parameters, but achieves better accuracy.

### [Python](./Python)

### [BrainScript](./BrainScript)

## Pre-trained Models

### Caffe-Converted

|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/AlexNet_ImageNet.model
|:---------|:---
|Source Caffe model website | https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
|Single crop top 5 error | 19.8%
