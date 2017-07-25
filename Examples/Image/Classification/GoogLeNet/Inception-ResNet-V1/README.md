# CNTK Examples: Image/Classification/GoogLeNet/Inception-ResNet-V1

## Overview

|Data:     |The ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) for image classification.
|:---------|:---
|Purpose   |This folder contains examples that demonstrate how to use CNTK to define Inception-ResNet-V1 (https://arxiv.org/abs/1602.07261) for image classification.
|Network   |Deep convolutional neural networks codenamed "Inception" (GoogLeNet) with residual connection version 1.
|Training  |Nesterov's Accelerated Gradient Descent.
|Comments  |See below.

## Running the example

### Getting the data
We use the ILSVRC2012 datasets to demonstrate how to train an Inception-ResNet-V1 network. Inception-ResNet-V1 was initially published by Researchers at Google Inc., and it is fine-tuned to have excellent classification accuracy and low computation cost. Its original version, GoogLeNet, won first place in the [ILSVRC](http://www.image-net.org/challenges/LSVRC/) 2014 detection challenge.


ILSVRC2012 datasets are not included in the CNTK distribution. You may obtain it through http://image-net.org.

## Details

We currently offer Inception-ResNet-V1 model, published in February 2016 (https://arxiv.org/abs/1602.07261). Only BrainScript version is available at this moment.

### [BrainScript](./BrainScript)
