# CNTK Examples: Image/Classification/GoogLeNet/InceptionV3

## Overview

|Data:     |The ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) for image classification.
|:---------|:---
|Purpose   |This folder contains examples that demonstrate how to use CNTK to define Inception V3 (https://arxiv.org/abs/1512.00567) for image classification.
|Network   |Deep convolutional neural networks codenamed "Inception" (GoogLeNet) version 3.
|Training  |RMSProp.
|Comments  |See below.

## Running the example

### Getting the data
We use the ILSVRC2012 datasets to demonstrate how to train an Inception V3 network. Inception V3 was initially published by Researchers at Google Inc., and it is fine-tuned to have excellent classification accuracy and low computation cost. Its original version, GoogLeNet, won first place in the [ILSVRC](http://www.image-net.org/challenges/LSVRC/) 2014 detection challenge.


ILSVRC2012 datasets are not included in the CNTK distribution. You may obtain it through http://image-net.org.

## Details

We currently offer the Inception V3 model, published in December 2015 (https://arxiv.org/abs/1512.00567). Only BrainScript version is available at this moment.

### [BrainScript](./BrainScript)
