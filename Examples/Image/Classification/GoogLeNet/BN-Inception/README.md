# CNTK Examples: Image/Classification/GoogLeNet/BN-Inception

## Overview

|Data:     |The ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) for image classification.
|:---------|:---
|Purpose   |This folder contains examples that demonstrate how to use CNTK to define BN-Inception (https://arxiv.org/abs/1502.03167) for image classification.
|Network   |Deep convolutional neural networks codenamed "Inception" (GoogLeNet) with Batch Normalization.
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Running the example

### Getting the data
We use the ILSVRC2012 datasets to demonstrate how to train a BN-Inception network. BN-Inception was initially published by Researchers at Google Inc., and it is firstly described in the Batch Normalization paper (https://arxiv.org/abs/1502.03167) to demonstrate the power of Batch Normalization with minor changes on the original GoogLeNet. It has been proved that it could increase the training speed and achieve better accuracy, compared with the GoogLeNet v1 which have been well known for winning first place in the [ILSVRC](http://www.image-net.org/challenges/LSVRC/) 2014 detection challenge.


ILSVRC2012 datasets are not included in the CNTK distribution. You may obtain it through http://image-net.org.

## Details

We currently offer the BN-Inception model (https://arxiv.org/abs/1502.03167). Both Python and BrainScript examples are available.

### [Python](./Python)

### [BrainScript](./BrainScript)
