# CNTK Examples: Image/Classification/ResNet

The recipes above are in BrainScript. [For Python click here](https://github.com/Microsoft/CNTK/blob/master/bindings/python/tutorials/CNTK_201A_CIFAR-10_DataLoader.ipynb).

## Overview

|Data:     |The CIFAR-10 dataset (http://www.cs.toronto.edu/~kriz/cifar.html) and the ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) for image classification.
|:---------|:---
|Purpose   |This folder contains a number of examples that demonstrate the usage of BrainScript to define residual network (http://arxiv.org/abs/1512.03385) for image classification.
|Network   |Deep convolutional residual networks (ResNet).
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Running the example

### Getting the data
We use the CIFAR-10 and ILSVRC2012 datasets to demonstrate how to train a deep convolutional residual network (ResNet). ResNet was invented by Researchers at [Microsoft Research](https://www.microsoft.com/en-us/research/), and it won first place in both [ILSVRC](http://www.image-net.org/challenges/LSVRC/) and [MS COCO](http://mscoco.org/) competitions in the year of 2015. The original implementation of ResNet was on Caffe (https://github.com/KaimingHe/deep-residual-networks).

CIFAR-10 and ILSVRC2012 datasets are not included in the CNTK distribution. The CIFAR-10 datasets can be downloaded and converted by following the instructions in [DataSets/CIFAR-10](../../DataSets/CIFAR-10). We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files in this folder assumes that by default. For ILSVRC2012 datasets, you may obtain it through http://image-net.org.

## Details

### ResNet20_CIFAR10.cntk

Our first example applies a relatively shallow ResNet on the CIFAR-10 dataset. We strictly follow the [ResNet paper](http://arxiv.org/abs/1512.03385) for the network architecture. That is, the network has a first layer of `3x3` convolutions, followed by `6n` layers with `3x3` convolution on the feature maps of size `{32, 16, 8}` respectively, with `2n` layers for each feature map size. Note for ResNet20, we have `n=3`. The network ends with a global average pooling, a 10-way fully-connected
layer, and softmax. [Batch normalization](https://arxiv.org/abs/1502.03167) is applied everywhere except the last fully-connected layer.

Other than the network architecture, the CIFAR-10 dataset is augmented with random translation, identical to that in [GettingStarted/ConvNet_CIFAR10_DataAug.cntk](../../GettingStarted/ConvNet_CIFAR10_DataAug.cntk). Please refer to the cntk configuration file [ResNet20_CIFAR10.cntk](./ResNet20_CIFAR10.cntk) for more details.

Run the example from the current folder using:

`cntk configFile=ResNet20_CIFAR10.cntk`

The network achieves an error rate of about `8.2%`, which is lower than the number reported in the original paper.

### ResNet110_CIFAR10.cntk

In this example we increase the depth of the ResNet to 110 layers. That is, we set `n=18`. Only very minor changes are made to the CNTK configuration file. To run this example, use:

`cntk configFile=ResNet110_CIFAR10.cntk`

The network achieves an error rate of about `6.2-6.5%`.

### ResNet50_ImageNet1K.cntk

This is an example using a 50-layer ResNet to train on ILSVRC2012 datasets. Compared with the CIFAR-10 examples, we introduced bottleneck blocks to reduce the amount of computation by replacing the two `3x3` convolutions by a `1x1` convolution, bottlenecked to 1/4 of feature maps, followed by a `3x3` convolution, and then a `1x1` convolution again, with the same number feature maps as input.

Run the example from the current folder using:

`cntk configFile=ResNet50_ImageNet1K.cntk`

### ResNet101_ImageNet1K.cntk

Increase the depth of the ResNet to 101 layers:

`cntk configFile=ResNet101_ImageNet1K.cntk`

### ResNet152_ImageNet1K.cntk

Further increase the depth of the ResNet to 152 layers:

`cntk configFile=ResNet152_ImageNet1K.cntk`
