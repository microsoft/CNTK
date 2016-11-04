# CNTK Examples: Image/Classification/ConvNet

## Overview

|Data:     |The MNIST dataset (http://yann.lecun.com/exdb/mnist/) of handwritten digits and the CIFAR-10 dataset (http://www.cs.toronto.edu/~kriz/cifar.html) for image classification.
|:---------|:---
|Purpose   |This folder contains a number of examples that demonstrate the usage of BrainScript to define convolutional neural networks for image classification.
|Network   |Convolutional neural networks.
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Running the example

### Getting the data

we use the MNIST and CIFAR-10 datasets to demonstrate how to train a `convolutional neural network (CNN)`. CNN has been one of the most popular neural networks for image-related tasks. A very well-known early work on CNN is the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). In 2012 Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton won the ILSVRC-2012 competition using a [CNN architecture](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). And most state-of-the-art neural networks on image classification tasks today adopts a modified CNN architecture, such as [VGG](../VGG), [GoogLeNet](../GoogLeNet), [ResNet](../ResNet), etc.

MNIST and CIFAR-10 datasets are not included in the CNTK distribution but can be easily downloaded and converted by following the instructions in [DataSets/MNIST](../../DataSets/MNIST) and [DataSets/CIFAR-10](../../DataSets/CIFAR-10). We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files in this folder assumes that by default.

## Details

### [Python](./Python)

### [BrainScript](./BrainScript)
