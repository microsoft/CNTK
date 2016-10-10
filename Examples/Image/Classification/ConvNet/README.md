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

### ConvNet_MNIST.cntk

Our first example applies CNN on the MNIST dataset. The network we use contains three convolution layers and two dense layers. Dropout is applied after the first dense layer. No data augmentation is used in this example. We start the training with no momentum, and add momentum after training for 5 epochs. Please refer to the cntk configuration file [ConvNet_MNIST.cntk](./ConvNet_MNIST.cntk) for more details.

Run the example from the current folder using:

`cntk configFile=ConvNet_MNIST.cntk`

The network achieves an error rate of `0.5%`, which is very good considering no data augmentation is used. This accuracy is comparable, if not better, than many other vanilla CNN implementations (http://yann.lecun.com/exdb/mnist/).

### ConvNet_CIFAR10.cntk

The second exmaple applies CNN on the CIFAR-10 dataset. The network contains four convolution layers and three dense layers. Max pooling is conducted for every two convolution layers. Dropout is applied after the first two dense layers. No data augmentation is used. Please refer to the cntk configuration file [ConvNet_CIFAR10.cntk](./ConvNet_CIFAR10.cntk) for more details.

Run the example from the current folder using:

`cntk configFile=ConvNet_CIFAR10.cntk`

The network achieves an error rate of `18.51%` after 30 epochs. This is comparable to the network published by [cuda-convnet](https://code.google.com/p/cuda-convnet/), which has 18% error with no data augmentation. One difference is that we do not use a `local response normalization layer`. This layer type is now rarely used in most state-of-the-art deep learning networks.

### ConvNet_CIFAR10_DataAug.cntk

The third example uses the same CNN as the previous example, but it improves by adding data augmentation to training. For this purpose, we use the `ImageReader` instead of the `CNTKTextFormatReader` to load the data. The ImageReader currently supports crop, flip, scale, color jittering, and mean subtraction.
For a reference on image reader and transforms, please check [here](https://github.com/Microsoft/CNTK/wiki/Image-reader).

Run the example from the current folder using:

`cntk configFile=ConvNet_CIFAR10_DataAug.cntk`

As seen in the cntk configuration file [ConvNet_CIFAR10_DataAug.cntk](./ConvNet_CIFAR10_DataAug.cntk), we use a fix crop ratio of `0.8` and scale the image to `32x32` pixels for training. Since all training images are pre-padded to `40x40` pixels, effectively we only perfrom translation transform without scaling. The accuracy of the network on test data is `14.39%`, which is a lot better than the previous model.
