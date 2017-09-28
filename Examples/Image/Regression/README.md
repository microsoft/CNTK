# CNTK Examples: Image/Regression

## Overview

|Data:     |The CIFAR-10 dataset (http://www.cs.toronto.edu/~kriz/cifar.html) of small images.
|:---------|:---
|Purpose   |This folder contains a number of examples that demonstrate the usage of BrainScript to define deep learning networks for image regression tasks.
|Network   |Convolution neural networks.
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Running the example

### Getting the data

we use the CIFAR-10 dataset to demonstrate how to perform regression on images. CIFAR-10 dataset is not included in the CNTK distribution but can be easily downloaded and converted by following the instructions in [DataSets/CIFAR-10](../DataSets/CIFAR-10). We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files in this folder assumes that by default.

## Details

### RegrSimple_CIFAR10.cntk

In this example, we set up a very simple task to have a neural network predict the average RGB values of images normalized to [0,1). To generate the ground truth labels for this regression task, the CIFAR-10 installation script in [DataSets/CIFAR-10](../DataSets/CIFAR-10) will generate two additional files, `train_regrLabels.txt` and `test_regrLabels.txt`, for train and test respectively.

Run the example from the current folder using:

`cntk configFile=RegrSimple_CIFAR10.cntk`

The network produces root-mean-square error (rmse) of around 0.1257.

You may examine the cntk configuration file [RegrSimple_CIFAR10.cntk](./RegrSimple_CIFAR10.cntk) for more details. Note the network is a linear one without nonlinearity. This is intended as we know that computing the average RGB values of images is a linear operation. The reader is a composite reader that uses the `ImageReader` to read images and the `CNTKTextFormatReader` to read the regression ground truth labels. The configuration file also demonstrates how to write the network prediction for the test data into an output file.
