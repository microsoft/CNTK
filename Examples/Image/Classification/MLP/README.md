# CNTK Examples: Image/Classification/MLP

## Overview

|Data:     |The MNIST dataset (http://yann.lecun.com/exdb/mnist/) of handwritten digits.
|:---------|:---
|Purpose   |This folder contains a number of examples that demonstrate the usage of BrainScript to define multi-layer perceptron (MLP) networks for image classification.
|Network   |Multi-layer perceptron.
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Running the example

### Getting the data

we use the MNIST dataset to demonstrate how to train a `multi-layer perceptron (MLP)` network. MLP is a feed-forward neural network that consists of multiple layers of nodes in a directed graph, where each layer fully connected to the next one. This is argueabally one of the simplest neural networks.

MNIST dataset is not included in the CNTK distribution but can be easily downloaded and converted by following the instructions in [DataSets/MNIST](../../DataSets/MNIST). We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files in this folder assumes that by default.

## Details

### MLP_MNIST.cntk

Similar to the `01_OneHidden.cntk` network in [GettingStarted](../../GettingStarted), MLP is "permutation invariant". In this particular example, we use 3 hidden layers, each containing `768`, `512` and `256` nodes, respectively. Dropout is applied after each hidden layer, with `droputRate=0.5`. The learning rate is gradually adjusted from `0.001` per sample to `0.0001`, and momentum as time constant is adjusted from `600` (effective momentum = `0.898824`) to `4096` (effective momentum = `0.984495`).

Run the example from the current folder using:

`cntk configFile=MLP_MNIST.cntk`

The network achieves an error rate of `1.45%`, which is about as good as one can have with MLP and no data augmentation (http://yann.lecun.com/exdb/mnist/).
