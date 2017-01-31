# CNTK Examples: Image - Getting Started

## Overview

|Data:     |The MNIST dataset (http://yann.lecun.com/exdb/mnist/) of handwritten digits.
|:---------|:---
|Purpose   |This folder contains a number of examples that demonstrate the usage of BrainScript to define basic networks for deep learning on image tasks.
|Network   |Simple feed-forward networks including dense layers, convolution layers, drop out and batch normalization for classification and regression tasks.
|Training  |Stochastic gradient descent both with and without momentum.
|Comments  |There are five configuration files, details are provided below.

## Running the example

### Getting the data

These examples use the MNIST dataset to demonstrate various network configurations. MNIST dataset is not included in the CNTK distribution but can be easily downloaded and converted by following the instructions in [DataSets/MNIST](../DataSets/MNIST). We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files in this folder assumes that by default.

### Setup

Compile the sources to generate the cntk executable (not required if you downloaded the binaries).

__Windows:__ Add the folder of the cntk executable to your path
(e.g. `set PATH=%PATH%;c:/src/cntk/x64/Release/;`)
or prefix the call to the cntk executable with the corresponding folder.

__Linux:__ Add the folder of the cntk executable to your path
(e.g. `export PATH=$PATH:$HOME/src/cntk/build/Release/bin/`)
or prefix the call to the cntk executable with the corresponding folder.

### Run

Run the example from the current folder (recommended) using:

`cntk configFile=01_OneHidden.cntk`

or run from any folder and specify the `GettingStarted` folder as the `currentDirectory`,
e.g. running from the `Image` folder using:

`cntk configFile=GettingStarted/01_OneHidden.cntk currentDirectory=GettingStarted`

An Output folder will be created in the `Image/GettingStarted` folder, which is used to store intermediate results and trained models.

## Details

There are five cntk configuration files in the current folder. These cntk configuration files use BrainScript, a custom script language for CNTK. To learn more about BrainScript, please follow the introduction of [BrainScript Basic Concepts](https://github.com/Microsoft/CNTK/wiki/BS-Basic-concepts).

### 01_OneHidden.cntk

This is a simple, one hidden layer network that produces `1.76%` of error. Since this model does not assume any spatial relationships between the pixels, it is often referred as "permutation invariant".

To run this example, use the following command:

`cntk configFile=01_OneHidden.cntk`

In this example, the MNIST images are first normalized to the range `[0,1)`, followed by a single dense hidden layer with 200 nodes. A [rectified linear unit (ReLU)](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf) activation function is added for nonlinearity. Afterwards, another dense linear layer is added to generate the output label. The training adopts cross entropy as the cost function after softmax.

In the `SGD` block, `learningRatesPerSample = 0.01*5:0.005` indicates using 0.01 as learning rate per sample for 5 epochs and then 0.005 for the rest. More details about the SGD block are explained [here](https://github.com/Microsoft/CNTK/wiki/SGD-Block).

The MNIST data is loaded with a simple CNTK text format reader. The train and test datasets are converted by running the Python script in [DataSets/MNIST](../DataSets/MNIST). For more information on the reader block, please refer [here](https://github.com/Microsoft/CNTK/wiki/Reader-block).

### 02_OneConv.cntk

In the second example, we add a convolution layer to the network. Convolution layers were inspired by biological process, and has been extremely popular in image-related tasks, where neighboring pixels have high correlation. One of the earliest papers on convolution neural networks can be found [here](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).

To run this example, use the following command:

`cntk configFile=02_OneConv.cntk`

After normalization, a convolution layer with `16` kernels at size `(5,5)` is added, followed by a ReLU nonlinearity. Then, we perform max pooling on the output feature map, with size `(2,2)` and stride `(2,2)`. A dense layer of 64 hidden nodes is then added, followed by another ReLU, and another dense layer to generate the output. This network achieves `1.22%` error rate, which is better than the previous network.
In practice, one would be stacking multiple convolution layers to improve classification accuracy. State-of-the-art convolution neural networks can achieve lower than 0.5% error rate on MNIST. Interested readers can find more examples in [Classification/ConvNet](../Classification/ConvNet).

### 03_OneConvdropout.cntk

In the third example, we demonstrate the use of dropout layers. Dropout is a network regularization technique that helps combat overfitting, in particular when the network contains many parameters. Dropout, together with ReLU activiation, are the two key techniques that enables Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton to win the ILSVRC-2012 competition, which has argueabally changed the course of computer vision research. Their paper can be found [here](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

To run this example, use the following command:

`cntk configFile=03_OneConvDropout.cntk`

Compared with the previous example, we added a dropout layer after max pooling. Dropout can also be added after dense layer if needed. The dropout rate is specified in the SGD block, as `dropoutRate = 0.5`.

With dropout, the accuracy of the network improves slightly to `1.10%` error rate.

### 04_OneConvBN.cntk

In the fourth example, we add [batch normalization](https://arxiv.org/abs/1502.03167) to the network. Batch normalization was designed to address the internal covariate shift problem caused by input and parameter changes during training. The technique has been proven to be very useful in training very deep and complicated networks.

In this example, we simply added a batch normalization layer to the `02_OneConv.cntk` network. To run this example, use the following command:

`cntk configFile=04_OneConvBN.cntk`

The network achieves around `0.96%` error rate, which is better than the previous examples. Due to the small training dataset and the extremely simple network, we have to stop the training early (10 epochs) in order to avoid overfitting.

This cntk configuration file also demonstrates the use of custom layer definition in BrainScript. Note `ConvBnReluPoolLayer` and `DenseBnReluLayer` are both custom layers that contains different basic layer types.

### 05_OneConvRegr.cntk

In the fifth example, we show how CNTK can be used to perform a regression task. To simplify our task and not introduce any new datasets, we assume the digit labels of MNIST is a regression target rather than a classification target. We then reuse the same network architecture in `02_OneConv`, only to replace the cost function with squared error. To run this example, use the following command:

`cntk configFile=05_OneConvRegr.cntk`

 The trained network achieves root-mean-square error (RMSE) of around 0.05. To see more sophisticated examples on regression tasks, please refer to [Regression](../Regression).

### 06_OneConvRegrMultiNode.cntk

In the sixth example, we show how to train CNTK with multiple process(GPUs) for a regression task. CNTK using MPI for the multiple nodes task, and CNTK currently support four parallel SGD algorithms: DataParallelSGD, BlockMomentumSGD, ModelAveragingSGD, DataParallelASGD. We reuse the same network architecture in `05_OneConvRegr`, only to add a parallel train block. To run this example on a machine, use the following command:

`mpiexec -n 2 cntk configFile=06_OneConvRegrMultiNode.cntk parallelTrain=True parallelizationMethod=DataParallelSGD`

You can change the parallelizationMethod to other three options. To see more detailed guide on multiple GPUs and machines tasks, please refer to [Multiple GPUs and machines](https://github.com/Microsoft/CNTK/wiki/Multiple-GPUs-and-machines).

### 07_Deconvolution.cntk

Example number seven shows how to use Deconvolution and Unpooling to generate a simple image auto encoder. It uses the MNIST dataset, which has a resolution of 28x28x1, encodes it into a 7x7x1 representation using convolution and pooling and decodes to the original resolution. The training criterion is root-mean-square error (RMSE). To run this example, use the following command for BrainScript:

`cntk configFile=07_Deconvolution_BS.cntk`

or this one for Python:

`python 07_Deconvolution_PY.py`

The rmse values for training and testing are 0.225 and 0.223 respectively. To visualize the encoded and decoded images run the following command (from a Python CNTK environment):

`python 07_Deconvolution_Visualizer.py`

The script uses by default the BrainScript model, set `use_brain_script_model=False` to use the Python model for visualization. The visualizations will be stored in the `Output` folder together with a text representation of the encoder and the decoder output.
