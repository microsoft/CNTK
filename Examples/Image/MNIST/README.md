# CNTK example: MNIST 

## Overview

|Data:     |The MNIST database (http://yann.lecun.com/exdb/mnist/) of handwritten digits.
|:---------|:---
|Purpose   |This example demonstrates usage of the NDL (Network Description Language) to define networks.
|Network   |NDLNetworkBuilder, simple feed forward and convolutional networks, cross entropy with softmax.
|Training  |Stochastic gradient descent both with and without momentum.
|Comments  |There are four config files, details are provided below.

## Running the example

### Getting the data

The MNIST dataset is not included in the CNTK distribution but can be easily 
downloaded and converted by running the following command from the 'AdditionalFiles' folder:

`python mnist_convert.py`

The script will download all required files and convert them to CNTK-supported format. 
The resulting files (Train-28x28_cntk_text.txt and Test-28x28_cntk_text.txt) will be stored in the 'Data' folder.
In case you don't have Python installed, there are 2 options:

1. Download and install latest version of Python 2.7 from: https://www.python.org/downloads/ 
Then install the numpy package by following instruction from: http://www.scipy.org/install.html#individual-packages

2. Alternatively install the Python Anaconda distribution which contains most of the 
popular Python packages including numpy: http://continuum.io/downloads

### Setup

Compile the sources to generate the cntk executable (not required if you downloaded the binaries).

__Windows:__ Add the folder of the cntk executable to your path 
(e.g. `set PATH=%PATH%;c:/src/cntk/x64/Debug/;`) 
or prefix the call to the cntk executable with the corresponding folder. 

__Linux:__ Add the folder of the cntk executable to your path 
(e.g. `export PATH=$PATH:$HOME/src/cntk/build/debug/bin/`) 
or prefix the call to the cntk executable with the corresponding folder. 

### Run

Run the example from the Image/MNIST/Data folder using:

`cntk configFile=../Config/01_OneHidden.cntk`

or run from any folder and specify the Data folder as the `currentDirectory`, 
e.g. running from the Image/MNIST folder using:

`cntk configFile=Config/01_OneHidden.cntk currentDirectory=Data`

The output folder will be created inside Image/MNIST/.

## Details

### Config files

There are four config files and the corresponding network description files in the 'Config' folder:

1. 01_OneHidden.ndl is a simple, one hidden layer network that produces 2.3% of error.
To run the sample, navigate to the Data folder and run the following command:  
`cntk configFile=../Config/01_OneHidden.cntk`

2. 02_Convolution.ndl is more interesting, convolutional network which has 2 convolutional and 2 max pooling layers. 
The network produces 0.87% of error after training for about 2 minutes on GPU.
To run the sample, navigate to the Data folder and run the following command:  
`cntk configFile=../Config/02_Convolution.cntk`

3. 03_ConvBatchNorm.ndl is almost identical to 02_Convolution.ndl 
except that it uses batch normalization for the convolutional and fully connected layers.
As a result, it achieves around 0.8% of error after training for just 2 epochs (and less than 30 seconds).
To run the sample, navigate to the Data folder and run the following command:  
`cntk configFile=../Config/03_ConvBatchNorm.cntk`

4. 04_DeConv.ndl illustrates the usage of Deconvolution and Unpooling. It is a network with one Convolution, one Pooling, one Unpooling and one Deconvolution layer. In fact it is an auto-encoder network where Rectified Linear Unit (ReLU) or Sigmoid layer is now replaced with Convolutional ReLU (for encoding) and Deconvolutional ReLU (for decoding) layers. The network goal is to reconstruct the original signal, with Mean Squared Error (MSE) used to minimize the reconstruction error. Generally such networks are used in semantic segmentation.  
To run the sample, navigate to the Data folder and run the following command:  
`cntk configFile=../Config/04_DeConv.cntk` 

For more details, refer to .ndl and the corresponding .cntk files.

### Additional files

The 'AdditionalFiles' folder contains the python script to download and convert the data. 
