# CNTK example: CIFAR-10

## Overview

|Data:     |The CIFAR-10 dataset (http://www.cs.toronto.edu/~kriz/cifar.html) of small images.
|:---------|:---
|Purpose   |This example demonstrates usage of the NDL (Network Description Language) to define networks.
|Network   |NDLNetworkBuilder, convolutional networks with batch normalization (including ResNet), cross entropy with softmax.
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Running the example

### Getting the data

CIFAR-10 dataset is not included in CNTK distribution but can be easily downloaded and converted by running the following commands from this folder:

```
python CifarDownload.py [-f <format: cudnn|legacy>]
python CifarConverter.py <path to CIFAR-10 dataset>
```

The scripts will download all required files and convert them to CNTK-supported format.
In case you don't have Python installed (you require Python 2.7 and numpy), we recommend to install the Python Anaconda distribution which contains most of the popular Python packages including numpy:
http://continuum.io/downloads

The download script has an optional `-f` parameter which specifies output format of the datasets. `cudnn` option (default) saves dataset in a spatial-major format used by cuDNN, while `legacy` - in CNTK legacy format. Use `cudnn` if CNTK is compiled with cuDNN and `legacy` otherwise.

The converter script takes a full path to the original CIFAR-10 dataset (in Python pickle format). The script will create `data` folder inside of provided path where it will store both train and test images (in `train` and `test` folders). It will also create appropriate mapping files for the CNTK ImageReader as well as mean file.

## Details

### Config files

1. 01_Convolution.ndl is a convolutional network which has 3 convolutional and 3 max pooling layers and resembles the network described here:
https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-80sec.cfg 
(main differences are usage of max pooling layers everywhere rather than mix of max and average pooling, as well as dropout in fully-connected layer).
The network produces 20.5% of error after training for about 3 minutes on GPU.
To run the sample, navigate to the sample folder and run the following command:
```
cntk configFile=01_Conv_ndl_deprecated.cntk
```
2. 02_BatchNormConv.ndl is a convolutional network which uses batch normalization technique (http://arxiv.org/abs/1502.03167).
To run the sample, navigate to the sample folder and run the following command:
```
cntk configFile=02_BatchNormConv_ndl_deprecated.cntk
```

3. 03_ResNet.ndl and 04_ResNet_56.ndl are very deep convolutional networks that use ResNet architecture and have 20 and 56 layers respectively (http://arxiv.org/abs/1512.03385).
With 03_ResNet_ndl_deprecated.cntk you should get around 8.2% of error after training for about 50 minutes. 04_ResNet_56_ndl_deprecated.cntk should produce around 6.4% of error after training for about 3 hours (see log files in the Output directory).

4. 05_ConvLocal_ndl_deprecated.cntk uses locally-connected convolution layers (see `conv_local3` and `conv_local4` in `05_ConvLocal_ndl_deprecated.cntk`) and resembles a network described here: https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-conv-local-11pct.cfg

5. 06_RegressionSimple.cntk shows how to train a regression model on image data. It uses a very simple network and a composite reader using both the ImageReader and CNTKTextFormatReader and defines a the RMSE (root mean square error) as the loss function. The value that the network learns to predict are simply the average rgb values of an image normalized to [0, 1]. To generate the ground truth labels for regression you need to run the CifarConverter.py script (since this example was added later you might need to rerun it to generate the regression files). See also here: https://github.com/Microsoft/CNTK/wiki/Train-a-regression-model-on-images

For more details, refer to .ndl and corresponding .cntk files.

