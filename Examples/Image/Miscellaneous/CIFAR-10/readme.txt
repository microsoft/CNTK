This example demonstrates usage of NDL to train a neural network on CIFAR-10 dataset (http://www.cs.toronto.edu/~kriz/cifar.html).
CIFAR-10 dataset is not included in CNTK distribution but can be easily downloaded and converted by running the following command from this folder:
python CIFAR_convert.py [-f <format: cudnn|legacy>]
The script will download all required files and convert them to CNTK-supported format.
In case you don't have a Python installed, there are 2 options:
1. Download and install latest version of Python 2.7 from: https://www.python.org/downloads/
Then install numpy package by following instruction from: http://www.scipy.org/install.html#individual-packages
2. Alternatively install Python Anaconda distribution which contains most of the popular Python packages including numpy:
http://continuum.io/downloads
-f parameter is optional and specifies output format of the datasets. 'cudnn' option (default) saves dataset in spatial-major format used by cuDNN 
while 'legacy' - in CNTK legacy format. Use 'cudnn' if CNTK is compiled with USE_CUDNN option and 'legacy' otherwise.

Short description of the network:

01_Convolution.ndl is a convolutional network which has 3 convolutional and 3 max pooling layers and resembles the network described here:
https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-80sec.cfg 
(main differences are usage of max pooling layers everywhere rather than mix of max and average pooling, as well as dropout in fully-connected layer).
The network produces 21% of error after training for about 3 minutes on GPU.
To run the sample, navigate to this folder and run the following command:
<path to CNTK executable> configFile=01_Conv.cntk configName=01_Conv

02_BatchNormConv.ndl is a convolutional network which uses batch normalization technique (http://arxiv.org/abs/1502.03167).

03_ResNet.ndl and 04_ResNet_56.ndl are very deep convolutional networks that use ResNet architecture and have 20 and 56 layers respectively (http://arxiv.org/abs/1512.03385).
With 03_ResNet.cntk you should get around 8.2% of error after training for about 50 minutes (see log files in the Output directory).

For more details, refer to .ndl and corresponding .cntk files.

