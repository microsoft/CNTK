This example demonstrates usage of NDL to train a neural network on CIFAR-10 dataset (http://www.cs.toronto.edu/~kriz/cifar.html).
CIFAR-10 dataset is not included in CNTK distribution but can be easily downloaded and converted by running the following command from this folder:
python CIFAR_convert.py
The script will download all required files and convert them to CNTK-supported format.
In case you don't have a Python installed, there are 2 options:
1. Download and install latest version of Python 2.7 from: https://www.python.org/downloads/
Then install numpy package by following instruction from: http://www.scipy.org/install.html#individual-packages
2. Alternatively install Python Anaconda distribution which contains most of the popular Python packages including numpy:
http://continuum.io/downloads

Short description of the network:

01_Convolution.ndl is a convolutional network which has 3 convolutional and 3 max pooling layers and resembles the network described here:
https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-80sec.cfg 
(main differences are usage of max pooling layers everywhere rather than mix of max and average pooling, as well as dropout in fully-connected layer).
The network produces 22% of error after training for about 4 minutes on GPU.
To run the sample, navigate to this folder and run the following command:
<path to CNTK executable> configFile=01_Conv.config configName=01_Conv

For more details, refer to .ndl and corresponding .config files.

