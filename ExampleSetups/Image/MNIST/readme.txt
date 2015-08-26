This example demonstrates usage of NDL to train 2 neural networks on MNIST dataset (http://yann.lecun.com/exdb/mnist/).
MNIST dataset is not included in CNTK distribution but can be easily downloaded and converted by running the following command from this folder:
python mnist_convert.py
The script will download all required files and convert them to CNTK-supported format.
In case you don't have a Python installed, there are 2 options:
1. Download and install latest version of Python 2.7 from: https://www.python.org/downloads/
Then install numpy package by following instruction from: http://www.scipy.org/install.html#individual-packages
2. Alternatively install Python Anaconda distribution which contains most of the popular Python packages including numpy:
http://continuum.io/downloads

Short description of the networks:

1. 01_OneHidden.ndl is a simple, one hidden layer network that produces 2.3% of error.
To run the sample, navigate to this folder and run the following command:
<path to CNTK executable> configFile=01_OneHidden.config configName=01_OneHidden

2. 02_Convolution.ndl is more interesting, convolutional network which has 2 convolutional and 2 max pooling layers. The network produces 0.87% of error after training for about 2 minutes on GPU.
To run the sample, navigate to this folder and run the following command:
<path to CNTK executable> configFile=02_Conv.config configName=02_Conv

For more details, refer to .ndl and corresponding .config files.

