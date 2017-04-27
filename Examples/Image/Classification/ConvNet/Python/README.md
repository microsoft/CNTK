# CNTK Examples: Image/Classification/ConvNet

## Python

### Getting the data

MNIST and CIFAR-10 datasets are not included in the CNTK distribution but can be easily downloaded and converted by following the instructions in [DataSets/MNIST](../../../DataSets/MNIST) and [DataSets/CIFAR-10](../../../DataSets/CIFAR-10). We recommend you to keep the downloaded data in the respective folder while downloading, as the scripts in this folder assume that by default.

### ConvNet_MNIST.py

Our first example applies CNN on the MNIST dataset. The network we use contains three convolution layers and two dense layers. Dropout is applied after the first dense layer. No data augmentation is used in this example.

Run the example from the current folder using:

`python ConvNet_MNIST.py`

The network achieves an error rate around `0.5%`, which is very good considering no data augmentation is used. This accuracy is comparable, if not better, than many other vanilla CNN implementations (http://yann.lecun.com/exdb/mnist/).

### ConvNet_CIFAR10.py

The second example applies CNN on the CIFAR-10 dataset. The network contains four convolution layers and three dense layers. Max pooling is conducted for every two convolution layers. Dropout is applied after the first two dense layers. No data augmentation is used.

Run the example from the current folder using:

`python ConvNet_CIFAR10.py`

The network achieves an error rate of around `18%` after 30 epochs. This is comparable to the network published by [cuda-convnet](https://code.google.com/p/cuda-convnet/), which has 18% error with no data augmentation. One difference is that we do not use a `local response normalization layer`. This layer type is now rarely used in most state-of-the-art deep learning networks.

### ConvNet_CIFAR10_DataAug.py

The third example uses the same CNN as the previous example, but it improves by adding data augmentation to training. For this purpose, we use the `ImageDeserializer` instead of the `CTFDeserializer` to load the data. The image deserializer currently supports crop, flip, scale, color jittering, and mean subtraction.
For a reference on image reader and transforms, please check [here](https://www.cntk.ai/pythondocs/cntk.io.html#cntk.io.ImageDeserializer).

Run the example from the current folder using:

`python ConvNet_CIFAR10_DataAug.py`

We use a fixed crop ratio of `0.8` and scale the image to `32x32` pixels for training. Since all training images are pre-padded to `40x40` pixels, effectively we only perform translation transform without scaling. The accuracy of the network on test data is around `14%`, which is a lot better than the previous model.

### ConvNetLRN_CIFAR10_DataAug.py

The fourth example added local response normalization (LRN) to the previous example. LRN was used in [AlexNet](../../AlexNet), but it has lost its popularity. We implemented the LRN with a 3D convolution.

Run the example from the current folder using:

`python ConvNetLRN_CIFAR10_DataAug.py`

All settings are identical to the previous example. The accuracy of the network on test data is slightly (`0.1-0.2%`) better than the previous example.

### ConvNet_CIFAR10_DataAug_Distributed.py

The fifth example uses the same CNN as ConvNet_CIFAR10_DataAug.py, but it adds support for distributed training with simple aggregation. For a reference on distributed training, please check [here](https://github.com/Microsoft/CNTK/wiki/Multiple-GPUs-and-machines#32-python).
Note that this example will run with a CPU-only build.

`mpiexec -n <#workers> python ConvNet_CIFAR10_DataAug_Distributed.py`
