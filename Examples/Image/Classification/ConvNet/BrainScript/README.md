# CNTK Examples: Image/Classification/ConvNet

## BrainScript

### ConvNet_MNIST.cntk

Our first example applies CNN on the MNIST dataset. The network we use contains three convolution layers and two dense layers. Dropout is applied after the first dense layer. No data augmentation is used in this example. We start the training with no momentum, and add momentum after training for 5 epochs. Please refer to the CNTK configuration file [ConvNet_MNIST.cntk](./ConvNet_MNIST.cntk) for more details.

Run the example from the current folder using:

`cntk configFile=ConvNet_MNIST.cntk`

The network achieves an error rate of `0.5%`, which is very good considering no data augmentation is used. This accuracy is comparable, if not better, than many other vanilla CNN implementations (http://yann.lecun.com/exdb/mnist/).

### ConvNet_CIFAR10.cntk

The second example applies CNN on the CIFAR-10 dataset. The network contains four convolution layers and three dense layers. Max pooling is conducted for every two convolution layers. Dropout is applied after the first two dense layers. No data augmentation is used. Please refer to the CTNK configuration file [ConvNet_CIFAR10.cntk](./ConvNet_CIFAR10.cntk) for more details.

Run the example from the current folder using:

`cntk configFile=ConvNet_CIFAR10.cntk`

The network achieves an error rate of around `18%` after 30 epochs. This is comparable to the network published by [cuda-convnet](https://code.google.com/p/cuda-convnet/), which has 18% error with no data augmentation. One difference is that we do not use a `local response normalization layer`. This layer type is now rarely used in most state-of-the-art deep learning networks.

### ConvNet_CIFAR10_DataAug.cntk

The third example uses the same CNN as the previous example, but it improves by adding data augmentation to training. For this purpose, we use the `ImageReader` instead of the `CNTKTextFormatReader` to load the data. The ImageReader currently supports crop, flip, scale, color jittering, and mean subtraction.
For a reference on image reader and transforms, please check [here](https://docs.microsoft.com/en-us/cognitive-toolkit/BrainScript-Image-Reader).

Run the example from the current folder using:

`cntk configFile=ConvNet_CIFAR10_DataAug.cntk`

As seen in the CNTK configuration file [ConvNet_CIFAR10_DataAug.cntk](./ConvNet_CIFAR10_DataAug.cntk), we use a fixed crop ratio of `0.8` and scale the image to `32x32` pixels for training. Since all training images are pre-padded to `40x40` pixels, effectively we only perform translation transform without scaling. The accuracy of the network on test data is around `14.2%`, which is a lot better than the previous model.

### ConvNetLRN_CIFAR10_DataAug.cntk

The fourth example added local response normalization (LRN) to the previous example. LRN is implemented as a BrainScript function using 3D convolution with a constant kernel. You may run the example from the current folder using:

`cntk configFile=ConvNetLRN_CIFAR10_DataAug.cntk`

This model achieves slightly better accuracy of `13.8%`, which demonstrates the effectiveness of LRN. Nevertheless, as mentioned earlier, LRN is now rarely used by state-of-the-art deep networks.
