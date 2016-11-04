# CNTK Examples: Image/Classification/ResNet

## Python

### TrainResNet_CIFAR10.py

This example code applies ResNet on the CIFAR-10 dataset. We strictly follow the [ResNet paper](http://arxiv.org/abs/1512.03385) for the network architecture. That is, the network has a first layer of `3x3` convolutions, followed by `6n` layers with `3x3` convolution on the feature maps of size `{32, 16, 8}` respectively, with `2n` layers for each feature map size. For ResNet20, we have `n=3`, for ResNet110, we have `n=18`. The network ends with a global average pooling, a 10-way fully-connected layer, and softmax. [Batch normalization](https://arxiv.org/abs/1502.03167) is applied everywhere except the last fully-connected layer.

We use a fixed crop ratio of `0.8` and scale the image to `32x32` pixels for training. Since all training images are pre-padded to `40x40` pixels, effectively we only perform translation transform without scaling. Run the example from the current folder using:

`python TrainResNet_CIFAR10.py resnet20`  
`python TrainResNet_CIFAR10.py resnet110`

for ResNet20 and ResNet110, respectively. The ResNet20 network achieves an error rate of about `8.2%`, and the ResNet110 network achieves an error rate of about `6.3%`.
