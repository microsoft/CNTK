# CNTK Examples: Image/Classification/ResNet

> Note: if, on Linux, you experience segmentation faults when trying to run these examples,
> please increase your stack size limit. This can be done by running `ulimit -s 65536` in your shell,
> and then running CNTK from the same session. You can also check you current limits using `ulimit -a`.

## BrainScript

### ResNet20_CIFAR10.cntk

Our first example applies a relatively shallow ResNet on the CIFAR-10 dataset. We strictly follow the [ResNet paper](http://arxiv.org/abs/1512.03385) for the network architecture. That is, the network has a first layer of `3x3` convolutions, followed by `6n` layers with `3x3` convolution on the feature maps of size `{32, 16, 8}` respectively, with `2n` layers for each feature map size. Note for ResNet20, we have `n=3`. The network ends with a global average pooling, a 10-way fully-connected
layer, and softmax. [Batch normalization](https://arxiv.org/abs/1502.03167) is applied everywhere except the last fully-connected layer.

We use a fixed crop ratio of `0.8` and scale the image to `32x32` pixels for training. Since all training images are pre-padded to `40x40` pixels, effectively we only perform translation transform without scaling. Run the example from the current folder using:

`cntk configFile=ResNet20_CIFAR10.cntk`

The network achieves an error rate of about `8.580%`, which is lower than the number reported in the original paper.

### ResNet110_CIFAR10.cntk

In this example we increase the depth of the ResNet to 110 layers. That is, we set `n=18`. Only very minor changes are made to the CNTK configuration file. To run this example, use:

`cntk configFile=ResNet110_CIFAR10.cntk`

The network achieves an error rate of about `6.180%`.

### ResNet50_ImageNet1K.cntk

This is an example using a 50-layer ResNet to train on ILSVRC2012 datasets. Compared with the CIFAR-10 examples, we introduced bottleneck blocks to reduce the amount of computation by replacing the two `3x3` convolutions by a `1x1` convolution, bottlenecked to 1/4 of feature maps, followed by a `3x3` convolution, and then a `1x1` convolution again, with the same number feature maps as input.

Run the example from the current folder using:

`cntk configFile=ResNet50_ImageNet1K.cntk`

### ResNet101_ImageNet1K.cntk

Increase the depth of the ResNet to 101 layers:

`cntk configFile=ResNet101_ImageNet1K.cntk`

### ResNet152_ImageNet1K.cntk

Further increase the depth of the ResNet to 152 layers:

`cntk configFile=ResNet152_ImageNet1K.cntk`
