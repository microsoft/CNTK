# CNTK Examples: Image/Classification/ResNet

## Python

### Getting the data

CIFAR-10 datasets are not included in the CNTK distribution but can be easily downloaded and converted by following the instructions in [DataSets/CIFAR-10](../../../DataSets/CIFAR-10). We recommend you to keep the downloaded data in the respective folder while downloading, as the scripts in this folder assume that by default.

### TrainResNet_CIFAR10.py

This example code applies ResNet on the CIFAR-10 dataset. We strictly follow the [ResNet paper](http://arxiv.org/abs/1512.03385) for the network architecture. That is, the network has a first layer of `3x3` convolutions, followed by `6n` layers with `3x3` convolution on the feature maps of size `{32, 16, 8}` respectively, with `2n` layers for each feature map size. For ResNet20, we have `n=3`, for ResNet110, we have `n=18`. The network ends with a global average pooling, a 10-way fully-connected layer, and softmax. [Batch normalization](https://arxiv.org/abs/1502.03167) is applied everywhere except the last fully-connected layer.

We use a fixed crop ratio of `0.8` and scale the image to `32x32` pixels for training. Since all training images are pre-padded to `40x40` pixels, effectively we only perform translation transform without scaling. Run the example from the current folder using:

`python TrainResNet_CIFAR10.py -n resnet20`
`python TrainResNet_CIFAR10.py -n resnet110`

for ResNet20 and ResNet110, respectively. The ResNet20 network achieves an error rate of about `8.23%`, and the ResNet110 network achieves an error rate of about `6.24%`.

### TrainResNet_CIFAR10_Distributed.py

[This example](./TrainResNet_CIFAR10_Distributed.py) is similar to TrainResNet_CIFAR10.py, but it adds support for distributed training via [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface). Details can be found [here](https://docs.microsoft.com/en-us/cognitive-toolkit/Multiple-GPUs-and-machines).
Note this example requires a multi-GPU machine or mpi hosts file to distribute to multiple machines.

Simple aggregation, ResNet20, with a 2-GPU machine:

`mpiexec -n 2 python TrainResNet_CIFAR10_Distributed.py -n resnet20 -q 32`

Quantized 1-bit aggregation with 50000 samples before distributed, ResNet20, with a 2-GPU machine:

`mpiexec -n 2 python TrainResNet_CIFAR10_Distributed.py -n resnet20 -q 1 -a 50000`

To run with maximum parallelization with minibatch size scaled according to #workers for 3 epochs:

`mpiexec -n 2 python TrainResNet_CIFAR10_Distributed.py -s True -e 3`

### TrainResNet_ImageNet_Distributed.py

This example is python implementation of ResNet-V2 model, which is originally described in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

This script supports distributed training feature. To run it in a distributed manner, please check [here](https://docs.microsoft.com/en-us/cognitive-toolkit/Multiple-GPUs-and-machines#42-running-parallel-training-with-python). For example, the command for distributed training on the same machine (with multiple GPUs) with Windows is:

`mpiexec -n <#workers> python TrainResNet_ImageNet_Distributed.py`

Simple aggregation, ResNet50, with a 8-GPU machine:

`mpiexec -n 8 python TrainResNet_ImageNet_Distributed.py -n resnet50`

In our experiments, we achieves `23.65%` top-1 error on ResNet50, `21.61%` top-1 error on ResNet101 and `20.93%` top-1 error on ResNet152.

For more parameter definitions, please use `-h` command to see the help text:

`python TrainResNet_ImageNet_Distributed.py -h`