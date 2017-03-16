# CNTK Examples: Image/Classification/GoogLeNet/BN-Inception

## Python

### Getting the data

CIFAR-10 datasets are not included in the CNTK distribution but can be easily downloaded and converted by following the instructions in [DataSets/CIFAR-10](../../../DataSets/CIFAR-10). We recommend you to keep the downloaded data in the respective folder while downloading, as the scripts in this folder assume that by default.

ILSVRC2012 datasets are not included in the CNTK distribution. You may obtain it through http://image-net.org.

### BN_Inception_CIFAR10.py

This example code applies BN-Inception model on the CIFAR-10 dataset. The network structure is slightly changed and simplified to fit the CIFAR dataset.

We use a fixed crop ratio of `0.8` and scale the image to `32x32` pixels for training. Since all training images are pre-padded to `40x40` pixels, effectively we only perform translation transform without scaling. Run the example from the current folder using:

`python BN_Inception_CIFAR10.py`

In our test, The BN-Inception model for CIFAR achieves an error rate of about `6.0%`.

For more parameter definitions, please use `-h` command to see the help text:

`python BN_Inception_CIFAR10.py -h`

### BN_Inception_CIFAR10_Distributed.py

[This example](./BN_Inception_CIFAR10_Distributed.py) is similar to BN_Inception_CIFAR10.py, but it adds support for distributed training via [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface). Details can be found in [here](https://github.com/Microsoft/CNTK/wiki/Multiple-GPUs-and-machines#32-python).
Note this example requires a multi-GPU machine or mpi hosts file to distribute to multiple machines.

Simple aggregation, BN-Inception, with a 2-GPU machine:

`mpiexec -n 2 python BN_Inception_CIFAR10_Distributed.py`

For more parameter definitions, please use `-h` command to see the help text:

`python BN_Inception_CIFAR10_Distributed.py -h`

### BN_Inception_ImageNet.py

This example is python implementation of BN-Inception model, which is described in [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).

We use a fixed crop ratio of `0.85` and scale the cropped image down to `224x224` pixels for training. Run the example from the current folder using:

`python BN_Inception_ImageNet.py`

For more parameter definitions, please use `-h` command to see the help text:

`python BN_Inception_ImageNet.py -h`

### BN_Inception_ImageNet_Distributed.py

[This example](./BN_Inception_ImageNet_Distributed.py) is similar to BN_Inception_ImageNet.py, but it adds  distributed training support.

To run it in a distributed manner, please check [here](https://github.com/Microsoft/CNTK/wiki/Multiple-GPUs-and-machines#32-python). For example, the command for distributed training on the same machine (with multiple GPUs) with Windows is:

`mpiexec -n <#workers> python BN_Inception_ImageNet_Distributed.py`

For more parameter definitions, please use `-h` command to see the help text:

`python BN_Inception_ImageNet_Distributed.py -h`