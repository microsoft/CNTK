# CNTK Examples: Image/Classification/GoogLeNet/InceptionV3

## Python

### Getting the data

ILSVRC2012 datasets are not included in the CNTK distribution. You may obtain it through http://image-net.org.

### InceptionV3_ImageNet.py

This example is python implementation of Inception-V3 model, which is described in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).

And we're using common SGD with Nesterov momentum in this implementation. Run the example from the current folder using:

`python InceptionV3_ImageNet.py`

For more parameter definitions, please use `-h` command to see the help text:

`python InceptionV3_ImageNet.py -h`

### InceptionV3_ImageNet_Distributed.py

[This example](./InceptionV3_ImageNet_Distributed.py) is similar to InceptionV3_ImageNet.py, but it adds distributed training support.

To run it in a distributed manner, please check [here](https://docs.microsoft.com/en-us/cognitive-toolkit/Multiple-GPUs-and-machines#42-running-parallel-training-with-python). For example, the command for distributed training on the same machine (with multiple GPUs) with Windows is:

`mpiexec -n <#workers> python InceptionV3_ImageNet_Distributed.py`

For more parameter definitions, please use `-h` command to see the help text:

`python InceptionV3_ImageNet_Distributed.py -h`