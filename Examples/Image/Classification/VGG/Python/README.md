# CNTK Examples: Image/Classification/VGG

## Python

### VGG16_ImageNet_Distributed.py

This is the VGG model that contains 16 layers, which was referred as `ConvNet configuration D` in the [original paper](https://arxiv.org/pdf/1409.1556v6.pdf).

Run the example from the current folder using:

`python VGG16_ImageNet_Distributed.py`

To run it in a distributed manner, please check [here](https://github.com/Microsoft/CNTK/wiki/Multiple-GPUs-and-machines#32-python). For example, the command for distributed training on the same machine (with multiple GPUs) with Windows is:

`mpiexec -n <#workers> python VGG16_ImageNet_Distributed.py`

### VGG19_ImageNet_Distributed.py

This is the VGG model that contains 19 layers, which was referred as `ConvNet configuration E` in the [original paper](https://arxiv.org/pdf/1409.1556v6.pdf).

Run the example from the current folder using:

`python VGG19_ImageNet_Distributed.py`

To run it in a distributed manner, please check [here](https://github.com/Microsoft/CNTK/wiki/Multiple-GPUs-and-machines#32-python). For example, the command for distributed training on the same machine (with multiple GPUs) with Windows is:

`mpiexec -n <#workers> python VGG19_ImageNet_Distributed.py` 
