# CNTK Examples: Image/Classification/AlexNet

## Python

### AlexNet_ImageNet_Distributed.py

Our AlexNet model is a slight variation of the Caffe implementation of AlexNet (https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet). We removed model parallelism since nowadays most modern GPUs can hold the model in memory. Our model's accuracy is about `59.9%` for top-1 category and `82.2%` for top-5 categories, using just the center crop. In comparison, the BLVC AlexNet accuracy is `57.1%` for top-1 category and `80.2%` for top-5 categories. Assuming the ImageNet data folder has been correctly set up, you may achieve our accuracy numbers by launching the command:

`python AlexNet_ImageNet_Distributed.py`

You may use this python script to train AlexNet on multiple GPUs or machines. For a reference on distributed training, please check [here](https://github.com/Microsoft/CNTK/wiki/Multiple-GPUs-and-machines#32-python). For example, the command for distributed training on the same machine (with multiple GPUs) with Windows is:

`mpiexec -n <#workers> python AlexNet_ImageNet_Distributed.py`
