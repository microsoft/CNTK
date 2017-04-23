# Multiverso Python Binding Benchmark

## Task Description
Perform CIFAR-10 classification with residual networks implementation based on Lasagne.

## Codebase
[Deep_Residual_Learning_CIFAR-10](https://github.com/Microsoft/multiverso/blob/master/binding/python/examples/theano/lasagne/Deep_Residual_Learning_CIFAR-10.py)

## Setup
Please follow [this guide](https://github.com/Microsoft/multiverso/wiki/Multiverso-Python-Theano-Lasagne-Binding) to setup your environment.

## Hardware
|||
| -------- |:--------:|
|Hosts|1|
|GPU|Tesla K40m * 8|
|CPU|Intel(R) Xeon(R) CPU E5-2680 v2 @ 2.80GHz|
|Memory| 251GB |


## Theano settings
Configuration of `~/.theanorc`
```
[global]
device = gpu
floatX = float32

[cuda]
root = /usr/local/cuda-7.5/

[lib]
cnmem = 1
```

## About the Model
|||
| :---- | -----: |
|Total epoch|82|
|Batch size|128|
|Depth|32|
|Learning rate change schedule|Initialized as 0.1, Changed to 0.01 from epoch 41, to 0.001 from epoch 61|
|number of parameters in model|    464,154|


Clarification
- An epoch represents all the processes divide all the data equally and go through them once together.
- A barrier is used at the end of each epoch.
- This experiment doesn't use warm start in ASGD.
- The time to load the data is not considered in the time of the experiment.


# The results
The results of 3 experiments with different configurations are shown as following.

|Short Name | # Process(es) | #GPU(s) per Process | Use multiverso | Batch size | Initial learning rate | Seconds per epoch | Best model validation accuracy |
| :---- | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| 1P1G0M | 1 | 1 | 0 | 128 | 0.1 | 175.4 | 92.69 % |
| 1P1G1M | 1 | 1 | 1 | 128 | 0.1 | 194.4 | 92.53 % |
| 8P1G1M | 8 | 1 | 1 | 64 | 0.05 | 34.1 | 92.11 % |

![accuracy_epoch](https://raw.githubusercontent.com/Microsoft/multiverso/master/binding/python/docs/imgs/accuracy_epoch.png)
![accuracy_time](https://raw.githubusercontent.com/Microsoft/multiverso/master/binding/python/docs/imgs/accuracy_time.png)
