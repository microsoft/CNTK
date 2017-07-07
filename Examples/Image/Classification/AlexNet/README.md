# CNTK Examples: Image/Classification/AlexNet

## Overview

|Data:     |The ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) for image classification.
|:---------|:---
|Purpose   |This folder contains examples that demonstrate how to use CNTK to define AlexNet (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) for image classification.
|Network   |AlexNet.
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Running the example

### Getting the data
We use the ILSVRC2012 datasets to demonstrate how to train the AlexNet which won the [ILSVRC](http://www.image-net.org/challenges/LSVRC/) 2012 challenge. AlexNet is an important milestone, as for the first time it was shown that deep convolutional neural networks can outperform traditional manual feature design for vision tasks by a significant margin.

ILSVRC2012 datasets are not included in the CNTK distribution. You may obtain it through http://image-net.org.

### Preparing the data for processing
Having downloaded the ILSVRC2012 dataset and storing it in $ILSVRC12 path, run the code in [readlabels.py](./readlabels.py) as follows :

```python
python readlabels.py $ILSVRC2012/Data/CLS-LOC/train $ILSVRC2012/Data/CLS-LOC/val $ILSVRC2012/Annotations/CLS-LOC/val 
```
This will create 3 files as follows :
#### train_map.txt 
A text file with the following format -
```
<Full path to image1 of training subset of CLS-LOC competition> <TAB> <Integer Label of the class of image1>
<Full path to image2 of	training subset	of CLS-LOC competition>	<TAB> <Integer Label of	the class of image2>
...
<Full path to image1281167 of	training subset	of CLS-LOC competition>	<TAB> <Integer Label of	the class of image1281167>
```
#### val_map.txt
Same format as above, but for the validation images of CLS-LOC competition
### classmappings.txt
A text file with the following format -
```
<Wordnet ID of class1> <TAB> <Integer label of class1>
<Wordnet ID of class2> <TAB> <Integer label of class2>
...
<Wordnet ID of class1000> <TAB> <Integer label of class1000>
```
## Details

We give examples for both Python and BrainScript. Compared to the original AlexNet, and the Caffe implementation of AlexNet (https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet), our model differs slightly in that we no longer split the convolution layers into two groups (model parallelism). As a result our model has very slightly more parameters, but achieves better accuracy.

### [Python](./Python)

### [BrainScript](./BrainScript)

## Pre-trained Models

### CNTK Pre-trained
Models pre-trained with CNTK scripts.

|CNTK model download path | https://www.cntk.ai/Models/CNTK_Pretrained/AlexNet_ImageNet_CNTK.model
|:---------|:---
|Training script | [AlexNet_ImageNet.cntk](./BrainScript/AlexNet_ImageNet.cntk)
|Single crop top 1 / top 5 error | 40.106% / 17.746%

### Caffe-Converted
Models converted from Caffe model zoo.

|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/AlexNet_ImageNet_Caffe.model
|:---------|:---
|Last updated | April, 28th, 2017
|Source Caffe model website | https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
|Single crop top 5 error | 19.8%
