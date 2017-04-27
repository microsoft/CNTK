# CNTK Examples: Image/Classification/VGG

## Overview

|Data:     |The ILSVRC2012 dataset (http://www.image-net.org/challenges/LSVRC/2012/) for image classification.
|:---------|:---
|Purpose   |This folder contains examples that demonstrate how to use CNTK to define VGG network (https://arxiv.org/abs/1409.1556) for image classification.
|Network   |VGG.
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Running the example

### Getting the data
We use the ILSVRC2012 datasets to demonstrate how to train the VGG model which was developed by the [Visual Geometry Group in University of Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). It won the second place in the ILSVRC-2014 challenge. VGG has been a very popular model for its simple architect and high accuracy.

ILSVRC2012 datasets are not included in the CNTK distribution. You may obtain it through http://image-net.org.

## Details

We give examples for both Python and BrainScript.

### [Python](./Python)

### [BrainScript](./BrainScript)

## Pre-trained Models

### Caffe-Converted

#### VGG16
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet.model
|:---------|:---
|Source Caffe model website | http://www.robots.ox.ac.uk/~vgg/research/very_deep/
|Single crop top 5 error | 10.11%

#### VGG19
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet.model
|:---------|:---
|Source Caffe model website | http://www.robots.ox.ac.uk/~vgg/research/very_deep/ 
|Single crop top 5 error | 10.18%
