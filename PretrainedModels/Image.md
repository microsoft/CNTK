# CNTK Pre-trained Image Models

This page contains pre-trained image models either converted from other toolkits or trained from scratch with CNTK. The list of available models includes:

* [AlexNet](#alexnet)
* [GoogLeNet](#googlenet)
* [ResNet](#resnet)
* [VGG](#vgg)

## AlexNet

### CNTK Pre-trained
#### AlexNet for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/CNTK_Pretrained/AlexNet_ImageNet_CNTK.model
|:---------|:---
|Training script | [AlexNet_ImageNet.cntk](../Examples/Image/Classification/AlexNet/BrainScript/AlexNet_ImageNet.cntk)
|Single crop top 1 / top 5 error | 40.106% / 17.746%

### Caffe-Converted
#### AlexNet for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/AlexNet_ImageNet_Caffe.model (Last updated: April 28, 2017)
|:---------|:---
|Source Caffe model website | https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
|Single crop top 5 error | 19.8%

## GoogLeNet

### CNTK Pre-trained
#### InceptionV3 for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/CNTK_Pretrained/InceptionV3_ImageNet_CNTK.model (Last updated: August 11, 2017)
|:---------|:---
|Training script | [InceptionV3_ImageNet_Distributed.py](../Examples/Image/Classification/GoogLeNet/InceptionV3/Python/InceptionV3_ImageNet_Distributed.py)
|Single crop top 1 / top 5 error | 21.520% / NA

### Caffe-Converted
#### BN-Inception for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/BNInception_ImageNet_Caffe.model (Last updated: April 28, 2017)
|:---------|:---
|Source Caffe model website | https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
|Single crop top 5 error | 11.50%

## ResNet

### CNTK Pre-trained

#### ResNet18 for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/CNTK_Pretrained/ResNet18_ImageNet_CNTK.model (Last updated: Nov. 16, 2017)
|:-------|:---
|Training script | [ResNet18_ImageNet1K.cntk](../Examples/Image/Classification/ResNet/BrainScript/ResNet18_ImageNet1K.cntk)
|Single crop top 1 / top 5 error | 28.752% / 9.700%

#### ResNet34 for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/CNTK_Pretrained/ResNet34_ImageNet_CNTK.model (Last updated: Nov. 16, 2017)
|:---------|:---
|Training script | [ResNet34_ImageNet1K.cntk](../Examples/Image/Classification/ResNet/BrainScript/ResNet34_ImageNet1K.cntk)
|Single crop top 1 / top 5 error | 26.114% / 8.386%

#### ResNet50 for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/CNTK_Pretrained/ResNet50_ImageNet_CNTK.model (Last updated: Nov. 16, 2017)
|:---------|:---
|Training script | [ResNet50_ImageNet1K.cntk](../Examples/Image/Classification/ResNet/BrainScript/ResNet50_ImageNet1K.cntk)
|Single crop top 1 / top 5 error | 23.358% / 6.740%

#### ResNet101 for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/CNTK_Pretrained/ResNet101_ImageNet_CNTK.model (Last updated: Nov. 16, 2017)
|:---------|:---
|Training script | [ResNet101_ImageNet1K.cntk](../Examples/Image/Classification/ResNet/BrainScript/ResNet101_ImageNet1K.cntk)
|Single crop top 1 / top 5 error | 21.822% / 6.042%

#### ResNet152 for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/CNTK_Pretrained/ResNet152_ImageNet_CNTK.model (Last updated: Nov. 16, 2017)
|:---------|:---
|Training script | [ResNet152_ImageNet1K.cntk](../Examples/Image/Classification/ResNet/BrainScript/ResNet152_ImageNet1K.cntk)
|Single crop top 1 / top 5 error | 21.300% / 5.760%

#### ResNet20 for CIFAR-10
|CNTK model download path | https://www.cntk.ai/Models/CNTK_Pretrained/ResNet20_CIFAR10_CNTK.model (Last updated: Nov. 16, 2017)
|:-------|:---
|Training script | [TrainResNet_CIFAR10.py --network resnet20](../Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py)
|Single crop top 1 error | 8.23%

#### ResNet110 for CIFAR-10
|CNTK model download path | https://www.cntk.ai/Models/CNTK_Pretrained/ResNet110_CIFAR10_CNTK.model (Last updated: Nov. 16, 2017)
|:-------|:---
|Training script | [TrainResNet_CIFAR10.py --network resnet110](../Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py)
|Single crop top 1 error | 6.24%

### Caffe-Converted

#### ResNet50 for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/ResNet50_ImageNet_Caffe.model (Last updated: April 28, 2017)
|:---------|:---
|Source Caffe model website | https://github.com/KaimingHe/deep-residual-networks
|Single crop top 5 error | 7.75%

#### ResNet101 for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/ResNet101_ImageNet_Caffe.model (Last updated: April 28, 2017)
|:---------|:---
|Source Caffe model website | https://github.com/KaimingHe/deep-residual-networks
|Single crop top 5 error | 7.12%

#### ResNet152 for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/ResNet152_ImageNet_Caffe.model (Last updated: April 28, 2017)
|:---------|:---
|Source Caffe model website | https://github.com/KaimingHe/deep-residual-networks
|Single crop top 5 error | 6.71%

## VGG

### Caffe-Converted

#### VGG16 for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet_Caffe.model (Last updated: April 28, 2017)
|:---------|:---
|Source Caffe model website | http://www.robots.ox.ac.uk/~vgg/research/very_deep/
|Single crop top 5 error | 10.11%

#### VGG19 for ImageNet 1K
|CNTK model download path | https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet_Caffe.model (Last updated: April 28, 2017)
|:---------|:---
|Source Caffe model website | http://www.robots.ox.ac.uk/~vgg/research/very_deep/
|Single crop top 5 error | 10.18%
