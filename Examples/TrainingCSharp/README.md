# CNTK C#/.NET API training examples

## Overview
This folder contains examples using CNTK C# API to build, train, and evaluate CNTK neural network models. 

### Getting the Data and Model
Data and model preparation are done with python scripts.

To install CIFAR10 dataset, change directory to Examples\Image\DataSets\CIFAR-10, run:
***
```python
python install_cifar10.py 
```

To install VGG flower and animal data and to download the ResNet model, change directory to Examples\Image\TransferLearning, run:
***
```python
python install_data_and_model.py
```

### Build and Run Examples
1. Install Nuget package CNTK.CPUOnly version v2.2.0 or higher for CSTrainingCPUOnlyExamples.
2. Install Nuget package CNTK.GPU version v2.2.0 or higher for CSTrainingGPUExamples
3. Run following examples:

#### LogisticRegression
A hello-world example to train and evaluate a logistic regression model using C#/API. See [CNTK 101: Logistic Regression and ML Primer](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_101_LogisticRegression.ipynb) for more details.
#### MNISTClassifier 
This class shows how to build and train a classifier for handwriting data (MNIST).  
#### CifarResNetClassifier 
This class shows how to do image classification using ResNet.
The model being built is a lite version of [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). See [Python Tutorials](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_201B_CIFAR-10_ImageHandsOn.ipynb) for more details.
#### TransferLearning 
This class demonstrates transfer learning using a pretrained ResNet model. 
See [Python Tutorials](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_301_Image_Recognition_with_Deep_Transfer_Learning.ipynb) for more details. 
#### LSTMSequenceClassifier 
This class shows how to build a recurrent neural network model from ground up and how to train the model.

