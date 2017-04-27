# CNTK Examples: Image/TransferLearning

## Overview

|Data:     |A data set containing images of 102 different types of flowers ([website](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)).
|:---------|:---
|Purpose   |Demonstrate how to perform transfer learning in CNTK.
|Network   |Pre-trained ResNet_18 model, which is modified to fit the flowers data set.
|Training  |In this example all layers (old and new) are trained with the same learning rate.

## Running the example

### Getting the data

We use the `Flowers` data set ([Examples/Image/DataSets/Flowers](../DataSets/Flowers)) and a pre-trained ResNet_18 model [Examples/Image/PretrainedModels/ResNet_18.model](../PretrainedModels). To download both run 

`python install_data_and_model.py`

### Details

Run `python TransferLearning.py` to train and evaluate the transfer learning model. The model achieves 93% accuracy on the Flowers data set after training for 20 epochs. A detailed walk through is provided in the ['Build your own image classifier using Transfer Learning'](https://github.com/Microsoft/CNTK/wiki/Build-your-own-image-classifier-using-Transfer-Learning) tutorial on the CNTK github wiki.
