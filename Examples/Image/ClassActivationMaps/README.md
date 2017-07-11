# CNTK Examples: Image/ClassActivationMap

## Overview

|Data:     |A toy dataset of license free images of mainly wolfs and sheep. 
|:---------|:---
|Purpose   |Demonstrate how to compute [class activation maps](http://cnnlocalization.csail.mit.edu/) for visualizing which regions of an image contributed to its classification result.
|Network   |A Resnet network trained on the Animals dataset, per Image/TransferLearning example.
|Training  |No training further training is required to obtain the maps;

## Running the example

### Training the model

Before runnning the example, please download the Animals dataset and run `TransferLearning_ext.py` example in Images/TransferLearning example. 


### Details

Run `python ClassActivationMap.py` to show the the class activation maps for an image.
