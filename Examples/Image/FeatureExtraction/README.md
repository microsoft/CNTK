# CNTK Examples: Image/FeatureExtraction

## Overview

|Data:     |A small toy data set of food items in a fridge (grocery).
|:---------|:---
|Purpose   |Demonstrate how to evaluate and write out different layers of a trained model using Python.
|Network   |Pre-trained ResNet_18 model.
|Training  |None, only evaluation of different layers of the model.

## Running the example

### Getting the data

We use the `grocery` toy data set ([Examples/Image/DataSets/Grocery](../DataSets/Grocery)) and a pre-trained ResNet_18 model [Examples/Image/PretrainedModels/ResNet_18.model](../PretrainedModels). To download both run 

`python install_data_and_model.py`

### Details

Run `python FeatureExtraction.py` to generate the output of a specific layer. Please refer to the comments in the Python code directly for how to choose different layers for evaluation.
