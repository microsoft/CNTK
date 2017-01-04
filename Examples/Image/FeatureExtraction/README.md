# CNTK Examples: Image/FeatureExtraction

## Overview

|Data:     |A small toy data set of food items in a fridge (grocery).
|:---------|:---
|Purpose   |Demonstrate how to evaluate and write out different layers of a trained model using python.
|Network   |Pre-trained AlexNet model.
|Training  |None, only evaluation of different layers of the model.

## Running the example

### Getting the data

We use the `grocery` toy data set. To download it go to the folder [DataSets/grocery](../DataSets/grocery) and run `python install_grocery.py`. 

Additionally the example requires a pre-trained AlexNet model. Download this model from [https://www.cntk.ai/Models/AlexNet/AlexNetBS.model](https://www.cntk.ai/Models/AlexNet/AlexNetBS.model) and store it in [Examples/Image/PretrainedModels](../PretrainedModels).

### Details

Run `python FeatureExtraction.py` to generate the output of a specific layer. Please refer to the comments in the python code directly for how to choose different layers for evaluation.
