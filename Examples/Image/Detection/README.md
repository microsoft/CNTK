# CNTK Examples: Image/Detection/DetectionDemo

## Overview

This folder contains an end-to-end demo to try different object detectors, base models and data sets. The script `DetectionDemo.py` will do the following:

* Train a detector on a specified training set
* Compute average precision per class and mAP on the test set
* Detect objects in a single image (without using a reader)
* Filter the detected objects using NMS (non maximum suppression) and display the boxes on the image

## Running the example

### Setup

To run the object detection demo you need a CNTK Python environment. Install the following additional packages:

```
pip install opencv-python easydict pyyaml future
```

The code uses prebuild Cython modules for parts of the region proposal network (see `Examples/Image/Detection/utils/cython_modules`). 
These binaries are contained in the repository for Python 3.5 under Windows and Python 3.5/3.6 under Linux. For some systems you might need to rename the corresponding module (e.g., rename `cython_bbox.cpython-35m.so` to `cython_bbox.so` for Python 3.5) in order for things to work. If you require other versions please follow the instructions at [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#installation-sufficient-for-the-demo).

If you want to use the debug output you need to run `pip install pydot_ng` ([website](https://pypi.python.org/pypi/pydot-ng)) and install [graphviz](http://graphviz.org/) to be able to plot the CNTK graphs (the GraphViz executable has to be in the system’s PATH).

### Getting the data and AlexNet model

We use a toy dataset of images captured from a refrigerator to demonstrate object detection with CNTK. Both the dataset and the pre-trained AlexNet model can be downloaded by running the following Python command from the Examples/Image/Detection/FastRCNN folder:

`python install_data_and_model.py`

After running the script, the toy dataset will be installed under the `Image/DataSets/Grocery` folder. The AlexNet model will be downloaded to the `Image/PretrainedModels` folder. 
We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files assume that by default.

To download the Pascal data and create the annotation file for Pascal in CNTK format run the following scripts:

```
python Examples/Image/DataSets/Pascal/install_pascalvoc.py
python Examples/Image/DataSets/Pascal/mappings/create_mappings.py
```

### Running the demo

To train and evaluate a detector run

`python DetectionDemo.py`

#### Changing the detector

Currently the demo supports 'FastRCNN' and 'FasterRCNN' detectors. Pass the name of the desired detector to the get_configuration() method from main:

```
    # Currently supported detectors: 'FastRCNN', 'FasterRCNN'
    cfg = get_configuration('FastRCNN')
```

#### Changing the data set

Change the `dataset_cfg` in the `get_configuration()` method:

```
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    from utils.configs.Grocery_config import cfg as dataset_cfg
```

#### Changing the base model

Change the `network_cfg` in the `get_configuration()` method:

```
    # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
```
