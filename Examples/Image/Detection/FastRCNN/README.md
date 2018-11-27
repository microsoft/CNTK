# CNTK Examples: Image/Detection/Fast R-CNN

## Overview

This folder contains an end-to-end solution for using Fast R-CNN to perform object detection. 
The original research paper for Fast R-CNN can be found at [https://arxiv.org/abs/1504.08083](https://arxiv.org/abs/1504.08083).
Base models that are supported by the current configuration are AlexNet and VGG16. 
Two image sets that are preconfigured are Pascal VOC 2007 and Grocery. 
Other base models or image sets can be used by adding a configuration file similar to the examples in
`Examples/Image/Detection/utils/configs` and importing it in `run_fast_rcnn.py`.

## Running the example

### Setup

To run Fast R-CNN you need a CNTK Python environment. Install the following additional packages:

```
pip install opencv-python easydict pyyaml dlib
```

The code uses prebuild Cython modules for parts of the region proposal network. These binaries are contained in the folder (`Examples/Image/Detection/utils/cython_modules`) for Python 3.5 for Windows and Python 3.5, and 3.6 for Linux.
If you require other versions please follow the instructions at [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#installation-sufficient-for-the-demo).

If you want to use the debug output you need to run `pip install pydot_ng` ([website](https://pypi.python.org/pypi/pydot-ng)) and install [graphviz](http://graphviz.org/) (GraphViz executable has to be in the system’s PATH) to be able to plot the CNTK graphs.

### Getting the data and AlexNet model

We use a toy dataset of images captured from a refrigerator to demonstrate Fast R-CNN. Both the dataset and the pre-trained AlexNet model can be downloaded by running the following Python command from the Examples/Image/Detection/FastRCNN folder:

`python install_data_and_model.py`

After running the script, the toy dataset will be installed under the `Examples/Image/DataSets/Grocery` folder. The AlexNet model will be downloaded to the `PretrainedModels` folder in the root CNTK folder. 
We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files expect that by default.

### Running Fast R-CNN on the example data

To train and evaluate Fast R-CNN run 

`python run_fast_rcnn.py`

### Running Fast R-CNN on Pascal VOC data

To download the Pascal data and create the annotation file for Pascal in CNTK format run the following scripts:

```
python Examples/Image/DataSets/Pascal/install_pascalvoc.py
python Examples/Image/DataSets/Pascal/mappings/create_mappings.py
```

Change the `dataset_cfg` in the `get_configuration()` method of `run_fast_rcnn.py` to

```
from utils.configs.Pascal_config import cfg as dataset_cfg
```

Now you're set to train on the Pascal VOC 2007 data using `python run_fast_rcnn.py`. Beware that training might take a while.

### Running Fast R-CNN on your own data

Preparing your own data and annotating it with ground truth bounding boxes is described [here](https://docs.microsoft.com/en-us/cognitive-toolkit/Object-Detection-using-Fast-R-CNN#train-on-your-own-data).
After storing your images in the described folder structure and annotating them please run

`python Examples/Image/Detection/utils/annotations/annotations_helper.py`

after changing the folder in that script to your data folder. Finally, create a `MyDataSet_config.py` in the `utils/configs` folder following the existing examples:

```
__C.CNTK.DATASET == "YourDataSet":
__C.CNTK.MAP_FILE_PATH = "../../DataSets/YourDataSet"
__C.CNTK.CLASS_MAP_FILE = "class_map.txt"
__C.CNTK.TRAIN_MAP_FILE = "train_img_file.txt"
__C.CNTK.TEST_MAP_FILE = "test_img_file.txt"
__C.CNTK.TRAIN_ROI_FILE = "train_roi_file.txt"
__C.CNTK.TEST_ROI_FILE = "test_roi_file.txt"
__C.CNTK.NUM_TRAIN_IMAGES = 500
__C.CNTK.NUM_TEST_IMAGES = 200
__C.CNTK.PROPOSAL_LAYER_SCALES = [8, 16, 32]
```

Change the `dataset_cfg` in the `get_configuration()` method of `run_fast_rcnn.py` to

```
from utils.configs.MyDataSet_config import cfg as dataset_cfg
```

and run `python run_fast_rcnn.py` to train and evaluate Fast R-CNN on your data.

## Technical details

### Parameters

All options and parameters are in `FastRCNN_config.py` in the `FastRCNN` folder and all of them are explained there. These include

```
# learning parameters
__C.CNTK.MAX_EPOCHS = 10
__C.CNTK.LR_PER_SAMPLE = [0.001] * 10 + [0.0001] * 10 + [0.00001]

# Number of regions of interest [ROIs] proposals
__C.NUM_ROI_PROPOSALS = 1000
# minimum relative width/height of an ROI
__C.roi_min_side_rel = 0.01
# maximum relative width/height of an ROI
__C.roi_max_side_rel = 1.0
```

### Fast R-CNN CNTK code

Most of the code is in `FastRCNN_train.py` and `FastRCNN_eval.py` (and `Examples/Image/Detection/utils/*.py` for helper methods). Please see those files for details.

### Algorithm 

All details regarding the Fast R-CNN algorithm can be found in the original research paper: [https://arxiv.org/abs/1504.08083](https://arxiv.org/abs/1504.08083).
