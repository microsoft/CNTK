# CNTK Examples: Image/Detection/Fast R-CNN

## Overview

|Data:     |A toy dataset of images captured from a refrigerator.
|:---------|:---
|Purpose   |This folder contains an end-to-end solution for using Fast R-CNN to perform object detection using a pre-trained AlexNet model and a set of user-supplied additional images.
|Network   |Convolutional neural networks, AlexNet.
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Introduction

`Fast R-CNN` is an object detection algorithm proposed by `Ross Girshick` in 2015. The paper is accepted to ICCV 2015, and archived at https://arxiv.org/abs/1504.08083. Fast R-CNN builds on previous work to efficiently classify object proposals using deep convolutional networks. Compared to previous work, Fast R-CNN employs a `region of interest pooling` scheme that allows training to be single stage, with a multi-task loss. It trains the very deep VGG16 network 9x faster than R-CNN, is 213x faster at test-time, and achieves a higher mAP on PASCAL VOC 2012.

In this example, we use [AlexNet](../../Classification/AlexNet) as a pre-trained model, and adapt it to a toy dataset of images captured from a refrigerator to detect objects inside.

## Running the example

### Getting the data and AlexNet model

we use a toy dataset of images captured from a refrigerator to demonstrate Fast-R-CNN. Both the dataset and the pre-trained AlexNet model can be downloaded by running the following Python command:

`python install_fastrcnn.py`

After running the script, the toy dataset will be installed under the `Image/DataSets/Grocery` folder. And the AlexNet model will be downloaded to the `Image/PretrainedModels` folder. We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files in this folder assumes that by default.

### Setup

Currently, CNTK only supports `Python 3.4`, and we hereby assume you already have it installed. To run the code in this example, you need to install a few additional packages. Under Python 3.4, run:

`pip install -r requirements.txt`

You will further need Scikit-Image and OpenCV to run these examples. Unfortunately, for Python 3.4, there is no direct way to install them other than downloading the corresponding wheel packages and install manually. For Windows users, visit http://www.lfd.uci.edu/~gohlke/pythonlibs/, and download:

    scikit_image-0.12.3-cp34-cp34m-win_amd64.whl
    opencv_python-3.1.0-cp34-cp34m-win_amd64.whl

Once you download the respective wheel binaries, install them with:

`pip install your_download_folder/scikit_image-0.12.3-cp34-cp34m-win_amd64.whl`  
`pip install your_download_folder/opencv_python-3.1.0-cp34-cp34m-win_amd64.whl`

Last but not least, in `PARAMETERS.py`: Change 'rootdir' to the absolute path of the FastRCNN folder of your CNTK repository clone (only forward slashes, has to end with forward slash). Also, make sure datasetName is set to "grocery".

### Preprocess data

The toy refrigerator data comes with labeled region of interests. To run Fast R-CNN training, the first step is to generate a large set of potential region proposals via selective search. For this purpose, you can run:

`python A1_GenerateInputROIs.py`

This script will go through all training, validation and testing images, and extract potential region of interests via selective search (https://staff.fnwi.uva.nl/th.gevers/pub/GeversIJCV2013.pdf).

To visualize the generated ROIs, you can run:

`python B1_VisualizeInputROIs.py`

Press any key to step through the images. Further, you may check the recall of the proposed regions by running:

`python B2_EvaluateInputROIs.py`

This will generate something like:

    2016-10-21 07:35:03  
    PARAMETERS: datasetName = grocery  
    PARAMETERS: cntk_nrRois = 2000  
    Processing subdir 'positive', image 0 of 20  
    Processing subdir 'testImages', image 0 of 5  
    Average number of rois per image 1312.92  
    At threshold 0.00: recall = 1.00  
    At threshold 0.10: recall = 1.00  
    At threshold 0.20: recall = 1.00  
    At threshold 0.30: recall = 1.00  
    At threshold 0.40: recall = 1.00  
    At threshold 0.50: recall = 1.00  
    At threshold 0.60: recall = 1.00  
    At threshold 0.70: recall = 1.00  
    At threshold 0.80: recall = 0.84  
    At threshold 0.90: recall = 0.28  
    At threshold 1.00: recall = 0.00  

It shows that up to threashold `0.70`, we have `100%` recall on the ground truth region of interests.

### Running Fast R-CNN training

Now you can start a full training of Fast R-CNN on the grocery data by running:

`python A2_RunCntk.py`

This python code will start training Fast R-CNN using the [fastrcnn.cntk](./fastrcnn.cntk) configuration file (in BrainScript).

If you carefully examine the [fastrcnn.cntk](./fastrcnn.cntk) file, you would notice we load the pre-trained AlexNet model, clone the network up to the `conv5_y` layer and freeze all bottom layer parameters, and then added pooling and dense layers on the top with trainable parameters. The training will run for 17 epochs, and reaching training error around `1.05%`. The script will also write the network output for the entire train and test dataset.

### Evaluate trained model

One the model has been trained for detection, you may run:

`python A3_ParseAndEvaluateOutput.py`

to parse and evaluate the output accuracy. You should see mean average precision (mAP) at around `0.86` for this simple toy example. You may further visualize the detection result on the test data with:

`python B3_VisualizeOutputROIs.py`

## Running Fast R-CNN on Pascal VOC data:

We are still working on enabling the script to run Fast R-CNN on Pascal VOC data, so we only briefly introduce the steps here. Please visit this paper later for more complete and accurate guidance.

- Download the PAscal VOC data to <CntkRoot>/Examples/Image/Datasets/Pascal
-- 2007 trainval: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
-- 2007 test:     http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
-- 2012 trainval: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
- Additionally you need selective_search_data: http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz
- In PARAMETERS.py set datasetName = "pascalVoc".

## Running on your own data

If you want to use another data set than the provided toy example or Pascal, you can capture a few images and label yourselves. We provide two Python scripts to help you.

First, store all your images in a single folder. Edit the `imgDir` in both `C1_DrawBboxesOnImages.py` and `C2_AssignLabelsToBboxes.py` to point to that folder. Save the Python scripts. In `C2_AssignLabelsToBboxes.py`, you may also define your own object categories. If you do modify these categories, you also need to edit `PARAMETERS.py` to reflect that.

In addition, you need to edit `PARAMETERS.py` by:
- Pick a new name and assign it to `datasetName`.
- Adjust `imgDir` to the directory where your images reside.
- Adjust parameters under `project-specific parameter` to your data, i.e. classes etc.

Now you are ready to use the two scripts to do data labeling. With `C1_DrawBboxesOnImages.py`, you may draw all region of interests you want to label. And with `C2_AssignLabelsToBboxes.py`, you can assign a label to each drawn region. The scripts will store the annotations in the correct format for CNTK Fast R-CNN. After all data is labeled, follow instructions earlier to train Fast R-CNN on your new data set.
