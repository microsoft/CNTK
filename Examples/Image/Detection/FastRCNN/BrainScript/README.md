# CNTK Examples: Image/Detection/FastRCNN

## Overview

|Data:     |A toy dataset of images captured from a refrigerator.
|:---------|:---
|Purpose   |This folder contains an end-to-end solution for using Fast R-CNN to perform object detection using a pre-trained AlexNet model and a set of user-supplied additional images.
|Network   |Convolutional neural networks, AlexNet.
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Tutorial

Check out the CNTK Tutorial on [Object Detection using Fast R-CNN](https://docs.microsoft.com/en-us/cognitive-toolkit/Object-Detection-using-Fast-R-CNN).

## Introduction

`Fast R-CNN` is an object detection algorithm proposed by `Ross Girshick` in 2015. The paper is accepted to ICCV 2015, and archived at https://arxiv.org/abs/1504.08083. Fast R-CNN builds on previous work to efficiently classify object proposals using deep convolutional networks. Compared to previous work, Fast R-CNN employs a `region of interest pooling` scheme that allows training to be single stage, with a multi-task loss. It trains the very deep VGG16 network 9x faster than R-CNN, is 213x faster at test-time, and achieves a higher mAP on PASCAL VOC 2012.

In this example, we use [AlexNet](../../../../../PretrainedModels) as a pre-trained model, and adapt it to a toy dataset of images captured from a refrigerator to detect objects inside.

## Running the example

### Getting the data and AlexNet model

We use a toy dataset of images captured from a refrigerator to demonstrate Fast-R-CNN. Both the dataset and the pre-trained AlexNet model can be downloaded by running the following Python command (run from Examples/Image/Detection/FastRCNN folder):

`python install_data_and_model.py`

After running the script, the toy dataset will be installed under the `Examples/Image/DataSets/Grocery` folder. The AlexNet model will be downloaded to the `PretrainedModels` folder in the root CNTK folder. We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files in this folder assumes that by default.

### Setup

Currently, CNTK supports `Python 3.5` and `Python 3.6`. We recommend to install anaconda python (http://continuum.io/downloads) and create a python 3.5 environment using: 
```
conda create --name cntk python=3.5.2 numpy scipy
activate cntk
```
To run the code in this example, you need to install a few additional packages. Under Python 3.5 (64bit version assumed), go to the FastRCNN folder and run:
```
pip install -r requirements.txt
```
You will further need Scikit-Image and OpenCV to run these examples. You can download the corresponding wheel packages and install them manually. For Windows users, visit http://www.lfd.uci.edu/~gohlke/pythonlibs/, and download:

    scikit_image-0.12.3-cp35-cp35m-win_amd64.whl  
    opencv_python-3.1.0-cp35-cp35m-win_amd64.whl

Once you download the respective wheel binaries, install them with:

`pip install your_download_folder/scikit_image-0.12.3-cp35-cp35m-win_amd64.whl`  
`pip install your_download_folder/opencv_python-3.1.0-cp35-cp35m-win_amd64.whl`

This example code assumes you are using 64bit version of Python 3.5 or 3.6, as the Fast R-CNN DLL files under [utils](./fastRCNN/utils) are prebuilt for this version. If your task requires the use of a different Python version, please recompile these DLL files yourself in the correct environment. 

The folder where cntk.exe resides needs to be in your PATH environment variable.

Last but not least, in `PARAMETERS.py`: make sure datasetName is set to "grocery".

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

`python A2_RunWithBSModel.py`

This python code will start training Fast R-CNN using the [fastrcnn.cntk](./fastrcnn.cntk) configuration file (in BrainScript).

If you carefully examine the [fastrcnn.cntk](./fastrcnn.cntk) file, you would notice we load the pre-trained AlexNet model, clone the network up to the `conv5_y` layer and freeze all bottom layer parameters, and then added pooling and dense layers on the top with trainable parameters. The training will run for 17 epochs, and reaching training error around `1.05%`. The script will also write the network output for the entire train and test dataset.

### Running Fast R-CNN distributed training

In case of distributed training, set `distributed_flg` to `True` in [PARAMETERS.py](./PARAMETERS.py).
It will cause `python A2_RunWithPyModel.py` for distributed learning with multi-GPU environment.
Note: This example requires a multi-GPU machine to distribute.

Simple aggregation with a 2-GPU machine:
`mpiexec -n 2 python A2_RunWithPyModel.py`

Please check 2 parameters `num_quantization_bits`, `warm_up` in [PARAMETERS.py](./PARAMETERS.py) for distributed learning.
Here is a [quick reference](https://docs.microsoft.com/en-us/cognitive-toolkit/Multiple-GPUs-and-machines#2-configuring-parallel-training-in-cntk-in-python) for distributed learning with python.

### Evaluate trained model

One the model has been trained for detection, you may run:

`python A3_ParseAndEvaluateOutput.py`

to parse and evaluate the output accuracy. You should see mean average precision (mAP) at around `0.86` for this simple toy example. You may further visualize the detection result on the test data with:

`python B3_VisualizeOutputROIs.py`

## Running Fast R-CNN on other data sets

To learn more about CNTK Fast R-CNN, e.g. how to run it on Pascal VOC data or on your own data set, please go to the CNTK tutorial on [Object Detection using Fast R-CNN](https://docs.microsoft.com/en-us/cognitive-toolkit/Object-Detection-using-Fast-R-CNN).
