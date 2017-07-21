# CNTK Examples: Video - Getting Started

## Overview

|Data:     |The UCF11 dataset (http://crcv.ucf.edu/data/UCF_YouTube_Action.php) of action.
|:---------|:---
|Purpose   |This folder contains a number of examples that demonstrate the usage of Python to define basic 3D convolution networks for deep learning on video tasks.
|Network   |Simple feed-forward networks including dense layers, 3D convolution layers and 3D pooling for action recognition.
|Training  |Stochastic gradient descent.

## Running the example

### Getting the data

This example use the UCF11 dataset to demonstrate 3D convolution. UCF11 dataset is not included in the CNTK distribution but can be easily downloaded and converted by following the instructions in [DataSets/UCF11](../DataSets/UCF11). We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files in this folder assumes that by default.

### Setup

You need to install the latest version of CNTK, please follow the instruction in:
  https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine  

All video examples use CNTK Python API, so make sure that you can call CNTK API from your python environment by following the instruction in the above link.

Also, all examples depend on `imageio` package, to install imageio do the following:

* For Anaconda: `conda install -c conda-forge imageio`
* For pip: `pip install imageio`

### Run

Run the example from the current folder (recommended) using:

`python Conv3D_UCF11.py`
