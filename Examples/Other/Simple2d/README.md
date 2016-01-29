# CNTK example: Simple2d 

## Overview

|Data     |Two dimensional synthetic data
|:--------|:---
|Purpose  |Showcase how to train a simple CNTK network (CPU and GPU) and how to use it for scoring (decoding)
|Network  |SimpleNetworkBuilder, 2 hidden layers with 50 sigmoid nodes each, cross entropy with softmax
|Training |Stochastic gradient descent with momentum
|Comments |There are two config files: Simple.cntk uses a single CPU or GPU, Multigpu.cntk uses data-parallel SGD for training on multiple GPUs

## Running the example

### Getting the data

The data for this example is already contained in the folder Simple2d/Data/.

### Setup

Compile the sources to generate the cntk executable (not required if you downloaded the binaries).

__Windows:__ Add the folder of the cntk executable to your path 
(e.g. `set PATH=%PATH%;c:/src/cntk/x64/Debug/;`) 
or prefix the call to the cntk executable with the corresponding folder. 

__Linux:__ Add the folder of the cntk executable to your path 
(e.g. `export PATH=$PATH:$HOME/src/cntk/build/debug/bin/`) 
or prefix the call to the cntk executable with the corresponding folder. 

### Run

Run the example from the Simple2d/Data folder using:

`cntk configFile=../Config/Simple.cntk`

or run from any folder and specify the Data folder as the `currentDirectory`, 
e.g. running from the Simple2d folder using:

`cntk configFile=Config/Simple.cntk currentDirectory=Data`

The output folder will be created inside Simple2d/.

## Details

### Config files

The config files define a `RootDir` variable and sevearal other variables for directories. 
The `ConfigDir` and `ModelDir` variables define the folders for additional config files and for model files. 
These variables will be overwritten when running on the Philly cluster. 
__It is therefore recommended to generally use `ConfigDir` and `ModelDir` in all config files.__ 
To run on CPU set `deviceId = -1`, to run on GPU set deviceId to "auto" or a specific value >= 0.

Both config files are nearly identical. 
Multigpu.cntk has some additional parameters for parallel training (see parallelTrain in the file).
Both files define the following three commands: train, test and output. 
By default only train and test are executed:

`command=Simple_Demo_Train:Simple_Demo_Test`

The prediction error on the test data is written to stdout. 
The trained models for each epoch are stored in the output models folder. 
In the case of the Multigpu config the console output is written to a file `stderr = DemoOut`.

### Additional files

The 'AdditionalFiles' folder contains the Matlab script that generates the 
training and test data as well as the plots that are provided in the folder. 
The data is synthetic 2d data representing two classes that are separated by a sinusoidal boundary. 
SimpleDemoDataReference.png shows a plot of the training data.

## Using a trained model

The Test (Simple_Demo_Test) and the Output (Simple_Demo_Output) commands 
specified in the config files use the trained model to compute labels for data 
specified in the SimpleDataTest.txt file. The Test command computes prediction 
error, cross entropy and perplexity for the test set and outputs them to the 
console. The Output command writes for each test instance the likelihood per 
label to a file `outputPath = $OutputDir$/SimpleOutput`. 
To use the Output command either set `command=Simple_Demo_Output` in the config 
file or add it to the command line. The model that is used to compute the labels 
in these commands is defined in the modelPath variable at the beginning of the 
file `modelPath=$modelDir$/simple.dnn`.
