# CNTK example: Speech AN4

## License

Contents of this directory is a modified version of AN4 dataset pre-processed and optimized for CNTK end-to-end testing. 
The data uses the format required by the HTKMLFReader. For details please refer to the documentation.
The [AN4 dataset](http://www.speech.cs.cmu.edu/databases/an4) is a part of CMU audio databases. 
This modified version of dataset is distributed under the terms of a AN4 license which can be found in 'AdditionalFiles/AN4.LICENSE.html'

See License.md in the root level folder of the CNTK repository for full license information.

## Overview

|Data:     |Speech data from the CMU Audio Database aka AN4 (http://www.speech.cs.cmu.edu/databases/an4)
|:---------|:---|
|Purpose:  |Showcase how to train feed forward and LSTM networks for speech data
|Network:  |SimpleNetworkBuilder for 2-layer FF, NdlNetworkBuilder for 3-layer LSTM network
|Training: |Data-parallel 1-Bit SGD with automatic mini batch rescaling (FF)
|Comments: |There are two config files: FeedForward.config and LSTM-NDL.config for FF and LSTM training respectively

## Running the example

### Getting the data

The data for this example is already contained in the folder AN4/Data/.

### Setup

Compile the sources to generate the cntk executable (not required if you downloaded the binaries).

__Windows:__ Add the folder of the cntk executable to your path 
(e.g. `set PATH=%PATH%;c:/src/cntk/x64/Debug/;`) 
or prefix the call to the cntk executable with the corresponding folder. 

__Linux:__ Add the folder of the cntk executable to your path 
(e.g. `export PATH=$PATH:$HOME/src/cntk/build/debug/bin/`) 
or prefix the call to the cntk executable with the corresponding folder. 

### Run

Run the example from the Speech/Data folder using:

`cntk configFile=../Config/FeedForward.config`

or run from any folder and specify the Data folder as the `currentDirectory`, 
e.g. running from the Speech folder using:

`cntk configFile=Config/FeedForward.config currentDirectory=Data`

The output folder will be created inside Speech/.

## Details

### Config files

The config files define a `RootDir` variable and sevearal other variables for directories. 
The `ConfigDir` and `ModelDir` variables define the folders for additional config files and for model files. 
These variables will be overwritten when running on the Philly cluster. 
__It is therefore recommended to generally use `ConfigDir` and `ModelDir` in all config files.__ 
To run on CPU set `deviceId = -1`, to run on GPU set deviceId to "auto" or a specific value >= 0.

The FeedForward.config file uses the SimpleNetworkBuilder to create a 2-layer 
feed forward network with sigmoid nodes and a softmax layer.
The LSTM-NDL.config file uses the NdlNetworkBuilder and refers to the lstmp-3layer-opt.ndl file. 
In the ndl file an LSTM component is defined and used to create a 3-layer LSTM network with a softmax layer. 
Both configuration only define and execute a single training task:

`command=speechTrain`

The trained models for each epoch are stored in the output models folder. 

### Additional files

The 'AdditionalFiles' folder contains the license terms for the AN4 audio database.
