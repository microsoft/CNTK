# CNTK example: Text 

## License

CNTK distribution contains a subset of the data of The Penn Treebank Project (https://www.cis.upenn.edu/~treebank/):

Marcus, Mitchell, Beatrice Santorini, and Mary Ann Marcinkiewicz. Treebank-2 LDC95T7. Web Download. Philadelphia: Linguistic Data Consortium, 1995.

See License.md in the root level folder of the CNTK repository for full license information.

## Overview

|Data      |The Penn Treebank Project (https://www.cis.upenn.edu/~treebank/) annotates naturally-occuring text for linguistic structure .
|:---------|:---|
|Purpose   |Showcase how to train a recurrent network for text data.
|Network   |SimpleNetworkBuilder for recurrent network with two hidden layers.
|Training  |Stochastic gradient descent with adjusted learning rate.
|Comments  |The provided configuration file performs class based RNN training.

## Running the example

### Getting the data

The data for this example is already contained in the folder PennTreebank/Data/.

### Setup

Compile the sources to generate the cntk executable (not required if you downloaded the binaries).

__Windows:__ Add the folder of the cntk executable to your path 
(e.g. `set PATH=%PATH%;c:/src/cntk/x64/Debug/;`) 
or prefix the call to the cntk executable with the corresponding folder. 

__Linux:__ Add the folder of the cntk executable to your path 
(e.g. `export PATH=$PATH:$HOME/src/cntk/build/debug/bin/`) 
or prefix the call to the cntk executable with the corresponding folder. 

### Run

Run the example from the Text/Data folder using:

`cntk configFile=../Config/rnn.config`

or run from any folder and specify the Data folder as the `currentDirectory`, 
e.g. running from the Text folder using:

`cntk configFile=Config/rnn.config currentDirectory=Data`

The output folder will be created inside Text/.

## Details

### Config files

The config files define a `RootDir` variable and sevearal other variables for directories. 
The `ConfigDir` and `ModelDir` variables define the folders for additional config files and for model files. 
These variables will be overwritten when running on the Philly cluster. 
__It is therefore recommended to generally use `ConfigDir` and `ModelDir` in all config files.__ 
To run on CPU set `deviceId = -1`, to run on GPU set deviceId to "auto" or a specific value >= 0.

The configuration contains three commands. 
The first writes the word and class information as three separate files into the data directory.
The training command uses the SimpleNetworkBuilder to build a recurrent network 
using `rnnType = CLASSLSTM` and the LMSequenceReader.
The test command evalutes the trained network agains the specified `testFile`.

The trained models for each epoch are stored in the output models folder. 

### Additional files

The 'AdditionalFiles' folder contains perplexity and expected results files for comparison.
