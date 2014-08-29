
These examples demonstrate several ways to train and evaluate acoustic models using CNTK. 
Below is a brief description of the examples. 

** Note that these examples are designed to demonstrate functionality of CNTK. The particular parameter settings do not necessarily result in state of the art performance. **

To Use:
=======
Modify the following files:
* globals.config in "configs" to reflect your current experimental setup)
* modify "DeviceNumber" in globals.config to specify CPU (<0) or GPU (>=0)
* all SCP files (lists of files) in "lib/scp" to point to your feature files

Run the command line with both globals.config and the desired config, separated by a +
* for example: cn.exe configFile=globals.config+TIMIT_TrainSimpleNetwork.config
* note that full paths to config files need to be provided if you are not inside the config directory

Path Definitions:
=================
* globals.config [defines paths to feature and label files and experiments]

Network Training Examples:
==========================
* TIMIT_TrainSimpleNetwork.config [train basic feedforward fully connected neural network]
* TIMIT_TrainNDLNetwork.config [train a neural network defined using NDL]
* TIMIT_TrainAutoEncoder.config [train autoencoder with bottleneck layer]
* TIMIT_TrainWithPreTrain.config [pre-train using layerwise discriminative pre-training, then do full network training]
* TIMIT_TrainMultiTask.config [train with multi-task learning with joint prediciton of senone labels and dialect region]
* TIMIT_TrainMultiInput.config [train with 2 different inputs: fbank and mfcc]


Network Evaluation Examples:
============================
* TIMIT_CrossValidateSimpleNetwork.config [evaluate the models at all or some epochs and report best performing model]
* TIMIT_EvalSimpleNetwork.config [evaluate a network]

Network Output Writing:
=======================
* TIMIT_WriteBottleneck.config [write bottleneck features from autoencoder model]
* TIMIT_WriteScaledLogLike.config [write scaled likelihoods from simple model]

Network Description Language (NDL) & Model Editing Language (MEL) files for experiments:
=======================================================================================
* ae.ndl
* classify.ndl
* mtl_fbank_mfcc.ndl
* mtl_senones_dr.ndl
* create_1layer.ndl
* add_layer.mel
* default_macros.ndl
