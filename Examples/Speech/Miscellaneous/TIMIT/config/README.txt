
These examples demonstrate several ways to train and evaluate acoustic models using CNTK. 
Below is a brief description of the examples. 

** Note that these examples are designed to demonstrate functionality of CNTK. The particular parameter settings do not necessarily result in state of the art performance. **

To Use:
=======
Modify the following files:
* globals.cntk in "configs" to reflect your current experimental setup)
* modify "DeviceNumber" in globals.cntk to specify CPU (<0) or GPU (>=0)
* all SCP files (lists of files) in "lib/scp" to point to your feature files

Run the command line with both globals.cntk and the desired config, separated by a +
* for example: cntk configFile=globals.cntk+TIMIT_TrainLSTM_ndl_deprecated.cntk
* note that full paths to config files need to be provided if you are not inside the config directory

Path Definitions:
=================
* globals.cntk [defines paths to feature and label files and experiments]

Network Training Examples:
==========================
* TIMIT_TrainNDLNetwork_ndl_deprecated.cntk [train a neural network defined using NDL]
* TIMIT_TrainAutoEncoder_ndl_deprecated.cntk [train autoencoder with bottleneck layer]
* TIMIT_TrainWithPreTrain_ndl_deprecated.cntk [pre-train using layerwise discriminative pre-training, then do full network training]
* TIMIT_TrainMultiTask_ndl_deprecated.cntk [train with multi-task learning with joint prediction of senone labels and dialect region]
* TIMIT_TrainMultiInput_ndl_deprecated.cntk [train with 2 different inputs: fbank and mfcc]
* TIMIT_TrainLSTM_ndl_deprecated.cntk [train single layer LSTM network]

Network Output Writing:
=======================
* TIMIT_WriteBottleneck.cntk [write bottleneck features from autoencoder model]

Network Description Language (NDL) & Model Editing Language (MEL) files for experiments:
=======================================================================================
* ae.ndl
* classify.ndl
* mtl_fbank_mfcc.ndl
* mtl_senones_dr.ndl
* create_1layer.ndl
* default_macros.ndl
* lstm.ndl
* add_layer.mel
