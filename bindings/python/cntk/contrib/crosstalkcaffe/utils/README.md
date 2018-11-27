# Global Configuration File (conf.json)

Global configuration file contains all parameters used during the converting. Basically, it contains 3 components:

## SourceSolver

Source solver records the parameters relating to Caffe. 

*Source (String)*: Only support `Caffe` currently.

*ModelPath (String)*: The path to Caffe network prototxt.

*WeightsPath (String)*: The path to Caffe weights file.

*PHASE (String)*: `1` for testing phase, `0` for training phase.

## ModelSolver

Model solver records the parameters relating to CNTK. Currently, there is only a single parameter.

*CNTKModelPath (String)*: The path to save the converted CNTK model.

## ValidSolver

Validation solver will compare the specified nodes during forward prop between Caffe and CNTK. For each compared node, it will statistics maximum/minimum value across the output tensor, and RMSE (Root Mean Square Error) between two platforms. 

*SavePath (String):* The dir to save the temporary files.

*ValInputs (Dict):* The dict of inputs information. Each item contains the name and value range of the input tensor. During validating, the tool will generate randomize value within the value range to fill the tensor and execute forward.

*ValNodes:* The names of validated nodes. Each item contains the name in CNTK and Caffe, respectively. 