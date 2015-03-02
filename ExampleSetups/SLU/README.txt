
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
* for example: cn.exe configFile=globals.config+rnnlu.config

* note that full paths to config files need to be provided if you are not inside the config directory
* for example 
* C:\dev\cntk5\CNTKSolution\x64\Release\cn.exe configFile=C:\dev\cntk5\ExampleSetups\SLU\globals.config+C:\dev\cntk5\ExampleSetups\SLU\rnnlu.config

Scoring
* ./score.sh 
* however, need to supply feature and lable files, which are not included in this 
* distribution due to copy right issues.

Path Definitions:
=================
* globals.config [defines paths to feature and label files and experiments]

Network Training Examples:
==========================
* rnnlu.config 

# iter 10, learning rate 0.1
accuracy:  98.01%; precision:  93.75%; recall:  94.04%; FB1:  93.89
# iter 20, learning rate 0.1
accuracy:  98.04%; precision:  94.05%; recall:  94.15%; FB1:  94.10
# iter 30, leraning rate 0.1
accuracy:  98.03%; precision:  94.05%; recall:  94.15%; FB1:  94.10

