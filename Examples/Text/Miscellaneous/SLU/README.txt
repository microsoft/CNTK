
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
* for example: cntk configFile=globals.cntk+rnnlu.cntk

* note that full paths to config files need to be provided if you are not inside the config directory
* for example 
*  C:\dev\cntk5\x64\release\CNTK.exe configFile=C:\dev\cntk5\ExampleSetups\SLU\globals.cntk+C:\dev\cntk5\ExampleSetups\SLU\rnnlu.cntk

Scoring
* ./score.sh 
* however, need to supply feature and lable files, which are not included in this 
* distribution due to copy right issues.

Path Definitions:
=================
* globals.cntk [defines paths to feature and label files and experiments]

Check training loss
==========================
$ grep Finish log_LSTM_LSTMTest.log
Finished Epoch[1]: [Training Set] Train Loss Per Sample = 0.62975813    EvalErr Per Sample = 0.62975813   Ave Learn Rate Per Sample = 0.1000000015  Epoch Time=5250.689
Finished Epoch[1]: [Validation Set] Train Loss Per Sample = 0.2035009  EvalErr Per Sample = 0.2035009

------ code changed and the following need to be verified ----
------ May 29 2015
--------------------------------------------------------------
Network Training Examples:
==========================
* rnnlu.cntk 

# iter 10, learning rate 0.1
accuracy:  98.01%; precision:  93.75%; recall:  94.04%; FB1:  93.89
# iter 20, learning rate 0.1
accuracy:  98.04%; precision:  94.05%; recall:  94.15%; FB1:  94.10
# iter 30, leraning rate 0.1
accuracy:  98.03%; precision:  94.05%; recall:  94.15%; FB1:  94.10

