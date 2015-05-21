set cnpath=d:\gitroot\cntk\CNTKSolution\x64\Release
set proc=%1
echo on
%cnpath%\cntk configFile=globals_%proc%.config+TIMIT_TrainSimpleNetwork.config
%cnpath%\cntk configFile=globals_%proc%.config+TIMIT_TrainNDLNetwork.config
%cnpath%\cntk configFile=globals_%proc%.config+TIMIT_TrainAutoEncoder.config
%cnpath%\cntk configFile=globals_%proc%.config+TIMIT_TrainMultiInput.config
%cnpath%\cntk configFile=globals_%proc%.config+TIMIT_TrainMultiTask.config
%cnpath%\cntk configFile=globals_%proc%.config+TIMIT_EvalSimpleNetwork.config
%cnpath%\cntk configFile=globals_%proc%.config+TIMIT_WriteScaledLogLike.config
%cnpath%\cntk configFile=globals_%proc%.config+TIMIT_WriteBottleneck.config




