set cnpath=d:\src\CNTK\CNTKSolution\x64\Release
set proc=%1
echo on
%cnpath%\cn.exe configFile=globals_%proc%.config+TIMIT_TrainSimpleNetwork.config
%cnpath%\cn.exe configFile=globals_%proc%.config+TIMIT_TrainNDLNetwork.config
%cnpath%\cn.exe configFile=globals_%proc%.config+TIMIT_TrainAutoEncoder.config
%cnpath%\cn.exe configFile=globals_%proc%.config+TIMIT_TrainMultiInput.config
%cnpath%\cn.exe configFile=globals_%proc%.config+TIMIT_TrainMultiTask.config
%cnpath%\cn.exe configFile=globals_%proc%.config+TIMIT_EvalSimpleNetwork.config




