::: the CNTK executable is found relative to this BAT file
set  THIS=%~dp0
set  ROOT=%THIS%\..\..\..

set  CNTK=%ROOT%\x64\Release\CNTK1.exe

::: directories we pass in to CNTK config

::: example setups are here
set WorkDir=%ROOT%\ExampleSetups\ASR\TIMIT
set  ExpDir=%THIS%\..\test_out
set  LibDir=%WorkDir%\lib
set  ScpDir=%LibDir%\scp

::: first argument is CPU or GPU
set proc=%1

%CNTK%  WorkDir=%WorkDir%  ExpDir=%ExpDir%  LibDir=%LibDir%  ScpDir=%ScpDir%  configFile=%THIS%\globals.config+%THIS%\select_%proc%.config+%THIS%\TIMIT_TrainSimpleNetwork.config
%CNTK%  WorkDir=%WorkDir%  ExpDir=%ExpDir%  LibDir=%LibDir%  ScpDir=%ScpDir%  configFile=%THIS%\globals.config+%THIS%\select_%proc%.config+%THIS%\TIMIT_TrainNDLNetwork.config
%CNTK%  WorkDir=%WorkDir%  ExpDir=%ExpDir%  LibDir=%LibDir%  ScpDir=%ScpDir%  configFile=%THIS%\globals.config+%THIS%\select_%proc%.config+%THIS%\TIMIT_TrainAutoEncoder.config
%CNTK%  WorkDir=%WorkDir%  ExpDir=%ExpDir%  LibDir=%LibDir%  ScpDir=%ScpDir%  configFile=%THIS%\globals.config+%THIS%\select_%proc%.config+%THIS%\TIMIT_TrainMultiInput.config
%CNTK%  WorkDir=%WorkDir%  ExpDir=%ExpDir%  LibDir=%LibDir%  ScpDir=%ScpDir%  configFile=%THIS%\globals.config+%THIS%\select_%proc%.config+%THIS%\TIMIT_TrainMultiTask.config
%CNTK%  WorkDir=%WorkDir%  ExpDir=%ExpDir%  LibDir=%LibDir%  ScpDir=%ScpDir%  configFile=%THIS%\globals.config+%THIS%\select_%proc%.config+%THIS%\TIMIT_EvalSimpleNetwork.config
%CNTK%  WorkDir=%WorkDir%  ExpDir=%ExpDir%  LibDir=%LibDir%  ScpDir=%ScpDir%  configFile=%THIS%\globals.config+%THIS%\select_%proc%.config+%THIS%\TIMIT_WriteScaledLogLike.config
%CNTK%  WorkDir=%WorkDir%  ExpDir=%ExpDir%  LibDir=%LibDir%  ScpDir=%ScpDir%  configFile=%THIS%\globals.config+%THIS%\select_%proc%.config+%THIS%\TIMIT_WriteBottleneck.config
