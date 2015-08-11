::: first argument is CPU or GPU
set PROC=%1
::: second argument is buildconfig (Release or Debug)
set BUILD=%2
echo running ASR test on %PROC%

::: the CNTK executable is found relative to this BAT file
set  THIS=%~dp0
set  ROOT=%THIS%..\..\..

set  CNTK=%ROOT%\x64\%2\CNTK.exe

::: directories we pass in to CNTK config

::: example setups are here
set WorkDir=%ROOT%\ExampleSetups\ASR\TIMIT
set  ExpDir=%THIS%..\test_out
set  LibDir=%WorkDir%\lib
set  ScpDir=%LibDir%\scp

::: run all tests
::: TODO: fix the log path, it seems it cannot be passed to CNTK currently on the command line
for %%t in (TrainSimpleNetwork TrainNDLNetwork TrainAutoEncoder TrainMultiInput TrainMultiTask EvalSimpleNetwork WriteScaledLogLike WriteBottleneck) do (
  echo ------
  echo running test TIMIT_%%t.config logging to %ExpDir%\%%t\log\log_TIMIT_%%t.log
  %CNTK%  WorkDir=%WorkDir%  ExpDir=%ExpDir%  LibDir=%LibDir%  ScpDir=%ScpDir%  configFile=%THIS%globals.config+%THIS%select_%PROC%.config+%THIS%TIMIT_%%t.config
  if ERRORLEVEL 1 (
    echo CNTK FAILED:
    findstr /I EXCEPTION %ExpDir%\%%t\log\log_TIMIT_%%t.log
  ) else (
    echo CNTK OUTPUT:
    findstr /I Finished %ExpDir%\%%t\log\log_TIMIT_%%t.log
    findstr /I EXCEPTION %ExpDir%\%%t\log\log_TIMIT_%%t.log
    echo REFERENCE:
    findstr /I Finished %THIS%..\%PROC%\%%t.output
  )
)
