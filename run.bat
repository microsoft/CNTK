setlocal
set CONFIG_DIR=%~dp0Tests\EndToEndTests\Image\AlexNet
set RUN_DIR=%~dp0AlexNetBenchRunDir
set CNTK_DIR=%~1
if not defined CNTK_DIR set CNTK_DIR=%~dp0
set BUILD_DIR=%BUILD_DIR%\x64\Release
set CNTK=%CNTK_DIR%\x64\Release\cntk.exe
if not exist "%CNTK%" echo ?Cannot find cntk.exe at %CNTK%&exit /b 1

if not defined DATA_DIR echo ?Specify path to val_256x256.zip in %%DATA_DIR%%%&exit /b 1

if not defined CNTK_DIR set CNTK_DIR=%~dp0
%CNTK% ^
  "configFile=%CONFIG_DIR%\AlexNetCommon.cntk" ^
  command=Train ^
  Train=[SGD=[maxEpochs=2]] ^
  Train=[SGD=[minibatchSize=48]] ^
  numMBsToShowResult=10 ^
  "currentDirectory=%DATA_DIR%" ^
  "RunDir=%RUN_DIR%" ^
  "DataDir=%DATA_DIR%" ^
  "ConfigDir=%CONFIG_DIR%" ^
  "OutputDir=%RUN_DIR%" ^
  DeviceId=0 ^
  timestamping=true ^
  "configFile=%CONFIG_DIR%\AlexNet.cntk" ^
  makeMode=false ^
  "stderr=%RUN_DIR%\bench" ^
  perfTraceLevel=1

findstr epochTime= "%RUN_DIR%\bench_Train.log"
