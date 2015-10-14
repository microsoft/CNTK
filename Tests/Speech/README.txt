How to run the Tests\Speech test
================================

Command lines for debugging
---------------------------

Note: Below, the current dir is set to be the data directory. This allows for local paths in data scripts.

--- Speech\QuickE2E:

COMMAND:     currentDirectory=$(SolutionDir)Tests\Speech\Data  configFile=$(SolutionDir)Tests\Speech\QuickE2E\cntk.config  stderr=$(SolutionDir)Tests\Speech\RunDir\QuickE2E\models\cntkSpeech.dnn.log  RunDir=$(SolutionDir)Tests\Speech\RunDir\QuickE2E  DataDir=.  DeviceId=Auto

Linux:
bin/cntk configFile=Tests/Speech/QuickE2E/cntk.config RunDir=Tests/Speech/RunDirL/QuickE2E DataDir=Tests/Speech/Data DeviceId=0

--- Speech\LSTM\Truncated:

COMMAND:     currentDirectory=$(SolutionDir)Tests\Speech\Data  configFile=$(SolutionDir)Tests\Speech\LSTM\cntk.config  stderr=$(SolutionDir)Tests\Speech\RunDir\LSTM\Truncated\models\cntkSpeech.dnn.log  RunDir=$(SolutionDir)Tests\Speech\RunDir\LSTM\Truncated  NdlDir=$(SolutionDir)Tests\Speech\LSTM  DataDir=.  DeviceId=Auto

--- Speech\LSTM\FullUtterance:

COMMAND:     currentDirectory=$(SolutionDir)Tests\Speech\Data  configFile=$(SolutionDir)Tests\Speech\LSTM\cntk.config  stderr=$(SolutionDir)Tests\Speech\RunDir\LSTM\FullUtterance\models\cntkSpeech.dnn.log  RunDir=$(SolutionDir)Tests\Speech\RunDir\LSTM\FullUtterance  NdlDir=$(SolutionDir)Tests\Speech\LSTM  DataDir=.  DeviceId=Auto Truncated=false speechTrain=[reader=[nbruttsineachrecurrentiter=1]] speechTrain=[SGD=[epochSize=2560]] speechTrain=[SGD=[maxEpochs=2]]  speechTrain=[SGD=[numMBsToShowResult=1]]

--- Speech\DiscriminativePreTraining:

currentDirectory=$(SolutionDir)Tests\Speech\Data  configFile=..\DNN\DiscriminativePreTraining\cntk_dpt.config  stderr=..\RunDir\DNN\DiscriminativePreTraining\models\cntkSpeech.dnn.log  RunDir=..\RunDir\DNN\DiscriminativePreTraining  DataDir=.  DeviceId=Auto

--- MNIST:

COMMAND:     currentDirectory=$(SolutionDir)ExampleSetups\Image\MNIST  configFile=02_Conv.config configName=02_Conv


Simple test
-----------

COMMAND:     currentDirectory=$(SolutionDir)Demos\Simple  configFile=Simple.config  stderr=RunDir\Simple.config.log  RootDir=$(SolutionDir)  DeviceNumber=-1
