How to run the Tests\Speech test
================================

Command lines for debugging in VS
---------------------------------

Note: Below, the current dir is set to be the data directory. This allows for local paths in data scripts.

--- Speech\QuickE2E:

COMMAND:     currentDirectory=$(SolutionDir)Tests\Speech\Data  configFile=$(SolutionDir)Tests\Speech\QuickE2E\cntk.config  stderr=$(SolutionDir)Tests\Speech\RunDir\QuickE2E\models\cntkSpeech.dnn.log  RunDir=$(SolutionDir)Tests\Speech\RunDir\QuickE2E  DataDir=.  DeviceId=Auto  makeMode=false

--- Speech\LSTM\Truncated:

COMMAND:     currentDirectory=$(SolutionDir)Tests\Speech\Data  configFile=$(SolutionDir)Tests\Speech\LSTM\cntk.config  stderr=$(SolutionDir)Tests\Speech\RunDir\LSTM\Truncated\models\cntkSpeech.dnn.log  RunDir=$(SolutionDir)Tests\Speech\RunDir\LSTM\Truncated  NdlDir=$(SolutionDir)Tests\Speech\LSTM  DataDir=.  DeviceId=Auto  makeMode=false

Using full BrainScript configuration

COMMAND:    --cd $(SolutionDir)Tests\Speech\Data  -f $(SolutionDir)Tests\Speech\LSTM\lstm.bs  -D stderr='$(SolutionDir)Tests\Speech\RunDir\LSTM\Truncated\models\cntkSpeech.dnn.log'  -D RunDir='$(SolutionDir)Tests\Speech\RunDir\LSTM\Truncated'  -D DataDir='.'  -D DeviceId='auto'  -D makeMode=false

--- Speech\LSTM\FullUtterance:

COMMAND:     currentDirectory=$(SolutionDir)Tests\Speech\Data  configFile=$(SolutionDir)Tests\Speech\LSTM\cntk.config  stderr=$(SolutionDir)Tests\Speech\RunDir\LSTM\FullUtterance\models\cntkSpeech.dnn.log  RunDir=$(SolutionDir)Tests\Speech\RunDir\LSTM\FullUtterance  NdlDir=$(SolutionDir)Tests\Speech\LSTM  DataDir=.  DeviceId=Auto  Truncated=false  speechTrain=[reader=[nbruttsineachrecurrentiter=1]] speechTrain=[SGD=[epochSize=2560]]  speechTrain=[SGD=[maxEpochs=2]]  speechTrain=[SGD=[numMBsToShowResult=1]]  makeMode=false

Using parallel sequences (difference to above: nbruttsineachrecurrentiter=4). Note that this will produce a different result since we are confused about what MB size means:

COMMAND:     currentDirectory=$(SolutionDir)Tests\Speech\Data  configFile=$(SolutionDir)Tests\Speech\LSTM\cntk.config  stderr=$(SolutionDir)Tests\Speech\RunDir\LSTM\FullUtterance\models\cntkSpeech.dnn.log  RunDir=$(SolutionDir)Tests\Speech\RunDir\LSTM\FullUtterance  NdlDir=$(SolutionDir)Tests\Speech\LSTM  DataDir=.  DeviceId=Auto  Truncated=false  speechTrain=[reader=[nbruttsineachrecurrentiter=4]]  speechTrain=[SGD=[epochSize=2560]]  speechTrain=[SGD=[learningRatesPerMB=0.125]]  speechTrain=[SGD=[maxEpochs=2]]  speechTrain=[SGD=[numMBsToShowResult=1]]  makeMode=false

Using full BrainScript configuration

COMMAND:     --cd $(SolutionDir)Tests\Speech\Data  -f $(SolutionDir)Tests\Speech\LSTM\lstm.bs  -D stderr='$(SolutionDir)Tests\Speech\RunDir\LSTM\FullUtterance\models\cntkSpeech.dnn.log'  -D RunDir='$(SolutionDir)Tests\Speech\RunDir\LSTM\FullUtterance'  -D NdlDir='$(SolutionDir)Tests\Speech\LSTM'  -D DataDir='.'  -D DeviceId='Auto'  -D Truncated=false  -D speechTrain=[reader=[nbruttsineachrecurrentiter=1];SGD=[epochSize=2560;maxEpochs=2;numMBsToShowResult=1]]  -D makeMode=false

--- Speech\DiscriminativePreTraining:

COMMAND:     currentDirectory=$(SolutionDir)Tests\Speech\Data  configFile=..\DNN\DiscriminativePreTraining\cntk_dpt.config  stderr=..\RunDir\DNN\DiscriminativePreTraining\models\cntkSpeech.dnn.log  RunDir=..\RunDir\DNN\DiscriminativePreTraining  DataDir=.  DeviceId=Auto

--- MNIST:

COMMAND:     currentDirectory=$(SolutionDir)ExampleSetups\Image\MNIST  configFile=02_Conv.config configName=02_Conv


Simple test
-----------

COMMAND:     currentDirectory=$(SolutionDir)Demos\Simple  configFile=Simple.config  stderr=RunDir\Simple.config.log  RootDir=$(SolutionDir)  DeviceNumber=-1
