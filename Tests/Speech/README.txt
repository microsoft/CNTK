How to run the Tests\Speech test
================================

Full test
---------

Install Cygwin with the python module.

Execute 'Tests/Testdriver.py run' script. This will run the test in various Tests (recursively). Note that the first time you may get an error about the missing YAML python module that you will need to install. 

Command lines for debugging
---------------------------

--- QuickE2E:

WORKING DIR: $(SolutionDir)Tests\Speech\Data
COMMAND:     configFile=$(SolutionDir)Tests\Speech\QuickE2E\cntk.config  stderr=$(SolutionDir)Tests\Speech\RunDir\QuickE2E\models\cntkSpeech.dnn.log  RunDir=$(SolutionDir)Tests\Speech\RunDir\QuickE2E  DataDir=$(SolutionDir)Tests\Speech\Data  DeviceId=Auto

# TODO: can stderr refer to RunDir?

--- LSTM:

WORKING DIR: $(SolutionDir)Tests\Speech\Data
COMMAND:     configFile=$(SolutionDir)Tests\Speech\LSTM\cntk.config  stderr=$(SolutionDir)Tests\Speech\RunDir\LSTM\models\cntkSpeech.dnn.log  RunDir=$(SolutionDir)Tests\Speech\RunDir\LSTM  NdlDir=$(SolutionDir)Tests\Speech\LSTM  DataDir=$(SolutionDir)Tests\Speech\Data  DeviceId=Auto

Simple test
-----------

../build/debug/bin/cntk configFile=/home/cbasoglu/src/cntk/.run-linux/Simple.conf
COMMAND:     configFile=$(SolutionDir)Demos\Simple\Simple.config  stderr=$(SolutionDir)Demos\Simple\RunDir\Simple.config.log  RootDir=$(SolutionDir)  DeviceNumber=-1
