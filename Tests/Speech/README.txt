How to run the Tests\Speech test
================================

Full test
---------

Install Cygwin with the python module.

Execute 'Tests/Testdriver.py run' script. This will run the test in Tests/Speech/QuickE2E directory for various configurations. Note that the first time you may get an error about the missing YAML python module that you will need to install. 

Simple command line for debugging
---------------------------------

WORKING DIR: $(SolutionDir)Tests\Speech\Data
COMMAND:     configFile=$(SolutionDir)Tests\Speech\QuickE2E\cntk.config  stderr=$(SolutionDir)Tests\Speech\RunDir\models\cntkSpeech.dnn.log  RunDir=$(SolutionDir)Tests\Speech\RunDir  DataDir=$(SolutionDir)Tests\Speech\Data  DeviceId=Auto


Simple test
-----------

../build/debug/bin/cntk configFile=/home/cbasoglu/src/cntk/.run-linux/Simple.conf
COMMAND:     configFile=$(SolutionDir)Demos\Simple\Simple.config  stderr=$(SolutionDir)Demos\Simple\RunDir\Simple.config.log  RootDir=$(SolutionDir)  DeviceNumber=-1
