
These scripts are similar to those in the TIMIT folder of the ExampleSetups except they use much fewer files (100 utterances) and fewer minibatches. See the README.txt file there for more details about these configurations. 

To test on CPU:
CNTK.exe  WorkDir=...  ExpDir=...  LibDir=...  ScpDir=...  configFile=globals.config+select_cpu.config+<DesiredConfigFile>

To test on GPU:
CNTK.exe  WorkDir=...  ExpDir=...  LibDir=...  ScpDir=...  configFile=globals.config+select_gpu.config+<DesiredConfigFile>
