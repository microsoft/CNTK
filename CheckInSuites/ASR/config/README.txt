
These scripts are similar to those in the TIMIT folder of the ExampleSetups except they use much fewer files (100 utterances) and fewer minibatches. See the README.txt file there for more details about these configurations. 

The globals_cpu.config and globals_gpu.config differ only in which device they use and where the results are stored. 

To test on CPU:
cntk configFile=globals_cpu.config+<DesiredConfigFile>

To test on GPU:
cntk configFile=globals_gpu.config+<DesiredConfigFile>
