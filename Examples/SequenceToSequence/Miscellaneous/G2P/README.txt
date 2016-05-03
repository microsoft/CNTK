
This example demonstrates the use of CNTK for letter-to-sound converstion using a
sequence-to-sequence model with attention.

Unfortunately, the data is not public. This shall be addressed in a future update.

To Use:
=======

Modify the following in G2P.cntk:
* pathnames
* deviceId to specify CPU (-1) or GPU (>=0)

Run:
* command line:  cntk  configFile=Examples/SequenceToSequence/Miscellaneous/G2P/G2P.cntk  RunRootDir=g2p
* VS Debugger:   configFile=$(SolutionDir)Examples/SequenceToSequence/Miscellaneous/G2P/G2P.cntk  RunRootDir=$(SolutionDir)g2p
