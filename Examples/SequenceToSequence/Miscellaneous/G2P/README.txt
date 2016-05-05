
This example demonstrates the use of CNTK for letter-to-sound converstion using a
sequence-to-sequence model with attention.

This example uses the CMUDict as a corpus. The data or a conversion script will be included soon.

To Use:
=======

Modify the following in G2P.cntk:
* pathnames
* deviceId to specify CPU (-1) or GPU (>=0)

Run:
* command line:  cntk  configFile=Examples/SequenceToSequence/Miscellaneous/G2P/G2P.cntk  RunRootDir=g2p
* VS Debugger:   configFile=$(SolutionDir)Examples/SequenceToSequence/Miscellaneous/G2P/G2P.cntk  RunRootDir=$(SolutionDir)g2p
