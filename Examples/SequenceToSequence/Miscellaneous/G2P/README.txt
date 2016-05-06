
This example demonstrates the use of CNTK for letter-to-sound conversion using a
sequence-to-sequence model with attention.

The code supports a number of alternative configurations. As configured currently, it implements
* a 3-hidden layer unidirectional LSTM encoder network, all hidden dimensions are 512
* a 3-hidden layer unidirectional LSTM decoder network, all hidden dimensions are 512
* encoder state is passed to the decoder by means of attention, with projection dimension 128 and maximum input length of 20 tokens
* embedding disabled (the vocabulary is very small)
* beam decoder with beam width 3

This example uses the CMUDict as a corpus. The data or a conversion script will be included soon.

To Use:
=======

Modify the following in G2P.cntk:
* pathnames
* deviceId to specify CPU (-1) or GPU (>=0 or "auto")

Run:
* command line:  cntk  configFile=Examples/SequenceToSequence/Miscellaneous/G2P/G2P.cntk  RunRootDir=g2p
* VS Debugger:   configFile=$(SolutionDir)Examples/SequenceToSequence/Miscellaneous/G2P/G2P.cntk  RunRootDir=$(SolutionDir)g2p
