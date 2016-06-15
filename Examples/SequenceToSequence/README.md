
This example demonstrates the use of CNTK for grapheme-to-phoneme (letter-to-sound)
conversion using a sequence-to-sequence model with attention, using the CMUDict dictionary.

The code supports a number of alternative configurations. As configured currently, it implements
* a 3-hidden layer unidirectional LSTM encoder network, all hidden dimensions are 512
* a 3-hidden layer unidirectional LSTM decoder network, all hidden dimensions are 512
* encoder state is passed to the decoder by means of attention, with projection dimension 128 and maximum input length of 20 tokens
* embedding is disabled (because the 'vocabulary' of the task, letters and phonemes, is very small)
* beam decoder with beam width 3

## To Use

Modify the following in G2P.cntk as needed:
* pathnames
* deviceId to specify CPU (-1) or GPU (>=0 or "auto")

Run:
* command line:  ```        cntk  configFile=Examples/SequenceToSequence/CMUDict/Config/G2P.cntk  RunRootDir=g2p```
* VS Debugger:   ```configFile=$(SolutionDir)Examples/SequenceToSequence/CMUDict/Config/G2P.cntk  RunRootDir=$(SolutionDir)Examples/SequenceToSequence/CMUDict```
