C:\dev\cntk3\CNTKSolution\x64\Release\cntk configFile=globals.cntk+rnnlu.cntk


# expected results, which has a copy at Expected.log is 
$ grep Finish c:/temp/exp/atis/ATIS/log_LSTM_LSTMTest.log
Finished Epoch[1]: [Training Set] Train Loss Per Sample = 4.7967326    EvalErr Per Sample = 4.7967326   Ave Learn Rate Per Sample = 0.1000000015  Epoch Time=0.177
Finished Epoch[1]: [Validation Set] Train Loss Per Sample = 4.6260059  EvalErr Per Sample = 4.6260059
Finished Epoch[2]: [Training Set] Train Loss Per Sample = 4.4580467    EvalErr Per Sample = 4.4580467   Ave Learn Rate Per Sample = 0.1000000015  Epoch Time=0.178
Finished Epoch[2]: [Validation Set] Train Loss Per Sample = 4.0801723  EvalErr Per Sample = 4.0801723
Finished Epoch[3]: [Training Set] Train Loss Per Sample = 3.6568716    EvalErr Per Sample = 3.6568716   Ave Learn Rate Per Sample = 0.1000000015  Epoch Time=0.171
Finished Epoch[3]: [Validation Set] Train Loss Per Sample = 2.6959986  EvalErr Per Sample = 2.6959986

del /q c:\temp\exp\atis
C:\dev\cntk3\CNTKSolution\x64\Release\cntk configFile=globals.cntk+rnnlu.ndl.cntk
#should have the same output as above using simple network builder. 

