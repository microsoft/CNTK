REM test.s26conditionalhashingbilstmn300n300
set TESTITER=%1
set MdlDir=%2
set BW=%3
set Q_ROOT=rpc://speech-data:6673
set QEXE=\\speechstore5\q
set PATH=\\speechstore5\q;%PATH%
set PATH=\\speechstore5\userdata\kaishengy\bin\DLLS;%PATH%

\\speechstore5\userdata\kaishengy\bin\binlujan29\cn.exe configFile=\\speechstore5\userdata\kaishengy\exp\lts\setups\global.lstm.config+\\speechstore5\userdata\kaishengy\exp\lts\setups\s2s.mpd.rnd.hiddenstate.2nets.500.100mb.2layers.config DeviceNumber=-1 command=LSTMTest Iter=%1 MdlDir=%2 bw=%3 LSTMTest=[beamWidth=$bw$] LSTMTest=[encoderModelPath=$MdlDir$\cntkdebug.dnn.encoder.$Iter$] LSTMTest=[decoderModelPath=$MdlDir$\cntkdebug.dnn.decoder.$Iter$] ExpDir=$MdlDir$\test_bw$bw$_iter$Iter$

