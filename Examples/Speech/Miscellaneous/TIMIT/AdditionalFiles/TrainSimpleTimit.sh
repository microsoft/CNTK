
HTK=/cygdrive/c/Users/mslaney/Projects/HTK/bin.win32
TIMIT=c:/Users/mslaney/Projects/TIMIT/timit/train

HCopyConfig=HCopyTimit.config
HCopyScript=HCopyTimit.scp

python TimitGetFiles.py $TIMIT

$HTK/hcopy.exe -C $HCopyConfig -S $HCopyScript


CNdir=../../../
rm -f Models/TrainSimple.dnn*

time $CNdir/cn.exe configFile=TrainSimpleTimit.cntk

