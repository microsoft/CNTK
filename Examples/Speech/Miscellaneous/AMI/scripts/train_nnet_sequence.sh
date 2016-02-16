#!/bin/bash

# Copyright 2015  Guoguo Chen
# Apache 2.0.

# Script for sequence discriminative training.


# Begin configuration section.
cmd=run.pl
num_iters=4
acwt=0.1
lmwt=1.0
learning_rate=0.00001
momentum=0
halving_factor=1.0 #ie. disable halving
do_smbr=true
one_silence_class=true # true : reduce insertions in sMBR/MPE FW/BW, more stable training,
sil_phone_list=
criterion="smbr"
minibatch_size=10
truncated=false
evaluate_period=100
one_silence_class=true

cntk_train_opts=
cntk_config=cntk_config/CNTK2_lstmp_smbr.cntk
default_macros=cntk_config/default_macros.ndl
model_ndl=cntk_config/lstmp-3layer.ndl
model_mel=cntk_config/lstmp-3layer-smbr.mel
device=-1
parallel_opts=
num_threads=1
feature_transform=NO_FEATURE_TRANSFORM
feat_dim=
clipping_per_sample="1#INF"
smooth_factor=0.2
init_sequence_model=
l2_reg_weight=0
dropout_rate=0
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 6 ]; then
  echo "Usage: steps/$0 <data> <lang> <srcdir> <ali> <denlats> <exp>"
  echo " e.g.: steps/$0 data/train_all data/lang exp/tri3b_dnn exp/tri3b_dnn_ali exp/tri3b_dnn_denlats exp/tri3b_dnn_smbr"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --num-iters <N>                                  # number of iterations to run"
  echo "  --acwt <float>                                   # acoustic score scaling"
  echo "  --lmwt <float>                                   # linguistic score scaling"
  echo "  --learn-rate <float>                             # learning rate for NN training"
  echo "  --do-smbr <bool>                                 # do sMBR training, otherwise MPE"
  
  exit 1;
fi

data=$(readlink -f $1)
lang=$(readlink -f $2)
srcdir=$(readlink -f $3)
alidir=$(readlink -f $4)
denlatdir=$(readlink -f $5)
dir=$(readlink -f $6)

mkdir -p $dir/log
mkdir -p $dir/configs
mkdir -p $dir/cntk_model

# Handles parallelization.
if [ $num_threads -gt 1 -a -z "$parallel_opts" ]; then
  parallel_opts="--num-threads $num_threads"
fi
cntk_train_opts="$cntk_train_opts numThreads=$num_threads"

# Checks files.
kaldi_model=$srcdir/final.mdl
original_cntk_model=$srcdir/cntk.mdl
cntk_label_mapping=$srcdir/cntk_label.mapping

for f in $lang/phones/silence.csl $data/feats.scp $alidir/ali.scp \
  $denlatdir/lat.scp $kaldi_model $original_cntk_model $cntk_label_mapping; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
if [ $feature_transform != "NO_FEATURE_TRANSFORM" ]; then
  [ ! -f $feature_transform ] &&\
    echo "$0: missing file $feature_transform" && exit 1;
fi

cp -L $srcdir/tree $dir || exit 1;
cp -L $original_cntk_model $dir/original_cntk.mdl || exit 1;
cp -L $kaldi_model $dir || exit 1;
cp -L $cntk_label_mapping $dir || exit 1;

# Silence phones.
if [ -z $sil_phone_list ]; then
  sil_phone_list=`cat $lang/phones/silence.csl` || exit 1;
fi

# Features to be fed to CNTK.
feats="scp:$data/feats.scp"
if [ -z $feat_dim ]; then feat_dim=$(feat-to-dim "$feats" -) || exit 1; fi
label_dim=$(am-info $kaldi_model 2>/dev/null | grep "pdfs" | awk '{print $4;}') || exit 1;

cntk_train_feats=$dir/cntk_train.feats
cntk_train_labels=$dir/cntk_train.labels
cntk_train_counts=$dir/cntk_train.counts
cntk_train_denlats=$dir/cntk_train.denlats
cntk_train_ali=$dir/cntk_train.ali

echo "$feats" > $cntk_train_feats
echo "scp:$denlatdir/lat.scp" > $cntk_train_denlats
echo "scp:$alidir/ali.scp" > $cntk_train_ali
echo "ark:ali-to-pdf $kaldi_model \"scp:$alidir/ali.scp\" ark:- | ali-to-post ark:- ark:- |" > $cntk_train_labels
(feat-to-len "$feats" ark,t:- > $cntk_train_counts) || exit 1;

# Copies CNTK config files.
cp -f $cntk_config $dir/configs/Train.config
cp -f $default_macros $dir/configs/default_macros.ndl
cp -f $model_ndl $dir/configs/model.ndl
cp -f $model_mel $dir/configs/edit.mel

tee $dir/configs/Base.config <<EOF
ExpDir=$dir
NdlDir=$dir/configs

smoothFactor=$smooth_factor
origModelName=$original_cntk_model
modelName=cntk_model/cntk.sequence

momentum=$momentum
lratePerSample=$learning_rate
maxEpochs=$num_iters
l2RegWeight=$l2_reg_weight
dropoutRate=$dropout_rate

featDim=$feat_dim
inputCounts=$cntk_train_counts
inputFeats=$cntk_train_feats
featureTransform=$feature_transform

labelDim=$label_dim
inputLabels=$cntk_train_labels
labelMapping=$cntk_label_mapping

InputDenLats=$cntk_train_denlats
InputAli=$cntk_train_ali
kaldiModel=$kaldi_model
silPhoneList=$sil_phone_list

criterion=$criterion

minibatchSize=$minibatch_size
truncated=$truncated
evaluatePeriod=$evaluate_period
acwt=$acwt
oneSilenceClass=$one_silence_class
clippingThresholdPerSample=$clipping_per_sample
EOF

cn_command="cntk configFile=${dir}/configs/Base.config"
cn_command="$cn_command configFile=$cntk_config"
cn_command="$cn_command $cntk_train_opts DeviceNumber=$device"

if [ ! -z $init_sequence_model ]; then
  cp -L $init_sequence_model $dir/cntk_model/cntk.sequence.0
fi

if [ ! -f $dir/cntk_model/cntk.sequence.0 ]; then
  echo "$0: start editing model"
  cn_command1="$cn_command command=EditModel"
  $cmd $parallel_opts JOB=1:1 $dir/log/cntk.edit.JOB.log $cn_command1 || exit 1; 
  echo "$0: editing model finished."
fi

echo "$0: start sequence training."
cn_command2="$cn_command command=TrainModel"
$cmd $parallel_opts JOB=1:1 $dir/log/cntk.train.JOB.log $cn_command2 || exit 1; 
echo "$0: sequence training finished."

exit 0
