#!/bin/bash

# Copyright 2015  Guoguo Chen
# Apache 2.0.

# Script for CNTK neural network training.

# Begin configuration section.
cmd=run.pl
learning_rate="0.2:1:1:1"
momentum="0:0.9"
max_epochs=50
minibatch_size=20
evaluate_period=100

cntk_train_opts=
cntk_config=cntk_config/CNTK2_lstmp.cntk
default_macros=cntk_config/default_macros.ndl
model_ndl=cntk_config/lstmp-3layer.ndl
device=-1
parallel_opts=
num_threads=1
feature_transform=NO_FEATURE_TRANSFORM
feat_dim=
clipping_per_sample="1#INF"
l2_reg_weight=0
dropout_rate=0
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: steps/$0 <data-dir> <ali-dir> <exp-dir>"
  echo " e.g.: steps/$0 data/sdm1/train exp/sdm1/tri3a_ali exp_cntk/sdm1/dnn"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."

  exit 1;
fi

data=${1%/}
alidir=$2
dir=$3

for f in $data/feats.scp $alidir/final.mdl $alidir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
if [ $feature_transform != "NO_FEATURE_TRANSFORM" ]; then
  [ ! -f $feature_transform ] &&\
    echo "$0: missing file $feature_transform" && exit 1;
fi

mkdir -p $dir/log
mkdir -p $dir/configs
mkdir -p $dir/cntk_model

cp -L $alidir/final.mdl $dir
cp -L $alidir/tree $dir

# Handles parallelization.
if [ $num_threads -gt 1 -a -z "$parallel_opts" ]; then
  parallel_opts="--num-threads $num_threads"
fi
cntk_train_opts="$cntk_train_opts numThreads=$num_threads"

if [ ! -d ${data}_tr90 -o ! -d ${data}_cv10 ]; then
  utils/subset_data_dir_tr_cv.sh $data ${data}_tr90 ${data}_cv10 || exit 1;
fi

cntk_tr_feats=$dir/cntk_train.feats
cntk_tr_labels=$dir/cntk_train.labels
cntk_tr_counts=$dir/cntk_train.counts
cntk_cv_feats=$dir/cntk_valid.feats
cntk_cv_labels=$dir/cntk_valid.labels
cntk_cv_counts=$dir/cntk_valid.counts
cntk_label_mapping=$dir/cntk_label.mapping
feats_tr="scp:${data}_tr90/feats.scp"
feats_cv="scp:${data}_cv10/feats.scp"
if [ -f $alidir/ali.scp ]; then
  labels="ark:ali-to-pdf $alidir/final.mdl scp:$alidir/ali.scp ark:- | ali-to-post ark:- ark:- |"
else
  labels="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
fi

# Prepares training files.
echo "$feats_tr" > $cntk_tr_feats
echo "$labels" > $cntk_tr_labels
(feat-to-len "$feats_tr" ark,t:- > $cntk_tr_counts) || exit 1;
echo "$feats_cv" > $cntk_cv_feats
echo "$labels" > $cntk_cv_labels
(feat-to-len "$feats_cv" ark,t:- > $cntk_cv_counts) || exit 1;
label_dim=$(am-info $alidir/final.mdl 2>/dev/null | grep "pdfs" | awk '{print $4;}') || exit 1;
for ((c = 0; c < label_dim; c++)); do
  echo $c
done > $cntk_label_mapping
if [ -z $feat_dim ]; then feat_dim=$(feat-to-dim "$feats_tr" -) || exit 1; fi

# Copies CNTK config files.
cp -f $cntk_config $dir/configs/Train.config
cp -f $default_macros $dir/configs/default_macros.ndl
cp -f $model_ndl $dir/configs/model.ndl

tee $dir/configs/Base.config <<EOF
ExpDir=$dir
modelName=cntk_model/cntk.nnet

momentum=$momentum
lratePerMB=$learning_rate
l2RegWeight=$l2_reg_weight
dropoutRate=$dropout_rate
maxEpochs=$max_epochs

labelDim=$label_dim
labelMapping=$cntk_label_mapping
featDim=$feat_dim
featureTransform=$feature_transform

inputCounts=$cntk_tr_counts
inputFeats=$cntk_tr_feats
inputLabels=$cntk_tr_labels
cvInputCounts=$cntk_cv_counts
cvInputFeats=$cntk_cv_feats
cvInputLabels=$cntk_cv_labels

minibatchSize=$minibatch_size
evaluatePeriod=$evaluate_period
clippingThresholdPerSample=$clipping_per_sample
EOF

cn_command="cntk configFile=${dir}/configs/Base.config"
cn_command="$cn_command configFile=${dir}/configs/Train.config"
cn_command="$cn_command $cntk_train_opts DeviceNumber=$device"
cn_command="$cn_command command=TrainModel"
$cmd $parallel_opts JOB=1:1 $dir/log/cntk.train.JOB.log $cn_command || exit 1;

echo "$0 successfuly finished.. $dir"

sleep 3
exit 0
