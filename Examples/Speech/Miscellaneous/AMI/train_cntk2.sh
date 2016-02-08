#!/bin/bash

# This is the 2nd version of CNTK recipe for AMI corpus.
# In this recipe, CNTK directly read Kaldi features and labels,
# which makes the whole pipline much simpler. Here, we only
# train the standard hybrid DNN model. To train LSTM and PAC-RNN
# models, you have to change the ndl file. -Liang (1/5/2015)

mic=ihm
expdir=exp_cntk/$mic
datadir=data-fmllr-tri4a/$mic
alidir=/exports/work/inf_hcrc_cstr_nst/llu/ami/s5b/exp/$mic/dnn5b_pretrain-dbn_dnn_ali

labelDim=3972  #number of output units, you have to check your gmm setup to get this number
featDim=600
stage=2

cn_gpu=/exports/work/inf_hcrc_cstr_nst/llu/cntk_v2/bin/cn.exe

feats_tr="scp:$datadir/train_tr90/feats.scp"
feats_cv="scp:$datadir/train_cv10/feats.scp"
labels_tr="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
#labels_cv="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir_cv/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
  
mkdir -p $expdir

if [ $stage -le 0 ] ; then
(feat-to-len "$feats_tr" ark,t:- > $expdir/cntk_train.counts) || exit 1;
echo "$feats_tr" > $expdir/cntk_train.feats
echo "$labels_tr" > $expdir/cntk_train.labels

(feat-to-len "$feats_cv" ark,t:- > $expdir/cntk_valid.counts) || exit 1;
echo "$feats_cv" > $expdir/cntk_valid.feats
echo "$labels_tr" > $expdir/cntk_valid.labels

for (( c=0; c<labelDim; c++)) ; do
  echo $c
done >$expdir/cntk_label.mapping

fi


if [ $stage -le 1 ] ; then

### setup the configuration files for training CNTK models ###
cp cntk_config/CNTK2.cntk $expdir/CNTK2.cntk
cp cntk_config/default_macros.ndl $expdir/default_macros.ndl
cp cntk_config/dnn_6layer.ndl $expdir/dnn_6layer.ndl
ndlfile=$expdir/dnn_6layer.ndl

tee $expdir/Base.config <<EOF
ExpDir=$expdir
logFile=train_cntk
modelName=cntk.dnn

labelDim=${labelDim}
featDim=${featDim}
labelMapping=${expdir}/cntk_label.mapping
featureTransform=NO_FEATURE_TRANSFORM

inputCounts=${expdir}/cntk_train.counts
inputFeats=${expdir}/cntk_train.feats
inputLabels=${expdir}/cntk_train.labels

cvInputCounts=${expdir}/cntk_valid.counts
cvInputFeats=${expdir}/cntk_valid.feats
cvInputLabels=${expdir}/cntk_valid.labels
EOF

## training command ##
$cn_gpu configFile=${expdir}/Base.config configFile=${expdir}/CNTK2.cntk DeviceNumber=0 action=TrainDNN ndlfile=$ndlfile

echo "$0 successfuly finished.. $dir"

fi


if [ $stage -le 2 ] ; then

config_write=cntk_config/CNTK2_write.cntk
cnmodel=$expdir/cntk.dnn.16
action=write
graphdir=/exports/work/inf_hcrc_cstr_nst/llu/ami/s5b/exp/$mic/tri4a/graph_ami_fsh.o3g.kn.pr1-7
cp $alidir/final.mdl $expdir

cntk_string="$cn_gpu configFile=$config_write DeviceNumber=0 modelName=$cnmodel labelDim=$labelDim featDim=$featDim action=$action ExpDir=$expdir"
scripts/decode_cntk2.sh  --nj 1 --cmd run.pl  --acwt 0.0833 $graphdir $datadir/eval $expdir/decode_ami_fsh.o3g.kn.pr1-7 "$cntk_string" || exit 1;


fi

sleep 3
exit 0

