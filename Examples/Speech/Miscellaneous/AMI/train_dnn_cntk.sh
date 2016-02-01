#!/bin/bash

# This example script trains a XNN on top of fMLLR features using CNTK. 
# The training is done in 3 stages,
#
# 1) DNN training
# 2) LSTM training
# 3) PAC-RNN training

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
#           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

. ./path_cntk.sh

# Config:
mic=sdm1
alidir=/exports/work/inf_hcrc_cstr_nst/llu/ami/s5b/exp/$mic/dnn5b_pretrain-dbn_dnn_ali
data_fmllr=/exports/work/inf_hcrc_cstr_nst/llu/ami/s5b/data-fmllr-tri4a/$mic
dir=/exports/work/inf_hcrc_cstr_nst/llu/ami/s5b/exp_cntk/$mic/dnn
labelDim=3951  #number of output units, you have to check your gmm setup to get this number
stage=6
featdir=`dirname $dir`;

cn_gpu="/exports/work/inf_hcrc_cstr_nst/llu/cntk_kaldi/bin/cn.exe"

# End of config.
. utils/parse_options.sh || exit 1;
#

nj=`cat $alidir/num_jobs` || exit 1;
mkdir -p $dir/log
echo $nj > $dir/num_jobs

if [ $stage -le 0 ]; then
echo "$0 convert the features to the HTK format"
mkdir -p $featdir/feat_train
feats="scp:$data_fmllr/train/split$nj/JOB/feats.scp"
$train_cmd JOB=1:$nj $featdir/log/train_feats.JOB.log \
    copy-feats "$feats" ark:- \| \
    copy-feats-to-htk --output-dir=$featdir/feat_train ark:- || exit 1;

#### Make sure you have splitted the eval feature list 
mkdir -p $featdir/feat_eval
feats="scp:$data_fmllr/eval/split16/JOB/feats.scp"
$train_cmd JOB=1:16 $dir/log/eval_feats.JOB.log \
    copy-feats "$feats" ark:- \| \
    copy-feats-to-htk --output-dir=$featdir/feat_eval ark:- || exit 1;
fi

if [ $stage -le 1 ]; then
echo "$0 extracting the alignment from $alidir "
    $train_cmd JOB=1:$nj $dir/log/ali2pdf.JOB.log \
    ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/ali.JOB.gz|" ark,t:$dir/pdf.JOB || exit 1;

    cat $dir/pdf.* > $dir/train_ali.all
    perl ./scripts/convert_kaldi_label.pl $dir $featdir/feat_train $dir "" || exit 1;
    
    cp $alidir/final.mdl $dir
    rm $dir/pdf.*
fi

if [ $stage -le 2 ]; then
echo "$0 creating the training and evaluation list"
    mkdir -p $dir/List
    find $featdir/feat_train/ | grep 'label$' > $dir/List/train.label.list
    sed -e 's/label$/fea/g' $dir/List/train.label.list > $dir/List/train.list

    ls $featdir/feat_eval/*.fea > $dir/List/eval.list
fi

if [ $stage -le 3 ]; then
echo "$0 create labels for cntk dnn training"
    mkdir -p $dir/lib
    python scripts/Convert_label_to_cntk.py -in $dir/List/train.list $dir/List/train.label.list 0 > $dir/lib/tmp.mlf
    sed -e "s,$dir\/dct\/,,g" $dir/lib/tmp.mlf >  $dir/lib/tmp2.mlf
    sed -e "s,.dct.bottle.htk,,g" $dir/lib/tmp2.mlf >  $dir/lib/trans_align_train.mlf
fi

if [ $stage -le 4 ]; then
echo "$0 create the mlf and features for cntk dnn training"
#    python scripts/Convert_label_to_cntk.py -in $dir/File_List/train.list $dir/File_List/train.label.list 5 > $dir/lib/trans_align_train_delay5.mlf
    python scripts/convert_scp_to_cntk.py -in $dir/List/train.list $dir/List/train.label.list > $dir/lib/train.scp.all
    perl scripts/rand_select_data.pl $dir/List/train.list $dir/List/train.label.list

    python scripts/select_by_id.py -in $dir/lib/train.scp.all $dir/List/train.list.train > $dir/lib/train.scp
    python scripts/select_by_id.py -in $dir/lib/train.scp.all $dir/List/train.list.test > $dir/lib/cv.scp

fi

if [ $stage -le 5 ]; then
echo "$0 create cntk files"
    mkdir -p $dir/ndl
    logFile=train_dnn
    labelMapping=$dir/lib/label.mapping
    cp cntk_config/default_macros.ndl $dir/ndl
    cp cntk_config/dnn_6layer.ndl $dir/ndl
 
    ndlfile=$dir/ndl/dnn_6layer.ndl
    inputFeat=$dir/lib/train.scp
    trainMLF=$dir/lib/trans_align_train.mlf
    cvInputFeat=$dir/lib/cv.scp
    for i in `seq 1 $labelDim` ; do echo $i ; done > $labelMapping

   # You can either submit your jobs to your cluser, or run it in your local GPU machine
   # $cuda_cmd $dir/log/train_cntk.log \
  $cn_gpu configFile=/exports/work/inf_hcrc_cstr_nst/llu/ami/s5b/cntk_config/CNTK.cntk modelName=cntkSpeech.dnn DeviceNumber=0 ExpDir=$dir logFile=$logFile labelMapping=$labelMapping ndlfile=$ndlfile inputFeat=$inputFeat trainMLF=$trainMLF labelDim=$labelDim cvInputFeat=$cvInputFeat featDim=600 action=TrainDNN phnLabel=no phnDim=no phnMapping=no inputSCP=no outputSCP=no
fi

if [ $stage -le 6 ] ; then
    echo "$0 create test feature list"
    perl scripts/convert_to_kaldi_list.pl noMap $dir/List/eval.list $data_fmllr/eval/feats.scp $data_fmllr/eval
    #utils/fix_data_dir.sh $data_fmllr/eval
fi

if [ $stage -le 7 ]; then
    action=write
    featDim=600
    cnmodel=$dir/cntkSpeech.dnn.17
    graphdir=/exports/work/inf_hcrc_cstr_nst/llu/ami/s5b/exp/$mic/tri4a/graph_ami_fsh.o3g.kn.pr1-7
    class_frame_counts=/exports/work/inf_hcrc_cstr_nst/llu/ami/s5b/exp/ihm/dnn5b_pretrain-dbn_dnn_realign/ali_train_pdf.counts
    cntk_string="$cn_gpu configFile=cntk_config/CNTK_write.cntk DeviceNumber=1 modelName=$cnmodel labelDim=$labelDim featDim=$featDim action=$action"
    scripts/decode_cntk.sh  --nj 16 --cmd "$decode_cmd" --acwt 0.0833 --class-frame-counts "$class_frame_counts" \
      $graphdir $data_fmllr/eval $dir/decode_ami_fsh.o3g.kn.pr1-7 "$cntk_string" || exit 1;
fi

exit 0



