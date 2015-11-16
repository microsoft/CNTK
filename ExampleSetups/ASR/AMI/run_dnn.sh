#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
#                2014  Guoguo Chen
#		 2015  Liang Lu
# Apache 2.0

# This example script trains a DNN on top of fMLLR features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Config:
mic=sdm1
gmmdir=exp/$mic/tri4a
data_fmllr=data-fmllr-tri4a
stage=0 # resume training with --stage=N
has_fisher=false

EVAL_SPK=$((`cut -d" " -f2 data/$mic/eval/utt2spk | sort | uniq -c | wc -l`))

# End of config.
. utils/parse_options.sh || exit 1;
#

if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # eval
  dir=$data_fmllr/$mic/eval
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_eval_ami_fsh.o3g.kn.pr1-7 \
     $dir data/$mic/eval $gmmdir $dir/log $dir/data || exit 1
  # train
  dir=$data_fmllr/$mic/train
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir data/$mic/train $gmmdir $dir/log $dir/data || exit 1
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs
  dir=exp/$mic/dnn5b_pretrain-dbn
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 3 $data_fmllr/$mic/train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/$mic/dnn5b_pretrain-dbn_dnn
  ali=${gmmdir}_ali
#  ali=exp/$mic/dnn5b_pretrain-dbn_dnn_ali
  feature_transform=exp/$mic/dnn5b_pretrain-dbn/final.feature_transform
  dbn=exp/$mic/dnn5b_pretrain-dbn/6.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/$mic/train_tr90 $data_fmllr/$mic/train_cv10 data/lang $ali $ali $dir || exit 1;

#   Decode with the trigram swbd language model.
  steps/nnet/decode.sh --nj $EVAL_SPK --cmd "$decode_cmd" \
    --config conf/decode_dnn.config --acwt 0.08333 \
    $gmmdir/graph_ami_fsh.o3g.kn.pr1-7 $data_fmllr/$mic/eval \
    $dir/decode_eval_ami_fsh.o3g.kn.pr1-7 || exit 1;
  if $has_fisher; then
    # Rescore with the 4gram swbd+fisher language model.
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_sw1_{tg,fsh_fg} data/eval2000 \
      $dir/decode_eval2000_sw1_{tg,fsh_fg}
  fi
fi

if [ $stage -le 3 ]; then
  # re-align and then retrain the nnet:
 srcdir=exp/$mic/dnn5b_pretrain-dbn_dnn
  steps/nnet/align.sh --nj 50 --cmd "$train_cmd" \
    $data_fmllr/$mic/train data/lang $srcdir ${srcdir}_ali || exit 1;
  cp $srcdir/final.mdl ${srcdir}_ali
fi

#Liang: have not tried sequence training
exit 0;


# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
dir=exp/$mic/dnn5b_pretrain-dbn_dnn_smbr
srcdir=exp/$mic/dnn5b_pretrain-dbn_dnn
acwt=0.0909

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 50 --cmd "$train_cmd" \
    $data_fmllr/$mic/train data/lang $srcdir ${srcdir}_ali || exit 1;
#  steps/nnet/make_denlats.sh --nj 10 --sub-split 100 --cmd "$decode_cmd" --config conf/decode_dnn.config \
#    --acwt $acwt $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

exit;

if [ $stage -le 4 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 1; do
    # Decode with the trigram swbd language model.
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
      --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_sw1_tg $data_fmllr/eval2000 \
      $dir/decode_eval2000_sw1_tg || exit 1;
    if $has_fisher; then
      # Rescore with the 4gram swbd+fisher language model.
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_sw1_{tg,fsh_fg} data/eval2000 \
        $dir/decode_eval2000_sw1_{tg,fsh_fg}
    fi
  done 
fi

# Re-generate lattices, run 4 more sMBR iterations
dir=exp/dnn5b_pretrain-dbn_dnn_smbr_i1lats
srcdir=exp/dnn5b_pretrain-dbn_dnn_smbr
acwt=0.0909

if [ $stage -le 5 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 250 --cmd "$train_cmd" \
    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 10 --sub-split 100 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --acwt $acwt $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 6 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 2 --acwt $acwt --do-smbr true \
    $data_fmllr/train_nodup data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 1 2; do
    # Decode with the trigram swbd language model.
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
      --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_sw1_tg $data_fmllr/eval2000 \
      $dir/decode_eval2000_sw1_tg || exit 1;
    if $has_fisher; then
      # Rescore with the 4gram swbd+fisher language model.
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_sw1_{tg,fsh_fg} data/eval2000 \
        $dir/decode_eval2000_sw1_{tg,fsh_fg}
    fi
  done 
fi

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
