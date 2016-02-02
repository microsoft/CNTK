#!/bin/bash

# WARNING: make sure your $srcdir has final.mdl which is a Kaldi model, and
#          cntk.mdl which is the CNTK model you want to use. You also need to
#          copy tree from the corresponding Kaldi directory.

. ./cmd.sh
. ./path.sh

# Config:
mic=ihm
data_fmllr=data-fmllr-tri4a/$mic/
srcdir=exp_cntk/$mic/dnn
acwt=0.1
feat_dim=600

# Removes possible trailing slash
srcdir=${srcdir%/}

# Alignment.
device=-1
alidir=${srcdir}_ali
mkdir -p $alidir/configs
cp -f cntk_config/Align.cntk $alidir/configs/Align.cntk
scripts/align.sh --num-threads 1 --nj 30 --cmd "$decode_cmd" \
  --feat-dim $feat_dim --device $device \
  --cntk-config $alidir/configs/Align.cntk \
  $data_fmllr/train data/lang $srcdir $alidir || exit 1;

# Denominator lattices.
device=-1
denlatdir=${srcdir}_denlats
mkdir -p $denlatdir/configs
cp -f cntk_config/Align.cntk $denlatdir/configs/Decode.config
scripts/make_denlats.sh --num-threads 1 --nj 20 --sub-split 15 \
  --feat-dim $feat_dim --cmd "$decode_cmd" --acwt $acwt \
  --device $device --cntk-config $denlatdir/configs/Decode.config \
  $data_fmllr/train data/lang $srcdir $denlatdir || exit 1;

# Sequence training.
# CAVEAT: I'm always setting device to 0 and this is OK on the Amazon cluster,
#         because each GPU machine only has one GPU card. Otherwise it can be
#         very tricky to use CNTK on SGE.
device=0
smbrdir=${srcdir}_smbr
mkdir -p $smbrdir/configs
cp -f cntk_config/CNTK2_smbr.cntk $smbrdir/configs/Train.config
cp -f cntk_config/dnn_6layer_smbr.ndl $smbrdir/configs/model.ndl
cp -f cntk_config/default_macros.ndl $smbrdir/configs/default_macros.ndl
scripts/train_sequence.sh --num-threads 1 --cmd "$cuda_cmd" \
  --learning-rate "0.00001" --num-iters 4 --feat-dim $feat_dim \
  --acwt $acwt --evaluate-period 100 --truncated false \
  --device $device --cntk-config $smbrdir/configs/Train.config \
  $data_fmllr/train data/lang $srcdir $alidir $denlatdir $smbrdir || exit 1;

exit 0

