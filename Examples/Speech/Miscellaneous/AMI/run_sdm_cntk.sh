#!/bin/bash

# Make sure you run_prepare_shared.sh and run_sdm.sh till tri3a_ali stage, we
# will start from there.

. ./cmd.sh
. ./path.sh

mic=sdm1

# ------------------------------------------------------------------------------
# Features for DNN training: 40 dimensional fbank + delta + delta-delta, with
#                            cepstral mean and variance normalization.
# ------------------------------------------------------------------------------
for x in train dev eval; do
  utils/copy_data_dir.sh data/$mic/$x data/$mic/${x}_120fbank || exit 1;
  rm data/$mic/${x}_120fbank/{cmvn,feats,raw_feats}.scp
  steps/make_fbank.sh --fbank-config cntk_config/40fbank.conf \
    --nj 10 --cmd "$train_cmd" data/$mic/${x}_120fbank \
    data/$mic/${x}_120fbank/log data/$mic/${x}_120fbank/data || exit 1;
  steps/compute_cmvn_stats.sh data/$mic/${x}_120fbank \
    data/$mic/${x}_120fbank/log data/$mic/${x}_120fbank/data || exit 1;

  mv data/$mic/${x}_120fbank/feats.scp \
    data/$mic/${x}_120fbank/raw_feats.scp
  copy-feats scp:data/$mic/${x}_120fbank/raw_feats.scp ark:- |\
    apply-cmvn --norm-means=true --norm-vars=true \
    --utt2spk=ark:data/$mic/${x}_120fbank/utt2spk \
    scp:data/$mic/${x}_120fbank/cmvn.scp ark:- ark:- |\
    add-deltas --delta-order=2 ark:- ark:- |\
    copy-feats --compress=true ark:- \
    ark,scp:data/$mic/${x}_120fbank/data/delta_${x}_120fbank.ark,data/$mic/${x}_120fbank/feats.scp
done
utils/subset_data_dir_tr_cv.sh data/$mic/train_120fbank \
  data/$mic/train_120fbank_tr90 data/$mic/train_120fbank_cv10 || exit 1;

# ------------------------------------------------------------------------------
# Features for LSTM training: 80 dimensional fbank, without cepstral mean and
#                             variance normalization.
# ------------------------------------------------------------------------------
for x in train dev eval; do
  utils/copy_data_dir.sh data/$mic/$x data/$mic/${x}_80fbank || exit 1;
  rm data/$mic/${x}_80fbank/{cmvn,feats}.scp
  steps/make_fbank.sh --fbank-config cntk_config/80fbank.conf \
    --nj 10 --cmd "$train_cmd" data/$mic/${x}_80fbank \
    data/$mic/${x}_80fbank/log data/$mic/${x}_80fbank/data || exit 1;
  steps/compute_cmvn_stats.sh data/$mic/${x}_80fbank \
    data/$mic/${x}_80fbank/log data/$mic/${x}_80fbank/data || exit 1;
done
utils/subset_data_dir_tr_cv.sh data/$mic/train_80fbank \
  data/$mic/train_80fbank_tr90 data/$mic/train_80fbank_cv10 || exit 1;

# ------------------------------------------------------------------------------
# DNN CE training and alignment.
# Make sure you point exp_cntk/sdm1/dnn/cntk.mdl to your CNTK model, sometimes
# it fails to generate automatically.
# ------------------------------------------------------------------------------
feat_dim=1320
cntk_train_opts=""
scripts/train_nnet.sh --num-threads 1 --device 0 --cmd "$cuda_cmd" \
  --feat-dim $feat_dim --cntk-train-opts "$cntk_train_opts" \
  --learning-rate "0.1:1" --momentum "0:0.9" \
  --max-epochs 50 --minibatch-size 256 --evaluate-period 100 \
  --cntk-config cntk_config/CNTK2_dnn.cntk \
  --default-macros cntk_config/default_macros.ndl \
  --model-ndl cntk_config/dnn_6layer.ndl \
  data/sdm1/train_120fbank exp/sdm1/tri3a_ali exp_cntk/sdm1/dnn

# Alignment.
device=-1
scripts/align.sh --num-threads 1 --nj 60 --cmd "$decode_cmd" \
  --feat-dim $feat_dim --device $device \
  --cntk-config cntk_config/Align.cntk \
  data/sdm1/train_120fbank data/lang \
  exp_cntk/sdm1/dnn exp_cntk/sdm1/dnn_ali

# ------------------------------------------------------------------------------
# Highway LSTM CE training.
# Make sure you point exp_cntk/sdm1/hlstmp/cntk.mdl to your CNTK model,
# sometimes it fails to generate automatically.
# ------------------------------------------------------------------------------
feat_dim=880
base_feat_dim=80
row_slice_start=800
num_utts_per_iter=40
cntk_train_opts=""
cntk_train_opts="baseFeatDim=$base_feat_dim"
cntk_train_opts="$cntk_train_opts RowSliceStart=$row_slice_start"
cntk_train_opts="$cntk_train_opts numUttsPerMinibatch=$num_utts_per_iter "
scripts/train_nnet.sh --num-threads 1 --device 0 --cmd "$cuda_cmd" \
  --feat-dim $feat_dim --cntk-train-opts "$cntk_train_opts" \
  --learning-rate "0.2:1" --momentum "0:0.9" \
  --max-epochs 50 --minibatch-size 20 --evaluate-period 100 \
  --clipping-per-sample 0.05 --l2-reg-weight 0.00001 --dropout-rate "0.1*5:0.8"\
  --cntk-config cntk_config/CNTK2_lstmp.cntk \
  --default-macros cntk_config/default_macros.ndl \
  --model-ndl cntk_config/lstmp-3layer-highway-dropout.ndl \
  data/sdm1/train_80fbank exp_cntk/sdm1/dnn_ali exp_cntk/sdm1/hlstmp

# ------------------------------------------------------------------------------
# Highway LSTM sMBR training.
# Note: you could try a larger learning rate, for example 0.000005.
# ------------------------------------------------------------------------------
data=data/sdm1/train_80fbank/
srcdir=exp_cntk/sdm1/hlstmp/
srcdir=${srcdir%/}

# Alignment.
device=-1
alidir=${srcdir}_ali
scripts/align.sh --num-threads 1 --nj 60 --cmd "$decode_cmd" \
  --feat-dim $feat_dim --device $device \
  --cntk-config cntk_config/Align.cntk \
  $data data/lang $srcdir $alidir || exit 1;

# Denominator lattices.
device=-1
denlatdir=${srcdir}_denlats
scripts/make_denlats.sh --num-threads 1 --nj 20 --sub-split 60 \
  --feat-dim $feat_dim --cmd "$decode_cmd" --acwt $acwt \
  --device $device --cntk-config cntk_config/Align.cntk \
  --ngram-order 2 \
  $data data/lang $srcdir $denlatdir || exit 1;

# Sequence training.
device=0
smbrdir=${srcdir}_smbr
cntk_train_opts="baseFeatDim=$base_feat_dim"
cntk_train_opts="$cntk_train_opts RowSliceStart=$row_slice_start"
cntk_train_opts="$cntk_train_opts numUttsPerMinibatch=$num_utts_per_iter "
scripts/train_nnet_sequence.sh --num-threads 1 --cmd "$cuda_cmd" --momentum 0.9\
  --learning-rate "0.000002*6" --num-iters 6 --feat-dim $feat_dim \
  --acwt 0.1 --evaluate-period 100 --truncated true --device $device \
  --minibatch-size 20 --cntk-train-opts "$cntk_train_opts" \
  --clipping-per-sample 0.05 --smooth-factor 0.1 \
  --l2-reg-weight 0.00001 --one-silence-class false --dropout-rate 0 \
  --cntk-config cntk_config/CNTK2_lstmp_smbr.cntk \
  --model-mel cntk_config/lstmp-smbr.mel \
  --model-ndl cntk_config/lstmp-3layer-highway-dropout.ndl \
  --default-macros cntk_config/default_macros.ndl \
  $data data/lang $srcdir $alidir $denlatdir $smbrdir || exit 1;
