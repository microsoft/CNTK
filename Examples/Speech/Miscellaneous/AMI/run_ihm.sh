#!/bin/bash -u

. ./cmd.sh
. ./path.sh

#INITIAL COMMENTS
#To run the whole recipe you're gonna to need
# 1) SRILM 
# 2) 

#1) some settings
#do not change this, it's for ctr-c ctr-v of training commands between ihm, sdm and mdm
mic=ihm
#path where AMI whould be downloaded or where is locally available
AMI_DIR=`pwd`/amicorpus/
# path to Fisher transcripts for background language model 
# when not set only in-domain LM will be build
FISHER_TRANS=/exports/work/inf_hcrc_cstr_general/corpora/fisher/transcripts/

norm_vars=false

#1)

#in case you want download AMI corpus, uncomment this line
#you need arount 130GB of free space to get whole data ihm+mdm
local/ami_download.sh ihm $AMI_DIR || exit 1;

#2) Data preparation

local/ami_text_prep.sh $AMI_DIR

local/ami_ihm_data_prep.sh $AMI_DIR || exit 1;

local/ami_ihm_scoring_data_prep.sh $AMI_DIR dev || exit 1;

local/ami_ihm_scoring_data_prep.sh $AMI_DIR eval || exit 1;

local/ami_prepare_dict.sh

utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

local/ami_train_lms.sh --fisher $FISHER_TRANS data/ihm/train/text data/ihm/dev/text data/local/dict/lexicon.txt data/local/lm

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
nj=16

prune-lm --threshold=1e-7 data/local/lm/$final_lm.gz /dev/stdout | \
   gzip -c > data/local/lm/$LM.gz

utils/format_lm.sh data/lang data/local/lm/$LM.gz data/local/dict/lexicon.txt data/lang_$LM

#local/ami_format_data.sh data/local/lm/$LM.gz

# 3) Building systems
# here starts the normal recipe, which is mostly shared across mic scenarios
# one difference is for sdm and mdm we do not adapt for speaker byt for environment only

mfccdir=mfcc_$mic
(
 steps/make_mfcc.sh --nj 5  --cmd "$train_cmd" data/$mic/eval exp/$mic/make_mfcc/eval $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$mic/eval exp/$mic/make_mfcc/eval $mfccdir || exit 1
)&
(
 steps/make_mfcc.sh --nj 5 --cmd "$train_cmd" data/$mic/dev exp/$mic/make_mfcc/dev $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$mic/dev exp/$mic/make_mfcc/dev $mfccdir || exit 1
)&
(
 steps/make_mfcc.sh --nj 16 --cmd "$train_cmd" data/$mic/train exp/$mic/make_mfcc/train $mfccdir || exit 1;
 steps/compute_cmvn_stats.sh data/$mic/train exp/$mic/make_mfcc/train $mfccdir || exit 1
)&

wait;

for dset in train eval dev; do utils/fix_data_dir.sh data/$mic/$dset; done

# 4) Train systems
 nj=16

 mkdir -p exp/$mic/mono
 steps/train_mono.sh --nj $nj --cmd "$train_cmd" --feat-dim 39 --norm-vars $norm_vars \
   data/$mic/train data/lang exp/$mic/mono >& exp/$mic/mono/train_mono.log || exit 1;

 mkdir -p exp/$mic/mono_ali
 steps/align_si.sh --nj $nj --cmd "$train_cmd" data/$mic/train data/lang exp/$mic/mono \
   exp/$mic/mono_ali >& exp/$mic/mono_ali/align.log || exit 1;

 mkdir -p exp/$mic/tri1
 steps/train_deltas.sh --cmd "$train_cmd" --norm-vars $norm_vars \
   5000 80000 data/$mic/train data/lang exp/$mic/mono_ali exp/$mic/tri1 \
   >& exp/$mic/tri1/train.log || exit 1;

 mkdir -p exp/$mic/tri1_ali
 steps/align_si.sh --nj $nj --cmd "$train_cmd" \
   data/$mic/train data/lang exp/$mic/tri1 exp/$mic/tri1_ali || exit 1;

 mkdir -p exp/$mic/tri2a
 steps/train_deltas.sh --cmd "$train_cmd" --norm-vars $norm_vars \
  5000 80000 data/$mic/train data/lang exp/$mic/tri1_ali exp/$mic/tri2a \
  >& exp/$mic/tri2a/train.log || exit 1;

 for lm_suffix in $LM; do
#  (
    graph_dir=exp/$mic/tri2a/graph_${lm_suffix}
    mkdir -p $graph_dir/mkgraph.log
    $highmem_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_${lm_suffix} exp/$mic/tri2a $graph_dir

    steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir data/$mic/dev exp/$mic/tri2a/decode_dev_${lm_suffix} 
   
    steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir data/$mic/eval exp/$mic/tri2a/decode_eval_${lm_suffix} 

#  ) &
 done

mkdir -p exp/$mic/tri2a_ali
steps/align_si.sh --nj $nj --cmd "$train_cmd" \
  data/$mic/train data/lang exp/$mic/tri2a exp/$mic/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT
mkdir -p exp/$mic/tri3a
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" \
  5000 80000 data/$mic/train data/lang exp/$mic/tri2_ali exp/$mic/tri3a \
  >& exp/$mic/tri3a/train.log || exit 1;

for lm_suffix in $LM; do
  (
    graph_dir=exp/$mic/tri3a/graph_${lm_suffix}
    mkdir -p $graph_dir/mkgraph.log
    $highmem_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_${lm_suffix} exp/$mic/tri3a $graph_dir

    steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir data/$mic/dev exp/$mic/tri3a/decode_dev_${lm_suffix} 

    steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir data/$mic/eval exp/$mic/tri3a/decode_eval_${lm_suffix} 
  ) 
done

# Train tri4a, which is LDA+MLLT+SAT
steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
  data/$mic/train data/lang exp/$mic/tri3a exp/$mic/tri3a_ali || exit 1;

mkdir -p exp/$mic/tri4a
steps/train_sat.sh  --cmd "$train_cmd" \
  5000 80000 data/$mic/train data/lang exp/$mic/tri3a_ali \
  exp/$mic/tri4a >& exp/$mic/tri4a/train.log || exit 1;

for lm_suffix in $LM; do
  (
    graph_dir=exp/$mic/tri4a/graph_${lm_suffix}
    mkdir -p $graph_dir/mkgraph.log
    $highmem_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_${lm_suffix} exp/$mic/tri4a $graph_dir

    steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd"  --config conf/decode.conf \
      $graph_dir data/$mic/dev exp/$mic/tri4a/decode_dev_${lm_suffix} 

    steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir data/$mic/eval exp/$mic/tri4a/decode_eval_${lm_suffix} 
  ) 
done

# MMI training starting from the LDA+MLLT+SAT systems
steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
  data/$mic/train data/lang exp/$mic/tri4a exp/$mic/tri4a_ali || exit 1
exit 0;

steps/make_denlats.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
  --transform-dir exp/$mic/tri4a_ali \
  data/$mic/train data/lang exp/$mic/tri4a exp/$mic/tri4a_denlats  || exit 1;

# 4 iterations of MMI seems to work well overall. The number of iterations is
# used as an explicit argument even though train_mmi.sh will use 4 iterations by
# default.
num_mmi_iters=4
steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 --num-iters $num_mmi_iters \
  data/$mic/train data/lang exp/$mic/tri4a_ali exp/$mic/tri4a_denlats \
  exp/$mic/tri4a_mmi_b0.1 || exit 1;

for lm_suffix in $LM; do
  (
    graph_dir=exp/$mic/tri4a/graph_${lm_suffix}
    
    for i in `seq 1 4`; do
         decode_dir=exp/$mic/tri4a_mmi_b0.1/decode_dev_${i}.mdl_${lm_suffix}
      steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
        --transform-dir exp/$mic/tri4a/decode_dev_${lm_suffix} --iter $i \
        $graph_dir data/$mic/dev $decode_dir 
    done
    
    i=3 #simply assummed
    decode_dir=exp/$mic/tri4a_mmi_b0.1/decode_eval_${i}.mdl_${lm_suffix}
    steps/decode.sh --nj $nj --cmd "$decode_cmd"  --config conf/decode.conf \
      --transform-dir exp/$mic/tri4a/decode_eval_${lm_suffix} --iter $i \
      $graph_dir data/$mic/eval $decode_dir 
  )
done

# here goes hybrid stuf
# in the ASRU paper we used different python nnet code, so someone needs to copy&adjust nnet or nnet2 switchboard commands



