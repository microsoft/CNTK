#!/bin/bash -u

. ./cmd.sh
. ./path.sh

#SDM - Signle Distant Microphone 
#Assuming initial transcrips, dict, lang and LM were build in run_ihm.sh

micid=1 #which mic from array should be used?
mic=sdm$micid
#AMI_DIR=/disk/data2/amicorpus/
AMI_DIR=`pwd`/amicorpus/

norm_vars=false

#1) Download AMI (single distant channel)

#local/ami_download.sh sdm $AMI_DIR

#2) Prepare sdm data directories

#local/ami_sdm_data_prep.sh $AMI_DIR $micid
#local/ami_sdm_scoring_data_prep.sh $AMI_DIR $micid dev
#local/ami_sdm_scoring_data_prep.sh $AMI_DIR $micid eval

#use the final LM
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

#jobs for SDM/MDM decodes - one per meeting on 16core local machine
DEV_SPK=$((`cut -d" " -f2 data/$mic/dev/utt2spk | sort | uniq -c | wc -l`))
EVAL_SPK=$((`cut -d" " -f2 data/$mic/eval/utt2spk | sort | uniq -c | wc -l`))
echo $DEV_SPK $EVAL_SPK
nj=16

#GENERATE FEATS
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

# TRAIN THE MODELS
 mkdir -p exp/$mic/mono
 steps/train_mono.sh --nj $nj --cmd "$train_cmd" --feat-dim 39 \
   data/$mic/train data/lang exp/$mic/mono >& exp/$mic/mono/train_mono.log || exit 1;

 mkdir -p exp/$mic/mono_ali
 steps/align_si.sh --nj $nj --cmd "$train_cmd" data/$mic/train data/lang exp/$mic/mono \
   exp/$mic/mono_ali >& exp/$mic/mono_ali/align.log || exit 1;

 mkdir -p exp/$mic/tri1
 steps/train_deltas.sh --cmd "$train_cmd" \
   5000 80000 data/$mic/train data/lang exp/$mic/mono_ali exp/$mic/tri1 \
   >& exp/$mic/tri1/train.log || exit 1;

 mkdir -p exp/$mic/tri1_ali
 steps/align_si.sh --nj $nj --cmd "$train_cmd" \
   data/$mic/train data/lang exp/$mic/tri1 exp/$mic/tri1_ali || exit 1;

 mkdir -p exp/$mic/tri2a
 steps/train_deltas.sh --cmd "$train_cmd" \
  5000 80000 data/$mic/train data/lang exp/$mic/tri1_ali exp/$mic/tri2a \
  >& exp/$mic/tri2a/train.log || exit 1;

 for lm_suffix in $LM; do
  (
    graph_dir=exp/$mic/tri2a/graph_${lm_suffix}
    $highmem_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_${lm_suffix} exp/$mic/tri2a $graph_dir
   
    steps/decode.sh --nj $DEV_SPK --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir data/$mic/dev exp/$mic/tri2a/decode_dev_${lm_suffix} 
   
    steps/decode.sh --nj $EVAL_SPK --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir data/$mic/eval exp/$mic/tri2a/decode_eval_${lm_suffix} 
  ) 
 done

#THE TARGET LDA+MLLT+SAT+BMMI PART GOES HERE:
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
    $highmem_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_${lm_suffix} exp/$mic/tri3a $graph_dir

    steps/decode.sh --nj $DEV_SPK --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir data/$mic/dev exp/$mic/tri3a/decode_dev_${lm_suffix} 

    steps/decode.sh --nj $EVAL_SPK --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir data/$mic/eval exp/$mic/tri3a/decode_eval_${lm_suffix} 
  ) 
done

# skip SAT, and build MMI models
steps/make_denlats.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config \
    data/$mic/train data/lang exp/$mic/tri3a exp/$mic/tri3a_denlats  || exit 1;


mkdir -p exp/$mic/tri3a_ali
steps/align_si.sh --nj $nj --cmd "$train_cmd" \
  data/$mic/train data/lang exp/$mic/tri3a exp/$mic/tri3a_ali || exit 1;

# 4 iterations of MMI seems to work well overall. The number of iterations is
# used as an explicit argument even though train_mmi.sh will use 4 iterations by
# default.
num_mmi_iters=4
steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 --num-iters $num_mmi_iters \
  data/$mic/train data/lang exp/$mic/tri3a_ali exp/$mic/tri3a_denlats \
  exp/$mic/tri3a_mmi_b0.1 || exit 1;

for lm_suffix in $LM; do
  (
    graph_dir=exp/$mic/tri3a/graph_${lm_suffix}

    for i in `seq 1 4`; do
      decode_dir=exp/$mic/tri3a_mmi_b0.1/decode_dev_${i}.mdl_${lm_suffix}
      steps/decode.sh --nj $DEV_SPK --cmd "$decode_cmd" --iter $i --config conf/decode.conf \
        $graph_dir data/$mic/dev $decode_dir
    done

    i=3 #simply assummed
    decode_dir=exp/$mic/tri3a_mmi_b0.1/decode_eval_${i}.mdl_${lm_suffix}
    steps/decode.sh --nj $EVAL_SPK --cmd "$decode_cmd" --iter $i --config conf/decode.conf \
      $graph_dir data/$mic/eval $decode_dir
  )
done

#By default we do no build systems adapted to sessions for AMI in distant scnearios as this does not help a lot (around 1%)
#But one can do this by running below code
exit;

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
    $highmem_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_${lm_suffix} exp/$mic/tri4a $graph_dir

    steps/decode_fmllr.sh --nj $DEV_SPK --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir data/$mic/dev exp/$mic/tri4a/decode_dev_${lm_suffix} 

    steps/decode_fmllr.sh --nj $EVAL_SPK --cmd "$decode_cmd" --config conf/decode.conf \
      $graph_dir data/$mic/eval exp/$mic/tri4a/decode_eval_${lm_suffix} 
  ) 
done

# MMI training starting from the LDA+MLLT+SAT systems
steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
  data/$mic/train data/lang exp/$mic/tri4a exp/$mic/tri4a_ali || exit 1

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
      steps/decode.sh --nj $DEV_SPK --cmd "$decode_cmd" --config conf/decode.conf \
        --transform-dir exp/$mic/tri4a/decode_dev_${lm_suffix} \
        $graph_dir data/$mic/dev $decode_dir 
    done
    
    wait;
    i=3 #simply assummed
    decode_dir=exp/$mic/tri4a_mmi_b0.1/decode_eval_${i}.mdl_${lm_suffix}
    steps/decode.sh --nj $EVAL_SPK --cmd "$decode_cmd" --config conf/decode.conf \
      --transform-dir exp/$mic/tri4a/decode_eval_${lm_suffix} \
      $graph_dir data/$mic/eval $decode_dir 
  )&
done

# here goes hybrid stuf
# in the ASRU paper we used different python nnet code, so someone needs to copy&adjust nnet or nnet2 switchboard commands

