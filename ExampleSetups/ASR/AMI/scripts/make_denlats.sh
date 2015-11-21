#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).
#           2015  Guoguo Chen
# Apache 2.0.

# Generates denominator lattices for MMI/MPE/sMBR training.

# Begin configuration section.
nj=4
cmd=run.pl
sub_split=1
beam=13.0
lattice_beam=7.0
acwt=0.1
max_active=5000
max_mem=20000000
cntk_forward_opts=
cntk_config=
device=-1
parallel_opts=
num_threads=1
feature_transform=NO_FEATURE_TRANSFORM
feat_dim=
ngram_order=1
srilm_options="-wbdiscount"   # By default, use Witten-Bell discounting in SRILM
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: steps/$0 [options] <data-dir> <lang-dir> <src-dir> <exp-dir>"
   echo "  e.g.: steps/$0 data/train data/lang exp/tri1 exp/tri1_denlats"
   echo "Works for plain features (or CMN, delta), forwarded through feature-transform."
   echo ""
   echo "Main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --sub-split <n-split>                            # e.g. 40; use this for "
   echo "                           # large databases so your jobs will be smaller and"
   echo "                           # will (individually) finish reasonably soon."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$(readlink -f $4)

sdata=$data/split$nj
mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

mkdir -p $dir

# If --ngram-order is larger than 1, we will have to use SRILM
if [ $ngram_order -gt 1 ]; then
  ngram_count=`which ngram-count`;
  if [ -z $ngram_count ]; then
    if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
      sdir=`pwd`/../../../tools/srilm/bin/i686-m64
    else
      sdir=`pwd`/../../../tools/srilm/bin/i686
    fi
    if [ -f $sdir/ngram-count ]; then
      echo Using SRILM tools from $sdir
      export PATH=$PATH:$sdir
    else
      echo You appear to not have SRILM tools installed, either on your path,
      echo or installed in $sdir.  See tools/install_srilm.sh for installation
      echo instructions.
      exit 1
    fi
  fi
fi

# Handles parallelization.
thread_string=
if [ $num_threads -gt 1 ]; then
  thread_string="-parallel --num-threads=$num_threads"
  if [ -z $parallel_opts ]; then
    parallel_opts="--num-threads $num_threads"
  fi
fi
cntk_forward_opts="$cntk_forward_opts numThreads=$num_threads"

# Checks files.
kaldi_model=$srcdir/final.mdl
cntk_model=$srcdir/cntk.mdl

for f in $sdata/1/feats.scp $data/text \
  $srcdir/tree $kaldi_model $cntk_model $cntk_config; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done
if [ $feature_transform != "NO_FEATURE_TRANSFORM" ]; then
  [ ! -f $feature_transform ] &&\
    echo "$0: missing file $feature_transform" && exit 1;
fi

cp -L $srcdir/tree $dir || exit 1;
cp -L $cntk_model $dir || exit 1;
cp -L $kaldi_model $dir || exit 1;

mkdir $dir/configs
cp -f $cntk_config $dir/configs/Decode.config

# Compiles decoding graph.
echo "Compiling decoding graph in $dir/dengraph"
cp -rH $lang $dir/
new_lang="$dir/"$(basename "$lang")
oov=`cat $lang/oov.int` || exit 1;
oov_txt=`cat $lang/oov.txt`
if [ -s $dir/dengraph/HCLG.fst ] && [ $dir/dengraph/HCLG.fst -nt $kaldi_model ]; then
  echo "Graph $dir/dengraph/HCLG.fst already exists: skipping graph creation."
else
  echo "Making ${ngram_order}-gram grammar FST in $new_lang"
  if [ $ngram_order -eq 1 ]; then
    cat $data/text | utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt | \
      awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
      utils/make_unigram_grammar.pl | fstcompile |\
      fstarcsort --sort_type=ilabel > $new_lang/G.fst || exit 1;
  else
    cat $data/text | awk -v voc=$lang/words.txt -v oov="$oov_txt" '
      BEGIN{ while((getline<voc)>0) { invoc[$1]=1; } } {
      for (x=2;x<=NF;x++) {
      if (invoc[$x]) { printf("%s ", $x); } else { printf("%s ", oov); } }
      printf("\n"); }' > $new_lang/text.tmp
    ngram-count -text $new_lang/text.tmp -order $ngram_order "$srilm_options" -lm - |\
      arpa2fst - | fstprint | utils/eps2disambig.pl | utils/s2eps.pl |\
      fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt  \
      --keep_isymbols=false --keep_osymbols=false |\
      fstrmepsilon | fstarcsort --sort_type=ilabel > $new_lang/G.fst || exit 1;
  fi
  utils/mkgraph.sh $new_lang $srcdir $dir/dengraph || exit 1;
fi

# Features to be fed to CNTK.
feats="scp:$sdata/JOB/feats.scp"
feats_one="scp:$sdata/1/feats.scp"
if [ -z $feat_dim ]; then feat_dim=$(feat-to-dim "$feats_one" -) || exit 1; fi
label_dim=$(am-info $kaldi_model 2>/dev/null | grep "pdfs" | awk '{print $4;}') || exit 1;

cntk_input_counts="$sdata/JOB/input_cntk.counts"
cntk_input_feats="$sdata/JOB/input_cntk_feats.scp"

$cmd JOB=1:$nj $dir/log/split_input_counts.JOB.log \
  feat-to-len "$feats" ark,t:"$cntk_input_counts" || exit 1;
$cmd JOB=1:$nj $dir/log/make_input_feats.JOB.log \
  echo scp:$sdata/JOB/feats.scp \> $cntk_input_feats || exit 1;

# Features to be generated by CNTK.
cntk_feats="cntk $cntk_forward_opts featureTransform=$feature_transform"
cntk_feats="$cntk_feats ExpDir=$dir configFile=$cntk_config DeviceNumber=$device"
cntk_feats="$cntk_feats modelName=$cntk_model labelDim=$label_dim featDim=$feat_dim"
cntk_feats="$cntk_feats inputCounts=$cntk_input_counts inputFeats=$cntk_input_feats"

# if this job is interrupted by the user, we want any background jobs to be
# killed too.
cleanup() {
  local pids=$(jobs -pr)
  [ -n "$pids" ] && kill $pids
}
trap "cleanup" INT QUIT TERM EXIT


echo "$0: generating denlats from data '$data', putting lattices in '$dir'"
if [ $sub_split -eq 1 ]; then
  # Generates lattices
  $cmd $parallel_opts JOB=1:$nj $dir/log/decode_den.JOB.log \
    $cntk_feats \| latgen-faster-mapped$thread_string --beam=$beam \
    --lattice-beam=$lattice_beam --acoustic-scale=$acwt --max-mem=$max_mem \
    --max-active=$max_active --word-symbol-table=$lang/words.txt $kaldi_model \
    $dir/dengraph/HCLG.fst ark:- "ark,scp:$dir/lat.JOB.ark,$dir/lat.JOB.scp" || exit 1;
else
  # each job from 1 to $nj is split into multiple pieces (sub-split), and we aim
  # to have at most two jobs running at each time.  The idea is that if we have
  # stragglers from one job, we can be processing another one at the same time.
  rm $dir/.error 2>/dev/null

  prev_pid=
  for n in `seq $[nj+1]`; do
    if [ $n -gt $nj ]; then
      this_pid=
    elif [ -f $dir/.done.$n ] && [ $dir/.done.$n -nt $kaldi_model ]; then
      echo "Not processing subset $n as already done (delete $dir/.done.$n if not)";
      this_pid=
    else
      mkdir -p $dir/log/$n
      sdata2=$data/split$nj/$n/split$sub_split;
      if [ ! -d $sdata2 ] || [ $sdata2 -ot $sdata/$n/feats.scp ]; then
        split_data.sh --per-utt $sdata/$n $sub_split || exit 1;
        feats_subset="scp:$sdata/$n/split$sub_split/JOB/feats.scp"
        cntk_input_counts_subset="$sdata/$n/split$sub_split/JOB/input_cntk.counts"
        cntk_input_feats_subset="$sdata/$n/split$sub_split/JOB/input_cntk_feats.scp"
        $cmd JOB=1:$sub_split $dir/log/$n/split_input_counts.JOB.log \
          feat-to-len "$feats_subset" ark,t:"$cntk_input_counts_subset" || exit 1;
        $cmd JOB=1:$sub_split $dir/log/$n/make_input_feats.JOB.log \
          echo scp:$sdata/$n/split$sub_split/JOB/feats.scp \> $cntk_input_feats_subset || exit 1;
      fi
      cntk_feats_subset=$(echo $cntk_feats | sed s:JOB/:$n/split$sub_split/JOB/:g)

      # Generates lattices
      $cmd $parallel_opts JOB=1:$sub_split $dir/log/$n/decode_den.JOB.log \
        $cntk_feats_subset \| latgen-faster-mapped$thread_string --beam=$beam \
        --lattice-beam=$lattice_beam --acoustic-scale=$acwt --max-mem=$max_mem \
        --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl \
        $dir/dengraph/HCLG.fst ark:- "ark:|gzip -c > $dir/lat.$n.JOB.gz" || touch $dir/.error &
      this_pid=$!
    fi
    if [ ! -z "$prev_pid" ]; then  # Wait for the previous job; merge the previous set of lattices.
      wait $prev_pid
      [ -f $dir/.error ] && echo "$0: error generating denominator lattices" && exit 1;
      rm $dir/.merge_error 2>/dev/null
      echo Merging archives for data subset $prev_n
      lattice-copy "ark:gzip -cdf $dir/lat.$prev_n.*.gz|" \
        ark,scp:$dir/lat.$prev_n.ark,$dir/lat.$prev_n.scp || \
        touch $dir/.merge_error;
      [ -f $dir/.merge_error ] &&\
        echo "$0: Merging lattices for subset $prev_n failed (or maybe some other error)" && exit 1;
      rm $dir/lat.$prev_n.*.gz
      touch $dir/.done.$prev_n
    fi
    prev_n=$n
    prev_pid=$this_pid
  done
fi

cat $dir/lat.*.scp > $dir/lat.scp || exit 1;

echo "$0: done generating denominator lattices."
