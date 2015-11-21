#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Computes training alignments using a model with delta or
# LDA+MLLT features.

# If you supply the "--use-graphs true" option, it will use the training
# graphs from the source directory (where the model is).  In this
# case the number of jobs must match with the source directory.


# Begin configuration section.  
nj=4
cmd=run.pl
use_graphs=false
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence during alignment.
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: steps/extract_feats.sh <data-dir> <src-dir> <out-dir>"
   exit 1;
fi

data=$1
srcdir=$2
dir=$3

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options.
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

#cp $srcdir/{tree,final.mdl} $dir || exit 1;
#cp $srcdir/final.occs $dir;


if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi

if [ ! -f $srcdir/final.mdl ]; then feat_type=simple; fi

echo "$0: feature type is $feat_type"


case $feat_type in
  simple) feats="scp:$sdata/JOB/feats.scp";;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

logdir=$dir/log

echo "$0: extract features"
if [ ! -f $dir/features ]; then
    split_segments=""
    for ((n=1; n<=nj; n++)); do
        split_segments="$split_segments $logdir/segments.$n"
    done

    $cmd JOB=1:$nj $logdir/extact_feats.JOB.log \
    copy-feats "$feats" ark:- \| \
    copy-feats-to-htk --output-dir=$dir/features ark:- \
     || exit 1;


fi



