# Note, this path.sh works on the Amazon cluster, you may want to create your
# own. I suggest putting all the Kaldi and CNTK related paths into a single
# path.sh file.

export KALDI_ROOT=/export/ws15-dnn-data/guoguo/kaldi-trunk/
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/:$PWD:$PATH
export LC_ALL=C

LMBIN=$KALDI_ROOT/tools/irstlm/bin
SRILM=$KALDI_ROOT/tools/srilm/bin/i686-m64
BEAMFORMIT=$KALDI_ROOT/tools/BeamformIt-3.51

export PATH=$PATH:$LMBIN:$BEAMFORMIT:$SRILM

LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH

# For CNTK.
export LD_LIBRARY_PATH=/export/ws15-dnn-data/tools/acml/gfortran64/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/export/ws15-dnn-data/guoguo/cntk-gcc/bin/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/export/ws15-dnn-data/guoguo/kaldi-trunk/src/lib/:$LD_LIBRARY_PATH
export PATH=/export/ws15-dnn-data/guoguo/cntk-gcc/bin/:$PATH

