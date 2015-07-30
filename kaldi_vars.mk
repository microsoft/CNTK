########## Set your Kaldi location and the Kaldi includes / libs you want to use ##########

############### make sure your kaldi is compiled by --share mode so we can find the lib ##############################
# KALDI_PATH = /usr/users/yzhang87/code/kaldi-trunk
KALDI_PATH = /export/ws15-dnn-data/guoguo/kaldi-trunk/

KALDI_INCLUDES = -I $(KALDI_PATH)/src
KALDI_LIBS = -L$(KALDI_PATH)/src/lib -lkaldi-util -lkaldi-matrix -lkaldi-base -lkaldi-hmm -lkaldi-cudamatrix -lkaldi-nnet -lkaldi-lat

########## Copy includes and defines from $(KALDI_PATH)/src/kaldi.mk ##########

FSTROOT = $(KALDI_PATH)/tools/openfst
ATLASINC = $(KALDI_PATH)/tools/ATLAS/include

KALDI_INCLUDES += \
-I $(ATLASINC) \
-I $(FSTROOT)/include

KALDI_DEFINES = \
-DKALDI_DOUBLEPRECISION=0 \
-DHAVE_POSIX_MEMALIGN \
-DHAVE_EXECINFO_H=1 \
-DHAVE_CXXABI_H \
-DHAVE_ATLAS \
-DHAVE_OPENFST_GE_10400 \

########## KALDI_LIBS, KALDI_INCLUDES, and KALDI_DEFINES are the only variables used outside this file ##########
