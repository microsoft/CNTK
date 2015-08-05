# Makefile for a Linux/GCC build of CNTK
#
# The Linux and Windows versions are not different branches, but rather build off the same
# source files, using different makefiles. This current makefile has the purpose of enabling
# work to make all sources compile with GCC, and also to check for GCC-compat regressions due to
# modifications which are currently done under Windows.
#
# This makefile will be extended/completed as we go.
#
# export PATH=$PATH:/usr/local/bin:<path_to_cuda>/bin
#
# In order to deviate from the default settings in this Makefile, please specify options on
# the make command line, like this, for example, to build debug with Kaldi for cuda,
#
# make DEBUG=1 USE_CUDA=1 USE_KALDI=1

# These are the options.  USE_ACML is the default for CPU math.
#DEBUG
#USE_ACML
#USE_MKL
#USE_CUDA
#USE_KALDI

# Paths.  This still needs some generification
ACML_PATH = /usr/local/acml5.3.1/ifort64
MKL_PATH = /usr/users/yzhang87/tools/composer_xe_2015.2.164
CUDA_PATH = /scratch/cuda-6.5
#CUDA_PATH = /usr/local/cuda-7.0

# Kaldi build should correspond to whether you are using cuda
KALDI_CPU_PATH = /usr/users/cyphers/kaldi-trunk
KALDI_CUDA_PATH = /usr/users/yzhang87/code/kaldi-trunk

# You need to install the deployment kit from https://developer.nvidia.com/gpu-deployment-kit
# This is for the default install location, /
GDK_PATH=/usr

#### Configure based on options above

CC = g++

ifndef ARCH
ARCH = $(shell uname -m)
endif

ifndef USE_MKL
ifndef USE_ACML
USE_ACML=1
endif
endif

INCLUDEPATH:= Common/Include Math/Math MachineLearning/CNTK
CPPFLAGS:= -D_POSIX_SOURCE -D_XOPEN_SOURCE=600 -D__USE_XOPEN2K
CXXFLAGS:= -msse3 -std=c++0x -std=c++11 -fopenmp -fpermissive -fPIC
LIBPATH:=
LIBS:=
LDFLAGS:=
ORIGINLIBDIR:='$$ORIGIN/../lib'
ORIGINDIR:='$$ORIGIN'

SEPARATOR = "=-----------------------------------------------------------="
TARGETS:=
SRC:=

all : alltargets

# Set up nvcc target architectures (will generate code to support them all, i.e. fat-binary)
GENCODE_SM20 := -gencode arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM30 := -gencode arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 := -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)

# Set up basic nvcc options and add CUDA targets from above
CUFLAGS = -std=c++11 -D_POSIX_SOURCE -D_XOPEN_SOURCE=600 -D__USE_XOPEN2K -m 64 $(GENCODE_FLAGS)

ifdef USE_CUDA
  DEVICE = gpu

  NVCC = $(CUDA_PATH)/bin/nvcc

  KALDI_PATH = $(KALDI_CUDA_PATH)

  # This is a suggested/default location for NVML
  INCLUDEPATH+=$(GDK_PATH)/include/nvidia/gdk
  NVMLPATH=$(GDK_PATH)/src/gdk/nvml/lib

  # Set up CUDA includes and libraries
  INCLUDEPATH += $(CUDA_PATH)/include
  LIBPATH += $(CUDA_PATH)/lib64
  LIBS += -lcublas -lcudart -lcuda -lcurand -lcusparse -lnvidia-ml

else
  DEVICE = cpu

  KALDI_PATH = $(KALDI_CPU_PATH)

  CPPFLAGS +=-DCPUONLY
endif

ifdef USE_ACML
  MATHLIB = acml
  INCLUDEPATH += $(ACML_PATH)/include
  LIBPATH += $(ACML_PATH)/lib
  LIBS += -lacml -lm -lpthread
  CPPFLAGS += -DUSE_ACML
endif

ifdef USE_MKL
  MATHLIB = mkl
  INCLUDEPATH += $(MKL_PATH)/mkl/include
  LIBPATH += $(MKL_PATH)/compiler/lib/intel64 $(MKL_PATH)/mkl/lib/intel64 $(MKL_PATH)/compiler/lib/mic $(MKL_PATH)/mkl/lib/mic
  LIBS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lm -liomp5 -lpthread
  CPPFLAGS += -DUSE_MKL
endif


ifdef USE_KALDI
  ########## Copy includes and defines from $(KALDI_PATH)/src/kaldi.mk ##########
  FSTROOT = $(KALDI_PATH)/tools/openfst
  ATLASINC = $(KALDI_PATH)/tools/ATLAS/include

  INCLUDEPATH += $(KALDI_PATH)/src $(ATLASINC) $(FSTROOT)/include
  CPPFLAGS+= -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -DHAVE_OPENFST_GE_10400

  KALDI_LIBPATH += $(KALDI_PATH)/src/lib
  KALDI_LIBS += -lkaldi-util -lkaldi-matrix -lkaldi-base -lkaldi-hmm -lkaldi-cudamatrix -lkaldi-nnet -lkaldi-lat
endif

# BUILDTYPE can also be release
ifdef DEBUG
  BUILDTYPE = debug
  CXXFLAGS += -g
  CUFLAGS += -O0 -G -lineinfo
else
  BUILDTYPE = release
  CXXFLAGS += -O4
  CUFLAGS += -O3 -use_fast_math -lineinfo
endif

#######

BUILDFOR:= $(ARCH).$(DEVICE).$(BUILDTYPE).$(MATHLIB)

OBJDIR:= .build/$(BUILDFOR)
BINDIR:= $(BUILDFOR)/bin
LIBDIR:= $(BUILDFOR)/lib

CNTKMATH:=cntkmath

# Define all sources that need to be built
COMMON_SRC =\
	Common/BestGpu.cpp \
	Common/ConfigFile.cpp \
	Common/DataReader.cpp \
	Common/DataWriter.cpp \
	Common/Eval.cpp \
	Common/File.cpp \
	Common/TimerUtility.cpp \
	Common/fileutil.cpp \

MATH_SRC =\
	Math/Math/CPUMatrix.cpp \
	Math/Math/CPUSparseMatrix.cpp \
	Math/Math/Matrix.cpp \

ifdef USE_CUDA
MATH_SRC +=\
	Math/Math/GPUMatrix.cu \
	Math/Math/GPUMatrixCUDAKernels.cu \
	Math/Math/GPUSparseMatrix.cu \
	Math/Math/GPUWatcher.cu \

else
MATH_SRC +=\
	Math/Math/NoGPU.cpp

endif

MATH_SRC+=$(COMMON_SRC)

MATH_OBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(patsubst %.cpp, $(OBJDIR)/%.o, $(MATH_SRC)))

CNTKMATH_LIB:= $(LIBDIR)/lib$(CNTKMATH).so
TARGETS += $(CNTKMATH_LIB)
SRC+=$(MATH_SRC)

RPATH=-Wl,-rpath,

$(CNTKMATH_LIB): $(MATH_OBJ)
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE) 
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBPATH) $(NVMLPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -fopenmp

LIBLIBPATH:=$(LIBDIR) $(LIBPATH)

BINARYREADER_SRC =\
	DataReader/BinaryReader/BinaryFile.cpp \
	DataReader/BinaryReader/BinaryReader.cpp \
	DataReader/BinaryReader/BinaryWriter.cpp \

BINARYREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(BINARYREADER_SRC))

BINARY_READER:= $(LIBDIR)/BinaryReader.so

#TARGETS += $(BINARY_READER)
#SRC+=$(BINARYREADER_SRC)

$(BINARY_READER): $(BINARYREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CC) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

HTKMLFREADER_SRC =\
	DataReader/HTKMLFReader_linux/DataReader.cpp \
	DataReader/HTKMLFReader_linux/DataWriter.cpp \
	DataReader/HTKMLFReader_linux/HTKMLFReader.cpp \
	DataReader/HTKMLFReader_linux/HTKMLFWriter.cpp \

HTKMLREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(HTKMLFREADER_SRC))

HTKMLREADER:=$(LIBDIR)/HTKMLFReader.so
TARGETS+=$(HTKMLREADER)
SRC+=$(HTKMLREADER_SRC)

$(LIBDIR)/HTKMLFReader.so: $(HTKMLREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CC) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

LMSEQUENCEREADER_SRC =\
	DataReader/LMSequenceReader/Exports.cpp \
	DataReader/LMSequenceReader/SequenceParser.cpp \
	DataReader/LMSequenceReader/SequenceReader.cpp \

LMSEQUENCEREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(LMSEQUENCEREADER_SRC))

LMSEQUENCEREADER:= $(LIBDIR)/LMSequenceReader.so
TARGETS+=$(LMSEQUENCEREADER)
SRC+=$(LMSEQUENCEREADER_SRC)

$(LMSEQUENCEREADER): $(LMSEQUENCEREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CC) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)



LUSEQUENCEREADER_SRC =\
	DataReader/LUSequenceReader/Exports.cpp \
	DataReader/LUSequenceReader/LUSequenceParser.cpp \
	DataReader/LUSequenceReader/LUSequenceReader.cpp \

LUSEQUENCEREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(LUSEQUENCEREADER_SRC))

LUSEQUENCEREADER:=$(LIBDIR)/LUSequenceReader.so
TARGETS+=$(LUSEQUENCEREADER)
SRC+=$(LUSEQUENCEREADER_SRC)

$(LUSEQUENCEREADER): $(LUSEQUENCEREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CC) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)



UCIFASTREADER_SRC =\
	DataReader/UCIFastReader/Exports.cpp \
	DataReader/UCIFastReader/UCIFastReader.cpp \
	DataReader/UCIFastReader/UCIParser.cpp \

UCIFASTREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(UCIFASTREADER_SRC))

UCIFASTREADER:=$(LIBDIR)/UCIFastReader.so
TARGETS += $(UCIFASTREADER)
SRC+=$(UCIFASTREADER_SRC)

$(UCIFASTREADER): $(UCIFASTREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CC) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

ifdef USE_KALDI
KALDIREADER_SRC = \
	DataReader/KaldiReader/DataReader.cpp \
	DataReader/KaldiReader/DataWriter.cpp \
	DataReader/KaldiReader/HTKMLFReader.cpp \
	DataReader/KaldiReader/HTKMLFWriter.cpp \

KALDIREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(KALDIREADER_SRC))

KALDIREADER:=$(LIBDIR)/KaldiReader.so
TARGETS+=$(KALDIREADER)
SRC+=$(KALDIREADER_SRC)

$(KALDIREADER): $(KALDIREADER_OBJ)
	@echo $(SEPARATOR)
	$(CC) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(KALDI_LIBPATH) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(KALDI_LIBPATH) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH) $(KALDI_LIBS)

KALDIWRITER:=$(LIBDIR)/KaldiWriter.so
TARGETS+=$(KALDIWRITER)

$(KALDIWRITER): $(KALDIREADER_OBJ)
	@echo $(SEPARATOR)
	$(CC) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)


KALDI2READER_SRC = \
	DataReader/Kaldi2Reader/DataReader.cpp \
	DataReader/Kaldi2Reader/DataWriter.cpp \
	DataReader/Kaldi2Reader/HTKMLFReader.cpp \
	DataReader/Kaldi2Reader/HTKMLFWriter.cpp \
	DataReader/Kaldi2Reader/KaldiSequenceTrainingDerivative.cpp \
	DataReader/Kaldi2Reader/UtteranceDerivativeBuffer.cpp \

KALDI2READER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(KALDI2READER_SRC))

KALDI2READER:=$(LIBDIR)/Kaldi2Reader.so
TARGETS+=$(KALDI2READER)
SRC+=$(KALDI2READER_SRC)

$(KALDI2READER): $(KALDI2READER_OBJ)
	@echo $(SEPARATOR)
	$(CC) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(KALDI_LIBPATH) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(KALDI_LIBPATH) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH) $(KALDI_LIBS)

endif

CN_SRC =\
	MachineLearning/CNTK/CNTK.cpp \
	MachineLearning/CNTK/ComputationNode.cpp \
	MachineLearning/CNTK/ModelEditLanguage.cpp \
	MachineLearning/CNTK/NetworkDescriptionLanguage.cpp \
	MachineLearning/CNTK/Profiler.cpp \
	MachineLearning/CNTK/SimpleNetworkBuilder.cpp \
	MachineLearning/CNTK/tests.cpp \
	MachineLearning/CNTKEval/CNTKEval.cpp \

CN_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(CN_SRC))

CNTK:=$(BINDIR)/cntk
TARGETS+=$(CNTK)

$(CNTK): $(CN_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building output for $(ARCH) with build type $(BUILDTYPE)
	$(CC) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -l$(CNTKMATH) -fopenmp

VPATH := $(sort  $(dir $(SRC)))

# Define object files
OBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(patsubst %.cpp, $(OBJDIR)/%.o, $(SRC)))

# C++ include dependencies generated by -MF compiler option
DEP := $(patsubst %.o, %.d, $(OBJ))

# Include all C++ dependencies, like header files, to ensure that a change in those
# will result in the rebuild.
-include ${DEP}

$(OBJDIR)/%.o : %.cu Makefile
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE) 
	@mkdir -p $(dir $@)
	$(NVCC) -c $< -o $@  $(CUFLAGS) $(INCLUDEPATH:%=-I%) -Xcompiler -fPIC

$(OBJDIR)/%.o : %.cpp Makefile
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE) 
	@mkdir -p $(dir $@)
	$(CC) -c $< -o $@ $(CPPFLAGS) $(CXXFLAGS) $(INCLUDEPATH:%=-I%) -MD -MP -MF ${@:.o=.d}

.PHONY: clean alltargets

clean:
	@echo $(SEPARATOR)
	@rm -rf $(OBJDIR)
	@rm -rf $(BINDIR)
	@rm -rf $(LIBDIR)
	@echo finished cleaning up the project 

alltargets : $(TARGETS)
	@echo $(SEPARATOR)
	@echo finished building for $(ARCH) with build type $(BUILDTYPE)
