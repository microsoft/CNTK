# Makefile for a Linux/GCC build of CNTK
#
# The Linux and Windows versions are not different branches, but rather build off the same
# source files, using different makefiles. This current makefile has the purpose of enabling
# work to make all sources compile with GCC, and also to check for GCC-compat regressions due to
# modifications which are currently done under Windows.
#
# This makefile will be extended/completed as we go.
#
# To use this Makefile, create a directory to build in and make a Config.make in the directory
# that provides
# ACML_PATH= path to ACML library installation
#   only needed if MATHLIB=acml
# MKL_PATH= path to MKL library installation
#   only needed if MATHLIB=mkl
# GDK_PATH= path to cuda gdk installation, so $(GDK_PATH)/include/nvidia/gdk/nvml.h exists
#   defaults to /usr
# BUILDTYPE= One of release or debug
#   defaults to release
# MATHLIB= One of acml or mkl
#   defaults to acml
# CUDA_PATH= Path to CUDA
#   If not specified, GPU will not be enabled
# KALDI_PATH= Path to Kaldi
#   If not specified, Kaldi plugins will not be built

ifndef BUILD_TOP
BUILD_TOP=.
endif

ifneq ("$(wildcard $(BUILD_TOP)/Config.make)","")
  include $(BUILD_TOP)/Config.make
else
  $(error Cannot fine $(BUILD_TOP)/Config.make.  Please see the README file for configuration instructions.)
endif

ifndef BUILDTYPE
$(info Defaulting BUILDTYPE=release)
BUILDTYPE=release
endif

ifndef MATHLIB
$(info DEFAULTING MATHLIB=acml)
MATHLIB = acml
endif

#### Configure based on options above

# The mpic++ wrapper only adds MPI specific flags to the g++ command line.
# The actual compiler/linker flags added can be viewed by running 'mpic++ --showme:compile' and 'mpic++ --showme:link'
CXX = mpic++

INCLUDEPATH:= Common/Include Math/Math MachineLearning/CNTK MachineLearning/CNTKComputationNetworkLib MachineLearning/CNTKSGDLib BrainScript
CPPFLAGS:= -D_POSIX_SOURCE -D_XOPEN_SOURCE=600 -D__USE_XOPEN2K
CXXFLAGS:= -msse3 -std=c++0x -std=c++11 -fopenmp -fpermissive -fPIC -Werror -Wno-error=literal-suffix
LIBPATH:=
LIBS:=
LDFLAGS:=

SEPARATOR = "=-----------------------------------------------------------="
ALL:=
SRC:=

# Make sure all is the first (i.e. default) target, but we can't actually define it
# this early in the file, so let buildall do the work.
all : buildall

# Set up basic nvcc options and add CUDA targets from above
CUFLAGS = -std=c++11 -D_POSIX_SOURCE -D_XOPEN_SOURCE=600 -D__USE_XOPEN2K -m 64

ifdef CUDA_PATH
  ifndef GDK_PATH
    $(info defaulting GDK_PATH to /usr)
    GDK_PATH=/usr
endif

  DEVICE = gpu

  NVCC = $(CUDA_PATH)/bin/nvcc

  # This is a suggested/default location for NVML
  INCLUDEPATH+=$(GDK_PATH)/include/nvidia/gdk
  NVMLPATH=$(GDK_PATH)/src/gdk/nvml/lib

# Set up CUDA includes and libraries
  INCLUDEPATH += $(CUDA_PATH)/include
  LIBPATH += $(CUDA_PATH)/lib64
  LIBS += -lcublas -lcudart -lcuda -lcurand -lcusparse -lnvidia-ml

else
  DEVICE = cpu

  CPPFLAGS +=-DCPUONLY
endif

ifeq ("$(MATHLIB)","acml")
  INCLUDEPATH += $(ACML_PATH)/include
  LIBPATH += $(ACML_PATH)/lib
  LIBS += -lacml -lm -lpthread
  CPPFLAGS += -DUSE_ACML
endif

ifeq ("$(MATHLIB)","mkl")
  INCLUDEPATH += $(MKL_PATH)/mkl/include
  LIBPATH += $(MKL_PATH)/compiler/lib/intel64 $(MKL_PATH)/mkl/lib/intel64 $(MKL_PATH)/compiler/lib/mic $(MKL_PATH)/mkl/lib/mic
  LIBS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lm -liomp5 -lpthread
  CPPFLAGS += -DUSE_MKL
endif


ifdef KALDI_PATH
  ########## Copy includes and defines from $(KALDI_PATH)/src/kaldi.mk ##########
  FSTROOT = $(KALDI_PATH)/tools/openfst
  ATLASINC = $(KALDI_PATH)/tools/ATLAS/include

  INCLUDEPATH += $(KALDI_PATH)/src $(ATLASINC) $(FSTROOT)/include
  CPPFLAGS+= -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -DHAVE_OPENFST_GE_10400

  KALDI_LIBPATH += $(KALDI_PATH)/src/lib
  KALDI_LIBS += -lkaldi-util -lkaldi-matrix -lkaldi-base -lkaldi-hmm -lkaldi-cudamatrix -lkaldi-nnet -lkaldi-lat
endif

# Set up nvcc target architectures (will generate code to support them all, i.e. fat-binary, in release mode)
# In debug mode we will rely on JIT to create code "on the fly" for the underlying architecture
GENCODE_SM20 := -gencode arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM30 := -gencode arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 := -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 := -gencode arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_FLAGS := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50)

ifeq ("$(BUILDTYPE)","debug")
  CXXFLAGS += -g
  CUFLAGS += -O0 -G -lineinfo -gencode arch=compute_20,code=\"compute_20\"
endif

ifeq ("$(BUILDTYPE)","release")
  CXXFLAGS += -O4
  CUFLAGS += -O3 -use_fast_math -lineinfo $(GENCODE_FLAGS)
endif

#######

OBJDIR:= $(BUILD_TOP)/.build
BINDIR:= $(BUILD_TOP)/bin
LIBDIR:= $(BUILD_TOP)/lib

ORIGINLIBDIR:='$$ORIGIN/../lib'
ORIGINDIR:='$$ORIGIN'

CNTKMATH:=cntkmath

########################################
# Math library
########################################

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
	Math/Math/MatrixQuantizer.cpp \
	Math/Math/MatrixQuantizerCPU.cpp \
	Math/Math/QuantizedMatrix.cpp \
	Math/Math/Matrix.cpp \
	Math/Math/CUDAPageLockedMemAllocator.cpp \

ifdef CUDA_PATH
MATH_SRC +=\
	Math/Math/GPUMatrix.cu \
	Math/Math/GPUMatrixCUDAKernels.cu \
	Math/Math/GPUSparseMatrix.cu \
	Math/Math/GPUWatcher.cu \
	Math/Math/MatrixQuantizerGPU.cu \

else
MATH_SRC +=\
	Math/Math/NoGPU.cpp

endif

MATH_SRC+=$(COMMON_SRC)

MATH_OBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(patsubst %.cpp, $(OBJDIR)/%.o, $(MATH_SRC)))

CNTKMATH_LIB:= $(LIBDIR)/lib$(CNTKMATH).so
ALL += $(CNTKMATH_LIB)
SRC+=$(MATH_SRC)

RPATH=-Wl,-rpath,

$(CNTKMATH_LIB): $(MATH_OBJ)
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE) 
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBPATH) $(NVMLPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -fopenmp

########################################
# BinaryReader plugin
########################################

BINARYREADER_SRC =\
	DataReader/BinaryReader/BinaryFile.cpp \
	DataReader/BinaryReader/BinaryReader.cpp \
	DataReader/BinaryReader/BinaryWriter.cpp \

BINARYREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(BINARYREADER_SRC))

BINARY_READER:= $(LIBDIR)/BinaryReader.so

#ALL += $(BINARY_READER)
#SRC+=$(BINARYREADER_SRC)

$(BINARY_READER): $(BINARYREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# HTKMLFReader plugin
########################################

HTKMLFREADER_SRC =\
	DataReader/HTKMLFReader/DataReader.cpp \
	DataReader/HTKMLFReader/DataWriter.cpp \
	DataReader/HTKMLFReader/HTKMLFReader.cpp \
	DataReader/HTKMLFReader/HTKMLFWriter.cpp \

HTKMLREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(HTKMLFREADER_SRC))

HTKMLREADER:=$(LIBDIR)/HTKMLFReader.so
ALL+=$(HTKMLREADER)
SRC+=$(HTKMLREADER_SRC)

$(LIBDIR)/HTKMLFReader.so: $(HTKMLREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# LMSequenceReader plugin
########################################

LMSEQUENCEREADER_SRC =\
	DataReader/LMSequenceReader/Exports.cpp \
	DataReader/LMSequenceReader/SequenceParser.cpp \
	DataReader/LMSequenceReader/SequenceReader.cpp \

LMSEQUENCEREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(LMSEQUENCEREADER_SRC))

LMSEQUENCEREADER:= $(LIBDIR)/LMSequenceReader.so
ALL+=$(LMSEQUENCEREADER)
SRC+=$(LMSEQUENCEREADER_SRC)

$(LMSEQUENCEREADER): $(LMSEQUENCEREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# LUSequenceReader plugin
########################################

LUSEQUENCEREADER_SRC =\
	DataReader/LUSequenceReader/Exports.cpp \
	DataReader/LUSequenceReader/LUSequenceParser.cpp \
	DataReader/LUSequenceReader/LUSequenceReader.cpp \

LUSEQUENCEREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(LUSEQUENCEREADER_SRC))

LUSEQUENCEREADER:=$(LIBDIR)/LUSequenceReader.so
ALL+=$(LUSEQUENCEREADER)
SRC+=$(LUSEQUENCEREADER_SRC)

$(LUSEQUENCEREADER): $(LUSEQUENCEREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# UCIFastReader plugin
########################################

UCIFASTREADER_SRC =\
	DataReader/UCIFastReader/Exports.cpp \
	DataReader/UCIFastReader/UCIFastReader.cpp \
	DataReader/UCIFastReader/UCIParser.cpp \

UCIFASTREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(UCIFASTREADER_SRC))

UCIFASTREADER:=$(LIBDIR)/UCIFastReader.so
ALL += $(UCIFASTREADER)
SRC+=$(UCIFASTREADER_SRC)

$(UCIFASTREADER): $(UCIFASTREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# Kaldi plugins
########################################

ifdef KALDI_PATH
KALDIREADER_SRC = \
	DataReader/KaldiReader/DataReader.cpp \
	DataReader/KaldiReader/DataWriter.cpp \
	DataReader/KaldiReader/HTKMLFReader.cpp \
	DataReader/KaldiReader/HTKMLFWriter.cpp \

KALDIREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(KALDIREADER_SRC))

KALDIREADER:=$(LIBDIR)/KaldiReader.so
ALL+=$(KALDIREADER)
SRC+=$(KALDIREADER_SRC)

$(KALDIREADER): $(KALDIREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(KALDI_LIBPATH) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(KALDI_LIBPATH) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH) $(KALDI_LIBS)

KALDIWRITER:=$(LIBDIR)/KaldiWriter.so
ALL+=$(KALDIWRITER)

$(KALDIWRITER): $(KALDIREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)


KALDI2READER_SRC = \
	DataReader/Kaldi2Reader/DataReader.cpp \
	DataReader/Kaldi2Reader/DataWriter.cpp \
	DataReader/Kaldi2Reader/HTKMLFReader.cpp \
	DataReader/Kaldi2Reader/HTKMLFWriter.cpp \
	DataReader/Kaldi2Reader/KaldiSequenceTrainingDerivative.cpp \
	DataReader/Kaldi2Reader/UtteranceDerivativeBuffer.cpp \

KALDI2READER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(KALDI2READER_SRC))

KALDI2READER:=$(LIBDIR)/Kaldi2Reader.so
ALL+=$(KALDI2READER)
SRC+=$(KALDI2READER_SRC)

$(KALDI2READER): $(KALDI2READER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(KALDI_LIBPATH) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(KALDI_LIBPATH) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH) $(KALDI_LIBS)

endif

########################################
# cntk
########################################

CNTK_SRC =\
	MachineLearning/CNTK/CNTK.cpp \
	MachineLearning/CNTK/ModelEditLanguage.cpp \
	MachineLearning/CNTK/NetworkDescriptionLanguage.cpp \
	MachineLearning/CNTK/SimpleNetworkBuilder.cpp \
	MachineLearning/CNTK/SynchronousExecutionEngine.cpp \
	MachineLearning/CNTK/tests.cpp \
	MachineLearning/CNTKComputationNetworkLib/ComputationNode.cpp \
	MachineLearning/CNTKComputationNetworkLib/ComputationNetwork.cpp \
	MachineLearning/CNTKComputationNetworkLib/ComputationNetworkBuilder.cpp \
	MachineLearning/CNTKComputationNetworkLib/NetworkBuilderFromConfig.cpp \
	MachineLearning/CNTKSGDLib/Profiler.cpp \
	MachineLearning/CNTKSGDLib/SGD.cpp \
	BrainScript/BrainScriptEvaluator.cpp \
	BrainScript/BrainScriptParser.cpp \
	BrainScript/BrainScriptTest.cpp \
	MachineLearning/CNTK/ExperimentalNetworkBuilder.cpp \

CNTK_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(CNTK_SRC))

CNTK:=$(BINDIR)/cntk
ALL+=$(CNTK)

$(CNTK): $(CNTK_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building output for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -l$(CNTKMATH) -fopenmp

########################################
# General compile and dependency rules
########################################

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
	$(NVCC) -c $< -o $@  $(CUFLAGS) $(INCLUDEPATH:%=-I%) -Xcompiler "-fPIC -Werror"

$(OBJDIR)/%.o : %.cpp Makefile
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE) 
	@mkdir -p $(dir $@)
	$(CXX) -c $< -o $@ $(CPPFLAGS) $(CXXFLAGS) $(INCLUDEPATH:%=-I%) -MD -MP -MF ${@:.o=.d}

.PHONY: clean buildall all

clean:
	@echo $(SEPARATOR)
	@rm -rf $(OBJDIR)
	@rm -rf $(ALL)
	@echo finished cleaning up the project 

buildall : $(ALL)
	@echo $(SEPARATOR)
	@echo finished building for $(ARCH) with build type $(BUILDTYPE)
