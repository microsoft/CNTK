# Makefile for a Linux/GCC build of CNTK
#
# The Linux and Windows versions are not different branches, but rather build off the same
# source files, using different makefiles. This current makefile has the purpose of enabling
# work to make all sources compile with GCC, and also to check for GCC-compat regressions due to
# modifications which are currently done under Windows.
#
# To use this Makefile, create a directory to build in and make a Config.make in the directory
# that provides
#   BUILDTYPE= One of release or debug
#     defaults to release
#   ACML_PATH= path to ACML library installation
#     only needed if MATHLIB=acml
#   MKL_PATH= path to MKL library installation
#     only needed if MATHLIB=mkl
#   GDK_PATH= path to cuda gdk installation, so $(GDK_PATH)/include/nvidia/gdk/nvml.h exists
#     defaults to /usr
#   MATHLIB= One of acml or mkl
#     defaults to acml
#   CUDA_PATH= Path to CUDA
#     If not specified, GPU will not be enabled
#   CUB_PATH= path to NVIDIA CUB installation, so $(CUB_PATH)/cub/cub.cuh exists
#     defaults to /usr/local/cub-1.4.1
#   CUDNN_PATH= path to NVIDIA cuDNN installation so $(CUDNN_PATH)/cuda/include/cudnn.h exists
#     If not specified, CNTK will be be built without cuDNN.
#   KALDI_PATH= Path to Kaldi
#     If not specified, Kaldi plugins will not be built
#   OPENCV_PATH= path to OpenCV 3.0.0 installation, so $(OPENCV_PATH) exists
#     defaults to /usr/local/opencv-3.0.0
#   LIBZIP_PATH= path to libzip installation, so $(LIBZIP_PATH) exists
#     defaults to /usr/local/
# These can be overridden on the command line, e.g. make BUILDTYPE=debug

ARCH=$(shell uname)

ifndef BUILD_TOP
BUILD_TOP=.
endif

ifneq ("$(wildcard $(BUILD_TOP)/Config.make)","")
  include $(BUILD_TOP)/Config.make
else
  $(error Cannot find $(BUILD_TOP)/Config.make.  Please see CNTK Wiki at https://github.com/Microsoft/cntk/wiki for configuration instructions.)
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

SOURCEDIR:= Source
INCLUDEPATH:= $(addprefix $(SOURCEDIR)/, Common/Include Math CNTK ActionsLib ComputationNetworkLib SGDLib SequenceTrainingLib CNTK/BrainScript Readers/ReaderLib)
# COMMON_FLAGS include settings that are passed both to NVCC and C++ compilers.
COMMON_FLAGS:= -D_POSIX_SOURCE -D_XOPEN_SOURCE=600 -D__USE_XOPEN2K -std=c++11
CPPFLAGS:= 
CXXFLAGS:= -msse3 -std=c++0x -fopenmp -fpermissive -fPIC -Werror -fcheck-new
LIBPATH:=
LIBS:=
LDFLAGS:=

CXXVER_GE480:= $(shell expr `$(CXX) -dumpversion | sed -e 's/\.\([0-9][0-9]\)/\1/g' -e 's/\.\([0-9]\)/0\1/g' -e 's/^[0-9]\{3,4\}$$/&00/'` \>= 40800)
ifeq ($(CXXVER_GE480),1)
	CXXFLAGS += -Wno-error=literal-suffix
endif

SEPARATOR = "=-----------------------------------------------------------="
ALL:=
SRC:=

# Make sure all is the first (i.e. default) target, but we can't actually define it
# this early in the file, so let buildall do the work.
all : buildall

# Set up basic nvcc options and add CUDA targets from above
CUFLAGS = -m 64

ifdef CUDA_PATH
  ifndef GDK_PATH
    $(info defaulting GDK_PATH to /usr)
    GDK_PATH=/usr
  endif

  ifndef CUB_PATH
    $(info defaulting CUB_PATH to /usr/local/cub-1.4.1)
    CUB_PATH=/usr/local/cub-1.4.1
  endif

  DEVICE = gpu

  NVCC = $(CUDA_PATH)/bin/nvcc

  # This is a suggested/default location for NVML
  INCLUDEPATH+=$(GDK_PATH)/include/nvidia/gdk
  INCLUDEPATH+=$(CUB_PATH)
  NVMLPATH=$(GDK_PATH)/src/gdk/nvml/lib

# Set up CUDA includes and libraries
  INCLUDEPATH += $(CUDA_PATH)/include
  LIBPATH += $(CUDA_PATH)/lib64
  LIBS += -lcublas -lcudart -lcuda -lcurand -lcusparse -lnvidia-ml

# Set up cuDNN if needed
  ifdef CUDNN_PATH
    INCLUDEPATH += $(CUDNN_PATH)/cuda/include
    LIBPATH += $(CUDNN_PATH)/cuda/lib64
    LIBS += -lcudnn
    COMMON_FLAGS +=-DUSE_CUDNN
  endif
else
  DEVICE = cpu

  COMMON_FLAGS +=-DCPUONLY
endif

ifeq ("$(MATHLIB)","acml")
  INCLUDEPATH += $(ACML_PATH)/include
  LIBPATH += $(ACML_PATH)/lib
  LIBS += -lacml_mp -liomp5 -lm -lpthread
  COMMON_FLAGS += -DUSE_ACML
endif

ifeq ("$(MATHLIB)","mkl")
  INCLUDEPATH += $(MKL_PATH)/mkl/include
  LIBPATH += $(MKL_PATH)/compiler/lib/intel64 $(MKL_PATH)/mkl/lib/intel64 $(MKL_PATH)/compiler/lib/mic $(MKL_PATH)/mkl/lib/mic
  LIBS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lm -liomp5 -lpthread
  COMMON_FLAGS += -DUSE_MKL
endif

ifeq ("$(MATHLIB)","openblas")
  INCLUDEPATH += $(OPENBLAS_PATH)/include
  LIBPATH += $(OPENBLAS_PATH)/lib
  LIBS += -lopenblas -lm -lpthread
  CPPFLAGS += -DUSE_OPENBLAS
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

ifeq ("$(BUILDTYPE)","debug")
  ifdef CNTK_CUDA_CODEGEN_DEBUG
    GENCODE_FLAGS := $(CNTK_CUDA_CODEGEN_DEBUG)
  else
    GENCODE_FLAGS := -gencode arch=compute_20,code=\"compute_20\" $(GENCODE_SM30)
  endif

  CXXFLAGS += -g
  LDFLAGS += -rdynamic
  COMMON_FLAGS += -D_DEBUG -DNO_SYNC
  CUFLAGS += -O0 -g -use_fast_math -lineinfo  $(GENCODE_FLAGS)
endif

ifeq ("$(BUILDTYPE)","release")
  ifdef CNTK_CUDA_CODEGEN_RELEASE
    GENCODE_FLAGS := $(CNTK_CUDA_CODEGEN_RELEASE)
  else
    GENCODE_FLAGS := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50)
  endif

  CXXFLAGS += -g -O4
  LDFLAGS += -rdynamic
  COMMON_FLAGS += -DNDEBUG -DNO_SYNC
  CUFLAGS += -O3 -g -use_fast_math -lineinfo $(GENCODE_FLAGS)
endif

ifdef CNTK_CUDA_DEVICE_DEBUGINFO
  CUFLAGS += -G
endif

#######

OBJDIR:= $(BUILD_TOP)/.build
BINDIR:= $(BUILD_TOP)/bin
LIBDIR:= $(BUILD_TOP)/lib

ORIGINLIBDIR:='$$ORIGIN/../lib'
ORIGINDIR:='$$ORIGIN'

CNTKMATH:=cntkmath


########################################
# Build info
########################################

BUILDINFO:= $(SOURCEDIR)/CNTK/buildinfo.h
GENBUILD:=Tools/generate_build_info

BUILDINFO_OUTPUT := $(shell $(GENBUILD) $(BUILD_TOP)/Config.make && echo Success)

ifneq ("$(BUILDINFO_OUTPUT)","Success")
  $(error Could not generate $(BUILDINFO))
endif


########################################
# Math library
########################################

# Define all sources that need to be built
READER_SRC =\
	$(SOURCEDIR)/Readers/ReaderLib/BlockRandomizer.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/Bundler.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/NoRandomizer.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/ReaderShim.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/ChunkRandomizer.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/SequenceRandomizer.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/SequencePacker.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/BpttPacker.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/PackerBase.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/FramePacker.cpp \

COMMON_SRC =\
	$(SOURCEDIR)/Common/Config.cpp \
	$(SOURCEDIR)/Common/DataReader.cpp \
	$(SOURCEDIR)/Common/DataWriter.cpp \
	$(SOURCEDIR)/Common/ExceptionWithCallStack.cpp \
	$(SOURCEDIR)/Common/Eval.cpp \
	$(SOURCEDIR)/Common/File.cpp \
	$(SOURCEDIR)/Common/TimerUtility.cpp \
	$(SOURCEDIR)/Common/fileutil.cpp \

MATH_SRC =\
	$(SOURCEDIR)/Math/CPUMatrix.cpp \
	$(SOURCEDIR)/Math/CPUSparseMatrix.cpp \
	$(SOURCEDIR)/Math/MatrixQuantizerImpl.cpp \
	$(SOURCEDIR)/Math/MatrixQuantizerCPU.cpp \
	$(SOURCEDIR)/Math/QuantizedMatrix.cpp \
	$(SOURCEDIR)/Math/Matrix.cpp \
	$(SOURCEDIR)/Math/TensorView.cpp \
	$(SOURCEDIR)/Math/CUDAPageLockedMemAllocator.cpp \
	$(SOURCEDIR)/Math/ConvolutionEngine.cpp \
	$(SOURCEDIR)/Math/BatchNormalizationEngine.cpp \

ifdef CUDA_PATH
MATH_SRC +=\
	$(SOURCEDIR)/Math/GPUMatrix.cu \
	$(SOURCEDIR)/Math/GPUTensor.cu \
	$(SOURCEDIR)/Math/GPUSparseMatrix.cu \
	$(SOURCEDIR)/Math/GPUWatcher.cu \
	$(SOURCEDIR)/Math/MatrixQuantizerGPU.cu \
	$(SOURCEDIR)/Math/CuDnnCommon.cu \
	$(SOURCEDIR)/Math/CuDnnConvolutionEngine.cu \
	$(SOURCEDIR)/Math/CuDnnBatchNormalization.cu \
	$(SOURCEDIR)/Math/GPUDataTransferer.cpp \

else
MATH_SRC +=\
	$(SOURCEDIR)/Math/NoGPU.cpp

endif

MATH_SRC+=$(COMMON_SRC)
MATH_SRC+=$(READER_SRC)

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
	$(SOURCEDIR)/Readers/BinaryReader/BinaryFile.cpp \
	$(SOURCEDIR)/Readers/BinaryReader/BinaryReader.cpp \
	$(SOURCEDIR)/Readers/BinaryReader/BinaryWriter.cpp \

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
	$(SOURCEDIR)/Readers/HTKMLFReader/Exports.cpp \
	$(SOURCEDIR)/Readers/HTKMLFReader/DataWriterLocal.cpp \
	$(SOURCEDIR)/Readers/HTKMLFReader/HTKMLFReader.cpp \
	$(SOURCEDIR)/Readers/HTKMLFReader/HTKMLFWriter.cpp \

HTKMLFREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(HTKMLFREADER_SRC))

HTKMLFREADER:=$(LIBDIR)/HTKMLFReader.so
ALL+=$(HTKMLFREADER)
SRC+=$(HTKMLFREADER_SRC)

$(LIBDIR)/HTKMLFReader.so: $(HTKMLFREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# ExperimentalHTKMLFReader plugin
########################################

EXPERIMENTALHTKMLFREADER_SRC =\
	$(SOURCEDIR)/Readers/HTKMLFReader/DataWriterLocal.cpp \
	$(SOURCEDIR)/Readers/HTKMLFReader/HTKMLFWriter.cpp \
	$(SOURCEDIR)/Readers/ExperimentalHTKMLFReader/ConfigHelper.cpp \
	$(SOURCEDIR)/Readers/ExperimentalHTKMLFReader/Exports.cpp \
	$(SOURCEDIR)/Readers/ExperimentalHTKMLFReader/HTKDataDeserializer.cpp \
	$(SOURCEDIR)/Readers/ExperimentalHTKMLFReader/HTKMLFReader.cpp \
	$(SOURCEDIR)/Readers/ExperimentalHTKMLFReader/MLFDataDeserializer.cpp \

EXPERIMENTALHTKMLFREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(EXPERIMENTALHTKMLFREADER_SRC))

EXPERIMENTALHTKMLFREADER:=$(LIBDIR)/ExperimentalHTKMLFReader.so
ALL+=$(EXPERIMENTALHTKMLFREADER)
SRC+=$(EXPERIMENTALHTKMLFREADER_SRC)

$(LIBDIR)/ExperimentalHTKMLFReader.so: $(EXPERIMENTALHTKMLFREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# LMSequenceReader plugin
########################################

LMSEQUENCEREADER_SRC =\
	$(SOURCEDIR)/Readers/LMSequenceReader/Exports.cpp \
	$(SOURCEDIR)/Readers/LMSequenceReader/SequenceParser.cpp \
	$(SOURCEDIR)/Readers/LMSequenceReader/SequenceReader.cpp \
	$(SOURCEDIR)/Readers/LMSequenceReader/SequenceWriter.cpp \

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
	$(SOURCEDIR)/Readers/LUSequenceReader/Exports.cpp \
	$(SOURCEDIR)/Readers/LUSequenceReader/DataWriterLocal.cpp \
	$(SOURCEDIR)/Readers/LUSequenceReader/LUSequenceParser.cpp \
	$(SOURCEDIR)/Readers/LUSequenceReader/LUSequenceReader.cpp \
	$(SOURCEDIR)/Readers/LUSequenceReader/LUSequenceWriter.cpp \

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
	$(SOURCEDIR)/Readers/UCIFastReader/Exports.cpp \
	$(SOURCEDIR)/Readers/UCIFastReader/UCIFastReader.cpp \
	$(SOURCEDIR)/Readers/UCIFastReader/UCIParser.cpp \

UCIFASTREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(UCIFASTREADER_SRC))

UCIFASTREADER:=$(LIBDIR)/UCIFastReader.so
ALL += $(UCIFASTREADER)
SRC+=$(UCIFASTREADER_SRC)

$(UCIFASTREADER): $(UCIFASTREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# LibSVMBinaryReader plugin
########################################

LIBSVMBINARYREADER_SRC =\
	$(SOURCEDIR)/Readers/LibSVMBinaryReader/Exports.cpp \
	$(SOURCEDIR)/Readers/LibSVMBinaryReader/LibSVMBinaryReader.cpp \

LIBSVMBINARYREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(LIBSVMBINARYREADER_SRC))

LIBSVMBINARYREADER:=$(LIBDIR)/LibSVMBinaryReader.so
ALL += $(LIBSVMBINARYREADER)
SRC+=$(LIBSVMBINARYREADER_SRC)

$(LIBSVMBINARYREADER): $(LIBSVMBINARYREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# SparsePCReader plugin
########################################

SPARSEPCREADER_SRC =\
	$(SOURCEDIR)/Readers/SparsePCReader/Exports.cpp \
	$(SOURCEDIR)/Readers/SparsePCReader/SparsePCReader.cpp \

SPARSEPCREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(SPARSEPCREADER_SRC))

SPARSEPCREADER:=$(LIBDIR)/SparsePCReader.so
ALL += $(SPARSEPCREADER)
SRC+=$(SPARSEPCREADER_SRC)

$(SPARSEPCREADER): $(SPARSEPCREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# CNTKTextFormatReader plugin
########################################

CNTKTEXTFORMATREADER_SRC =\
	$(SOURCEDIR)/Readers/CNTKTextFormatReader/Exports.cpp \
	$(SOURCEDIR)/Readers/CNTKTextFormatReader/Indexer.cpp \
	$(SOURCEDIR)/Readers/CNTKTextFormatReader/TextParser.cpp \
	$(SOURCEDIR)/Readers/CNTKTextFormatReader/CNTKTextFormatReader.cpp \
	$(SOURCEDIR)/Readers/CNTKTextFormatReader/TextConfigHelper.cpp \

CNTKTEXTFORMATREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(CNTKTEXTFORMATREADER_SRC))

CNTKTEXTFORMATREADER:=$(LIBDIR)/CNTKTextFormatReader.so
ALL += $(CNTKTEXTFORMATREADER)
SRC+=$(CNTKTEXTFORMATREADER_SRC)

$(CNTKTEXTFORMATREADER): $(CNTKTEXTFORMATREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)


########################################
# Kaldi plugins
########################################

ifdef KALDI_PATH

KALDI2READER_SRC = \
	$(SOURCEDIR)/Readers/Kaldi2Reader/DataReader.cpp \
	$(SOURCEDIR)/Readers/Kaldi2Reader/DataWriter.cpp \
	$(SOURCEDIR)/Readers/Kaldi2Reader/HTKMLFReader.cpp \
	$(SOURCEDIR)/Readers/Kaldi2Reader/HTKMLFWriter.cpp \
	$(SOURCEDIR)/Readers/Kaldi2Reader/KaldiSequenceTrainingDerivative.cpp \
	$(SOURCEDIR)/Readers/Kaldi2Reader/UtteranceDerivativeBuffer.cpp \

KALDI2READER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(KALDI2READER_SRC))

KALDI2READER:=$(LIBDIR)/Kaldi2Reader.so
ALL+=$(KALDI2READER)
SRC+=$(KALDI2READER_SRC)

$(KALDI2READER): $(KALDI2READER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(KALDI_LIBPATH) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(KALDI_LIBPATH) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH) $(KALDI_LIBS)

endif

########################################
# ImageReader plugin
########################################

ifdef OPENCV_PATH

IMAGE_READER_LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

ifdef LIBZIP_PATH
  CPPFLAGS += -DUSE_ZIP
  INCLUDEPATH += $(LIBZIP_PATH)/lib/libzip/include
  IMAGE_READER_LIBS += -lzip
endif

IMAGEREADER_SRC =\
  $(SOURCEDIR)/Readers/ImageReader/Exports.cpp \
  $(SOURCEDIR)/Readers/ImageReader/ImageConfigHelper.cpp \
  $(SOURCEDIR)/Readers/ImageReader/ImageDataDeserializer.cpp \
  $(SOURCEDIR)/Readers/ImageReader/ImageTransformers.cpp \
  $(SOURCEDIR)/Readers/ImageReader/ImageReader.cpp \
  $(SOURCEDIR)/Readers/ImageReader/ZipByteReader.cpp \

IMAGEREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(IMAGEREADER_SRC))

IMAGEREADER:=$(LIBDIR)/ImageReader.so
ALL += $(IMAGEREADER)
SRC+=$(IMAGEREADER_SRC)

INCLUDEPATH += $(OPENCV_PATH)/include
LIBPATH += $(OPENCV_PATH)/lib $(OPENCV_PATH)/release/lib

$(IMAGEREADER): $(IMAGEREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH) $(IMAGE_READER_LIBS)
endif

########################################
# 1bit SGD setup
########################################

ifeq ("$(CNTK_ENABLE_1BitSGD)","true")

ifeq (,$(wildcard Source/1BitSGD/*.h))
  $(error Build with 1bit-SGD was requested but cannot find the code. Please check https://github.com/Microsoft/CNTK/wiki/Enabling-1bit-SGD for instructions)
endif

  INCLUDEPATH += $(SOURCEDIR)/1BitSGD

  COMMON_FLAGS += -DQUANTIZED_GRADIENT_AGGREGATION
endif

########################################
# cntk
########################################

CNTK_SRC =\
	$(SOURCEDIR)/CNTK/CNTK.cpp \
	$(SOURCEDIR)/CNTK/ModelEditLanguage.cpp \
	$(SOURCEDIR)/CNTK/tests.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNode.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNodeScripting.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/InputAndParamNodes.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ReshapingNodes.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/SpecialPurposeNodes.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetwork.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetworkEvaluation.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetworkAnalysis.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetworkEditing.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetworkBuilder.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetworkScripting.cpp \
	$(SOURCEDIR)/SGDLib/Profiler.cpp \
	$(SOURCEDIR)/SGDLib/SGD.cpp \
	$(SOURCEDIR)/ActionsLib/TrainActions.cpp \
	$(SOURCEDIR)/ActionsLib/EvalActions.cpp \
	$(SOURCEDIR)/ActionsLib/OtherActions.cpp \
	$(SOURCEDIR)/ActionsLib/SpecialPurposeActions.cpp \
	$(SOURCEDIR)/ActionsLib/NetworkFactory.cpp \
	$(SOURCEDIR)/ActionsLib/NetworkDescriptionLanguage.cpp \
	$(SOURCEDIR)/ActionsLib/SimpleNetworkBuilder.cpp \
	$(SOURCEDIR)/ActionsLib/NDLNetworkBuilder.cpp \
	$(SOURCEDIR)/SequenceTrainingLib/latticeforwardbackward.cpp \
	$(SOURCEDIR)/SequenceTrainingLib/parallelforwardbackward.cpp \
	$(SOURCEDIR)/CNTK/BrainScript/BrainScriptEvaluator.cpp \
	$(SOURCEDIR)/CNTK/BrainScript/BrainScriptParser.cpp \
	$(SOURCEDIR)/CNTK/BrainScript/BrainScriptTest.cpp \
	$(SOURCEDIR)/Common/BestGpu.cpp \
	$(SOURCEDIR)/Common/MPIWrapper.cpp \


ifdef CUDA_PATH
CNTK_SRC +=\
	$(SOURCEDIR)/Math/cudalatticeops.cu \
	$(SOURCEDIR)/Math/cudalattice.cpp \
	$(SOURCEDIR)/Math/cudalib.cpp \

else
CNTK_SRC +=\
	$(SOURCEDIR)/SequenceTrainingLib/latticeNoGPU.cpp \

endif

CNTK_OBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(patsubst %.cpp, $(OBJDIR)/%.o, $(CNTK_SRC)))

CNTK:=$(BINDIR)/cntk
ALL+=$(CNTK)
SRC+=$(CNTK_SRC)

$(CNTK): $(CNTK_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building output for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(NVMLPATH)) $(patsubst %,$(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -l$(CNTKMATH) -fopenmp

# deployable resources: standard library of BS
CNTK_CORE_BS:=$(BINDIR)/cntk.core.bs
ALL += $(CNTK_CORE_BS)
$(CNTK_CORE_BS): $(SOURCEDIR)/CNTK/BrainScript/CNTKCoreLib/CNTK.core.bs
	@mkdir -p $(dir $@)
	@echo bin-placing deployable resource files
	cp -f $^ $@

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

MAKEFILES :=  Makefile $(BUILD_TOP)/Config.make
$(OBJDIR)/%.o : %.cu $(MAKEFILES)

	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE)
	@mkdir -p $(dir $@)
	$(NVCC) -c $< -o $@ $(COMMON_FLAGS) $(CUFLAGS) $(INCLUDEPATH:%=-I%) -Xcompiler "-fPIC -Werror"

$(OBJDIR)/%.o : %.cpp $(MAKEFILES)
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE)
	@mkdir -p $(dir $@)
	$(CXX) -c $< -o $@ $(COMMON_FLAGS) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDEPATH:%=-I%) -MD -MP -MF ${@:.o=.d}

.PHONY: clean buildall all

clean:
	@echo $(SEPARATOR)
	@rm -rf $(OBJDIR)
	@rm -rf $(ALL)
	@rm -rf $(BUILDINFO)
	@echo finished cleaning up the project

buildall : $(ALL)
	@echo $(SEPARATOR)
	@echo finished building for $(ARCH) with build type $(BUILDTYPE)
