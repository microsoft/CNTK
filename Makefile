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
#   MKL_PATH= path to CNTK custom MKL installation
#     only needed if MATHLIB=mkl
#   CNTK_CUSTOM_MKL_VERSION=3
#     version for the CNTK custom MKL installation
#   MKL_THREADING=parallel|sequential
#     only needed if MATHLIB=mkl
#   GDK_INCLUDE_PATH= path to CUDA GDK include path, so $(GDK_INCLUDE_PATH)/nvml.h exists
#     defaults to /usr/include/nvidia/gdk
#   GDK_NVML_LIB_PATH= path to CUDA GDK (stub) library path, so $(GDK_NVML_LIB_PATH)/libnvidia-ml.so exists
#     defaults to /usr/src/gdk/nvml/lib
#   MATHLIB= mkl
#     defaults to mkl
#   CUDA_PATH= Path to CUDA
#     If not specified, GPU will not be enabled
#   CUB_PATH= path to NVIDIA CUB installation, so $(CUB_PATH)/cub/cub.cuh exists
#     defaults to /usr/local/cub-1.4.1
#   CUDNN_PATH= path to NVIDIA cuDNN installation so $(CUDNN_PATH)/cuda/include/cudnn.h exists
#     CuDNN version needs to be 5.0 or higher.
#   KALDI_PATH= Path to Kaldi
#     If not specified, Kaldi plugins will not be built
#   OPENCV_PATH= path to OpenCV 3.1.0 installation, so $(OPENCV_PATH) exists
#     defaults to /usr/local/opencv-3.1.0
#   PROTOBUF_PATH= path to Protocol Buffers 3.1.0 installation, so $(PROTOBUF_PATH) exists
#     defaults to /usr/local/protobuf-3.1.0
#   LIBZIP_PATH= path to libzip installation, so $(LIBZIP_PATH) exists
#     defaults to /usr/local/
#   BOOST_PATH= path to Boost installation, so $(BOOST_PATH)/include/boost/test/unit_test.hpp
#     defaults to /usr/local/boost-1.60.0
#   PYTHON_SUPPORT=true iff CNTK v2 Python module should be build
#   SWIG_PATH= path to SWIG (>= 3.0.10)
#   PYTHON_VERSIONS= list of Python versions to build for
#     A Python version is identified by "27", "34", "35", or "36".
#   PYTHON27_PATH= path to Python 2.7 interpreter
#   PYTHON34_PATH= path to Python 3.4 interpreter
#   PYTHON35_PATH= path to Python 3.5 interpreter
#   PYTHON36_PATH= path to Python 3.6 interpreter
#   MPI_PATH= path to MPI installation, so $(MPI_PATH) exists
#     defaults to /usr/local/mpi
# These can be overridden on the command line, e.g. make BUILDTYPE=debug

# TODO: Build static libraries for common dependencies that are shared by multiple 
# targets, e.g. eval and CNTK.

ARCH=$(shell uname)

ifndef BUILD_TOP
BUILD_TOP=.
endif

ifneq ("$(wildcard $(BUILD_TOP)/Config.make)","")
  include $(BUILD_TOP)/Config.make
else
  $(error Cannot find $(BUILD_TOP)/Config.make.  Please see the CNTK documentation at https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux for configuration instructions.)
endif

ifndef BUILDTYPE
$(info Defaulting BUILDTYPE=release)
BUILDTYPE=release
endif

ifndef MATHLIB
$(info DEFAULTING MATHLIB=mkl)
MATHLIB = mkl
endif

#### Configure based on options above

# The mpic++ wrapper only adds MPI specific flags to the g++ command line.
# The actual compiler/linker flags added can be viewed by running 'mpic++ --showme:compile' and 'mpic++ --showme:link'
ifneq ($(HAS_MPI),0)
CXX = $(MPI_PATH)/bin/mpic++
endif

SSE_FLAGS = -msse4.1 -mssse3

PROTOC = $(PROTOBUF_PATH)/bin/protoc

# Settings for ARM64 architectures that use a crosscompiler on a host machine.
#CXX = aarch64-linux-gnu-g++
#SSE_FLAGS =

SOURCEDIR:= Source
INCLUDEPATH:= $(addprefix $(SOURCEDIR)/, Common/Include CNTKv2LibraryDll CNTKv2LibraryDll/API CNTKv2LibraryDll/proto ../Examples/Extensibility/CPP Math CNTK ActionsLib ComputationNetworkLib SGDLib SequenceTrainingLib CNTK/BrainScript Readers/ReaderLib PerformanceProfilerDll)
INCLUDEPATH+=$(PROTOBUF_PATH)/include
# COMMON_FLAGS include settings that are passed both to NVCC and C++ compilers.
COMMON_FLAGS:= -DHAS_MPI=$(HAS_MPI) -D_POSIX_SOURCE -D_XOPEN_SOURCE=600 -D__USE_XOPEN2K -std=c++11
CPPFLAGS:= 
CXXFLAGS:= $(SSE_FLAGS) -std=c++0x -fopenmp -fpermissive -fPIC -Werror -fcheck-new
LIBPATH:=
LIBS_LIST:=
LDFLAGS:=

CXXVER_GE480:= $(shell expr `$(CXX) -dumpversion | sed -e 's/\.\([0-9][0-9]\)/\1/g' -e 's/\.\([0-9]\)/0\1/g' -e 's/^[0-9]\{3,4\}$$/&00/'` \>= 40800)
ifeq ($(CXXVER_GE480),1)
	CXXFLAGS += -Wno-error=literal-suffix
endif

SEPARATOR = "=-----------------------------------------------------------="
ALL:=
ALL_LIBS:=
PYTHON_LIBS:=
JAVA_LIBS:=
LIBS_FULLPATH:=
SRC:=

# Make sure all is the first (i.e. default) target, but we can't actually define it
# this early in the file, so let buildall do the work.
all : buildall

# Set up basic nvcc options and add CUDA targets from above
CUFLAGS = -m 64

ifdef CUDA_PATH
  ifndef GDK_INCLUDE_PATH
    GDK_INCLUDE_PATH=/usr/include/nvidia/gdk
    $(info defaulting GDK_INCLUDE_PATH to $(GDK_INCLUDE_PATH))
  endif

  ifndef GDK_NVML_LIB_PATH
    GDK_NVML_LIB_PATH=/usr/src/gdk/nvml/lib
    $(info defaulting GDK_NVML_LIB_PATH to $(GDK_NVML_LIB_PATH))
  endif

  ifndef CUB_PATH
    $(info defaulting CUB_PATH to /usr/local/cub-1.4.1)
    CUB_PATH=/usr/local/cub-1.4.1
  endif

  DEVICE = gpu

  NVCC = $(CUDA_PATH)/bin/nvcc

  INCLUDEPATH+=$(GDK_INCLUDE_PATH)
  INCLUDEPATH+=$(CUB_PATH)

# Set up CUDA includes and libraries
  INCLUDEPATH += $(CUDA_PATH)/include
  LIBPATH += $(CUDA_PATH)/lib64
  LIBS_LIST += cublas cudart cuda curand cusparse nvidia-ml

# Set up cuDNN if needed
  ifdef CUDNN_PATH
    INCLUDEPATH += $(CUDNN_PATH)/cuda/include
    LIBPATH += $(CUDNN_PATH)/cuda/lib64
    LIBS_LIST += cudnn
    COMMON_FLAGS +=-DUSE_CUDNN
  endif

# Set up NCCL if needed
  ifdef NCCL_PATH
    INCLUDEPATH += $(NCCL_PATH)/include
    LIBPATH += $(NCCL_PATH)/lib
    LIBS_LIST += nccl
    COMMON_FLAGS += -DUSE_NCCL
  endif
else
  DEVICE = cpu

  COMMON_FLAGS +=-DCPUONLY
endif

ifeq ("$(MATHLIB)","mkl")
  INCLUDEPATH += $(MKL_PATH)/$(CNTK_CUSTOM_MKL_VERSION)/include
  LIBS_LIST += m
ifeq ("$(MKL_THREADING)","sequential")
  LIBPATH += $(MKL_PATH)/$(CNTK_CUSTOM_MKL_VERSION)/x64/sequential
  LIBS_LIST += mkl_cntk_s
else
  LIBPATH += $(MKL_PATH)/$(CNTK_CUSTOM_MKL_VERSION)/x64/parallel
  LIBS_LIST += mkl_cntk_p iomp5 pthread
endif
  COMMON_FLAGS += -DUSE_MKL
endif

ifeq ($(CUDA_GDR),1)
  COMMON_FLAGS += -DUSE_CUDA_GDR
endif

ifeq ("$(MATHLIB)","openblas")
  INCLUDEPATH += $(OPENBLAS_PATH)/include
  LIBPATH += $(OPENBLAS_PATH)/lib
  LIBS_LIST += openblas m pthread
  CPPFLAGS += -DUSE_OPENBLAS
endif


ifdef KALDI_PATH
  ########## Copy includes and defines from $(KALDI_PATH)/src/kaldi.mk ##########
  FSTROOT = $(KALDI_PATH)/tools/openfst
  ATLASINC = $(KALDI_PATH)/tools/ATLAS/include

  INCLUDEPATH += $(KALDI_PATH)/src $(ATLASINC) $(FSTROOT)/include
  CPPFLAGS += -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -DHAVE_OPENFST_GE_10400

  KALDI_LIBPATH += $(KALDI_PATH)/src/lib
  KALDI_LIBS_LIST := kaldi-util kaldi-matrix kaldi-base kaldi-hmm kaldi-cudamatrix kaldi-nnet kaldi-lat
  KALDI_LIBS := $(addprefix -l,$(KALDI_LIBS_LIST))
endif

ifdef SUPPORT_AVX2
  CPPFLAGS += -mavx2
endif

# Set up nvcc target architectures (will generate code to support them all, i.e. fat-binary, in release mode)
# In debug mode we only include cubin/PTX for 30 and rely on PTX / JIT to generate the required native cubin format
# see also http://docs.nvidia.com/cuda/pascal-compatibility-guide/index.html#building-applications-with-pascal-support
GENCODE_SM30 := -gencode arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 := -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 := -gencode arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM52 := -gencode arch=compute_52,code=\"sm_52,compute_52\"
GENCODE_SM60 := -gencode arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM61 := -gencode arch=compute_61,code=\"sm_61,compute_61\"

# Should we relocate *.gcno and *.gcda files using -fprofile-dir option?
# Use GCOV_PREFIX and GCOV_PREFIX_STRIP if relocating:
# For example, if the object file /user/build/foo.o was built with -fprofile-arcs, the final executable will try to create the data file
# /user/build/foo.gcda when running on the target system. This will fail if the corresponding directory does not exist and it is unable
# to create it. This can be overcome by, for example, setting the environment as 'GCOV_PREFIX=/target/run' and 'GCOV_PREFIX_STRIP=1'.
# Such a setting will name the data file /target/run/build/foo.gcda
ifdef CNTK_CODE_COVERAGE
  CXXFLAGS += -fprofile-arcs -ftest-coverage
  LDFLAGS += -lgcov --coverage
endif

ifeq ("$(BUILDTYPE)","debug")
  ifdef CNTK_CUDA_CODEGEN_DEBUG
    GENCODE_FLAGS := $(CNTK_CUDA_CODEGEN_DEBUG)
  else
    GENCODE_FLAGS := $(GENCODE_SM30)
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
    GENCODE_FLAGS := $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM61)
  endif

  CXXFLAGS += -g -O4
  LDFLAGS += -rdynamic
  COMMON_FLAGS += -DNDEBUG -DNO_SYNC
  CUFLAGS += -O3 -g -use_fast_math $(GENCODE_FLAGS)
endif

ifdef CNTK_CUDA_DEVICE_DEBUGINFO
  CUFLAGS += -G
endif

# Create the library link options for the linker.
# LIBS_LIST must not be changed beyond this point.
LIBS:= $(addprefix -l,$(LIBS_LIST))

OBJDIR:= $(BUILD_TOP)/.build
BINDIR:= $(BUILD_TOP)/bin
LIBDIR:= $(BUILD_TOP)/lib
PYTHONDIR:= $(BUILD_TOP)/python

ORIGINLIBDIR:='$$ORIGIN/../lib'
ORIGINDIR:='$$ORIGIN'

########################################
# Components VERSION info
########################################

CNTK_COMPONENT_VERSION := 2.2
ifeq ("$(BUILDTYPE)","debug")
CNTK_COMPONENT_VERSION := $(CNTK_COMPONENT_VERSION)d
endif

CPPFLAGS += -DCNTK_COMPONENT_VERSION="$(CNTK_COMPONENT_VERSION)"

CNTKMATH:=Cntk.Math-$(CNTK_COMPONENT_VERSION)
RPATH=-Wl,-rpath,

########################################
# Build info
########################################

BUILDINFO:= $(SOURCEDIR)/CNTKv2LibraryDll/buildinfo.h
GENBUILD:=Tools/generate_build_info

BUILDINFO_OUTPUT := $(shell $(GENBUILD) $(BUILD_TOP)/Config.make && echo Success)

ifneq ("$(BUILDINFO_OUTPUT)","Success")
  $(error Could not generate $(BUILDINFO))
endif

########################################
# Performance profiler library
########################################

PERF_PROFILER:=Cntk.PerformanceProfiler-$(CNTK_COMPONENT_VERSION)

PP_SRC =\
	$(SOURCEDIR)/PerformanceProfilerDll/PerformanceProfiler.cpp \
	$(SOURCEDIR)/Common/File.cpp \
	$(SOURCEDIR)/Common/fileutil.cpp \
	$(SOURCEDIR)/Common/ExceptionWithCallStack.cpp \

PP_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(PP_SRC))

PERF_PROFILER_LIB:= $(LIBDIR)/lib$(PERF_PROFILER).so
ALL_LIBS += $(PERF_PROFILER_LIB)
PYTHON_LIBS += $(PERF_PROFILER_LIB)
SRC += $(PP_SRC)

$(PERF_PROFILER_LIB): $(PP_OBJ)
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,$(RPATH)%, $(ORIGINDIR)) -o $@ $^


########################################
# Math library
########################################

# Define all sources that need to be built
READER_SRC =\
	$(SOURCEDIR)/Readers/ReaderLib/BlockRandomizer.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/Bundler.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/NoRandomizer.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/LTNoRandomizer.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/LTTumblingWindowRandomizer.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/LocalTimelineRandomizerBase.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/ReaderShim.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/ChunkRandomizer.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/SequenceRandomizer.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/SequencePacker.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/TruncatedBpttPacker.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/PackerBase.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/FramePacker.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/ReaderBase.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/Index.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/IndexBuilder.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/BufferedFileReader.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/DataDeserializerBase.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/ChunkCache.cpp \
	$(SOURCEDIR)/Readers/ReaderLib/ReaderUtil.cpp \

COMMON_SRC =\
	$(SOURCEDIR)/Common/Config.cpp \
	$(SOURCEDIR)/Common/Globals.cpp \
	$(SOURCEDIR)/Common/DataReader.cpp \
	$(SOURCEDIR)/Common/DataWriter.cpp \
	$(SOURCEDIR)/Common/ExceptionWithCallStack.cpp \
	$(SOURCEDIR)/Common/Eval.cpp \
	$(SOURCEDIR)/Common/File.cpp \
	$(SOURCEDIR)/Common/TimerUtility.cpp \
	$(SOURCEDIR)/Common/fileutil.cpp \
	$(SOURCEDIR)/Common/Sequences.cpp \
	$(SOURCEDIR)/Common/EnvironmentUtil.cpp \

MATH_SRC =\
	$(SOURCEDIR)/Math/BatchNormalizationEngine.cpp \
	$(SOURCEDIR)/Math/CUDAPageLockedMemAllocator.cpp \
	$(SOURCEDIR)/Math/CPUMatrixFloat.cpp \
	$(SOURCEDIR)/Math/CPUMatrixDouble.cpp \
	$(SOURCEDIR)/Math/CPURNGHandle.cpp \
	$(SOURCEDIR)/Math/CPUSparseMatrix.cpp \
	$(SOURCEDIR)/Math/ConvolutionEngine.cpp \
	$(SOURCEDIR)/Math/MatrixQuantizerImpl.cpp \
	$(SOURCEDIR)/Math/MatrixQuantizerCPU.cpp \
	$(SOURCEDIR)/Math/Matrix.cpp \
	$(SOURCEDIR)/Math/QuantizedMatrix.cpp \
	$(SOURCEDIR)/Math/DataTransferer.cpp \
	$(SOURCEDIR)/Math/RNGHandle.cpp \
	$(SOURCEDIR)/Math/TensorView.cpp \
	$(SOURCEDIR)/Math/NcclComm.cpp \

ifdef CUDA_PATH
MATH_SRC +=\
	$(SOURCEDIR)/Math/CuDnnBatchNormalization.cu \
	$(SOURCEDIR)/Math/CuDnnCommon.cu \
	$(SOURCEDIR)/Math/CuDnnConvolutionEngine.cu \
	$(SOURCEDIR)/Math/CuDnnRNN.cpp \
	$(SOURCEDIR)/Math/GPUDataTransferer.cpp \
	$(SOURCEDIR)/Math/GPUMatrix.cu \
	$(SOURCEDIR)/Math/GPUSparseMatrix.cu \
	$(SOURCEDIR)/Math/GPUTensor.cu \
	$(SOURCEDIR)/Math/GPUWatcher.cu \
	$(SOURCEDIR)/Math/GPURNGHandle.cu \
	$(SOURCEDIR)/Math/MatrixQuantizerGPU.cu \

else
MATH_SRC +=\
	$(SOURCEDIR)/Math/NoGPU.cpp

endif

MATH_SRC+=$(COMMON_SRC)
MATH_SRC+=$(READER_SRC)

MATH_OBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(patsubst %.cpp, $(OBJDIR)/%.o, $(MATH_SRC)))

CNTKMATH_LIB:= $(LIBDIR)/lib$(CNTKMATH).so
ALL_LIBS += $(CNTKMATH_LIB)
PYTHON_LIBS += $(CNTKMATH_LIB)
JAVA_LIBS += $(CNTKMATH_LIB)
SRC+=$(MATH_SRC)

$(CNTKMATH_LIB): $(MATH_OBJ) | $(PERF_PROFILER_LIB)
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBPATH) $(LIBDIR) $(GDK_NVML_LIB_PATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -fopenmp -l$(PERF_PROFILER)


# Any executable using Common or ReaderLib needs to link these libraries. 
READER_LIBS := $(CNTKMATH_LIB) $(PERF_PROFILER_LIB)
L_READER_LIBS := -l$(CNTKMATH) -l$(PERF_PROFILER)


########################################
# CNTKLibrary
########################################

CNTK_COMMON_SRC =\
	$(SOURCEDIR)/Common/BestGpu.cpp \
	$(SOURCEDIR)/Common/MPIWrapper.cpp \

COMPUTATION_NETWORK_LIB_SRC =\
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNode.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNodeScripting.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/InputAndParamNodes.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/RecurrentNodes.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/LinearAlgebraNodes.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ReshapingNodes.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/RNNNodes.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/SpecialPurposeNodes.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetwork.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetworkEvaluation.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetworkAnalysis.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetworkEditing.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetworkBuilder.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/ComputationNetworkScripting.cpp \
	$(SOURCEDIR)/ComputationNetworkLib/TrainingNodes.cpp \

SEQUENCE_TRAINING_LIB_SRC =\
	$(SOURCEDIR)/SequenceTrainingLib/latticeforwardbackward.cpp \
	$(SOURCEDIR)/SequenceTrainingLib/parallelforwardbackward.cpp \

ifdef CUDA_PATH
SEQUENCE_TRAINING_LIB_SRC +=\
	$(SOURCEDIR)/Math/cudalatticeops.cu \
	$(SOURCEDIR)/Math/cudalattice.cpp \
	$(SOURCEDIR)/Math/cudalib.cpp \

else
SEQUENCE_TRAINING_LIB_SRC +=\
	$(SOURCEDIR)/SequenceTrainingLib/latticeNoGPU.cpp \

endif

CNTKLIBRARY_COMMON_SRC =\
	$(SOURCEDIR)/CNTKv2LibraryDll/BackCompat.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/Common.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/Function.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/PrimitiveFunction.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/CompositeFunction.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/UserDefinedFunction.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/NDArrayView.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/NDMask.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/Trainer.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/Evaluator.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/Utils.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/Value.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/Variable.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/Learner.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/Serialization.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/DistributedCommunicator.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/DistributedLearnerBase.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/DataParallelDistributedLearner.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/ProgressWriter.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/CNTK.pb.cc \
	$(SOURCEDIR)/CNTKv2LibraryDll/tensorboard/tensorboard.pb.cc \
	$(SOURCEDIR)/CNTKv2LibraryDll/tensorboard/TensorBoardFileWriter.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/tensorboard/TensorBoardUtils.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/protobuf/graph.pb.cc \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/defs/experiments/defs.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/defs/generator/defs.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/defs/logical/defs.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/defs/math/defs.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/defs/nn/defs.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/defs/reduction/defs.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/defs/rnn/defs.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/defs/tensor/defs.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/core/constants.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/core/status.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/core/utils.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/core/opsignature.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/core/op.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/core/shape_inference.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/core/graph.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/core/model.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/Operators.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/CNTKToONNX.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/ONNXToCNTK.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/proto/onnx/ONNX.cpp \

CNTKLIBRARY_SRC =\
	$(SOURCEDIR)/CNTKv2LibraryDll/ComputeInputStatistics.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/MinibatchSource.cpp \
	$(SOURCEDIR)/CNTKv2LibraryDll/TrainingSession.cpp \

CNTKLIBRARY_SRC+=$(CNTKLIBRARY_COMMON_SRC)
CNTKLIBRARY_SRC+=$(CNTK_COMMON_SRC)
CNTKLIBRARY_SRC+=$(COMPUTATION_NETWORK_LIB_SRC)
CNTKLIBRARY_SRC+=$(SEQUENCE_TRAINING_LIB_SRC)

CNTKLIBRARY:=Cntk.Core-$(CNTK_COMPONENT_VERSION)

CNTKLIBRARY_OBJ:=\
	$(patsubst %.cu, $(OBJDIR)/%.o, $(filter %.cu, $(CNTKLIBRARY_SRC))) \
	$(patsubst %.pb.cc, $(OBJDIR)/%.pb.o, $(filter %.pb.cc, $(CNTKLIBRARY_SRC))) \
	$(patsubst %.cpp, $(OBJDIR)/%.o, $(filter %.cpp, $(CNTKLIBRARY_SRC)))

CNTKLIBRARY_LIB:=$(LIBDIR)/lib$(CNTKLIBRARY).so
ALL_LIBS+=$(CNTKLIBRARY_LIB)
PYTHON_LIBS+=$(CNTKLIBRARY_LIB)
JAVA_LIBS+=$(CNTKLIBRARY_LIB)
SRC+=$(CNTKLIBRARY_SRC)

OPENCV_LIBS:=-lopencv_core -lopencv_imgproc -lopencv_imgcodecs

$(CNTKLIBRARY_LIB): $(CNTKLIBRARY_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $@ for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH))  -o $@ $^ $(LIBS) $(OPENCV_LIBS) -l$(CNTKMATH) $(PROTOBUF_PATH)/lib/libprotobuf.a -ldl -fopenmp


########################################
# C++ extensibility examples library
########################################

CPP_EXTENSIBILITY_EXAMPLES_LIBRARY_SRC =\
	$(SOURCEDIR)/../Examples/Extensibility/CPPLib/CPPExtensibilityExamplesLibrary.cpp \

CPP_EXTENSIBILITY_EXAMPLES_LIBRARY_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(CPP_EXTENSIBILITY_EXAMPLES_LIBRARY_SRC))

CPP_EXTENSIBILITY_EXAMPLES_LIB:= $(LIBDIR)/Cntk.ExtensibilityExamples-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS += $(CPP_EXTENSIBILITY_EXAMPLES_LIB)
PYTHON_LIBS += $(CPP_EXTENSIBILITY_EXAMPLES_LIB)
SRC += $(CPP_EXTENSIBILITY_EXAMPLES_LIBRARY_SRC)

$(CPP_EXTENSIBILITY_EXAMPLES_LIB): $(CPP_EXTENSIBILITY_EXAMPLES_LIBRARY_OBJ) | $(CNTKLIBRARY_LIB)
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR)) $(patsubst %,$(RPATH)%, $(LIBDIR) $(ORIGINDIR)) -o $@ $^ -l$(CNTKLIBRARY)


##############################################
# Binary convolution example library
##############################################

BINARY_CONVOLUTION_EXAMPLE_LIBRARY_SRC =\
	$(SOURCEDIR)/../Examples/Extensibility/BinaryConvolution/BinaryConvolutionLib/BinaryConvolutionLib.cpp \

BINARY_CONVOLUTION_EXAMPLE_LIBRARY_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(BINARY_CONVOLUTION_EXAMPLE_LIBRARY_SRC))

BINARY_CONVOLUTION_EXAMPLE_LIB:= $(LIBDIR)/Cntk.BinaryConvolutionExample-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS += $(BINARY_CONVOLUTION_EXAMPLE_LIB)
PYTHON_LIBS += $(BINARY_CONVOLUTION_EXAMPLE_LIB)
SRC += $(BINARY_CONVOLUTION_EXAMPLE_LIBRARY_SRC)

$(BINARY_CONVOLUTION_EXAMPLE_LIB): $(BINARY_CONVOLUTION_EXAMPLE_LIBRARY_OBJ) | $(CNTKLIBRARY_LIB)
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR)) $(patsubst %,$(RPATH)%, $(LIBDIR) $(ORIGINDIR)) -o $@ $^ -l$(CNTKLIBRARY) $(SOURCEDIR)/../Examples/Extensibility/BinaryConvolution/BinaryConvolutionLib/halide/halide_convolve_nofeatures.a


##############################################
# Native implementation of the Proposal Layer
##############################################

PROPOSAL_LAYER_LIBRARY_SRC =\
	$(SOURCEDIR)/../Examples/Extensibility/ProposalLayer/ProposalLayerLib/ProposalLayerLib.cpp \

PROPOSAL_LAYER_LIBRARY_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(PROPOSAL_LAYER_LIBRARY_SRC))

PROPOSAL_LAYER_LIB:= $(LIBDIR)/Cntk.ProposalLayerLib-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS += $(PROPOSAL_LAYER_LIB)
PYTHON_LIBS += $(PROPOSAL_LAYER_LIB)
SRC += $(PROPOSAL_LAYER_LIBRARY_SRC)

$(PROPOSAL_LAYER_LIB): $(PROPOSAL_LAYER_LIBRARY_OBJ) | $(CNTKLIBRARY_LIB)
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(LIBDIR) $(LIBPATH) $(ORIGINDIR)) -o $@ $^ -l$(CNTKLIBRARY) $(OPENCV_LIBS)


########################################
# LibEval
########################################

EVAL:=Cntk.Eval-$(CNTK_COMPONENT_VERSION)

SGDLIB_SRC=\
	$(SOURCEDIR)/SGDLib/ASGDHelper.cpp \
	$(SOURCEDIR)/SGDLib/Profiler.cpp \
	$(SOURCEDIR)/SGDLib/SGD.cpp \
	$(SOURCEDIR)/SGDLib/PostComputingActions.cpp \

SGDLIB_SRC+=$(CNTKLIBRARY_COMMON_SRC)

EVAL_SRC=\
	$(SOURCEDIR)/EvalDll/CNTKEval.cpp \
	$(SOURCEDIR)/CNTK/BrainScript/BrainScriptEvaluator.cpp \
	$(SOURCEDIR)/CNTK/BrainScript/BrainScriptParser.cpp \
	$(SOURCEDIR)/CNTK/ModelEditLanguage.cpp \
	$(SOURCEDIR)/ActionsLib/EvalActions.cpp \
	$(SOURCEDIR)/ActionsLib/NetworkFactory.cpp \
	$(SOURCEDIR)/ActionsLib/NetworkDescriptionLanguage.cpp \
	$(SOURCEDIR)/ActionsLib/SimpleNetworkBuilder.cpp \
	$(SOURCEDIR)/ActionsLib/NDLNetworkBuilder.cpp \

EVAL_SRC+=$(SGDLIB_SRC)
EVAL_SRC+=$(COMPUTATION_NETWORK_LIB_SRC)
EVAL_SRC+=$(CNTK_COMMON_SRC)
EVAL_SRC+=$(SEQUENCE_TRAINING_LIB_SRC)

EVAL_OBJ:=\
	$(patsubst %.cu, $(OBJDIR)/%.o, $(filter %.cu, $(EVAL_SRC))) \
	$(patsubst %.pb.cc, $(OBJDIR)/%.pb.o, $(filter %.pb.cc, $(EVAL_SRC))) \
	$(patsubst %.cpp, $(OBJDIR)/%.o, $(filter %.cpp, $(EVAL_SRC)))

EVAL_LIB:=$(LIBDIR)/lib$(EVAL).so
ALL_LIBS+=$(EVAL_LIB)
SRC+=$(EVAL_SRC)

$(EVAL_LIB): $(EVAL_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo Building $(EVAL_LIB) for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -l$(CNTKMATH) -ldl $(lMULTIVERSO) $(PROTOBUF_PATH)/lib/libprotobuf.a

########################################
# Eval Sample clients
########################################
EVAL_CLIENT:=$(BINDIR)/cppevalclient

EVAL_CLIENT_SRC=\
	$(SOURCEDIR)/../Examples/Evaluation/LegacyEvalDll/CPPEvalClient/CPPEvalClient.cpp

EVAL_CLIENT_OBJ:=$(patsubst %.cpp, $(OBJDIR)/%.o, $(EVAL_CLIENT_SRC))

ALL+=$(EVAL_CLIENT)
SRC+=$(EVAL_CLIENT_SRC)

$(EVAL_CLIENT): $(EVAL_CLIENT_OBJ) | $(EVAL_LIB) $(READER_LIBS)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $(EVAL_CLIENT) for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH)) $(patsubst %,$(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -l$(EVAL) $(L_READER_LIBS) $(lMULTIVERSO) $(OPENCV_LIBS)

EVAL_EXTENDED_CLIENT:=$(BINDIR)/cppevalextendedclient

EVAL_EXTENDED_CLIENT_SRC=\
	$(SOURCEDIR)/../Examples/Evaluation/LegacyEvalDll/CPPEvalExtendedClient/CPPEvalExtendedClient.cpp

EVAL_EXTENDED_CLIENT_OBJ:=$(patsubst %.cpp, $(OBJDIR)/%.o, $(EVAL_EXTENDED_CLIENT_SRC))

ALL+=$(EVAL_EXTENDED_CLIENT)
SRC+=$(EVAL_EXTENDED_CLIENT_SRC)

$(EVAL_EXTENDED_CLIENT): $(EVAL_EXTENDED_CLIENT_OBJ) | $(EVAL_LIB) $(READER_LIBS)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $(EVAL_EXTENDED_CLIENT) for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH)) $(patsubst %,$(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -l$(EVAL) $(L_READER_LIBS) $(lMULTIVERSO) $(OPENCV_LIBS)

########################################
# Eval V2 Sample client
########################################
CNTKLIBRARY_CPP_EVAL_EXAMPLES:=$(BINDIR)/CNTKLibraryCPPEvalExamples

#ifdef CUDA_PATH
CNTKLIBRARY_CPP_EVAL_EXAMPLES_SRC=\
	$(SOURCEDIR)/../Examples/Evaluation/CNTKLibraryCPPEvalGPUExamples/CNTKLibraryCPPEvalGPUExamples.cpp\
	$(SOURCEDIR)/../Examples/Evaluation/CNTKLibraryCPPEvalCPUOnlyExamples/CNTKLibraryCPPEvalExamples.cpp

#else
CNTKLIBRARY_CPP_EVAL_EXAMPLES_SRC=\
	$(SOURCEDIR)/../Examples/Evaluation/CNTKLibraryCPPEvalCPUOnlyExamples/CNTKLibraryCPPEvalCPUOnlyExamples.cpp\
	$(SOURCEDIR)/../Examples/Evaluation/CNTKLibraryCPPEvalCPUOnlyExamples/CNTKLibraryCPPEvalExamples.cpp
#endif

CNTKLIBRARY_CPP_EVAL_EXAMPLES_OBJ:=$(patsubst %.cpp, $(OBJDIR)/%.o, $(CNTKLIBRARY_CPP_EVAL_EXAMPLES_SRC))

ALL+=$(CNTKLIBRARY_CPP_EVAL_EXAMPLES)
SRC+=$(CNTKLIBRARY_CPP_EVAL_EXAMPLES_SRC)

$(CNTKLIBRARY_CPP_EVAL_EXAMPLES): $(CNTKLIBRARY_CPP_EVAL_EXAMPLES_OBJ) | $(CNTKLIBRARY_LIB) $(READER_LIBS)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $(CNTKLIBRARY_CPP_EVAL_EXAMPLES) for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH)) $(patsubst %,$(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -l$(CNTKLIBRARY) $(L_READER_LIBS)

########################################
# Eval V2 Sample test 
########################################
CNTKLIBRARY_CPP_EVAL_TEST:=$(BINDIR)/CNTKLibraryCPPEvalExamplesTest

CNTKLIBRARY_CPP_EVAL_TEST_SRC=\
	$(SOURCEDIR)/../Examples/Evaluation/CNTKLibraryCPPEvalCPUOnlyExamples/CNTKLibraryCPPEvalExamples.cpp\
	$(SOURCEDIR)/../Tests/EndToEndTests/EvalClientTests/CNTKLibraryCPPEvalExamplesTest/CNTKLibraryCPPEvalExamplesTest.cpp\
	$(SOURCEDIR)/../Tests/EndToEndTests/EvalClientTests/CNTKLibraryCPPEvalExamplesTest/EvalMultithreads.cpp\
	$(SOURCEDIR)/../Tests/EndToEndTests/CNTKv2Library/Common/Common.cpp

CNTKLIBRARY_CPP_EVAL_TEST_OBJ:=$(patsubst %.cpp, $(OBJDIR)/%.o, $(CNTKLIBRARY_CPP_EVAL_TEST_SRC))

ALL+=$(CNTKLIBRARY_CPP_EVAL_TEST)
SRC+=$(CNTKLIBRARY_CPP_EVAL_TEST_SRC)

$(CNTKLIBRARY_CPP_EVAL_TEST): $(CNTKLIBRARY_CPP_EVAL_TEST_OBJ) | $(CNTKLIBRARY_LIB) $(READER_LIBS)
	@mkdir -p $(dir $@)
	@echo building $(CNTKLIBRARY_CPP_EVAL_TEST) for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH)) $(patsubst %,$(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -l$(CNTKLIBRARY) $(L_READER_LIBS)

########################################
# HTKMLFReader plugin
########################################

HTKMLFREADER_SRC =\
	$(SOURCEDIR)/Readers/HTKMLFReader/Exports.cpp \
	$(SOURCEDIR)/Readers/HTKMLFReader/DataWriterLocal.cpp \
	$(SOURCEDIR)/Readers/HTKMLFReader/HTKMLFReader.cpp \
	$(SOURCEDIR)/Readers/HTKMLFReader/HTKMLFWriter.cpp \

HTKMLFREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(HTKMLFREADER_SRC))

HTKMLFREADER:=$(LIBDIR)/Cntk.Reader.HTKMLF-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS+=$(HTKMLFREADER)
SRC+=$(HTKMLFREADER_SRC)

$(HTKMLFREADER): $(HTKMLFREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# CompositeDataReader plugin
########################################

COMPOSITEDATAREADER_SRC =\
	$(SOURCEDIR)/Readers/CompositeDataReader/CompositeDataReader.cpp \
	$(SOURCEDIR)/Readers/CompositeDataReader/Exports.cpp \

COMPOSITEDATAREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(COMPOSITEDATAREADER_SRC))

COMPOSITEDATAREADER:=$(LIBDIR)/Cntk.Composite-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS+=$(COMPOSITEDATAREADER)
PYTHON_LIBS+=$(COMPOSITEDATAREADER)
SRC+=$(COMPOSITEDATAREADER_SRC)

$(COMPOSITEDATAREADER): $(COMPOSITEDATAREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# HTKDeserializers plugin
########################################

HTKDESERIALIZERS_SRC =\
	$(SOURCEDIR)/Readers/HTKMLFReader/DataWriterLocal.cpp \
	$(SOURCEDIR)/Readers/HTKMLFReader/HTKMLFWriter.cpp \
	$(SOURCEDIR)/Readers/HTKDeserializers/ConfigHelper.cpp \
	$(SOURCEDIR)/Readers/HTKDeserializers/Exports.cpp \
	$(SOURCEDIR)/Readers/HTKDeserializers/HTKDeserializer.cpp \
	$(SOURCEDIR)/Readers/HTKDeserializers/HTKMLFReader.cpp \
	$(SOURCEDIR)/Readers/HTKDeserializers/MLFDeserializer.cpp \
	$(SOURCEDIR)/Readers/HTKDeserializers/MLFIndexBuilder.cpp \
	$(SOURCEDIR)/Readers/HTKDeserializers/MLFUtils.cpp \

HTKDESERIALIZERS_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(HTKDESERIALIZERS_SRC))

HTKDESERIALIZERS:=$(LIBDIR)/Cntk.Deserializers.HTK-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS+=$(HTKDESERIALIZERS)
PYTHON_LIBS+=$(HTKDESERIALIZERS)
SRC+=$(HTKDESERIALIZERS_SRC)

$(HTKDESERIALIZERS): $(HTKDESERIALIZERS_OBJ) | $(CNTKMATH_LIB)
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

LMSEQUENCEREADER:= $(LIBDIR)/Cntk.Reader.LMSequence-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS+=$(LMSEQUENCEREADER)
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

LUSEQUENCEREADER:=$(LIBDIR)/Cntk.Reader.LUSequence-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS+=$(LUSEQUENCEREADER)
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

UCIFASTREADER:=$(LIBDIR)/Cntk.Reader.UCIFast-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS += $(UCIFASTREADER)
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

LIBSVMBINARYREADER:=$(LIBDIR)/Cntk.Reader.SVMBinary-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS += $(LIBSVMBINARYREADER)
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

SPARSEPCREADER:=$(LIBDIR)/Cntk.Reader.SparsePC-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS += $(SPARSEPCREADER)
SRC+=$(SPARSEPCREADER_SRC)

$(SPARSEPCREADER): $(SPARSEPCREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)

########################################
# CNTKBinaryReader plugin
########################################

CNTKBINARYREADER_SRC =\
	$(SOURCEDIR)/Readers/CNTKBinaryReader/Exports.cpp \
	$(SOURCEDIR)/Readers/CNTKBinaryReader/BinaryChunkDeserializer.cpp \
	$(SOURCEDIR)/Readers/CNTKBinaryReader/BinaryConfigHelper.cpp \
	$(SOURCEDIR)/Readers/CNTKBinaryReader/CNTKBinaryReader.cpp \

CNTKBINARYREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(CNTKBINARYREADER_SRC))

CNTKBINARYREADER:=$(LIBDIR)/Cntk.Deserializers.Binary-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS += $(CNTKBINARYREADER)
PYTHON_LIBS += $(CNTKBINARYREADER)
SRC+=$(CNTKBINARYREADER_SRC)

$(CNTKBINARYREADER): $(CNTKBINARYREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH)


########################################
# CNTKTextFormatReader plugin
########################################

CNTKTEXTFORMATREADER_SRC =\
	$(SOURCEDIR)/Readers/CNTKTextFormatReader/Exports.cpp \
	$(SOURCEDIR)/Readers/CNTKTextFormatReader/TextParser.cpp \
	$(SOURCEDIR)/Readers/CNTKTextFormatReader/CNTKTextFormatReader.cpp \
	$(SOURCEDIR)/Readers/CNTKTextFormatReader/TextConfigHelper.cpp \

CNTKTEXTFORMATREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(CNTKTEXTFORMATREADER_SRC))

CNTKTEXTFORMATREADER:=$(LIBDIR)/Cntk.Deserializers.TextFormat-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS += $(CNTKTEXTFORMATREADER)
PYTHON_LIBS += $(CNTKTEXTFORMATREADER)
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

KALDI2READER:=$(LIBDIR)/Cntk.Reader.Kaldi2-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS+=$(KALDI2READER)
SRC+=$(KALDI2READER_SRC)

$(KALDI2READER): $(KALDI2READER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(KALDI_LIBPATH) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(KALDI_LIBPATH) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH) $(KALDI_LIBS)

endif

########################################
# ImageReader plugin
########################################

ifdef OPENCV_PATH
ifdef BOOST_PATH

INCLUDEPATH += $(BOOST_PATH)/include

IMAGEREADER_LIBS_LIST := opencv_core opencv_imgproc opencv_imgcodecs

ifdef LIBZIP_PATH
  CPPFLAGS += -DUSE_ZIP
  # Both directories are needed for building libzip
  INCLUDEPATH += $(LIBZIP_PATH)/include $(LIBZIP_PATH)/lib/libzip/include
  LIBPATH += $(LIBZIP_PATH)/lib
  IMAGEREADER_LIBS_LIST += zip
endif

IMAGEREADER_LIBS:= $(addprefix -l,$(IMAGEREADER_LIBS_LIST))

IMAGEREADER_SRC =\
  $(SOURCEDIR)/Readers/ImageReader/Base64ImageDeserializer.cpp \
  $(SOURCEDIR)/Readers/ImageReader/ImageDeserializerBase.cpp \
  $(SOURCEDIR)/Readers/ImageReader/Exports.cpp \
  $(SOURCEDIR)/Readers/ImageReader/ImageConfigHelper.cpp \
  $(SOURCEDIR)/Readers/ImageReader/ImageDataDeserializer.cpp \
  $(SOURCEDIR)/Readers/ImageReader/ImageTransformers.cpp \
  $(SOURCEDIR)/Readers/ImageReader/ImageReader.cpp \
  $(SOURCEDIR)/Readers/ImageReader/ZipByteReader.cpp \

IMAGEREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(IMAGEREADER_SRC))

IMAGEREADER:=$(LIBDIR)/Cntk.Deserializers.Image-$(CNTK_COMPONENT_VERSION).so
ALL_LIBS += $(IMAGEREADER)
PYTHON_LIBS += $(IMAGEREADER)
SRC+=$(IMAGEREADER_SRC)

INCLUDEPATH += $(OPENCV_PATH)/include
LIBPATH += $(OPENCV_PATH)/lib $(OPENCV_PATH)/release/lib

$(IMAGEREADER): $(IMAGEREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CXX) $(LDFLAGS) -shared $(patsubst %,-L%, $(LIBDIR) $(LIBPATH)) $(patsubst %,$(RPATH)%, $(ORIGINDIR) $(LIBPATH)) -o $@ $^ -l$(CNTKMATH) $(IMAGEREADER_LIBS)
endif
endif

########################################
# 1bit SGD setup
########################################

ifeq ("$(CNTK_ENABLE_1BitSGD)","true")

ifeq (,$(wildcard Source/1BitSGD/*.h))
  $(error Build with 1bit-SGD was requested but cannot find the code. Please check https://docs.microsoft.com/en-us/cognitive-toolkit/Enabling-1bit-SGD for instructions)
endif

  INCLUDEPATH += $(SOURCEDIR)/1BitSGD 

  COMMON_FLAGS += -DCNTK_PARALLEL_TRAINING_SUPPORT
  # temporarily adding to 1bit, need to work with others to fix it
endif

 
########################################
# ASGD(multiverso) setup
########################################


ifeq ("$(CNTK_ENABLE_ASGD)","true")

ifeq (,$(wildcard Source/Multiverso/include/multiverso/*.h))
  $(error Build with Multiverso was requested but cannot find the code. Please check https://docs.microsoft.com/en-us/cognitive-toolkit/Multiple-GPUs-and-machines#8-data-parallel-training-with-parameter-server to learn more.)
endif

lMULTIVERSO:=-lmultiverso

INCLUDEPATH += $(SOURCEDIR)/Multiverso/include
COMMON_FLAGS += -DASGD_PARALLEL_SUPPORT

MULTIVERSO_LIB:=$(LIBDIR)/libmultiverso.so

ALL_LIBS+=$(MULTIVERSO_LIB)
ifeq ("$(BUILDTYPE)","release")
MULTIVERSO_CMAKE_BUILDTYPE=Release
endif
ifeq ("$(BUILDTYPE)","debug")
MULTIVERSO_CMAKE_BUILDTYPE=Debug
endif

# TODO need to align Multiverso OpenMP with the one we use (libiomp). For now, disabled.
$(MULTIVERSO_LIB): 
	@echo "Build Multiverso lib"
	@mkdir -p $(LIBDIR)
	@mkdir -p $(BINDIR)
	@mkdir -p $(SOURCEDIR)/Multiverso/build/$(BUILDTYPE)
	@cmake -DCMAKE_VERBOSE_MAKEFILE=TRUE \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DOpenMP_CXX_FLAGS="" \
		-DOpenMP_C_FLAGS="" \
		-DBoost_NO_BOOST_CMAKE=TRUE \
		-DBoost_NO_SYSTEM_PATHS=TRUE \
		-DBOOST_ROOT:PATHNAME=$(BOOST_PATH) \
		-DBOOST_LIBRARY_DIRS:FILEPATH=$(BOOST_PATH) \
		-DLIBRARY_OUTPUT_PATH=$(shell readlink -f $(LIBDIR)) \
		-DEXECUTABLE_OUTPUT_PATH=$(shell readlink -f $(BINDIR)) \
		-DCMAKE_BUILD_TYPE=$(MULTIVERSO_CMAKE_BUILDTYPE) \
		-B./Source/Multiverso/build/$(BUILDTYPE) -H./Source/Multiverso
	@make VERBOSE=1 -C ./Source/Multiverso/build/$(BUILDTYPE) -j multiverso

UNITTEST_MULTIVERSO_SRC = \
	$(SOURCEDIR)/Multiverso/Test/unittests/test_array.cpp \
	$(SOURCEDIR)/Multiverso/Test/unittests/test_blob.cpp \
	$(SOURCEDIR)/Multiverso/Test/unittests/test_kv.cpp \
	$(SOURCEDIR)/Multiverso/Test/unittests/test_message.cpp \
	$(SOURCEDIR)/Multiverso/Test/unittests/test_multiverso.cpp \
	$(SOURCEDIR)/Multiverso/Test/unittests/test_node.cpp \
	$(SOURCEDIR)/Multiverso/Test/unittests/test_sync.cpp \

UNITTEST_MULTIVERSO_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(UNITTEST_MULTIVERSO_SRC))

UNITTEST_MULTIVERSO := $(BINDIR)/multiversotests

ALL += $(UNITTEST_MULTIVERSO)

$(UNITTEST_MULTIVERSO): $(UNITTEST_MULTIVERSO_OBJ) | $(MULTIVERSO_LIB)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $@ for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(BOOSTLIB_PATH)) $(patsubst %, $(RPATH)%, $(ORIGINLIBDIR) $(BOOSTLIB_PATH)) -o $@ $^ $(BOOSTLIBS) $(lMULTIVERSO) -ldl
endif

########################################
# cntk
########################################

CNTK_SRC =\
	$(SOURCEDIR)/CNTK/CNTK.cpp \
	$(SOURCEDIR)/CNTK/ModelEditLanguage.cpp \
	$(SOURCEDIR)/CNTK/tests.cpp \
	$(SOURCEDIR)/ActionsLib/TrainActions.cpp \
	$(SOURCEDIR)/ActionsLib/EvalActions.cpp \
	$(SOURCEDIR)/ActionsLib/OtherActions.cpp \
	$(SOURCEDIR)/ActionsLib/SpecialPurposeActions.cpp \
	$(SOURCEDIR)/ActionsLib/NetworkFactory.cpp \
	$(SOURCEDIR)/ActionsLib/NetworkDescriptionLanguage.cpp \
	$(SOURCEDIR)/ActionsLib/SimpleNetworkBuilder.cpp \
	$(SOURCEDIR)/ActionsLib/NDLNetworkBuilder.cpp \
	$(SOURCEDIR)/CNTK/BrainScript/BrainScriptEvaluator.cpp \
	$(SOURCEDIR)/CNTK/BrainScript/BrainScriptParser.cpp \

CNTK_SRC+=$(SGDLIB_SRC)
CNTK_SRC+=$(CNTK_COMMON_SRC)
CNTK_SRC+=$(COMPUTATION_NETWORK_LIB_SRC)
CNTK_SRC+=$(SEQUENCE_TRAINING_LIB_SRC)

CNTK_OBJ :=\
	$(patsubst %.cu, $(OBJDIR)/%.o, $(filter %.cu, $(CNTK_SRC))) \
	$(patsubst %.pb.cc, $(OBJDIR)/%.pb.o, $(filter %.pb.cc, $(CNTK_SRC))) \
	$(patsubst %.cpp, $(OBJDIR)/%.o, $(filter %.cpp, $(CNTK_SRC)))

CNTK:=$(BINDIR)/cntk
ALL+=$(CNTK)
SRC+=$(CNTK_SRC)

$(CNTK): $(CNTK_OBJ) | $(READER_LIBS) $(MULTIVERSO_LIB)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $@ for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH)) $(patsubst %,$(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH)) -o $@ $^ $(LIBS) $(L_READER_LIBS) $(lMULTIVERSO) -ldl -fopenmp $(PROTOBUF_PATH)/lib/libprotobuf.a $(OPENCV_LIBS)

# deployable resources: standard library of BS
CNTK_CORE_BS:=$(BINDIR)/cntk.core.bs
ALL += $(CNTK_CORE_BS)
$(CNTK_CORE_BS): $(SOURCEDIR)/CNTK/BrainScript/CNTKCoreLib/CNTK.core.bs
	@mkdir -p $(dir $@)
	@echo bin-placing deployable resource files
	cp -f $^ $@

########################################
# V2Library EndToEndTests
########################################
CNTKLIBRARY_END_TO_END_TESTS_PATH =\
	Tests/EndToEndTests/CNTKv2Library

CNTKLIBRARY_END_TO_END_COMMON_SRC_PATH =\
	$(CNTKLIBRARY_END_TO_END_TESTS_PATH)/Common

INCLUDEPATH+=$(CNTKLIBRARY_END_TO_END_COMMON_SRC_PATH)

CNTKLIBRARY_END_TO_END_TESTS_SRC_PATH =\
	$(CNTKLIBRARY_END_TO_END_TESTS_PATH)/EndToEndTests

CNTKLIBRARY_END_TO_END_TESTS_SRC =\
	$(CNTKLIBRARY_END_TO_END_COMMON_SRC_PATH)/Common.cpp \
	$(CNTKLIBRARY_END_TO_END_TESTS_SRC_PATH)/Main.cpp \
	$(CNTKLIBRARY_END_TO_END_TESTS_SRC_PATH)/CifarResNet.cpp \
	$(CNTKLIBRARY_END_TO_END_TESTS_SRC_PATH)/MNISTClassifier.cpp \
	$(CNTKLIBRARY_END_TO_END_TESTS_SRC_PATH)/Seq2Seq.cpp \
	$(CNTKLIBRARY_END_TO_END_TESTS_SRC_PATH)/SequenceClassification.cpp \
	$(CNTKLIBRARY_END_TO_END_TESTS_SRC_PATH)/TruncatedLSTMAcousticModel.cpp \
	$(CNTKLIBRARY_END_TO_END_TESTS_SRC_PATH)/FrameMode.cpp \

CNTKLIBRARY_END_TO_END_TESTS:=$(BINDIR)/V2LibraryEndToEndTests
CNTKLIBRARY_END_TO_END_TESTS_OBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(patsubst %.cpp, $(OBJDIR)/%.o, $(CNTKLIBRARY_END_TO_END_TESTS_SRC)))

ALL+=$(CNTKLIBRARY_END_TO_END_TESTS)
SRC+=$(CNTKLIBRARY_END_TO_END_TESTS_SRC)

$(CNTKLIBRARY_END_TO_END_TESTS): $(CNTKLIBRARY_END_TO_END_TESTS_OBJ) | $(CNTKLIBRARY_LIB) $(READER_LIBS)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $@ for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH)) $(patsubst %,$(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH)) -o $@ $^ $(LIBS) -l$(CNTKLIBRARY) $(L_READER_LIBS)

########################################
# Unit Tests
########################################

# only build unit tests when Boost is available
ifdef BOOST_PATH

INCLUDEPATH += $(BOOST_PATH)/include

BOOSTLIB_PATH = $(BOOST_PATH)/lib
BOOSTLIBS := -lboost_unit_test_framework -lboost_filesystem -lboost_system

UNITTEST_EVAL_SRC = \
	$(SOURCEDIR)/../Tests/UnitTests/EvalTests/EvalExtendedTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/EvalTests/stdafx.cpp

UNITTEST_EVAL_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(UNITTEST_EVAL_SRC))

UNITTEST_EVAL := $(BINDIR)/evaltests

ALL += $(UNITTEST_EVAL)
SRC += $(UNITTEST_EVAL_SRC)

$(UNITTEST_EVAL) : $(UNITTEST_EVAL_OBJ) | $(EVAL_LIB) $(READER_LIBS)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $@ for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH) $(BOOSTLIB_PATH)) $(patsubst %, $(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH) $(BOOSTLIB_PATH)) -o $@ $^ $(BOOSTLIBS) $(LIBS) -l$(EVAL) $(L_READER_LIBS) $(lMULTIVERSO) $(OPENCV_LIBS) 

#TODO: create project specific makefile or rules to avoid adding project specific path to the global path
INCLUDEPATH += $(SOURCEDIR)/Readers/CNTKTextFormatReader

UNITTEST_READER_SRC = \
	$(SOURCEDIR)/../Tests/UnitTests/ReaderTests/CNTKBinaryReaderTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/ReaderTests/CNTKTextFormatReaderTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/ReaderTests/HTKLMFReaderTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/ReaderTests/ImageReaderTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/ReaderTests/ReaderLibTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/ReaderTests/ReaderUtilTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/ReaderTests/stdafx.cpp \
	$(SOURCEDIR)/Readers/CNTKTextFormatReader/TextParser.cpp \

UNITTEST_READER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(UNITTEST_READER_SRC))

UNITTEST_READER := $(BINDIR)/readertests

ALL += $(UNITTEST_READER)
SRC += $(UNITTEST_READER_SRC)

$(UNITTEST_READER): $(UNITTEST_READER_OBJ) | $(HTKMLFREADER) $(HTKDESERIALIZERS) $(UCIFASTREADER) $(COMPOSITEDATAREADER) $(IMAGEREADER) $(READER_LIBS)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $@ for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(BOOSTLIB_PATH)) $(patsubst %, $(RPATH)%, $(ORIGINLIBDIR) $(BOOSTLIB_PATH)) -o $@ $^ $(BOOSTLIBS) $(L_READER_LIBS) -ldl -fopenmp

UNITTEST_NETWORK_SRC = \
	$(SOURCEDIR)/../Tests/UnitTests/NetworkTests/AccumulatorNodeTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/NetworkTests/BatchNormalizationTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/NetworkTests/CropNodeTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/NetworkTests/OperatorEvaluation.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/NetworkTests/stdafx.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/NetworkTests/TestHelpers.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/NetworkTests/EditDistanceTests.cpp \
	$(SOURCEDIR)/CNTK/ModelEditLanguage.cpp \
	$(SOURCEDIR)/ActionsLib/TrainActions.cpp \
	$(SOURCEDIR)/ActionsLib/EvalActions.cpp \
	$(SOURCEDIR)/ActionsLib/OtherActions.cpp \
	$(SOURCEDIR)/ActionsLib/SpecialPurposeActions.cpp \
	$(SOURCEDIR)/ActionsLib/NetworkFactory.cpp \
	$(SOURCEDIR)/ActionsLib/NetworkDescriptionLanguage.cpp \
	$(SOURCEDIR)/ActionsLib/SimpleNetworkBuilder.cpp \
	$(SOURCEDIR)/ActionsLib/NDLNetworkBuilder.cpp \
	$(SOURCEDIR)/CNTK/BrainScript/BrainScriptEvaluator.cpp \
	$(SOURCEDIR)/CNTK/BrainScript/BrainScriptParser.cpp \

UNITTEST_NETWORK_SRC += $(COMPUTATION_NETWORK_LIB_SRC)
UNITTEST_NETWORK_SRC += $(CNTK_COMMON_SRC)
UNITTEST_NETWORK_SRC += $(SEQUENCE_TRAINING_LIB_SRC)
UNITTEST_NETWORK_SRC += $(SGDLIB_SRC)

UNITTEST_NETWORK_OBJ :=\
	$(patsubst %.cu, $(OBJDIR)/%.o, $(filter %.cu, $(UNITTEST_NETWORK_SRC))) \
	$(patsubst %.pb.cc, $(OBJDIR)/%.pb.o, $(filter %.pb.cc, $(UNITTEST_NETWORK_SRC))) \
	$(patsubst %.cpp, $(OBJDIR)/%.o, $(filter %.cpp, $(UNITTEST_NETWORK_SRC)))

UNITTEST_NETWORK := $(BINDIR)/networktests

ALL += $(UNITTEST_NETWORK)
SRC += $(UNITTEST_NETWORK_SRC)

$(UNITTEST_NETWORK): $(UNITTEST_NETWORK_OBJ) | $(READER_LIBS) $(CNTKTEXTFORMATREADER) $(MULTIVERSO_LIB)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $@ for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH) $(BOOSTLIB_PATH)) $(patsubst %, $(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH) $(BOOSTLIB_PATH)) -o $@ $^ $(BOOSTLIBS) $(LIBS) $(lMULTIVERSO) $(L_READER_LIBS) -ldl -fopenmp  $(PROTOBUF_PATH)/lib/libprotobuf.a $(OPENCV_LIBS)  

UNITTEST_MATH_SRC = \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/BatchNormalizationEngineTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/constants.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/ConvolutionEngineTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/CPUMatrixTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/CPUSparseMatrixTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/fixtures.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/QuantizersTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/QuantizedOperationsTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/TensorTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/GPUMatrixCudaBlasTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/GPUMatrixTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/GPUSparseMatrixTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/MatrixBlasTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/MatrixDataSynchronizationTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/MatrixFileWriteReadTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/MatrixQuantizerTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/MatrixSparseDenseInteractionsTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/MatrixTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/MatrixLearnerTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/MathTests/stdafx.cpp \

UNITTEST_MATH_SRC += $(CNTK_COMMON_SRC)
UNITTEST_MATH_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(UNITTEST_MATH_SRC))

UNITTEST_MATH := $(BINDIR)/mathtests

ALL += $(UNITTEST_MATH)
SRC += $(UNITTEST_MATH_SRC)

$(UNITTEST_MATH): $(UNITTEST_MATH_OBJ) | $(READER_LIBS)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $@ for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH) $(BOOSTLIB_PATH)) $(patsubst %, $(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH) $(BOOSTLIB_PATH)) -o $@ $^ $(BOOSTLIBS) $(LIBS)  $(L_READER_LIBS) -ldl -fopenmp

UNITTEST_BRAINSCRIPT_SRC = \
	$(SOURCEDIR)/CNTK/BrainScript/BrainScriptEvaluator.cpp \
	$(SOURCEDIR)/CNTK/BrainScript/BrainScriptParser.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/BrainScriptTests/ParserTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/BrainScriptTests/ComputationNetworkTests.cpp \
	$(SOURCEDIR)/../Tests/UnitTests/BrainScriptTests/stdafx.cpp

UNITTEST_BRAINSCRIPT_SRC += $(COMPUTATION_NETWORK_LIB_SRC)
UNITTEST_BRAINSCRIPT_SRC += $(SEQUENCE_TRAINING_LIB_SRC)

UNITTEST_BRAINSCRIPT_OBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(patsubst %.cpp, $(OBJDIR)/%.o, $(UNITTEST_BRAINSCRIPT_SRC)))

UNITTEST_BRAINSCRIPT := $(BINDIR)/brainscripttests

ALL += $(UNITTEST_BRAINSCRIPT)
SRC += $(UNITTEST_BRAINSCRIPT_SRC)

$(UNITTEST_BRAINSCRIPT): $(UNITTEST_BRAINSCRIPT_OBJ) | $(READER_LIBS)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $@ for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH) $(BOOSTLIB_PATH)) $(patsubst %, $(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH) $(BOOSTLIB_PATH)) -o $@ $^ $(BOOSTLIBS) $(LIBS) -ldl $(L_READER_LIBS) -fopenmp

########################################
# CNTKLibrary tests
########################################
CNTKLIBRARY_TESTS_SRC_PATH =\
	Tests/UnitTests/V2LibraryTests

CNTKLIBRARY_TESTS_SRC =\
	$(CNTKLIBRARY_END_TO_END_COMMON_SRC_PATH)/Common.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/FeedForwardTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/NDArrayViewTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/RecurrentFunctionTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/BlockTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/TensorTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/ValueTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/SerializationTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/LearnerTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/FunctionTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/DeviceSelectionTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/MinibatchSourceTest.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/UserDefinedFunctionTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/LoadLegacyModelTests.cpp \
	$(CNTKLIBRARY_TESTS_SRC_PATH)/stdafx.cpp

CNTKLIBRARY_TESTS := $(BINDIR)/v2librarytests
CNTKLIBRARY_TESTS_OBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(patsubst %.cpp, $(OBJDIR)/%.o, $(CNTKLIBRARY_TESTS_SRC)))

ALL += $(CNTKLIBRARY_TESTS)
SRC += $(CNTKLIBRARY_TESTS_SRC)

$(CNTKLIBRARY_TESTS): $(CNTKLIBRARY_TESTS_OBJ) | $(CNTKLIBRARY_LIB) $(READER_LIBS)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building $@ for $(ARCH) with build type $(BUILDTYPE)
	$(CXX) $(LDFLAGS) $(patsubst %,-L%, $(LIBDIR) $(LIBPATH) $(GDK_NVML_LIB_PATH) $(BOOSTLIB_PATH)) $(patsubst %, $(RPATH)%, $(ORIGINLIBDIR) $(LIBPATH) $(BOOSTLIB_PATH)) -o $@ $^ $(BOOSTLIBS) $(LIBS) -ldl -l$(CNTKLIBRARY) $(L_READER_LIBS)

unittests: $(UNITTEST_EVAL) $(UNITTEST_READER) $(UNITTEST_NETWORK) $(UNITTEST_MATH) $(UNITTEST_BRAINSCRIPT) $(CNTKLIBRARY_TESTS)

endif

ifeq ("$(PYTHON_SUPPORT)","true")

# Libraries needed for the run-time (i.e., excluding test binaries)
# TODO MPI doesn't appear explicitly here, hidden by mpic++ usage (but currently, it should be user installed)
PYTHON_LIBS_LIST := $(LIBS_LIST) $(IMAGEREADER_LIBS_LIST)
PYTHON_LIBS_EXCLUDE_LIST := m pthread nvidia-ml
PYTHON_EXTRA_LIBS_BASENAMES:=$(addsuffix .so,$(addprefix lib,$(filter-out $(PYTHON_LIBS_EXCLUDE_LIST),$(PYTHON_LIBS_LIST))))

# TODO dependencies
# TODO intermediate build results should go below $OBJDIR
.PHONY: python
python: $(PYTHON_LIBS)
	@bash -c '\
            set -x -e; \
            declare -A py_paths; \
            py_paths[27]=$(PYTHON27_PATH); \
            py_paths[34]=$(PYTHON34_PATH); \
            py_paths[35]=$(PYTHON35_PATH); \
            py_paths[36]=$(PYTHON36_PATH); \
            export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$$(echo $(GDK_NVML_LIB_PATH) $(LIBPATH) | tr " " :); \
            ldd $$(find $(LIBDIR) -maxdepth 1 -type f -print) | grep "not found" && false; \
            export CNTK_COMPONENT_VERSION=$(CNTK_COMPONENT_VERSION); \
            export CNTK_LIBRARIES="$(PYTHON_LIBS)"; \
            export CNTK_EXTRA_LIBRARIES=$$(ldd $(LIBDIR)/* | grep "^\s.*=> " | cut -d ">" -f 2- --only-delimited | cut -d "(" -f 1 --only-delimited | sort -u | grep -Ff <(echo $(PYTHON_EXTRA_LIBS_BASENAMES) | xargs -n1)); \
            test -x $(SWIG_PATH); \
            export CNTK_LIB_PATH=$$(readlink -f $(LIBDIR)); \
            PYTHONDIR=$$(readlink -f $(PYTHONDIR)); \
            test $$? -eq 0; \
            cd bindings/python; \
            export PATH=$(SWIG_PATH):$$PATH; \
            for ver in $(PYTHON_VERSIONS); \
            do \
                test -x $${py_paths[$$ver]}; \
                $${py_paths[$$ver]} setup.py \
                    build_ext --inplace \
                    bdist_wheel \
                        --dist-dir $$PYTHONDIR || exit $$?; \
            done'

ALL += python

endif

ifeq ("$(JAVA_SUPPORT)","true")

BINDINGS_DIR=bindings
JAVA_SWIG_DIR=$(BINDINGS_DIR)/java/Swig
JAVA_TEST_DIR=Tests/EndToEndTests/EvalClientTests/JavaEvalTest
GENERATED_JAVA_DIR=$(JAVA_SWIG_DIR)/com/microsoft/CNTK
JDK_BIN_PATH=$(JDK_PATH)/bin
JDK_INCLUDE_PATH:=$(JDK_PATH)/include
JDK_INCLUDE_PATH+=$(JDK_INCLUDE_PATH)/linux

.PHONY: java
java: $(JAVA_LIBS)
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE)
	mkdir -p $(GENERATED_JAVA_DIR)
	rm -f $(GENERATED_JAVA_DIR)/*.java $(GENERATED_JAVA_DIR)/*.class
	$(SWIG_PATH)/swig -c++ -java -package com.microsoft.CNTK $(INCLUDEPATH:%=-I%) -I$(BINDINGS_DIR)/common -outdir $(GENERATED_JAVA_DIR) $(JAVA_SWIG_DIR)/cntk_java.i
	$(JDK_BIN_PATH)/javac $(GENERATED_JAVA_DIR)/*.java
	mkdir -p $(LIBDIR)/java
	cd $(JAVA_SWIG_DIR) && $(JDK_BIN_PATH)/jar -cvf cntk.jar com
	cp $(JAVA_SWIG_DIR)/cntk.jar $(LIBDIR)/java
	javac -cp $(JAVA_SWIG_DIR) $(JAVA_TEST_DIR)/src/Main.java -d $(LIBDIR)/java
	$(CXX) $(LDFLAGS) -shared $(COMMON_FLAGS) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDEPATH:%=-I%) $(JDK_INCLUDE_PATH:%=-I%) $(patsubst %,$(RPATH)%, $(ORIGINDIR)) -L$(LIBDIR) $(JAVA_SWIG_DIR)/cntk_java_wrap.cxx -l$(CNTKMATH) -l$(CNTKLIBRARY) -o $(LIBDIR)/libCntk.Core.JavaBinding-$(CNTK_COMPONENT_VERSION).so

ALL += java

endif

########################################
# General compile and dependency rules
########################################

ALL += $(ALL_LIBS)

VPATH := $(sort $(dir $(SRC)))

# Define object files
OBJ := \
	$(patsubst %.cu, $(OBJDIR)/%.o, $(filter %.cu, $(SRC))) \
	$(patsubst %.pb.cc, $(OBJDIR)/%.pb.o, $(filter %.pb.cc, $(SRC))) \
	$(patsubst %.cpp, $(OBJDIR)/%.o, $(filter %.cpp, $(SRC)))

# C++ include dependencies generated by -MF compiler option
DEP := $(patsubst %.o, %.d, $(OBJ))

# Include all C++ dependencies, like header files, to ensure that a change in those
# will result in the rebuild.
-include ${DEP}

BUILD_CONFIGURATION := Makefile $(BUILD_TOP)/Config.make

%.pb.cc : %.proto $(BUILD_CONFIGURATION)
	@echo $(SEPARATOR)
	@echo compiling protobuf $<
	$(PROTOC) --proto_path=$(dir $<) --cpp_out=$(dir $<) $<

$(OBJDIR)/%.o : %.cu $(BUILD_CONFIGURATION)
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE)
	@mkdir -p $(dir $@)
	$(NVCC) -c $< -o $@ $(COMMON_FLAGS) $(CUFLAGS) $(INCLUDEPATH:%=-I%) -Xcompiler "-fPIC -Werror"

$(OBJDIR)/%.pb.o : %.pb.cc $(BUILD_CONFIGURATION) 
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE)
	@mkdir -p $(dir $@)
	$(CXX) -c $< -o $@ $(COMMON_FLAGS) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDEPATH:%=-I%) -MD -MP -MF ${@:.o=.d}

$(OBJDIR)/%.o : %.cpp $(BUILD_CONFIGURATION) 
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE)
	@mkdir -p $(dir $@)
	$(CXX) -c $< -o $@ $(COMMON_FLAGS) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDEPATH:%=-I%) -MD -MP -MF ${@:.o=.d}

.PHONY: clean buildall all unittests

clean:
	@echo $(SEPARATOR)
	@rm -rf $(OBJDIR)
	@rm -rf $(ALL)
	@rm -rf $(BUILDINFO)
	@echo finished cleaning up the project

buildall : $(ALL)
	@echo $(SEPARATOR)
	@echo finished building for $(ARCH) with build type $(BUILDTYPE)
