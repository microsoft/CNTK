# Makefile for a Linux/GCC build of CNTK
#
# The Linux and Windows versions are not different branches, but rather build off the same
# source files, using different makefiles. This current makefile has the purpose of enabling
# work to make all sources compile with GCC, and also to check for GCC-compat regressions due to
# modifications which are currently done under Windows.
#
# This makefile will be extended/completed as we go.
#
# You will need to modify PATH and LD_LIBRARY_PATH environment variables to run CNTK
# export LD_LIBRARY_PATH=<path_to_math_lib>/ifort64/lib:<path_to_cuda>/lib64:/usr/local/lib
# export PATH=$PATH:/usr/local/bin:<path_to_cuda>/bin
#
# In order to deviate from the default settings in this Makefile, please specify options on
# the make command line, like this, for example (to build release):
#
# make BUILDTYPE=release -j

CC = g++
NVCC = nvcc
ARCH = x86_64

# DEVICE can also be cpu
DEVICE = gpu

# BUILDTYPE can also be release
BUILDTYPE = debug

# MATHLIB can also be mkl
MATHLIB = acml

# This is a suggested/default location for ACML library
MATHLIB_PATH = /usr/local/acml5.3.1/ifort64

# This is a suggested/default location for CUDA
CUDA_PATH = /usr/local/cuda-7.0

# This is a suggested/default location for NVML
NVML_INCLUDE = /usr/include/nvidia/gdk
NVML_LIB = /usr/src/gdk/nvml/lib
#######

BUILDFOR = $(ARCH).$(DEVICE).$(BUILDTYPE).$(MATHLIB)

OBJDIR = .build/$(BUILDFOR)
BINDIR = bin/$(BUILDFOR)

# Set up debug vs release compiler settings, both nvcc and gcc
ifeq ($(BUILDTYPE),debug)
	BUILDTYPE_OPT = -g
	GPU_BUILDTYPE_OPT = -O0 -G -lineinfo
else
	BUILDTYPE_OPT = -O3 -flto
	GPU_BUILDTYPE_OPT = -O3 -use_fast_math -lineinfo
endif

# Set up math library defines and libraries
ifeq ($(MATHLIB),mkl)
	MATHLIB_INCLUDE = $(MATHLIB_PATH)/mkl/include
	MATHLIB_LIB = -L$(MATHLIB_PATH)/compiler/lib/intel64 -L$(MATHLIB_PATH)/mkl/lib/intel64 -L$(MATHLIB_PATH)/compiler/lib/mic -L$(MATHLIB_PATH)/mkl/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lm -liomp5 -lpthread
	MATHLIB_DEFINE = -DUSE_MKL
else
	MATHLIB_INCLUDE = $(MATHLIB_PATH)/include
	MATHLIB_LIB = -L$(MATHLIB_PATH)/lib -lacml -lm -lpthread
	MATHLIB_DEFINE = -DUSE_ACML
endif

# Set up CUDA includes and libraries
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64 -L$(NVML_LIB) -lcublas -lcudart -lcurand -lcusparse -lnvidia-ml

# Set up final list of libs to use
ifeq ($(DEVICE),gpu)
	LINK_LIBS = $(CUDA_LIB) $(MATHLIB_LIB)
else
	LINK_LIBS = $(MATHLIB_LIB)
endif

# Compile CNTK math into its own shared library to ensure that any change to its
# global variables, like CUDA streams is made in one place and has global effect.
# Otherwise, different clients of CNTK math would observe different states.
CNTKMATH_LINK_LIB = -L$(BINDIR) -lcntkmath
CNTKMATH_LIB = $(BINDIR)/libcntkmath.so

# Set up gcc includes and libraries
INCFLAGS_COMMON = -I Common/Include -I Math/Math -I MachineLearning/CNTK -I $(MATHLIB_INCLUDE)
CFLAGS_COMMON = -msse3 -std=c++0x -std=c++11 -D_POSIX_SOURCE -D_XOPEN_SOURCE=600 -D__USE_XOPEN2K $(MATHLIB_DEFINE) -fopenmp -fpermissive -fPIC

ifeq ($(DEVICE),gpu)
	INCFLAGS = $(INCFLAGS_COMMON) -I $(CUDA_INCLUDE) -I $(NVML_INCLUDE)
	CFLAGS = $(CFLAGS_COMMON)
else
	INCFLAGS = $(INCFLAGS_COMMON)
	CFLAGS = $(CFLAGS_COMMON) -DCPUONLY
endif

# Set up nvcc target architectures (will generate code to support them all, i.e. fat-binary)
GENCODE_SM20 := -gencode arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM30 := -gencode arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 := -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)

# Set up basic nvcc options and add GPU targets from above
NVCCFLAGS = -std=c++11 -D_POSIX_SOURCE -D_XOPEN_SOURCE=600 -D__USE_XOPEN2K -m 64 $(GENCODE_FLAGS)

# Set up linker option to embed ORIGIN, i.e. directory where cntk is into the search path option
# at runtime. This will try to resolve all dependent binaries in the same directory where cntk binary resides
LDFLAGS=-Wl,-rpath,'$$ORIGIN'

# Define all sources that need to be built
COMMON_SRC = Common/fileutil.cpp Common/DataWriter.cpp Common/ConfigFile.cpp Common/DataReader.cpp \
			 Common/Eval.cpp Common/File.cpp Common/BestGpu.cpp Common/TimerUtility.cpp

MATH_COMMON_SRC = Math/Math/Matrix.cpp Math/Math/CPUMatrix.cpp Math/Math/CPUSparseMatrix.cpp

ifeq ($(DEVICE),gpu)
	MATH_SRC = $(MATH_COMMON_SRC) Math/Math/GPUMatrix.cu Math/Math/GPUMatrixCUDAKernels.cu Math/Math/GPUSparseMatrix.cu Math/Math/GPUWatcher.cu
else
	MATH_SRC = $(MATH_COMMON_SRC) Math/Math/NoGPU.cpp
endif

CN_SRC =  MachineLearning/CNTK/NetworkDescriptionLanguage.cpp MachineLearning/CNTK/CNTK.cpp MachineLearning/CNTK/ComputationNode.cpp \
		  MachineLearning/CNTK/ModelEditLanguage.cpp MachineLearning/CNTK/SimpleNetworkBuilder.cpp MachineLearning/CNTK/tests.cpp \
		  MachineLearning/CNTK/Profiler.cpp MachineLearning/CNTKEval/CNTKEval.cpp

BINARYREADER_SRC = DataReader/BinaryReader/BinaryWriter.cpp DataReader/BinaryReader/BinaryReader.cpp DataReader/BinaryReader/BinaryFile.cpp
HTKMLFREADER_SRC = DataReader/HTKMLFReader_linux/HTKMLFWriter.cpp DataReader/HTKMLFReader_linux/DataWriter.cpp DataReader/HTKMLFReader_linux/DataReader.cpp DataReader/HTKMLFReader_linux/HTKMLFReader.cpp
SEQUENCEREADER_SRC = DataReader/LMSequenceReader/SequenceReader.cpp DataReader/LMSequenceReader/SequenceParser.cpp DataReader/LMSequenceReader/Exports.cpp
LUSEQUENCEREADER_SRC = DataReader/LUSequenceReader/LUSequenceReader.cpp DataReader/LUSequenceReader/LUSequenceParser.cpp DataReader/LUSequenceReader/Exports.cpp
UCIFASTREADER_SRC = DataReader/UCIFastReader/UCIParser.cpp DataReader/UCIFastReader/UCIFastReader.cpp DataReader/UCIFastReader/Exports.cpp 

READER_SRC = $(UCIFASTREADER_SRC) $(LUSEQUENCEREADER_SRC) $(HTKMLFREADER_SRC) $(SEQUENCEREADER_SRC) $(BINARYREADER_SRC)
CORE_SRC = $(CN_SRC) $(COMMON_SRC)
SRC =  $(READER_SRC) $(CORE_SRC) $(MATH_SRC)

VPATH := $(sort  $(dir $(SRC)))

# Define object files
OBJ_TMP := $(patsubst %.cpp, $(OBJDIR)/%.o, $(SRC))
ifeq ($(DEVICE),gpu)
	OBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(OBJ_TMP))
else
	OBJ := $(OBJ_TMP)
endif

CORE_OBJ_TMP := $(patsubst %.cpp, $(OBJDIR)/%.o, $(CORE_SRC))
ifeq ($(DEVICE),gpu)
	CORE_OBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(CORE_OBJ_TMP))
else
	CORE_OBJ := $(CORE_OBJ_TMP)
endif

COMMON_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(COMMON_SRC))

MATH_OBJ_TMP := $(patsubst %.cpp, $(OBJDIR)/%.o, $(MATH_SRC))
ifeq ($(DEVICE),gpu)
	MATH_OBJ := $(patsubst %.cu, $(OBJDIR)/%.o, $(MATH_OBJ_TMP))
else
	MATH_OBJ := $(MATH_OBJ_TMP)
endif

UCIFASTREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(UCIFASTREADER_SRC))
LUSEQUENCEREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(LUSEQUENCEREADER_SRC))
SEQUENCEREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(SEQUENCEREADER_SRC))
HTKMLFREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(HTKMLFREADER_SRC))
BINARYREADER_OBJ := $(patsubst %.cpp, $(OBJDIR)/%.o, $(BINARYREADER_SRC))

# C++ include dependencies generated by -MF compiler option
DEP := $(patsubst %.o, %.d, $(OBJ))

SEPARATOR = "=-----------------------------------------------------------="

# Define build targets
all: $(BINDIR)/cntk $(BINDIR)/UCIFastReader.so $(BINDIR)/LMSequenceReader.so $(BINDIR)/LUSequenceReader.so $(BINDIR)/HTKMLFReader.so
	@echo $(SEPARATOR)
	@echo finished building for $(ARCH) with build type $(BUILDTYPE)

$(BINDIR)/UCIFastReader.so: $(UCIFASTREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CC) $(BUILDTYPE_OPT) -fPIC -shared -o $@ $^ $(CNTKMATH_LINK_LIB)

$(BINDIR)/LMSequenceReader.so: $(SEQUENCEREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CC) $(BUILDTYPE_OPT) -fPIC -shared -o $@ $^ $(CNTKMATH_LINK_LIB)

$(BINDIR)/LUSequenceReader.so: $(LUSEQUENCEREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CC) $(BUILDTYPE_OPT) -fPIC -shared -o $@ $^ $(CNTKMATH_LINK_LIB)

$(BINDIR)/HTKMLFReader.so: $(HTKMLFREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CC) $(BUILDTYPE_OPT) -fPIC -shared -o $@ $^ $(CNTKMATH_LINK_LIB)

$(BINDIR)/BinaryReader.so: $(BINARYREADER_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	$(CC) $(BUILDTYPE_OPT) -fPIC -shared -o $@ $^ $(CNTKMATH_LINK_LIB)

$(BINDIR)/cntk: $(CORE_OBJ) | $(CNTKMATH_LIB)
	@echo $(SEPARATOR)
	@mkdir -p $(dir $@)
	@echo building output for $(ARCH) with build type $(BUILDTYPE)
	$(CC) $(BUILDTYPE_OPT) $(LDFLAGS) -o $@ $^ $(LINK_LIBS) $(CNTKMATH_LINK_LIB) -fopenmp -ldl -fPIC 

$(CNTKMATH_LIB): $(MATH_OBJ) $(COMMON_OBJ)
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE) 
	@mkdir -p $(dir $@)
	$(CC) $(BUILDTYPE_OPT) -fPIC -shared -o $@ $^ $(LINK_LIBS) -fopenmp

# Include all C++ dependencies, like header files, to ensure that a change in those
# will result in the rebuild.
-include ${DEP}

ifeq ($(DEVICE),gpu)
$(OBJDIR)/%.o : %.cu Makefile
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE) 
	@mkdir -p $(dir $@)
	$(NVCC) -c $< -o $@ $(GPU_BUILDTYPE_OPT) $(NVCCFLAGS) $(INCFLAGS) -Xcompiler -fPIC
endif

$(OBJDIR)/%.o : %.cpp Makefile
	@echo $(SEPARATOR)
	@echo creating $@ for $(ARCH) with build type $(BUILDTYPE) 
	@mkdir -p $(dir $@)
	$(CC) -c $< -o $@ $(BUILDTYPE_OPT) $(CPPFLAGS) $(CFLAGS) $(INCFLAGS) -MD -MP -MF ${@:.o=.d}

.PHONY: clean

clean:
	@echo $(SEPARATOR)
	@rm -rf $(OBJDIR)
	@rm -rf $(BINDIR)
	@echo finished cleaning up the project 
