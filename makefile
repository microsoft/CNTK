# WORK IN PROGRESS, not currently complete nor usable

# makefile for a Linux/GCC build of CNTK
# This needs ACML_PATH. E.g. in tcsh, say: setenv ACML_PATH C:/AMD/acml5.3.1/ifort64_mp

# This is work in progress and not at all complete or usable.
#
# The Linux and Windows versions are not different branches, but rather build off the same
# source files, using different makefiles. This current makefile has the purpose of enabling
# work to make all sources compile with GCC, and also to check for GCC-compat regressions due to
# modifications which are currently done under Windows.
#
# The planned steps are:
#  - runnable non-GPU GCC-built version under Cygwin
#     - get all CPU-only sources to compile with GCC/x64 under Cygwin    --currently ongoing work
#     - port the dynamic-loading mechanism
#  - runnable non-GPU version on actual Linux
#  - enable CUDA on Linux (=makefile code and figuring out the right compiler options)
#
# Any help is welcome, of course!
#
# This makefile will be extended/completed as we go.

.SUFFIXES: 

#SRC := Common/File.cpp Math/Math/Matrix.cpp
#OBJ := $(SRC:%.cpp=%.obj)
#DEP := $(OBJ:%.obj=%.dep)

INCFLAGS = -I Common/Include -I Math/Math -I MachineLearning/cn -I $(ACML_PATH)/include

COMMON_SRC = Common/fileutil.cpp Common/DataWriter.cpp Common/ConfigFile.cpp Common/DataReader.cpp \
             Common/Eval.cpp Common/File.cpp Common/NetworkDescriptionLanguage.cpp Common/BestGpu.cpp

MATH_SRC = Math/Math/Matrix.obj Math/Math/CPUMatrix.obj Math/Math/CPUSparseMatrix.obj Math/Math/GPUDummy.obj

CN_SRC =  MachineLearning/cn/cn.cpp MachineLearning/cn/ComputationNode.cpp MachineLearning/cn/ModelEditLanguage.cpp MachineLearning/cn/NetworkDescriptionLanguage.cpp MachineLearning/cn/PTaskGraphBuilder.cpp MachineLearning/cn/SimpleNetworkBuilder.cpp MachineLearning/cn/tests.cpp

SRC = $(CN_SRC) $(MATH_SRC) $(COMMON_SRC)

all:	${SRC:.cpp=.obj}


CFLAGS = -std=c++0x -std=c++11 -DCPUONLY -D_POSIX_SOURCE -D_XOPEN_SOURCE=600 -D__USE_XOPEN2K -fpermissive

%.obj:	%.cpp
	gcc -c -o $@ $(CPPFLAGS) $(CFLAGS) $(INCFLAGS) -MD -MP -MF ${@:.obj=.dep} $<

# .dep files created by -MD option in the gcc call
-include $(DEP) 
