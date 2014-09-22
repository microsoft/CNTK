CSOURCES = CPUMatrix.cpp CPUSparseMatrix.cpp Matrix.cpp 

OBJECTS = CPUMatrix.o CPUSparseMatrix.o Matrix.o \
	GPUSparseMatrix.o GPUWatcher.o \
	GPUMatrixCUDAKernels.o GPUMatrix.o 

INCLUDES = -I../../Common/Include -I/opt/acml5.3.1/gfortran64_mp_int64/include 

DEPS = 

CFLAGS =  $(INCLUDES)  \
	-D BASETYPES_NO_UNSAFECRTOVERLOAD -DBASETYPES_NO_STRPRINTF \
	-DLINUX -D_FILEUTIL_  -Wnon-template-friend  -std=c++11

NVCFLAGS = -DLINUX -I../../Common/Include -D_FILEUTIL_ -arch sm_11

CXX = gcc
NVCC = nvcc

all: 	libmatrixmat.so

libmatrixmat.so: $(OBJECTS)
	$(CXX) -shared $(OBJECTS) -o libmatrixmat.so

libmatrixmat.so: $(OBJECTS)


%.o: %.cpp $(DEPS)
	$(CXX) -c $(CFLAGS) $< -o $@

%.o: %.cu $(DEPS)
	$(NVCC) -c $(NVCFLAGS) $< -o $@
