CSOURCES = CPUMatrix.cpp CPUSparseMatrix.cpp Matrix.cpp 

OBJECTS = CPUMatrix.o CPUSparseMatrix.o Matrix.o \
	GPUMatrixCUDAKernels.o GPUMatrix.o \
	GPUWatcher.o GPUSparseMatrix.o  

INCLUDES = -I../../Common/Include -I/opt/acml5.3.1/gfortran64_mp_int64/include 

DEPS = 

CFLAGS =  $(INCLUDES)  \
	-D BASETYPES_NO_UNSAFECRTOVERLOAD -DBASETYPES_NO_STRPRINTF \
	-DLINUX -Wnon-template-friend  -std=c++11 # -D_FILEUTIL_  

NVCFLAGS = -DLINUX -D BASETYPES_NO_UNSAFECRTOVERLOAD -DBASETYPES_NO_STRPRINTF \
	-I../../Common/Include -arch=compute_20  -std=c++11 # -D_FILEUTIL_ 

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
