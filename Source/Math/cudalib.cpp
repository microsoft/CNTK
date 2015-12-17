// cudalib.cpp -- all CUDA calls (but not cublas) are encapsulated here
// All actual CUDA API calls go here, to keep the header out of our other headers.
//
// F. Seide, V-hansu

#define _CRT_SECURE_NO_WARNINGS 1    // so we can use getenv()...

#include "Basics.h"
#include <cuda_runtime_api.h>           // for CUDA API
#include <cuda.h>                       // for device API
#include "cudalib.h"
#include "cudadevice.h"
#include <string>
#include <assert.h>
#include <cublas_v2.h>

#undef NOMULTIDEVICE       // define this to disable any context/driver stuff

#ifndef NOMULTIDEVICE
#pragma comment (lib, "cuda.lib")       // link CUDA device API
#endif
#pragma comment (lib, "cudart.lib")     // link CUDA runtime
#pragma comment (lib, "cublas.lib")

namespace msra { namespace cuda {

static int devicesallocated = -1;    // -1 means not initialized

// allows to write cudaFunction() || "error"   (CUDA runtime)
static void operator|| (cudaError_t rc, const char * msg)
{
    if (rc != cudaSuccess)
        RuntimeError("%s: %s (cuda error %d)", msg, cudaGetErrorString(rc), (int)rc);
}

cudaStream_t GetCurrentStream() { return cudaStreamDefault; }

// synchronize with ongoing thread
void join() 
{ 
    cudaDeviceSynchronize() || "cudaDeviceSynchronize failed";
} 

// allocate a stack to store the devices that have been pushed
const int stackSize = 20;
static int curStack = 0;
static size_t deviceStack[stackSize] = {0};


// memory allocation
void * mallocbytes (size_t nelem, size_t sz)
{
    for (size_t retry = 0; ; retry++)
    {
        try
        {
            //fprintf (stderr, "mallocbytes: allocating %d elements of size %d, %d bytes\n", (int) nelem, (int) sz, (int) (nelem * sz));        // comment out by [v-hansu] to get rid out annoying output
            void * p;
            cudaMalloc (&p, nelem * sz) || "cudaMalloc failed";
            return p;
        }
        catch (const std::exception & e)
        {
            fprintf (stderr, "mallocbytes: failed with error %s\n", e.what());
            if (retry >= 5)
                throw;
        }
    }
}

void freebytes (void * p) { cudaFree (p) || "cudaFree failed"; }

void memcpyh2d (void * dst, size_t byteoffset, const void * src, size_t nbytes)
{
    cudaMemcpy (byteoffset + (char*) dst, src, nbytes, cudaMemcpyHostToDevice) || "cudaMemcpy failed";
}

void memcpyd2h (void * dst, const void * src, size_t byteoffset, size_t nbytes)
{
    cudaMemcpy (dst, byteoffset + (const char *) src, nbytes, cudaMemcpyDeviceToHost) || "cudaMemcpy failed";
}

};};
