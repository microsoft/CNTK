#include "stdafx.h"
#include "CUDAPageLockedMemAllocator.h"
#include "BestGpu.h"    // for CPUONLY
#ifndef CPUONLY
#include <cuda_runtime_api.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

    CUDAPageLockedMemAllocator::CUDAPageLockedMemAllocator(int deviceID)
        : m_deviceID(deviceID)
    {
    }

    char* CUDAPageLockedMemAllocator::Malloc(size_t size)
    {
#ifndef CPUONLY
        void* p;
        cudaSetDevice(m_deviceID);

        // Note: I ask for cudaHostAllocDefault but cudaHostGetFlags() shows that it is allocated as 'cudaHostAllocMapped'
        cudaHostAlloc(&p, size, cudaHostAllocDefault) || "Malloc in CUDAPageLockedMemAllocator failed";

        return (char*)p;
#else
        return (char*) malloc(size);
#endif
    }

    void CUDAPageLockedMemAllocator::Free(char* p)
    {
#ifndef CPUONLY
        cudaSetDevice(m_deviceID);
        cudaFreeHost(p) || "Free in CUDAPageLockedMemAllocator failed";
#else
        free(p);
#endif
    }
}}}
