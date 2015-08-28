#include "stdafx.h"
#include "CUDAPageLockedMemAllocator.h"
#include <cuda_runtime_api.h>

namespace Microsoft { namespace MSR { namespace CNTK {

    CUDAPageLockedMemAllocator::CUDAPageLockedMemAllocator(int deviceID)
        : m_deviceID(deviceID)
    {
    }

    char* CUDAPageLockedMemAllocator::Malloc(size_t size)
    {
        void* p;
        cudaSetDevice(m_deviceID);

        // Note: I ask for cudaHostAllocDefault but cudaHostGetFlags() shows that it is allocated as 'cudaHostAllocMapped'
        cudaHostAlloc(&p, size, cudaHostAllocDefault) || "Malloc in CUDAPageLockedMemAllocator failed";

        return (char*)p;
    }

    void CUDAPageLockedMemAllocator::Free(char* p)
    {
        cudaSetDevice(m_deviceID);
        cudaFreeHost(p) || "Free in CUDAPageLockedMemAllocator failed";
    }
}}}
