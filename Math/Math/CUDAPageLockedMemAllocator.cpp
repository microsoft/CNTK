#include "stdafx.h"
#include "CUDAPageLockedMemAllocator.h"
#ifndef CPUONLY
#include <cuda_runtime_api.h>
#endif // !CPUONLY
#include "BestGpu.h"

namespace Microsoft { namespace MSR { namespace CNTK {

#ifndef CPUONLY
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

    int CUDAPageLockedMemAllocator::GetDeviceID() const
    {
        return m_deviceID;
    }
#else
    // Dummy definitions when compiling for CPUONLY
    CUDAPageLockedMemAllocator::CUDAPageLockedMemAllocator(int)
    {
    }

    int CUDAPageLockedMemAllocator::GetDeviceID() const
    {
        return -1;
    }

    char* CUDAPageLockedMemAllocator::Malloc(size_t)
    {
        return NULL;
    }

    void CUDAPageLockedMemAllocator::Free(char*)
    {
    }
#endif
}}}
