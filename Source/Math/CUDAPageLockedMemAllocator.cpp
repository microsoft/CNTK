#include "stdafx.h"
#include "CUDAPageLockedMemAllocator.h"
#include "BestGpu.h" // for CPUONLY
#ifndef CPUONLY
#include <cuda_runtime_api.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

#ifndef CPUONLY
CUDAPageLockedMemAllocator::CUDAPageLockedMemAllocator(int deviceID)
    : m_deviceID(deviceID)
{
}

void* CUDAPageLockedMemAllocator::Malloc(size_t size, int deviceId)
{
    void* p;
    cudaSetDevice(deviceId);

    // Note: I ask for cudaHostAllocDefault but cudaHostGetFlags() shows that it is allocated as 'cudaHostAllocMapped'
    cudaHostAlloc(&p, size, cudaHostAllocDefault) || "Malloc in CUDAPageLockedMemAllocator failed";

    return p;
}

void CUDAPageLockedMemAllocator::Free(void* p, int deviceId)
{
    cudaSetDevice(deviceId);
    cudaFreeHost(p) || "Free in CUDAPageLockedMemAllocator failed";
}

void* CUDAPageLockedMemAllocator::Malloc(size_t size)
{
    return Malloc(size, m_deviceID);
}

void CUDAPageLockedMemAllocator::Free(void* p)
{
    Free(p, m_deviceID);
}

int CUDAPageLockedMemAllocator::GetDeviceId() const
{
    return m_deviceID;
}
#else
// Dummy definitions when compiling for CPUONLY
CUDAPageLockedMemAllocator::CUDAPageLockedMemAllocator(int)
{
}

int CUDAPageLockedMemAllocator::GetDeviceId() const
{
    return -1;
}

void* CUDAPageLockedMemAllocator::Malloc(size_t)
{
    return nullptr;
}

void* CUDAPageLockedMemAllocator::Malloc(size_t, int)
{
	return nullptr;
}

void CUDAPageLockedMemAllocator::Free(void*)
{
}

void CUDAPageLockedMemAllocator::Free(void*, int)
{
}
#endif
} } }
