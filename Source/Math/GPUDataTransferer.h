#pragma once

#ifndef CPUONLY
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif // !CPUONLY
#include <vector>
#include <memory>

#ifdef _WIN32
#ifndef MATH_API
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#endif /* MATH_API */
#else  // no DLLs in Linux
#define MATH_API
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

class MATH_API GPUDataTransferer
{
public:
    GPUDataTransferer(int deviceId, bool useConcurrentStreams);
    ~GPUDataTransferer();

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(GPUDataTransferer);

    template <class ElemType>
    void CopyGPUToCPUAsync(ElemType* gpuBuffer, size_t numElements, ElemType* cpuBuffer);
    void CopyGPUToCPUAsync(void* gpuBuffer, size_t totalSize, void* cpuBuffer);
    void WaitForCopyGPUToCPUAsync();

    template <class ElemType>
    void CopyCPUToGPUAsync(ElemType* cpuBuffer, size_t numElements, ElemType* gpuBuffer);
    void CopyCPUToGPUAsync(void* cpuBuffer, size_t totalSize, void* gpuBuffer);
    void WaitForCopyCPUToGPUAsync();

#ifndef CPUONLY
    static cudaStream_t GetFetchStream();
#endif // !CPUONLY

private:
#ifndef CPUONLY
    static void SyncEvent(cudaEvent_t ev);
#endif // !CPUONLY

private:
#ifndef CPUONLY
    static cudaStream_t m_fetchStream;
    static cudaStream_t m_assignStream;

    mutable cudaEvent_t m_fetchCompleteEvent;
    mutable cudaEvent_t m_assignCompleteEvent;
#endif // !CPUONLY

    int m_deviceId;
};

} } }
