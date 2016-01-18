#pragma once

#ifndef CPUONLY
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif // !CPUONLY
#include <vector>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class GPUDataTransferer
{
public:
    GPUDataTransferer(int deviceId, bool useConcurrentStreams);
    ~GPUDataTransferer();

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(GPUDataTransferer);

    void CopyGPUToCPUAsync(ElemType* gpuBuffer, size_t numElements, ElemType* cpuBuffer);
    void WaitForCopyGPUToCPUAsync();

    void CopyCPUToGPUAsync(ElemType* cpuBuffer, size_t numElements, ElemType* gpuBuffer);
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
