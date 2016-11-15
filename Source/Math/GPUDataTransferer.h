#pragma once

#ifndef CPUONLY
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif // !CPUONLY

#include "Basics.h"

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

#include "DataTransferer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class MATH_API GranularGPUDataTransferer : public DataTransferer
{
public:
#ifndef CPUONLY
    GranularGPUDataTransferer(int deviceId, const cudaStream_t& fetchStream, const cudaStream_t& assignStream, bool blocking = false);
#else
    GranularGPUDataTransferer() {}
#endif // !CPUONLY

    ~GranularGPUDataTransferer();

    void CopyGPUToCPUAsync(const void* gpuBuffer, size_t numElements, size_t elementSize, void* cpuBuffer) override;
    void RecordGPUToCPUCopy() override;
    void WaitForCopyGPUToCPU() override;

    void CopyCPUToGPUAsync(const void* cpuBuffer, size_t numElements, size_t elementSize, void* gpuBuffer) override;
    void RecordCPUToGPUCopy() override;
    void WaitForCopyCPUToGPU() override;

    void RecordComputeStreamSyncPoint() override;
    void WaitForSyncPointOnFetchStreamAsync() override;
    void WaitForSyncPointOnAssignStreamAsync() override;

#ifndef CPUONLY
private:
    // Not owned by this class, are always injected.
    const cudaStream_t& m_fetchStream;
    const cudaStream_t& m_assignStream;

protected:
    mutable cudaEvent_t m_fetchCompleteEvent;
    mutable cudaEvent_t m_assignCompleteEvent;
    mutable cudaEvent_t m_syncEvent;
#endif // !CPUONLY

protected:
    int m_deviceId;

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(GranularGPUDataTransferer);

    friend class GPUDataTransferer;
};

class MATH_API GPUDataTransferer
{
#pragma warning(push)
#pragma warning(disable : 4251) // Using std::unique pointer on the dll boundary.
    std::unique_ptr<GranularGPUDataTransferer> m_inner;
#pragma warning(pop)

public:
    GPUDataTransferer(int deviceId, bool useConcurrentStreams);
    ~GPUDataTransferer();

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(GPUDataTransferer);

    // GPU to CPU
    void CopyGPUToCPUAsync(void* gpuBuffer, size_t totalSize, void* cpuBuffer);

    template <class ElemType>
    void CopyGPUToCPUAsync(ElemType* gpuBuffer, size_t numElements, ElemType* cpuBuffer)
    {
        CopyGPUToCPUAsync(static_cast<void*>(gpuBuffer), numElements * sizeof(ElemType), cpuBuffer);
    }

    void WaitForCopyGPUToCPUAsync();

    // CPU to GPU
    void CopyCPUToGPUAsync(void* cpuBuffer, size_t totalSize, void* gpuBuffer);

    template <class ElemType>
    void CopyCPUToGPUAsync(ElemType* cpuBuffer, size_t numElements, ElemType* gpuBuffer)
    {
        CopyCPUToGPUAsync(static_cast<void*>(cpuBuffer), numElements * sizeof(ElemType), gpuBuffer);
    }

    void WaitForCopyCPUToGPUAsync();

#ifndef CPUONLY
    static cudaStream_t GetFetchStream();
#endif // !CPUONLY

private:
#ifndef CPUONLY
    static void SyncEvent(cudaEvent_t ev);

    static cudaStream_t s_fetchStream;
    static cudaStream_t s_assignStream;
#endif // !CPUONLY
};

class PrefetchGPUDataTransferer : public GranularGPUDataTransferer
{
public:
    PrefetchGPUDataTransferer(int deviceId);

private:
#ifndef CPUONLY
    static cudaStream_t s_prefetchStream;
    static cudaStream_t s_gpuToCpuStream;
#endif

    DISABLE_COPY_AND_MOVE(PrefetchGPUDataTransferer);
};

}}}