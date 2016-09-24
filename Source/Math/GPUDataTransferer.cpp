#include "stdafx.h"
#include "Basics.h"
#include "GPUDataTransferer.h"
#include "GPUMatrix.h"

#pragma comment(lib, "cudart.lib")

#pragma warning(disable : 4267) // conversion from 'size_t' to 'unsigned int'; happens in CUDA <<<a,b>>> syntax if a and b are size_t
#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this
#pragma warning(disable : 4702) // unreachable code; triggered for unknown reasons

namespace Microsoft { namespace MSR { namespace CNTK {

// CUDA failed
// Since the outer code sometimes does not recover properly, as an option we log and die right away.
// This is needed for our GCD farm which has intermittent CUDA errors that sometimes cause the DBN tool, when running with MPI, to hang instead of terminating.
static void cudafail(const char* msg)
{
    // TODO: get from an env variable
    bool dieoncudafailure = false;
    if (!dieoncudafailure)
    {
        RuntimeError("%s", msg);
    }
    fprintf(stderr, "%s\n", msg);
    fprintf(stderr, "cudafail: terminating\n"), fflush(stderr);
#ifdef WIN32
    TerminateProcess(GetCurrentProcess(), EXIT_FAILURE); // fail the hard way to ensure it won't hang elsewhere
#else
    exit(1);
#endif
}

// allows to write cudaFunction() || "error"   (CUDA runtime)
static
#ifdef WIN32
    __declspec(noinline)
#endif
        void
        operator||(cudaError_t rc, const char* msg)
{
    if (rc != cudaSuccess)
    {
        char buf[1000];
        sprintf_s(buf, 1000, "%s: %s (cuda error %d)", msg, cudaGetErrorString(rc), rc);
        cudafail(buf);
    }
}

// same but for event
void GPUDataTransferer::SyncEvent(cudaEvent_t ev)
{
    auto rc = cudaEventQuery(ev);
    if (rc != cudaErrorNotReady)
    {
        // if Event is ready then no need to wait
        rc || "cudaEventQuery failed";
        return;
    }
    // we must wait
    cudaEventSynchronize(ev) || "cudaEventSynchronize failed";
}

//streams
cudaStream_t GPUDataTransferer::m_fetchStream = NULL;

cudaStream_t GPUDataTransferer::m_assignStream = NULL;

cudaStream_t GPUDataTransferer::GetFetchStream()
{
    return m_fetchStream;
}

GPUDataTransferer::GPUDataTransferer(int deviceId, bool useConcurrentStreams)
    : m_deviceId(deviceId)
{
    PrepareDevice(m_deviceId);

    // events
    // Note: Do NOT use cudaEventBlockingSync (which supposedly yields the process)--it will totally break cudaEventSynchronize(), causing it to take 50 or 100 ms randomly.
    cudaEventCreateWithFlags(&m_fetchCompleteEvent, cudaEventDisableTiming) || "cudaEventCreateWithFlags failed";
    cudaEventCreateWithFlags(&m_assignCompleteEvent, cudaEventDisableTiming) || "cudaEventCreateWithFlags failed";

#pragma warning(disable : 4127)
    if (useConcurrentStreams && (m_fetchStream == NULL))
    {
        cudaStreamCreateWithFlags(&m_fetchStream, cudaStreamNonBlocking) || "cudaStreamCreateWithFlags failed";
        cudaStreamCreateWithFlags(&m_assignStream, cudaStreamNonBlocking) || "cudaStreamCreateWithFlags failed";
    }
}

GPUDataTransferer::~GPUDataTransferer()
{
    // BUGBUG: we don't destroy our streams (they are static variables); we need a static destructor, I am too lazy now
    // TODO: Check for error code and throw if !std::uncaught_exception()
    cudaEventDestroy(m_assignCompleteEvent);
    cudaEventDestroy(m_fetchCompleteEvent);
}

template <class ElemType>
void GPUDataTransferer::CopyGPUToCPUAsync(ElemType* gpuBuffer, size_t numElements, ElemType* cpuBuffer)
{
    CopyGPUToCPUAsync((void*)gpuBuffer, numElements * sizeof(ElemType), (void*)cpuBuffer);
}

void GPUDataTransferer::CopyGPUToCPUAsync(void* gpuBuffer, size_t totalSize, void* cpuBuffer)
{
    PrepareDevice(m_deviceId);

    cudaMemcpyAsync(cpuBuffer, gpuBuffer, totalSize, cudaMemcpyDeviceToHost, m_fetchStream) || "cudaMemcpyAsync failed";
    cudaEventRecord(m_fetchCompleteEvent, m_fetchStream) || "cudaEventRecord failed";
}

void GPUDataTransferer::WaitForCopyGPUToCPUAsync()
{
    PrepareDevice(m_deviceId);

    SyncEvent(m_fetchCompleteEvent);
}

template <class ElemType>
void GPUDataTransferer::CopyCPUToGPUAsync(ElemType* cpuBuffer, size_t numElements, ElemType* gpuBuffer)
{
    CopyCPUToGPUAsync((void*)cpuBuffer, numElements * sizeof(ElemType), (void*)gpuBuffer);
}

void GPUDataTransferer::CopyCPUToGPUAsync(void* cpuBuffer, size_t totalSize, void* gpuBuffer)
{
    PrepareDevice(m_deviceId);

    cudaMemcpyAsync(gpuBuffer, cpuBuffer, totalSize, cudaMemcpyHostToDevice, m_assignStream) || "cudaMemcpyAsync failed";
    cudaEventRecord(m_assignCompleteEvent, m_assignStream) || "cudaEventRecord failed";
}

void GPUDataTransferer::WaitForCopyCPUToGPUAsync()
{
    PrepareDevice(m_deviceId);

    SyncEvent(m_assignCompleteEvent);
}

//explicit
template void GPUDataTransferer::CopyGPUToCPUAsync<float>(float*, size_t, float*);
template void GPUDataTransferer::CopyGPUToCPUAsync<double>(double*, size_t, double*);
template void GPUDataTransferer::CopyCPUToGPUAsync<float>(float*, size_t, float*);
template void GPUDataTransferer::CopyCPUToGPUAsync<double>(double*, size_t, double*);

} } }
