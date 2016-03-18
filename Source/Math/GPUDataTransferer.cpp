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
template <class ElemType>
void GPUDataTransferer<ElemType>::SyncEvent(cudaEvent_t ev)
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
template <class ElemType>
cudaStream_t GPUDataTransferer<ElemType>::m_fetchStream = NULL;

template <class ElemType>
cudaStream_t GPUDataTransferer<ElemType>::m_assignStream = NULL;

template <class ElemType>
cudaStream_t GPUDataTransferer<ElemType>::GetFetchStream()
{
    return m_fetchStream;
}

template <class ElemType>
GPUDataTransferer<ElemType>::GPUDataTransferer(int deviceId, bool useConcurrentStreams)
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

template <class ElemType>
GPUDataTransferer<ElemType>::~GPUDataTransferer()
{
    // BUGBUG: we don't destroy our streams (they are static variables); we need a static destructor, I am too lazy now
    // TODO: Check for error code and throw if !std::uncaught_exception()
    cudaEventDestroy(m_assignCompleteEvent);
    cudaEventDestroy(m_fetchCompleteEvent);
}

template <class ElemType>
void GPUDataTransferer<ElemType>::CopyGPUToCPUAsync(ElemType* gpuBuffer, size_t numElements, ElemType* cpuBuffer)
{
    PrepareDevice(m_deviceId);

    cudaMemcpyAsync(cpuBuffer, gpuBuffer, numElements * sizeof(ElemType), cudaMemcpyDeviceToHost, m_fetchStream) || "cudaMemcpyAsync failed";
    cudaEventRecord(m_fetchCompleteEvent, m_fetchStream) || "cudaEventRecord failed";
}

template <class ElemType>
void GPUDataTransferer<ElemType>::WaitForCopyGPUToCPUAsync()
{
    PrepareDevice(m_deviceId);

    SyncEvent(m_fetchCompleteEvent);
}

template <class ElemType>
void GPUDataTransferer<ElemType>::CopyCPUToGPUAsync(ElemType* cpuBuffer, size_t numElements, ElemType* gpuBuffer)
{
    PrepareDevice(m_deviceId);

    cudaMemcpyAsync(gpuBuffer, cpuBuffer, numElements * sizeof(ElemType), cudaMemcpyHostToDevice, m_assignStream) || "cudaMemcpyAsync failed";
    cudaEventRecord(m_assignCompleteEvent, m_assignStream) || "cudaEventRecord failed";
}

template <class ElemType>
void GPUDataTransferer<ElemType>::WaitForCopyCPUToGPUAsync()
{
    PrepareDevice(m_deviceId);

    SyncEvent(m_assignCompleteEvent);
}

//explicit
template class GPUDataTransferer<float>;
template class GPUDataTransferer<double>;
} } }
