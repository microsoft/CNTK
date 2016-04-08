#include "stdafx.h"
#include "MatrixQuantizerGPU.h"
#include "MatrixQuantizer_kernel.cu"
#include "GPUMatrix.h"
#include "GPUDataTransferer.h"

#pragma comment(lib, "cudart.lib") // instruct linker to reference these libs
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cusparse.lib")
#pragma comment(lib, "curand.lib")

#pragma warning(disable : 4267) // conversion from 'size_t' to 'unsigned int'; happens in CUDA <<<a,b>>> syntax if a and b are size_t
#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this
#pragma warning(disable : 4702) // unreachable code; triggered for unknown reasons

namespace Microsoft { namespace MSR { namespace CNTK {

// CUDA failed
// Since the outer code sometimes does not recover properly, as an option we log and die right away.
// This is needed for our GCD farm which has intermittent CUDA errors that sometimes cause the DBN tool, when running with MPI, to hang instead of terminating.
void cudafail(const char* msg)
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

template <class ElemType>
void MatrixQuantizerGPU<ElemType>::Sync()
{
    cudaDeviceSynchronize() || "cudaDeviceSynchronize failed";
}

// wait until stream has completed all scheduled operations
template <class ElemType>
void MatrixQuantizerGPU<ElemType>::SyncStream(cudaStream_t stream)
{
    cudaStreamSynchronize(stream) || "cudaStreamSynchronize failed";
}

// same but for event
template <class ElemType>
void MatrixQuantizerGPU<ElemType>::SyncEvent(cudaEvent_t ev)
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
cudaStream_t MatrixQuantizerGPU<ElemType>::m_computeStream = NULL;

template <class ElemType>
cudaStream_t MatrixQuantizerGPU<ElemType>::m_fetchStream = NULL;

template <class ElemType>
cudaStream_t MatrixQuantizerGPU<ElemType>::m_assignStream = NULL;

template <class ElemType>
cudaStream_t MatrixQuantizerGPU<ElemType>::GetComputeStream()
{
    return m_computeStream;
}

template <class ElemType>
cudaStream_t MatrixQuantizerGPU<ElemType>::GetFetchStream()
{
    return m_fetchStream;
}

template <class ElemType>
cudaStream_t MatrixQuantizerGPU<ElemType>::GetAssignStream()
{
    return m_assignStream;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// computestream: the stream the caller issued the quant op on
template <class ElemType>
void MatrixQuantizerGPU<ElemType>::RecordQuantizeCompleteEvent(cudaStream_t computestream) const
{
    // schedule to flag the quantize-complete event (on main stream)
    cudaEventRecord(m_quantizeCompleteEvent, computestream) || "cudaEventRecord failed";

    // when running synchronously (for time measurements), then we (CPU) wait right here
    if (m_forceSync)
    {
        SyncStream(computestream);
    }
}

template <class ElemType>
void MatrixQuantizerGPU<ElemType>::SyncQuantizeCompleEventAndFetchAndRecordFetchCompleteEvent(char* cpuBuffer, char* gpuBuffer, size_t size) const
{
    // schedule fetch stream to wait until the last quantize op is complete, i.e. the data in the buffer is now valid
    // wait until commencement
    cudaStreamWaitEvent(GetFetchStream(), m_quantizeCompleteEvent, 0 /*flags 'must be 0'*/) || "cudaStreamWaitEvent failed";

    // schedule to fetch that quantized data into CPU buffer (on a separate transfer stream)
    cudaMemcpyAsync(cpuBuffer, gpuBuffer, size, cudaMemcpyDeviceToHost, GetFetchStream()) || "cudaMemcpyAsync failed";

    cudaEventRecord(m_fetchCompleteEvent, GetFetchStream()) || "cudaEventRecord failed"; // for next GPU operation

    // when running synchronously (for time measurements), then we (CPU) wait right here
    if (m_forceSync)
    {
        SyncStream(GetFetchStream());
    }
}

template <class ElemType>
void MatrixQuantizerGPU<ElemType>::SyncAssignCompleteEvent(cudaStream_t computestream) const
{
    // schedule to wait for the assign-complete event (on main/compute stream)     --CPU buffer free once main stream does anything after this
    cudaStreamWaitEvent(computestream, m_assignCompleteEvent, 0 /*flags 'must be 0'*/) || "cudaStreamWaitEvent failed";

    // Note that the NVidia doc says somewhat confusingly:
    //  * If \p stream is NULL, any future work submitted in any stream will wait for
    //  * \p event to complete before beginning execution. This effectively creates a
    //  * barrier for all future work submitted to the device on this thread.
    // -> it says that this may bring the whole machinery to stall. Or does cudaStreamWaitEvent() honor cudaStreamNonBlocking?
    // According to NVidia (Jiri Kraus), this works as expected.
}

template <class ElemType>
QuantizedMatrix<ElemType>& MatrixQuantizerGPU<ElemType>::GetTempGPUQuantizedMatrix(size_t numRows, size_t numCols, size_t nBits, bool& newlyAllocated)
{
    newlyAllocated = false;

    // Check if the existing one is good for our needs
    if ((m_tempGPUQuantizedMatrix != nullptr) && (m_tempGPUQuantizedMatrix->GetNumBits() == nBits) && (m_tempGPUQuantizedMatrix->GetNumRows() >= numRows) && (m_tempGPUQuantizedMatrix->GetNumCols() >= numCols))
    {
        return *m_tempGPUQuantizedMatrix;
    }

    if (m_tempGPUQuantizedMatrix != nullptr)
    {
        delete m_tempGPUQuantizedMatrix;
        m_tempGPUQuantizedMatrix = nullptr;
    }

    m_tempGPUQuantizedMatrix = new QuantizedMatrix<ElemType>(numRows, numCols, nBits, (short) this->GetDeviceId());
    newlyAllocated = true;

    return *m_tempGPUQuantizedMatrix;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///cpubuffer should be page-locked memory allocated, otherwise CUDA will not be efficient (hence we don't use STL)
template <class ElemType>
MatrixQuantizerGPU<ElemType>::MatrixQuantizerGPU(int deviceId, bool useDedicatedComputeStream, bool forceSync /*= false*/)
    : MatrixQuantizerImpl<ElemType>(deviceId), m_quantizeCompleteEvent(NULL), m_fetchCompleteEvent(NULL), m_tempMatrixZeroingCompleteEvent(NULL), m_assignCompleteEvent(NULL), m_forceSync(forceSync), m_tempGPUQuantizedMatrix(nullptr), m_quantizeOpIncludedFetch(false)
{
    PrepareDevice(this->GetDeviceId());

    // events
    // Note: Do NOT use cudaEventBlockingSync (which supposedly yields the process)--it will totally break cudaEventSynchronize(), causing it to take 50 or 100 ms randomly.
    cudaEventCreateWithFlags(&m_tempMatrixZeroingCompleteEvent, cudaEventDisableTiming) || "cudaEventCreateWithFlags failed";
    cudaEventCreateWithFlags(&m_quantizeCompleteEvent, cudaEventDisableTiming) || "cudaEventCreateWithFlags failed";
    cudaEventCreateWithFlags(&m_fetchCompleteEvent, cudaEventDisableTiming) || "cudaEventCreateWithFlags failed";
    cudaEventCreateWithFlags(&m_assignCompleteEvent, cudaEventDisableTiming) || "cudaEventCreateWithFlags failed";

#pragma warning(disable : 4127)
    if (useDedicatedComputeStream && (m_computeStream == NULL))
    {
        cudaStreamCreateWithFlags(&m_computeStream, cudaStreamNonBlocking) || "cudaStreamCreateWithFlags failed";
    }

    if (m_fetchStream == NULL)
    {
        cudaStreamCreateWithFlags(&m_fetchStream, cudaStreamNonBlocking) || "cudaStreamCreateWithFlags failed";
        cudaStreamCreateWithFlags(&m_assignStream, cudaStreamNonBlocking) || "cudaStreamCreateWithFlags failed";
    }
}

template <class ElemType>
MatrixQuantizerGPU<ElemType>::~MatrixQuantizerGPU()
{
    if (nullptr != m_tempGPUQuantizedMatrix)
    {
        delete m_tempGPUQuantizedMatrix;
        m_tempGPUQuantizedMatrix = nullptr;
    }

    // BUGBUG: we don't destroy our streams (they are static variables); we need a static destructor, I am too lazy now
    // TODO: Check for error code and throw if !std::uncaught_exception()
    cudaEventDestroy(m_assignCompleteEvent);
    cudaEventDestroy(m_fetchCompleteEvent);
    cudaEventDestroy(m_quantizeCompleteEvent);
    cudaEventDestroy(m_tempMatrixZeroingCompleteEvent);
}

template <class ElemType>
void MatrixQuantizerGPU<ElemType>::QuantizeAsync(const Matrix<ElemType>& inMatrix, const Matrix<ElemType>& inResidual, QuantizedMatrix<ElemType>& outQMatrix, Matrix<ElemType>& outResidual, bool zeroThresholdFor1Bit)
{
    // Verify various input matrix parameter's dimensions
    assert((inMatrix.GetNumRows() == outQMatrix.GetNumRows()) && (inMatrix.GetNumCols() == outQMatrix.GetNumCols()));
    assert((inMatrix.GetNumRows() == inResidual.GetNumRows()) && (inMatrix.GetNumCols() == inResidual.GetNumCols()));
    assert((inMatrix.GetNumRows() == outResidual.GetNumRows()) && (inMatrix.GetNumCols() == outResidual.GetNumCols()));

    size_t nBits = outQMatrix.GetNumBits();

    PrepareDevice(this->GetDeviceId());
    if (m_forceSync)
    {
        Sync();
    }

    bool GPUMatrixNewlyAllocated = false;
    QuantizedMatrix<ElemType>& outQMatrixGPU = (outQMatrix.GetDeviceId() == CPUDEVICE) ? GetTempGPUQuantizedMatrix(outQMatrix.GetNumRows(), outQMatrix.GetNumCols(), nBits, GPUMatrixNewlyAllocated) : outQMatrix;

    // If we newly allocated the target GPU matrix then the aysnc zeroing of the matrix is still in procgress on
    // the main compute stream. We must synchroniz with the mail compute stream in case the quantization
    // compute stream is different from the main compute stream
    if (GPUMatrixNewlyAllocated && (GetComputeStream() != GetStream()))
    {
        cudaEventRecord(m_tempMatrixZeroingCompleteEvent, GetStream()) || "cudaEventRecord failed";
        cudaStreamWaitEvent(GetComputeStream(), m_tempMatrixZeroingCompleteEvent, 0 /*flags 'must be 0'*/) || "cudaStreamWaitEvent failed";
    }

    // Do the quantization on compute sstream and insert event into stream
    _QuantizeMatrix<ElemType>(inMatrix.Data(), inResidual.Data(),
                              inMatrix.GetNumRows(), inMatrix.GetNumCols(),
                              outQMatrixGPU.Buffer(), nBits, GetComputeStream(),
                              outResidual.Data(), zeroThresholdFor1Bit);

    RecordQuantizeCompleteEvent(GetComputeStream());

    // copy from gpu to cpu if needed
    m_quantizeOpIncludedFetch = false;
    if (outQMatrix.GetDeviceId() == CPUDEVICE)
    {
        SyncQuantizeCompleEventAndFetchAndRecordFetchCompleteEvent(outQMatrix.Buffer(), outQMatrixGPU.Buffer(), outQMatrixGPU.GetSize());
        m_quantizeOpIncludedFetch = true;
    }
}

template <class ElemType>
void MatrixQuantizerGPU<ElemType>::WaitQuantizeAsyncDone()
{
    PrepareDevice(this->GetDeviceId());

    if (m_quantizeOpIncludedFetch)
    {
        SyncEvent(m_fetchCompleteEvent);
    }
    else
    {
        SyncEvent(m_quantizeCompleteEvent);
    }
}

template <class ElemType>
void MatrixQuantizerGPU<ElemType>::UnquantizeAsync(QuantizedMatrix<ElemType>& inQMatrix, Matrix<ElemType>& outMatrix, bool add /*= false*/)
{
    // The outMatrix should be on the same GPU as m_inMatrix
    assert(outMatrix.GetDeviceId() == this->GetDeviceId());

    PrepareDevice(this->GetDeviceId());

    size_t nBits = inQMatrix.GetNumBits();

    // Verify  input matrix parameter's dimensions
    assert((inQMatrix.GetNumRows() == outMatrix.GetNumRows()) && (inQMatrix.GetNumCols() == outMatrix.GetNumCols()));

    bool GPUMatrixNewlyAllocated = false;
    QuantizedMatrix<ElemType>& inQMatrixGPU = (inQMatrix.GetDeviceId() == CPUDEVICE) ? GetTempGPUQuantizedMatrix(inQMatrix.GetNumRows(), inQMatrix.GetNumCols(), nBits, GPUMatrixNewlyAllocated) : inQMatrix;

    if (inQMatrix.GetDeviceId() == CPUDEVICE)
    {
        // If the intermediate GPU Matrix was newly allocated, we need to wait for its zeroing to finish
        // before assigning the inQMatrix contents
        if (GPUMatrixNewlyAllocated)
        {
            cudaEventRecord(m_tempMatrixZeroingCompleteEvent, GetStream()) || "cudaEventRecord failed";
            cudaStreamWaitEvent(GetAssignStream(), m_tempMatrixZeroingCompleteEvent, 0 /*flags 'must be 0'*/) || "cudaStreamWaitEvent failed";
        }

        // schedule assign to GPU (on transfer stream)
        cudaMemcpyAsync(inQMatrixGPU.Buffer(), inQMatrix.Buffer(), inQMatrix.GetSize(), cudaMemcpyHostToDevice, GetAssignStream()) || "cudaMemcpyAsync failed";

        // schedule to flag the assign-complete event
        cudaEventRecord(m_assignCompleteEvent, GetAssignStream()) || "cudaEventRecord failed"; // for subsequent GPU operation to consume this buffer

        if (m_forceSync)
        {
            SyncStream(GetAssignStream());
        }

        // let the computing stream wait for the assign complete
        SyncAssignCompleteEvent(GetComputeStream());
    }

    // do the actually unquantization
    _UnquantizeMatrix(inQMatrixGPU.Buffer(), inQMatrixGPU.GetSize(),
                      outMatrix.Data(), outMatrix.GetNumRows(), outMatrix.GetNumCols(),
                      nBits, add, GetComputeStream());

    // Record the event of unquantization
    RecordQuantizeCompleteEvent(GetComputeStream());
}

template <class ElemType>
void MatrixQuantizerGPU<ElemType>::WaitUnquantizeAsyncDone()
{
    PrepareDevice(this->GetDeviceId());
    SyncEvent(m_quantizeCompleteEvent);
}

//explicit
template class MatrixQuantizerGPU<float>;
template class MatrixQuantizerGPU<double>;

GPUMatrixComputeStreamEvent::GPUMatrixComputeStreamEvent(int deviceId)
    : MatrixComputeStreamEvent(deviceId)
{
    // Note: Do NOT use cudaEventBlockingSync (which supposedly yields the process)--it will totally break cudaEventSynchronize(), causing it to take 50 or 100 ms randomly.
    cudaEventCreateWithFlags(&m_mainGPUComputeStreamCUDAEvent, cudaEventDisableTiming) || "cudaEventCreateWithFlags failed";

    // Record an event on the main GPU compute stream
    cudaEventRecord(m_mainGPUComputeStreamCUDAEvent, GetStream()) || "cudaEventRecord failed";
}

GPUMatrixComputeStreamEvent::~GPUMatrixComputeStreamEvent()
{
    // TODO: Check for error code and throw if !std::uncaught_exception()
    cudaEventDestroy(m_mainGPUComputeStreamCUDAEvent) || "cudaEventDestroy failed";
}

void GPUMatrixComputeStreamEvent::SynchronizeEvent()
{
    cudaEventSynchronize(m_mainGPUComputeStreamCUDAEvent) || "cudaEventSynchronize failed";
}

template <typename ElemType>
void GPUMatrixComputeStreamEvent::SynchronizeQuantizationComputeStreamWithEvent()
{
    cudaStreamWaitEvent(MatrixQuantizerGPU<ElemType>::GetComputeStream(), m_mainGPUComputeStreamCUDAEvent, 0 /*flags 'must be 0'*/) || "cudaStreamWaitEvent failed";
}

template <typename ElemType>
void GPUMatrixComputeStreamEvent::SynchronizeDataTransferFetchStreamWithEvent()
{
    cudaStreamWaitEvent(GPUDataTransferer<ElemType>::GetFetchStream(), m_mainGPUComputeStreamCUDAEvent, 0 /*flags 'must be 0'*/) || "cudaStreamWaitEvent failed";
}

// Explicit template instantiations
template void GPUMatrixComputeStreamEvent::SynchronizeQuantizationComputeStreamWithEvent<float>();
template void GPUMatrixComputeStreamEvent::SynchronizeQuantizationComputeStreamWithEvent<double>();
template void GPUMatrixComputeStreamEvent::SynchronizeDataTransferFetchStreamWithEvent<float>();
template void GPUMatrixComputeStreamEvent::SynchronizeDataTransferFetchStreamWithEvent<double>();
} } }
