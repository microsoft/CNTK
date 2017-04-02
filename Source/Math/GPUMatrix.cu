//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "Basics.h"
#include "BestGpu.h"

#ifndef CPUONLY

#include "GPUMatrix.h"
#include "GPUMatrixCUDAKernels.cuh"
//#include "GPUSparseMatrix.h"
#include "GPUTensor.h"
#include "CommonMatrix.h"
#define TENSOR_OPS_DECL __device__ __host__
#include "TensorOps.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cublas_v2.h"
#include <assert.h>
#include <memory>
#include "CntkBatchNormalization.cuh"
#include "Convolution.cuh"
#include "CuDnnRNN.h"

#pragma comment(lib, "cudart.lib") // instruct linker to reference these libs
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cusparse.lib")
#pragma comment(lib, "curand.lib")

#pragma warning(disable : 4267) // conversion from 'size_t' to 'unsigned int'; happens in CUDA <<<a,b>>> syntax if a and b are size_t
#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this
#pragma warning(disable : 4702) // unreachable code; triggered for unknown reasons

#define DEFAULT_THREAD_PER_DIM 16

#define UNCONST(t, c, uc) GPUMatrix<t>& uc = const_cast<GPUMatrix<t>&>(c);

#ifdef _WIN32
// thread local storage to access the current stream, initialize to default stream
__declspec(thread)
#endif
    cudaStream_t t_stream = cudaStreamDefault;

#define DEFAULT_THREAD_PER_DIM 16

extern int _ConvertSMVer2Cores(int major, int minor); // forward declaration

// SetStream - set the stream that will be used by the GPU routines
void MATH_API SetStream(cudaStream_t stream)
{
    t_stream = stream;
}

// GetStream - get the stream that will be used by the GPU routines
cudaStream_t MATH_API GetStream()
{
    return t_stream;
}

// Helper macro patterns for elementwise methods
#define DEF_ELEMWISE_INPLACE_FUNC(f)                                      \
    template <class ElemType>                                             \
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::Inplace##f()                \
    {                                                                     \
        performElementWiseFunction(ElementWiseOperator::op##f, Data());   \
        return *this;                                                     \
    }
#define DEF_ELEMWISE_ASSIGN_FUNC(f)                                                       \
    template <class ElemType>                                                             \
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::Assign##f##Of(const GPUMatrix<ElemType>& a) \
    {                                                                                     \
        if (a.IsEmpty())                                                                  \
            LogicError("Assign##f##Of: Matrix a is empty.");                              \
        if (this != &a)                                                                   \
            RequireSize(a.GetNumRows(), a.GetNumCols());                                  \
        performElementWiseFunction(ElementWiseOperator::op##f, a.Data());                 \
        return *this;                                                                     \
    }

template <>
const char* CudaErrString<cudaError_t>(cudaError_t x)
{
    cudaDeviceSynchronize();
    return cudaGetErrorString(x);
}
template <>
const char* CudaErrString<cublasStatus_t>(cublasStatus_t e)
{
    cudaDeviceSynchronize();
    switch (e)
    {
    case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
    default:                             return "(look for CUBLAS_STATUS_xxx in cublas_api.h)";
    }
}
template <>
const char* CudaErrString<curandStatus>(curandStatus)
{
    cudaDeviceSynchronize();
    return "(see curand.h & look for curandStatus or CURAND_STATUS_xxx)";
}

namespace Microsoft { namespace MSR { namespace CNTK {

/*static*/ std::vector<cudaDeviceProp> GridDim::s_cachedDeviceProps;
/*static*/ std::once_flag GridDim::s_cachedDevicePropsInitFlag;

/*static*/ bool SyncGuard::s_isSyncEnabled = false;

/*static*/ void SyncGuard::EnableSync()
{
    s_isSyncEnabled = true;
}

/*static*/ bool SyncGuard::IsSyncEnabled() 
{
    return s_isSyncEnabled; 
}

SyncGuard::SyncGuard(bool forceSync /*= false*/)
    : m_forceSync(forceSync)
{
    m_done = nullptr;
    if (m_forceSync || s_isSyncEnabled)
    {
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaEventCreate(&m_done));
    }
}

SyncGuard::~SyncGuard()
{
    if (m_forceSync || s_isSyncEnabled)
    {
        // The regular use of this destructor is to synchronize the GPU, but also
        // to check for errors. So this destructor is where CUDA errors would be thrown.
        // If this destructor runs during stack unwinding, then a different error has
        // already happened that should be reported; so we only clean up the resource.
        if (std::uncaught_exception())
            cudaEventDestroy(m_done);
        else
        {
            // failures in a prior launch might be reported here
            CUDA_CALL(cudaEventRecord(m_done));
            CUDA_CALL(cudaEventSynchronize(m_done));
            CUDA_CALL(cudaEventDestroy(m_done));
        }
    }
}

template <typename AllocatedElemType>
AllocatedElemType* TracingGPUMemoryAllocator::Allocate(int deviceId, size_t numRows, size_t numCols)
{
    if (IsTraceEnabled())
    {
        auto freeAndTotalMemory = GetFreeAndTotalMemoryInMBs(deviceId);
        fprintf(stderr, "Allocating Matrix<%s> (Rows = %d, Cols = %d) buffer on DeviceId = %d; GPU Memory Free = %d MB of %d MB\n", typeid(AllocatedElemType).name(), (int)numRows, (int)numCols, (int)deviceId, (int)freeAndTotalMemory.first, (int)freeAndTotalMemory.second);
        Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
    }

    AllocatedElemType* deviceBufferPtr = AllocateNoTrace<AllocatedElemType>(deviceId, numRows * numCols);

    if (IsTraceEnabled())
    {
        fprintf(stderr, "Allocated DeviceData = %p\n", (void*) deviceBufferPtr);
    }

    return deviceBufferPtr;
}

template <typename AllocatedElemType>
AllocatedElemType* TracingGPUMemoryAllocator::Allocate(int deviceId, size_t numElements)
{
    if (IsTraceEnabled())
    {
        auto freeAndTotalMemory = GetFreeAndTotalMemoryInMBs(deviceId);
        fprintf(stderr, "Allocating array<%s> (NumElements = %d) on DeviceId = %d; GPU Memory Free = %d MB of %d MB\n", typeid(AllocatedElemType).name(), (int)numElements, (int)deviceId, (int)freeAndTotalMemory.first, (int)freeAndTotalMemory.second);
        Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
    }

    AllocatedElemType* deviceBufferPtr = AllocateNoTrace<AllocatedElemType>(deviceId, numElements);
    
    if (IsTraceEnabled())
    {
        fprintf(stderr, "Allocated DeviceData = %p\n", (void*)deviceBufferPtr);
    }

    return deviceBufferPtr;
}

template <typename AllocatedElemType>
void TracingGPUMemoryAllocator::Free(int deviceId, AllocatedElemType* bufferPtr, bool ignoreCUDARetCode /*= false*/)
{
    PrepareDevice(deviceId);
    if (ignoreCUDARetCode)
        cudaFree((void*) bufferPtr);
    else
        CUDA_CALL(cudaFree((void*) bufferPtr));

    if (IsTraceEnabled())
    {
        auto freeAndTotalMemory = GetFreeAndTotalMemoryInMBs(deviceId);
        fprintf(stderr, "Freed buffer<%s> DeviceData = %p on DeviceId = %d; GPU Memory Free = %d MB of %d MB\n", typeid(AllocatedElemType).name(), (void*) bufferPtr, (int) deviceId, (int) freeAndTotalMemory.first, (int) freeAndTotalMemory.second);
        Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
    }
}

template <typename AllocatedElemType>
AllocatedElemType* TracingGPUMemoryAllocator::AllocateNoTrace(int deviceId, size_t numElements)
{
    AllocatedElemType* deviceBufferPtr;

    PrepareDevice(deviceId);
    CUDA_CALL(cudaMalloc((void**) &deviceBufferPtr, sizeof(AllocatedElemType) * numElements));

    return deviceBufferPtr;
}

std::pair<size_t, size_t> TracingGPUMemoryAllocator::GetFreeAndTotalMemoryInMBs(int deviceId)
{
    PrepareDevice(deviceId);

    size_t free, total;
    CUDA_CALL(cudaMemGetInfo(&free, &total));

    size_t numBytesPerMB = 1 << 20;
    return {free / numBytesPerMB, total / numBytesPerMB};
}

// PrepareDevice - Setup the correct cuda context for an operation
// deviceId - the device on which the operation will take place
void PrepareDevice(DEVICEID_TYPE deviceId)
{
    THREAD_LOCAL static DEVICEID_TYPE currentDevice = DEVICEID_NOTYETDETERMINED;
    // and if we last set the device to be this device we are good
    if (deviceId == currentDevice)
        return;
    CUDA_CALL(cudaSetDevice(deviceId));
    currentDevice = deviceId;
}

#pragma region DeviceBoundNumber class

template <class ElemType>
DeviceBoundNumber<ElemType>::DeviceBoundNumber(const DeviceBoundNumber<ElemType>& /*deepCopy*/)
{
    NOT_IMPLEMENTED;
}

template <class ElemType>
DeviceBoundNumber<ElemType>::DeviceBoundNumber(DeviceBoundNumber<ElemType>&& shallowCopy)
{
    ShallowCopyFrom(shallowCopy.m_data, shallowCopy.m_computeDevice);
    shallowCopy.m_data = NULL;
}

template <class ElemType>
void DeviceBoundNumber<ElemType>::ShallowCopyFrom(ElemType* newVal, int newValsDevceId)
{
    m_computeDevice = newValsDevceId;
    m_data = newVal;
}

template <class ElemType>
DeviceBoundNumber<ElemType>::~DeviceBoundNumber()
{
    if (m_data != NULL)
    {
        if (m_computeDevice < 0)
        {
            delete m_data;
            m_data = NULL;
        }
        else
        {
            TracingGPUMemoryAllocator::Free<ElemType>(m_computeDevice, m_data);
        }
    }
}

#pragma endregion DeviceBoundNumber class

#pragma region Helper functions
template <class ElemType>
cublasHandle_t _initCUBLAS(int devId)
{
    PrepareDevice((DEVICEID_TYPE) devId);
    cublasHandle_t cuHandle;
    CUBLAS_CALL(cublasCreate(&cuHandle));
    return cuHandle;
}

template <class ElemType>
void GPUMatrix<ElemType>::SetDevice(DEVICEID_TYPE deviceId)
{
    assert(deviceId >= 0);
    CUDA_CALL(cudaSetDevice(deviceId));
}

// PrepareDevice - Setup the correct cuda context for an operation
// deviceId - the device on which the operation will take place
//            defaults to -1, which means use matrices current device
template <class ElemType>
DEVICEID_TYPE GPUMatrix<ElemType>::PrepareDevice(DEVICEID_TYPE deviceId /*=-1*/) const
{
    // if default value use current compute device
    DEVICEID_TYPE newId = deviceId >= 0 ? deviceId : GetComputeDeviceId();

    Microsoft::MSR::CNTK::PrepareDevice(newId);
    return newId;
}

template <class ElemType>
ElemType* GPUMatrix<ElemType>::CopyToArray() const
{
    size_t numElements = GetNumElements();
    if (numElements != 0)
    {
        PrepareDevice();
        ElemType* pArray = new ElemType[numElements];
        CUDA_CALL(cudaMemcpy(pArray, Data(), sizeof(ElemType) * m_numRows * m_numCols, cudaMemcpyDeviceToHost));
        return pArray;
    }
    else
    {
        return NULL;
    }
}

//memory will be allocated by the callee if not enough but need to be deleted by the caller after it's done
//return number of elements copied
template <class ElemType>
size_t GPUMatrix<ElemType>::CopyToArray(ElemType*& arrayCopyTo, size_t& currentArraySize) const
{
    size_t numElements = GetNumElements();

    if (numElements > currentArraySize)
    {
        delete arrayCopyTo;
        arrayCopyTo = new ElemType[numElements];
        currentArraySize = numElements;
    }

    if (numElements != 0)
    {
        PrepareDevice();
        CUDA_CALL(cudaMemcpy(arrayCopyTo, Data(), sizeof(ElemType) * numElements, cudaMemcpyDeviceToHost));
    }

    return numElements;
}

template <class ElemType>
void GPUMatrix<ElemType>::CopySection(size_t numRows, size_t numCols, ElemType* dst, size_t colStride) const
{
    CUBLAS_CALL(cublasGetMatrix((int) numRows, (int) numCols, sizeof(ElemType),
                                Data(), (int) GetNumRows(), dst, (int) colStride));
}
template <class ElemType>
void GPUMatrix<ElemType>::ChangeDeviceTo(DEVICEID_TYPE to_id)
{
    if (to_id == CPUDEVICE)
        LogicError("to_id must be valid GPU");
    if (GetComputeDeviceId() == to_id)
        return;

    ElemType* d_dst = TracingGPUMemoryAllocator::Allocate<ElemType>(to_id, m_numRows, m_numCols);

    SetSizeAllocated(m_numRows * m_numCols);

    // check to make sure we have something to copy (on init we often have zero sized allocations)
    if (GetSizeAllocated() > 0)
    {
#if 0 // see the backlog item # 1220
        // IOMMU DMAR needs to be disabled for CUDA P2P, otherwise it will silently hang.
        // Unfortunately, cudaDeviceCanAccessPeer returns true irrespective of the IOMMU settings.
        // More details: https://bugzilla.kernel.org/show_bug.cgi?id=188271
        // http://docs.nvidia.com/cuda/gpudirect-rdma/#supported-systems
        // TODO: enable UVA p2p access once this is fixed.

        // first try peer access
        int canAccessPeer = false;
        CUDA_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, to_id, GetComputeDeviceId()));
        if (canAccessPeer)
        {
            cudaError_t cudaStatus = cudaDeviceEnablePeerAccess(GetComputeDeviceId(), 0);
            if (cudaStatus != cudaErrorPeerAccessAlreadyEnabled)
            {
                CUDA_CALL(cudaStatus);
            }
            CUDA_CALL(cudaMemcpyPeer(d_dst, to_id, Data(), GetComputeDeviceId(), sizeof(ElemType) * m_numRows * m_numCols));
        }
        else
#endif
        {
            // peer access didn't work, just copy normal
            // make this more efficient by keeping some buffers available for each copy
            ElemType* h_dst = NULL;
            PrepareDevice();
            CUDA_CALL(cudaMallocHost((void**) &h_dst, sizeof(ElemType) * m_numRows * m_numCols));
            CUDA_CALL(cudaMemcpy(h_dst, Data(), sizeof(ElemType) * m_numRows * m_numCols, cudaMemcpyDeviceToHost));
            PrepareDevice((DEVICEID_TYPE) to_id);
            CUDA_CALL(cudaMemcpy(d_dst, h_dst, sizeof(ElemType) * m_numRows * m_numCols, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaFreeHost(h_dst));
        }
    }

    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), Buffer());
    SetBuffer(d_dst, m_numRows * m_numCols * sizeof(ElemType));

    PrepareDevice((DEVICEID_TYPE) to_id);
    SetComputeDeviceId(to_id);
}

template <class ElemType>
void GPUMatrix<ElemType>::performElementWiseFunction(ElementWiseOperator kind, const ElemType* src)
{
    PrepareDevice();
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    switch (kind)
    {
    case ElementWiseOperator::opSigmoid:
        return _elementWiseSigmoidOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(src, Data(), N);
    case ElementWiseOperator::opTanh:
        return _elementWiseTanhOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(src, Data(), N);
    case ElementWiseOperator::opSqrt:
        return _elementWiseSqrtOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(src, Data(), N);
    case ElementWiseOperator::opExp:
        return _elementWiseExpOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(src, Data(), N);
    case ElementWiseOperator::opLog:
        return _elementWiseLogOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(src, Data(), N);
    case ElementWiseOperator::opAbs:
        return _elementWiseAbsOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(src, Data(), N);
    case ElementWiseOperator::opLinearRectifierDerivative:
        return _elementWiseLinRectDerivativeOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(src, Data(), N);
    case ElementWiseOperator::opCosine:
        return _elementWiseCosineOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(src, Data(), N);
    case ElementWiseOperator::opNegativeSine:
        return _elementWiseNegativeSineOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(src, Data(), N);
    case ElementWiseOperator::opSigmoidDerivative:
        return _elementWiseSigmoidDerivativeOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(src, Data(), N);
    default: LogicError("performElementWiseFunction: unexpected op code %d", (int)kind);
    }
}

#pragma endregion Helper functions

#pragma region Constructors and Destructor

// should only be used by constructors
template <class ElemType>
void GPUMatrix<ElemType>::ZeroInit(int deviceId)
{
    BaseMatrix<ElemType>::ZeroInit();
    SetComputeDeviceId(deviceId);
}

template <class ElemType>
GPUMatrix<ElemType>::GPUMatrix(int deviceId)
{
    ZeroInit(deviceId);
};

template <class ElemType>
GPUMatrix<ElemType>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId)
{
    ZeroInit(deviceId);
    m_numRows = numRows;
    m_numCols = numCols;
    SetSizeAllocated(GetNumElements());

    if (GetNumElements() != 0)
    {
        SetBuffer(TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), m_numRows, m_numCols), GetNumElements() * sizeof(ElemType));
        CUDA_CALL(cudaMemset(Buffer(), 0, sizeof(ElemType) * GetSizeAllocated()));
    }
};

template <class ElemType>
GPUMatrix<ElemType>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId, ElemType* pArray, const size_t matrixFlags)
{
    ZeroInit(deviceId);
    SetValue(numRows, numCols, deviceId, pArray, matrixFlags);
};

template <class ElemType>
GPUMatrix<ElemType>::GPUMatrix(const GPUMatrix<ElemType>& deepCopyFrom)
{
    ZeroInit();
    SetValue(deepCopyFrom);
}

template <class ElemType>
GPUMatrix<ElemType>::GPUMatrix(GPUMatrix<ElemType>&& moveFrom)
{
    ShallowCopyFrom(moveFrom);
    moveFrom.ZeroValues();
}

//assignment operator, deep copy
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator=(const GPUMatrix<ElemType>& deepCopyFrom)
{
    if (this != &deepCopyFrom)
    {
        SetValue(deepCopyFrom);
    }
    return *this;
}

//move assignment operator, shallow copy
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator=(GPUMatrix<ElemType>&& moveFrom)
{
    if (this != &moveFrom)
    {
        ShallowCopyFrom(moveFrom);
        moveFrom.ZeroValues();
    }
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>::~GPUMatrix(void)
{
}

// TODO: This should be in the storage object.
// Clear will clear your storage, zeroinit just drops it on the ground.
template <class ElemType>
void GPUMatrix<ElemType>::Clear()
{
    VerifyWritable(__FUNCTION__);
    //if (OwnBuffer() && m_pArray != NULL)
    if (m_sob != nullptr)
    {
        if (GetComputeDeviceId()>= 0)
        {
            // BUG: We do not check the CUDA return code for cudaFree here since this may get called
            // during processExit when cudaFree will fail. The destruction of CUDA objects during
            // process exit must be avoided
            ReleaseStorageMemory();
        }
    }

    ZeroInit(GetComputeDeviceId());
}
#pragma endregion Constructors and Destructor

template <class ElemType>
std::unique_ptr<GPUMatrix<ElemType>> GPUMatrix<ElemType>::GetOrCreateWorkspace() const
{
    // REVIEW alexeyk: not thread-safe, fine for now.
    if (m_workspace == nullptr)
        m_workspace = std::make_unique<conc_stack<std::unique_ptr<GPUMatrix<ElemType>>>>();
    assert(m_workspace != nullptr);
    auto deviceId = GetComputeDeviceId();
    return m_workspace->pop_or_create([deviceId]()
                                      {
                                          return std::make_unique<GPUMatrix<ElemType>>(deviceId);
                                      });
}

template <class ElemType>
void GPUMatrix<ElemType>::ReleaseWorkspace(std::unique_ptr<GPUMatrix<ElemType>> src) const
{
    assert(m_workspace != nullptr);
    m_workspace->push(std::move(src));
}

#pragma region Basic Operators
template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::ColumnSlice(size_t startColumn, size_t numCols) const
{
    if (startColumn + numCols > GetNumCols())
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) GetNumCols());

    GPUMatrix<ElemType> slice(GetComputeDeviceId());

    slice.ShallowCopyFrom(*this);
    slice.m_numCols = numCols;
    slice.m_sliceViewOffset = m_sliceViewOffset + startColumn * GetNumRows();

    return slice;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignColumnSlice(const GPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols)
{
    if (numCols == 0)
        LogicError("The slice cannot have 0 columns.");

    if (startColumn + numCols > fromMatrix.GetNumCols())
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) fromMatrix.GetNumCols());

    Clear();

    ShallowCopyFrom(fromMatrix);
    m_numCols = numCols;
    m_sliceViewOffset = fromMatrix.m_sliceViewOffset + startColumn * GetNumRows();

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::SetColumnSlice(const GPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols)
{
    if (startColumn + numCols > GetNumCols())
        LogicError("The slice is out of range of the destination matrix.");
    if (numCols > fromMatrix.GetNumCols())
        InvalidArgument("The slice (%d) is out of range of the source matrix (%d).", (int) numCols, (int) fromMatrix.GetNumCols());
    if (m_numRows != fromMatrix.m_numRows)
        LogicError("The number of rows in source and destination matrices do not match");

    if (m_numRows * numCols > 0) // TODO: remove if unnecessary
        CUDA_CALL(cudaMemcpy(Data() + m_sliceViewOffset + LocateColumn(startColumn), fromMatrix.Data(), sizeof(ElemType) * m_numRows * numCols, cudaMemcpyDeviceToDevice));
    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::CopyColumnsStrided(const GPUMatrix<ElemType>& fromMatrix, size_t numCols, size_t srcNumColsStride, size_t destNumColsStride)
{
    if ((((numCols - 1) * srcNumColsStride) + 1) > fromMatrix.m_numCols)
        LogicError("The numCols to copy and srcNumColsStride specified is out of range of the source matrix.");
    if ((((numCols - 1) * destNumColsStride) + 1) > m_numCols)
        LogicError("The numCols to copy and srcNumColsStride specified is out of range of the destination matrix.");
    if (m_numRows != fromMatrix.m_numRows)
        LogicError("The number of rows in source and destination matrices do not match");

    if ((m_numRows * numCols) > 0)
    {
        // Launch a kernel to do the strided copy
        CUDA_LONG N = (CUDA_LONG)(m_numRows * numCols);
        int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
        PrepareDevice();
        SyncGuard syncGuard;
        _copyColumnsStrided<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), fromMatrix.Data(), N, (CUDA_LONG) m_numRows, (CUDA_LONG) destNumColsStride, (CUDA_LONG) srcNumColsStride);
    }
}

//for each column of a, we assign all rows of a to this starting from startIndex
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignToRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (a.IsEmpty())
        LogicError("AddToRowSliceValuesOf: input matrix a is empty.");

    if (a.GetNumRows() != numRows)
        LogicError("AddToRowSliceValuesOf: a.GetNumRows() != numRows.");

    if (startIndex + numRows > GetNumRows())
        LogicError("AddToRowSliceValuesOf: startIndex + numRows exceeds GetNumRows().");

    if (a.GetNumCols() != GetNumCols())
        LogicError("AddToRowSliceValuesOf: columns does not match.");

    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _assignToRowSliceValuesOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), N, (CUDA_LONG) startIndex, (CUDA_LONG) GetNumRows(), (CUDA_LONG) a.GetNumRows());
    return *this;
}

//for each column of a, we assign numRows starting from startIndex to this
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (a.IsEmpty())
        LogicError("AssignRowSliceValuesOf: input matrix a is empty.");

    if (startIndex + numRows > a.GetNumRows())
        LogicError("AssignRowSliceValuesOf: startIndex + numRows exceeds a.GetNumRows().");

    RequireSize(numRows, a.GetNumCols());

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _assignRowSliceValuesOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), N, (CUDA_LONG) startIndex, (CUDA_LONG) numRows, (CUDA_LONG) a.GetNumRows());
    return *this;
}

//for the row slice of this starting from startIndex we add a to it.
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddToRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (a.IsEmpty())
        LogicError("AddToRowSliceValuesOf: input matrix a is empty.");

    if (a.GetNumRows() != numRows)
        LogicError("AddToRowSliceValuesOf: a.GetNumRows() != numRows.");

    if (startIndex + numRows > GetNumRows())
        LogicError("AddToRowSliceValuesOf: startIndex + numRows exceeds GetNumRows().");

    if (a.GetNumCols() != GetNumCols())
        LogicError("AddToRowSliceValuesOf: columns does not match.");

    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _addToRowSliceValuesOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), N, (CUDA_LONG) startIndex, (CUDA_LONG) GetNumRows(), (CUDA_LONG) a.GetNumRows());
    return *this;
}

//for each column of this, we add row slice of a starting from startIndex
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddWithRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (a.IsEmpty())
        LogicError("AddWithRowSliceValuesOf: input matrix a is empty.");

    if (GetNumRows() != numRows)
        LogicError("AddWithRowSliceValuesOf: GetNumRows() != numRows.");

    if (startIndex + numRows > a.GetNumRows())
        LogicError("AddWithRowSliceValuesOf: startIndex + numRows exceeds a.GetNumRows().");

    if (a.GetNumCols() != GetNumCols())
        LogicError("AddWithRowSliceValuesOf: columns does not match.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _addWithRowSliceValuesOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), N, (CUDA_LONG) startIndex, (CUDA_LONG) GetNumRows(), (CUDA_LONG) a.GetNumRows());
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::Diagonal() const
{
    size_t m = GetNumRows();
    size_t n = GetNumCols();
    if (m != n)
        LogicError("Diagonal can be called only for square matrix. (rows=%d, cols=%d)", (int) m, (int) n);

    GPUMatrix<ElemType> diag(1, n, GetComputeDeviceId());

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _assignToDiagonalValuesOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(diag.Data(), Data(), N, (CUDA_LONG) n);
    return diag;
}

// c = c - 1.0 for a specific position
template <class ElemType>
void GPUMatrix<ElemType>::MinusOneAt(GPUMatrix<ElemType>& c, const size_t position)
{
    assert(position < c.GetNumElements());

    CUDA_LONG n = (CUDA_LONG) c.GetNumElements();
    CUDA_LONG p = (CUDA_LONG) position;

    int blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
    // BUGBUG: PrepareDevice() missing?
    SyncGuard syncGuard;
    _minusOneAt<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(c.Data(), p, n);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignRepeatOf(const GPUMatrix<ElemType>& a, const size_t numRowRepeats, const size_t numColRepeats)
{
    if (this == &a)
        LogicError("AssignRepeatOf: a is the same as [this]. Does not support inplace repeat.");

    if (a.IsEmpty())
        LogicError("AssignRepeatOf: Matrix a is empty.");

    RequireSize(a.GetNumRows() * numRowRepeats, a.GetNumCols() * numColRepeats);

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    CUDA_LONG n = (CUDA_LONG) a.GetNumCols(), m = (CUDA_LONG) a.GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _assignRepeatOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), N, m, n, (CUDA_LONG) GetNumRows());
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddToRowRepeatValuesOf(const GPUMatrix<ElemType>& a, const size_t numRepeats)
{
    if (a.IsEmpty())
        LogicError("AddToRowRepeatValuesOf: input matrix a is empty.");

    if (a.GetNumRows() != GetNumRows() * numRepeats)
        LogicError("AddToRowSliceValuesOf: a.GetNumRows() != GetNumRows() * numRepeats.");

    RequireSize(a.GetNumRows() / numRepeats, a.GetNumCols());

    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _addToRowRepeatValuesOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), N, (CUDA_LONG) a.GetNumRows(), (CUDA_LONG) a.GetNumCols(), (CUDA_LONG) GetNumRows());
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignPositiveAndShiftedNegSample(const GPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber)
{
    if (this == &a)
        LogicError("AssignPositiveAndShiftedNegSample: a is the same as [this]. Does not support inplace assignment.");

    if (a.IsEmpty())
        LogicError("AssignPositiveAndShiftedNegSample: Matrix a is empty.");

    RequireSize(a.GetNumRows() * (posNumber + negNumber), a.GetNumCols());

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    CUDA_LONG n = (CUDA_LONG) a.GetNumCols(), m = (CUDA_LONG) a.GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _assignPositiveAndShiftedNegSample<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), N, m, n, (CUDA_LONG) GetNumRows(), posNumber, shiftNumber);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddFoldedPositiveAndShiftedNegSample(const GPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber)
{
    if (this == &a)
        LogicError("AddFoldedPositiveAndShiftedNegSample: a is the same as [this]. Does not support inplace assignment.");

    if (a.IsEmpty())
        LogicError("AddFoldedPositiveAndShiftedNegSample: Matrix a is empty.");

    if (a.GetNumRows() != GetNumRows() * (posNumber + negNumber) || a.GetNumCols() != GetNumCols())
        LogicError("AddFoldedPositiveAndShiftedNegSample: dimensions mismatch.");

    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    CUDA_LONG n = (CUDA_LONG) a.GetNumCols(), m = (CUDA_LONG) a.GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _addFoldedPositiveAndShiftedNegSample<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), N, m, n, (CUDA_LONG) GetNumRows(), posNumber, shiftNumber);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::Transpose() const
{
    if (IsEmpty())
        LogicError("Transpose: Matrix is empty.");

    GPUMatrix<ElemType> c(GetComputeDeviceId());
    c.AssignTransposeOf(*this);
    return c;
}

// GetCublasHandle - get a cublas handle for the given GPU, should only need one per GPU
// computeDevice - The compute device for which the cublas handle is desired
// returns: cublas handle
// NOTE: we currently don't bother to ever free the CUBLAS handle, it will be freed automatically by CUDA when the process ends
template <class ElemType>
cublasHandle_t GPUMatrix<ElemType>::GetCublasHandle(int computeDevice /*=-1*/)
{
    // if the compute device is not passed, get the current device from CUDA
    if (computeDevice < 0)
        cudaGetDevice(&computeDevice);

    if (computeDevice < 0 || computeDevice >= MaxGpus)
        LogicError("GetCublasHandle: Maximum GPU exceeded");
    cublasHandle_t cuHandle = s_cuHandle[computeDevice];
    if (cuHandle == NULL)
    {
        s_cuHandle[computeDevice] = cuHandle = _initCUBLAS<ElemType>(computeDevice);
    }
    CUBLAS_CALL(cublasSetStream(cuHandle, t_stream));

    return cuHandle;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignTransposeOf(const GPUMatrix<ElemType>& a)
{
    if (this == &a)
        LogicError("AssignTransposeOf: a is the same as [this]. Does not support inplace transpose.");

    if (a.IsEmpty())
        LogicError("AssignTransposeOf: Matrix a is empty.");

    if (GetNumRows() != a.GetNumCols() || GetNumCols() != a.GetNumRows())
        RequireSize(a.GetNumCols(), a.GetNumRows());

    cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_T;
    int m = (int) a.m_numCols;
    int n = (int) a.m_numRows;
    ElemType alpha = 1;
    ElemType beta = 0;
    cublasStatus_t st;
    if (sizeof(ElemType) == sizeof(float))
        st = cublasSgeam(cuHandle, transA, transB, m, n, reinterpret_cast<float*>(&alpha), reinterpret_cast<float*>(a.Data()), (int) a.m_numRows, reinterpret_cast<float*>(&beta), reinterpret_cast<float*>(a.Data()), (int) a.m_numRows, reinterpret_cast<float*>(Data()), (int) m_numRows);
    else if (sizeof(ElemType) == sizeof(double))
        st = cublasDgeam(cuHandle, transA, transB, m, n, reinterpret_cast<double*>(&alpha), reinterpret_cast<double*>(a.Data()), (int) a.m_numRows, reinterpret_cast<double*>(&beta), reinterpret_cast<double*>(a.Data()), (int) a.m_numRows, reinterpret_cast<double*>(Data()), (int) m_numRows);
    else
        RuntimeError("Unsupported template argument in GPUMatrix");
    if (st != CUBLAS_STATUS_SUCCESS)
        RuntimeError("AssignTransposeOf failed");
    m_numRows = a.m_numCols;
    m_numCols = a.m_numRows;
    return *this;
}

template <class ElemType>
__global__ void _doGatherColumnsOf(ElemType* us, size_t usStride, const ElemType beta, const ElemType* idx, size_t idxStride, const ElemType* a, size_t aStride, size_t aCols, const ElemType alpha, CUDA_LONG numElements)
{
    CUDA_LONG id = GridDim::GetLinearThreadId();
    if (id >= numElements) // note: there are no __syncthread() calls inside
        return;

    // id = i + jOut * usStride;
    // Each thread processes one element of the output matrix.
    CUDA_LONG i    = id % usStride; // row index into 'us' and 'a'
    CUDA_LONG jOut = id / usStride; // col index into 'us' and 'idx'

    auto jInF = idx[jOut * idxStride]; // this is the column we need to get
    if (::isnan(jInF) || jInF < 0)     // negative index means gap
        return;
    size_t jIn = (size_t)jInF;
    //if (jIn >= aCols)
    //    return; // actually a failure

    const ElemType&  ra = a[    i + jIn  *  aStride  ];
    ElemType&       rus = us[id/*i + jOut * usStride*/];

    ElemType res = ra * alpha;
    if (beta != 0)
        res += rus * beta;
    rus = res;
}

// *this[:,j] = a[:,idx[j]] * alpha + *this[:,j] * beta
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::DoGatherColumnsOf(ElemType beta, const GPUMatrix<ElemType>& idx, const GPUMatrix<ElemType>& a, ElemType alpha)
{
    if (idx.GetNumRows() != 1) // index is 1-dimensional only
        InvalidArgument("DoGatherColumnsOf: Map must be a row vector.");

    if (beta == 0)
        RequireSize(a.GetNumRows(), idx.GetNumCols()); // output has same column format as a, but number of columns comes from idx
    else
        VerifySize(a.GetNumRows(), idx.GetNumCols());

    if (idx.GetComputeDeviceId() != a.GetComputeDeviceId() || GetComputeDeviceId() != a.GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");
    a.PrepareDevice();

    // launch the kernel
    CUDA_LONG NN = (CUDA_LONG)GetNumElements(); // linear space identifying each individual input element
    SyncGuard syncGuard;
    GridDim grid(NN);
    _doGatherColumnsOf<ElemType><<<grid.m_blocksPerGrid, grid.m_threadsPerBlock, 0, t_stream>>>(Data(), GetNumRows(), beta, idx.Data(), idx.GetNumRows(), a.Data(), a.GetNumRows(), a.GetNumCols(), alpha, grid.m_N);

    // Note: The following fails silently (no error, immediate or delayed) for numcols = 10000 under CUDA 7.0.
    //_doGatherColumnsOf<ElemType><<<GetNumCols(), GetNumRows(), 0, t_stream>>>(Data(), GetNumRows(), beta, idx.Data(), idx.GetNumRows(), a.Data(), a.GetNumRows(), a.GetNumCols(), alpha);

    return *this;
}

// little helper for debugging
template <class ElemType>
static void Peek(const GPUMatrix<ElemType>& m, const char* which)
{
    size_t rows = m.GetNumRows();
    size_t cols = m.GetNumCols();
    ElemType buf[10000] = { 0 };
    size_t n = min(rows * cols, _countof(buf));
    CUDA_CALL(cudaMemcpy(buf, m.Data(), sizeof(ElemType) * n, cudaMemcpyDeviceToHost));
    UNUSED(which); UNUSED(rows); UNUSED(cols); sin(1.0f); // set breakpoint here
    //CUDA_CALL(cudaMemcpy(const_cast<ElemType*>(m.Data()), buf, sizeof(ElemType) * n, cudaMemcpyHostToDevice));
}

#define ALLOW_ATOMIC_SCATTER // allow to disable this, until we know atomicAdd() works properly here

template <class ElemType>
__global__ void _doScatterColumnsOf(ElemType* us, size_t usStride, size_t usCols, const ElemType* idx, size_t idxStride, const ElemType* a, size_t aStride, const ElemType alpha, CUDA_LONG numElements)
{
    CUDA_LONG id = GridDim::GetLinearThreadId();
    if (id >= numElements) // note: there are no __syncthread() calls inside
        return;

    // id = i + jIn  *  aStride
    // Each thread processes one element of a
    CUDA_LONG i   = id % aStride; // row index into 'a' and 'us'
    CUDA_LONG jIn = id / aStride; // col index into 'a' and 'idx'

    auto jOutF = idx[jIn * idxStride];  // this is the column we copy/add into
    if (::isnan(jOutF) || jOutF < 0)    // negative index means gap
        return;
    size_t jOut = (size_t)jOutF;
    //if (jOut >= usCols)
    //    return; // actually a failure  --TODO: This should not be necessary. Why is it?

    const ElemType&  ra =  a[id/*i + jIn  *  aStride*/];
    ElemType&       rus = us[    i + jOut * usStride  ];

    ElemType res = ra * alpha;
    if (res != 0)             // avoid memory conflict if e.g. an entire column has no gradient
#ifdef ALLOW_ATOMIC_SCATTER
        atomicAdd(&rus, res); // rus += res;
#else
        rus += res;
#endif
    // Note: atomicAdd() is supposed to be fast in case of no conflict (the simple case of Scatter())
}

// *this[:,idx[j]] = a[:,j] * alpha + *this[:,idx[j]] * beta
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::DoScatterColumnsOf(ElemType beta, const GPUMatrix<ElemType>& idx, const GPUMatrix<ElemType>& a, ElemType alpha)
{
    if (idx.GetNumRows() != 1) // index is 1-dimensional only
        InvalidArgument("DoScatterColumnsOf: Map must be a row vector.");
    if (idx.GetNumCols() != a.GetNumCols())
        InvalidArgument("DoScatterColumnsOf: Map must have width of input vector.");
    if (a.GetNumRows() != GetNumRows())
        InvalidArgument("DoScatterColumnsOf: Output must have same height as input vector.");

    if (idx.GetComputeDeviceId() != a.GetComputeDeviceId() || GetComputeDeviceId() != a.GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");
    a.PrepareDevice();

    auto& us = *this;

#ifndef ALLOW_ATOMIC_SCATTER // verify that atomicAdd is not needed  --this is not efficient
    {
        vector<ElemType> buf(idx.GetNumRows() * idx.GetNumCols()); // idx(,)are the column(s) we copy/add into
        CUDA_CALL(cudaMemcpy(buf.data(), idx.Data(), sizeof(ElemType) * buf.size(), cudaMemcpyDeviceToHost));
        vector<bool> writtenTo(GetNumCols(), false); // remember whether an output column is in fact a target
        for (size_t i = 0; i < buf.size(); i++)
        {
            auto colF = buf[i];
            if (std::isnan(colF) || colF < 0)
                continue;
            size_t col = (size_t)colF;
            if (col >= GetNumCols())
                LogicError("DoScatterColumnsOf: Index value out of bounds.");
            if (writtenTo[col])
                LogicError("DoScatterColumnsOf: #ifndef ALLOW_ATOMIC_SCATTER then columns must be unique. Column idx(%d,%d)=%d is used twice.", (int)(i % idx.GetNumCols()), (int)(i / idx.GetNumCols()), (int)col);
            else
                writtenTo[col] = true;
        }
    }
#endif

    // pre-scale with beta upfront
    // Scatter may add more than one source column to the same target, so we must pre-scale with beta, and then just keep adding.
    Scale(beta, us); // if beta is 0, then this will be a memset()

    // launch the kernel
    CUDA_LONG NN = (CUDA_LONG)(a.GetNumElements()); // linear space identifying each individual input element
    SyncGuard syncGuard;
    GridDim grid(NN);
    _doScatterColumnsOf<ElemType><<<grid.m_blocksPerGrid, grid.m_threadsPerBlock, 0, t_stream>>>(Data(), GetNumRows(), GetNumCols(), idx.Data(), idx.GetNumRows(), a.Data(), a.GetNumRows(), alpha, NN);

    //SyncGuard syncGuard;
    //_doScatterColumnsOf<ElemType><<<a.GetNumCols(), a.GetNumRows(), 0, t_stream>>>(Data(), GetNumRows(), GetNumCols(), idx.Data(), idx.GetNumRows(), a.Data(), a.GetNumRows(), alpha, NN);

    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const ElemType v)
{
    if (IsEmpty())
        return;

    CUDA_LONG N = (CUDA_LONG) GetNumElements();

    // Check if value is zero, which can be set using cudaMemset
    bool isZero = true;
    const char* valArray = reinterpret_cast<const char*>(&v);

    for (int i = 0; i < sizeof(ElemType); i++)
    {
        if (valArray[i] != 0)
        {
            isZero = false;
            break;
        }
    }

    if (isZero)
    {
        CUDA_CALL(cudaMemset(Data(), 0, N * sizeof(ElemType)));
    }
    else
    {
        int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
        PrepareDevice();
        SyncGuard syncGuard;
        _setValue<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), v, N);
    }
}

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const ElemType* d_v) // d_v is pointer to the the value in GPU memory
{
    if (IsEmpty())
        LogicError("SetValue: Matrix is empty.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _setValue<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), d_v, N);
}

template <class ElemType>
void GPUMatrix<ElemType>::MaskColumnsValue(const GPUMatrix<char>& columnsMask, ElemType val, size_t numColsPerMaskEntry)
{
    if (GetNumCols() != (columnsMask.GetNumCols() * numColsPerMaskEntry))
        RuntimeError("Matrix number of columns must equal 'number of columns in column mask * numColsPerMaskEntry'.");

    if (GetComputeDeviceId() != columnsMask.GetComputeDeviceId())
        RuntimeError("Matrix and column mask must be on the same device");

    int blocksPerGrid = (int)columnsMask.GetNumCols();
    PrepareDevice();
    SyncGuard syncGuard;
    _maskColumnsValue<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), columnsMask.Data(), (CUDA_LONG) GetNumCols(), (CUDA_LONG) GetNumRows(), val, numColsPerMaskEntry);
}

template <class ElemType>
void GPUMatrix<ElemType>::SetColumn(const ElemType* colPointer, size_t colInd)
{
    if (IsEmpty())
        LogicError("SetValue: Matrix is empty.");
    if (colPointer == NULL)
        return;
    CUDA_CALL(cudaMemcpy(Data() + LocateColumn(colInd), colPointer, sizeof(ElemType) * m_numRows, cudaMemcpyHostToDevice));
}

template <class ElemType>
void GPUMatrix<ElemType>::SetColumn(const GPUMatrix<ElemType>& valMat, size_t colInd)
{
    if (IsEmpty())
        LogicError("SetColumn: Matrix is empty.");
    if (valMat.GetNumCols() != 1)
        LogicError("SetColumn: only support one column matrix now.");
    CUDA_CALL(cudaMemcpy(Data() + LocateColumn(colInd), valMat.Data(), sizeof(ElemType) * m_numRows, cudaMemcpyDeviceToDevice));
}

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const GPUMatrix<ElemType>& deepCopyFrom)
{
    if (this == &deepCopyFrom)
        return;

    SetValue(deepCopyFrom.GetNumRows(), deepCopyFrom.GetNumCols(), deepCopyFrom.GetComputeDeviceId(), deepCopyFrom.Data(), matrixFlagSetValueOnDevice);
}

#if 0
template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const CPUMatrix<ElemType>& /*deepCopyFrom*/)
{
    NOT_IMPLEMENTED;
}

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const CPUSparseMatrix<ElemType>& /*deepCopyFrom*/)
{
    NOT_IMPLEMENTED;
}

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const GPUSparseMatrix<ElemType>& deepCopyFrom)
{
    deepCopyFrom.CopyToDenseMatrix(*this);
}
#endif

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const size_t numRows, const size_t numCols, int deviceId, ElemType* pArray, size_t matrixFlags, DataTransferer* transferer)
{
    // handle externally managed case
    // BUGBUG: This is super super ugly, and needs to be fixed, but if matrixFlags has the right value, then we can't free anything,
    // and everything gets wonky. This should be fixed, and would go away if it is made a shared_ptr.
    if (matrixFlags & matrixFlagDontOwnBuffer)
    {
        // free the existing array if it used to be an owned array
        if ( Buffer() != NULL)
        {
            TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), Buffer());
        }
        m_numRows = numRows;
        m_numCols = numCols;
        SetBuffer(pArray, GetNumElements() * sizeof(ElemType), true);
        SetSizeAllocated(GetNumElements());
        SetFormat(matrixFormatDense);
        SetComputeDeviceId(deviceId);
    }
    else
    {
        if (transferer && (matrixFlags & matrixFlagSetValueOnDevice))
            RuntimeError("Asynchronous data copy from device to device is currently not supported.");

        // if the devices are different move it now
        if (GetComputeDeviceId() != deviceId && deviceId >= 0)
        {
            Clear();
            ZeroInit(deviceId);
        }

        // now RequireSize/allocate as necessary
        RequireSize(numRows, numCols);

        // copy over the content to the buffer
        PrepareDevice();
        if (pArray != NULL)
        {
            if (!(matrixFlags & matrixFormatRowMajor))
            {
                if (transferer)
                    transferer->CopyCPUToGPUAsync(pArray, GetNumElements(), sizeof(ElemType), Data());
                else
                    CUDA_CALL(cudaMemcpy(Data(), pArray, sizeof(ElemType) * GetNumElements(), (matrixFlags & matrixFlagSetValueOnDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
            }
            else // row major: must transpose (this is not meant to be efficient, but very useful for defining inline matrices for test code)
            {
                vector<ElemType> transposed(GetNumElements());
                for (size_t i = 0; i < numRows; i++)
                    for (size_t j = 0; j < numCols; j++)
                        transposed[i + numRows * j] = pArray[j + numCols * i];

                if (transferer)
                    transferer->CopyCPUToGPUAsync(transposed.data(), GetNumElements(), sizeof(ElemType), Data());
                else
                    CUDA_CALL(cudaMemcpy(Data(), transposed.data(), sizeof(ElemType) * GetNumElements(), (matrixFlags & matrixFlagSetValueOnDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
            }
        }
    }
    SetFormat(matrixFormatDense);
}

template <class ElemType>
void GPUMatrix<ElemType>::SetDiagonalValue(const ElemType v)
{
    CUDA_LONG N = (CUDA_LONG) GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _setDiagonalValue<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), v, N, (CUDA_LONG) GetNumRows());
}

template <class ElemType>
void GPUMatrix<ElemType>::SetDiagonalValue(const GPUMatrix<ElemType>& vector)
{
    if (IsEmpty() || vector.IsEmpty())
        LogicError("SetDiagonalValue: Matrix is empty.");

    if (GetNumRows() != GetNumCols())
        LogicError("SetDiagonalValue: NumRows and NumCols do not agree.");

    if (vector.GetNumRows() != 1 && vector.GetNumCols() != 1)
        LogicError("SetDiagonalValue: input vector must be a vector.");

    if (vector.GetNumElements() == 1) // reduce to simple form
        SetDiagonalValue(vector.Data()[0]);

    else if (vector.GetNumRows() != GetNumRows() && vector.GetNumCols() != GetNumRows())
        LogicError("SetDiagonalValue: input vector's dimension does not agree with [this].");
    else
    {
        CUDA_LONG N = (CUDA_LONG) GetNumRows();
        int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
        PrepareDevice();
        SyncGuard syncGuard;
        _setDiagonalValueFromVector<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), vector.Data(), N);
    }
}

template <class ElemType>
void GPUMatrix<ElemType>::SetUniformRandomValue(const ElemType low, const ElemType high, unsigned long seed)
{
    PrepareDevice();
    CreateCurandObject(seed, __FUNCTION__); // TODO call ResetCurandObject() instead?

    cudaEvent_t done = nullptr;
    CUDA_CALL(cudaEventCreate(&done)); // TODO: why not condition on do_sync, so that we can use SyncGuard?
    if (sizeof(ElemType) == sizeof(float))
        CURAND_CALL(curandGenerateUniform(((curandGenerator_t*) s_curandGenerator)[0], reinterpret_cast<float*>(Data()), GetNumElements()));
    else
        CURAND_CALL(curandGenerateUniformDouble(((curandGenerator_t*) s_curandGenerator)[0], reinterpret_cast<double*>(Data()), GetNumElements()));
    CUDA_CALL(cudaEventRecord(done));
    CUDA_CALL(cudaEventSynchronize(done));
    // CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaEventDestroy(done));

    size_t N = GetNumElements();
    size_t blocksPerGrid = (size_t) ceil(N / (double) GridDim::maxThreadsPerBlock);

    SyncGuard syncGuard;
    _rescaleToRange<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), N, low, high);
}

template <class ElemType>
void GPUMatrix<ElemType>::SetGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed)
{
    PrepareDevice();
    CreateCurandObject(seed, __FUNCTION__); // TODO call ResetCurandObject() instead?

    // TODO: Why not use SyncGuard?
    if (sizeof(ElemType) == sizeof(float))
        CURAND_CALL(curandGenerateNormal(((curandGenerator_t*) s_curandGenerator)[0], reinterpret_cast<float*>(Data()), GetNumElements(), (float) mean, (float) sigma));
    else
        CURAND_CALL(curandGenerateNormalDouble(((curandGenerator_t*) s_curandGenerator)[0], reinterpret_cast<double*>(Data()), GetNumElements(), (double) mean, (double) sigma));
    // CURAND_CALL(curandDestroyGenerator(gen));
}

//maskRate: percentage of values masked out (similar to dropout rate)
//scaleValue: which scale value to set to the left ones (unmasked items).
template <class ElemType>
void GPUMatrix<ElemType>::SetUniformRandomMask(const ElemType maskRate, const ElemType scaleValue, RNGHandle& rngHandle)
{
    PrepareDevice();

    GPURNGHandle* gpuRNGHandle = dynamic_cast<GPURNGHandle*>(&rngHandle);
    assert(gpuRNGHandle != nullptr);

    cudaEvent_t done = nullptr;
    CUDA_CALL(cudaEventCreate(&done)); // TODO: why not condition on do_sync, so that we can use SyncGuard?
    if (sizeof(ElemType) == sizeof(float))
        CURAND_CALL(curandGenerateUniform(gpuRNGHandle->Generator(), reinterpret_cast<float*>(Data()), GetNumElements()));
    else
        CURAND_CALL(curandGenerateUniformDouble(gpuRNGHandle->Generator(), reinterpret_cast<double*>(Data()), GetNumElements()));
    CUDA_CALL(cudaEventRecord(done));
    CUDA_CALL(cudaEventSynchronize(done));
    CUDA_CALL(cudaEventDestroy(done));

    size_t N = GetNumElements();
    size_t blocksPerGrid = (size_t) ceil(N / (double) GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _setMaskAndScale<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), N, maskRate, scaleValue);
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::Adagrad(GPUMatrix<ElemType>& gradients, const bool needAveMultiplier)
{
    size_t numColsNeeded = gradients.GetNumCols();
    if (needAveMultiplier)
        numColsNeeded += gradients.GetNumCols();

    if (IsEmpty() || GetNumCols() < numColsNeeded)
    {
        RequireSize(gradients.GetNumRows(), numColsNeeded);
        SetValue(0.0);
    }

    assert(GetNumRows() == gradients.GetNumRows() && GetNumCols() == numColsNeeded);

    size_t n = gradients.GetNumElements();

    ElemType* multipliers = nullptr;
    if (needAveMultiplier)
        multipliers = Data() + n; // temp memory used to store multipliers,

    int blocksPerGrid = (n + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    _adagrad<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(Data(), gradients.Data(), n, multipliers);

    if (!needAveMultiplier)
        return 1;

    cublasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());
    if (sizeof(ElemType) == sizeof(float))
    {
        float aveMultiplier = 0;
        CUBLAS_CALL(cublasSasum(cuHandle, (CUDA_LONG) n, reinterpret_cast<float*>(multipliers), 1, &aveMultiplier));
        return (ElemType) aveMultiplier / n;
    }
    else
    {
        double aveMultiplier = 0;
        CUBLAS_CALL(cublasDasum(cuHandle, (CUDA_LONG) n, reinterpret_cast<double*>(multipliers), 1, &aveMultiplier));
        return (ElemType) aveMultiplier / n;
    }
}

template <class ElemType>
void GPUMatrix<ElemType>::FSAdagrad(GPUMatrix<ElemType>& gradients,
                                    GPUMatrix<ElemType>& functionValues,
                                    ElemType learnRatePerSample,
                                    ElemType momentum,
                                    ElemType adaWeight,
                                    ElemType adaMul,
                                    bool unitGainMomentum)
{
    size_t numColsNeeded = 2 * gradients.GetNumCols();

    if (IsEmpty() || (GetNumCols() < numColsNeeded))
    {
        RequireSize(gradients.GetNumRows(), numColsNeeded);
        SetValue(0.0);
    }

    assert((GetNumRows() == gradients.GetNumRows()) && (GetNumCols() == numColsNeeded));

    size_t n = gradients.GetNumElements();
    int blocksPerGrid = (n + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    _fsadagrad<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(n, gradients.Data(), Data(), Data()+ n, functionValues.Data(),
                                                                         learnRatePerSample, momentum, adaWeight, adaMul, unitGainMomentum);
}

template <class ElemType>
void GPUMatrix<ElemType>::Adam(GPUMatrix<ElemType>& gradients,
    GPUMatrix<ElemType>& functionValues,
    ElemType learnRatePerSample,
    ElemType momentum,
    ElemType adaWeight,
    ElemType adaMul,
    bool unitGainMomentum)
{
    size_t numColsNeeded = 2 * gradients.GetNumCols();

    if (IsEmpty() || (GetNumCols() < numColsNeeded))
    {
        RequireSize(gradients.GetNumRows(), numColsNeeded);
        SetValue(0.0);
    }

    assert((GetNumRows() == gradients.GetNumRows()) && (GetNumCols() == numColsNeeded));

    size_t n = gradients.GetNumElements();
    int blocksPerGrid = (n + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    _adam<ElemType> << <blocksPerGrid, GridDim::maxThreadsPerBlock >> >(n, gradients.Data(), Data(), Data() + n, functionValues.Data(),
        learnRatePerSample, momentum, adaWeight, adaMul, unitGainMomentum);
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::RmsProp(GPUMatrix<ElemType>& gradients,
                                      ElemType RMS_GAMMA,
                                      ElemType RMS_WGT_INC,
                                      ElemType RMS_WGT_MAX,
                                      ElemType RMS_WGT_DEC,
                                      ElemType RMS_WGT_MIN,
                                      const bool needAveMultiplier)
{
    const ElemType floor = 1e-6f;
    static ElemType* upd_gpu = (ElemType*) 0;

    size_t n = gradients.GetNumElements();
    int blocksPerGrid = (GetNumElements() + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;

    size_t numColsNeeded = gradients.GetNumCols() * 3;
    if (needAveMultiplier)
        numColsNeeded += gradients.GetNumCols();

    if (IsEmpty() || GetNumCols() < numColsNeeded)
    {
        RequireSize(gradients.GetNumRows(), numColsNeeded);
        SetValue(0.0);

        ElemType* avars = Data();         // accumulated variances for RMS scaling
        ElemType* signs = Data() + n;     // sign of previous gradient
        ElemType* steps = Data() + 2 * n; // current step size
        // Data()+3*n is temp memory used to store multipliers, no need to initialize

        _rmsprop_init<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(avars, signs, steps, gradients.Data(), n);
    }
    assert(GetNumRows() == gradients.GetNumRows() && GetNumCols() == numColsNeeded);

    ElemType* avars = Data();         // accumulated variances for RMS scaling
    ElemType* signs = Data() + n;     // sign of previous gradient
    ElemType* steps = Data() + 2 * n; // current step size

    ElemType* multipliers = nullptr;
    if (needAveMultiplier)
        multipliers = Data() + 3 * n; // temp memory used to store multipliers,

    if (!upd_gpu)
    {
        const ElemType upd[] = {
            2, 2, 0,
            2, 2, 0,
            1, 1, 1,
            2, 2, 0,
            1, 2, 1,
            0, 2, 2,
            1, 1, 1,
            0, 2, 2,
            0, 2, 2,
        };

        upd_gpu = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 27);
        CUDA_CALL(cudaMemcpy(upd_gpu, upd, sizeof(ElemType) * _countof(upd), cudaMemcpyHostToDevice));
    }

    _rmsprop<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(avars, signs, steps, gradients.Data(), n,
                                                                       RMS_GAMMA, RMS_WGT_INC, RMS_WGT_MAX, RMS_WGT_DEC, RMS_WGT_MIN,
                                                                       floor, upd_gpu, multipliers);

    if (!needAveMultiplier)
        return 1;

    cublasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());
    if (sizeof(ElemType) == sizeof(float))
    {
        float aveMultiplier = 0;
        CUBLAS_CALL(cublasSasum(cuHandle, (CUDA_LONG) n, reinterpret_cast<float*>(multipliers), 1, &aveMultiplier));
        return aveMultiplier / n;
    }
    else
    {
        double aveMultiplier = 0;
        CUBLAS_CALL(cublasDasum(cuHandle, (CUDA_LONG) n, reinterpret_cast<double*>(multipliers), 1, &aveMultiplier));
        return (ElemType) aveMultiplier / n;
    }
}

template <class ElemType>
void GPUMatrix<ElemType>::AdaDelta(GPUMatrix<ElemType>& gradients, GPUMatrix<ElemType>& functionValues, ElemType rho, ElemType epsilon)
{
    size_t numColsNeeded = 2 * gradients.GetNumCols();

    if (IsEmpty() || (GetNumCols() < numColsNeeded))
    {
        RequireSize(gradients.GetNumRows(), numColsNeeded);
        SetValue(0.0);
    }

    assert((GetNumRows() == gradients.GetNumRows()) && (GetNumCols() == numColsNeeded));

    size_t n = gradients.GetNumElements();
    int blocksPerGrid = (n + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    _adadelta<ElemType> << <blocksPerGrid, GridDim::maxThreadsPerBlock >> >(n, gradients.Data(), Data(), Data() + n, functionValues.Data(), rho, epsilon);
}

template <class ElemType>
void GPUMatrix<ElemType>::Reshape(const size_t numRows, const size_t numCols)
{
    assert(numRows * numCols == GetNumElements());
    if (numRows * numCols != GetNumElements())
        InvalidArgument("Reshape: total number of elements does not match.");

    m_numRows = numRows;
    m_numCols = numCols;
}

template <class ElemType>
void GPUMatrix<ElemType>::RequireSize(const size_t numRows, const size_t numCols, bool growOnly)
{
    if (GetNumRows() != numRows || GetNumCols() != numCols)
        Resize(numRows, numCols, growOnly);
}

template <class ElemType>
void GPUMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, bool growOnly)
{
    if (GetNumRows() == numRows && GetNumCols() == numCols)
        return;

    VerifyResizable(__FUNCTION__);

    size_t numElements = numRows * numCols;
    if (numElements > GetSizeAllocated() ||                     // grow allocation
        (!growOnly && numElements != GetSizeAllocated()))   // shrink allocation if not growOnly
    {
        // If the buffer exists, free it before allocate
        if (Buffer())
        {
            TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), Buffer());
        }

        // reallocate buffer if numElements > 0
        ElemType* pArray = nullptr;
        if (numElements > 0)
        {
            pArray = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), numRows, numCols);
        }

        SetBuffer(pArray, numElements * sizeof(ElemType));
        SetSizeAllocated(numElements);
    }
    
    // success
    m_sliceViewOffset = 0;
    m_numRows = numRows;
    m_numCols = numCols;
}

template <class ElemType>
size_t GPUMatrix<ElemType>::LocateElement(const size_t row, const size_t col) const
{
    assert(row < m_numRows && col < m_numCols);
    return LocateColumn(col) + row; // matrix in column-wise storage
}

template <class ElemType>
size_t GPUMatrix<ElemType>::LocateColumn(const size_t col) const
{
    assert(col < GetNumCols());
    return col * m_numRows; // matrix in column-wise storage
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::Get00Element() const
{
    ElemType res = 0;
    CUDA_CALL(cudaMemcpy(&res, Data(), sizeof(ElemType), cudaMemcpyDeviceToHost));
    return res;
}
#pragma endregion Basic Operators

#pragma region Member BLAS Functions
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator+=(ElemType alpha)
{
    if (IsEmpty())
        LogicError("operator+=: Matrix is empty.");
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _addValue<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), alpha, N);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator+(ElemType alpha) const
{
    if (IsEmpty())
        LogicError("operator+: Matrix is empty.");

    GPUMatrix<ElemType> c(*this);
    c += alpha;
    return c;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSumOf(const ElemType alpha, const GPUMatrix<ElemType>& a)
{
    SetValue(a);
    (*this) += alpha;
    return (*this);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator+=(const GPUMatrix<ElemType>& a)
{
    ScaleAndAdd(1, a, *this);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator+(const GPUMatrix<ElemType>& a) const
{
    if (GetNumElements() == 1)
    {
        GPUMatrix<ElemType> c(a);
        c += Get00Element();
        return c;
    }
    else if (a.GetNumElements() == 1)
    {
        GPUMatrix<ElemType> c(*this);
        c += a.Get00Element();
        return c;
    }
    else
    {
        GPUMatrix<ElemType> c(*this); // this implementation will introduce a copy overhead. but make resue of the code
        c += a;
        return c;
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSumOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    SetValue(a);
    (*this) += b;
    return (*this);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator-=(ElemType alpha)
{
    if (IsEmpty())
        LogicError("operato-=: Matrix is empty.");
    return operator+=(-1 * alpha);
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator-(ElemType alpha) const
{
    if (IsEmpty())
        LogicError("operator-: Matrix is empty.");
    return operator+(-1 * alpha);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignDifferenceOf(const ElemType alpha, const GPUMatrix<ElemType>& a)
{
    RequireSize(a.m_numRows, a.m_numCols);
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _assignDifferenceOf1<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), alpha, a.Data(), N);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignDifferenceOf(const GPUMatrix<ElemType>& a, const ElemType alpha)
{
    RequireSize(a.m_numRows, a.m_numCols);
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _assignDifferenceOf2<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), alpha, a.Data(), N);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator-=(const GPUMatrix<ElemType>& a)
{
    ScaleAndAdd(-1, a, *this);

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator-(const GPUMatrix<ElemType>& a) const
{
    GPUMatrix<ElemType> c(*this); // this implementation will introduce a copy overhead. but make resue of the code
    c -= a;
    return c;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignDifferenceOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (this != &a)
    {
        RequireSize(a.GetNumRows(), a.GetNumCols());
        SetValue(a);
    }
    (*this) -= b;
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator*=(ElemType alpha)
{
    Scale(alpha, *this);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator*(ElemType alpha) const
{
    GPUMatrix<ElemType> c(GetNumRows(), GetNumCols(), GetComputeDeviceId());
    Scale(alpha, *this, c);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignProductOf(const ElemType alpha, const GPUMatrix<ElemType>& a)
{
    Scale(alpha, a, *this);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignProductOf(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB)
{
    if (a.GetNumElements() == 1)
    {
        if (transposeB)
            AssignTransposeOf(b);
        (*this) *= a.Get00Element();
    }
    else if (b.GetNumElements() == 1)
    {
        if (transposeA)
            AssignTransposeOf(a);
        (*this) *= b.Get00Element();
    }
    else
        Multiply(a, transposeA, b, transposeB, *this);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator*(const GPUMatrix<ElemType>& a) const
{
    const GPUMatrix<ElemType>& us = *this;
    if (GetNumElements() == 1)
    {
        GPUMatrix<ElemType> c(GetComputeDeviceId());
        c.AssignProductOf(Get00Element(), a);
        return c;
    }
    else if (a.GetNumElements() == 1)
    {
        GPUMatrix<ElemType> c(GetComputeDeviceId());
        c.AssignProductOf(a.Get00Element(), us);
        return c;
    }
    else
    {
        GPUMatrix<ElemType> c(GetNumRows(), a.GetNumCols(), GetComputeDeviceId());
        Multiply(*this, a, c);
        return c;
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator/=(ElemType alpha)
{
    (*this) *= 1 / alpha;
    return (*this);
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator/(ElemType alpha) const
{
    return ((*this) * (1 / alpha));
}

//element-wise power
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator^=(ElemType alpha)
{
    GPUMatrix<ElemType>& us = *this;
    ElementWisePower(alpha, us, us);
    return us;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator^(ElemType alpha) const
{
    GPUMatrix<ElemType> c(GetNumRows(), GetNumCols(), GetComputeDeviceId());
    ElementWisePower(alpha, *this, c);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementPowerOf(const GPUMatrix<ElemType>& a, const ElemType power)
{
    ElementWisePower(power, a, *this);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddElementProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AddElementProductOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match [this].");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _addElementProductOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), b.Data(), N);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::ColumnElementMultiplyWith(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("ColumnElementMultiplyWith: Matrix is empty.");

    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
        InvalidArgument("ColumnElementMultiplyWith: The input matrix should be a col vector and match [this]'s rows.");

    CUDA_LONG N = (CUDA_LONG) a.GetNumRows();
    CUDA_LONG M = (CUDA_LONG) GetNumCols();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _columnElementMultiplyWith<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), N, M);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::RowElementMultiplyWith(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("RowElementMultiplyWith: Matrix is empty.");

    if (!(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols()))
        InvalidArgument("RowElementMultiplyWith: The input matrix should be a row vector and match [this]'s columns.");

    CUDA_LONG N = (CUDA_LONG) GetNumRows();
    CUDA_LONG M = (CUDA_LONG) a.GetNumCols();
    int blocksPerGrid = (int) ceil(1.0 * M / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _rowElementMultiplyWith<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(Data(), a.Data(), N, M);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::RowElementDivideBy(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("RowElementDivideBy: Matrix is empty.");

    if (!(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols()))
        InvalidArgument("RowElementDivideBy: The input matrix should be a row vector and match [this]'s columns.");

    CUDA_LONG N = (CUDA_LONG) GetNumRows();
    CUDA_LONG M = (CUDA_LONG) a.GetNumCols();
    int blocksPerGrid = (int) ceil(1.0 * M / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _rowElementDivideBy<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(Data(), a.Data(), N, M);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::ColumnElementDivideBy(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("ColumnElementDivideBy: Matrix is empty.");

    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
        InvalidArgument("ColumnElementDivideBy: The input matrix should be a col vector and match [this]'s rows.");

    CUDA_LONG N = (CUDA_LONG) a.GetNumRows();
    CUDA_LONG M = (CUDA_LONG) GetNumCols();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _ColumnElementDivideBy<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), N, M);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::ElementInverse()
{
    if (IsEmpty())
        LogicError("ElementInverse: Matrix is empty.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _elemInverse<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), N);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementInverseOf(const GPUMatrix<ElemType>& a)
{
    SetValue(a);
    return ElementInverse();
}

DEF_ELEMWISE_INPLACE_FUNC(Sigmoid)

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSigmoidOf(const GPUMatrix<ElemType>& a)
{
    RequireSize(a.GetNumRows(), a.GetNumCols());
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    // _elementWIseSigmoidOnCuda has an implementation that avoids possible overflow errors, but has a slight accuracy regression.
#if 0
    _elementWiseSigmoidOnCuda<<<blocksPerGrid, threadsPerBlock, 0, t_stream>>>(a.Data(), Data(), N);
#else
    _assignSigmoidOf<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(a.Data(), Data(), N);
#endif
    return *this;
}

DEF_ELEMWISE_INPLACE_FUNC(SigmoidDerivative)
DEF_ELEMWISE_ASSIGN_FUNC(SigmoidDerivative)

template <class ElemType>
void GPUMatrix<ElemType>::AssignNoiseContrastiveEstimation(const GPUMatrix<ElemType>& a,
                                                           const GPUMatrix<ElemType>& b, const GPUMatrix<ElemType>& bias, size_t sampleCount, GPUMatrix<ElemType>& tmp, GPUMatrix<ElemType>& c)
//this:   samples+probs
// a  :   hidden
// b  :   embedding
// tmp:   softmax
// c  :   loglikelihood
{
    UNCONST(ElemType, a, my_a);
    UNCONST(ElemType, b, my_b);
    UNCONST(ElemType, bias, my_bias);
    SyncGuard syncGuard;
    // a: dim * minibatch
    // b: dim * |vocab|
    int p = 512;
    int width = a.GetNumRows(); // dimension of hidden vector

    while (p / 2 > width)
        p = p / 2;

    // note: kernel has hard-coded dimension of 512
    _computeNceOutputMax512Threads<ElemType> << <GetNumElements() / 2, p >> >(
        Data(),
        sampleCount,
        m_numRows / 2,
        my_a.Data(), // a
        a.GetNumRows(),
        my_b.Data(), // b
        my_bias.Data(),
        tmp.Data()); // tmp

    p = 512;
    while (p / 2 > GetNumElements() / 2)
        p = p / 2;
    // summing up objective must be done in one block
    // note: kernel has hard-coded dimension of 512
    _assignNoiseContrastiveEstimationMax512Threads<ElemType> << <1, p >> >(
        Data(),
        sampleCount,
        m_numRows / 2,
        my_a.Data(),
        a.GetNumCols(),
        my_b.Data(),
        tmp.Data(),
        c.Data());
}

template <class ElemType>
void GPUMatrix<ElemType>::AssignNCEDerivative(GPUMatrix<ElemType>& tmp, const GPUMatrix<ElemType>& a,
                                              const GPUMatrix<ElemType>& b, size_t inputIndex, GPUMatrix<ElemType>& c)
{
    UNCONST(ElemType, a, my_a);
    UNCONST(ElemType, b, my_b);
    SyncGuard syncGuard;
    int p = 512;
    int width = a.GetNumRows();
    while (p / 2 > width)
        p = p / 2;

    _assignNceDerivativeNew<ElemType><<<(tmp.GetNumElements() + p - 1) / p, p>>>(
        Data(),
        tmp.GetNumCols(),
        m_numRows / 2,
        my_a.Data(),
        a.GetNumRows(),
        my_b.Data(),
        tmp.Data(),
        c.Data(),
        inputIndex);
}

template <class ElemType>
void GPUMatrix<ElemType>::AssignSoftmaxSum(const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
{
    UNCONST(ElemType, a, my_a);
    SyncGuard syncGuard;
    int p = 512;
    int width = a.GetNumRows();
    while (p / 2 > width)
        p = p / 2;

    // note: kernel has hard-coded dimension of 512
    _assignSoftmaxSumMax512Threads<ElemType> << <1, p >> >(
        my_a.Data(),
        width,
        Data(),
        c.Data());
}

template <class ElemType>
void GPUMatrix<ElemType>::AssignNCEUnnormalizedEval(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    assert(a.GetComputeDeviceId() == b.GetComputeDeviceId());
    assert(GetNumRows() == a.GetNumRows());
    assert(GetNumCols() == b.GetNumRows());
    assert(a.GetNumCols() == b.GetNumRows());
    UNUSED(a);
    UNUSED(b);
    UNUSED(c); // TODO: this function seems like a stub
    /*
        EnsureAuxMemory();
        int p = 512;
        int width = a.GetNumCols();
        while (p / 2 > width) p = p / 2;

        // this kernel need be launched in nnz blocks
        _sparseInnerProductDenseTimesDense<ElemType> << <m_nz, p >> >(
        m_dVal,
        m_buf,
        m_dCol,
        m_nz,
        GetNumRows(),
        a.Buffer(),
        b.Buffer(),
        b.GetNumRows(),
        m_res);

        // sum up the results
        _reductionSum32<ElemType> << <1, 32 >> >(m_res, c.Buffer(), m_nz);*/
}

DEF_ELEMWISE_INPLACE_FUNC(Tanh)
DEF_ELEMWISE_ASSIGN_FUNC(Tanh)

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceLogSoftmax(const bool isColWise)
{
    if (IsEmpty())
        LogicError("InplaceLogSoftmax: Matrix is empty.");

    PrepareDevice();
    if (isColWise)
    {
        CUDA_LONG N = (CUDA_LONG) GetNumCols(); // one kernel per column
        int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        _logSoftMaxColWise<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), (CUDA_LONG) m_numCols, (CUDA_LONG) m_numRows);
    }
    else
    {
        CUDA_LONG N = (CUDA_LONG) GetNumRows(); // one kernel per column
        int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        _logSoftMaxRowWise<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), (CUDA_LONG) m_numCols, (CUDA_LONG) m_numRows);
    }
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignLogSoftmaxOf(const GPUMatrix<ElemType>& a, const bool isColWise)
{
    RequireSize(a.GetNumRows(), a.GetNumCols());
    if (isColWise)
    {
        PrepareDevice();
        CUDA_LONG N = (CUDA_LONG) GetNumCols();
        CUDA_LONG M = (CUDA_LONG) GetNumRows();
        SyncGuard syncGuard;
        // note: kernel uses hard-coded thread dimension
        _assignColumnwiseLogSoftmaxOf512Threads<<<N, 512, 0, t_stream>>>(a.Data(), Data(), N, M);
    }
    else
    {
        NOT_IMPLEMENTED;
    }

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceHardmax(const bool isColWise)
{
    return AssignHardmaxOf(*this, isColWise);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignHardmaxOf(const GPUMatrix<ElemType>& a, const bool isColWise)
{
    RequireSize(a.GetNumRows(), a.GetNumCols());
    if (isColWise)
    {
        PrepareDevice();
        CUDA_LONG N = (CUDA_LONG) GetNumCols();
        CUDA_LONG M = (CUDA_LONG) GetNumRows();
        SyncGuard syncGuard;
        // note: kernel uses hard-coded thread dimension
        _assignColumnwiseHardmaxOf512Threads << <N, 512, 0, t_stream >> >(a.Data(), Data(), N, M);
    }
    else
    {
        NOT_IMPLEMENTED;
    }

    return *this;
}

DEF_ELEMWISE_INPLACE_FUNC(Sqrt)
DEF_ELEMWISE_ASSIGN_FUNC(Sqrt)

DEF_ELEMWISE_INPLACE_FUNC(Exp)
DEF_ELEMWISE_ASSIGN_FUNC(Exp)

DEF_ELEMWISE_INPLACE_FUNC(Log)
DEF_ELEMWISE_ASSIGN_FUNC(Log)

DEF_ELEMWISE_INPLACE_FUNC(Abs)
DEF_ELEMWISE_ASSIGN_FUNC(Abs)

DEF_ELEMWISE_INPLACE_FUNC(LinearRectifierDerivative)
DEF_ELEMWISE_ASSIGN_FUNC(LinearRectifierDerivative)

DEF_ELEMWISE_INPLACE_FUNC(Cosine)
DEF_ELEMWISE_ASSIGN_FUNC(Cosine)

DEF_ELEMWISE_INPLACE_FUNC(NegativeSine)
DEF_ELEMWISE_ASSIGN_FUNC(NegativeSine)

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceTruncateBottom(const ElemType threshold)
{
    return AssignTruncateBottomOf(*this, threshold);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignTruncateBottomOf(const GPUMatrix<ElemType>& a, const ElemType threshold)
{
    if (a.IsEmpty())
        LogicError("AssignTruncateBottomOf: Matrix a is empty.");

    if (this != &a)
    {
        RequireSize(a.GetNumRows(), a.GetNumCols());
    }

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _assignTruncateBottom<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), threshold, N);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceTruncateTop(const ElemType threshold)
{
    return AssignTruncateTopOf(*this, threshold);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignTruncateTopOf(const GPUMatrix<ElemType>& a, const ElemType threshold)
{
    if (a.IsEmpty())
        LogicError("AssignTruncateTopOf: Matrix a is empty.");

    if (this != &a)
    {
        RequireSize(a.GetNumRows(), a.GetNumCols());
    }

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _assignTruncateTop<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), threshold, N);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceTruncate(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("InplaceTruncate: Matrix is empty.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _inplaceTruncate<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), threshold, N);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceSoftThreshold(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("InplaceSoftThreshold: Matrix is empty.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _inplaceSoftThreshold<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), threshold, N);
    return *this;
}
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::SetToZeroIfAbsLessThan(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("SetToZeroIfAbsLessThan: Matrix is empty.");
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    _setToZeroIfAbsLessThan<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), threshold, N);
    return *this;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::SumOfAbsElements() const
{
    if (IsEmpty())
        LogicError("SumOfAbsElements: Matrix is empty");

    cublasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());
    if (sizeof(ElemType) == sizeof(float))
    {
        float res = 0;
        CUBLAS_CALL(cublasSasum(cuHandle, (CUDA_LONG) GetNumElements(), reinterpret_cast<float*>(Data()), 1, &res));
        return res;
    }
    else
    {
        double res = 0;
        CUBLAS_CALL(cublasDasum(cuHandle, (CUDA_LONG) GetNumElements(), reinterpret_cast<double*>(Data()), 1, &res));
        return ElemType(res);
    }
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::SumOfElements() const
{
    if (IsEmpty())
        LogicError("SumOfElements: Matrix is empty");

    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);
    ElemType h_sum;

    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    _reductionSum1024Threads<ElemType> << <1, 1024, 0, t_stream >> >(Data(), d_sum, (CUDA_LONG)GetNumElements());
    CUDA_CALL(cudaMemcpy(&h_sum, d_sum, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_sum);
    return h_sum;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSumOfElements(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSumOfElements: Matrix a is empty");

    RequireSize(1, 1);

    PrepareDevice();
    SyncGuard syncGuard;
    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    _reductionSumAndAssign1024Threads<ElemType> << <1, 1024 >> >(Data(), a.Data(), (CUDA_LONG)a.GetNumElements(), (CUDA_LONG)GetNumElements());
    return (*this);
}

template <class ElemType>
DeviceBoundNumber<ElemType> GPUMatrix<ElemType>::Sum_AsDeviceBoundNum() const
{
    if (IsEmpty())
        LogicError("Matrix is empty");
    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);

    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    _reductionSum1024Threads<ElemType> << <1, 1024, 0, t_stream >> >(Data(), d_sum, (CUDA_LONG)GetNumElements());
    DeviceBoundNumber<ElemType> result;
    result.ShallowCopyFrom(d_sum, GetComputeDeviceId());
    return result;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::AbsoluteMax() const
{
    cublasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());
    ElemType res;
    if (sizeof(ElemType) == sizeof(float))
    {
        int resInd = 0;
        cublasIsamax(cuHandle, (CUDA_LONG)GetNumElements(), reinterpret_cast<float*>(Data()), 1, &resInd);
        resInd--;
        CUDA_CALL(cudaMemcpy(reinterpret_cast<float*>(&res), reinterpret_cast<float*>(Data() + resInd), sizeof(float), cudaMemcpyDeviceToHost));
        return res;
    }
    else
    {
        int resInd = 0;
        cublasIdamax(cuHandle, (CUDA_LONG)GetNumElements(), reinterpret_cast<double*>(Data()), 1, &resInd);
        resInd--;

        CUDA_CALL(cudaMemcpy(reinterpret_cast<double*>(&res), Data() + resInd, sizeof(double), cudaMemcpyDeviceToHost));

        return res;
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::ElementMultiplyWith(const GPUMatrix<ElemType>& a)
{
    if (IsEmpty() || a.IsEmpty())
        LogicError("ElementMultiplyWith: Matrix is empty.");

    GPUMatrix<ElemType>& us = *this;
    assert(us.GetNumRows() == a.GetNumRows() && us.GetNumCols() == a.GetNumCols());
    if (us.GetNumRows() != a.GetNumRows() || us.GetNumCols() != a.GetNumCols())
        InvalidArgument("The matrix dimensions do not match.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _elemMul<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), N);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    RequireSize(a.GetNumRows(), a.GetNumCols());
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _assignElementProductOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), b.Data(), N);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::ElementDivideBy(const GPUMatrix<ElemType>& a)
{
    return AssignElementDivisionOf(*this, a);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementDivisionOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementDivisionOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    RequireSize(a.GetNumRows(), a.GetNumCols());
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _assignElementDivisionOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), b.Data(), N);
    return *this;
}

template <class ElemType>
bool GPUMatrix<ElemType>::IsEqualTo(const GPUMatrix<ElemType>& a, const ElemType threshold /*= 1e-8*/) const
{
    return AreEqual(*this, a, threshold);
}

template <class ElemType>
void GPUMatrix<ElemType>::VectorSum(const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c, const bool isColWise)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }

    a.PrepareDevice();

    if (a.IsEmpty())
        LogicError("VectorSum:  Input matrix is empty.");

    const CUDA_LONG n = (CUDA_LONG) a.GetNumRows();
    const CUDA_LONG m = (CUDA_LONG) a.GetNumCols();
    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    int blocksPerGrid = 0;
    if (isColWise) // col-wise
    {
        c.RequireSize(1, m);
        blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
    }
    else
    {
        c.RequireSize(n, 1);
        blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
    }

    SyncGuard syncGuard;
    _vectorSum<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(c.Data(), a.Data(), n, m, isColWise);
}
template <class ElemType>
void GPUMatrix<ElemType>::VectorNorm1(GPUMatrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorNorm1: Matrix is empty.");

    const CUDA_LONG n = (CUDA_LONG) GetNumRows();
    const CUDA_LONG m = (CUDA_LONG) GetNumCols();
    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    PrepareDevice();
    c.ChangeDeviceTo(GetComputeDeviceId());

    int blocksPerGrid = 0;
    if (isColWise) // col-wise
    {
        c.RequireSize(1, m);
        blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
    }
    else
    {
        c.RequireSize(n, 1);
        blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
    }

    SyncGuard syncGuard;
    _vectorNorm1<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(c.Data(), Data(), n, m, isColWise);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignVectorNorm1Of(GPUMatrix<ElemType>& a, const bool isColWise)
{
    a.VectorNorm1(*this, isColWise);
    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::VectorNorm2(GPUMatrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorNorm2: Matrix is empty.");

    const CUDA_LONG n = (CUDA_LONG) GetNumRows();
    const CUDA_LONG m = (CUDA_LONG) GetNumCols();
    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    PrepareDevice();
    c.ChangeDeviceTo(GetComputeDeviceId());

    int blocksPerGrid = 0;
    if (isColWise) // col-wise
    {
        c.RequireSize(1, m);
        blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
    }
    else
    {
        c.RequireSize(n, 1);
        c.ChangeDeviceTo(GetComputeDeviceId());
        blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
    }

    SyncGuard syncGuard;
    _vectorNorm2<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(c.Data(), Data(), n, m, isColWise);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignVectorNorm2Of(GPUMatrix<ElemType>& a, const bool isColWise)
{
    a.VectorNorm2(*this, isColWise);
    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::VectorNormInf(GPUMatrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorMax: Matrix is empty.");

    // this implementation is not efficient
    GPUMatrix<ElemType> tmp(GetComputeDeviceId());
    GPUMatrix<ElemType> tmp1(GetComputeDeviceId());
    tmp.AssignAbsOf((*this));
    tmp.VectorMax(tmp1, c, isColWise);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignVectorNormInfOf(GPUMatrix<ElemType>& a, const bool isColWise)
{
    a.VectorNormInf(*this, isColWise);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignInnerProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const bool isColWise)
{
    InnerProduct(a, b, *this, isColWise);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignKhatriRaoProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignKhatriRaoProductOf: Matrix is empty.");

    CUDA_LONG cols = a.GetNumCols();
    assert(cols == b.GetNumCols());
    if (!(cols == b.GetNumCols()))
        InvalidArgument("AssignKhatriRaoProductOf: The input matrix dimensions do not match.");

    CUDA_LONG rowsA = (CUDA_LONG) a.GetNumRows();
    CUDA_LONG rowsB = (CUDA_LONG) b.GetNumRows();
    RequireSize(rowsA * rowsB, cols);
    float N = (float) GetNumElements();
    int blocksPerGrid = (int) ceil(N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _assignKhatriRaoProductOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), b.Data(), rowsA, rowsB, cols);
    return *this;
}

//column-wise reshaped product. Used to compute KhatriRaoProduct Gradient
//   this = reshape each column of a from (K1xK2,1) to (K1, K2)
//   if each column of a is not transposed, each (K1, K2) times each column of b (K2, frames).
//   the output is a (K1, frames) matrix
//   if each column of a is tranposed, each (K1, K2)^T times each column of b(K1, frames) and output is (K2, frames)
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddColumnReshapeProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const bool transposeAColumn)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AddColumnReshapeProductOf: Matrix is empty.");

    CUDA_LONG cols = a.GetNumCols();
    assert(cols == b.GetNumCols());
    if (!(cols == b.GetNumCols()))
        InvalidArgument("AddColumnReshapeProductOf: The input matrix dimensions do not match.");

    CUDA_LONG rowsA = (CUDA_LONG) a.GetNumRows();
    CUDA_LONG rowsB = (CUDA_LONG) b.GetNumRows();
    if (rowsA % rowsB != 0)
        InvalidArgument("AddColumnReshapeProductOf: number of rows in a should be multiples of that in b.");

    CUDA_LONG rowsC = rowsA / rowsB;
    if (rowsC != GetNumRows() || cols != GetNumCols())
        InvalidArgument("AddColumnReshapeProductOf: This matrix does not have the right size.");

    float N = (float) GetNumElements();
    int blocksPerGrid = (int) ceil(N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _addColumnReshapeProductOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), b.Data(), rowsB, rowsC, cols, transposeAColumn);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddWithScaleOf(ElemType alpha, const GPUMatrix<ElemType>& a)
{
    ScaleAndAdd(alpha, a, *this);
    return *this;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::FrobeniusNorm() const
{
    if (IsEmpty())
        LogicError("FrobeniusNorm: Matrix is empty.");

    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);

    ElemType h_sum = 0;
    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    _reductionSum21024Threads<ElemType> << <1, 1024, 0, t_stream >> >(Data(), d_sum, (CUDA_LONG)GetNumElements(), true);
    CUDA_CALL(cudaMemcpy(&h_sum, d_sum, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_sum);

    return (h_sum);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignFrobeniusNormOf(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignFrobeniusNormOf: Matrix a is empty.");

    RequireSize(1, 1);

    PrepareDevice();
    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    _reductionSum21024Threads<ElemType> << <1, 1024, 0, t_stream >> >(a.Data(), Data(), (CUDA_LONG)a.GetNumElements(), true);

    return *this;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::MatrixNormInf() const
{
    if (IsEmpty())
        LogicError("MatrixNormInf: Matrix is empty.");

    ElemType* d_maxAbs = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);

    ElemType h_maxAbs = 0;
    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    _reductionMatrixNormInf1024Threads<ElemType> << <1, 1024, 0, t_stream >> >(Data(), d_maxAbs, (CUDA_LONG)GetNumElements());
    CUDA_CALL(cudaMemcpy(&h_maxAbs, d_maxAbs, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_maxAbs);
    return h_maxAbs;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::MatrixNorm1() const
{
    if (IsEmpty())
        LogicError("MatrixNorm1: Matrix is empty.");
    return SumOfAbsElements();
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::MatrixNorm0() const
{
    if (IsEmpty())
        LogicError("MatrixNorm0: Matrix is empty.");

    ElemType* d_nz = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);
    ElemType h_nz = 0;
    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    _reductionMatrixNorm01024Threads<ElemType> << <1, 1024, 0, t_stream >> >(Data(), d_nz, (CUDA_LONG)GetNumElements());
    CUDA_CALL(cudaMemcpy(&h_nz, d_nz, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_nz);
    return h_nz;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSignOf(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSignOf: Matrix a is empty.");

    if (this != &a)
        RequireSize(a.GetNumRows(), a.GetNumCols());

    PrepareDevice();
    int blocksPerGrid = (int) ceil(1.0 * GetNumElements() / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _assignSignOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), (CUDA_LONG) GetNumElements());
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddSignOf(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AddSignOf: Matrix a is empty.");

    if (this != &a)
        RequireSize(a.GetNumRows(), a.GetNumCols());

    PrepareDevice();
    int blocksPerGrid = (int) ceil(1.0 * GetNumElements() / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _addSignOf<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), (CUDA_LONG) GetNumElements());
    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::VectorMax(GPUMatrix<ElemType>& maxIndexes, GPUMatrix<ElemType>& maxValues, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorMax: Matrix is empty.");

    const GPUMatrix<ElemType>& us = *this;
    const CUDA_LONG m = (CUDA_LONG) GetNumRows();
    const CUDA_LONG n = (CUDA_LONG) GetNumCols();
    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    PrepareDevice();
    SyncGuard syncGuard;
    if (isColWise)
    {
        maxValues.RequireSize(1, n);
        maxIndexes.RequireSize(1, n);

        int blocksPerGrid = n; // we'll have 1 block processing 1 column
        // note: kernel has hard-coded dimension of 512
        _vectorMaxMinReduce512Threads<ElemType, true><<<blocksPerGrid, 512, 0, t_stream>>>(us.Data(), maxIndexes.Data(), maxValues.Data(), m, n);

        /*int blocksPerGrid=(int)ceil(1.0*n/GridDim::maxThreadsPerBlock);
            _vectorMax<ElemType><<<blocksPerGrid,GridDim::maxThreadsPerBlock,0,t_stream>>>(us.Data(),maxIndexes.Data(),maxValues.Data(),m,n,isColWise);*/
    }
    else
    {
        maxValues.RequireSize(m, 1);
        maxIndexes.RequireSize(m, 1);
        int blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
        _vectorMax<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(us.Data(), maxIndexes.Data(), maxValues.Data(), m, n, isColWise);
    }
}

__global__ void _initIndicesForSort(uint64_t* indexes, CUDA_LONG crow, CUDA_LONG ccol)
{
    CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= crow * ccol)
        return;
    uint32_t irow = id % crow;
    uint32_t icol = id / crow;
    indexes[id] = (static_cast<uint64_t>(irow) << 32) | icol;
}

template <class ElemType>
void GPUMatrix<ElemType>::VectorMax(GPUMatrix<ElemType>& maxIndexes, GPUMatrix<ElemType>& maxValues, const bool isColWise, int topK) const
{
    if (IsEmpty())
        LogicError("VectorMax: Matrix is empty.");

    if (topK == 1)
    {
        VectorMax(maxIndexes, maxValues, isColWise);
        return;
    }

    if (!isColWise)
        RuntimeError("Row-wise TopK max is not supported.");

    const GPUMatrix<ElemType>& us = *this;
    const CUDA_LONG m = (CUDA_LONG) GetNumRows();
    const CUDA_LONG n = (CUDA_LONG) GetNumCols();
    assert(topK <= m);
    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    PrepareDevice();
    SyncGuard syncGuard;
    maxValues.RequireSize(topK, n);
    maxIndexes.RequireSize(topK, n);

    // To sort matrix columns we use 2-pass _stable_ sort algorithm:
    // 1. Sort by values (descending) with corresponding row/col indexes.
    // 2. Sort by col indices (ascending) with corresponding values/row indices.
    // Indices are stored as 64-bit ints where low 32 bits represent column and high 32 bits - row index.
    // On the second pass only first 32 bits of the index are used in sorting, so SortPairs has
    // begin_bit and end_bit set accordingly.

    CUDA_LONG celt = static_cast<CUDA_LONG>(GetNumElements());
    ElemType* inVal = us.Data();
    ElemType* outVal1 = nullptr;
    ElemType* outVal2 = nullptr;
    uint64_t* inIdx = nullptr;
    uint64_t* outIdx = nullptr;
    // Determine temp buffer size needed for SortPairsDescending to sort values on the first pass.
    size_t cbtemp = 0;
    // If first param is nullptr then no actual work is done except writing result to cbtemp.
    CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(nullptr, cbtemp, inVal, outVal1, inIdx, outIdx, celt, 0, sizeof(ElemType) * 8, t_stream));
    size_t ctemp1 = (cbtemp + sizeof(ElemType) - 1) / sizeof(ElemType);
    // Determine temp buffer size needed for SortPairs to sort indices on the second pass.
    cbtemp = 0;
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr, cbtemp, outIdx, inIdx, outVal1, outVal2, celt, 0, 32, t_stream));
    size_t ctemp2 = (cbtemp + sizeof(ElemType) - 1) / sizeof(ElemType);
    size_t ctemp = std::max(ctemp1, ctemp2);
    cbtemp = ctemp * sizeof(ElemType);
    // ElemType count needed to store indices, accounting for natural alignment for uint64_t type.
    size_t cidx = ((celt + 1) * sizeof(uint64_t) - 1 + sizeof(ElemType) - 1) / sizeof(ElemType);
    // Get temp workspace.
    auto workspace = GetOrCreateWorkspace();
    // RequireSize to store: output values for the 1st and 2nd passes, input indices, output indices, and temp storage.
    workspace->RequireSize(m, 2 * n + (2 * cidx + ctemp + m - 1) / m);
    outVal1 = workspace->Data();
    outVal2 = outVal1 + celt;
    inIdx = reinterpret_cast<uint64_t*>(outVal2 + celt);
    // Align indices pointer if needed.
    size_t cbAlign = reinterpret_cast<size_t>(inIdx) % sizeof(uint64_t);
    if (cbAlign != 0)
        reinterpret_cast<uint8_t*&>(inIdx) += sizeof(uint64_t) - cbAlign;
    outIdx = inIdx + celt;
    void* ptmp = outIdx + celt;
    assert(reinterpret_cast<ElemType*>(reinterpret_cast<uint8_t*>(ptmp) + cbtemp) <= workspace->Data() + workspace->GetNumElements());

    // Initialize indices.
    const int ThreadsPerBlock = 128;
    int cblock = (celt + ThreadsPerBlock - 1) / ThreadsPerBlock;
    _initIndicesForSort<<<cblock, ThreadsPerBlock, 0, t_stream>>>(inIdx, m, n);
    // Sort by values.
    CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(ptmp, cbtemp, inVal, outVal1, inIdx, outIdx, celt, 0, sizeof(ElemType) * 8, t_stream));
    // Sort by column indices. outIdx contains indices after the first pass so it's used as an input.
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(ptmp, cbtemp, outIdx, inIdx, outVal1, outVal2, celt, 0, 32, t_stream));
    // Copy results.
    cblock = (topK * n + ThreadsPerBlock - 1) / ThreadsPerBlock;
    _copyTopKResults<<<cblock, ThreadsPerBlock, 0, t_stream>>>(inIdx, outVal2, maxIndexes.Data(), maxValues.Data(), m, n, topK);

    ReleaseWorkspace(std::move(workspace));

}

template <class ElemType>
void GPUMatrix<ElemType>::VectorMin(GPUMatrix<ElemType>& minIndexes, GPUMatrix<ElemType>& minValues, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorMax: Matrix is empty.");

    const GPUMatrix<ElemType>& us = *this;
    const int m = (int) GetNumRows();
    const int n = (int) GetNumCols();

    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow
    PrepareDevice();
    SyncGuard syncGuard;
    if (isColWise)
    {
        minValues.RequireSize(1, n);
        minIndexes.RequireSize(1, n);

        int blocksPerGrid = n; // we'll have 1 block processing 1 column
        // note: kernel has hard-coded dimension of 512
        _vectorMaxMinReduce512Threads<ElemType, false> << <blocksPerGrid, 512, 0, t_stream >> >(us.Data(), minIndexes.Data(), minValues.Data(), m, n);

        /*
            int blocksPerGrid=(int)ceil(1.0*n/GridDim::maxThreadsPerBlock);
            _vectorMin<ElemType><<<blocksPerGrid,GridDim::maxThreadsPerBlock,0,t_stream>>>(us.Data(),minIndexes.Data(),minValues.Data(),m,n,isColWise);*/
    }
    else
    {
        minValues.RequireSize(m, 1);
        minIndexes.RequireSize(m, 1);
        int blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
        _vectorMin<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(us.Data(), minIndexes.Data(), minValues.Data(), m, n, isColWise);
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignNumOfDiff(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, bool searchInCol)
{
    if (a.GetNumCols() != b.GetNumCols())
        InvalidArgument("AssignNumOfDiff: a and b must have the same number of columns.");
    if (!searchInCol && a.GetNumRows() != b.GetNumRows())
        InvalidArgument("AssignNumOfDiff: a and b must have the same number of rows.");

    RequireSize(1, 1); // result should be one element

    PrepareDevice();
    SyncGuard syncGuard;
    if (!searchInCol)
    {
        // int blocksPerGrid=(int)ceil(1.0*a.GetNumElements()/GridDim::maxThreadsPerBlock);
        // _assignNumOfDiff1024Threads<ElemType><<<blocksPerGrid,GridDim::maxThreadsPerBlock,0,t_stream>>>(a.Data(), b.Data(), Data(), a.GetNumElements());
        // note: kernel has hard-coded dimension of 1024
        _assignNumOfDiff1024Threads<ElemType> << <1, 1024, 0, t_stream >> >(a.Data(), b.Data(), Data(), (CUDA_LONG)a.GetNumElements());
    }
    else
    {
        const int blockSize = 1024;
        _assignNumOfDiffCol<blockSize><<<1, blockSize, 0, t_stream>>>(a.Data(), b.Data(), Data(),
                                                                      static_cast<CUDA_LONG>(b.GetNumRows()), static_cast<CUDA_LONG>(a.GetNumCols()));
    }
    return *this;
}

#pragma endregion Member BLAS Functions

#pragma region Other helper functions
template <class ElemType>
void GPUMatrix<ElemType>::Print(const char* /*matrixName*/, size_t /*rowStart*/, size_t /*rowEnd*/, size_t /*colStart*/, size_t /*colEnd*/) const
{
    NOT_IMPLEMENTED;
}

template <class ElemType>
void GPUMatrix<ElemType>::Print(const char* matrixName /*=nullptr*/) const
{
    size_t elemCount = GetNumRows() * GetNumCols();
    vector<ElemType> localCopy(elemCount);
    cudaMemcpy(localCopy.data(), Data(), elemCount * sizeof(ElemType), cudaMemcpyDeviceToHost);

    fprintf(stderr, "\n###### ");
    if (matrixName != nullptr)
        fprintf(stderr, "%s ", matrixName);
    fprintf(stderr, "(%lu, %lu) ######\n\n", (unsigned long)GetNumRows(), (unsigned long)GetNumCols());

    if (IsEmpty())
    {
        fprintf(stderr, "(empty)\n");
        return;
    }

    // CNTK is using column-major storage
    for (size_t i = 0; i < GetNumRows(); i++)
    {
        for (size_t j = 0; j < GetNumCols(); j++)
        {
            fprintf(stderr, "%.10f\t", localCopy[i + j * GetNumRows()]);
        }
        fprintf(stderr, "\n");
    }
}

//helpfer function used for convolution neural network
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignPackedConvolutionInput(const GPUMatrix<ElemType>& inputSubBatch,
                                                                       const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                                       const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                                       const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                                       const bool zeroPadding)
{
    assert(verticalSubsample <= kernelHeight && horizontalSubsample <= kernelWidth);

    size_t packedInputRows = kernelWidth * kernelHeight * inputChannels;
    size_t packedInputColsPerSample = outputWidth * outputHeight;
    size_t smallBatchSize = inputSubBatch.GetNumCols();
    RequireSize(packedInputRows, packedInputColsPerSample * smallBatchSize);
    if (zeroPadding)
        SetValue((ElemType) 0);

    PrepareDevice();
    int numThreadPerBlock = GridDim::maxThreadsPerBlock;
#if 1
    int blocksPerGrid = (smallBatchSize * inputWidth * inputHeight * inputChannels + numThreadPerBlock - 1) / numThreadPerBlock;
#else
    dim3 blocksPerGrid((inputWidth * inputHeight * inputChannels + numThreadPerBlock - 1) / numThreadPerBlock, smallBatchSize);
#endif
    SyncGuard syncGuard;
    _assignPackedConvolutionInput<<<blocksPerGrid, numThreadPerBlock, 0, t_stream>>>(Data(),
                                                                                     inputSubBatch.Data(),
                                                                                     smallBatchSize,
                                                                                     inputWidth, inputHeight, inputChannels,
                                                                                     outputWidth, outputHeight, outputChannels,
                                                                                     kernelWidth, kernelHeight, horizontalSubsample, verticalSubsample, zeroPadding);

    return *this;
}

//helpfer function used for convolution neural network
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::UnpackConvolutionInput(GPUMatrix<ElemType>& inputSubBatch,
                                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                                 const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                                 const bool zeroPadding) const
{
    assert(verticalSubsample <= kernelHeight && horizontalSubsample <= kernelWidth);

    size_t smallBatchSize = inputSubBatch.GetNumCols();

    PrepareDevice();
    int numThreadPerBlock = GridDim::maxThreadsPerBlock;
#if 1
    int blocksPerGrid = (smallBatchSize * inputWidth * inputHeight * inputChannels + numThreadPerBlock - 1) / numThreadPerBlock;
#else
    dim3 blocksPerGrid((inputWidth * inputHeight * inputChannels + numThreadPerBlock - 1) / numThreadPerBlock, smallBatchSize);
#endif
    SyncGuard syncGuard;
    _unpackConvolutionInput<<<blocksPerGrid, numThreadPerBlock, 0, t_stream>>>(Data(),
                                                                               inputSubBatch.Data(),
                                                                               smallBatchSize,
                                                                               inputWidth, inputHeight, inputChannels,
                                                                               outputWidth, outputHeight, outputChannels,
                                                                               kernelWidth, kernelHeight, horizontalSubsample, verticalSubsample, zeroPadding);

    return inputSubBatch;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignMaxPoolingResult(const GPUMatrix<ElemType>& inputBatch, const size_t channels,
                                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                                 const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    assert(verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

    unsigned int batchSize = inputBatch.GetNumCols();
    RequireSize(outputSizePerSample, batchSize);

    int numThreadPerBlock = GridDim::maxThreadsPerBlock;
    int blocksPerGrid = (batchSize * outputSizePerSample + numThreadPerBlock - 1) / numThreadPerBlock;

    PrepareDevice();
    SyncGuard syncGuard;
    _assignMaxPoolingResult<<<blocksPerGrid, numThreadPerBlock, 0, t_stream>>>(Data(), inputBatch.Data(), batchSize, channels,
                                                                               inputWidth, inputHeight, inputSizePerSample,
                                                                               outputWidth, outputHeight, outputSizePerSample,
                                                                               windowWidth, windowHeight, horizontalSubsample, verticalSubsample);

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddMaxPoolingGradient(const GPUMatrix<ElemType>& outputGradientBatch, const GPUMatrix<ElemType>& inputBatch, const GPUMatrix<ElemType>& outputBatch,
                                                                const size_t channels,
                                                                const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                                const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                                const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    assert(verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

    unsigned int batchSize = outputGradientBatch.GetNumCols();
    int numThreadPerBlock = GridDim::maxThreadsPerBlock;

    PrepareDevice();
    SyncGuard syncGuard;

    int blocksPerGrid = (batchSize * inputSizePerSample + numThreadPerBlock - 1) / numThreadPerBlock;
    _addMaxPoolingGradient<<<blocksPerGrid, numThreadPerBlock, 0, t_stream>>>(Data(), outputGradientBatch.Data(), inputBatch.Data(), outputBatch.Data(), batchSize, channels,
                                                                              inputWidth, inputHeight, inputSizePerSample,
                                                                              outputWidth, outputHeight, outputSizePerSample,
                                                                              windowWidth, windowHeight, horizontalSubsample, verticalSubsample);

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignAveragePoolingResult(const GPUMatrix<ElemType>& inputBatch, const size_t channels,
                                                                     const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                                     const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                                     const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    assert(verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

    unsigned int batchSize = inputBatch.GetNumCols();
    RequireSize(outputSizePerSample, batchSize);

    int numThreadPerBlock = GridDim::maxThreadsPerBlock;
    int blocksPerGrid = (batchSize * outputSizePerSample + numThreadPerBlock - 1) / numThreadPerBlock;

    PrepareDevice();
    SyncGuard syncGuard;
    _assignAveragePoolingResult<<<blocksPerGrid, numThreadPerBlock, 0, t_stream>>>(Data(), inputBatch.Data(), batchSize, channels,
                                                                                   inputWidth, inputHeight, inputSizePerSample,
                                                                                   outputWidth, outputHeight, outputSizePerSample,
                                                                                   windowWidth, windowHeight, horizontalSubsample, verticalSubsample);

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddAveragePoolingGradient(const GPUMatrix<ElemType>& outputGradientBatch,
                                                                    const size_t channels,
                                                                    const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                                    const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                                    const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    assert(verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

    size_t batchSize = outputGradientBatch.GetNumCols();
    int numThreadPerBlock = GridDim::maxThreadsPerBlock;

    PrepareDevice();
    SyncGuard syncGuard;
    size_t blocksPerGrid = (batchSize * inputSizePerSample + numThreadPerBlock - 1) / numThreadPerBlock;
    _addAveragePoolingGradient<<<blocksPerGrid, numThreadPerBlock, 0, t_stream>>>(Data(), outputGradientBatch.Data(), (CUDA_LONG) batchSize, channels,
                                                                                  inputWidth, inputHeight, inputSizePerSample,
                                                                                  outputWidth, outputHeight, outputSizePerSample,
                                                                                  windowWidth, windowHeight, horizontalSubsample, verticalSubsample);

    return *this;
}

#pragma endregion Other helper functions

template <class ElemType>
void GPUMatrix<ElemType>::ConvolutionForward(const GPUMatrix<ElemType>& kernel, const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIwht,
                                             const GPUMatrix<int>& mpRowRun, const GPUMatrix<int>& runs, GPUMatrix<ElemType>& output) const
{
    const int BlockSize = 128;
    auto gdim = dim3((output.GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
    kConvolutionForward<<<gdim, BlockSize, 0, t_stream>>>((int)GetNumCols(), kernel.Data(), mpRowCol.Data(), mpRowIwht.Data(), mpRowRun.Data(),
                                                            runs.Data(), Data(), (int)GetNumRows(), output.Data(), (int)output.GetNumRows());
}

template <class ElemType>
void GPUMatrix<ElemType>::ConvolutionBackwardData(const GPUMatrix<ElemType>& kernel, const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIwht,
                                                  const GPUMatrix<int>& mpRowRun, const GPUMatrix<int>& runs, GPUMatrix<ElemType>& grad) const
{
    const int BlockSize = 128;
    auto gdim = dim3((GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
    kConvolutionBackwardData<<<gdim, BlockSize, 0, t_stream>>>((int)GetNumCols(), kernel.Data(), mpRowCol.Data(), mpRowIwht.Data(), mpRowRun.Data(),
                                                                 runs.Data(), Data(), (int)GetNumRows(), grad.Data(), (int)grad.GetNumRows());
}

template <class ElemType>
void GPUMatrix<ElemType>::ConvolutionBackwardKernel(const GPUMatrix<ElemType>& in, const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIwht,
                                                    const GPUMatrix<int>& mpRowRun, const GPUMatrix<int>& runs, GPUMatrix<ElemType>& kernelGrad) const
{
    const int BlockSize = 128;
    auto gdim = dim3((GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
    kConvolutionBackwardKernel<<<gdim, BlockSize, 0, t_stream>>>((int)GetNumCols(), (int)in.GetNumRows(), (int)GetNumRows(),
                                                                   in.Data(), mpRowCol.Data(), mpRowIwht.Data(), mpRowRun.Data(),
                                                                   runs.Data(), Data(), kernelGrad.Data());
}

template <class ElemType>
void GPUMatrix<ElemType>::MaxPoolingForward(const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIndices, const GPUMatrix<int>& indices, GPUMatrix<ElemType>& output) const
{
    const int BlockSize = 128;
    auto gdim = dim3((output.GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
    kMaxPoolingForward<<<gdim, BlockSize, 0, t_stream>>>((int)GetNumCols(), mpRowCol.Data(), mpRowIndices.Data(), indices.Data(),
                                                           Data(), (int)GetNumRows(), output.Data(), (int)output.GetNumRows());
}

template <class ElemType>
void GPUMatrix<ElemType>::MaxPoolingBackward(const GPUMatrix<ElemType>& out, const GPUMatrix<ElemType>& in,
                                             const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIndices, const GPUMatrix<int>& indices,
                                             GPUMatrix<ElemType>& grad) const
{
    const int BlockSize = 128;
    auto gdim = dim3((GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
    kMaxPoolingBackward<<<gdim, BlockSize, 0, t_stream>>>((int)GetNumCols(), out.Data(), in.Data(),
                                                            mpRowCol.Data(), mpRowIndices.Data(), indices.Data(),
                                                            Data(), (int)GetNumRows(), grad.Data(), (int)grad.GetNumRows());
}

template <class ElemType>
void GPUMatrix<ElemType>::ROIPoolingForward(const size_t numRois, const size_t numImg, const size_t channels, const size_t width, const size_t height,
                                            const size_t pooledWidth, const size_t pooledHeight, const GPUMatrix<ElemType>& roiData, GPUMatrix<ElemType>& output, 
                                            GPUMatrix<ElemType>& argmax) const
{
    PrepareDevice();
    SyncGuard syncGuard;

    int count = numRois * numImg * channels * pooledHeight * pooledWidth;
    const int blockSize = GridDim::maxThreadsPerBlock;
    auto numThreads = dim3((int)floor((double)(count + blockSize - 1) / blockSize));
    kROIPoolingForward<<<numThreads, blockSize, 0, t_stream>>>(count, numRois, numImg, channels, width, height, 
                                                               pooledWidth, pooledHeight, Data(), roiData.Data(), output.Data(), argmax.Data());
}

template <class ElemType>
void GPUMatrix<ElemType>::ROIPoolingBackward(const size_t numRois, const size_t numImg, const size_t channels, const size_t width, const size_t height,
                                             const size_t pooledWidth, const size_t pooledHeight, const GPUMatrix<ElemType>& roiData, GPUMatrix<ElemType>& grad, 
                                             GPUMatrix<ElemType>& argmax) const
{
    PrepareDevice();
    SyncGuard syncGuard;

    int count = numImg * channels * height * width;
    const int blockSize = GridDim::maxThreadsPerBlock;
    auto numThreads = dim3((int)floor((double)(count + blockSize - 1) / blockSize));
    kROIPoolingBackward<<<numThreads, blockSize, 0, t_stream>>>(count, numRois, numImg, channels, width, height, 
                                                                pooledWidth, pooledHeight, Data(), roiData.Data(), grad.Data(), argmax.Data());
}

template <class ElemType>
void GPUMatrix<ElemType>::MaxUnpooling(const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIndices, const GPUMatrix<int>& indices, const GPUMatrix<ElemType>& poolInput, GPUMatrix<ElemType>& input) const
{
    const int BlockSize = 128;
    auto gdim = dim3((GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
    kMaxUnpooling<<<gdim, BlockSize, 0, t_stream>>>((int)GetNumCols(), mpRowCol.Data(), mpRowIndices.Data(), indices.Data(),
                                                     Data(), poolInput.Data(), (int)GetNumRows(), input.Data(), (int)input.GetNumRows());
}

template <class ElemType>
void GPUMatrix<ElemType>::AveragePoolingForward(const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIndices, const GPUMatrix<int>& indices, GPUMatrix<ElemType>& output) const
{
    const int BlockSize = 128;
    auto gdim = dim3((output.GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
    kAveragePoolingForward<<<gdim, BlockSize, 0, t_stream>>>((int)GetNumCols(), mpRowCol.Data(), mpRowIndices.Data(), indices.Data(),
                                                               Data(), (int)GetNumRows(), output.Data(), (int)output.GetNumRows());
}

template <class ElemType>
void GPUMatrix<ElemType>::AveragePoolingBackward(const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIndices, const GPUMatrix<int>& indices, GPUMatrix<ElemType>& grad) const
{
    const int BlockSize = 128;
    auto gdim = dim3((GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
    kAveragePoolingBackward<<<gdim, BlockSize, 0, t_stream>>>((int)GetNumCols(), mpRowCol.Data(), mpRowIndices.Data(), indices.Data(),
                                                                Data(), (int)GetNumRows(), grad.Data(), (int)grad.GetNumRows());
}

// returns savedMean/savedInvStdDev which are the actual values used to perform the normalization, except for blendFactor 1, in which case they are unused and set to empty
template <class ElemType>
void GPUMatrix<ElemType>::BatchNormalizationForward(const GPUMatrix<ElemType>& scale, const GPUMatrix<ElemType>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor,
                                                    GPUMatrix<ElemType>& runMean, GPUMatrix<ElemType>& runVariance, GPUMatrix<ElemType>& out, double epsilon,
                                                    GPUMatrix<ElemType>& savedMean, GPUMatrix<ElemType>& savedInvStdDev) const
{
    assert((GetNumRows() % scale.GetNumRows()) == 0);

    bool spatial = GetNumRows() != scale.GetNumRows();
    size_t vectorSize = GetNumRows();
    size_t spatialSize = spatial ? (GetNumRows() / scale.GetNumRows()) : 1;
    size_t batchSize = GetNumCols();
    bool normalizeRunningStats;

    assert(0 < vectorSize && vectorSize <= std::numeric_limits<int>::max());
    assert(0 < batchSize  && batchSize  <= std::numeric_limits<int>::max());

    SyncGuard syncGuard;
    if (inferenceOnly)
    {
        // Pick running statistics for normalizing. No update reuqired, and
        // saved statistics do not need to be produced.
        assert(expAvgFactor == 0 && blendFactor == 1);
        normalizeRunningStats = true;
        savedMean.RequireSize(0, 0);
        savedInvStdDev.RequireSize(0, 0);
    }
    else
    {
        // Compute data mean and inverse standard deviation (into savedMean and
        // savedInvStdDev), and update running mean and variance.
        // TODO expAvgFactor == 0 && blendFactor == 1 can be optimized (no need for update).
        normalizeRunningStats = false;
        savedMean.RequireSize(runMean);
        savedInvStdDev.RequireSize(runMean);
        if (spatial)
        {
            Call<ComputeSpatialBatchMeanAndInvStdDev, ElemType>(spatialSize, vectorSize, spatialSize, batchSize, Data(),
                                                                expAvgFactor, blendFactor,
                                                                runMean.Data(), runVariance.Data(), epsilon,
                                                                savedMean.Data(), savedInvStdDev.Data(), GetStream());
        }
        else
        {
            Call<ComputeBatchMeanAndInvStdDev, ElemType>(vectorSize, vectorSize, batchSize, Data(),
                                                         expAvgFactor, blendFactor,
                                                         runMean.Data(), runVariance.Data(), epsilon,
                                                         savedMean.Data(), savedInvStdDev.Data(), GetStream());
        }
    }

    Call<NormalizeBatchTraining, ElemType>(spatial ? spatialSize : vectorSize, vectorSize, spatialSize, batchSize, spatial,
                                           normalizeRunningStats, epsilon,
                                           Data(), out.Data(),
                                           scale.Data(), bias.Data(),
                                           runMean.Data(), runVariance.Data(),
                                           savedMean.Data(), savedInvStdDev.Data(),
                                           GetStream());
}

// savedMean/savedInvStdDev are the interpolated mean/inverse standard deviation as used in ForwardProp().
// For blendFactor=1, they are not used and can be uninitialized or empty.
template <class ElemType>
void GPUMatrix<ElemType>::BatchNormalizationBackward(const GPUMatrix<ElemType>& in, GPUMatrix<ElemType>& grad, const GPUMatrix<ElemType>& scale, double blendFactor,
                                                     const GPUMatrix<ElemType>& savedMean, const GPUMatrix<ElemType>& savedInvStdDev,
                                                     GPUMatrix<ElemType>& scaleGrad, GPUMatrix<ElemType>& biasGrad) const
{
    assert((GetNumRows() % scale.GetNumRows()) == 0);

    bool spatial = GetNumRows() != scale.GetNumRows();
    size_t vectorSize = GetNumRows();
    size_t spatialSize = spatial ? (GetNumRows() / scale.GetNumRows()) : 1;
    size_t batchSize = GetNumCols();

    assert(0 < vectorSize && vectorSize <= std::numeric_limits<int>::max());
    assert(0 < batchSize  && batchSize  <= std::numeric_limits<int>::max());

    SyncGuard syncGuard;
    if (spatial)
    {
        Call<ComputeSpatialScaleAndBiasGradients, ElemType>(spatialSize, vectorSize, spatialSize, batchSize, in.Data(), Data(), scaleGrad.Data(), biasGrad.Data(),
                                                            savedMean.Data(), savedInvStdDev.Data(), GetStream());
    }
    else
    {
        Call<ComputeScaleAndBiasGradients, ElemType>(vectorSize, vectorSize, batchSize, in.Data(), Data(), scaleGrad.Data(), biasGrad.Data(),
                                                     savedMean.Data(), savedInvStdDev.Data(), GetStream());
    }
    ElemType mbStatsWeight = (ElemType)(1 - blendFactor); // weight for contribution from actual MB stats (0 if none, e.g. locked BN node)
    Call<BackpropagateBatchNormGradients, ElemType>(spatial ? spatialSize : vectorSize, vectorSize, spatialSize, batchSize, spatial,
                                                    in.Data(), Data(), grad.Data(), scale.Data(), mbStatsWeight, scaleGrad.Data(), biasGrad.Data(), savedMean.Data(), savedInvStdDev.Data(), GetStream());
}

#pragma region RNN Functions

template <class ElemType>
void GPUMatrix<ElemType>::RNNForward(const GPUMatrix<ElemType> &inputX, const GPUMatrix<ElemType> &paramW, size_t xDim, size_t yDim, const vector<size_t>& numSequencesForFrame, const RnnAttributes& rnnAttributes, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace)
{
    // numLayers, hiddenSize are input parameters
    if (!m_rnnExecutor)
        m_rnnExecutor = std::make_unique<CuDnnRNNExecutor<ElemType>>(xDim, yDim, rnnAttributes);
    m_rnnExecutor->ForwardCore(paramW, inputX, *this, numSequencesForFrame, rnnAttributes, reserve, workspace);
}

template <class ElemType>
void GPUMatrix<ElemType>::RNNBackwardData(const GPUMatrix<ElemType>& outputDY, const GPUMatrix<ElemType>& paramW, GPUMatrix<ElemType>& outputDX, const RnnAttributes& rnnAttributes, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace)
{
    if (!m_rnnExecutor)
        LogicError("RNNBackwardData called, but RNNWrapper object is not yet initialized");
    m_rnnExecutor->BackwardDataCore(*this, outputDY, paramW, outputDX, rnnAttributes, reserve, workspace);
}

template <class ElemType>
void GPUMatrix<ElemType>::RNNBackwardWeights(const GPUMatrix<ElemType>& inputX, const GPUMatrix<ElemType>& outputY, GPUMatrix<ElemType>& dw, const RnnAttributes& rnnAttributes, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace)
{
    if (!m_rnnExecutor)
        LogicError("RNNBackwardWeights called, but RNNWrapper object is not yet initialized");
    m_rnnExecutor->BackwardWeightsCore(inputX, outputY, dw, rnnAttributes, reserve, workspace);
}

#pragma region Static BLAS Functions
// float/double overloads of cublasSgemm()/cublasDgemm()
static cublasStatus_t cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
static cublasStatus_t cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
static cublasStatus_t cublas_axpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
{
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}
static cublasStatus_t cublas_axpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy)
{
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

template <class ElemType>
void GPUMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB,
                                                 ElemType beta, GPUMatrix<ElemType>& c)
{
    a.PrepareDevice();
    if ((a.GetComputeDeviceId() != b.GetComputeDeviceId()) || (b.GetComputeDeviceId() != c.GetComputeDeviceId())) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");

    cublasHandle_t cuHandle = GetCublasHandle(b.GetComputeDeviceId());
    cublasOperation_t transA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int m = int(transposeA ? a.m_numCols : a.m_numRows);
    int n = int(transposeB ? b.m_numRows : b.m_numCols);
    int k = int(transposeA ? a.m_numRows : a.m_numCols);
    int l = int(transposeB ? b.m_numCols : b.m_numRows);

    if (beta == 0)
        c.RequireSize(m, n);
    else
        c.VerifySize(m, n); // Can't resize if beta != 0

    if (!(m > 0 && k > 0 && l > 0 && n > 0))
        RuntimeError("!(m>0 && k>0 && l>0 && n>0)"); // converting from size_t to int may cause overflow
    if (k != l)
        RuntimeError("matrix dim mismatch in MultiplyAndWeightedAdd");
    CUBLAS_CALL(cublas_gemm(cuHandle, transA, transB, m, n, k, &alpha, a.Data(), (int) a.m_numRows, b.Data(), (int) b.m_numRows, &beta, c.Data(), (int) c.m_numRows));
}

template <class ElemType>
void GPUMatrix<ElemType>::Multiply1x1AndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, ElemType beta, GPUMatrix<ElemType>& c)
{
    a.PrepareDevice();
    if ((a.GetComputeDeviceId() != b.GetComputeDeviceId()) || (b.GetComputeDeviceId() != c.GetComputeDeviceId())) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");
    CUDA_LONG N = (CUDA_LONG) c.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _multiply1x1AndWeightedAdd<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(alpha, a.Data(), b.Data(), beta, c.Data(), N);
}

template <class ElemType>
void GPUMatrix<ElemType>::MultiplyAndAdd(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, GPUMatrix<ElemType>& c)
{
    return GPUMatrix<ElemType>::MultiplyAndWeightedAdd(1, a, transposeA, b, transposeB, 1, c);
}

template <class ElemType>
void GPUMatrix<ElemType>::Multiply(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, GPUMatrix<ElemType>& c)
{
    return GPUMatrix<ElemType>::MultiplyAndWeightedAdd(1, a, transposeA, b, transposeB, 0, c);
}

template <class ElemType>
void GPUMatrix<ElemType>::Multiply(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    return GPUMatrix<ElemType>::MultiplyAndWeightedAdd(1, a, false, b, false, 0, c);
}

template <class ElemType>
void GPUMatrix<ElemType>::ColumnwiseScaleAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& v, ElemType beta, GPUMatrix<ElemType>& c)
{
    if (v.GetNumRows() != 1 && v.GetNumCols() != 1)
        InvalidArgument("the argument v must be a vector"); // v is a vector

    if (beta == 0)
        c.RequireSize(a.GetNumRows(), a.GetNumCols());
    else
        c.VerifySize(a.GetNumRows(), a.GetNumCols()); // Can't resize if beta != 0

    int blocksPerGrid = (int)ceil(1.0 * c.GetNumElements() / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _columnwiseScaleAndWeightedAdd<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream >>>(alpha, a.Data(), v.Data(), beta, c.Data(), a.GetNumRows(), a.GetNumCols());
}

/// <summary>Matrix-scalar multiply with col-major matrices: c = alpha * a + c</summary>
/// if a is a column vector, add to all columns of c
/// if a is a row vector, add to all rows of c
/// if a is a scalar, add to all elements of c
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
/*static*/ void GPUMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        if (a.IsEmpty() && c.IsEmpty())
            return;
        a.PrepareDevice();
        if (a.IsEmpty() || c.IsEmpty())
            LogicError("ScaleAndAdd:  one of the input matrices is empty.");
        // if (a.GetNumRows() != 1 && a.GetNumCols() != 1) // a is not a col or row vector
        if (a.GetNumRows() == c.GetNumRows() && a.GetNumCols() == c.GetNumCols()) // dimensions match
        {
            const int m = (int) a.GetNumRows();
            const int n = (int) a.GetNumCols();
            const int len = m * n;
            const int incx = 1;
            const int incy = 1;

            assert(m > 0 && n > 0 && len > 0); // converting from size_t to int may cause overflow
            assert((int) c.GetNumRows() == m && (int) c.GetNumCols() == n);
            if ((int) c.GetNumRows() != m || (int) c.GetNumCols() != n)
                InvalidArgument("dimension of matrix c does not match dimension of matrix a.");

            cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
            // TODO: Overload the call to cublas_axpy to remove these ugly if/else statements.
            if (sizeof(ElemType) == sizeof(float))
            {
                CUBLAS_CALL(cublasSaxpy(cuHandle, len, reinterpret_cast<float*>(&alpha), reinterpret_cast<float*>(a.Data()), incx, reinterpret_cast<float*>(c.Data()), incy));
            }
            else if (sizeof(ElemType) == sizeof(double))
            {
                CUBLAS_CALL(cublasDaxpy(cuHandle, len, reinterpret_cast<double*>(&alpha), reinterpret_cast<double*>(a.Data()), incx, reinterpret_cast<double*>(c.Data()), incy));
            }
            else
            {
                RuntimeError("Unsupported template argument in GPUMatrix");
            }
        }
        else if (a.GetNumElements() == 1)
        {
            CUDA_LONG N = (CUDA_LONG) c.GetNumElements();
            int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
            c.PrepareDevice();
            SyncGuard syncGuard;
            _scaleAndAddScalar<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(c.Data(), N, alpha, a.Data(), c.Data());
        }
        else if (a.GetNumCols() == 1) // col vector, add it to all columns
        {
            CUDA_LONG m = (CUDA_LONG) c.GetNumRows();
            CUDA_LONG n = (CUDA_LONG) c.GetNumCols();
            if (m != (CUDA_LONG) a.GetNumRows())
                InvalidArgument("To add column vector, rows should match.");

            int blocksPerGrid = (int) (ceil(1.0 * m * n / GridDim::maxThreadsPerBlock));
            SyncGuard syncGuard;
#ifdef VALIDATION
            printf(">>>> CUDA compute device is %d\n", a.GetComputeDeviceId());
            printf(">>>> a.Data()= %p, c.Data()= %p, alpha = %f, m = %ld, n = %ld\n", a.Data(), c.Data(), alpha, m, n);
            for (int i = 0; i < 2; i++)
            {
                ElemType buffer[10] = {-1.234f};
                cudaError_t error = cudaMemcpy(buffer, !i ? a.Data(): c.Data(), sizeof(buffer), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                if (error == cudaError::cudaSuccess)
                    printf("buffer valid\n");
            }
#endif

            _matrixVectorColumnWiseAddWithThreadPerElem<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(a.Data(), c.Data(), c.Data(), alpha, m, n);
        }
        else if (a.GetNumRows() == 1) // row vector, add it to all rows
        {
            cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
            int m = (int) c.GetNumRows();
            int n = (int) c.GetNumCols();
            assert(n == (int) a.GetNumCols());
            if (n != (int) a.GetNumCols())
                InvalidArgument("To add row vector, cols should match.");

            // TODO: Overload the call to cublas_axpy to remove these ugly if/else statements.
            if (sizeof(ElemType) == sizeof(double))
            {
                foreach_row (i, c)
                {
                    CUBLAS_CALL(cublasDaxpy(cuHandle, n, reinterpret_cast<double*>(&alpha), reinterpret_cast<double*>(a.Data()), 1, reinterpret_cast<double*>(c.Data()+ i), m));
                }
            }
            else
            {
                foreach_row (i, c)
                {
                    CUBLAS_CALL(cublasSaxpy(cuHandle, n, reinterpret_cast<float*>(&alpha), reinterpret_cast<float*>(a.Data()), 1, reinterpret_cast<float*>(c.Data()+ i), m));
                }
            }
        }
        else
            InvalidArgument("dimension of matrix c does not match dimension of matrix a.");
    }
}

/// <summary>Matrix-scalar multiply with col-major matrices: c = alpha * a + b</summary>
/// if a is a column vector, add to all columns of b
/// if a is a row vector, add to all rows of b
/// if a is a scalar, add to all elements of b
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
/*static*/ void GPUMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId() || a.GetComputeDeviceId() != b.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        if (a.IsEmpty() && b.IsEmpty())
            return;
        a.PrepareDevice();
        if (a.IsEmpty() || b.IsEmpty())
            LogicError("ScaleAndAdd: One of the input matrices is empty.");
        c.RequireSize(b.GetNumRows(), b.GetNumCols());
        // if (a.GetNumRows() != 1 && a.GetNumCols() != 1) // a is not a col or row vector
        if (a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()) // dimensions match
        {
            /*
                const int m = (int)a.GetNumRows();
                const int n = (int)a.GetNumCols();
                const int len = m * n;
                const int incx = 1;
                const int incy = 1;
                assert (m>0 && n>0 && len>0); // converting from size_t to int may cause overflow
                */
            CUDA_LONG N = (CUDA_LONG) c.GetNumElements();
            int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
            c.PrepareDevice();
            SyncGuard syncGuard;
            _matrixMatrixAddOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(alpha, a.Data(), b.Data(), c.Data(), N);
        }
        else if (a.GetNumElements() == 1)
        {
            CUDA_LONG N = (CUDA_LONG) c.GetNumElements();
            int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
            c.PrepareDevice();
            SyncGuard syncGuard;
            _scaleAndAddScalar<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(c.Data(), N, alpha, a.Data(), b.Data());
        }
        else if (a.GetNumCols() == 1) // col vector, add it to all columns
        {
            CUDA_LONG m = (CUDA_LONG) c.GetNumRows();
            CUDA_LONG n = (CUDA_LONG) c.GetNumCols();
            if (m != (CUDA_LONG) a.GetNumRows())
                InvalidArgument("To add column vector, rows should match.");

            int blocksPerGrid = (int) (ceil(1.0 * m * n / GridDim::maxThreadsPerBlock));
            SyncGuard syncGuard;
            _matrixVectorColumnWiseAddWithThreadPerElem<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(a.Data(), b.Data(), c.Data(), alpha, m, n);

        }
        else if (a.GetNumRows() == 1) // row vector, add it to all rows
        {
            CUDA_LONG m = (CUDA_LONG) c.GetNumRows();
            CUDA_LONG n = (CUDA_LONG) c.GetNumCols();
            if (m != (CUDA_LONG) a.GetNumRows())
                InvalidArgument("To add column vector, rows should match.");

            int blocksPerGrid = (int) (ceil(1.0 * m * n / GridDim::maxThreadsPerBlock));
            SyncGuard syncGuard;
            _matrixVectorRowWiseAddWithThreadPerElem<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(a.Data(), b.Data(), c.Data(), alpha, m, n);
        }
        else
            InvalidArgument("Dimension of matrix c does not match dimension of matrix a.");
    }
}

/// <summary>c += alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void GPUMatrix<ElemType>::AddScaledDifference(const ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        a.PrepareDevice();

        assert(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
               a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols());

        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
              a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols()))
        {
            InvalidArgument("AddScaledDifference: a, b, and c must have same dimension.");
        }

        if (a.IsEmpty())
            LogicError("AddScaledDifference: Input matrix a is empty.");

        CUDA_LONG n = (CUDA_LONG) a.GetNumElements();
        int blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        _addScaledDifference<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(alpha, a.Data(), b.Data(), c.Data(), n);
    }
}

/// <summary> c = alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void GPUMatrix<ElemType>::AssignScaledDifference(const ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        a.PrepareDevice();

        assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());

        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
            InvalidArgument("AssignScaledDifference: a, b must have same dimension.");

        if (a.IsEmpty())
            LogicError("AssignScaledDifference: Input matrix a is empty.");

        if (&c != &a && &c != &b)
            c.RequireSize(a.GetNumRows(), a.GetNumCols());

        CUDA_LONG n = (CUDA_LONG) a.GetNumElements();
        int blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        _assignScaledDifference<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(alpha, a.Data(), b.Data(), c.Data(), n);
    }
}

/// <summary>c += alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">1X1 matrix</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void GPUMatrix<ElemType>::AddScaledDifference(const GPUMatrix<ElemType>& alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    assert(alpha.GetNumElements() == 1);
    if (!(alpha.GetNumElements() == 1))
        InvalidArgument("AddScaledDifference: alpha must be a 1X1 matrix.");

    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        a.PrepareDevice();

        assert(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
               a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols());

        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
              a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols()))
        {
            InvalidArgument("AddScaledDifference: a, b, and c must have same dimension.");
        }

        if (a.IsEmpty())
            LogicError("AddScaledDifference: Input matrix a is empty.");

        CUDA_LONG n = (CUDA_LONG) a.GetNumElements();
        int blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        _addScaledDifference<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(alpha.Data(), a.Data(), b.Data(), c.Data(), n);
    }
}

/// <summary> c = alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void GPUMatrix<ElemType>::AssignScaledDifference(const GPUMatrix<ElemType>& alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    assert(alpha.GetNumElements() == 1);
    if (!(alpha.GetNumElements() == 1))
        InvalidArgument("AddScaledDifference: alpha must be a 1X1 matrix.");

    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        a.PrepareDevice();

        assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());

        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        {
            InvalidArgument("AssignScaledDifference: a, b must have same dimension.");
        }

        if (a.IsEmpty())
            LogicError("AssignScaledDifference: Input matrix a is empty.");

        c.RequireSize(a.GetNumRows(), a.GetNumCols());

        CUDA_LONG n = (CUDA_LONG) a.GetNumElements();
        int blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        _assignScaledDifference<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(alpha.Data(), a.Data(), b.Data(), c.Data(), n);
    }
}

//c[ci,cj] += a[ai,aj]
template <class ElemType>
void GPUMatrix<ElemType>::AddElementToElement(ElemType beta, const GPUMatrix<ElemType>& a, const size_t ai, const size_t aj, GPUMatrix<ElemType>& c, const size_t ci, const size_t cj)
{
    if (ai >= a.GetNumRows() || aj >= a.GetNumCols() ||
        ci >= c.GetNumRows() || cj >= c.GetNumCols())
        InvalidArgument("AddElementToElement: Index out of range.");

    a.PrepareDevice();
    SyncGuard syncGuard;
    _addElementToElement<ElemType><<<1, 1, 0, t_stream>>>(beta, a.Data(), (CUDA_LONG) a.LocateElement(ai, aj), c.Data(), (CUDA_LONG) c.LocateElement(ci, cj));
}

template <class ElemType>
/*static*/ void GPUMatrix<ElemType>::Scale(ElemType alpha, GPUMatrix<ElemType>& a)
{
    if (alpha == 0) // if 0 then do not access the value, so that we can use this to multiply uninitialized matrices with beta=0
    {
        CUDA_CALL(cudaMemset(a.Data(), 0, a.m_numRows * a.m_numCols * sizeof(ElemType)));
        return;
    }

    cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
    // TODO: Overload the call to cublas_axpy to remove these ugly if/else statements.
    if (sizeof(ElemType) == sizeof(float))
    {
        float alph = (float) alpha;
        CUBLAS_CALL(cublasSscal(cuHandle, int(a.m_numRows * a.m_numCols), &alph, (float*) a.Data(), 1));
    }
    else if (sizeof(ElemType) == sizeof(double))
    {
        double alph = alpha;
        CUBLAS_CALL(cublasDscal(cuHandle, int(a.m_numRows * a.m_numCols), &alph, (double*) a.Data(), 1));
    }
    else
    {
        RuntimeError("Unsupported template argument in GPUMatrix");
    }
}

template <class ElemType>
/*static*/ void GPUMatrix<ElemType>::Scale(GPUMatrix<ElemType>& alpha, GPUMatrix<ElemType>& a)
{
    if (alpha.GetNumElements() != 1)
    {
        RuntimeError("Matrix alpha must be 1x1");
    }
    cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
    cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_DEVICE);
    if (sizeof(ElemType) == sizeof(float))
    {
        CUBLAS_CALL(cublasSscal(cuHandle, int(a.m_numRows * a.m_numCols), (float*) alpha.Data(), (float*) a.Data(), 1));
    }
    else if (sizeof(ElemType) == sizeof(double))
    {
        CUBLAS_CALL(cublasDscal(cuHandle, int(a.m_numRows * a.m_numCols), (double*) alpha.Data(), (double*) a.Data(), 1));
    }
    else
    {
        cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_HOST);
        RuntimeError("Unsupported template argument in GPUMatrix");
    }
    cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_HOST);
}

template <class ElemType> // c = alpha * a
/*static*/ void GPUMatrix<ElemType>::Scale(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
{
    c = a;
    Scale(alpha, c);
}

template <class ElemType>
void GPUMatrix<ElemType>::InnerProduct(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const bool isColWise)
{
    if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || b.GetComputeDeviceId() != c.GetComputeDeviceId()) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");

    if (a.IsEmpty() || b.IsEmpty())
        LogicError("Scale:  one of the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int k = (int) b.GetNumRows();
    const int l = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("Matrices a and b should have same dimension.");

    if (isColWise)
        c.RequireSize(1, n);
    else
        c.RequireSize(m, 1);

    if ((isColWise && m == 1) || (!isColWise && n == 1)) // in this case it's equivalent to element-wise product
    {
        c.AssignElementProductOf(a, b);
    }
    else
    {
        c.PrepareDevice();

        int blocksPerGrid = 0;
        if (isColWise) // col-wise
        {
            c.RequireSize(1, n);
            blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
        }
        else
        {
            c.RequireSize(m, 1);
            blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
        }

        SyncGuard syncGuard;
        _innerProduct<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(c.Data(), a.Data(), b.Data(), m, n, isColWise);
    }
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::InnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProductOfMatrices:  one of the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int k = (int) b.GetNumRows();
    const int l = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("InnerProductOfMatrices: Matrices a and b should have same dimension.");

    cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
    if (sizeof(ElemType) == sizeof(double))
    {
        double tmp = 0;
        CUBLAS_CALL(cublasDdot(cuHandle, m * n, reinterpret_cast<double*>(a.Data()), 1, reinterpret_cast<double*>(b.Data()), 1, &tmp));
        return ElemType(tmp);
        // return (ElemType)ddot((int)a.GetNumElements(), reinterpret_cast <double*>(a.Data()), 1, reinterpret_cast <double*>(b.Data()), 1);
    }
    else
    {
        float tmp = 0;
        CUBLAS_CALL(cublasSdot(cuHandle, m * n, reinterpret_cast<float*>(a.Data()), 1, reinterpret_cast<float*>(b.Data()), 1, &tmp));
        return tmp;
        // return (ElemType)sdot((int)a.GetNumElements(), reinterpret_cast <float*>(a.Data()), 1, reinterpret_cast <float*>(b.Data()), 1);
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignInnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProductOfMatrices:  one of the input matrices is empty.");

    RequireSize(1, 1);

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int k = (int) b.GetNumRows();
    const int l = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("InnerProductOfMatrices: Matrices a and b should have same dimension.");

    cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
    cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_DEVICE);
    if (sizeof(ElemType) == sizeof(double))
    {
        CUBLAS_CALL(cublasDdot(cuHandle, m * n, reinterpret_cast<double*>(a.Data()), 1, reinterpret_cast<double*>(b.Data()), 1, reinterpret_cast<double*>(Data())));
    }
    else
    {
        CUBLAS_CALL(cublasSdot(cuHandle, m * n, reinterpret_cast<float*>(a.Data()), 1, reinterpret_cast<float*>(b.Data()), 1, reinterpret_cast<float*>(Data())));
    }
    cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_HOST);
    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::ElementWisePower(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        if (a.IsEmpty())
            LogicError("ElementWisePower:  The input matrix a is empty.");

        c.RequireSize(a.GetNumRows(), a.GetNumCols());

        a.PrepareDevice();
        SyncGuard syncGuard;
        CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
        int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
        _elementWisePowerOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(alpha, a.Data(), c.Data(), N);
    }
}

template <class ElemType>
bool GPUMatrix<ElemType>::AreEqual(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const ElemType threshold /*= 1e-8*/)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AreEqual: one of the input matrices is empty.");

    if (a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
        return false;

    bool bResult = false;

    long* res = new long[1];
    res[0] = 1;
    long* d_res = TracingGPUMemoryAllocator::Allocate<long>(a.GetComputeDeviceId(), 1);
    CUDA_CALL(cudaMemcpy(d_res, res, sizeof(long) * 1, cudaMemcpyHostToDevice));
    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    _areEqual<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(a.Data(), b.Data(), N, threshold, d_res);
    CUDA_CALL(cudaMemcpy(res, d_res, sizeof(long) * 1, cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<long>(a.GetComputeDeviceId(), d_res);
    if (res[0] != 0)
        bResult = true;
    delete[] res;
    return bResult;
}

// see Matrix<ElemType>::TensorShuffleScaleAndAdd() for comments
template <class ElemType>
void GPUMatrix<ElemType>::TensorShuffleScaleAndAdd(ElemType keepWeight, const GPUMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    CUDA_LONG N = (CUDA_LONG) c.GetNumElements();
    assert(N == (CUDA_LONG) a.GetNumElements() && N == (CUDA_LONG) b.GetNumElements());
    assert(a.GetComputeDeviceId() == c.GetComputeDeviceId() && b.GetComputeDeviceId() == c.GetComputeDeviceId());
    a.PrepareDevice();
    SyncGuard syncGuard;
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    _tensorShuffleScaleAndAdd<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(keepWeight, a.Data(), D, S, M, K, T, scaleFactor, b.Data(), c.Data());
}

template <class ElemType>
bool GPUMatrix<ElemType>::HasElement(const GPUMatrix<ElemType>& a, const ElemType v)
{
    if (a.IsEmpty())
        LogicError("HasElement: the input matrix is empty.");

    bool bResult = false;
    ElemType* res = new ElemType[2];
    res[0] = v;
    res[1] = 0;
    ElemType* d_res = TracingGPUMemoryAllocator::Allocate<ElemType>(a.GetComputeDeviceId(), 2);
    CUDA_CALL(cudaMemcpy(d_res, res, sizeof(ElemType) * 2, cudaMemcpyHostToDevice));
    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    _hasElement<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(a.Data(), N, d_res);
    CUDA_CALL(cudaMemcpy(res, d_res, sizeof(ElemType) * 2, cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(a.GetComputeDeviceId(), d_res);
    if (res[1] != 0)
        bResult = true;
    else
        bResult = false;

    delete[] res;
    return bResult;
}

template <class ElemType>
void GPUMatrix<ElemType>::CreateCurandObject(unsigned long seed, const char* caller)
{
    assert(caller != nullptr);

    if (s_curandGenerator == NULL)
    {
        unsigned long long cudaSeed = (seed == USE_TIME_BASED_SEED) ? time(NULL) : seed;
        if (GetMathLibTraceLevel() > 0)
        {
            fprintf(stderr, "%s (GPU): creating curand object with seed %llu, sizeof(ElemType)==%lu\n",
                    caller, cudaSeed, (unsigned long)sizeof(ElemType));
        }
        s_curandGenerator = new curandGenerator_t;
        // Create pseudo-random number generator
        CURAND_CALL(curandCreateGenerator(&(((curandGenerator_t*) s_curandGenerator)[0]), CURAND_RNG_PSEUDO_XORWOW));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(((curandGenerator_t*) s_curandGenerator)[0], cudaSeed));
        CURAND_CALL(curandSetGeneratorOrdering(((curandGenerator_t*) s_curandGenerator)[0], CURAND_ORDERING_PSEUDO_SEEDED));
    }
}

template <class ElemType>
void GPUMatrix<ElemType>::ResetCurandObject(unsigned long seed, const char* caller)
{
    assert(caller != nullptr);

    if (s_curandGenerator && (seed != USE_TIME_BASED_SEED))
    {
        // Note: this might be slow.
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(((curandGenerator_t*) s_curandGenerator)[0], seed));
        CURAND_CALL(curandSetGeneratorOffset(((curandGenerator_t*) s_curandGenerator)[0], 0));
    }
    else
    {
        CreateCurandObject(seed, caller);
    }
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::Ones(const size_t rows, const size_t cols, int deviceId)
{
    GPUMatrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    c.SetValue(1);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::Zeros(const size_t rows, const size_t cols, int deviceId)
{
    GPUMatrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    // c.SetValue(0);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::Eye(const size_t rows, int deviceId)
{
    GPUMatrix<ElemType> c(rows, rows, deviceId); // will initialize to 0
    c.SetDiagonalValue(1);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::RandomUniform(const size_t rows, const size_t cols, int deviceId, const ElemType low, const ElemType high, unsigned long seed)
{
    GPUMatrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    c.SetUniformRandomValue(low, high, seed);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::RandomGaussian(const size_t rows, const size_t cols, int deviceId, const ElemType mean, const ElemType sigma, unsigned long seed)
{
    GPUMatrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    c.SetGaussianRandomValue(mean, sigma, seed);
    return c;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::GetLearnRateForBlock_Helper(const GPUMatrix<ElemType>& Gradients, const GPUMatrix<ElemType>& SmoothedGradients)
{
    ElemType* d_res = TracingGPUMemoryAllocator::Allocate<ElemType>(Gradients.GetComputeDeviceId(), 1);

    // Compute inner product of matrices and keep it on device
    const int m = (int) Gradients.GetNumRows();
    const int n = (int) Gradients.GetNumCols();
    const int k = (int) SmoothedGradients.GetNumRows();
    const int l = (int) SmoothedGradients.GetNumCols();
    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("InnerProductOfMatrices: Matrices a and b should have same dimension.");

    if (sizeof(ElemType) == sizeof(double))
    {
        cublasHandle_t cuHandle = GetCublasHandle(Gradients.GetComputeDeviceId());
        cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_DEVICE);
        CUBLAS_CALL(cublasDdot(cuHandle, m * n, reinterpret_cast<double*>(Gradients.Data()), 1, reinterpret_cast<double*>(SmoothedGradients.Data()), 1, reinterpret_cast<double*>(d_res)));
        cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_HOST);
    }
    else
    {
        cublasHandle_t cuHandle = GetCublasHandle(Gradients.GetComputeDeviceId());
        cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_DEVICE);
        CUBLAS_CALL(cublasSdot(cuHandle, m * n, reinterpret_cast<float*>(Gradients.Data()), 1, reinterpret_cast<float*>(SmoothedGradients.Data()), 1, reinterpret_cast<float*>(d_res)));
        cublasSetPointerMode(cuHandle, CUBLAS_POINTER_MODE_HOST);
    }
    // d_res[0] should now contain inner product of matrices
    // Compute squared Frobenius norms (squared sums of elements)
    // note: kernel has hard-coded dimension of 512
    _lrHelper512Threads<ElemType> << <1, 512, 0, t_stream >> >(Gradients.Data(), SmoothedGradients.Data(), (CUDA_LONG)Gradients.GetNumElements(), d_res);
    ElemType res;
    CUDA_CALL(cudaMemcpy(&res, d_res, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(Gradients.GetComputeDeviceId(), d_res);
    return res;
}
// The inputs are two row vectors [a1 a2 a3 a4] [b1 b2 b3 b4]
// The outputs are one matrix of size (nt+1)*4
// The first row is just element multiplication
// The rest rows will be with shift
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementProductOfWithShiftNeg(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const size_t shift, const size_t nt)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    if (!(a.GetNumRows() == 1))
        InvalidArgument("The input matrix must be a row vector.");

    RequireSize(nt + 1, a.GetNumCols());
    int BS = a.GetNumCols();

    // the output matrix is of size (nt+1, BS)
    dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
    dim3 block_tail((nt + 1 + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (BS + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

    a.PrepareDevice();
    SyncGuard syncGuard;
    _assignElementProductOfWithShiftNeg<ElemType><<<block_tail, thread_tail, 0, t_stream>>>(Data(), a.Data(), b.Data(), shift, nt + 1, BS);
    //      _assignElementProductOf<ElemType> << <block_tail, thread_tail, 0, t_stream >> >(Data(), a.Data(), b.Data(), nt);

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignOneHot(const GPUMatrix<ElemType>& a, vector<size_t>& shape, size_t axis)
{
    if(a.IsEmpty())
        LogicError("AssignOneHot: Matrix a is empty.");

    if (axis >= shape.size())
        LogicError("AssignOneHot: axis is not correct");

    size_t item_size = 1;
    for (size_t i = 0; i < shape.size() && i < axis; i++)
        item_size *= shape[i];

    size_t num_class = shape[axis];

    auto nCols = a.GetNumCols();
    auto nRows = num_class * a.GetNumRows();
    this->RequireSize(nRows, nCols);
    this->PrepareDevice();
    
    CUDA_CALL(cudaMemset(Data(), 0, nCols * nRows * sizeof(ElemType)));


    CUDA_LONG N = (CUDA_LONG)a.GetNumElements();
    int blocksPerGrid = (int)ceil(((double)N) / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _assignOneHot<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(a.Data(), Data(), num_class, item_size, N);
    
    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::InnerProductWithShiftNeg(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const size_t shift, const size_t nt)
{
    if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || b.GetComputeDeviceId() != c.GetComputeDeviceId()) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");

    if (a.IsEmpty() || b.IsEmpty())
        LogicError("Scale:  one of the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int k = (int) b.GetNumRows();
    const int l = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("Matrices a and b should have same dimension.");

    c.RequireSize(nt + 1, n);

    if (true)
    {
        c.PrepareDevice();

        dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
        dim3 block_tail((nt + 1 + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (n + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

        SyncGuard syncGuard;
        _innerProductWithShiftNeg<ElemType><<<block_tail, thread_tail, 0, t_stream>>>(c.Data(), a.Data(), b.Data(), m, n, shift, nt + 1);
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::GetARowByIndex(const GPUMatrix<ElemType>& a, const size_t m)
{
    if (a.IsEmpty())
        LogicError("GetARowByIndex: Matrix is empty.");

    RequireSize(1, a.GetNumCols());

    int n = a.GetNumRows();
    int P = a.GetNumCols();

    if (m >= n)
        LogicError("GetARowByIndex: m is out of range.");

    int blocksPerGrid = (int) ceil(((double) P) / GridDim::maxThreadsPerBlock);

    a.PrepareDevice();
    SyncGuard syncGuard;
    _getARowByIndex<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), n, P, m);
    //      _assignElementProductOf<ElemType> << <block_tail, thread_tail, 0, t_stream >> >(Data(), a.Data(), b.Data(), nt);
    return *this;
}

// Calculate CTC score
// prob (input): the posterior output from the network
// alpha, beta (output): alpha and beta for forward-backward calculation. 
// phoneSeq (input): phone ID sequence for each utterance in this minibatch, each col is one utterance 
// phoneBoundary (input): phone boundary (frame index) of each phone for each utterance in this minibatch, each col is one utterance 
// totalScore (output): total CTC score
// uttToChanInd (input):  map from utterance ID to minibatch channel ID. We need this because each channel may contain more than one utterance.
// uttBeginFrame(input): the positon of the first frame of each utterance in the minibatch channel. We need this because each channel may contain more than one utterance.
// uttFrameNum (input): the frame number of each utterance. The size of this vector =  the number of all utterances in this minibatch
// uttPhoneNum (input): the phone number of each utterance. The size of this vector =  the number of all utterances in this minibatch
// numParallelSequences (input): channel number in this minibatch
// maxFrameNum (input): the maximum channel frame number
// delayConstraint -- label output delay constraint introduced during training that allows to have shorter delay during inference.
//      Alpha and Beta scores outside of the delay boundary are set to zero.
//      Setting this parameter smaller will result in shorted delay between label output during decoding, yet may hurt accuracy
//      delayConstraint=-1 means no constraint
template<class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignCTCScore(const GPUMatrix<ElemType>& prob,
    GPUMatrix<ElemType>& alpha,
    GPUMatrix<ElemType>& beta,
    const GPUMatrix<ElemType> phoneSeq,
    const GPUMatrix<ElemType> phoneBoundary,
    ElemType &totalScore,
    const std::vector<size_t>& uttToChanInd,
    const std::vector<size_t> & uttBeginFrame,
    const std::vector<size_t> & uttFrameNum,
    const std::vector<size_t> & uttPhoneNum,
    const size_t numParallelSequences,
    const size_t maxFrameNum, 
    const size_t blankTokenId,
    const int delayConstraint, 
    const bool isColWise)
{
    if (isColWise)
    {
        PrepareDevice();
        // Total number of phones
        long totalPhoneNum = prob.GetNumRows();
        size_t uttNum = uttFrameNum.size();

        // Max number of phones in utterances in this minibatch
        size_t maxPhoneNum = phoneSeq.GetNumRows();

        size_t *gpuFrameNum;
        CUDA_CALL(cudaMalloc((void **)&gpuFrameNum, uttNum * sizeof(size_t)));
        CUDA_CALL(cudaMemcpy(gpuFrameNum, uttFrameNum.data(), uttNum * sizeof(size_t), cudaMemcpyHostToDevice));

        size_t *gpuPhoneNum;
        CUDA_CALL(cudaMalloc((void **)&gpuPhoneNum, uttNum * sizeof(size_t)));
        CUDA_CALL(cudaMemcpy(gpuPhoneNum, uttPhoneNum.data(), uttNum * sizeof(size_t), cudaMemcpyHostToDevice));

        size_t *gpuBeginFrame;
        CUDA_CALL(cudaMalloc((void **)&gpuBeginFrame, uttNum * sizeof(size_t)));
        CUDA_CALL(cudaMemcpy(gpuBeginFrame, uttBeginFrame.data(), uttNum * sizeof(size_t), cudaMemcpyHostToDevice));

        size_t *gpuUttToChanInd;
        CUDA_CALL(cudaMalloc((void **)&gpuUttToChanInd, uttNum * sizeof(size_t)));
        CUDA_CALL(cudaMemcpy(gpuUttToChanInd, uttToChanInd.data(), uttNum * sizeof(size_t), cudaMemcpyHostToDevice));

        ElemType *gpuScores;
        CUDA_CALL(cudaMalloc((void **)&gpuScores, uttNum * sizeof(ElemType)));

        cudaEvent_t done = nullptr;
        CUDA_CALL(cudaEventCreate(&done));
        dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
        // x dimension is for utterances
        // y dimention is for phone sequence in each utterance
        // Ensure that we allocate correct number of blocks for given number of utterances and max number of phones in those utterances 
        dim3 block_tail((uttNum + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (maxPhoneNum + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
        for (long t = 0; t < maxFrameNum; t++)
        {
            _assignAlphaScore << <block_tail, thread_tail, 0, t_stream >> >(prob.Data(), alpha.Data(), phoneSeq.Data(), phoneBoundary.Data(), gpuUttToChanInd,
                gpuFrameNum, gpuBeginFrame, gpuPhoneNum, numParallelSequences, uttNum, t, maxPhoneNum, totalPhoneNum, blankTokenId, delayConstraint);
        }

        for (long t = maxFrameNum - 1; t >= 0; t--)
        {
            _assignBetaScore << <block_tail, thread_tail, 0, t_stream >> >(prob.Data(), beta.Data(), phoneSeq.Data(), phoneBoundary.Data(), gpuUttToChanInd,
                gpuFrameNum, gpuBeginFrame, gpuPhoneNum, numParallelSequences, uttNum, t, maxPhoneNum, totalPhoneNum, blankTokenId, delayConstraint);
        }

        _assignTotalScore << <uttNum, 1, 0, t_stream >> > (beta.Data(), gpuScores, uttNum, gpuUttToChanInd, gpuBeginFrame, numParallelSequences, maxPhoneNum);

        dim3 block_tail_2((uttNum + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (maxFrameNum + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

        _assignCTCScore << < block_tail_2, thread_tail, 0, t_stream >> >(Data(), prob.Data(), alpha.Data(), beta.Data(), phoneSeq.Data(), uttNum, gpuUttToChanInd,
            gpuBeginFrame, gpuPhoneNum, gpuFrameNum, numParallelSequences, maxPhoneNum, totalPhoneNum);

        vector<ElemType>scores(uttNum);
        CUDA_CALL(cudaMemcpyAsync(scores.data(), gpuScores, sizeof(ElemType) * uttNum, cudaMemcpyDeviceToHost, t_stream));

        for (size_t utt = 0; utt < uttFrameNum.size(); utt++)
        {
            totalScore += scores[utt];
        }

        CUDA_CALL(cudaFree(gpuFrameNum));
        CUDA_CALL(cudaFree(gpuPhoneNum));
        CUDA_CALL(cudaFree(gpuBeginFrame));
        CUDA_CALL(cudaFree(gpuUttToChanInd));
        CUDA_CALL(cudaFree(gpuScores));

        CUDA_CALL(cudaEventRecord(done));
        CUDA_CALL(cudaEventSynchronize(done));
        CUDA_CALL(cudaEventDestroy(done));
    }
    else
    {
        NOT_IMPLEMENTED;
    }

    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::ConductRowElementMultiplyWithShift(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const size_t shift, const bool isafixed)
{
    if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || b.GetComputeDeviceId() != c.GetComputeDeviceId()) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");

    if (a.IsEmpty() || b.IsEmpty())
        LogicError("Scale:  one of the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int O = (int) b.GetNumRows();
    const int P = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && O > 0 && P > 0); // converting from size_t to int may cause overflow
    if (m != 1 || n != P)
        InvalidArgument("Matrices a and b should have same dimension.");

    c.RequireSize(O, P);

    if (true)
    {
        c.PrepareDevice();

        dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
        dim3 block_tail((O + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (P + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

        SyncGuard syncGuard;
        _conductRowElementMultiplyWithShift<ElemType><<<block_tail, thread_tail, 0, t_stream>>>(c.Data(), a.Data(), b.Data(), O, P, shift, isafixed);
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementProductOfWithShift(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const size_t shift)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOfWithShift: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    // int O = a.GetNumRows();
    int P = a.GetNumCols();

    RequireSize(1, P);
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    _assignElementProductOfWithShift<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), a.Data(), b.Data(), shift, N);
    return *this;
}

//sequence training
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::DropFrame(const GPUMatrix<ElemType>& label, const GPUMatrix<ElemType>& gamma, const ElemType& threshhold)
{
    if (IsEmpty())
        LogicError("DropFrame: Matrix is empty.");

    PrepareDevice();

    long N = (long) GetNumCols(); // one kernel per column
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _DropFrame<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(Data(), label.Data(), gamma.Data(), threshhold, (long) m_numCols, (long) m_numRows);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSequenceError(const ElemType hsmoothingWeight, const GPUMatrix<ElemType>& label,
                                                              const GPUMatrix<ElemType>& dnnoutput, const GPUMatrix<ElemType>& gamma, ElemType alpha)
{
    if (IsEmpty())
        LogicError("AssignSequenceError: Matrix is empty.");

    PrepareDevice();

    SyncGuard syncGuard;
    long N = (LONG64) label.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    _AssignSequenceError<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(hsmoothingWeight, Data(), label.Data(), dnnoutput.Data(), gamma.Data(), alpha, N);
    return *this;
}

#pragma endregion Static BLAS Functions

/// f = logadd(f, vec) to get the logadd sum of vector elments
template <class ElemType>
ElemType GPUMatrix<ElemType>::LogSumOfElements() const
{
    if (IsEmpty())
        LogicError("SumOfElements: Matrix is empty");

    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);

    ElemType h_sum;
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);

    _reductionLogAddSum<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(Data(),
                                                                                  d_sum, 1, N);
    CUDA_CALL(cudaMemcpy(&h_sum, d_sum, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_sum);

    return h_sum;
}

template <class ElemType>
void GPUMatrix<ElemType>::RCRFBackwardCompute(
    const GPUMatrix<ElemType>& alpha, GPUMatrix<ElemType>& beta,
    const GPUMatrix<ElemType>& /*lbls*/,
    const GPUMatrix<ElemType>& pos_scores, const GPUMatrix<ElemType>& pair_scores, const int shift)
{
    if (alpha.IsEmpty() || pos_scores.IsEmpty() || pair_scores.IsEmpty())
        LogicError("RCRFBackwardCompute: one of the input matrices is empty.");

    if (alpha.GetNumRows() != pos_scores.GetNumRows() || alpha.GetNumCols() != pos_scores.GetNumCols())
        LogicError("RCRFBackwardCompute: matrix dimensions mismatched.");

    size_t iNumLab = alpha.GetNumRows();
    size_t iNumPos = alpha.GetNumCols();

    alpha.PrepareDevice();
    beta.RequireSize(iNumLab, iNumPos);

    ElemType* d_zeta = TracingGPUMemoryAllocator::Allocate<ElemType>(alpha.GetComputeDeviceId(), iNumLab);

    CUDA_LONG N = iNumLab;
    // TODO: change all three '512' to 'GridDim::maxThreadsPerBlock' (not doing this now since I cannot test it)
    int blocksPerGrid = (int) ceil(1.0 * N / 512);
    size_t szMemSize;
    for (int t = iNumPos - 1; t >= 0; t--)
    {
        szMemSize = sizeof(ElemType) * iNumLab;
        // This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == iNumLab.
        assert(iNumLab <= 1024);
        _rcrfBackwardComputeZetaMax1024Labels<ElemType> << <blocksPerGrid, 512, szMemSize >> >(t, iNumPos, alpha.Data(), d_zeta, pair_scores.Data(), iNumLab, shift);
        szMemSize = iNumLab * 3;
        szMemSize *= sizeof(ElemType);
        // This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == 3 * iNumLab.
        assert(iNumLab <= 1024);
        _rcrfBackwardComputeMax1024Labels<ElemType> << <blocksPerGrid, 512, szMemSize >> >(t, iNumPos, alpha.Data(), beta.Data(),
                                                                                           d_zeta, pair_scores.Data(), iNumLab, shift);
    }
    /*
        error = cudaGetErrorString(cudaPeekAtLastError());
        printf("%s\n", error);
        error = cudaGetErrorString(cudaThreadSynchronize());
        printf("%s\n", error);
        */

    TracingGPUMemoryAllocator::Free<ElemType>(alpha.GetComputeDeviceId(), d_zeta);
}

/**
    Compute the gradient for the first order Markov transition probabilities
    It uses equations derived in R. Collobert's paper "Natural language processing (almost) from scratch"
    */
template <class ElemType>
void GPUMatrix<ElemType>::RCRFTransGrdCompute(const GPUMatrix<ElemType>& lbls,
                                              const GPUMatrix<ElemType>& alpha,
                                              const GPUMatrix<ElemType>& beta,
                                              const GPUMatrix<ElemType>& pair_scores,
                                              GPUMatrix<ElemType>& grd,
                                              const int startLbl,
                                              const int shift)
{
    assert(shift == 1);
    int iNumPos = alpha.GetNumCols();
    int iNumLab = alpha.GetNumRows();

    ElemType* d_zeta = TracingGPUMemoryAllocator::Allocate<ElemType>(alpha.GetComputeDeviceId(), iNumLab);

    CUDA_LONG N = iNumLab;
    // TODO: change all three '512' to 'GridDim::maxThreadsPerBlock' (not doing this now since I cannot test it)
    int blocksPerGrid = (int)ceil(1.0 * N / 512);
    size_t szMemSize;
    for (int t = 0; t < iNumPos; t++)
    {
        szMemSize = sizeof(ElemType) * iNumLab;
        // This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == iNumLab.
        assert(iNumLab <= 1024);
        // BUGBUG: This is launched with 512 threads per block, but allocates shared mem as if there is only one block. Likewise for all 4 of these functions.
        _rcrfTransGrdComputeZetaMax1024Labels<ElemType> << <blocksPerGrid, 512, szMemSize >> >(t - 1, iNumPos, alpha.Data(), d_zeta, pair_scores.Data(), iNumLab, startLbl, shift);
        szMemSize = iNumLab * 3;
        szMemSize *= sizeof(ElemType);
        // This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == iNumLab.
        assert(iNumLab <= 1024);
        _rcrfTransGrdComputeMax1024Labels<ElemType> << <blocksPerGrid, 512, szMemSize >> >(t, startLbl, alpha.Data(), beta.Data(),
                                                                                           d_zeta, pair_scores.Data(), lbls.Data(), grd.Data(), iNumPos, iNumLab, shift);
    }
    TracingGPUMemoryAllocator::Free<ElemType>(alpha.GetComputeDeviceId(), d_zeta);
};

// -----------------------------------------------------------------------
// TensorView entry points from Matrix.cpp
// -----------------------------------------------------------------------

// helper to provide a vector of ones of at least the given number of elements
// TODO: Use this to implement ComputationNode::ConstOnes? Or do we even need that anymore?
template <class ElemType>
static shared_ptr<GPUMatrix<ElemType>> GetOnesVector(size_t N, DEVICEID_TYPE deviceId)
{
    // using a dynamically allocated array so this will never get freed, avoiding free-after-DLL-unload issues.
    // and using shared_ptrs since we don't want to leak more than CacheSize elements
    // when using a plain array we would have to control lifetime of the object and destructor would be called for every element in the array at the end
    const int CacheSize = 32;
    static shared_ptr<GPUMatrix<ElemType>> * onesCache = new shared_ptr<GPUMatrix<ElemType>>[CacheSize]; // cache of objects

    if (deviceId >= CacheSize){
        LogicError("GetOnesVector: onesCache[] too small (%d entries), increase (you need %d) and recompile.", CacheSize, (int)deviceId + 1);
    }

    auto p = onesCache[deviceId];
    if (!p || p->GetNumRows() < N) // must (re-)allocate
    {
        p = make_shared<GPUMatrix<ElemType>>(GPUMatrix<ElemType>::Ones(N, 1, deviceId));
        onesCache[deviceId] = p; // this will replace the pointer thread-safely (although weird race conditions may happen where a larger entry is overwritten by a smaller one; will still run correctly)
    }
    return p;
}

// perform unary operation 'op' on a giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
// This binds the N-ariness to a template parameter N, and gets the data pointers out from the matrix objects.
template <class ElemType>
void GPUMatrix<ElemType>::TensorOp(ElemType beta, const GPUMatrix<ElemType>& a, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const array<size_t, 2>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opSum    &&
        reductionOp != ElementWiseOperator::opLogSum &&
        reductionOp != ElementWiseOperator::opMin    &&
        reductionOp != ElementWiseOperator::opMax    &&
        reductionOp != ElementWiseOperator::opElementwiseProduct)
        InvalidArgument("TensorOp: Unary reduction operations other than opMax, opMin, opSum, and opLogSum are not implemented.");

    a.PrepareDevice();
    if (a.GetComputeDeviceId() != GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");

    // special case: linear processing
    // The case statement has measurable impact for unary ops (but not for binary ops it seems, due to double mem access).
    // Linear gap-free unary ops happen so regularly that we will eliminate the case statement from the CUDA kernel, and instead expand all.
    if (regularOpDims.size() == 1 && regularStrides[0][0] == 1 && regularStrides[1][0] == 1 && reducingOpDims.size() == 0)
    {
        // special case: for copy, use cudaMemcpy() instead, or cublas_axpy()
        // TODO: We should observe if these actually make a speed difference, and if not, remove these special cases.
        if (op == ElementWiseOperator::opCopy && beta == 0 && alpha == 1)
            return CUDA_CALL(cudaMemcpy(Data()+ offsets[1], a.Data()+ offsets[0], sizeof(ElemType) * regularOpDims[0], cudaMemcpyDeviceToDevice));
        else if (op == ElementWiseOperator::opCopy && beta == 1)
            return CUBLAS_CALL(cublas_axpy(GetCublasHandle(GetComputeDeviceId()), (int) regularOpDims[0], &alpha, a.Data()+ offsets[0], 1, Data()+ offsets[1], 1));
        else
            return LaunchUnaryTensorOp<ElemType>(beta, a.Data()+ offsets[0], Data()+ offsets[1], alpha, op, regularOpDims[0]);
    }

    // special case: sum-reducing a matrix onto a column vector; can be done with SGEMM
    // Note: A minor risk is that with this, our own reduction function will rarely be used.
    // That function was tested to give the same results with 'double', and nearly the same with 'float' (different summation order matters).
    else if (op == ElementWiseOperator::opCopy && // we are just adding to target without any further operation
             reductionOp == ElementWiseOperator::opSum &&
#ifdef _DEBUG
             sizeof(ElemType) == sizeof(float) && // in debug don't shortcut 'double' so we have some test of our own codepath
#endif
             regularOpDims.size() == 1 && regularStrides[0][0] == 1 && regularStrides[1][0] == 1 && // we are processing a column
             reducingOpDims.size() == 1 && reducingStrides[0][0] >= (ptrdiff_t) regularOpDims[0])   // reducing across columns and no overlap
    {
        assert(reducingStrides[1][0] == 0);
        auto ARows = regularOpDims[0];    // vertical steps
        auto ACols = reducingOpDims[0];   // horizontal steps (reduction)
        auto ALd = reducingStrides[0][0]; // horizontal step width through matrix
        cublasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
        CUBLAS_CALL(cublas_gemm(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, (int) /*CRows=*/ARows, /*CCols=*/1, (int) ACols, &alpha,
                                /*A00=*/a.Data()+ offsets[0], (int) ALd,
                                /*B00=*/GetOnesVector<ElemType>(ACols, a.GetComputeDeviceId())->Data(), (int) /*BRows=*/ACols, &beta,
                                /*C00=*/Data()+ offsets[1], (int) /*CRows=*/ARows));
        return;
    }

    // TODO: Add a special case for tensor bias reduction. cudnn is ~7% faster on Image/QuickE2E.

    // regular case
    else
        return TensorOpN<ElemType, 2>(beta, array<ElemType*, 2>{a.Data(), Data()}, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
}

// perform binary operation 'op' on a and b giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
template <class ElemType>
void GPUMatrix<ElemType>::TensorOp(ElemType beta, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const array<size_t, 3>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opSum)
        InvalidArgument("TensorOp: The only permitted binary reduction operation is opSum.");

    a.PrepareDevice();
    if (a.GetComputeDeviceId() != GetComputeDeviceId() || b.GetComputeDeviceId() != GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");

    return TensorOpN<ElemType, 3>(beta, array<ElemType*, 3>{a.Data(), b.Data(), Data()}, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
}

// perform ternary operation 'op' on a, and c giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
template <class ElemType>
void GPUMatrix<ElemType>::TensorOp(ElemType beta, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const GPUMatrix<ElemType>& c, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const array<size_t, 4>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opSum)
        InvalidArgument("TensorOp: The only permitted ternary reduction operation is opSum.");

    a.PrepareDevice();
    if (a.GetComputeDeviceId() != GetComputeDeviceId() || b.GetComputeDeviceId() != GetComputeDeviceId() || c.GetComputeDeviceId() != GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");
    return TensorOpN<ElemType, 4>(beta, array<ElemType*, 4>{a.Data(), b.Data(), c.Data(), Data()}, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
}

template <class ElemType>
void GPUMatrix<ElemType>::TensorArgOp(const GPUMatrix<ElemType>& a, ElementWiseOperator reductionOp,
                                      const array<size_t, 2>& offsets,
                                      const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                                      const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opArgmin &&
        reductionOp != ElementWiseOperator::opArgmax)
        InvalidArgument("TensorOp: Arg reduction operations other than opArgmax, and opArgmin are not implemented.");

    a.PrepareDevice();
    if (a.GetComputeDeviceId() != GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");
    return TensorOpN<ElemType, 2>((ElemType) 0, array<ElemType*, 2>{a.Data(), Data()}, (ElemType) 1, ElementWiseOperator::opCopy, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
}

// =======================================================================
// explicit instantiations business
// =======================================================================

template class GPUMatrix<float>;
template class GPUMatrix<double>;
template class DeviceBoundNumber<float>;
template class DeviceBoundNumber<double>;

template <class ElemType>
cublasHandle_t GPUMatrix<ElemType>::s_cuHandle[GPUMatrix<ElemType>::MaxGpus] = {0};

template <class ElemType>
void* GPUMatrix<ElemType>::s_curandGenerator = NULL;

// We use Matrix<char> as the backing store for QuantizedMatrix
// Let's explicitly instantiate the methods we need for that purpose
template GPUMatrix<char>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId);
template GPUMatrix<char>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId, char* pArray, const size_t matrixFlags);
template GPUMatrix<char>::GPUMatrix(const GPUMatrix<char>&);
template GPUMatrix<char>::GPUMatrix(GPUMatrix<char>&&);
template char* GPUMatrix<char>::CopyToArray() const;
template void GPUMatrix<char>::ChangeDeviceTo(int);
template void GPUMatrix<char>::Resize(size_t, size_t, bool);
template void GPUMatrix<char>::RequireSize(size_t, size_t, bool);

template GPUMatrix<char>::~GPUMatrix();
template GPUMatrix<char> GPUMatrix<char>::ColumnSlice(size_t startColumn, size_t numCols) const;
template GPUMatrix<char>& GPUMatrix<char>::operator=(GPUMatrix<char>&&);
template GPUMatrix<char>::GPUMatrix(int);
template void GPUMatrix<char>::SetValue(const char);
template void GPUMatrix<char>::SetValue(const size_t numRows, const size_t numCols, int deviceId, char* pArray, size_t matrixFlags, DataTransferer* transferer);
//template void GPUMatrix<char>::SetValue(CPUMatrix<char> const&);
template void GPUMatrix<char>::SetValue(GPUMatrix<char> const&);
//template void GPUMatrix<char>::SetValue(CPUSparseMatrix<char> const&);
//template void GPUMatrix<char>::SetValue(GPUSparseMatrix<char> const&);
template void GPUMatrix<char>::CopySection(size_t numRows, size_t numCols, char* dst, size_t colStride) const;
template void GPUMatrix<char>::Reshape(const size_t, const size_t);
template GPUMatrix<char>& GPUMatrix<char>::operator*=(char);
template DEVICEID_TYPE GPUMatrix<char>::PrepareDevice(DEVICEID_TYPE deviceId) const;

// Support <short>
template GPUMatrix<short>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId);
template GPUMatrix<short>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId, short* pArray, const size_t matrixFlags);
template GPUMatrix<short>::GPUMatrix(const GPUMatrix<short>&);
template GPUMatrix<short>::GPUMatrix(GPUMatrix<short>&&);
template short* GPUMatrix<short>::CopyToArray() const;
template void GPUMatrix<short>::ChangeDeviceTo(int);
template void GPUMatrix<short>::Resize(size_t, size_t, bool);
template void GPUMatrix<short>::RequireSize(size_t, size_t, bool);

template GPUMatrix<short>::~GPUMatrix();
template GPUMatrix<short> GPUMatrix<short>::ColumnSlice(size_t startColumn, size_t numCols) const;
template GPUMatrix<short>& GPUMatrix<short>::operator=(GPUMatrix<short>&&);
template GPUMatrix<short>::GPUMatrix(int);
template void GPUMatrix<short>::SetValue(const short);
template void GPUMatrix<short>::SetValue(const size_t numRows, const size_t numCols, int deviceId, short* pArray, size_t matrixFlags, DataTransferer* transferer);
//template void GPUMatrix<short>::SetValue(CPUMatrix<short> const&);
template void GPUMatrix<short>::SetValue(GPUMatrix<short> const&);
//template void GPUMatrix<short>::SetValue(CPUSparseMatrix<short> const&);
//template void GPUMatrix<short>::SetValue(GPUSparseMatrix<short> const&);
template void GPUMatrix<short>::CopySection(size_t numRows, size_t numCols, short* dst, size_t colStride) const;
template void GPUMatrix<short>::Reshape(const size_t, const size_t);
template GPUMatrix<short>& GPUMatrix<short>::operator*=(short);
template DEVICEID_TYPE GPUMatrix<short>::PrepareDevice(DEVICEID_TYPE deviceId) const;

template GPUMatrix<int>::GPUMatrix(const size_t, const size_t, int, int*, const size_t);
template GPUMatrix<int>::~GPUMatrix();

template int* TracingGPUMemoryAllocator::Allocate<int>(int, size_t);
template size_t* TracingGPUMemoryAllocator::Allocate<size_t>(int, size_t);
template long* TracingGPUMemoryAllocator::Allocate<long>(int, size_t);
template short* TracingGPUMemoryAllocator::Allocate<short>(int, size_t);
template char* TracingGPUMemoryAllocator::Allocate<char>(int, size_t);
template float* TracingGPUMemoryAllocator::Allocate<float>(int, size_t);
template double* TracingGPUMemoryAllocator::Allocate<double>(int, size_t);

template void TracingGPUMemoryAllocator::Free<int>(int, int*, bool);
template void TracingGPUMemoryAllocator::Free<size_t>(int, size_t*, bool);
template void TracingGPUMemoryAllocator::Free<short>(int, short*, bool);
template void TracingGPUMemoryAllocator::Free<char>(int, char*, bool);
template void TracingGPUMemoryAllocator::Free<float>(int, float*, bool);
template void TracingGPUMemoryAllocator::Free<double>(int, double*, bool);

}}}

// !!!!This is from helper_cuda.h which comes with CUDA samples!!!! Consider if it is beneficial to just include all helper_cuda.h
// TODO: This is duplicated in BestGpu.cpp
// Beginning of GPU Architecture definitions
int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
        {
            {0x10, 8},   // Tesla Generation (SM 1.0) G80 class
            {0x11, 8},   // Tesla Generation (SM 1.1) G8x class
            {0x12, 8},   // Tesla Generation (SM 1.2) G9x class
            {0x13, 8},   // Tesla Generation (SM 1.3) GT200 class
            {0x20, 32},  // Fermi Generation (SM 2.0) GF100 class
            {0x21, 48},  // Fermi Generation (SM 2.1) GF10x class
            {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
            {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
            {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }
    return nGpuArchCoresPerSM[7].Cores;
};
// end of GPU Architecture definitions

//inline CUDA_LONG _GetFreeMemoryOnCUDADevice(int devId)
//{
//    CUdevice cudaDevice;
//    CUresult result = cuDeviceGet(&cudaDevice, devId);
//    if(result!= CUDA_SUCCESS)
//    {
//        return 0;
//    }
//
//    // create cuda context
//    CUcontext cudaContext;
//    result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);
//    if(result != CUDA_SUCCESS)
//    {
//        return 0;
//    }
//
//    // get the amount of free memory on the graphics card
//    size_t free;
//    size_t total;
//    result = cuMemGetInfo(&free, &total);
//    if (result!=CUDA_SUCCESS)
//    {
//        return 0;
//    }
//    else
//        return (CUDA_LONG)free;
//}

#endif // CPUONLY
