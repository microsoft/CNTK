//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Basics.h"
#include "BestGpu.h"

#ifndef CPUONLY

#include "GPUSparseMatrix.h"
#include "GPUMatrix.h"
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cublas_v2.h"
#include "GPUMatrixCUDAKernels.cuh"
#include <functional>
#include "CommonMatrix.h"
#include <iostream> // for cout/cerr
#include <assert.h>

typedef unsigned char byte;

#pragma warning(disable : 4267) // conversion from 'size_t' to 'unsigned int'; happens in CUDA <<<a,b>>> syntax if a and b are size_t
#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this

#ifdef _WIN32
// thread local storage to access the current stream, initalize to default stream
extern __declspec(thread)
#else
static
#endif
    cudaStream_t t_stream;

template <>
const char* CudaErrString<cusparseStatus_t>(cusparseStatus_t)
{
    cudaDeviceSynchronize();
    return "(see cusparse.h & look for cusparseStatus_t or CUSPARSE_STATUS_xxx)";
}

namespace Microsoft { namespace MSR { namespace CNTK {

#pragma region Constructors and Destructor

template <class ElemType>
GPUSPARSE_INDEX_TYPE GPUSparseMatrix<ElemType>::SecondaryIndexValueAt(size_t idx) const
{
    GPUSPARSE_INDEX_TYPE value;
    CUDA_CALL(cudaMemcpy(&value, SecondaryIndexLocation() + idx, sizeof(GPUSPARSE_INDEX_TYPE), cudaMemcpyDeviceToHost));

    return value;
}

//-------------------------------------------------------------------------
// construction and conversion
//-------------------------------------------------------------------------

template <class ElemType>
void GPUSparseMatrix<ElemType>::ZeroInit(const MatrixFormat matrixFormat, const DEVICEID_TYPE computeDevice)
{
    if (matrixFormat != MatrixFormat::matrixFormatSparseCSC && matrixFormat != MatrixFormat::matrixFormatSparseCSR &&
        matrixFormat != MatrixFormat::matrixFormatSparseBlockCol && matrixFormat != MatrixFormat::matrixFormatSparseBlockRow)
    {
        LogicError("GPUSparseMatrix:  unsupported sparse matrix format");
        // BUGBUG: Then why even define others?
    }
    Base::ZeroInit();
    m_computeDevice            = computeDevice; // current GPU device Id
    m_format                   = matrixFormat;

    m_sliceOf                  = nullptr;
    m_totalBufferSizeAllocated = 0;
    m_blockSize                = 0;
    m_rowToId                  = nullptr;
    m_tempHostBuffer           = nullptr;
    m_tempHostBufferSize       = 0;
}

template <class ElemType>
GPUSparseMatrix<ElemType>::GPUSparseMatrix(const size_t numRows, const size_t numCols, const size_t numNZ, DEVICEID_TYPE computeDevice, const MatrixFormat matrixFormat /*= MatrixFormat::matrixFormatSparseCSR*/)
{
    ZeroInit(matrixFormat, computeDevice);
    Resize(numRows, numCols, numNZ, true, false);
}

template <class ElemType>
GPUSparseMatrix<ElemType>::GPUSparseMatrix(DEVICEID_TYPE computeDevice, const MatrixFormat matrixFormat /*= MatrixFormat::matrixFormatSparseCSR*/)
{
    ZeroInit(matrixFormat, computeDevice);
}

template <class ElemType>
GPUSparseMatrix<ElemType>::GPUSparseMatrix(const GPUMatrix<ElemType>& deepCopy, const MatrixFormat matrixFormat /*= MatrixFormat::matrixFormatSparseCSR*/)
{
    ZeroInit(matrixFormat, deepCopy.GetComputeDeviceId());
    if (!deepCopy.IsEmpty())
        SetValue(deepCopy, matrixFormat);
}

template <class ElemType>
GPUSparseMatrix<ElemType>::GPUSparseMatrix(const GPUSparseMatrix<ElemType>& deepCopy)
{

    ZeroInit(deepCopy.GetFormat(), deepCopy.GetComputeDeviceId());
    DeepCopy(deepCopy);
}

// PrepareDevice - Setup the correct cuda context for an operation
// deviceId - the device on which the operation will take place
//            defaults to -1, which means use matrices current device
template <class ElemType>
DEVICEID_TYPE GPUSparseMatrix<ElemType>::PrepareDevice(DEVICEID_TYPE deviceId /*=-1*/) const
{
    // if default value use current compute device
    DEVICEID_TYPE newId = deviceId >= 0 ? deviceId : m_computeDevice;

    Microsoft::MSR::CNTK::PrepareDevice(newId);
    return newId;
}

template <class ElemType>
/*private*/ void GPUSparseMatrix<ElemType>::DeepCopy(const GPUSparseMatrix<ElemType>& deepCopy)
{
    ChangeDeviceTo(deepCopy.m_computeDevice);
    deepCopy.PrepareDevice();

    Resize(deepCopy.m_numRows, deepCopy.m_numCols, deepCopy.GetNumNZElements(), deepCopy.m_format, true, false);
    m_externalBuffer  = false;
    m_sliceOf         = nullptr;
    m_nz              = deepCopy.m_nz;
    m_sliceViewOffset = 0; // reset to zero as we only start copying the indices starting from the offset in the source matrix

    CUDA_CALL(cudaMemcpy(BufferPointer(), deepCopy.NzValues(), NzSize(), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(MajorIndexLocation(), deepCopy.MajorIndexLocationWithSliceViewOffset(), MajorIndexSize(), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(SecondaryIndexLocation(), deepCopy.SecondaryIndexLocation(), SecondaryIndexSize(), cudaMemcpyDeviceToDevice));

    if (deepCopy.m_sliceViewOffset > 0)
    {
        int blocksPerGrid = (int) ceil(1.0 * SecondaryIndexCount() / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        _shiftColCSCIndexFromSliceViewToAbsolute<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            SecondaryIndexLocation(),
            SecondaryIndexCount(),
            GetNumNZElements());
    }

    // TODO: to copy other varibles used only for class based LM
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::SetValue(const GPUSparseMatrix<ElemType>& deepCopy)
{
    if (!OwnBuffer())
        LogicError("Cannot SetValue on managed external matrix");

    DeepCopy(deepCopy);
}

// from CPU
template <class ElemType>
void GPUSparseMatrix<ElemType>::SetValue(const CPUSparseMatrix<ElemType>& deepCopy)
{
    if (!OwnBuffer())
        LogicError("Cannot SetValue on Managed external matrix");

    SetFormat(deepCopy.GetFormat());
    if (deepCopy.IsEmpty())
    {
        Reset();
        return;
    }

    if (deepCopy.GetFormat() == matrixFormatSparseCSR)
    {
        SetMatrixFromCSRFormat(deepCopy.RowLocation(), deepCopy.ColLocation(), deepCopy.BufferPointer(), deepCopy.GetNumElemAllocated(), deepCopy.GetNumRows(), deepCopy.GetNumCols());
    }
    else if (deepCopy.GetFormat() == matrixFormatSparseCSC)
    {
        SetMatrixFromCSCFormat(deepCopy.ColLocation(), deepCopy.RowLocation(), deepCopy.BufferPointer(), deepCopy.GetNumElemAllocated(), deepCopy.GetNumRows(), deepCopy.GetNumCols());
    }
    else
        NOT_IMPLEMENTED;
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::CopyToCPUSparseMatrix(CPUSparseMatrix<ElemType>& cpuSparseMatrix) const
{
    if (!cpuSparseMatrix.OwnBuffer())
        LogicError("Cannot CopyToCPUSparseMatrix on managed external matrix");

    cpuSparseMatrix.SetFormat(GetFormat());
    if (IsEmpty())
    {
        cpuSparseMatrix.Reset();
        return;
    }

    if (this->GetFormat() == matrixFormatSparseCSR)
    {
        // we need to do conversion because CPUSparseMatrix uses size_t for indexes while GPUSparseMatrix uses int
        cpuSparseMatrix.Resize(GetNumRows(), GetNumCols(), GetNumElemAllocated(), true, false);
        cpuSparseMatrix.SetNzCount(GetNumNZElements());

        PrepareDevice();

        if (sizeof(GPUSPARSE_INDEX_TYPE) == sizeof(CPUSPARSE_INDEX_TYPE))
        {
            CUDA_CALL(cudaMemcpy(cpuSparseMatrix.RowLocation(), RowLocation(), RowSize(), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(cpuSparseMatrix.ColLocation(), ColLocation(), ColSize(), cudaMemcpyDeviceToHost));
        }
        else
        {
            GPUSPARSE_INDEX_TYPE* h_CSRRow = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(RowSize());
            CUDA_CALL(cudaMemcpy(h_CSRRow, RowLocation(), RowSize(), cudaMemcpyDeviceToHost));
            CopyBuffer(cpuSparseMatrix.RowLocation(), h_CSRRow, SecondaryIndexCount());

            GPUSPARSE_INDEX_TYPE* h_Col = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(ColSize());
            CUDA_CALL(cudaMemcpy(h_Col, ColLocation(), ColSize(), cudaMemcpyDeviceToHost));
            CopyBuffer(cpuSparseMatrix.ColLocation(), h_Col, MajorIndexCount());
        }

        CUDA_CALL(cudaMemcpy(cpuSparseMatrix.BufferPointer(), BufferPointer(), GetSizeElemAllocated(), cudaMemcpyDeviceToHost));
    }
    else if (this->GetFormat() == matrixFormatSparseCSC)
    {
        // we need to do conversion because CPUSparseMatrix uses size_t for indexes while GPUSparseMatrix uses int
        cpuSparseMatrix.Resize(GetNumRows(), GetNumCols(), GetNumNZElements(), true, false);
        cpuSparseMatrix.SetNzCount(GetNumNZElements());

        PrepareDevice();
        if (sizeof(GPUSPARSE_INDEX_TYPE) == sizeof(CPUSPARSE_INDEX_TYPE))
        {
            CUDA_CALL(cudaMemcpy(cpuSparseMatrix.RowLocation(), RowLocation(), RowSize(), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(cpuSparseMatrix.ColLocation(), ColLocation(), ColSize(), cudaMemcpyDeviceToHost));
        }
        else
        {
            GPUSPARSE_INDEX_TYPE* h_CSCCol = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(ColSize());
            CUDA_CALL(cudaMemcpy(h_CSCCol, ColLocation(), ColSize(), cudaMemcpyDeviceToHost));
            CopyBuffer(cpuSparseMatrix.ColLocation(), h_CSCCol, SecondaryIndexCount());

            GPUSPARSE_INDEX_TYPE* h_Row = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(RowSize());
            CUDA_CALL(cudaMemcpy(h_Row, RowLocation(), RowSize(), cudaMemcpyDeviceToHost));
            CopyBuffer(cpuSparseMatrix.RowLocation(), h_Row, MajorIndexCount());
        }

        CUDA_CALL(cudaMemcpy(cpuSparseMatrix.BufferPointer(), BufferPointer(), GetSizeElemAllocated(), cudaMemcpyDeviceToHost));
    }
    else
        NOT_IMPLEMENTED;
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::CopyToDenseMatrix(GPUMatrix<ElemType>& denseMatrix) const
{
    if (IsEmpty())
    {
        denseMatrix.Resize(0, 0);
        return;
    }

    PrepareDevice();
    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    denseMatrix.Resize(m_numRows, m_numCols);

    SyncGuard syncGuard;
    CUSPARSE_CALL(cusparseSetStream(cusparseHandle, t_stream));
    if (m_format == MatrixFormat::matrixFormatSparseCSR)
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseScsr2dense(cusparseHandle, int(m_numRows), int(m_numCols), descr, (float*) BufferPointer(), RowLocation(), ColLocation(), (float*) denseMatrix.BufferPointer(), int(m_numRows)));
        }
        else
        {
            CUSPARSE_CALL(cusparseDcsr2dense(cusparseHandle, int(m_numRows), int(m_numCols), descr, (double*) BufferPointer(), RowLocation(), ColLocation(), (double*) denseMatrix.BufferPointer(), int(m_numRows)));
        }
    }
    else if (m_format == MatrixFormat::matrixFormatSparseCSC)
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseScsc2dense(cusparseHandle, int(m_numRows), int(m_numCols), descr, (float*) BufferPointer(), RowLocation(), ColLocation(), (float*) denseMatrix.BufferPointer(), int(m_numRows)));
        }
        else
        {
            CUSPARSE_CALL(cusparseDcsc2dense(cusparseHandle, int(m_numRows), int(m_numCols), descr, (double*) BufferPointer(), RowLocation(), ColLocation(), (double*) denseMatrix.BufferPointer(), int(m_numRows)));
        }
    }
    else
    {
        NOT_IMPLEMENTED;
    }
    CUSPARSE_CALL(cusparseDestroy(cusparseHandle));

}

template <class ElemType>
void GPUSparseMatrix<ElemType>::ConvertToSparseFormat(MatrixFormat newFormat, GPUSparseMatrix<ElemType>& outMatrix) const
{
    if (!outMatrix.OwnBuffer())
        LogicError("Cannot ConvertToSparseFormat to managed external matrix");

    if (IsEmpty())
    {
        outMatrix.ZeroInit(newFormat, GetComputeDeviceId());
        return;
    }

    MatrixFormat oldFormat = GetFormat();
    if (oldFormat == newFormat)
    {
        outMatrix.SetValue(*this);
        return;
    }

    PrepareDevice();
    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));

    SyncGuard syncGuard;
    CUSPARSE_CALL(cusparseSetStream(cusparseHandle, t_stream));

    outMatrix.ChangeDeviceTo(GetComputeDeviceId());
    outMatrix.Resize(m_numRows, m_numCols, m_nz, newFormat, true, false);
    outMatrix.SetNzCount(m_nz);

    if ((oldFormat == matrixFormatSparseCSR && newFormat == matrixFormatSparseCSC) || (oldFormat == matrixFormatSparseCSC && newFormat == matrixFormatSparseCSR))
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, int(m_numRows), int(m_numCols), int(m_elemSizeAllocated),
                                           (float*) BufferPointer(), RowLocation(), ColLocation(), (float*) outMatrix.BufferPointer(),
                                           outMatrix.RowLocation(), outMatrix.ColLocation(), CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));
        }
        else
        {
            CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, int(m_numRows), int(m_numCols), int(m_elemSizeAllocated),
                                           (double*) BufferPointer(), RowLocation(), ColLocation(), (double*) outMatrix.BufferPointer(),
                                           outMatrix.RowLocation(), outMatrix.ColLocation(), CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));
        }
    }
    else
    {
        NOT_IMPLEMENTED;
    }

    CUSPARSE_CALL(cusparseDestroy(cusparseHandle));
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::ConvertToSparseFormat(MatrixFormat newFormat)
{
    if (IsEmpty())
    {
        SetFormat(newFormat);
        return;
    }

    MatrixFormat oldFormat = GetFormat();
    if (oldFormat == newFormat)
        return;

    GPUSparseMatrix<ElemType> tempMatrix(GetComputeDeviceId(), newFormat);
    ConvertToSparseFormat(newFormat, tempMatrix);

    *this = std::move(tempMatrix);
}

template <class ElemType>
GPUMatrix<ElemType> GPUSparseMatrix<ElemType>::CopyToDenseMatrix() const
{
    GPUMatrix<ElemType> res(GetComputeDeviceId());
    if (!IsEmpty())
        CopyToDenseMatrix(res);
    return res;
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::ChangeDeviceTo(DEVICEID_TYPE to_id)
{
    if (!OwnBuffer())
        LogicError("Cannot change device on Managed external matrix");
    if (to_id == CPUDEVICE)
        LogicError("to_id must be valid GPU");
    if (m_computeDevice == to_id)
        return;

    if (m_totalBufferSizeAllocated == 0) // nothing to move
    {
        assert(m_pArray == nullptr);
    }
    else
    {
        ElemType* d_dst = reinterpret_cast<ElemType*>(TracingGPUMemoryAllocator::Allocate<char>(to_id, m_totalBufferSizeAllocated));

        // first try peer access
        int canAccessPeer = false;
        CUDA_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, to_id, m_computeDevice));
        if (canAccessPeer)
        {
            cudaError_t cudaStatus = cudaDeviceEnablePeerAccess(m_computeDevice, 0);
            if (cudaStatus != cudaErrorPeerAccessAlreadyEnabled)
            {
                CUDA_CALL(cudaStatus);
            }
            CUDA_CALL(cudaMemcpyPeer(d_dst, to_id, m_pArray, m_computeDevice, m_totalBufferSizeAllocated));
        }
        else
        {
            // peer access didn't work, just copy normal
            // make this more efficient by keeping some buffers available for each copy
            ElemType* h_dst = NULL;
            PrepareDevice();
            CUDA_CALL(cudaMallocHost((void**) &h_dst, m_totalBufferSizeAllocated));
            CUDA_CALL(cudaMemcpy(h_dst, m_pArray, m_totalBufferSizeAllocated, cudaMemcpyDeviceToHost));
            PrepareDevice((DEVICEID_TYPE) to_id);
            CUDA_CALL(cudaMemcpy(d_dst, h_dst, m_totalBufferSizeAllocated, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaFreeHost(h_dst));
        }

        TracingGPUMemoryAllocator::Free<ElemType>(m_computeDevice, m_pArray);
        m_pArray = d_dst;
    }

    SetComputeDeviceId(PrepareDevice(to_id));
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::SetValue(const GPUMatrix<ElemType>& denseMatrix)
{
    if (!OwnBuffer())
        LogicError("Cannot SetValue on Managed external matrix");

    SetValue(denseMatrix, GetFormat());
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::SetValue(const GPUMatrix<ElemType>& denseMatrix, const MatrixFormat matrixFormat)
{
    if (!OwnBuffer())
        LogicError("Cannot SetValue on Managed external matrix");

    if (matrixFormat != matrixFormatSparseCSR && matrixFormat != matrixFormatSparseCSC)
    {
        NOT_IMPLEMENTED;
    }

    PrepareDevice();
    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    int numRows = (int) denseMatrix.GetNumRows(); // m
    int numCols = (int) denseMatrix.GetNumCols(); // n

    int* nnzPerRowOrCol = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(m_computeDevice, ((matrixFormat & matrixFormatRowMajor) ? numRows : numCols));
    int nnzTotalDevHostPtr = -1;

    {
        SyncGuard syncGuard;
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseSnnz(cusparseHandle, (matrixFormat & matrixFormatRowMajor) ? CUSPARSE_DIRECTION_ROW : CUSPARSE_DIRECTION_COLUMN, (int) numRows, (int) numCols, descr,
                                       reinterpret_cast<float*>(denseMatrix.BufferPointer()), (int) numRows, nnzPerRowOrCol, &nnzTotalDevHostPtr));
        }
        else
        {
            CUSPARSE_CALL(cusparseDnnz(cusparseHandle, (matrixFormat & matrixFormatRowMajor) ? CUSPARSE_DIRECTION_ROW : CUSPARSE_DIRECTION_COLUMN, (int) numRows, (int) numCols, descr,
                                       reinterpret_cast<double*>(denseMatrix.BufferPointer()), (int) numRows, nnzPerRowOrCol, &nnzTotalDevHostPtr));
        }
        // ~SyncGuard
    }

    Resize(numRows, numCols, nnzTotalDevHostPtr, matrixFormat, true, false);
    SetNzCount(nnzTotalDevHostPtr);

    SyncGuard syncGuard;
    if (m_format == MatrixFormat::matrixFormatSparseCSR)
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseSdense2csr(cusparseHandle, (int) m_numRows, (int) m_numCols, descr, reinterpret_cast<float*>(denseMatrix.BufferPointer()),
                                             (int) m_numRows, nnzPerRowOrCol, reinterpret_cast<float*>(BufferPointer()), RowLocation(), ColLocation()));
        }
        else
        {
            CUSPARSE_CALL(cusparseDdense2csr(cusparseHandle, (int) m_numRows, (int) m_numCols, descr, reinterpret_cast<double*>(denseMatrix.BufferPointer()),
                                             (int) m_numRows, nnzPerRowOrCol, reinterpret_cast<double*>(BufferPointer()), RowLocation(), ColLocation()));
        }
    }
    else if (m_format == MatrixFormat::matrixFormatSparseCSC)
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseSdense2csc(cusparseHandle, (int) m_numRows, (int) m_numCols, descr, reinterpret_cast<float*>(denseMatrix.BufferPointer()),
                                             (int) m_numRows, nnzPerRowOrCol, reinterpret_cast<float*>(BufferPointer()), RowLocation(), ColLocation()));
        }
        else
        {
            CUSPARSE_CALL(cusparseDdense2csc(cusparseHandle, (int) m_numRows, (int) m_numCols, descr, reinterpret_cast<double*>(denseMatrix.BufferPointer()),
                                             (int) m_numRows, nnzPerRowOrCol, reinterpret_cast<double*>(BufferPointer()), RowLocation(), ColLocation()));
        }
    }
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::operator=(const GPUSparseMatrix<ElemType>& deepCopy)
{
    Clear();
    if (this != &deepCopy)
        SetValue(deepCopy);

    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>::GPUSparseMatrix(GPUSparseMatrix<ElemType>&& moveFrom)
{
    Base::ShallowCopyFrom(moveFrom);
    // TODO: implement this using operator= or a shared function
    m_totalBufferSizeAllocated = moveFrom.m_totalBufferSizeAllocated;
    m_sliceOf                  = moveFrom.m_sliceOf;
    m_blockSize                = moveFrom.m_blockSize;
    m_rowToId                  = moveFrom.m_rowToId;
    m_tempHostBuffer           = moveFrom.m_tempHostBuffer;
    m_tempHostBufferSize       = moveFrom.m_tempHostBufferSize;
    moveFrom.ZeroInit(moveFrom.m_format, moveFrom.m_computeDevice); // so that memory in moveFrom is not freed
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::operator=(GPUSparseMatrix<ElemType>&& moveFrom)
{
    if (this != &moveFrom)
    {
        if (OwnBuffer())
            ReleaseMemory(); // always delete the data pointer since we will use the pointer from moveFrom
        Base::ShallowCopyFrom(moveFrom);
        m_totalBufferSizeAllocated = moveFrom.m_totalBufferSizeAllocated;
        m_sliceOf                  = moveFrom.m_sliceOf;
        m_blockSize                = moveFrom.m_blockSize;
        m_rowToId                  = moveFrom.m_rowToId;
        m_tempHostBuffer           = moveFrom.m_tempHostBuffer;
        m_tempHostBufferSize       = moveFrom.m_tempHostBufferSize;
        moveFrom.ZeroInit(moveFrom.m_format, moveFrom.m_computeDevice);
    }

    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>::~GPUSparseMatrix()
{
    ReleaseMemory();
}

template <class ElemType>
/*private*/ void GPUSparseMatrix<ElemType>::ReleaseMemory()
{
    // If OwnBuffer() is false then this matrix is a view over another matrix.
    // In that case we shouldn't free anything.
    if (OwnBuffer())
    {
        delete[](byte*) m_tempHostBuffer;
        m_tempHostBuffer = nullptr;

        TracingGPUMemoryAllocator::Free<ElemType>(m_computeDevice, m_pArray);
        m_pArray = nullptr;

        TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(m_computeDevice, m_rowToId);
        m_rowToId = nullptr;
    }
    ZeroInit(m_format, m_computeDevice);
}

//ResizeAsAndCopyIndexFrom - Resize this sparse matrix to have the same element structure as the passed matrix
// a - sparse matrix whose structure we want to clone
// remark: this was done for element wise operations where the structure will be identical after an operation
template <class ElemType>
void GPUSparseMatrix<ElemType>::ResizeAsAndCopyIndexFrom(const GPUSparseMatrix<ElemType>& a, const bool growOnly /*= true*/)
{
    Resize(a.m_numRows, a.m_numCols, a.m_nz, a.m_format, growOnly, false);
    SetNzCount(a.m_nz);

    CUDA_CALL(cudaMemcpy(MajorIndexLocation(), a.MajorIndexLocation(), MajorIndexSize(), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(SecondaryIndexLocation(), a.SecondaryIndexLocation(), SecondaryIndexSize(), cudaMemcpyDeviceToDevice));
}

//-------------------------------------------------------------------------
// main operations
//-------------------------------------------------------------------------

template <class ElemType>
void GPUSparseMatrix<ElemType>::Reshape(const size_t numRows, const size_t numCols)
{
    if (m_numRows == numRows && m_numCols == numCols)
        return;

    if (!OwnBuffer())
        LogicError("GPUSparseMatrix::Reshape: Cannot Reshape since the buffer is managed externally.");

    if (m_format != MatrixFormat::matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    if (m_numRows * m_numCols != numRows * numCols)
        LogicError("GPUSparseMatrix::Reshape: new matrix size does not match current size, can't be reshaped. Did you mean to resize?");

    size_t bufferSizeNeeded = BufferSizeNeeded(numRows, numCols, m_elemSizeAllocated, m_format);

    ElemType* pArray = reinterpret_cast<ElemType*>(TracingGPUMemoryAllocator::Allocate<char>(m_computeDevice, bufferSizeNeeded));

    if (m_pArray != nullptr)
    {
        CUDA_CALL(cudaMemcpy(pArray, BufferPointer(), GetSizeElemAllocated(), cudaMemcpyDeviceToDevice));

        GPUSPARSE_INDEX_TYPE* majorIndexInNewBuffer = (GPUSPARSE_INDEX_TYPE*) (pArray + m_elemSizeAllocated);
        GPUSPARSE_INDEX_TYPE* secondaryIndexInNewBuffer = majorIndexInNewBuffer + MajorIndexCount(numRows, numCols, m_elemSizeAllocated, m_format);

        int blocksPerGrid = (int) ceil(1.0 * numCols / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        _reshape<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            m_numRows,                // old row count
            m_numCols,                // old col count
            numRows,                  // new row count
            numCols,                  // new col count
            MajorIndexLocation(),     // old row index array
            SecondaryIndexLocation(), // old column index array
            majorIndexInNewBuffer,    // new row index array
            secondaryIndexInNewBuffer // new column index array
            );
        TracingGPUMemoryAllocator::Free<ElemType>(m_computeDevice, m_pArray);
    }

    m_pArray = pArray;
    m_numRows = numRows;
    m_numCols = numCols;
    m_totalBufferSizeAllocated = bufferSizeNeeded;

    // following are generated dynamically and no need to save
    if (m_rowToId != nullptr)
        TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(m_computeDevice, m_rowToId);

    m_rowToId = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(m_computeDevice, m_elemSizeAllocated);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const bool growOnly, bool keepExistingValues)
{
    Resize(numRows, numCols, numNZElemToReserve, GetFormat(), growOnly, keepExistingValues);
}

// WARNING: When memory is reallocated, existing information will be lost.
// TODO: add keepExistingValues (default to true) argument so that the existing values are kept even after reallocation
template <class ElemType>
void GPUSparseMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly /*= true*/, bool keepExistingValues /*=true*/)
{
    if (!OwnBuffer())
        LogicError("Cannot Resize since the buffer is managed externally.");

    if (matrixFormat != m_format || m_numRows != numRows || m_numCols != numCols)
        keepExistingValues = false;

    size_t bufferSizeNeeded = BufferSizeNeeded(numRows, numCols, numNZElemToReserve, matrixFormat);
    bool reallocate = (m_totalBufferSizeAllocated < bufferSizeNeeded || (!growOnly && m_totalBufferSizeAllocated > bufferSizeNeeded));

    if (reallocate)
    {
        ElemType* pArray = reinterpret_cast<ElemType*>(TracingGPUMemoryAllocator::Allocate<char>(m_computeDevice, bufferSizeNeeded));

        if (m_pArray != nullptr)
        {
            if (keepExistingValues)
            {
                if (m_nz > numNZElemToReserve || m_totalBufferSizeAllocated > bufferSizeNeeded)
                    LogicError("Resize: To keep values m_nz should <= numNZElemToReserve.");

                CUDA_CALL(cudaMemcpy(pArray, BufferPointer(), GetSizeElemAllocated(), cudaMemcpyDeviceToDevice));

                GPUSPARSE_INDEX_TYPE* majorIndexInNewBuffer = (GPUSPARSE_INDEX_TYPE*) (pArray + numNZElemToReserve);

                CUDA_CALL(cudaMemcpy(majorIndexInNewBuffer, MajorIndexLocation(), MajorIndexSize(), cudaMemcpyDeviceToDevice));

                GPUSPARSE_INDEX_TYPE* secondaryIndexInNewBuffer = majorIndexInNewBuffer + MajorIndexCount(numRows, numCols, numNZElemToReserve, matrixFormat);
                CUDA_CALL(cudaMemcpy(secondaryIndexInNewBuffer, SecondaryIndexLocation(), SecondaryIndexSize(), cudaMemcpyDeviceToDevice));
            }
            else
                m_nz = 0;

            TracingGPUMemoryAllocator::Free<ElemType>(m_computeDevice, m_pArray);
        }

        m_pArray = pArray;

        // following are generated dynamically and no need to save
        if (m_rowToId != nullptr)
            TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(m_computeDevice, m_rowToId);

        m_rowToId = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(m_computeDevice, numNZElemToReserve);

        m_totalBufferSizeAllocated = bufferSizeNeeded;
        m_elemSizeAllocated = numNZElemToReserve;
    }
    else // if requested size is smaller, keeping original values does not make sense
    {
        m_elemSizeAllocated = ElemCountFromBufferSize(numRows, numCols, matrixFormat, m_totalBufferSizeAllocated);
    }

    m_numRows = numRows;
    m_numCols = numCols;
    m_format = matrixFormat;
}

// Reset matrix to 0.
template <class ElemType>
void GPUSparseMatrix<ElemType>::Reset()
{
    if (!OwnBuffer())
        LogicError("Cannot Reset since the buffer is managed externally.");

    m_nz = 0;
    m_blockSize = 0;
}

// copy features to GPU
template <class ElemType>
void GPUSparseMatrix<ElemType>::SetMatrixFromCSRFormat(const GPUSPARSE_INDEX_TYPE* h_CSRRow, const GPUSPARSE_INDEX_TYPE* h_Col, const ElemType* h_Val,
                                                       const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice /*= false*/, const DEVICEID_TYPE devId /*= -1*/)
{
    if (!OwnBuffer())
        LogicError("Cannot Set since the buffer is managed externally.");

    if (h_CSRRow == nullptr || h_Col == nullptr || h_Val == nullptr)
        LogicError("SetMatrixFromCSRFormat: nullptr passed in.");

    SetComputeDeviceId(PrepareDevice(devId));

    m_format = matrixFormatSparseCSR;
    Resize(numRows, numCols, nz, true, false);
    SetNzCount(nz);

    cudaMemcpyKind kind = IsOnDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    CUDA_CALL(cudaMemcpy(BufferPointer(), h_Val, NzSize(), kind));

    if (sizeof(CPUSPARSE_INDEX_TYPE) == sizeof(GPUSPARSE_INDEX_TYPE))
    {
        CUDA_CALL(cudaMemcpy(RowLocation(), h_CSRRow, RowSize(), kind));
        CUDA_CALL(cudaMemcpy(ColLocation(), h_Col, ColSize(), kind));
    }
    else
    {
        GPUSPARSE_INDEX_TYPE* pCol = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(RowSize() + ColSize());
        CopyBuffer(pCol, h_Col, MajorIndexCount());

        GPUSPARSE_INDEX_TYPE* pRow = pCol + MajorIndexCount();
        CopyBuffer(pRow, h_CSRRow, SecondaryIndexCount());

        CUDA_CALL(cudaMemcpy(RowLocation(), pRow, RowSize(), kind));
        CUDA_CALL(cudaMemcpy(ColLocation(), pCol, ColSize(), kind));
    }
}

// this function will allocate memory while the caller needs to release it
template <class ElemType>
void GPUSparseMatrix<ElemType>::GetMatrixFromCSRFormat(CPUSPARSE_INDEX_TYPE*& h_CSRRow, CPUSPARSE_INDEX_TYPE*& h_Col, ElemType*& h_Val, size_t& numElemAllocated, size_t& nz, size_t& numRows, size_t& numCols) const
{
    if (!OwnBuffer())
        LogicError("Cannot Set since the buffer is managed externally.");

    if (h_CSRRow != nullptr || h_Col != nullptr || h_Val != nullptr)
        LogicError("GetMatrixFromCSRFormat: Passed pointers must be nullptr");

    numElemAllocated = GetNumElemAllocated();
    nz = GetNumNZElements();
    numRows = GetNumRows();
    numCols = GetNumCols();

    if (IsEmpty() || nz == 0)
        return;
    else
    {
        h_Val = new ElemType[numElemAllocated];
        h_CSRRow = new CPUSPARSE_INDEX_TYPE[m_numRows + 1];
        h_Col = new CPUSPARSE_INDEX_TYPE[nz];

        PrepareDevice();
        CUDA_CALL(cudaMemcpy(h_Val, BufferPointer(), GetSizeElemAllocated(), cudaMemcpyDeviceToHost));

        if (sizeof(CPUSPARSE_INDEX_TYPE) == sizeof(GPUSPARSE_INDEX_TYPE))
        {
            CUDA_CALL(cudaMemcpy(h_CSRRow, RowLocation(), RowSize(), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(h_Col, ColLocation(), ColSize(), cudaMemcpyDeviceToHost));
        }
        else
        {
            GPUSPARSE_INDEX_TYPE* pCol = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(RowSize() + ColSize());
            GPUSPARSE_INDEX_TYPE* pRow = pCol + MajorIndexCount();

            CUDA_CALL(cudaMemcpy(pRow, RowLocation(), RowSize(), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(pCol, ColLocation(), ColSize(), cudaMemcpyDeviceToHost));

            CopyBuffer(h_Col, pCol, MajorIndexCount());
            CopyBuffer(h_CSRRow, pRow, SecondaryIndexCount());
        }
    }
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::SetMatrixFromCSCFormat(const CPUSPARSE_INDEX_TYPE* h_CSCCol, const CPUSPARSE_INDEX_TYPE* h_Row, const ElemType* h_Val,
                                                       const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice /*= false*/, const DEVICEID_TYPE devId /*= -1*/)
{
    if (!OwnBuffer())
        LogicError("Cannot Set since the buffer is managed externally.");

    if (h_CSCCol == nullptr || h_Row == nullptr || h_Val == nullptr)
        LogicError("SetMatrixFromCSCFormat: nullptr passed in.");

    SetComputeDeviceId(PrepareDevice(devId));
    m_format = matrixFormatSparseCSC;
    Resize(numRows, numCols, nz, true, false);
    SetNzCount(nz);

    cudaMemcpyKind kind = IsOnDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    CUDA_CALL(cudaMemcpy(BufferPointer(), h_Val, NzSize(), kind));

    if (sizeof(CPUSPARSE_INDEX_TYPE) == sizeof(GPUSPARSE_INDEX_TYPE))
    {
        CUDA_CALL(cudaMemcpy(RowLocation(), h_Row, RowSize(), kind));
        CUDA_CALL(cudaMemcpy(ColLocation(), h_CSCCol, ColSize(), kind));
    }
    else
    {
        GPUSPARSE_INDEX_TYPE* pCol = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(RowSize() + ColSize());
        GPUSPARSE_INDEX_TYPE* pRow = pCol + SecondaryIndexCount();

        CopyBuffer(pCol, h_CSCCol, SecondaryIndexCount());
        CopyBuffer(pRow, h_Row, MajorIndexCount());

        CUDA_CALL(cudaMemcpy(RowLocation(), pRow, RowSize(), kind));
        CUDA_CALL(cudaMemcpy(ColLocation(), pCol, ColSize(), kind));
    }
}

// this function will allocate memory while the caller needs to release it
template <class ElemType>
void GPUSparseMatrix<ElemType>::GetMatrixFromCSCFormat(GPUSPARSE_INDEX_TYPE*& h_CSCCol, GPUSPARSE_INDEX_TYPE*& h_Row, ElemType*& h_Val, size_t& numElemAllocated, size_t& nz, size_t& numRows, size_t& numCols) const
{
    if (h_CSCCol != nullptr || h_Row != nullptr || h_Val != nullptr)
        LogicError("GetMatrixFromCSCFormat: Passed pointers must be nullptr");

    numElemAllocated = GetNumElemAllocated();
    nz = GetNumNZElements();
    numRows = GetNumRows();
    numCols = GetNumCols();

    if (IsEmpty())
        return;
    else
    {
        h_Val = new ElemType[numElemAllocated];
        h_CSCCol = new GPUSPARSE_INDEX_TYPE[m_numRows + 1];
        h_Row = new GPUSPARSE_INDEX_TYPE[nz];

        PrepareDevice();
        CUDA_CALL(cudaMemcpy(h_Val, BufferPointer(), GetSizeElemAllocated(), cudaMemcpyDeviceToHost));

        if (sizeof(CPUSPARSE_INDEX_TYPE) == sizeof(GPUSPARSE_INDEX_TYPE))
        {
            CUDA_CALL(cudaMemcpy(h_Row, RowLocation(), RowSize(), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(h_CSCCol, ColLocation(), ColSize(), cudaMemcpyDeviceToHost));
        }
        else
        {
            GPUSPARSE_INDEX_TYPE* pCol = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(RowSize() + ColSize());
            GPUSPARSE_INDEX_TYPE* pRow = pCol + SecondaryIndexCount();

            CUDA_CALL(cudaMemcpy(pRow, RowLocation(), RowSize(), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(pCol, ColLocation(), ColSize(), cudaMemcpyDeviceToHost));

            CopyBuffer(h_CSCCol, pCol, SecondaryIndexCount());
            CopyBuffer(h_Row, pRow, MajorIndexCount());
        }
    }
}

#pragma endregion Constructors and Destructor

#pragma region Static BLAS Functions

// dense X sparse = dense
template <class ElemType>
void GPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& lhs, const bool transposeA,
                                                       const GPUSparseMatrix<ElemType>& rhs, const bool transposeB, ElemType beta, GPUMatrix<ElemType>& c)
{
    if (lhs.GetComputeDeviceId() != rhs.GetComputeDeviceId() || (lhs.GetComputeDeviceId() != c.GetComputeDeviceId()))
        RuntimeError("GPUSparseMatrix::MultiplyAndWeightedAdd: All matrices must be on the same GPU");

    if (lhs.IsEmpty() || rhs.IsEmpty())
        LogicError("GPUSparseMatrix::MultiplyAndWeightedAdd:  one of the input matrix is empty.");

    int m = transposeA ? (int) lhs.GetNumCols() : (int) lhs.GetNumRows();
    int k = transposeA ? (int) lhs.GetNumRows() : (int) lhs.GetNumCols();
    int l = transposeB ? (int) rhs.GetNumCols() : (int) rhs.GetNumRows();
    int n = transposeB ? (int) rhs.GetNumRows() : (int) rhs.GetNumCols();

    assert(m > 0 && k > 0 && l > 0 && n > 0); // converting from size_t to int may cause overflow
    assert(k == l);
    if (k != l)
    {
        InvalidArgument("GPUSparseMatrix::MultiplyAndWeightedAdd: The inner dimensions of a and b must match.");
    }

    if (beta == 0)
        c.Resize(m, n);
    else
        c.VerifySize(m, n); // Can't resize if beta != 0

    c.PrepareDevice();
    if (rhs.m_format == MatrixFormat::matrixFormatSparseCSC)
    {
        ConvolveAndWeightedAdd(alpha, lhs, transposeA, rhs, transposeB, beta, c, 1, 1, false, false);
    }
    else if (rhs.m_format == matrixFormatSparseCSR)
    {
        GPUSparseMatrix<ElemType> tempMatrix(rhs.GetComputeDeviceId(), matrixFormatSparseCSC);
        rhs.ConvertToSparseFormat(matrixFormatSparseCSC, tempMatrix);
        MultiplyAndWeightedAdd(alpha, lhs, transposeA, tempMatrix, transposeB, beta, c);
    }
    else
    {
        NOT_IMPLEMENTED;
    }
}

// dense X sparse = dense
template <class ElemType>
void GPUSparseMatrix<ElemType>::ConvolveAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& lhs, const bool transposeA,
                                                       const GPUSparseMatrix<ElemType>& rhs, const bool transposeB, ElemType beta,
                                                       GPUMatrix<ElemType>& c, size_t numChannels, size_t horizontalSubsample, bool padding, bool channelwise)
{
    if (lhs.GetComputeDeviceId() != rhs.GetComputeDeviceId() || (lhs.GetComputeDeviceId() != c.GetComputeDeviceId()))
        RuntimeError("GPUSparseMatrix<ElemType>::ConvolveAndWeightedAdd: All matrices must be on the same GPU");

    if (lhs.IsEmpty() || rhs.IsEmpty())
        LogicError("GPUSparseMatrix<ElemType>::ConvolveAndWeightedAdd:  one of the input matrix is empty.");

    int m = transposeA ? (int) lhs.GetNumCols() : (int) lhs.GetNumRows();
    int k = transposeA ? (int) lhs.GetNumRows() : (int) lhs.GetNumCols();
    int l = transposeB ? (int) rhs.GetNumCols() : (int) rhs.GetNumRows();
    int n = transposeB ? (int) rhs.GetNumRows() : (int) rhs.GetNumCols();

    assert(m > 0 && k > 0 && l > 0 && n > 0); // converting from size_t to int may cause overflow

    int numSteps = 0;
    if (padding)
        numSteps = (int) ceil(1.0 * l / (horizontalSubsample * numChannels));
    else if (l >= k)
        numSteps = 1 + (l - k) / (horizontalSubsample * numChannels);

    if (numSteps == 0)
        LogicError("ConvolveAndWeightedAdd: number of steps is zero. Matrix dimensions are incorrect or set padding to true.");

    int cRows = m * numSteps;
    int cCols = n;

    if (beta == 0)
        c.Resize(cRows, cCols);
    else
        c.VerifySize(cRows, cCols); // Can't resize if beta != 0

    c.PrepareDevice();
    if (rhs.m_format == MatrixFormat::matrixFormatSparseCSC)
    {
        if (!transposeB)
        {
            int blocksPerGrid = (int) ceil(1.0 * cRows * cCols / GridDim::maxThreadsPerBlock);
            SyncGuard syncGuard;
            _dense1DConvMultSparseCSCAndWeightedAddToDense<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
                m,                   // rowDense
                k,                   // colDense
                n,                   // colSparse
                numChannels,         // number of input channels
                numSteps,            // convolution num steps
                horizontalSubsample, // convolution step size
                channelwise,         // channelwise or pixelwise multiplication
                alpha,
                reinterpret_cast<const ElemType*>(lhs.BufferPointer()), // dense
                transposeA,
                reinterpret_cast<const ElemType*>(rhs.BufferPointer()), // sparse nz values
                rhs.RowLocation(),
                rhs.ColLocation(),
                beta,
                reinterpret_cast<ElemType*>(c.BufferPointer()) // dense target
                );
        }
        else
        {
            if (beta != 1.0)
            {
                RuntimeError("Only support c += alpha * a operation");
            }
            int blocksPerGrid = (int) ceil(1.0 * cRows / GridDim::maxThreadsPerBlock);
            SyncGuard syncGuard;
            for (int rowInB = 0; rowInB < l; rowInB++)
            {
                _dense1DConvMultSparseCSCTransposeAndAddToDense<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
                    m,                   // rowDense
                    k,                   // colDense
                    n,                   // colSparse
                    numChannels,         // number of input channels
                    numSteps,            // convolution num steps
                    horizontalSubsample, // convolution step size
                    channelwise,         // channelwise or pixelwise multiplication
                    rowInB,
                    alpha,
                    reinterpret_cast<const ElemType*>(lhs.BufferPointer()), // dense
                    transposeA,
                    reinterpret_cast<const ElemType*>(rhs.BufferPointer()), // sparse nz values
                    rhs.RowLocation(),
                    rhs.ColLocation(),
                    reinterpret_cast<ElemType*>(c.BufferPointer()) // dense target
                    );
            }
        }
    }
    else
    {
        NOT_IMPLEMENTED;
    }
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::TensorShuffleScaleAndAdd(ElemType keepWeight, const GPUSparseMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const GPUSparseMatrix<ElemType>& b, GPUSparseMatrix<ElemType>& c)
{
    if (!c.OwnBuffer())
        LogicError("Cannot modify externally managed matrix");

    if (a.GetComputeDeviceId() != c.GetComputeDeviceId() || b.GetComputeDeviceId() != c.GetComputeDeviceId())
        RuntimeError("GPUSparseMatrix<ElemType>::TensorShuffleScaleAndAdd: All matrices must be on the same GPU");

    if (a.m_format != MatrixFormat::matrixFormatSparseCSC || b.m_format != MatrixFormat::matrixFormatSparseCSC || c.m_format != MatrixFormat::matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    // Can't distribute the operations if we need to move values across columns
    if (a.GetNumCols() != T || keepWeight != 0 || scaleFactor != 1)
        NOT_IMPLEMENTED;

    if (a.GetNumRows() != D * S * M * K)
        LogicError("GPUSparseMatrix<ElemType>::TensorShuffleScaleAndAdd: tensor dimensions and underlying matrix dimensions don't match");

    c.Resize(a.GetNumRows(), a.GetNumCols(), a.GetNumElemAllocated(), true, false);
    c.SetNzCount(a.GetNumNZElements());

    if (c.m_nz > 0)
    {
        c.PrepareDevice();
        SyncGuard syncGuard;
        CUDA_LONG N = (CUDA_LONG) c.GetNumElemAllocated();
        int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
        _tensorShuffleScaleAndAddRowSparse<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            reinterpret_cast<const ElemType*>(a.BufferPointer()), // source nz values
            a.RowLocation(),
            a.ColLocation(),
            reinterpret_cast<ElemType*>(c.BufferPointer()), // target nz values
            c.RowLocation(),
            c.ColLocation(),
            D, S, M, K, T,
            c.GetNumElemAllocated());
    }
    else
    {
        CUDA_CALL(cudaMemset(c.BufferPointer(), 0, c.BufferSizeAllocated()));
    }
}

// backward pass from hidden layer to feature weight
// dense X sparse = sparse
template <class ElemType>
void GPUSparseMatrix<ElemType>::MultiplyAndAdd(ElemType alpha, const GPUMatrix<ElemType>& lhs, const bool transposeA,
                                               const GPUSparseMatrix<ElemType>& rhs, const bool transposeB, GPUSparseMatrix<ElemType>& c)
{
    if (!c.OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (lhs.GetComputeDeviceId() != rhs.GetComputeDeviceId())
        RuntimeError("GPUSparseMatrix::MultiplyAndAdd: All matrices must be on the same GPU");

    int m = transposeA ? (int) lhs.GetNumCols() : (int) lhs.GetNumRows();
    int k = transposeA ? (int) lhs.GetNumRows() : (int) lhs.GetNumCols();
    int l = transposeB ? (int) rhs.GetNumCols() : (int) rhs.GetNumRows();
    int n = transposeB ? (int) rhs.GetNumRows() : (int) rhs.GetNumCols();

    assert(m > 0 && k > 0 && l > 0 && n > 0);
    (void) m;
    (void) n; // converting from size_t to int may cause overflow
    assert(k == l);
    if (k != l)
    {
        InvalidArgument("GPUSparseMatrix::MultiplyAndAdd: The inner dimensions of a and b must match.");
    }

    if (!transposeA && !transposeB)
    {
        NOT_IMPLEMENTED;
    }
    else if (!transposeA && transposeB)
    {
        if (rhs.GetFormat() != matrixFormatSparseCSC)
            NOT_IMPLEMENTED;

        c.SetFormat(matrixFormatSparseBlockCol);

        lhs.PrepareDevice();

        int blocksPerGrid = 0;
        SyncGuard syncGuard;

        // based on the size of m_nz in rhs and numCols in the resulted matrix we use different approaches
        if (n * 10 < GridDim::maxThreadsPerBlock * rhs.m_nz)
        {
            c.Resize(m, n, 1, true, false); // reserve memory for BlockId2ColOrRow() and ColOrRow2BlockId()

            size_t* blockSize = TracingGPUMemoryAllocator::Allocate<size_t>(lhs.GetComputeDeviceId(), 1);
            CUDA_CALL(cudaMemset(blockSize, 0, sizeof(size_t)));

            CUDA_CALL(cudaMemset(c.BlockId2ColOrRow(), 0, sizeof(GPUSPARSE_INDEX_TYPE) * (n)));

            blocksPerGrid = (int) ceil(((double) rhs.m_nz) / GridDim::maxThreadsPerBlock);
            _findColsWithValues<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
                rhs.RowLocation(), c.BlockId2ColOrRow(), rhs.m_nz);
                
            blocksPerGrid = (int) ceil(((double) n) / GridDim::maxThreadsPerBlock);
            _determineBlockIds<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
                c.BlockId2ColOrRow(), c.ColOrRow2BlockId(), n, blockSize);

                
            CUDA_CALL(cudaMemcpy(&c.m_blockSize, blockSize, sizeof(size_t), cudaMemcpyDeviceToHost));
            TracingGPUMemoryAllocator::Free<size_t>(lhs.GetComputeDeviceId(), blockSize);

            size_t nnz = m * c.m_blockSize;
            c.Resize(m, n, nnz, true, true); // we need to keep the col2blockid and blockid2col info when resizing.
            c.m_nz = nnz;
            CUDA_CALL(cudaMemset(c.BufferPointer(), 0, sizeof(ElemType) * (c.m_elemSizeAllocated)));

            LONG64 N = (LONG64) lhs.GetNumElements(); // here we process for each row in lhs and each column in rhs (==columns in lhs)
            blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
            _denseMulSparseCSCTransposeToSparseBlockCol2<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
                alpha,
                lhs.BufferPointer(),
                m,
                l,
                rhs.BufferPointer(),
                rhs.RowLocation(),
                rhs.ColLocation(),
                c.ColOrRow2BlockId(),
                c.BufferPointer());
        }
        else
        {
            c.m_blockSize = rhs.IdentifyRowsWithValues();
            size_t nnz = m * c.m_blockSize;
            c.Resize(m, n, nnz, true, false);
            c.m_nz = nnz;
            CUDA_CALL(cudaMemset(c.BufferPointer(), 0, sizeof(ElemType) * (c.m_elemSizeAllocated)));
            CUDA_CALL(cudaMemset(c.BlockId2ColOrRow(), 0, sizeof(GPUSPARSE_INDEX_TYPE) * (c.m_blockSize)));

            LONG64 N = (LONG64) lhs.GetNumElements(); // here we process for each row in lhs and each column in rhs (==columns in lhs)
            blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
            _denseMulSparseCSCTransposeToSparseBlockCol<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
                alpha,
                lhs.BufferPointer(),
                m,
                l,
                rhs.BufferPointer(),
                rhs.RowLocation(),
                rhs.ColLocation(),
                rhs.m_rowToId,
                c.BufferPointer(),
                c.BlockId2ColOrRow());
        }

    }
    else if (transposeA && !transposeB)
    {
        NOT_IMPLEMENTED;
    }
    else
    {
        NOT_IMPLEMENTED;
    }
}

// find the rows of rhs with values
template <class ElemType>
size_t GPUSparseMatrix<ElemType>::IdentifyRowsWithValues() const
{
    if (GetFormat() != matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    map<size_t, GPUSPARSE_INDEX_TYPE> indexer;
    GPUSPARSE_INDEX_TYPE* rowToId = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(sizeof(GPUSPARSE_INDEX_TYPE) * m_nz * 2);
    GPUSPARSE_INDEX_TYPE* h_Row = rowToId + m_nz;
    CUDA_CALL(cudaMemcpy(h_Row, RowLocation(), sizeof(GPUSPARSE_INDEX_TYPE) * m_nz, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < m_nz; i++)
    {
        size_t row = h_Row[i];
        if (indexer.find(row) == indexer.end())
        {
            size_t id = indexer.size(); // We need to assign size to a temp variable due to difference in Linux and Windows
            indexer[row] = id;
        }
        rowToId[i] = indexer[row];
    }
    CUDA_CALL(cudaMemcpy(m_rowToId, rowToId, sizeof(GPUSPARSE_INDEX_TYPE) * m_nz, cudaMemcpyHostToDevice));
    return indexer.size();
}

// used for gradients udpate
template <class ElemType>
void GPUSparseMatrix<ElemType>::ScaleAndAdd(const ElemType alpha, const GPUSparseMatrix<ElemType>& lhs, GPUMatrix<ElemType>& rhs)
{
    if (lhs.GetNumRows() != rhs.GetNumRows() || lhs.GetNumCols() != rhs.GetNumCols())
        LogicError("ScaleAndAdd: dimension mismatch");

    if (lhs.GetComputeDeviceId() != rhs.GetComputeDeviceId())
        RuntimeError("GPUSparseMatrix::ScaleAndAdd: All matrices must be on the same GPU");

    if (lhs.m_format == matrixFormatSparseBlockCol || lhs.m_format == matrixFormatSparseBlockRow)
    {
        bool blockCol = (lhs.m_format == matrixFormatSparseBlockCol);

        SyncGuard syncGuard;
        LONG64 N = (LONG64) lhs.GetNumNZElements();
        int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
        _scaleSparseBlockAndAddToDense<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(
            alpha,
            blockCol,
            lhs.GetNumRows(),
            lhs.GetNumCols(),
            lhs.m_blockSize,
            lhs.BufferPointer(),
            lhs.BlockId2ColOrRow(),
            rhs.BufferPointer());

    }
    else
    {
        ScaleAndAdd(alpha, lhs, 1, rhs, rhs);
    }
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceTruncate(const ElemType threshold)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    CUDA_LONG N = (CUDA_LONG) GetNumNZElements();

    CUDA_LONG blocksPerGrid = (CUDA_LONG) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    ElemType* values = NzValues();
    _inplaceTruncate<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(values, threshold, N);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceSoftThreshold(const ElemType threshold)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    CUDA_LONG N = (CUDA_LONG) GetNumNZElements();

    CUDA_LONG blocksPerGrid = (CUDA_LONG) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    ElemType* values = NzValues();
    _inplaceSoftThreshold<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(values, threshold, N);
    return *this;
}

// normal update for smoothed gradients c and current gradients (this)
template <class ElemType>
void GPUSparseMatrix<ElemType>::NormalGrad(GPUMatrix<ElemType>& c, const ElemType momentum)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (c.IsEmpty())
    {
        c.Resize(GetNumRows(), GetNumCols());
        c.SetValue(0.0);
    }

    if (m_format == matrixFormatSparseBlockCol || m_format == matrixFormatSparseBlockRow)
    {
        bool isBlockCol = (m_format == MatrixFormat::matrixFormatSparseBlockCol);
        SyncGuard syncGuard;
        LONG64 N = (LONG64) GetNumNZElements();
        int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);

        _normalGradForSparseBlock<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(
            momentum,
            isBlockCol,
            GetNumRows(),
            GetNumCols(),
            m_blockSize,
            BufferPointer(),
            BlockId2ColOrRow(),
            c.BufferPointer());
    }
    else
    {
        NOT_IMPLEMENTED;
    }
}

template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::Adagrad(GPUMatrix<ElemType>& c, const bool needAveMultiplier)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    size_t numColsNeeded = GetNumCols();
    if (needAveMultiplier)
        numColsNeeded += GetNumCols();

    if (c.IsEmpty() || c.GetNumCols() < numColsNeeded)
    {
        c.Resize(GetNumRows(), numColsNeeded);
        c.SetValue(0.0);
    }

    assert(c.GetNumRows() == GetNumRows() && c.GetNumCols() == numColsNeeded);

    size_t n = this->GetNumElements();

    ElemType* multipliers = nullptr;
    if (needAveMultiplier)
        multipliers = c.GetArray() + n; // temp memory used to store multipliers,

    if (m_format == MatrixFormat::matrixFormatSparseCSC || m_format == MatrixFormat::matrixFormatSparseCSR)
    {
        NOT_IMPLEMENTED;
    }
    else if (m_format == MatrixFormat::matrixFormatSparseBlockCol || m_format == MatrixFormat::matrixFormatSparseBlockRow)
    {
        int blocksPerGrid = (m_nz + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
        bool colMajor = (m_format == MatrixFormat::matrixFormatSparseBlockCol ? true : false);
        size_t len = colMajor ? GetNumRows() : GetNumCols();
        _adagrad4BlockSparse<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(c.GetArray(), c.GetNumRows(), BufferPointer(), BlockId2ColOrRow(), multipliers, colMajor, len, m_nz);
    }
    else
        NOT_IMPLEMENTED;

    if (!needAveMultiplier)
        return 1;

    cublasHandle_t cuHandle = GPUMatrix<ElemType>::GetCublasHandle(GetComputeDeviceId());
    if (sizeof(ElemType) == sizeof(float))
    {
        float aveMultiplier = 0;
        CUBLAS_CALL(cublasSasum(cuHandle, (LONG64) m_nz, reinterpret_cast<float*>(multipliers), 1, &aveMultiplier));
        return (ElemType) aveMultiplier / m_nz;
    }
    else
    {
        double aveMultiplier = 0;
        CUBLAS_CALL(cublasDasum(cuHandle, (LONG64) m_nz, reinterpret_cast<double*>(multipliers), 1, &aveMultiplier));
        return (ElemType) aveMultiplier / m_nz;
    }
}

// sparse X dense = dense
template <class ElemType>
void GPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& a, const bool transposeA,
                                                       const GPUMatrix<ElemType>& b, const bool transposeD, ElemType beta, GPUMatrix<ElemType>& c)
{
    if (a.m_format != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    if (transposeD)
        NOT_IMPLEMENTED;

    if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || (b.GetComputeDeviceId() != a.GetComputeDeviceId()))
        RuntimeError("MultiplyAndWeightedAdd: All matrices must be on the same GPU");

    a.PrepareDevice();
    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseOperation_t oper = transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

    int m = (int) a.GetNumRows();
    int n = (int) b.GetNumCols();
    assert(n == (int) c.GetNumCols());
    int k = (int) a.GetNumCols();

    SyncGuard syncGuard;
    if (sizeof(ElemType) == sizeof(float))
    {
        CUSPARSE_CALL(cusparseScsrmm(cusparseHandle, oper, m, n, k, (int) a.GetNumElemAllocated(), reinterpret_cast<float*>(&alpha), descr, reinterpret_cast<const float*>(a.BufferPointer()),
                                     a.RowLocation(), a.ColLocation(), reinterpret_cast<float*>(b.BufferPointer()),
                                     (int) b.GetNumRows(), reinterpret_cast<float*>(&beta), reinterpret_cast<float*>(c.BufferPointer()), (int) c.GetNumRows()));
    }
    else
    {
        CUSPARSE_CALL(cusparseDcsrmm(cusparseHandle, oper, m, n, k, (int) a.GetNumElemAllocated(), reinterpret_cast<double*>(&alpha), descr, reinterpret_cast<const double*>(a.BufferPointer()),
                                     a.RowLocation(), a.ColLocation(), reinterpret_cast<double*>(b.BufferPointer()),
                                     (int) b.GetNumRows(), reinterpret_cast<double*>(&beta), reinterpret_cast<double*>(c.BufferPointer()), (int) c.GetNumRows()));
    }
    CUSPARSE_CALL(cusparseDestroy(cusparseHandle));
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::Multiply(const GPUSparseMatrix<ElemType>& S, const GPUMatrix<ElemType>& D, GPUMatrix<ElemType>& C)
{
    C.Resize(S.GetNumRows(), D.GetNumCols());

    MultiplyAndWeightedAdd(1, S, false, D, false, 0, C);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::Multiply(const GPUMatrix<ElemType>& D, const GPUSparseMatrix<ElemType>& S, GPUMatrix<ElemType>& C)
{
    C.Resize(S.GetNumCols(), D.GetNumRows());

    MultiplyAndWeightedAdd(1, D, false, S, false, 0, C);
}

// ElemCountFromBufferSize - Return the elemCountAllocated for a particular buffersize
// totalBufferSize - total buffer we have to use
// return: size of allocated elements/index slots available
template <class ElemType>
size_t GPUSparseMatrix<ElemType>::ElemCountFromBufferSize(const size_t numRows, const size_t numCols, const MatrixFormat format, const size_t totalBufferSize) const
{
    size_t elemSizeAllocated;
    if (format == matrixFormatSparseCSC)
    {
        elemSizeAllocated = (totalBufferSize - sizeof(GPUSPARSE_INDEX_TYPE) * (numCols + 1)) / (sizeof(GPUSPARSE_INDEX_TYPE) + sizeof(ElemType));
    }
    else if (format == matrixFormatSparseCSR)
    {
        elemSizeAllocated = (totalBufferSize - sizeof(GPUSPARSE_INDEX_TYPE) * (numRows + 1)) / (sizeof(GPUSPARSE_INDEX_TYPE) + sizeof(ElemType));
    }
    else if (format == matrixFormatSparseBlockCol)
    {
        elemSizeAllocated = (totalBufferSize - sizeof(GPUSPARSE_INDEX_TYPE) * 2 * numCols) / sizeof(ElemType);
    }
    else if (format == matrixFormatSparseBlockCol || format == matrixFormatSparseBlockRow)
    {
        elemSizeAllocated = (totalBufferSize - sizeof(GPUSPARSE_INDEX_TYPE) * 2 * numRows) / sizeof(ElemType);
    }
    else // uncompressed COO format
    {
        elemSizeAllocated = totalBufferSize / (2 * sizeof(GPUSPARSE_INDEX_TYPE) + sizeof(ElemType));
    }
    return elemSizeAllocated;
}

template <class ElemType>
size_t GPUSparseMatrix<ElemType>::ElemCountFromBufferSize() const
{
    return ElemCountFromBufferSize(m_numRows, m_numCols, m_format, m_totalBufferSizeAllocated);
}

// PrepareBuffer - Get the dimensions start buffer, computes the starting row/column of each value
// m - rows in the source
// n - cols in the source
// canReuseBuffer - target matrix can be reused for temporary space
// func - function to call to count elements in the result (returns count, and fills csrRowPtr array)
template <class ElemType>
void GPUSparseMatrix<ElemType>::PrepareBuffer(size_t m, size_t n, bool canReuseBuffer, std::function<size_t(GPUSPARSE_INDEX_TYPE* csrRowPtrC)> func)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (this->m_format != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    PrepareDevice();

    GPUSPARSE_INDEX_TYPE* csrRowPtrC = nullptr;
    GPUSparseMatrix<ElemType>& c = *this;
    size_t cSize = c.BufferSizeAllocated();
    size_t rowBufferRequired = (m + 1) * sizeof(GPUSPARSE_INDEX_TYPE);
    bool allocatedBuffer = false;

    // do we have enough memory to store just the row buffer?
    if (cSize >= rowBufferRequired && c.BufferPointer() != nullptr && canReuseBuffer)
    {
        csrRowPtrC = (GPUSPARSE_INDEX_TYPE*) c.BufferPointer();
    }
    else
    {
        csrRowPtrC = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(m_computeDevice, rowBufferRequired / sizeof(GPUSPARSE_INDEX_TYPE));
        allocatedBuffer = true;
    }

    // get the non-zero count from the function (and
    size_t nnzC = func(csrRowPtrC);

    // now we know the number of Non-zeros in the result set, set the output size
    c.Resize(m, n, nnzC, true, false);
    c.m_nz = nnzC;

    CUDA_CALL(cudaMemcpy(c.SecondaryIndexLocation(), csrRowPtrC, c.SecondaryIndexSize(), cudaMemcpyDeviceToDevice));

    // if we allocated the buffer, free it here
    if (allocatedBuffer)
        TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(m_computeDevice, csrRowPtrC);
}

// Multiply - multiply one spares matrix by another sparse matrix
// S1 - first sparse matrix
// transposeS1 - transpose first matrix?
// S2 - second sparse matrix
// transposeS2 - tanspose second matrix?
// c - result matrix
// NOTE: if c has enough space allocated, it will be reused, otherwise it will be freed and a new memory block used
template <class ElemType>
void GPUSparseMatrix<ElemType>::Multiply(const GPUSparseMatrix<ElemType>& S1, bool transposeS1, const GPUSparseMatrix<ElemType>& S2, bool transposeS2, GPUSparseMatrix<ElemType>& c)
{
    if (!c.OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (S1.m_format != matrixFormatSparseCSR || S2.m_format != matrixFormatSparseCSR || c.m_format != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    if (S1.GetComputeDeviceId() != S2.GetComputeDeviceId())
        RuntimeError("Sparse matrix multiply: both matrices must be on the same device");

    S1.PrepareDevice();
    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));
    cusparseMatDescr_t descrA = 0, descrB = 0, descrC = 0;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrA));
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrB));
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrC));
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
    cusparseOperation_t operA = transposeS1 ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t operB = transposeS2 ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

    int m = int(transposeS1 ? S1.GetNumCols() : S1.GetNumRows());
    int n = int(transposeS2 ? S2.GetNumRows() : S2.GetNumCols());
    int k = int(transposeS1 ? S1.GetNumRows() : S1.GetNumCols());
    int l = int(transposeS2 ? S2.GetNumCols() : S2.GetNumRows());
    if (k != l)
        RuntimeError("Sparse matrix multiply: dimensionality mismatch");

    int nnzA = (int) S1.GetNumNZElements();
    int nnzB = (int) S2.GetNumNZElements();

    SyncGuard syncGuard;
    // Step 1
    c.PrepareBuffer(m, n, false, // false means we cannot reuse the "c" buffer if it exists for temporaries
                    [&](GPUSPARSE_INDEX_TYPE* csrRowPtrC) -> size_t
                    {
                        int nnzTotal = -1;
                        CUSPARSE_CALL(cusparseXcsrgemmNnz(cusparseHandle, operA, operB, m, n, k, descrA, nnzA, S1.RowLocation(), S1.ColLocation(), descrB, nnzB,
                                                          S2.RowLocation(), S2.ColLocation(), descrC, csrRowPtrC, &nnzTotal));
                        return nnzTotal;
                    });

    // Step 2
    if (sizeof(float) == sizeof(ElemType))
    {
        CUSPARSE_CALL(cusparseScsrgemm(cusparseHandle, operA, operB, m, n, k, descrA, nnzA, (const float*) S1.BufferPointer(), S1.RowLocation(), S1.ColLocation(),
                                       descrB, nnzB, (const float*) S2.BufferPointer(), S2.RowLocation(), S2.ColLocation(),
                                       descrC, (float*) c.BufferPointer(), c.RowLocation(), c.ColLocation()));
    }
    else
    {
        CUSPARSE_CALL(cusparseDcsrgemm(cusparseHandle, operA, operB, m, n, k, descrA, nnzA, (const double*) S1.BufferPointer(), S1.RowLocation(), S1.ColLocation(),
                                       descrB, nnzB, (const double*) S2.BufferPointer(), S2.RowLocation(), S2.ColLocation(),
                                       descrC, (double*) c.BufferPointer(), c.RowLocation(), c.ColLocation()));
    }
    cusparseDestroy(cusparseHandle);
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignProductOf(const GPUSparseMatrix<ElemType>& a, const bool transposeA, const GPUSparseMatrix<ElemType>& b, const bool transposeB)
{
    Multiply(a, transposeA, b, transposeB, *this);
    return *this;
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& a, ElemType beta, const GPUSparseMatrix<ElemType>& b, GPUSparseMatrix<ElemType>& c)
{
    if (!c.OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (a.m_format != matrixFormatSparseCSR || b.m_format != matrixFormatSparseCSR || c.m_format != matrixFormatSparseCSR)
    {
        NOT_IMPLEMENTED;
    }

    if (a.GetNumCols() != b.GetNumCols() || a.GetNumRows() != b.GetNumRows())
        RuntimeError("Dimensions mismatch in ScaleAndAdd");
    if (a.GetComputeDeviceId() != b.GetComputeDeviceId())
        RuntimeError("ScaleAndAdd: matrices must be on the same device");

    int m = (int) a.GetNumRows();
    int n = (int) a.GetNumCols();
    int nnzA = (int) a.GetNumNZElements();
    int nnzB = (int) b.GetNumNZElements();

    a.PrepareDevice();
    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));
    cusparseMatDescr_t descrA = 0, descrB = 0, descrC = 0;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrA));
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrB));
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrC));
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);

    SyncGuard syncGuard;
    // Step 1
    bool inOutParameter = (&b == &c);
    c.PrepareBuffer(m, n, !inOutParameter, 
                    [&](GPUSPARSE_INDEX_TYPE* csrRowPtrC) -> size_t
                    {
                        int nnzTotal = -1;
                        CUSPARSE_CALL(cusparseXcsrgeamNnz(cusparseHandle, m, n, descrA, nnzA, a.RowLocation(), a.ColLocation(), descrB, nnzB, b.RowLocation(), b.ColLocation(), descrC, csrRowPtrC, &nnzTotal));
                        return nnzTotal;
                    });

    // Step 2
    if (sizeof(ElemType) == sizeof(float))
    {
        CUSPARSE_CALL(cusparseScsrgeam(cusparseHandle, m, n, reinterpret_cast<const float*>(&alpha), descrA, nnzA, reinterpret_cast<const float*>(a.BufferPointer()), a.RowLocation(), a.ColLocation(),
                                       reinterpret_cast<const float*>(&beta), descrB, nnzB, reinterpret_cast<const float*>(b.BufferPointer()), b.RowLocation(), b.ColLocation(), descrC, reinterpret_cast<float*>(c.BufferPointer()), c.RowLocation(), c.ColLocation()));
    }
    else
    {
        CUSPARSE_CALL(cusparseDcsrgeam(cusparseHandle, m, n, reinterpret_cast<const double*>(&alpha), descrA, nnzA, reinterpret_cast<const double*>(a.BufferPointer()), a.RowLocation(), a.ColLocation(),
                                       reinterpret_cast<const double*>(&beta), descrB, nnzB, reinterpret_cast<const double*>(b.BufferPointer()), b.RowLocation(), b.ColLocation(), descrC, reinterpret_cast<double*>(c.BufferPointer()), c.RowLocation(), c.ColLocation()));
    }
    cusparseDestroy(cusparseHandle);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& a, ElemType beta, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    if (a.m_format != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    if (a.GetNumRows() != b.GetNumRows() || a.GetNumRows() != c.GetNumRows() || a.GetNumCols() != b.GetNumCols() || a.GetNumCols() != c.GetNumCols())
        LogicError("ScaleAndAdd: dimension mismatch");
    if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || a.GetComputeDeviceId() != c.GetComputeDeviceId())
        RuntimeError("ScaleAndAdd: matrices must be on the same device");
    b.PrepareDevice();
    // copy b to c
    CUDA_CALL(cudaMemcpy(c.BufferPointer(), b.BufferPointer(), sizeof(ElemType) * b.GetNumElements(), cudaMemcpyDeviceToDevice));
    if (beta != 1)
    {
        c *= beta;
    }
    SyncGuard syncGuard;
    CUDA_LONG M = (CUDA_LONG) a.GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * M / GridDim::maxThreadsPerBlock);
    _sparseCSRPlusDense<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(alpha, a.BufferPointer(), a.RowLocation(), a.ColLocation(), c.BufferPointer(), M);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const GPUMatrix<ElemType>& a, ElemType beta, const GPUSparseMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    ScaleAndAdd(beta, b, alpha, a, c);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::Scale(ElemType alpha, GPUSparseMatrix<ElemType>& a)
{
    if (!a.OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (a.IsEmpty())
        return;

    CUDA_LONG N = (CUDA_LONG) a.GetNumNZElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _scaleArray<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(alpha, a.NzValues(), N);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::ElementWisePower(ElemType alpha, const GPUSparseMatrix<ElemType>& a, GPUSparseMatrix<ElemType>& c)
{
    if (!c.OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        if (a.IsEmpty())
            LogicError("ElementWisePower:  The input matrix a is empty.");

        c.ResizeAsAndCopyIndexFrom(a);

        SyncGuard syncGuard;
        a.PrepareDevice();
        CUDA_LONG N = (CUDA_LONG) a.GetNumNZElements();
        int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
        _elementWisePowerOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(alpha, a.NzValues(), c.NzValues(), N);
    }
}

// sparse x dense = scalar
template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::InnerProductOfMatrices(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.m_format != matrixFormatSparseCSR && a.m_format != matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    if (a.GetComputeDeviceId() != b.GetComputeDeviceId())
        RuntimeError("a and b must be on the same device");

    int m = (int) a.GetNumRows();
    int n = (int) a.GetNumCols();
    int nnz = (int) a.GetNumNZElements();

    ElemType* cscValA = nullptr;
    GPUSPARSE_INDEX_TYPE* cscRowIndA = nullptr;
    GPUSPARSE_INDEX_TYPE* cscColPtrA = nullptr;

    cusparseAction_t cpVals = CUSPARSE_ACTION_NUMERIC;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
    cusparseHandle_t cusparseHandle = 0;

    if (a.m_format == matrixFormatSparseCSR) // need to put a in ColumnMajor format
    {
        cscValA = TracingGPUMemoryAllocator::Allocate<ElemType>(a.GetComputeDeviceId(), nnz);
        cscRowIndA = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), nnz);
        cscColPtrA = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), (n + 1));

        CUSPARSE_CALL(cusparseCreate(&cusparseHandle));
        SyncGuard syncGuard;
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const float*>(a.BufferPointer()), a.RowLocation(), a.ColLocation(), reinterpret_cast<float*>(cscValA), cscRowIndA, cscColPtrA, cpVals, idxBase));
        }
        else
        {
            CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const double*>(a.BufferPointer()), a.RowLocation(), a.ColLocation(), reinterpret_cast<double*>(cscValA), cscRowIndA, cscColPtrA, cpVals, idxBase));
        }
    }
    else if (a.m_format == matrixFormatSparseCSC)
    {
        cscValA = (ElemType*) a.BufferPointer();
        cscRowIndA = a.RowLocation();
        cscColPtrA = a.ColLocation();
    }
    else
    {
        NOT_IMPLEMENTED;
    }
    // Given sparse matrix in column major format, calculate indices for corresponding sparse vector
    GPUSPARSE_INDEX_TYPE* vectArray = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), a.m_nz);
    CUDA_LONG M = n;
    CUDA_LONG N = m;
    // GPUSPARSE_INDEX_TYPE* h_vectArray= new int[a.m_nz];
    int blocksPerGrid = (int) ceil(1.0 * M / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _getSparseVectorRepresntationForCSCMatrix<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(cscColPtrA, cscRowIndA, vectArray, M, N);
    TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), cscRowIndA);
    TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), cscColPtrA);
    // CUDA_CALL(cudaMemcpy(h_vectArray,vectArray,sizeof(GPUSPARSE_INDEX_TYPE)*a.m_nz,cudaMemcpyDeviceToHost));

    // Actual dot product
    ElemType res = 0;
    if (sizeof(ElemType) == sizeof(float))
    {
        CUSPARSE_CALL(cusparseSdoti(cusparseHandle, (int) a.m_nz, reinterpret_cast<float*>(cscValA), vectArray,
                                    reinterpret_cast<float*>(b.BufferPointer()),
                                    reinterpret_cast<float*>(&res), idxBase));
    }
    else
    {
        CUSPARSE_CALL(cusparseDdoti(cusparseHandle, (int) a.m_nz, reinterpret_cast<double*>(cscValA), vectArray,
                                    reinterpret_cast<double*>(b.BufferPointer()),
                                    reinterpret_cast<double*>(&res), idxBase));
    }
    TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), vectArray);
    TracingGPUMemoryAllocator::Free<ElemType>(a.GetComputeDeviceId(), cscValA);
    CUSPARSE_CALL(cusparseDestroy(cusparseHandle));
    return res;
}

template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::InnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b)
{
    return GPUSparseMatrix<ElemType>::InnerProductOfMatrices(b, a);
}

// This is an utility function useful for debugging issues with sparse matrices.
// It just checks that the CSC format indices are not corrupted / pointing to invalid memory.
template <class ElemType>
bool GPUSparseMatrix<ElemType>::IsValid() const
{
    if (m_format != MatrixFormat::matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    long* res = new long[4];
    res[0] = 1;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    long* d_res = TracingGPUMemoryAllocator::Allocate<long>(m_computeDevice, 4);
    CUDA_CALL(cudaMemcpy(d_res, res, sizeof(long) * 4, cudaMemcpyHostToDevice));

    SyncGuard syncGuard;
    int blocksPerGrid = (int) ceil((1.0 * SecondaryIndexCount()) / GridDim::maxThreadsPerBlock);
    _isValid<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(MajorIndexLocation(), SecondaryIndexLocation(), GetNumRows(), GetNumCols(), GetNumNZElements(), d_res);

    CUDA_CALL(cudaMemcpy(res, d_res, sizeof(long) * 4, cudaMemcpyDeviceToHost));

    if (res[0] == 1)
    {
        return true;
    }
    else
    {
        fprintf(stderr, "GPUSparseMatrix::IsValid returned false (additional info: %ld %ld %ld %ld)\n", res[0], res[1], res[2], res[3]);
        return false;
    }
}

template <class ElemType>
/*static*/ bool GPUSparseMatrix<ElemType>::AreEqual(const GPUSparseMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b,
                                                    const ElemType threshold)
{
    if (a.GetNumNZElements() != b.GetNumNZElements() || a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
        return false;

    if (a.m_format != b.m_format)
        NOT_IMPLEMENTED;

    long* res = new long[3];
    res[0] = 1;
    res[1] = 1;
    res[2] = 1;
    long* d_res = TracingGPUMemoryAllocator::Allocate<long>(a.GetComputeDeviceId(), 3);
    CUDA_CALL(cudaMemcpy(d_res, res, sizeof(long) * 3, cudaMemcpyHostToDevice));

    int blocksPerGrid = (int) ceil(1.0 * a.GetNumNZElements() / GridDim::maxThreadsPerBlock);
    _areEqual<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(a.NzValues(), b.NzValues(), (CUDA_LONG) a.GetNumNZElements(), threshold, d_res);
    _areEqual<int><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(a.ColLocation(), b.ColLocation(), (CUDA_LONG) a.GetNumNZElements(), (int) threshold, d_res + 1);
    blocksPerGrid = (int) ceil((1.0 * a.GetNumRows() + 1.0) / GridDim::maxThreadsPerBlock);
    _areEqual<int><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(a.RowLocation(), b.RowLocation(), (CUDA_LONG) a.GetNumRows() + 1, (int) threshold, d_res + 2);

    CUDA_CALL(cudaMemcpy(res, d_res, sizeof(long) * 3, cudaMemcpyDeviceToHost));
    if (res[0] * res[1] * res[2] == 1)
        return true;
    else
        return false;
}

template <class ElemType>
/*static*/ bool GPUSparseMatrix<ElemType>::AreEqual(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b,
                                                    const ElemType threshold)
{
    if (a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
        return false;
    GPUSparseMatrix<ElemType> c(b.GetComputeDeviceId(), b.GetFormat());
    c.SetValue(a);
    return AreEqual(c, b, threshold);
}

template <class ElemType>
/*static*/ bool GPUSparseMatrix<ElemType>::AreEqual(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b,
                                                    const ElemType threshold)
{
    if (a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
        return false;
    GPUSparseMatrix<ElemType> c(a.GetComputeDeviceId(), a.GetFormat());
    c.SetValue(b);
    return AreEqual(a, c, threshold);
}

template <class ElemType>
bool GPUSparseMatrix<ElemType>::IsEqualTo(const GPUSparseMatrix<ElemType>& a, const ElemType threshold) const
{
    return AreEqual(*this, a, threshold);
}

template <class ElemType>
bool GPUSparseMatrix<ElemType>::IsEqualTo(const GPUMatrix<ElemType>& a, const ElemType threshold) const
{
    return AreEqual(*this, a, threshold);
}

#pragma endregion Static BLAS Functions

#pragma region Member BLAS Functions

template <class ElemType>
DEVICEID_TYPE GPUSparseMatrix<ElemType>::GetComputeDeviceId() const
{
    // for externally managed memory the CUDA context will have the current device
    if (!OwnBuffer())
    {
        DEVICEID_TYPE devId;
        CUDA_CALL(cudaGetDevice(&devId));
        return devId;
    }
    else
        return m_computeDevice;
}

// sparse x dense = dense
template <class ElemType>
GPUMatrix<ElemType> GPUSparseMatrix<ElemType>::ElementProductOf(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (!b.OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (a.m_format != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    if (a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
        LogicError("ElementProductOf: matrix dimensions mismatch");

    b.PrepareDevice();
    GPUMatrix<ElemType> c(b.GetNumRows(), b.GetNumCols(), b.GetComputeDeviceId());

    SyncGuard syncGuard;
    CUDA_LONG M = (CUDA_LONG) a.GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * M / GridDim::maxThreadsPerBlock);
    _sparseCSRElemMulDense<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(a.BufferPointer(), a.RowLocation(), a.ColLocation(), b.BufferPointer(), c.BufferPointer(), M);
    return c;
}

// sparse x dense = dense
template <class ElemType>
GPUMatrix<ElemType> GPUSparseMatrix<ElemType>::ElementProductOf(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b)
{
    return GPUSparseMatrix<ElemType>::ElementProductOf(b, a);
}

template <class ElemType>
GPUSparseMatrix<ElemType> GPUSparseMatrix<ElemType>::operator+(const GPUSparseMatrix<ElemType>& a) const
{
    GPUSparseMatrix<ElemType> res(GetComputeDeviceId(), GetFormat());
    GPUSparseMatrix<ElemType>::ScaleAndAdd(1, *this, 1, a, res);
    return res;
}

template <class ElemType>
GPUSparseMatrix<ElemType> GPUSparseMatrix<ElemType>::operator-(const GPUSparseMatrix<ElemType>& a) const
{
    GPUSparseMatrix<ElemType> res(GetComputeDeviceId(), GetFormat());
    GPUSparseMatrix<ElemType>::ScaleAndAdd(1, *this, -1, a, res);
    return res;
}

// TODO: This is an unusual use of this operator. Remove this.
template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::operator^=(ElemType alpha)
{
    GPUSparseMatrix<ElemType>& us = *this;
    ElementWisePower(alpha, us, us);
    return us;
}

// TODO: This is an unusual use of this operator. Remove this.
template <class ElemType>
GPUSparseMatrix<ElemType> GPUSparseMatrix<ElemType>::operator^(ElemType alpha) const
{
    GPUSparseMatrix<ElemType> c(GetComputeDeviceId(), GetFormat());
    ElementWisePower(alpha, *this, c);
    return c;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::operator*=(ElemType alpha)
{
    GPUSparseMatrix<ElemType>& us = *this;
    if (alpha != 1)
        Scale(alpha, us);
    return us;
}

template <class ElemType>
GPUSparseMatrix<ElemType> GPUSparseMatrix<ElemType>::operator*(ElemType alpha) const
{
    GPUSparseMatrix<ElemType> c(*this);
    if (alpha != 1)
        Scale(alpha, c);
    return c;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignElementPowerOf(const GPUSparseMatrix<ElemType>& a, const ElemType power)
{
    ElementWisePower(power, a, *this);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType> GPUSparseMatrix<ElemType>::Transpose() const
{
    int m = (int) GetNumRows();
    int n = (int) GetNumCols();
    int nnz = (int) GetNumNZElements();
    cusparseAction_t cpVals = CUSPARSE_ACTION_NUMERIC;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;

    assert(GetFormat() & matrixFormatCompressed); // for now this only supports compressed formats
    PrepareDevice();
    GPUSparseMatrix c(GetComputeDeviceId(), GetFormat());
    c.Resize(n, m, nnz, GetFormat(), true, false);
    c.m_nz = nnz;

    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));

    SyncGuard syncGuard;
    if (m_format == MatrixFormat::matrixFormatSparseCSR)
    {
        if (nnz > 0)
        {
            if (sizeof(ElemType) == sizeof(float))
            {
                CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const float*>(this->BufferPointer()), this->RowLocation(), this->ColLocation(),
                                               reinterpret_cast<float*>(c.BufferPointer()), c.ColLocation(), c.RowLocation(), cpVals, idxBase));
            }
            else
            {
                CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const double*>(this->BufferPointer()), this->RowLocation(), this->ColLocation(),
                                               reinterpret_cast<double*>(c.BufferPointer()), c.ColLocation(), c.RowLocation(), cpVals, idxBase));
            }
        }
        else
        {
            CUDA_CALL(cudaMemset(c.BufferPointer(), 0, c.BufferSizeAllocated()));
        }
    }
    else if (m_format == matrixFormatSparseCSC)
    {
        if (nnz > 0)
        {
            if (sizeof(ElemType) == sizeof(float))
            {
                CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, n, m, nnz, reinterpret_cast<const float*>(this->BufferPointer()), this->ColLocation(), this->RowLocation(),
                                               reinterpret_cast<float*>(c.BufferPointer()), c.RowLocation(), c.ColLocation(), cpVals, idxBase));
            }
            else
            {
                CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, n, m, nnz, reinterpret_cast<const double*>(this->BufferPointer()), this->ColLocation(), this->RowLocation(),
                                               reinterpret_cast<double*>(c.BufferPointer()), c.RowLocation(), c.ColLocation(), cpVals, idxBase));
            }
        }
        else
        {
            CUDA_CALL(cudaMemset(c.BufferPointer(), 0, c.BufferSizeAllocated()));
        }
    }
    else
    {
        NOT_IMPLEMENTED;
    }
    CUSPARSE_CALL(cusparseDestroy(cusparseHandle));
    return c;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignTransposeOf(const GPUSparseMatrix<ElemType>& a)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (this == &a)
        LogicError("AssignTransposeOf: a is the same as [this]. Does not support inplace transpose.");

    if (a.IsEmpty())
        LogicError("AssignTransposeOf: Matrix a is empty.");

    *this = a.Transpose();
    return *this;
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::InplaceTranspose()
{
    if (IsEmpty())
        return;
    // transfer converted block over to this pointer
    *this = std::move(Transpose());
}

template <class ElemType>
GPUSparseMatrix<ElemType> GPUSparseMatrix<ElemType>::ColumnSlice(size_t startColumn, size_t numCols) const
{
    if (startColumn + numCols > m_numCols)
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) m_numCols);

    if (m_format != MatrixFormat::matrixFormatSparseCSC && (startColumn != 0 || numCols != m_numCols))
        NOT_IMPLEMENTED;

    GPUSparseMatrix<ElemType> slice(m_computeDevice);
    slice.m_computeDevice            = m_computeDevice;
    slice.m_numRows                  = m_numRows;
    slice.m_numCols                  = numCols;
    slice.m_nz                       = (numCols == m_numCols) ? m_nz : SecondaryIndexValueAt(startColumn + numCols) - SecondaryIndexValueAt(startColumn);
    slice.m_elemSizeAllocated        = m_elemSizeAllocated;
    slice.m_totalBufferSizeAllocated = m_totalBufferSizeAllocated;
    slice.m_pArray                   = m_pArray;
    slice.m_format                   = m_format;
    slice.m_externalBuffer           = true;
    slice.m_sliceOf                  = const_cast<GPUSparseMatrix<ElemType>*>(this); // BUGBUG: ColumnSlice() returns a reference to a mutable matrix, even if itself is 'const'; should not be.
    slice.m_blockSize                = m_blockSize;
    slice.m_rowToId                  = m_rowToId;
    slice.m_tempHostBuffer           = m_tempHostBuffer;
    slice.m_tempHostBufferSize       = m_tempHostBufferSize;
    slice.m_sliceViewOffset          = startColumn; // Just shift the compressed index location to the new startColumn - that's it!

    return slice;
}

template <class ElemType>
GPUMatrix<ElemType> GPUSparseMatrix<ElemType>::CopyColumnSliceToDense(size_t startColumn, size_t numCols) const
{
    int m = (int) GetNumRows();
    int n = (int) GetNumCols();

    // if (numCols == 0)
    //    LogicError("The slice cannot have 0 columns.");

    if (startColumn + numCols > n)
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) n);

    if (m_format != MatrixFormat::matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    GPUMatrix<ElemType> slice(m, numCols, GetComputeDeviceId());

    PrepareDevice();
    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    SyncGuard syncGuard;
    CUSPARSE_CALL(cusparseSetStream(cusparseHandle, t_stream));
    if (sizeof(ElemType) == sizeof(float))
    {
        CUSPARSE_CALL(cusparseScsc2dense(cusparseHandle, m, numCols, descr, (float*) BufferPointer(), RowLocation(), ColLocation() + startColumn, (float*) slice.BufferPointer(), m));
    }
    else
    {
        CUSPARSE_CALL(cusparseDcsc2dense(cusparseHandle, m, numCols, descr, (double*) BufferPointer(), RowLocation(), ColLocation() + startColumn, (double*) slice.BufferPointer(), m));
    }

    CUSPARSE_CALL(cusparseDestroy(cusparseHandle));

    return slice;
}

template <class ElemType>
GPUMatrix<ElemType> GPUSparseMatrix<ElemType>::DiagonalToDense() const
{
    int m = (int) GetNumRows();
    int n = (int) GetNumCols();

    if (m != n)
        LogicError("Diagonal can be called only for square matrix. (rows=%d, cols=%d)", m, n);

    if (m_format != MatrixFormat::matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    GPUMatrix<ElemType> tmp(m, n, GetComputeDeviceId());

    // TODO: Implement optimized diagonal functions for sparse matrices. For now copy to dense first.
    CopyToDenseMatrix(tmp);

    return tmp.Diagonal();
}

template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::SumOfAbsElements() const
{
    if (IsEmpty())
        return 0;

    cublasHandle_t cuHandle = GPUMatrix<ElemType>::GetCublasHandle(GetComputeDeviceId());
    if (sizeof(ElemType) == sizeof(float))
    {
        float res = 0;
        cublasSasum(cuHandle, (int) GetNumNZElements(), reinterpret_cast<const float*>(NzValues()), 1, &res);
        return res;
    }
    else
    {
        double res = 0;
        cublasDasum(cuHandle, (int) GetNumNZElements(), reinterpret_cast<const double*>(NzValues()), 1, &res);
        return ElemType(res);
    }
}

template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::SumOfElements() const
{
    if (IsEmpty())
        LogicError("SumOfElements: Matrix is empty");

    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(m_computeDevice, 1);
    ElemType h_sum;
    // WARNING: THIS kernel is not the most efficient way!
    _reductionSum<ElemType><<<1, 1024>>>(NzValues(), d_sum, (LONG64) GetNumNZElements());
    CUDA_CALL(cudaMemcpy(&h_sum, d_sum, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(m_computeDevice, d_sum);

    return h_sum;
}

// sqrt(sum all elements^2)
template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::FrobeniusNorm() const
{
    if (IsEmpty())
        return 0;

    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(m_computeDevice, 1);
    ElemType h_sum = 0;
    // WARNING: THIS kernel is not the most efficient way!
    _reductionSum2<ElemType><<<1, 1024>>>(NzValues(), d_sum, (int) GetNumNZElements());
    CUDA_CALL(cudaMemcpy(&h_sum, d_sum, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(m_computeDevice, d_sum);

    if (sizeof(ElemType) == sizeof(float))
        return (ElemType) sqrtf((float) h_sum);
    else
        return (ElemType) sqrt((double) h_sum);
}

template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::MatrixNormInf() const
{
    if (IsEmpty())
        return 0;

    ElemType* d_maxAbs = TracingGPUMemoryAllocator::Allocate<ElemType>(m_computeDevice, 1);
    ElemType h_maxAbs = 0;
    // WARNING: THIS kernel is not the most efficient way!
    _reductionMatrixNormInf<ElemType><<<1, 1024>>>(NzValues(), d_maxAbs, (int) GetNumNZElements());
    CUDA_CALL(cudaMemcpy(&h_maxAbs, d_maxAbs, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(m_computeDevice, d_maxAbs);

    if (sizeof(ElemType) == sizeof(float))
        return h_maxAbs;
    else
        return h_maxAbs;
}

template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::MatrixNorm1() const
{
    return SumOfAbsElements();
}

#pragma endregion Member BLAS Functions

#pragma region Other Functions

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::ElementInverse()
{
#if 1
    // Note: This makes no sense because sparse matrices are defined by having lots of zeroes.
    NOT_IMPLEMENTED;
#else
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (IsEmpty())
        LogicError("ElementInverse: Matrix is empty.");

    CUDA_LONG N = (CUDA_LONG) GetNumNZElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _elemInverse<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(NzValues(), N);
    return *this;
#endif
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignElementInverseOf(const GPUSparseMatrix<ElemType>& a)
{
#if 1
    // Note: This makes no sense because sparse matrices are defined by having lots of zeroes.
    UNUSED(a); NOT_IMPLEMENTED;
#else
    SetValue(a);
    return ElementInverse();
#endif
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceSigmoid()
{
#if 1
    // Note: This makes no sense because sigmoid(0) != 0.
    NOT_IMPLEMENTED;
#else
    performElementWiseFunction(ElementWiseOperator::opSigmoid, *this);
    return *this;
#endif
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignSigmoidOf(const GPUSparseMatrix<ElemType>& a)
{
#if 1
    // Note: This makes no sense because sigmoid(0) != 0.
    UNUSED(a); NOT_IMPLEMENTED;
#else
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());
    performElementWiseFunction(ElementWiseOperator::opSigmoid, a);
    return *this;
#endif
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceLinearRectifierDerivative()
{
    performElementWiseFunction(ElementWiseOperator::opLinearRectifierDerivative, *this);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignLinearRectifierDerivativeOf(const GPUSparseMatrix<ElemType>& a)
{
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());
    performElementWiseFunction(ElementWiseOperator::opLinearRectifierDerivative, a);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceTanh()
{
    performElementWiseFunction(ElementWiseOperator::opTanh, *this);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignTanhOf(const GPUSparseMatrix<ElemType>& a)
{
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());
    performElementWiseFunction(ElementWiseOperator::opTanh, a);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceSqrt()
{
    performElementWiseFunction(ElementWiseOperator::opSqrt, *this);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignSqrtOf(const GPUSparseMatrix<ElemType>& a)
{
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());
    performElementWiseFunction(ElementWiseOperator::opSqrt, a);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceExp()
{
#if 1
    // Note: This makes no sense because exp(0) != 0.
    NOT_IMPLEMENTED;
#else
    performElementWiseFunction(ElementWiseOperator::opExp, *this);
    return *this;
#endif
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignExpOf(const GPUSparseMatrix<ElemType>& a)
{
#if 1
    // Note: This makes no sense because exp(0) != 0.
    UNUSED(a); NOT_IMPLEMENTED;
#else
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());
    performElementWiseFunction(ElementWiseOperator::opExp, a);
    return *this;
#endif
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceLog()
{
#if 1
    // Note: This makes no sense because log(0) != 0.
    NOT_IMPLEMENTED;
#else
    performElementWiseFunction(ElementWiseOperator::opLog, *this);
    return *this;
#endif
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignLogOf(const GPUSparseMatrix<ElemType>& a)
{
#if 1
    // Note: This makes no sense because log(0) != 0.
    UNUSED(a); NOT_IMPLEMENTED;
#else
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());
    performElementWiseFunction(ElementWiseOperator::opLog, a);
    return *this;
#endif
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceAbs()
{
    performElementWiseFunction(ElementWiseOperator::opAbs, *this);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignAbsOf(const GPUSparseMatrix<ElemType>& a)
{
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());
    performElementWiseFunction(ElementWiseOperator::opAbs, a);
    return *this;
}

// TODO: Check whether these functions always map 0 to 0.
template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceTruncateBottom(const ElemType threshold)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (IsEmpty())
        LogicError("InplaceTruncateBottom: Matrix is empty.");
    CUDA_LONG N = (CUDA_LONG) GetNumNZElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _assignTruncateBottom<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(NzValues(), NzValues(), threshold, N);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignTruncateBottomOf(const GPUSparseMatrix<ElemType>& a, const ElemType threshold)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (a.IsEmpty())
        LogicError("AssignTruncateBottomOf: Matrix a is empty.");

    if (this != &a)
    {
        // Resize(a.GetNumRows(), a.GetNumCols());
        ResizeAsAndCopyIndexFrom(a);
    }
    CUDA_LONG N = (CUDA_LONG) GetNumNZElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _assignTruncateBottom<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(NzValues(), a.NzValues(), threshold, N);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceTruncateTop(const ElemType threshold)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (IsEmpty())
        LogicError("InplaceTruncateTop: Matrix is empty.");
    CUDA_LONG N = (CUDA_LONG) GetNumNZElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _assignTruncateTop<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(NzValues(), NzValues(), threshold, N);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignTruncateTopOf(const GPUSparseMatrix<ElemType>& a, const ElemType threshold)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (a.IsEmpty())
        LogicError("AssignTruncateTopOf: Matrix a is empty.");

    if (this != &a)
    {
        ResizeAsAndCopyIndexFrom(a);
    }

    CUDA_LONG N = (CUDA_LONG) GetNumNZElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _assignTruncateTop<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(NzValues(), a.NzValues(), threshold, N);
    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::SetToZeroIfAbsLessThan(const ElemType threshold)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (IsEmpty())
        LogicError("SetToZeroIfAbsLessThan: Matrix is empty.");
    CUDA_LONG N = (CUDA_LONG) GetNumNZElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _setToZeroIfAbsLessThan<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(NzValues(), threshold, N);
    return *this;
}

#pragma endregion

#pragma region Helper Functions

//outBuffer should be allocated to be >= size by the caller
template <class ElemType>
template <class OutType, class InType>
/*private*/ void GPUSparseMatrix<ElemType>::CopyBuffer(OutType* outBuffer, const InType* inBuffer, const size_t size)
{
#pragma omp parallel for
    for (size_t i = 0; i < (size & ~3); i += 4)
    {
        outBuffer[i] = inBuffer[i];
        outBuffer[i + 1] = inBuffer[i + 1];
        outBuffer[i + 2] = inBuffer[i + 2];
        outBuffer[i + 3] = inBuffer[i + 3];
    }
    // handle remaining stuffs
    for (size_t i = size & ~3; i < size; i++)
    {
        outBuffer[i] = inBuffer[i];
    }
}

template <class ElemType>
void* GPUSparseMatrix<ElemType>::ReserveTempHostBuffer(const size_t sizeInByte) const
{
    if (m_tempHostBufferSize < sizeInByte)
    {
        delete[](byte*) m_tempHostBuffer;
        m_tempHostBuffer = new byte[sizeInByte];
        m_tempHostBufferSize = sizeInByte;
    }
    return (void*) m_tempHostBuffer;
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::performElementWiseFunction(ElementWiseOperator kind, const GPUSparseMatrix<ElemType>& src)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    CUDA_LONG N = (CUDA_LONG) GetNumNZElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    switch (kind)
    {
    case ElementWiseOperator::opSigmoid:
        return _elementWiseSigmoidOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(src.NzValues(), NzValues(), N);
    case ElementWiseOperator::opTanh:
        return _elementWiseTanhOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(src.NzValues(), NzValues(), N);
    case ElementWiseOperator::opSqrt:
        return _elementWiseSqrtOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(src.NzValues(), NzValues(), N);
    case ElementWiseOperator::opExp:
        return _elementWiseExpOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(src.NzValues(), NzValues(), N);
    case ElementWiseOperator::opLog:
        return _elementWiseLogOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(src.NzValues(), NzValues(), N);
    case ElementWiseOperator::opAbs:
        return _elementWiseAbsOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(src.NzValues(), NzValues(), N);
    case ElementWiseOperator::opLinearRectifierDerivative:
        return _elementWiseLinRectDerivativeOnCuda<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(src.NzValues(), NzValues(), N);
    default:
        NOT_IMPLEMENTED;
    }
}

#pragma endregion Helper Functions

template class MATH_API GPUSparseMatrix<float>;
template class MATH_API GPUSparseMatrix<double>;

// We use Matrix<char> as the backing store for QuantizedMatrix
// Let's explicitly instantiate the methods we need for that purpose
template GPUSparseMatrix<char>::GPUSparseMatrix(DEVICEID_TYPE computeDevice, const MatrixFormat matrixFormat);
template GPUSparseMatrix<char>::GPUSparseMatrix(const size_t numRows, const size_t numCols, const size_t numNZ, DEVICEID_TYPE computeDevice, const MatrixFormat matrixFormat);
template GPUSparseMatrix<char>::GPUSparseMatrix(GPUSparseMatrix<char> const&);
template GPUSparseMatrix<char>::GPUSparseMatrix(GPUSparseMatrix<char>&&);
template void GPUSparseMatrix<char>::SetValue(CPUSparseMatrix<char> const&);
template void GPUSparseMatrix<char>::SetValue(GPUSparseMatrix<char> const&);
template void GPUSparseMatrix<char>::SetValue(GPUMatrix<char> const&);
template void GPUSparseMatrix<char>::CopyToDenseMatrix(GPUMatrix<char>&) const;
template void GPUSparseMatrix<char>::CopyToCPUSparseMatrix(CPUSparseMatrix<char>&) const;
template void GPUSparseMatrix<char>::ChangeDeviceTo(int);
template void GPUSparseMatrix<char>::Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const bool growOnly, bool keepExistingValues);
template void GPUSparseMatrix<char>::Reset();
template GPUSparseMatrix<char>::~GPUSparseMatrix();
template GPUSparseMatrix<char> GPUSparseMatrix<char>::ColumnSlice(size_t startColumn, size_t numCols) const;
template GPUMatrix<char> GPUSparseMatrix<char>::CopyColumnSliceToDense(size_t startColumn, size_t numCols) const;
template GPUSparseMatrix<char>& GPUSparseMatrix<char>::operator=(GPUSparseMatrix<char>&& deepCopy);

template GPUSparseMatrix<int>::GPUSparseMatrix(DEVICEID_TYPE, const MatrixFormat);
template GPUSparseMatrix<int>::~GPUSparseMatrix();
template void GPUSparseMatrix<int>::Resize(const size_t, const size_t, const size_t, const bool, bool);

template <class ElemType>
MATH_API File& operator>>(File& stream, GPUSparseMatrix<ElemType>& us)
{
    if (!us.OwnBuffer())
        LogicError("Cannot read into a managed external matrix");

    stream.GetMarker(fileMarkerBeginSection, std::wstring(L"BMAT"));
    size_t elsize;
    stream >> elsize;
    if (sizeof(ElemType) != elsize)
        RuntimeError("Template argument size doesn't match those in file");
    std::wstring matrixName;

    // now prepare this header to receive the data being read
    size_t nz, colnum, rownum;
    int format;

    // read in the header information
    stream >> matrixName >> format >> nz >> colnum >> rownum;

    us.m_format = (MatrixFormat) format;
    if (us.m_format != matrixFormatSparseCSC && us.m_format != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    us.Resize(rownum, colnum, nz, true, false);
    us.SetNzCount(nz);

    if (nz > 0)
    {
        size_t compressedSize = (us.m_format == matrixFormatSparseCSC) ? colnum + 1 : rownum + 1;
        ElemType* dataBuffer = new ElemType[nz];
        CPUSPARSE_INDEX_TYPE* unCompressedIndex = new CPUSPARSE_INDEX_TYPE[nz];
        CPUSPARSE_INDEX_TYPE* compressedIndex = new CPUSPARSE_INDEX_TYPE[compressedSize];

        // read in the sparse matrix info
        for (size_t i = 0; i < nz; ++i)
        {
            stream >> dataBuffer[i];
        }
        for (size_t i = 0; i < nz; ++i)
        {
            size_t val;
            stream >> val;
            unCompressedIndex[i] = val;
        }
        for (size_t i = 0; i < compressedSize; ++i)
        {
            size_t val;
            stream >> val;
            compressedIndex[i] = val;
        }

        if (us.m_format == matrixFormatSparseCSC)
            us.SetMatrixFromCSCFormat(compressedIndex, unCompressedIndex, dataBuffer, nz, rownum, colnum);
        else if (us.m_format == matrixFormatSparseCSR)
            us.SetMatrixFromCSRFormat(compressedIndex, unCompressedIndex, dataBuffer, nz, rownum, colnum);

        delete[] dataBuffer;
        delete[] unCompressedIndex;
        delete[] compressedIndex;
    }

    stream.GetMarker(fileMarkerEndSection, std::wstring(L"EMAT"));

    return stream;
}

template MATH_API File& operator>>(File& stream, GPUSparseMatrix<float>& us);
template MATH_API File& operator>>(File& stream, GPUSparseMatrix<double>& us);

template <class ElemType>
MATH_API File& operator<<(File& stream, const GPUSparseMatrix<ElemType>& us)
{
    if (us.m_format != matrixFormatSparseCSC && us.m_format != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    stream.PutMarker(fileMarkerBeginSection, std::wstring(L"BMAT"));
    stream << sizeof(ElemType);
	std::wstring s(L"nnmatrix");
	stream << s;

    size_t nz = us.GetNumNZElements(), numElemAllocated = us.GetNumElemAllocated(), numRows = us.GetNumRows(), numCols = us.GetNumCols();
    size_t compressedSize = us.SecondaryIndexCount();
    int format = us.GetFormat();

    stream << format << nz << numCols << numRows;

    if (nz > 0)
    {
        ElemType* dataBuffer = nullptr;
        CPUSPARSE_INDEX_TYPE* compressedIndex = nullptr;
        CPUSPARSE_INDEX_TYPE* unCompressedIndex = nullptr;

        if (us.m_format == matrixFormatSparseCSC)
            us.GetMatrixFromCSCFormat(compressedIndex, unCompressedIndex, dataBuffer, numElemAllocated, nz, numRows, numCols);
        else if (us.m_format == matrixFormatSparseCSR)
            us.GetMatrixFromCSRFormat(compressedIndex, unCompressedIndex, dataBuffer, numElemAllocated, nz, numRows, numCols);
        else
            NOT_IMPLEMENTED;

        for (size_t i = 0; i < nz; ++i)
        {
            stream << dataBuffer[i];
        }
        for (size_t i = 0; i < nz; ++i)
        {
            size_t val = unCompressedIndex[i];
            stream << val;
        }
        for (size_t i = 0; i < compressedSize; ++i)
        {
            size_t val = compressedIndex[i];
            stream << val;
        }

        delete[] dataBuffer;
        delete[] unCompressedIndex;
        delete[] compressedIndex;
    }

    stream.PutMarker(fileMarkerEndSection, std::wstring(L"EMAT"));

    return stream;
}

template MATH_API File& operator<<(File& stream, const GPUSparseMatrix<float>& us);
template MATH_API File& operator<<(File& stream, const GPUSparseMatrix<double>& us);

}}}

#endif // CPUONLY
