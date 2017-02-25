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
    if (idx + m_sliceViewOffset == 0) return 0;
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
    Base::ZeroInit(matrixFormat, computeDevice);
}

template <class ElemType>
GPUSparseMatrix<ElemType>::GPUSparseMatrix(const size_t numRows, const size_t numCols, const size_t numNZ, DEVICEID_TYPE computeDevice, const MatrixFormat matrixFormat /*= MatrixFormat::matrixFormatSparseCSR*/)
{
    ZeroInit(matrixFormat, computeDevice);
    RequireSizeAndAllocate(numRows, numCols, numNZ, true, false);
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
    DEVICEID_TYPE newId = deviceId >= 0 ? deviceId : GetComputeDeviceId();

    Microsoft::MSR::CNTK::PrepareDevice(newId);
    return newId;
}

template <class ElemType>
/*private*/ void GPUSparseMatrix<ElemType>::DeepCopy(const GPUSparseMatrix<ElemType>& deepCopy)
{
    ChangeDeviceTo(deepCopy.GetComputeDeviceId());
    deepCopy.PrepareDevice();

    RequireSizeAndAllocate(deepCopy.GetNumRows(), deepCopy.GetNumCols(), deepCopy.GetNumNZElements(), deepCopy.GetFormat(), true, false);
    m_sliceViewOffset = 0; // reset to zero as we only start copying the indices starting from the offset in the source matrix

    CUDA_CALL(cudaMemcpy(Data(), deepCopy.NzValues(), deepCopy.NzSize(), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(MajorIndexLocation(), deepCopy.MajorIndexLocationWithSliceViewOffset(), deepCopy.MajorIndexSize(), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(SecondaryIndexLocation(), deepCopy.SecondaryIndexLocation(), deepCopy.SecondaryIndexSize(), cudaMemcpyDeviceToDevice));

    if (deepCopy.m_sliceViewOffset > 0)
    {
        int blocksPerGrid = (int) ceil(1.0 * SecondaryIndexCount() / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        _shiftColCSCIndexFromSliceViewToAbsolute<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            SecondaryIndexLocation(),
            SecondaryIndexCount(),
            GetNumNZElements());
    }

    // TODO: to copy other variables used only for class based LM
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::SetValue(const GPUSparseMatrix<ElemType>& deepCopy)
{
    VerifyWritable(__FUNCTION__);

    DeepCopy(deepCopy);
}

// from CPU
template <class ElemType>
void GPUSparseMatrix<ElemType>::SetValue(const CPUSparseMatrix<ElemType>& deepCopy)
{
    VerifyWritable(__FUNCTION__);

    SetFormat(deepCopy.GetFormat());
    if (deepCopy.IsEmpty())
    {
        Reset();
        return;
    }

    if (deepCopy.GetFormat() == matrixFormatSparseCSR)
    {
        SetMatrixFromCSRFormat(deepCopy.RowLocation(), deepCopy.ColLocation(), deepCopy.Data(), deepCopy.GetNumElemAllocated(), deepCopy.GetNumRows(), deepCopy.GetNumCols());
    }
    else if (deepCopy.GetFormat() == matrixFormatSparseCSC)
    {
        SetMatrixFromCSCFormat(deepCopy.ColLocation(), deepCopy.RowLocation(), deepCopy.Data(), deepCopy.GetNumElemAllocated(), deepCopy.GetNumRows(), deepCopy.GetNumCols());
    }
    else
        NOT_IMPLEMENTED;
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::CopyToCPUSparseMatrix(CPUSparseMatrix<ElemType>& cpuSparseMatrix) const
{
    cpuSparseMatrix.VerifyWritable(__FUNCTION__);

    cpuSparseMatrix.SetFormat(GetFormat());
    if (IsEmpty())
    {
        cpuSparseMatrix.Reset();
        return;
    }

    if (this->GetFormat() == matrixFormatSparseCSR)
    {
        // we need to do conversion because CPUSparseMatrix uses size_t for indexes while GPUSparseMatrix uses int
        cpuSparseMatrix.RequireSizeAndAllocate(GetNumRows(), GetNumCols(), GetNumElemAllocated(), true, false);

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
            ConvertBuffer(cpuSparseMatrix.RowLocation(), h_CSRRow, SecondaryIndexCount());

            GPUSPARSE_INDEX_TYPE* h_Col = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(ColSize());
            CUDA_CALL(cudaMemcpy(h_Col, ColLocation(), ColSize(), cudaMemcpyDeviceToHost));
            ConvertBuffer(cpuSparseMatrix.ColLocation(), h_Col, MajorIndexCount());
        }

        CUDA_CALL(cudaMemcpy(cpuSparseMatrix.Data(), Data(), GetSizeElemAllocated(), cudaMemcpyDeviceToHost));
    }
    else if (this->GetFormat() == matrixFormatSparseCSC)
    {
        // we need to do conversion because CPUSparseMatrix uses size_t for indexes while GPUSparseMatrix uses int
        cpuSparseMatrix.RequireSizeAndAllocate(GetNumRows(), GetNumCols(), GetNumElemAllocated(), true, false);

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
            ConvertBuffer(cpuSparseMatrix.ColLocation(), h_CSCCol, SecondaryIndexCount());

            GPUSPARSE_INDEX_TYPE* h_Row = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(RowSize());
            CUDA_CALL(cudaMemcpy(h_Row, RowLocation(), RowSize(), cudaMemcpyDeviceToHost));
            ConvertBuffer(cpuSparseMatrix.RowLocation(), h_Row, MajorIndexCount());
        }

        CUDA_CALL(cudaMemcpy(cpuSparseMatrix.Data(), Data(), GetSizeElemAllocated(), cudaMemcpyDeviceToHost));
    }
    else if (this->GetFormat() == matrixFormatSparseBlockCol)
    {
        cpuSparseMatrix.RequireSizeAndAllocate(GetNumRows(), GetNumCols(), GetNumNZElements(), true, false);

        PrepareDevice();
        std::vector<GPUSPARSE_INDEX_TYPE> temp(GetBlockSize());
        CUDA_CALL(cudaMemcpy(temp.data(), BlockId2ColOrRow(), GetBlockSize() * sizeof(GPUSPARSE_INDEX_TYPE), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < temp.size(); ++i)
            cpuSparseMatrix.BlockIdsLocation()[i] = temp[i];

        cpuSparseMatrix.SetBlockSize(GetBlockSize());

        CUDA_CALL(cudaMemcpy(cpuSparseMatrix.Data(), Data(), NzSize(), cudaMemcpyDeviceToHost));
    }
    else
        NOT_IMPLEMENTED;
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::CopyToDenseMatrix(GPUMatrix<ElemType>& denseMatrix) const
{
    if (IsEmpty())
    {
        denseMatrix.RequireSize(0, 0);
        return;
    }

    PrepareDevice();
    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    denseMatrix.RequireSize(GetNumRows(), GetNumCols());

    SyncGuard syncGuard;
    CUSPARSE_CALL(cusparseSetStream(cusparseHandle, t_stream));
    if (GetFormat() == MatrixFormat::matrixFormatSparseCSR)
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseScsr2dense(cusparseHandle, int(GetNumRows()), int(GetNumCols()), descr, (float*) Buffer(), RowLocation(), ColLocation(), (float*) denseMatrix.Data(), int(GetNumRows())));
        }
        else
        {
            CUSPARSE_CALL(cusparseDcsr2dense(cusparseHandle, int(GetNumRows()), int(GetNumCols()), descr, (double*) Buffer(), RowLocation(), ColLocation(), (double*) denseMatrix.Data(), int(GetNumRows())));
        }
    }
    else if (GetFormat() == MatrixFormat::matrixFormatSparseCSC)
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseScsc2dense(cusparseHandle, int(GetNumRows()), int(GetNumCols()), descr, (float*) Buffer(), RowLocation(), ColLocation(), (float*) denseMatrix.Data(), int(GetNumRows())));
        }
        else
        {
            CUSPARSE_CALL(cusparseDcsc2dense(cusparseHandle, int(GetNumRows()), int(GetNumCols()), descr, (double*) Buffer(), RowLocation(), ColLocation(), (double*) denseMatrix.Data(), int(GetNumRows())));
        }
    }
    else if (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol || GetFormat() == MatrixFormat::matrixFormatSparseBlockRow)
    {
        denseMatrix.SetValue((ElemType)0);
        ScaleAndAdd(1, *this, denseMatrix);
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
    outMatrix.VerifyWritable(__FUNCTION__);

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
    outMatrix.RequireSizeAndAllocate(GetNumRows(), GetNumCols(), NzCount(), newFormat, true, false);

    if ((oldFormat == matrixFormatSparseCSR && newFormat == matrixFormatSparseCSC) || (oldFormat == matrixFormatSparseCSC && newFormat == matrixFormatSparseCSR))
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, int(GetNumRows()), int(GetNumCols()), int(GetSizeAllocated()),
                                           (float*) Data(), RowLocation(), ColLocation(), (float*) outMatrix.Data(),
                                           outMatrix.RowLocation(), outMatrix.ColLocation(), CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));
        }
        else
        {
            CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, int(GetNumRows()), int(GetNumCols()), int(GetSizeAllocated()),
                                           (double*) Data(), RowLocation(), ColLocation(), (double*) outMatrix.Data(),
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
    VerifyWritable(__FUNCTION__);
    if (to_id == CPUDEVICE)
        LogicError("to_id must be valid GPU");
    if (GetComputeDeviceId()== to_id)
        return;

    if (BufferSizeAllocated() == 0) // nothing to move
    {
        assert(Buffer() == nullptr);
    }
    else
    {
        ElemType* d_dst = reinterpret_cast<ElemType*>(TracingGPUMemoryAllocator::Allocate<char>(to_id, BufferSizeAllocated()));

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
            CUDA_CALL(cudaMemcpyPeer(d_dst, to_id, Buffer(), GetComputeDeviceId(), BufferSizeAllocated()));
        }
        else
        {
            // peer access didn't work, just copy normal
            // make this more efficient by keeping some buffers available for each copy
            ElemType* h_dst = NULL;
            PrepareDevice();
            CUDA_CALL(cudaMallocHost((void**) &h_dst, BufferSizeAllocated()));
            CUDA_CALL(cudaMemcpy(h_dst, Buffer(), BufferSizeAllocated(), cudaMemcpyDeviceToHost));
            PrepareDevice((DEVICEID_TYPE) to_id);
            CUDA_CALL(cudaMemcpy(d_dst, h_dst, BufferSizeAllocated(), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaFreeHost(h_dst));
        }

        TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), Buffer());
        SetBuffer(d_dst, BufferSizeAllocated());
    }

    SetComputeDeviceId(PrepareDevice(to_id));
}

#if 0
template <class ElemType>
void GPUSparseMatrix<ElemType>::SetValue(const CPUMatrix<ElemType>& /*denseMatrix*/)
{
    NOT_IMPLEMENTED;
}
#endif

template <class ElemType>
void GPUSparseMatrix<ElemType>::SetValue(const GPUMatrix<ElemType>& denseMatrix)
{
    VerifyWritable(__FUNCTION__);

    SetValue(denseMatrix, GetFormat());
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::SetValue(const GPUMatrix<ElemType>& denseMatrix, const MatrixFormat matrixFormat)
{
    VerifyWritable(__FUNCTION__);

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

    int* nnzPerRowOrCol = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(GetComputeDeviceId(), ((matrixFormat & matrixFormatRowMajor) ? numRows : numCols));
    int nnzTotalDevHostPtr = -1;

    {
        SyncGuard syncGuard;
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseSnnz(cusparseHandle, (matrixFormat & matrixFormatRowMajor) ? CUSPARSE_DIRECTION_ROW : CUSPARSE_DIRECTION_COLUMN, (int) numRows, (int) numCols, descr,
                                       reinterpret_cast<float*>(denseMatrix.Data()), (int) numRows, nnzPerRowOrCol, &nnzTotalDevHostPtr));
        }
        else
        {
            CUSPARSE_CALL(cusparseDnnz(cusparseHandle, (matrixFormat & matrixFormatRowMajor) ? CUSPARSE_DIRECTION_ROW : CUSPARSE_DIRECTION_COLUMN, (int) numRows, (int) numCols, descr,
                                       reinterpret_cast<double*>(denseMatrix.Data()), (int) numRows, nnzPerRowOrCol, &nnzTotalDevHostPtr));
        }
        // ~SyncGuard
    }

    RequireSizeAndAllocate(numRows, numCols, nnzTotalDevHostPtr, matrixFormat, true, false);

    SyncGuard syncGuard;
    if (GetFormat() == MatrixFormat::matrixFormatSparseCSR)
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseSdense2csr(cusparseHandle, (int) GetNumRows(), (int) GetNumCols(), descr, reinterpret_cast<float*>(denseMatrix.Data()),
                                             (int) GetNumRows(), nnzPerRowOrCol, reinterpret_cast<float*>(Data()), RowLocation(), ColLocation()));
        }
        else
        {
            CUSPARSE_CALL(cusparseDdense2csr(cusparseHandle, (int) GetNumRows(), (int) GetNumCols(), descr, reinterpret_cast<double*>(denseMatrix.Data()),
                                             (int) GetNumRows(), nnzPerRowOrCol, reinterpret_cast<double*>(Data()), RowLocation(), ColLocation()));
        }
    }
    else if (GetFormat() == MatrixFormat::matrixFormatSparseCSC)
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseSdense2csc(cusparseHandle, (int) GetNumRows(), (int) GetNumCols(), descr, reinterpret_cast<float*>(denseMatrix.Data()),
                                             (int) GetNumRows(), nnzPerRowOrCol, reinterpret_cast<float*>(Data()), RowLocation(), ColLocation()));
        }
        else
        {
            CUSPARSE_CALL(cusparseDdense2csc(cusparseHandle, (int) GetNumRows(), (int) GetNumCols(), descr, reinterpret_cast<double*>(denseMatrix.Data()),
                                             (int) GetNumRows(), nnzPerRowOrCol, reinterpret_cast<double*>(Data()), RowLocation(), ColLocation()));
        }
    }
}

template <class ElemType>
GPUSPARSE_INDEX_TYPE* GPUSparseMatrix<ElemType>::GetCondensedVector() const
{
    if (GetFormat() == MatrixFormat::matrixFormatSparseCSC || GetFormat() == MatrixFormat::matrixFormatSparseCSR)
    {
        PrepareDevice();
        GPUSPARSE_INDEX_TYPE* pArray = new GPUSPARSE_INDEX_TYPE[SecondaryIndexCount()];
        CUDA_CALL(cudaMemcpy(pArray, SecondaryIndexLocation(), sizeof(GPUSPARSE_INDEX_TYPE) * SecondaryIndexCount(), cudaMemcpyDeviceToHost));
        return pArray;
    }
    else
    {
        return NULL;
    }
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::MaskColumnsValue(const GPUMatrix<char>& columnsMask, ElemType val)
{
    VerifyWritable(__FUNCTION__);

    size_t n = GetNumCols();
    if (n != columnsMask.GetNumCols())
        RuntimeError("Matrix and column mask must have equal number of columns");

    if (val != 0)
        LogicError("MaskColumnsValue is not implmented for a non-zero mask for sparse matrices.");

#ifdef _DEBUG
    if (GetFormat() == MatrixFormat::matrixFormatSparseCSC)
    {
        // TODO: We could do this on the GPU, but for now C++ is easier.
        // Download the binary columns mask
        char* maskedCols = columnsMask.CopyToArray();

        // If we're CSC, we only need to verify that the columns to be zeroed are empty, since val == 0.
        // So just download the condensed column vector.
        GPUSPARSE_INDEX_TYPE* colVector = GetCondensedVector();

        // Verify that if the column is to be masked, there are no elements in it.
        #pragma omp parallel for
        for (long j = 0; j < n; j++)
            if (maskedCols[j] == 0 && colVector[j + 1] != colVector[j])
                RuntimeError("GPUSparseMatrix attempted to mask column %d, but it has %d elements in it.", (int)j, (int)(colVector[j + 1] - colVector[j]));
    }
    else
        NOT_IMPLEMENTED;
#endif
}


template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::operator=(const GPUSparseMatrix<ElemType>& deepCopy)
{
    if (this != &deepCopy)
        SetValue(deepCopy);

    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>::GPUSparseMatrix(GPUSparseMatrix<ElemType>&& moveFrom)
{
    Base::ShallowCopyFrom(moveFrom);
    moveFrom.ZeroValues(); // so that memory in moveFrom is not freed
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::operator=(GPUSparseMatrix<ElemType>&& moveFrom)
{
    if (this != &moveFrom)
    {
        Base::ShallowCopyFrom(moveFrom);
        moveFrom.ZeroValues();
    }

    return *this;
}

template <class ElemType>
GPUSparseMatrix<ElemType>::~GPUSparseMatrix()
{
    ZeroValues();
}

//ResizeAsAndCopyIndexFrom - Resize this sparse matrix to have the same element structure as the passed matrix
// a - sparse matrix whose structure we want to clone
// remark: this was done for element wise operations where the structure will be identical after an operation
template <class ElemType>
void GPUSparseMatrix<ElemType>::ResizeAsAndCopyIndexFrom(const GPUSparseMatrix<ElemType>& a, const bool growOnly /*= true*/)
{
    RequireSizeAndAllocate(a.GetNumRows(), a.GetNumCols(), a.NzCount(), a.GetFormat(), growOnly, false);

    CUDA_CALL(cudaMemcpy(MajorIndexLocation(), a.MajorIndexLocation(), MajorIndexSize(), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(SecondaryIndexLocation(), a.SecondaryIndexLocation(), SecondaryIndexSize(), cudaMemcpyDeviceToDevice));
}

//-------------------------------------------------------------------------
// main operations
//-------------------------------------------------------------------------

template <class ElemType>
void GPUSparseMatrix<ElemType>::Reshape(const size_t numRows, const size_t numCols)
{
    if (GetNumRows() == numRows && GetNumCols() == numCols)
        return;

    VerifyWritable(__FUNCTION__);

    if (GetFormat() != MatrixFormat::matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    if (GetNumRows() * GetNumCols() != numRows * numCols)
        LogicError("GPUSparseMatrix::Reshape: new matrix size does not match current size, can't be reshaped. Did you mean to resize?");

    size_t bufferSizeNeeded = BufferSizeNeeded(numRows, numCols, GetSizeAllocated(), GetFormat());

    ElemType* pArray = reinterpret_cast<ElemType*>(TracingGPUMemoryAllocator::Allocate<char>(GetComputeDeviceId(), bufferSizeNeeded));

    if (Buffer() != nullptr)
    {
        CUDA_CALL(cudaMemcpy(pArray, Data(), GetSizeElemAllocated(), cudaMemcpyDeviceToDevice));

        GPUSPARSE_INDEX_TYPE* majorIndexInNewBuffer = (GPUSPARSE_INDEX_TYPE*) (pArray + GetSizeAllocated());
        GPUSPARSE_INDEX_TYPE* secondaryIndexInNewBuffer = majorIndexInNewBuffer + MajorIndexCount(numRows, numCols, GetSizeAllocated(), GetFormat());

        int blocksPerGrid = (int) ceil(1.0 * numCols / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        _reshape<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            GetNumRows(),                // old row count
            GetNumCols(),                // old col count
            numRows,                  // new row count
            numCols,                  // new col count
            MajorIndexLocation(),     // old row index array
            SecondaryIndexLocation(), // old column index array
            majorIndexInNewBuffer,    // new row index array
            secondaryIndexInNewBuffer // new column index array
            );
        TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), Buffer());
    }

    SetBuffer(pArray, bufferSizeNeeded);
    SetNumRows(numRows);
    SetNumCols(numCols);
}

// WARNING: When memory is reallocated, existing information will be lost.
// TODO: add keepExistingValues (default to true) argument so that the existing values are kept even after reallocation
template <class ElemType>
void GPUSparseMatrix<ElemType>::Allocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const bool growOnly /*= true*/, bool keepExistingValues /*= true*/)
{
    // BugBug: This doesn't work because allocate is called from Resize sometimes and resize expects allocate to know the old values not the new values, so this won't work.
    if (GetNumRows() != numRows || GetNumCols() != numCols)
        LogicError("Error, calling allocate with dimensions (%d, %d), but the matrix has dimension (%d, %d).", (int)numRows, (int)numCols, (int)GetNumRows(), (int)GetNumCols());

    size_t bufferSizeNeeded = BufferSizeNeeded(numRows, numCols, numNZElemToReserve, GetFormat());
    bool reallocate = (BufferSizeAllocated() < bufferSizeNeeded || (!growOnly && BufferSizeAllocated() > bufferSizeNeeded));

    if (reallocate)
    {
        // Note that we are allocating one buffer for all of our data structures. Thus the ElemType* nzValues array lives directly next to
        // the GPUSPARSE_INDEX_TYPE* rowIndices/colIndices in sparseCSC/CSR formats. Thus we allocate the number of bytes, and then set the
        // start pointer to an ElemType*.
        char* buf = TracingGPUMemoryAllocator::Allocate<char>(GetComputeDeviceId(), bufferSizeNeeded);
        ElemType* pArray = (ElemType*)(buf);

        // Note this is required due to m_nz 
        CUDA_CALL(cudaMemset(pArray, 0, bufferSizeNeeded));
        if (Buffer() != nullptr)
        {
            if (keepExistingValues)
            {
                if (NzCount() > numNZElemToReserve || BufferSizeAllocated() > bufferSizeNeeded)
                    LogicError("Resize: To keep values m_nz should <= numNZElemToReserve.");

                CUDA_CALL(cudaMemcpy(pArray, Data(), GetSizeElemAllocated(), cudaMemcpyDeviceToDevice));

                GPUSPARSE_INDEX_TYPE* majorIndexInNewBuffer = (GPUSPARSE_INDEX_TYPE*) (pArray + numNZElemToReserve);

                CUDA_CALL(cudaMemcpy(majorIndexInNewBuffer, MajorIndexLocation(), MajorIndexSize(), cudaMemcpyDeviceToDevice));

                GPUSPARSE_INDEX_TYPE* secondaryIndexInNewBuffer = majorIndexInNewBuffer + MajorIndexCount(numRows, numCols, numNZElemToReserve, GetFormat());
                CUDA_CALL(cudaMemcpy(secondaryIndexInNewBuffer, SecondaryIndexLocation(), SecondaryIndexSize(), cudaMemcpyDeviceToDevice));
            }
            TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), Buffer());
        }

        SetBuffer(pArray, bufferSizeNeeded);
        SetSizeAllocated(numNZElemToReserve);
    }
    else // if requested size is smaller, keeping original values does not make sense
    {
        SetSizeAllocated(ElemCountFromBufferSize(numRows, numCols, GetFormat(), BufferSizeAllocated()));
        CUDA_CALL(cudaMemset(Buffer(), 0, BufferSizeAllocated()));
    }
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::RequireSizeAndAllocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve /*= 10000*/, const bool growOnly /*= true*/, bool keepExistingValues /*= false*/)
{
    RequireSizeAndAllocate(numRows, numCols, numNZElemToReserve, GetFormat(), growOnly, keepExistingValues);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::RequireSizeAndAllocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly /*= true*/, bool keepExistingValues /*= true*/)
{
    RequireSize(numRows, numCols, matrixFormat, growOnly);
    
    size_t bufferSizeNeeded = BufferSizeNeeded(numRows, numCols, numNZElemToReserve, matrixFormat);
    bool reallocate = (BufferSizeAllocated() < bufferSizeNeeded || (!growOnly && BufferSizeAllocated() > bufferSizeNeeded));

    if (reallocate)
        Allocate(numRows, numCols, numNZElemToReserve, growOnly, keepExistingValues);

}

template <class ElemType>
void GPUSparseMatrix<ElemType>::RequireSize(const size_t numRows, const size_t numCols, const bool growOnly /*= true*/)
{
    RequireSize(numRows, numCols, GetFormat(), growOnly);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::RequireSize(const size_t numRows, const size_t numCols, const MatrixFormat matrixFormat, const bool growOnly /*= true*/)
{
    if (GetFormat() != matrixFormat || GetNumRows() != numRows || GetNumCols() != numCols)
        Resize(numRows, numCols, 0, matrixFormat, growOnly);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve /*= 10000*/, const bool growOnly /*= true*/)
{
    Resize(numRows, numCols, numNZElemToReserve, GetFormat(), growOnly);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly /*= true*/)
{
    VerifyResizable(__FUNCTION__);

    m_sliceViewOffset = 0;
    SetNumRows(numRows);
    SetNumCols(numCols);
    SetNumStorageRows(numRows);
    SetNumStorageCols(numCols);
    SetFormat(matrixFormat);

    // If we really did resize the number of rows/columns, then we changed the number of nz elements allocated. That is, if we used to have a buffer capable of
    // stroring 100 nz elements and 10 columns in CSC format, but we resized to 20 columns, we can no longer store 100 elements, we can only store 95. 
    // Thus we must reset the number of nz elements which can be stored. So let's compute it now.
    size_t newNzElem = ComputeMaxNZElemFromBufferSize(numRows, numCols, BufferSizeAllocated(), matrixFormat);
    SetSizeAllocated(newNzElem);

    size_t bufferSizeNeeded = BufferSizeNeeded(numRows, numCols, numNZElemToReserve, matrixFormat);
    bool reallocate = (BufferSizeAllocated() < bufferSizeNeeded || (!growOnly && BufferSizeAllocated() > bufferSizeNeeded));

    if (reallocate)
        Allocate(numRows, numCols, numNZElemToReserve, growOnly, false);
    else
        ClearNzCount();
}

// Reset matrix to 0.
template <class ElemType>
void GPUSparseMatrix<ElemType>::Reset()
{
    VerifyWritable(__FUNCTION__);

    ClearNzCount();
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::ClearNzCount()
{
    // We are now going to reset m_nz to 0. 
    // To reset m_nz to 0, we must do 2 things.
    //    1. We must clear the secondary column index.
    //    2. Set the block size to 0.
    // These requirements can be deduced by the NzCount method.
    CUDA_CALL(cudaMemset(Buffer(), 0, BufferSizeAllocated()));
    SetBlockSize(0);
}


// copy features to GPU
template <class ElemType>
void GPUSparseMatrix<ElemType>::SetMatrixFromCSRFormat(const GPUSPARSE_INDEX_TYPE* h_CSRRow, const GPUSPARSE_INDEX_TYPE* h_Col, const ElemType* h_Val,
                                                       const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice /*= false*/, const DEVICEID_TYPE devId /*= -1*/)
{
    VerifyWritable(__FUNCTION__);

    if (h_CSRRow == nullptr || h_Col == nullptr || h_Val == nullptr)
        LogicError("SetMatrixFromCSRFormat: nullptr passed in.");

    SetComputeDeviceId(PrepareDevice(devId));

    SetFormat(matrixFormatSparseCSR);
    RequireSizeAndAllocate(numRows, numCols, nz, true, false);

    cudaMemcpyKind kind = IsOnDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    CUDA_CALL(cudaMemcpy(Data(), h_Val, nz * sizeof(ElemType), kind));

    if (sizeof(CPUSPARSE_INDEX_TYPE) == sizeof(GPUSPARSE_INDEX_TYPE))
    {
        // ColSize doesn't work since it requires NzCount() to be usable (RowSize doesn't, since it's the fixed, compressed,
        // dimension. Since NzCount is not available (because the sparse indices which is where the NzCount is copmuted from
        // haven't been copied in yet), we just tell it how many bytes to copy. That is, nz * sizeof(GPUSPARSE_INDEX_TYPE);
        CUDA_CALL(cudaMemcpy(RowLocation(), h_CSRRow, RowSize(), kind));
        CUDA_CALL(cudaMemcpy(ColLocation(), h_Col, nz * sizeof(GPUSPARSE_INDEX_TYPE), kind));
        assert(nz == NzCount());
    }
    else
    {
        GPUSPARSE_INDEX_TYPE* pCol = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(RowSize() + nz);
        ConvertBuffer(pCol, h_Col, MajorIndexCount());

        GPUSPARSE_INDEX_TYPE* pRow = pCol + MajorIndexCount();
        ConvertBuffer(pRow, h_CSRRow, nz);

        CUDA_CALL(cudaMemcpy(RowLocation(), pRow, RowSize(), kind));
        CUDA_CALL(cudaMemcpy(ColLocation(), pCol, nz * sizeof(GPUSPARSE_INDEX_TYPE), kind));
    }
}

// this function will allocate memory while the caller needs to release it
template <class ElemType>
void GPUSparseMatrix<ElemType>::GetMatrixFromCSRFormat(CPUSPARSE_INDEX_TYPE*& h_CSRRow, CPUSPARSE_INDEX_TYPE*& h_Col, ElemType*& h_Val, size_t& numElemAllocated, size_t& nz, size_t& numRows, size_t& numCols) const
{
    VerifyWritable(__FUNCTION__);

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
        h_CSRRow = new CPUSPARSE_INDEX_TYPE[GetNumRows() + 1];
        h_Col = new CPUSPARSE_INDEX_TYPE[nz];

        PrepareDevice();
        CUDA_CALL(cudaMemcpy(h_Val, Data(), GetSizeElemAllocated(), cudaMemcpyDeviceToHost));

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

            ConvertBuffer(h_Col, pCol, MajorIndexCount());
            ConvertBuffer(h_CSRRow, pRow, SecondaryIndexCount());
        }
    }
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::SetMatrixFromCSCFormat(const CPUSPARSE_INDEX_TYPE* h_CSCCol, const CPUSPARSE_INDEX_TYPE* h_Row, const ElemType* h_Val,
    const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice /*= false*/, const DEVICEID_TYPE devId /*= -1*/, DataTransferer* transferer /*= nullptr*/)
{
    VerifyWritable(__FUNCTION__);

    if (h_CSCCol == nullptr || h_Row == nullptr || h_Val == nullptr)
        LogicError("SetMatrixFromCSCFormat: nullptr passed in.");

    SetComputeDeviceId(PrepareDevice(devId));
    SetFormat(matrixFormatSparseCSC);
    RequireSizeAndAllocate(numRows, numCols, nz, true, false);

    if (transferer && IsOnDevice)
        RuntimeError("Currently it is prohibited to copy data asynchronous from device to device.");

    // m_nz doesn't exist anymore. How are we going to deal with the NzSize, RowSize, and ColSize? Do it ourselves of course.

    cudaMemcpyKind kind = IsOnDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    if (transferer)
    {
        // TODO: All RequireSizeAndAllocate should be async and use a transferer.
        // Currently there are some memset operations that can be still executing on the default stream,
        // Here we have to wait for them to finish.
        transferer->RecordComputeStreamSyncPoint();
        transferer->WaitForSyncPointOnAssignStreamAsync();
        transferer->CopyCPUToGPUAsync(h_Val, nz, sizeof(ElemType), Data());
    }
    else
        CUDA_CALL(cudaMemcpy(Data(), h_Val, nz * sizeof(ElemType), kind));

    if (sizeof(CPUSPARSE_INDEX_TYPE) == sizeof(GPUSPARSE_INDEX_TYPE))
    {
        if (transferer)
        {
            transferer->CopyCPUToGPUAsync(h_Row, nz, sizeof(GPUSPARSE_INDEX_TYPE), RowLocation());
            transferer->CopyCPUToGPUAsync(h_CSCCol, numCols + 1, sizeof(GPUSPARSE_INDEX_TYPE), ColLocation());
        }
        else
        {
            CUDA_CALL(cudaMemcpy(RowLocation(), h_Row, sizeof(GPUSPARSE_INDEX_TYPE) * nz, kind));
            CUDA_CALL(cudaMemcpy(ColLocation(), h_CSCCol, sizeof(GPUSPARSE_INDEX_TYPE) * (numCols + 1), kind));
        }
    }
    else
    {
        size_t allocSize = sizeof(GPUSPARSE_INDEX_TYPE) * nz + sizeof(GPUSPARSE_INDEX_TYPE) * (numCols + 1);
        GPUSPARSE_INDEX_TYPE* pCol = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(allocSize);
        GPUSPARSE_INDEX_TYPE* pRow = pCol + nz;

        ConvertBuffer(pCol, h_CSCCol, (numCols+1));
        ConvertBuffer(pRow, h_Row, nz);

        if (transferer)
        {
            transferer->CopyCPUToGPUAsync(pRow, nz, sizeof(GPUSPARSE_INDEX_TYPE), RowLocation());
            transferer->CopyCPUToGPUAsync(pCol, numCols + 1, sizeof(GPUSPARSE_INDEX_TYPE), ColLocation());
        }
        else
        {
            CUDA_CALL(cudaMemcpy(RowLocation(), pRow, sizeof(GPUSPARSE_INDEX_TYPE) * nz, kind));
            CUDA_CALL(cudaMemcpy(ColLocation(), pCol, sizeof(GPUSPARSE_INDEX_TYPE) * (numCols + 1), kind));
        }
    }
}

#if 0 // add it back with test
template <class ElemType>
void GPUSparseMatrix<ElemType>::SetMatrixFromSBCFormat(const size_t* blockIds, const ElemType* val, const size_t numBlocks, const size_t numRows, const size_t numCols)
{
    VerifyWritable(__FUNCTION__);

    if (blockIds == nullptr || val == nullptr)
        LogicError("SetMatrixFromSBCFormat: nullptr passed in.");

    SetFormat(matrixFormatSparseBlockCol);
    SetBlockSize(numBlocks);

    if (numBlocks == 0) return; // ====>

    size_t nz = numBlocks * numRows;
    RequireSizeAndAllocate(numRows, numCols, nz, true, false);

    static std::vector<GPUSPARSE_INDEX_TYPE> gpuBlockId2Col(numBlocks);
    static std::vector<GPUSPARSE_INDEX_TYPE> gpuCol2BlockId(numCols);

    std::fill(gpuBlockId2Col.begin(), gpuBlockId2Col.end(), Id_NotAssigned);
    std::fill(gpuCol2BlockId.begin(), gpuCol2BlockId.end(), Id_NotAssigned);

    #pragma omp parallel for
    for (int i = 0; i < numBlocks; ++i)
    {
        gpuBlockId2Col[i] = (GPUSPARSE_INDEX_TYPE)blockIds[i];
        gpuCol2BlockId[blockIds[i]] = i;
    }

    CUDA_CALL(cudaMemcpy(Data(), val, nz * sizeof(ElemType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(BlockId2ColOrRow(), &gpuBlockId2Col[0], numBlocks * sizeof(GPUSPARSE_INDEX_TYPE), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(ColOrRow2BlockId(), &gpuCol2BlockId[0], numBlocks * sizeof(GPUSPARSE_INDEX_TYPE), cudaMemcpyHostToDevice));
}
#endif

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
        h_CSCCol = new GPUSPARSE_INDEX_TYPE[GetNumRows() + 1];
        h_Row = new GPUSPARSE_INDEX_TYPE[nz];

        PrepareDevice();
        CUDA_CALL(cudaMemcpy(h_Val, Data(), GetSizeElemAllocated(), cudaMemcpyDeviceToHost));

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

            ConvertBuffer(h_CSCCol, pCol, SecondaryIndexCount());
            ConvertBuffer(h_Row, pRow, MajorIndexCount());
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

    // BUGBUG: Below we fail if one of the factors is empty.That is wrong. We should be able to handle empty factors (e.g. worker of a minibatch got 0 samples).
    // Probably one should test further down and exit early, but we need to make sure that c is correct for beta != 0.
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
        InvalidArgument("GPUSparseMatrix::MultiplyAndWeightedAdd: The inner dimensions of a (= %d) and b (= %d) don't match.", k, l);
    }

    if (beta == 0)
        c.RequireSize(m, n);
    else
        c.VerifySize(m, n); // Can't resize if beta != 0

    c.PrepareDevice();
    if (rhs.GetFormat() == MatrixFormat::matrixFormatSparseCSC)
    {
        ConvolveAndWeightedAdd(alpha, lhs, transposeA, rhs, transposeB, beta, c, 1, 1, false, false);
    }
    else if (rhs.GetFormat() == matrixFormatSparseCSR)
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
        c.RequireSize(cRows, cCols);
    else
        c.VerifySize(cRows, cCols); // Can't resize if beta != 0

    c.PrepareDevice();
    if (rhs.GetFormat() == MatrixFormat::matrixFormatSparseCSC)
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
                reinterpret_cast<const ElemType*>(lhs.Data()), // dense
                transposeA,
                reinterpret_cast<const ElemType*>(rhs.Buffer()), // sparse nz values. Note that because of the offsets we use the array
                rhs.RowLocation(),
                rhs.ColLocation(),
                beta,
                reinterpret_cast<ElemType*>(c.Data()) // dense target
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
                    reinterpret_cast<const ElemType*>(lhs.Data()), // dense
                    transposeA,
                    reinterpret_cast<const ElemType*>(rhs.Buffer()), // sparse nz values
                    rhs.RowLocation(),
                    rhs.ColLocation(),
                    reinterpret_cast<ElemType*>(c.Data()) // dense target
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
void GPUSparseMatrix<ElemType>::TensorShuffleScaleAndAdd(ElemType keepWeight, const GPUSparseMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, 
    ElemType scaleFactor, const GPUSparseMatrix<ElemType>& b, GPUSparseMatrix<ElemType>& c)
{
    c.VerifyWritable(__FUNCTION__);

    if (a.GetComputeDeviceId() != c.GetComputeDeviceId() || b.GetComputeDeviceId() != c.GetComputeDeviceId())
        RuntimeError("GPUSparseMatrix<ElemType>::TensorShuffleScaleAndAdd: All matrices must be on the same GPU");

    if (a.GetFormat() != MatrixFormat::matrixFormatSparseCSC || b.GetFormat() != MatrixFormat::matrixFormatSparseCSC || c.GetFormat() != MatrixFormat::matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    // Can't distribute the operations if we need to move values across columns
    if (a.GetNumCols() != T || keepWeight != 0 || scaleFactor != 1)
        NOT_IMPLEMENTED;

    if (a.GetNumRows() != D * S * M * K)
        LogicError("GPUSparseMatrix<ElemType>::TensorShuffleScaleAndAdd: tensor dimensions and underlying matrix dimensions don't match");

    c.RequireSizeAndAllocate(a.GetNumRows(), a.GetNumCols(), a.NzCount(), true, false);

    if (a.NzCount() > 0)
    {
        c.PrepareDevice();
        SyncGuard syncGuard;
        CUDA_LONG N = (CUDA_LONG) a.NzCount();
        int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
        _tensorShuffleScaleAndAddRowSparse<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            reinterpret_cast<const ElemType*>(a.Buffer()), // source nz values
            a.RowLocation(),
            a.ColLocation(),
            reinterpret_cast<ElemType*>(c.Buffer()), // target nz values
            c.RowLocation(),
            c.ColLocation(),
            D, S, M, K, T,
            a.NzCount());
    }
    else
    {
        CUDA_CALL(cudaMemset(c.Buffer(), 0, c.BufferSizeAllocated()));
    }
}

// backward pass from hidden layer to feature weight
// dense X sparse = sparse
template <class ElemType>
void GPUSparseMatrix<ElemType>::MultiplyAndAdd(ElemType alpha, const GPUMatrix<ElemType>& lhs, const bool transposeA,
                                               const GPUSparseMatrix<ElemType>& rhs, const bool transposeB, GPUSparseMatrix<ElemType>& c)
{
    c.VerifyWritable(__FUNCTION__);

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
        InvalidArgument("GPUSparseMatrix::MultiplyAndAdd: The inner dimensions of a (= %d) and b (= %d) don't match.", k, l);
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
        size_t rhs_nz = rhs.NzCount();

        size_t blockSizePrev = c.GetBlockSize();
        if (blockSizePrev == 0)
        {
            c.Resize(m, n, 0);
            CUDA_CALL(cudaMemset(c.ColOrRow2BlockId(), Id_NotAssigned, sizeof(GPUSPARSE_INDEX_TYPE) * (n)));
            CUDA_CALL(cudaMemset(c.BlockId2ColOrRow(), Id_NotAssigned, sizeof(GPUSPARSE_INDEX_TYPE) * (n)));
        }

        size_t* blockSize = TracingGPUMemoryAllocator::Allocate<size_t>(lhs.GetComputeDeviceId(), 1);
        CUDA_CALL(cudaMemcpy(blockSize, &blockSizePrev, sizeof(size_t), cudaMemcpyHostToDevice));

        blocksPerGrid = (int) ceil(((double) rhs_nz) / GridDim::maxThreadsPerBlock);
        _findColsWithValues<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            rhs.RowLocation(), c.ColOrRow2BlockId(), rhs_nz);
                
        blocksPerGrid = (int) ceil(((double) n) / GridDim::maxThreadsPerBlock);
        _determineBlockIds<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            c.BlockId2ColOrRow(), c.ColOrRow2BlockId(), n, blockSize);

        size_t blockSizeCurr;
        CUDA_CALL(cudaMemcpy(&blockSizeCurr, blockSize, sizeof(size_t), cudaMemcpyDeviceToHost));
        TracingGPUMemoryAllocator::Free<size_t>(lhs.GetComputeDeviceId(), blockSize);
        c.SetBlockSize(blockSizeCurr);

        if (blockSizeCurr > blockSizePrev)
        {
            // zero initialize new blocks
            size_t nnz = m * blockSizeCurr;
            c.RequireSizeAndAllocate(m, n, nnz, true, true); // we need to keep the col2blockid and blockid2col info when resizing.
            CUDA_CALL(cudaMemset(c.Data() + m * blockSizePrev, 0, sizeof(ElemType) * m * (blockSizeCurr - blockSizePrev)));
        }

        LONG64 N = (LONG64) lhs.GetNumElements(); // here we process for each row in lhs and each column in rhs (==columns in lhs)
        blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
        _denseMulSparseCSCTransposeToSparseBlockCol2<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            alpha,
            lhs.Data(),
            m,
            l,
            rhs.Data(),
            rhs.RowLocation(),
            rhs.ColLocation(),
            c.ColOrRow2BlockId(),
            c.Data());
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

    let nnz = NzCount();
    this->ReserveTempDeviceBuffer(nnz);
    map<size_t, GPUSPARSE_INDEX_TYPE> indexer;
    GPUSPARSE_INDEX_TYPE* rowToId = (GPUSPARSE_INDEX_TYPE*) ReserveTempHostBuffer(sizeof(GPUSPARSE_INDEX_TYPE) * nnz * 2);

    // In the first nnz values of the 'rowToId' we will store the block ids of the nonzero-values (to be computed below).
    // In the next nnz values of 'rowToId' we store the row-ids of the non-zero values (copied from GPU).
    GPUSPARSE_INDEX_TYPE* h_Row = rowToId + nnz;
    CUDA_CALL(cudaMemcpy(h_Row, RowLocation(), sizeof(GPUSPARSE_INDEX_TYPE) * nnz, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nnz; i++)
    {
        size_t row = h_Row[i];
        if (indexer.find(row) == indexer.end())
        {
            size_t id = indexer.size(); // We need to assign size to a temp variable due to difference in Linux and Windows
            indexer[row] = id;
        }
        rowToId[i] = indexer[row];
    }
    CUDA_CALL(cudaMemcpy(GetTempDeviceBuffer(), rowToId, sizeof(GPUSPARSE_INDEX_TYPE) * nnz, cudaMemcpyHostToDevice));
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

    if (lhs.GetFormat() == matrixFormatSparseBlockCol || lhs.GetFormat() == matrixFormatSparseBlockRow)
    {
        bool blockCol = (lhs.GetFormat() == matrixFormatSparseBlockCol);

        SyncGuard syncGuard;
        LONG64 N = (LONG64) lhs.GetNumNZElements();
        int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
        _scaleSparseBlockAndAddToDense<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(
            alpha,
            blockCol,
            lhs.GetNumRows(),
            lhs.GetNumCols(),
            lhs.GetBlockSize(),
            lhs.Data(),
            lhs.BlockId2ColOrRow(),
            rhs.Data());

    }
    else
    {
        ScaleAndAdd(alpha, lhs, 1, rhs, rhs);
    }
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceTruncate(const ElemType threshold)
{
    VerifyWritable(__FUNCTION__);

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
    VerifyWritable(__FUNCTION__);

    CUDA_LONG N = (CUDA_LONG) GetNumNZElements();

    CUDA_LONG blocksPerGrid = (CUDA_LONG) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    ElemType* values = NzValues();
    _inplaceSoftThreshold<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(values, threshold, N);
    return *this;
}

// A helper method used in MomentumSGDUpdate and NesterovAcceleratedMomentumSGDUpdate.
// Modifies the smoothed gradients "c", as well as the current gradients "this" on which this method is invoked. 
// Classic momentum (unitGainFactor == 1.0):
// 1) c = momentum * c + this
// Unit-gain momentum (unitGainFactor == 1.0 - momentum):
// 1) c = momentum * c + (1.0 - momentum) * this
// 2) this = c
// TODO: NormalGrad is a misnomer here. Come up with a better name.
template <class ElemType>
void GPUSparseMatrix<ElemType>::NormalGrad(GPUMatrix<ElemType>& c, const ElemType momentum, bool unitGainMomentum)
{
    VerifyWritable(__FUNCTION__);

    if (c.IsEmpty())
    {
        c.RequireSize(GetNumRows(), GetNumCols());
        c.SetValue(0.0);
    }

    if (GetFormat() == matrixFormatSparseBlockCol || GetFormat() == matrixFormatSparseBlockRow)
    {
        bool isBlockCol = (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol);
        SyncGuard syncGuard;
        LONG64 N = (LONG64) GetNumNZElements();
        int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);

        _normalGradForSparseBlock<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(
            momentum,
            isBlockCol,
            GetNumRows(),
            GetNumCols(),
            GetBlockSize(),
            Data(),
            BlockId2ColOrRow(),
            c.Data(),
            unitGainMomentum);
    }
    else
    {
        NOT_IMPLEMENTED;
    }
}

template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::Adagrad(GPUMatrix<ElemType>& c, const bool needAveMultiplier)
{
    VerifyWritable(__FUNCTION__);

    size_t numColsNeeded = GetNumCols();
    if (needAveMultiplier)
        numColsNeeded += GetNumCols();

    if (c.IsEmpty() || c.GetNumCols() < numColsNeeded)
    {
        c.RequireSize(GetNumRows(), numColsNeeded);
        c.SetValue(0.0);
    }

    assert(c.GetNumRows() == GetNumRows() && c.GetNumCols() == numColsNeeded);

    size_t n = this->GetNumElements();

    ElemType* multipliers = nullptr;
    if (needAveMultiplier)
        multipliers = c.Buffer() + n; // temp memory used to store multipliers,

    if (GetFormat() == MatrixFormat::matrixFormatSparseCSC || GetFormat() == MatrixFormat::matrixFormatSparseCSR)
    {
        NOT_IMPLEMENTED;
    }
    else if (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol || GetFormat() == MatrixFormat::matrixFormatSparseBlockRow)
    {
        let nz = NzCount();
        int blocksPerGrid = (nz + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
        bool colMajor = GetFormat() == MatrixFormat::matrixFormatSparseBlockCol;
        size_t len = colMajor ? GetNumRows() : GetNumCols();
        _adagrad4BlockSparse<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(c.Buffer(), c.GetNumRows(), Data(), BlockId2ColOrRow(), multipliers, colMajor, len, nz);
    }
    else
        NOT_IMPLEMENTED;

    if (!needAveMultiplier)
        return 1;

    let nz = NzCount();
    cublasHandle_t cuHandle = GPUMatrix<ElemType>::GetCublasHandle(GetComputeDeviceId());
    if (sizeof(ElemType) == sizeof(float))
    {
        float aveMultiplier = 0;
        CUBLAS_CALL(cublasSasum(cuHandle, (LONG64) nz, reinterpret_cast<float*>(multipliers), 1, &aveMultiplier));
        return (ElemType) aveMultiplier / nz;
    }
    else
    {
        double aveMultiplier = 0;
        CUBLAS_CALL(cublasDasum(cuHandle, (LONG64) nz, reinterpret_cast<double*>(multipliers), 1, &aveMultiplier));
        return (ElemType) aveMultiplier / nz;
    }
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::FSAdagrad(
    GPUMatrix<ElemType>& c,
    GPUMatrix<ElemType>& functionValues,
    ElemType learnRatePerSample,
    ElemType momentum,
    ElemType adaWeight,
    ElemType adaMul,
    bool unitGainMomentum)
{
    if (GetFormat() != MatrixFormat::matrixFormatSparseBlockCol)
    {
        NOT_IMPLEMENTED;
    }

    size_t numColsNeeded = 2 * GetNumCols();

    if (c.IsEmpty() || (c.GetNumCols() < numColsNeeded))
    {
        c.RequireSize(GetNumRows(), numColsNeeded);
        c.SetValue(0.0);
    }

    assert((c.GetNumRows() == GetNumRows()) && (c.GetNumCols() == numColsNeeded));

    size_t n = GetNumElements();
    int blocksPerGrid = (n + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    _fsadagrad4BlockSparseCol<ElemType> << <blocksPerGrid, GridDim::maxThreadsPerBlock >> >(
        n, Data(), ColOrRow2BlockId(), GetNumRows(),
        c.Data(), c.Data() + n, functionValues.Data(),
        learnRatePerSample, momentum, adaWeight, adaMul, unitGainMomentum);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::Adam(
    GPUMatrix<ElemType>& c,
    GPUMatrix<ElemType>& functionValues,
    ElemType learnRatePerSample,
    ElemType momentum,
    ElemType adaWeight,
    ElemType adaMul,
    bool unitGainMomentum)
{
    if (GetFormat() != MatrixFormat::matrixFormatSparseBlockCol)
    {
        NOT_IMPLEMENTED;
    }

    size_t numColsNeeded = 2 * GetNumCols();

    if (c.IsEmpty() || (c.GetNumCols() < numColsNeeded))
    {
        c.RequireSize(GetNumRows(), numColsNeeded);
        c.SetValue(0.0);
    }

    assert((c.GetNumRows() == GetNumRows()) && (c.GetNumCols() == numColsNeeded));

    size_t n = GetNumElements();
    int blocksPerGrid = (n + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    _adam4BlockSparseCol<ElemType> << <blocksPerGrid, GridDim::maxThreadsPerBlock >> >(
        n, Data(), ColOrRow2BlockId(), GetNumRows(),
        c.Data(), c.Data() + n, functionValues.Data(),
        learnRatePerSample, momentum, adaWeight, adaMul, unitGainMomentum);
}

template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::RmsProp(GPUMatrix<ElemType>& c,
    ElemType RMS_GAMMA,
    ElemType RMS_WGT_INC,
    ElemType RMS_WGT_MAX,
    ElemType RMS_WGT_DEC,
    ElemType RMS_WGT_MIN,
    const bool needAveMultiplier)
{
    if (GetFormat() != MatrixFormat::matrixFormatSparseBlockCol)
    {
        NOT_IMPLEMENTED;
    }

    const ElemType floor = 1e-6f;
    static ElemType* upd_gpu = (ElemType*)0;

    size_t n = GetNumElements();
    int blocksPerGrid = (c.GetNumElements() + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;

    size_t numColsNeeded = GetNumCols() * 3;
    if (needAveMultiplier)
        numColsNeeded += GetNumCols();

    if (c.IsEmpty() || c.GetNumCols() < numColsNeeded)
    {
        c.RequireSize(GetNumRows(), numColsNeeded);
        c.SetValue(0.0);

        ElemType* avars = c.Data();         // accumulated variances for RMS scaling
        ElemType* signs = c.Data() + n;     // sign of previous gradient
        ElemType* steps = c.Data() + 2 * n; // current step size
                                            // Data()+3*n is temp memory used to store multipliers, no need to initialize

        _rmsprop_init4BlockSparseCol<ElemType> << <blocksPerGrid, GridDim::maxThreadsPerBlock >> >(
            avars, signs, steps, 
            Data(), ColOrRow2BlockId(), GetNumRows(),
            n);
    }
    assert(c.GetNumRows() == GetNumRows() && c.GetNumCols() == numColsNeeded);

    ElemType* avars = c.Data();         // accumulated variances for RMS scaling
    ElemType* signs = c.Data() + n;     // sign of previous gradient
    ElemType* steps = c.Data() + 2 * n; // current step size

    ElemType* multipliers = nullptr;
    if (needAveMultiplier)
        multipliers = c.Data() + 3 * n; // temp memory used to store multipliers,

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

    _rmsprop4BlockSparseCol<ElemType> << <blocksPerGrid, GridDim::maxThreadsPerBlock >> >(
        avars, signs, steps,
        Data(), ColOrRow2BlockId(), GetNumRows(),
        n,
        RMS_GAMMA, RMS_WGT_INC, RMS_WGT_MAX, RMS_WGT_DEC, RMS_WGT_MIN,
        floor, upd_gpu, multipliers);

    if (!needAveMultiplier)
        return 1;

    cublasHandle_t cuHandle = GPUMatrix<ElemType>::GetCublasHandle(GetComputeDeviceId());
    if (sizeof(ElemType) == sizeof(float))
    {
        float aveMultiplier = 0;
        CUBLAS_CALL(cublasSasum(cuHandle, (CUDA_LONG)n, reinterpret_cast<float*>(multipliers), 1, &aveMultiplier));
        return aveMultiplier / n;
    }
    else
    {
        double aveMultiplier = 0;
        CUBLAS_CALL(cublasDasum(cuHandle, (CUDA_LONG)n, reinterpret_cast<double*>(multipliers), 1, &aveMultiplier));
        return (ElemType)aveMultiplier / n;
    }
}

// sparse X dense = dense
template <class ElemType>
void GPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& a, const bool transposeA,
                                                       const GPUMatrix<ElemType>& b, const bool transposeB, ElemType beta, GPUMatrix<ElemType>& c)
{
    if (transposeB)
        NOT_IMPLEMENTED;

    // Note: This function is written for 'a' being in CSR format. If 'a' is CSC, we reinterpret it as CSR by transposing it.
    if (a.GetFormat() != matrixFormatSparseCSR && a.GetFormat() != matrixFormatSparseCSC)
        NOT_IMPLEMENTED;
    const bool reinterpretAsCSR = a.GetFormat() == matrixFormatSparseCSC;

    if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || (b.GetComputeDeviceId() != a.GetComputeDeviceId()))
        RuntimeError("MultiplyAndWeightedAdd: All matrices must be on the same GPU");

    a.PrepareDevice();
    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseOperation_t oper = (transposeA != reinterpretAsCSR) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

    int n = (int)b.GetNumCols();
    int m = (int)(reinterpretAsCSR ? a.GetNumCols() : a.GetNumRows());
    int k = (int)(reinterpretAsCSR ? a.GetNumRows() : a.GetNumCols());
    assert(n == (int) c.GetNumCols());

    const auto& aRowLocation = reinterpretAsCSR ? a.ColLocation() : a.RowLocation();
    const auto& aColLocation = reinterpretAsCSR ? a.RowLocation() : a.ColLocation();

    SyncGuard syncGuard;
    if (sizeof(ElemType) == sizeof(float))
    {
        CUSPARSE_CALL(cusparseScsrmm(cusparseHandle, oper, m, n, k, (int) a.GetNumNZElements(), reinterpret_cast<float*>(&alpha), descr, reinterpret_cast<const float*>(a.Buffer()),
                                     aRowLocation, aColLocation, reinterpret_cast<float*>(b.Data()),
                                     (int) b.GetNumRows(), reinterpret_cast<float*>(&beta), reinterpret_cast<float*>(c.Data()), (int) c.GetNumRows()));
    }
    else
    {
        CUSPARSE_CALL(cusparseDcsrmm(cusparseHandle, oper, m, n, k, (int) a.GetNumNZElements(), reinterpret_cast<double*>(&alpha), descr, reinterpret_cast<const double*>(a.Buffer()),
                                     aRowLocation, aColLocation, reinterpret_cast<double*>(b.Data()),
                                     (int) b.GetNumRows(), reinterpret_cast<double*>(&beta), reinterpret_cast<double*>(c.Data()), (int) c.GetNumRows()));
    }
    CUSPARSE_CALL(cusparseDestroy(cusparseHandle));
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::Multiply(const GPUSparseMatrix<ElemType>& S, const GPUMatrix<ElemType>& D, GPUMatrix<ElemType>& C)
{
    C.RequireSize(S.GetNumRows(), D.GetNumCols());

    MultiplyAndWeightedAdd(1, S, false, D, false, 0, C);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::Multiply(const GPUMatrix<ElemType>& D, const GPUSparseMatrix<ElemType>& S, GPUMatrix<ElemType>& C)
{
    C.RequireSize(S.GetNumCols(), D.GetNumRows());

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
    return ElemCountFromBufferSize(GetNumRows(), GetNumCols(), GetFormat(), BufferSizeAllocated());
}

// PrepareBuffer - Get the dimensions start buffer, computes the starting row/column of each value
// m - rows in the source
// n - cols in the source
// canReuseBuffer - target matrix can be reused for temporary space
// func - function to call to count elements in the result (returns count, and fills csrRowPtr array)
template <class ElemType>
void GPUSparseMatrix<ElemType>::PrepareBuffer(size_t m, size_t n, bool canReuseBuffer, std::function<size_t(GPUSPARSE_INDEX_TYPE* csrRowPtrC)> func)
{
    VerifyWritable(__FUNCTION__);

    if (this->GetFormat() != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    PrepareDevice();

    GPUSPARSE_INDEX_TYPE* csrRowPtrC = nullptr;
    GPUSparseMatrix<ElemType>& c = *this;
    size_t cSize = c.BufferSizeAllocated();
    size_t rowBufferRequired = (m + 1) * sizeof(GPUSPARSE_INDEX_TYPE);
    bool allocatedBuffer = false;

    // do we have enough memory to store just the row buffer?
    if (cSize >= rowBufferRequired && c.Data() != nullptr && canReuseBuffer)
    {
        csrRowPtrC = (GPUSPARSE_INDEX_TYPE*) c.Data();
    }
    else
    {
        csrRowPtrC = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(GetComputeDeviceId(), rowBufferRequired / sizeof(GPUSPARSE_INDEX_TYPE));
        allocatedBuffer = true;
    }

    // get the non-zero count from the function (and
    size_t nnzC = func(csrRowPtrC);

    // now we know the number of Non-zeros in the result set, set the output size
    c.RequireSizeAndAllocate(m, n, nnzC, true, false);

    CUDA_CALL(cudaMemcpy(c.SecondaryIndexLocation(), csrRowPtrC, c.SecondaryIndexSize(), cudaMemcpyDeviceToDevice));

    // if we allocated the buffer, free it here
    if (allocatedBuffer)
        TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(GetComputeDeviceId(), csrRowPtrC);
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
    c.VerifyWritable(__FUNCTION__);

    if (S1.GetFormat() != matrixFormatSparseCSR || S2.GetFormat() != matrixFormatSparseCSR || c.GetFormat() != matrixFormatSparseCSR)
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
        CUSPARSE_CALL(cusparseScsrgemm(cusparseHandle, operA, operB, m, n, k, descrA, nnzA, (const float*) S1.Buffer(), S1.RowLocation(), S1.ColLocation(),
                                       descrB, nnzB, (const float*) S2.Buffer(), S2.RowLocation(), S2.ColLocation(),
                                       descrC, (float*) c.Data(), c.RowLocation(), c.ColLocation()));
    }
    else
    {
        CUSPARSE_CALL(cusparseDcsrgemm(cusparseHandle, operA, operB, m, n, k, descrA, nnzA, (const double*) S1.Buffer(), S1.RowLocation(), S1.ColLocation(),
                                       descrB, nnzB, (const double*) S2.Buffer(), S2.RowLocation(), S2.ColLocation(),
                                       descrC, (double*) c.Data(), c.RowLocation(), c.ColLocation()));
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
    if (a.GetFormat() != matrixFormatSparseCSR || b.GetFormat() != matrixFormatSparseCSR )
    {
        NOT_IMPLEMENTED;
    }
    if (c.m_sob == nullptr)
        c.ZeroInit(a.GetFormat(), a.GetComputeDeviceId());

    if (a.GetNumCols() != b.GetNumCols() || a.GetNumRows() != b.GetNumRows())
        RuntimeError("Dimensions mismatch in ScaleAndAdd");
    if (a.GetComputeDeviceId() != b.GetComputeDeviceId())
        RuntimeError("ScaleAndAdd: matrices must be on the same device");

    c.SetFormat(a.GetFormat());
    c.SetComputeDeviceId(a.GetComputeDeviceId());
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
        CUSPARSE_CALL(cusparseScsrgeam(cusparseHandle, m, n, reinterpret_cast<const float*>(&alpha), descrA, nnzA, reinterpret_cast<const float*>(a.Data()), a.RowLocation(), a.ColLocation(),
                                       reinterpret_cast<const float*>(&beta), descrB, nnzB, reinterpret_cast<const float*>(b.Data()), b.RowLocation(), b.ColLocation(), descrC, reinterpret_cast<float*>(c.Data()), c.RowLocation(), c.ColLocation()));
    }
    else
    {
        CUSPARSE_CALL(cusparseDcsrgeam(cusparseHandle, m, n, reinterpret_cast<const double*>(&alpha), descrA, nnzA, reinterpret_cast<const double*>(a.Data()), a.RowLocation(), a.ColLocation(),
                                       reinterpret_cast<const double*>(&beta), descrB, nnzB, reinterpret_cast<const double*>(b.Data()), b.RowLocation(), b.ColLocation(), descrC, reinterpret_cast<double*>(c.Data()), c.RowLocation(), c.ColLocation()));
    }
    cusparseDestroy(cusparseHandle);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& a, ElemType beta, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    if (a.GetFormat() != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    if (a.GetNumRows() != b.GetNumRows() || a.GetNumRows() != c.GetNumRows() || a.GetNumCols() != b.GetNumCols() || a.GetNumCols() != c.GetNumCols())
        LogicError("ScaleAndAdd: dimension mismatch");
    if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || a.GetComputeDeviceId() != c.GetComputeDeviceId())
        RuntimeError("ScaleAndAdd: matrices must be on the same device");
    b.PrepareDevice();
    // copy b to c
    CUDA_CALL(cudaMemcpy(c.Data(), b.Data(), sizeof(ElemType) * b.GetNumElements(), cudaMemcpyDeviceToDevice));
    if (beta != 1)
    {
        c *= beta;
    }
    SyncGuard syncGuard;
    CUDA_LONG M = (CUDA_LONG) a.GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * M / GridDim::maxThreadsPerBlock);
    _sparseCSRPlusDense<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(alpha, a.Data(), a.RowLocation(), a.ColLocation(), c.Data(), M);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const GPUMatrix<ElemType>& a, ElemType beta, const GPUSparseMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    ScaleAndAdd(beta, b, alpha, a, c);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::Scale(ElemType alpha, GPUSparseMatrix<ElemType>& a)
{
    a.VerifyWritable(__FUNCTION__);

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
    c.VerifyWritable(__FUNCTION__);

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
    if (a.GetFormat() != matrixFormatSparseCSR && a.GetFormat() != matrixFormatSparseCSC)
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
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));

    bool allocTemp = (a.GetFormat() == matrixFormatSparseCSR);

    if (allocTemp) // need to put a in ColumnMajor format
    {
        cscValA = TracingGPUMemoryAllocator::Allocate<ElemType>(a.GetComputeDeviceId(), nnz);
        cscRowIndA = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), nnz);
        cscColPtrA = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), (n + 1));

        SyncGuard syncGuard;
        if (sizeof(ElemType) == sizeof(float))
        {
            CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const float*>(a.Data()), a.RowLocation(), a.ColLocation(), reinterpret_cast<float*>(cscValA), cscRowIndA, cscColPtrA, cpVals, idxBase));
        }
        else
        {
            CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const double*>(a.Data()), a.RowLocation(), a.ColLocation(), reinterpret_cast<double*>(cscValA), cscRowIndA, cscColPtrA, cpVals, idxBase));
        }
    }
    else if (a.GetFormat() == matrixFormatSparseCSC)
    {
        cscValA = (ElemType*) a.Data();
        cscRowIndA = a.RowLocation();
        cscColPtrA = a.ColLocation();
    }
    else
    {
        NOT_IMPLEMENTED;
    }
    let a_nz = a.NzCount();
    // Given sparse matrix in column major format, calculate indices for corresponding sparse vector
    GPUSPARSE_INDEX_TYPE* vectArray = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), a_nz);
    CUDA_LONG M = n;
    CUDA_LONG N = m;
    // GPUSPARSE_INDEX_TYPE* h_vectArray= new int[a.m_nz];
    int blocksPerGrid = (int) ceil(1.0 * M / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _getSparseVectorRepresntationForCSCMatrix<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(cscColPtrA, cscRowIndA, vectArray, M, N);
    if (allocTemp)
    {
        TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), cscRowIndA);
        TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), cscColPtrA);
    }
    // CUDA_CALL(cudaMemcpy(h_vectArray,vectArray,sizeof(GPUSPARSE_INDEX_TYPE)*a.m_nz,cudaMemcpyDeviceToHost));

    // Actual dot product
    ElemType res = 0;
    if (sizeof(ElemType) == sizeof(float))
    {
        CUSPARSE_CALL(cusparseSdoti(cusparseHandle, (int) a_nz, reinterpret_cast<float*>(cscValA), vectArray,
                                    reinterpret_cast<float*>(b.Data()),
                                    reinterpret_cast<float*>(&res), idxBase));
    }
    else
    {
        CUSPARSE_CALL(cusparseDdoti(cusparseHandle, (int) a_nz, reinterpret_cast<double*>(cscValA), vectArray,
                                    reinterpret_cast<double*>(b.Data()),
                                    reinterpret_cast<double*>(&res), idxBase));
    }
    TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(a.GetComputeDeviceId(), vectArray);
    if (allocTemp)
    {
        TracingGPUMemoryAllocator::Free<ElemType>(a.GetComputeDeviceId(), cscValA);
    }
    CUSPARSE_CALL(cusparseDestroy(cusparseHandle));
    return res;
}

template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::InnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b)
{
    return GPUSparseMatrix<ElemType>::InnerProductOfMatrices(b, a);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::InnerProduct(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const bool isColWise)
{
    if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || b.GetComputeDeviceId() != c.GetComputeDeviceId()) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");

    if (a.IsEmpty() || b.IsEmpty())
        LogicError("Scale:  one of the input matrices is empty.");

    if (a.GetFormat() != MatrixFormat::matrixFormatSparseCSC)
    {
        NOT_IMPLEMENTED;
    }

    const int m = (int)a.GetNumRows();
    const int n = (int)a.GetNumCols();
    const int k = (int)b.GetNumRows();
    const int l = (int)b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("Matrices a and b should have same dimension.");

    if (isColWise)
        c.RequireSize(1, n);
    else
        c.RequireSize(m, 1);

    c.PrepareDevice();

    int blocksPerGrid = 0;
    if (isColWise) // col-wise
    {
        blocksPerGrid = (int)ceil(1.0 * n / GridDim::maxThreadsPerBlock);
    }
    else
    {
        blocksPerGrid = (int)ceil(1.0 * m / GridDim::maxThreadsPerBlock);
    }

    SyncGuard syncGuard;
    _innerProduct4SparseCSC<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
        c.Data(),
        a.Data(), a.RowLocation(), a.ColLocation(),
        b.Data(),
        m, n, isColWise);
}

// This is an utility function useful for debugging issues with sparse matrices.
// It just checks that the CSC format indices are not corrupted / pointing to invalid memory.
template <class ElemType>
bool GPUSparseMatrix<ElemType>::IsValid() const
{
    if (GetFormat() != MatrixFormat::matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    long* res = new long[4];
    res[0] = 1;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    long* d_res = TracingGPUMemoryAllocator::Allocate<long>(GetComputeDeviceId(), 4);
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

    if (a.GetFormat() != b.GetFormat())
        NOT_IMPLEMENTED;

    long* res = new long[3];
    res[0] = 1;
    res[1] = 1;
    res[2] = 1;
    long* d_res = TracingGPUMemoryAllocator::Allocate<long>(a.GetComputeDeviceId(), 3);
    CUDA_CALL(cudaMemcpy(d_res, res, sizeof(long) * 3, cudaMemcpyHostToDevice));

    int blocksPerGrid = (int) ceil(1.0 * a.GetNumNZElements() / GridDim::maxThreadsPerBlock);
    _areEqual<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(a.NzValues(), b.NzValues(), (CUDA_LONG) a.GetNumNZElements(), threshold, d_res);
    _areEqual<int><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(a.MajorIndexLocation(), b.MajorIndexLocation(), (CUDA_LONG) a.MajorIndexCount(), (int) threshold, d_res + 1);
    blocksPerGrid = (int) ceil((1.0 * a.SecondaryIndexCount()) / GridDim::maxThreadsPerBlock);
    _areEqual<int><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(a.SecondaryIndexLocation(), b.SecondaryIndexLocation(), (CUDA_LONG) a.SecondaryIndexCount(), (int) threshold, d_res + 2);

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

// sparse x dense = dense
template <class ElemType>
GPUMatrix<ElemType> GPUSparseMatrix<ElemType>::ElementProductOf(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.GetFormat() != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    if (a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
        LogicError("ElementProductOf: matrix dimensions mismatch");

    b.PrepareDevice();
    GPUMatrix<ElemType> c(b.GetNumRows(), b.GetNumCols(), b.GetComputeDeviceId());

    SyncGuard syncGuard;
    CUDA_LONG M = (CUDA_LONG) a.GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * M / GridDim::maxThreadsPerBlock);
    _sparseCSRElemMulDense<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(a.Data(), a.RowLocation(), a.ColLocation(), b.Data(), c.Data(), M);
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
    c.RequireSizeAndAllocate(n, m, nnz, GetFormat(), true, false);

    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CALL(cusparseCreate(&cusparseHandle));

    SyncGuard syncGuard;
    if (GetFormat() == MatrixFormat::matrixFormatSparseCSR)
    {
        if (nnz > 0)
        {
            if (sizeof(ElemType) == sizeof(float))
            {
                CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const float*>(Data()), RowLocation(), ColLocation(),
                                               reinterpret_cast<float*>(c.Data()), c.ColLocation(), c.RowLocation(), cpVals, idxBase));
            }
            else
            {
                CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const double*>(Data()), RowLocation(), ColLocation(),
                                               reinterpret_cast<double*>(c.Data()), c.ColLocation(), c.RowLocation(), cpVals, idxBase));
            }
        }
        else
        {
            CUDA_CALL(cudaMemset(c.Buffer(), 0, c.BufferSizeAllocated()));
        }
    }
    else if (GetFormat() == matrixFormatSparseCSC)
    {
        if (nnz > 0)
        {
            if (sizeof(ElemType) == sizeof(float))
            {
                CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, n, m, nnz, reinterpret_cast<const float*>(this->Data()), this->ColLocation(), this->RowLocation(),
                                               reinterpret_cast<float*>(c.Data()), c.RowLocation(), c.ColLocation(), cpVals, idxBase));
            }
            else
            {
                CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, n, m, nnz, reinterpret_cast<const double*>(this->Data()), this->ColLocation(), this->RowLocation(),
                                               reinterpret_cast<double*>(c.Data()), c.RowLocation(), c.ColLocation(), cpVals, idxBase));
            }
        }
        else
        {
            CUDA_CALL(cudaMemset(c.Buffer(), 0, c.BufferSizeAllocated()));
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
    VerifyWritable(__FUNCTION__);

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
    if (startColumn + numCols > GetNumCols())
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) GetNumCols());

    if (GetFormat() != MatrixFormat::matrixFormatSparseCSC && (startColumn != 0 || numCols != GetNumCols()))
        NOT_IMPLEMENTED;

    GPUSparseMatrix<ElemType> slice(GetComputeDeviceId());
    slice.ShallowCopyFrom(*this);
    slice.SetNumCols(numCols);
    slice.m_sliceViewOffset          = m_sliceViewOffset + startColumn; // Just shift the compressed index location to the new startColumn - that's it!
    // Note: m_nz is missing from here because it does not exist. We must compute it every time.

    return slice;
}
    
template <class ElemType>
void GPUSparseMatrix<ElemType>::AssignColumnSliceToDense(GPUMatrix<ElemType>& slice, size_t startColumn, size_t numCols) const
{
    int m = (int) GetNumRows();
    int n = (int) GetNumCols();

    // We can either error out or RequireSize. Because RequireSize will error out if it's not allowed, I think this makes more sense.
    slice.RequireSize(m, numCols);

    if (startColumn + numCols > n)
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) n);

    if (GetFormat() != MatrixFormat::matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

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
        CUSPARSE_CALL(cusparseScsc2dense(cusparseHandle, m, numCols, descr, (float*) Buffer(), RowLocation(), ColLocation() + startColumn, (float*) slice.Data(), m));
    }
    else
    {
        CUSPARSE_CALL(cusparseDcsc2dense(cusparseHandle, m, numCols, descr, (double*) Buffer(), RowLocation(), ColLocation() + startColumn, (double*) slice.Data(), m));
    }

    CUSPARSE_CALL(cusparseDestroy(cusparseHandle));

}
template <class ElemType>
GPUMatrix<ElemType> GPUSparseMatrix<ElemType>::CopyColumnSliceToDense(size_t startColumn, size_t numCols) const
{
    GPUMatrix<ElemType> slice(GetNumRows(), numCols, GetComputeDeviceId());

    AssignColumnSliceToDense(slice, startColumn, numCols);

    return slice;
}

template <class ElemType>
GPUMatrix<ElemType> GPUSparseMatrix<ElemType>::DiagonalToDense() const
{
    int m = (int) GetNumRows();
    int n = (int) GetNumCols();

    if (m != n)
        LogicError("Diagonal can be called only for square matrix. (rows=%d, cols=%d)", m, n);

    if (GetFormat() != MatrixFormat::matrixFormatSparseCSC)
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

    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);
    ElemType h_sum;
    // WARNING: THIS kernel is not the most efficient way!
    _reductionSum1024Threads<ElemType><<<1, 1024>>>(NzValues(), d_sum, (LONG64) GetNumNZElements());
    CUDA_CALL(cudaMemcpy(&h_sum, d_sum, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_sum);

    return h_sum;
}

// sqrt(sum all elements^2)
template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::FrobeniusNorm() const
{
    if (IsEmpty())
        return 0;

    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);
    ElemType h_sum = 0;
    // WARNING: THIS kernel is not the most efficient way!
    _reductionSum21024Threads<ElemType><<<1, 1024>>>(NzValues(), d_sum, (int) GetNumNZElements());
    CUDA_CALL(cudaMemcpy(&h_sum, d_sum, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_sum);

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

    ElemType* d_maxAbs = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);
    ElemType h_maxAbs = 0;
    // WARNING: THIS kernel is not the most efficient way!
    _reductionMatrixNormInf1024Threads<ElemType><<<1, 1024>>>(NzValues(), d_maxAbs, (int) GetNumNZElements());
    CUDA_CALL(cudaMemcpy(&h_maxAbs, d_maxAbs, sizeof(ElemType), cudaMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_maxAbs);

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
        RequireSize(a.GetNumRows(), a.GetNumCols());
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
        RequireSize(a.GetNumRows(), a.GetNumCols());
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
        RequireSize(a.GetNumRows(), a.GetNumCols());
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
        RequireSize(a.GetNumRows(), a.GetNumCols());
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
        RequireSize(a.GetNumRows(), a.GetNumCols());
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
        RequireSize(a.GetNumRows(), a.GetNumCols());
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
        RequireSizeAndAllocate(a.GetNumRows(), a.GetNumCols(), a.NzCount());
    performElementWiseFunction(ElementWiseOperator::opAbs, a);
    return *this;
}

// TODO: Check whether these functions always map 0 to 0.
template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::InplaceTruncateBottom(const ElemType threshold)
{
    VerifyWritable(__FUNCTION__);

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
    VerifyWritable(__FUNCTION__);

    if (a.IsEmpty())
        LogicError("AssignTruncateBottomOf: Matrix a is empty.");

    if (this != &a)
    {
        // RequireSize(a.GetNumRows(), a.GetNumCols());
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
    VerifyWritable(__FUNCTION__);

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
    VerifyWritable(__FUNCTION__);

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
    VerifyWritable(__FUNCTION__);

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
/*private*/ void GPUSparseMatrix<ElemType>::ConvertBuffer(OutType* outBuffer, const InType* inBuffer, const size_t size)
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
    if (GetTempHostBufferSize() < sizeInByte)
    {
        delete[](byte*) GetTempHostBuffer();
        SetTempHostBuffer(new byte[sizeInByte]);
        SetTempHostBufferSize(sizeInByte);
    }
    return (void*) GetTempHostBuffer();
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::performElementWiseFunction(ElementWiseOperator kind, const GPUSparseMatrix<ElemType>& src)
{
    VerifyWritable(__FUNCTION__);

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
template GPUSparseMatrix<char>::GPUSparseMatrix(DEVICEID_TYPE, const MatrixFormat);
template GPUSparseMatrix<char>::GPUSparseMatrix(const size_t, const size_t, const size_t, DEVICEID_TYPE, const MatrixFormat);
template GPUSparseMatrix<char>::GPUSparseMatrix(GPUSparseMatrix<char> const&);
template GPUSparseMatrix<char>::GPUSparseMatrix(GPUSparseMatrix<char>&&);
template void GPUSparseMatrix<char>::SetValue(CPUSparseMatrix<char> const&);
template void GPUSparseMatrix<char>::SetValue(GPUSparseMatrix<char> const&);
template void GPUSparseMatrix<char>::SetValue(GPUMatrix<char> const&);
//template void GPUSparseMatrix<char>::SetValue(CPUMatrix<char> const&);
template GPUMatrix<char> GPUSparseMatrix<char>::CopyToDenseMatrix() const;
template void GPUSparseMatrix<char>::CopyToDenseMatrix(GPUMatrix<char>&) const;
template void GPUSparseMatrix<char>::CopyToCPUSparseMatrix(CPUSparseMatrix<char>&) const;
template void GPUSparseMatrix<char>::ChangeDeviceTo(int);
template void GPUSparseMatrix<char>::Resize(const size_t, const size_t, const size_t, const bool);
template void GPUSparseMatrix<char>::RequireSizeAndAllocate(const size_t, const size_t, const size_t, const bool, const bool);
template void GPUSparseMatrix<char>::Reset();
template GPUSPARSE_INDEX_TYPE GPUSparseMatrix<char>::SecondaryIndexValueAt(size_t) const;
template GPUSparseMatrix<char>::~GPUSparseMatrix();
template GPUSparseMatrix<char> GPUSparseMatrix<char>::ColumnSlice(size_t, size_t) const;
template GPUMatrix<char> GPUSparseMatrix<char>::CopyColumnSliceToDense(size_t, size_t) const;
template GPUSparseMatrix<char>& GPUSparseMatrix<char>::operator=(GPUSparseMatrix<char>&&);
template void GPUSparseMatrix<char>::Reshape(const size_t, const size_t);
template void GPUSparseMatrix<char>::ScaleAndAdd(char, GPUSparseMatrix<char> const &, GPUMatrix<char> &);

// Support <short>
template GPUSparseMatrix<short>::GPUSparseMatrix(DEVICEID_TYPE, const MatrixFormat);
template GPUSparseMatrix<short>::GPUSparseMatrix(const size_t, const size_t, const size_t, DEVICEID_TYPE, const MatrixFormat);
template GPUSparseMatrix<short>::GPUSparseMatrix(GPUSparseMatrix<short> const&);
template GPUSparseMatrix<short>::GPUSparseMatrix(GPUSparseMatrix<short>&&);
template void GPUSparseMatrix<short>::SetValue(CPUSparseMatrix<short> const&);
template void GPUSparseMatrix<short>::SetValue(GPUSparseMatrix<short> const&);
template void GPUSparseMatrix<short>::SetValue(GPUMatrix<short> const&);
//template void GPUSparseMatrix<short>::SetValue(CPUMatrix<short> const&);
template GPUMatrix<short> GPUSparseMatrix<short>::CopyToDenseMatrix() const;
template void GPUSparseMatrix<short>::CopyToDenseMatrix(GPUMatrix<short>&) const;
template void GPUSparseMatrix<short>::CopyToCPUSparseMatrix(CPUSparseMatrix<short>&) const;
template void GPUSparseMatrix<short>::ChangeDeviceTo(int);
template void GPUSparseMatrix<short>::Resize(const size_t, const size_t, const size_t, const bool);
template void GPUSparseMatrix<short>::RequireSizeAndAllocate(const size_t, const size_t, const size_t, const bool, const bool);
template void GPUSparseMatrix<short>::Reset();
template GPUSPARSE_INDEX_TYPE GPUSparseMatrix<short>::SecondaryIndexValueAt(size_t) const;
template GPUSparseMatrix<short>::~GPUSparseMatrix();
template GPUSparseMatrix<short> GPUSparseMatrix<short>::ColumnSlice(size_t, size_t) const;
template GPUMatrix<short> GPUSparseMatrix<short>::CopyColumnSliceToDense(size_t, size_t) const;
template GPUSparseMatrix<short>& GPUSparseMatrix<short>::operator=(GPUSparseMatrix<short>&&);
template void GPUSparseMatrix<short>::Reshape(const size_t, const size_t);
template void GPUSparseMatrix<short>::ScaleAndAdd(short, GPUSparseMatrix<short> const &, GPUMatrix<short> &);

template GPUSparseMatrix<int>::GPUSparseMatrix(DEVICEID_TYPE, const MatrixFormat);
template GPUSparseMatrix<int>::~GPUSparseMatrix();
template void GPUSparseMatrix<int>::RequireSizeAndAllocate(const size_t, const size_t, const size_t, const bool, const bool);

template <class ElemType>
MATH_API File& operator>>(File& stream, GPUSparseMatrix<ElemType>& us)
{
    us.VerifyWritable(__FUNCTION__);

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

    us.SetFormat((MatrixFormat) format);
    if (us.GetFormat() != matrixFormatSparseCSC && us.GetFormat() != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    us.RequireSizeAndAllocate(rownum, colnum, nz, true, false);

    if (nz > 0)
    {
        size_t compressedSize = (us.GetFormat() == matrixFormatSparseCSC) ? colnum + 1 : rownum + 1;
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

        if (us.GetFormat() == matrixFormatSparseCSC)
            us.SetMatrixFromCSCFormat(compressedIndex, unCompressedIndex, dataBuffer, nz, rownum, colnum);
        else if (us.GetFormat() == matrixFormatSparseCSR)
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
    if (us.GetFormat() != matrixFormatSparseCSC && us.GetFormat() != matrixFormatSparseCSR)
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

        if (us.GetFormat() == matrixFormatSparseCSC)
            us.GetMatrixFromCSCFormat(compressedIndex, unCompressedIndex, dataBuffer, numElemAllocated, nz, numRows, numCols);
        else if (us.GetFormat() == matrixFormatSparseCSR)
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
