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
    UpdateCachedNzCount(0);
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

    // If the source is a slice, then this copy is only the content of the slice.
    RequireSizeAndAllocate(deepCopy.GetNumRows(), deepCopy.GetNumCols(), deepCopy.GetNumNZElements(), deepCopy.GetFormat(), true, false);
    m_sliceViewOffset = 0; // reset to zero as we only start copying the indices starting from the offset in the source matrix

    // BUGBUG? I suspect Data() here should be Buffer() for CSC, although Data() is the same because m_sliceViewOffset == 0
    CUDA_CALL(cudaMemcpy(Data_IThinkThisShouldBeBuffer(), deepCopy.NzValues(), deepCopy.NzBytes(), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(MajorIndexLocation(), deepCopy.MajorIndexLocationWithSliceViewOffset(), deepCopy.MajorIndexSize(), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(SecondaryIndexLocation(), deepCopy.SecondaryIndexLocation(), deepCopy.SecondaryIndexSize(), cudaMemcpyDeviceToDevice));

    // When slicing not from the start, the offset array must be updated.
    if (deepCopy.m_sliceViewOffset > 0)
    {
        int blocksPerGrid = (int) ceil(1.0 * SecondaryIndexCount() / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        _shiftColCSCIndexFromSliceViewToAbsolute<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            SecondaryIndexLocation(),
            SecondaryIndexCount(),
            GetNumNZElements());
    }

    UpdateCachedNzCount(deepCopy.NzCount()); // in case of a slice, the sources NZCount already reflects the count of the slice
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
        // BUGBUG: Does this handle slice-view offset correctly? We should only copy parts.
        SetMatrixFromCSRFormat(deepCopy.RowLocation(), deepCopy.ColLocation(), deepCopy.Data(), deepCopy.GetNumElemAllocated(), deepCopy.GetNumRows(), deepCopy.GetNumCols());
    }
    else if (deepCopy.GetFormat() == matrixFormatSparseCSC)
    {
        // BUGBUG: Does this handle slice-view offset correctly? We should only copy parts.
        SetMatrixFromCSCFormat(deepCopy.ColLocation(), deepCopy.RowLocation(), deepCopy.Data(), deepCopy.GetNumElemAllocated(), deepCopy.GetNumRows(), deepCopy.GetNumCols());
    }
    else if (deepCopy.GetFormat() == matrixFormatSparseBlockCol)
    {
        SetMatrixFromSBCFormat(deepCopy.BlockIdsLocation(), deepCopy.Data(), deepCopy.GetBlockSize(), deepCopy.GetNumRows(), deepCopy.GetNumCols());
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
            // BUGBUG: Should this be RowLocationWithSliceViewOffset()?
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

        CUDA_CALL(cudaMemcpy(cpuSparseMatrix.Data(), Data_IThinkThisShouldBeBuffer(), GetSizeElemAllocated(), cudaMemcpyDeviceToHost));
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

        CUDA_CALL(cudaMemcpy(cpuSparseMatrix.Data(), Data(), NzBytes(), cudaMemcpyDeviceToHost));
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
    denseMatrix.RequireSize(GetNumRows(), GetNumCols());

    SyncGuard syncGuard;
    if (GetFormat() == MatrixFormat::matrixFormatSparseCSR || GetFormat() == MatrixFormat::matrixFormatSparseCSC)
    {
        cusparseHandle_t cusparseHandle = 0;
        CUSPARSE_CALL(cusparseCreate(&cusparseHandle));
        cusparseMatDescr_t descr = 0;
        CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        CUSPARSE_CALL(cusparseSetStream(cusparseHandle, t_stream));

        if (GetFormat() == MatrixFormat::matrixFormatSparseCSR)
        {
            if (sizeof(ElemType) == sizeof(float))
                CUSPARSE_CALL(cusparseScsr2dense(cusparseHandle, int(GetNumRows()), int(GetNumCols()), descr, (float*)Buffer(), RowLocation(), ColLocation(), (float*)denseMatrix.Data(), int(GetNumRows())));
            else
                CUSPARSE_CALL(cusparseDcsr2dense(cusparseHandle, int(GetNumRows()), int(GetNumCols()), descr, (double*)Buffer(), RowLocation(), ColLocation(), (double*)denseMatrix.Data(), int(GetNumRows())));
        }
        else
        {
            if (sizeof(ElemType) == sizeof(float))
                CUSPARSE_CALL(cusparseScsc2dense(cusparseHandle, int(GetNumRows()), int(GetNumCols()), descr, (float*)Buffer(), RowLocation(), ColLocation(), (float*)denseMatrix.Data(), int(GetNumRows())));
            else
                CUSPARSE_CALL(cusparseDcsc2dense(cusparseHandle, int(GetNumRows()), int(GetNumCols()), descr, (double*)Buffer(), RowLocation(), ColLocation(), (double*)denseMatrix.Data(), int(GetNumRows())));
        }
        CUSPARSE_CALL(cusparseDestroy(cusparseHandle));
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
}

// if the matrix contains strictly one-hot data, then return a vector of the indices; otherwise NULL
template <class ElemType>
size_t* GPUSparseMatrix<ElemType>::TryCopyToArrayAsOneHot() const
{
    if (GetFormat() != MatrixFormat::matrixFormatSparseCSC) // only CSC format for now
        return nullptr;
    let n = GetNumCols();
    if (NzCount() != n) // if not, we know it is not one-hot
        return nullptr;
    // all values must be 1
    vector<ElemType> valBuf(n);
    CUDA_CALL(cudaMemcpy(valBuf.data(), Data(), valBuf.size() * sizeof(*valBuf.data()), cudaMemcpyDeviceToHost)); // Data() includes slice-view offset
    if (any_of(valBuf.begin(), valBuf.end(), [](ElemType val) { return val != 1; }))
        return nullptr;
    // each column must contain exactly one element
    vector<GPUSPARSE_INDEX_TYPE> secondaryIndexBuf(n+1);
    CUDA_CALL(cudaMemcpy(secondaryIndexBuf.data(), SecondaryIndexLocation(), secondaryIndexBuf.size() * sizeof(*secondaryIndexBuf.data()), cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < n; j++)
        if (secondaryIndexBuf[j + 1] != secondaryIndexBuf[j] + 1)
            return nullptr;
    // OK! We can get the array now
    vector<GPUSPARSE_INDEX_TYPE> majorIndexBuf(n);
    CUDA_CALL(cudaMemcpy(majorIndexBuf.data(), MajorIndexLocationWithSliceViewOffset(), majorIndexBuf.size() * sizeof(*majorIndexBuf.data()), cudaMemcpyDeviceToHost)); // note: includes slice-view offset
    unique_ptr<size_t[]> res(new size_t[n]);
    for (size_t j = 0; j < n; j++)
        res[j] = majorIndexBuf[j];
    return res.release();
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
            // BUGBUG? I suspect Data() here should be Buffer().
            CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, int(GetNumRows()), int(GetNumCols()), int(GetSizeAllocated()),
                                           (float*) Data_IThinkThisShouldBeBuffer(), RowLocation(), ColLocation(), (float*) outMatrix.Data_IThinkThisShouldBeBuffer(),
                                           outMatrix.RowLocation(), outMatrix.ColLocation(), CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));
        }
        else
        {
            // BUGBUG? I suspect Data() here should be Buffer().
            CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, int(GetNumRows()), int(GetNumCols()), int(GetSizeAllocated()),
                                           (double*) Data_IThinkThisShouldBeBuffer(), RowLocation(), ColLocation(), (double*) outMatrix.Data_IThinkThisShouldBeBuffer(),
                                           outMatrix.RowLocation(), outMatrix.ColLocation(), CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));
        }
        InvalidateCachedNzCount();
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

#ifdef WIN32
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
            CUDA_CALL(cudaMemcpyPeer(d_dst, to_id, Buffer(), GetComputeDeviceId(), BufferSizeAllocated()));
        }
        else
#endif
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

// set value from a dense matrix
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
            // BUGBUG? I suspect Data() here should be Buffer().
            CUSPARSE_CALL(cusparseSdense2csr(cusparseHandle, (int) GetNumRows(), (int) GetNumCols(), descr, reinterpret_cast<float*>(denseMatrix.Data()),
                                             (int) GetNumRows(), nnzPerRowOrCol, reinterpret_cast<float*>(Data_IThinkThisShouldBeBuffer()), RowLocation(), ColLocation()));
        }
        else
        {
            // BUGBUG? I suspect Data() here should be Buffer().
            CUSPARSE_CALL(cusparseDdense2csr(cusparseHandle, (int) GetNumRows(), (int) GetNumCols(), descr, reinterpret_cast<double*>(denseMatrix.Data()),
                                             (int) GetNumRows(), nnzPerRowOrCol, reinterpret_cast<double*>(Data_IThinkThisShouldBeBuffer()), RowLocation(), ColLocation()));
        }
    }
    else if (GetFormat() == MatrixFormat::matrixFormatSparseCSC)
    {
        if (sizeof(ElemType) == sizeof(float))
        {
            // BUGBUG? I suspect Data() here should be Buffer().
            CUSPARSE_CALL(cusparseSdense2csc(cusparseHandle, (int) GetNumRows(), (int) GetNumCols(), descr, reinterpret_cast<float*>(denseMatrix.Data()),
                                             (int) GetNumRows(), nnzPerRowOrCol, reinterpret_cast<float*>(Data_IThinkThisShouldBeBuffer()), RowLocation(), ColLocation()));
        }
        else
        {
            // BUGBUG? I suspect Data() here should be Buffer().
            CUSPARSE_CALL(cusparseDdense2csc(cusparseHandle, (int) GetNumRows(), (int) GetNumCols(), descr, reinterpret_cast<double*>(denseMatrix.Data()),
                                             (int) GetNumRows(), nnzPerRowOrCol, reinterpret_cast<double*>(Data_IThinkThisShouldBeBuffer()), RowLocation(), ColLocation()));
        }
    }
    UpdateCachedNzCount(nnzTotalDevHostPtr);
}

///
/// adjusts the sparse block column matrix with the new Col2BlockId
/// For each column, if new Col2BlockId contains valid index, a corresponding block exists at the index
/// if old col2BlockId[i] contains value at that column, it would be copied over; otherwise the block would be filled with zeros
///
template <class ElemType>
void GPUSparseMatrix<ElemType>::AdjustCol2BlockId(const GPUSPARSE_INDEX_TYPE* cpuCol2BlockId, size_t numBlocks, bool useBlockId2Col)
{
    if (GetFormat() != MatrixFormat::matrixFormatSparseBlockCol)
        LogicError("Expected sparse block col matrix");

    // create new buffer
    size_t numRows = GetNumRows();
    size_t numCols = GetNumCols();
    size_t numNZ = numBlocks * numRows;
    size_t bufferSizeNeeded = BufferSizeNeeded(GetNumRows(), GetNumCols(), numNZ, GetFormat());
    ElemType* pArray = reinterpret_cast<ElemType*>(TracingGPUMemoryAllocator::Allocate<char>(GetComputeDeviceId(), bufferSizeNeeded));
    GPUSPARSE_INDEX_TYPE* newBlockId2Col = (GPUSPARSE_INDEX_TYPE*)(pArray + numNZ);
    GPUSPARSE_INDEX_TYPE* newCol2BlockId = newBlockId2Col + numCols;

    CUDA_CALL(cudaMemset(newBlockId2Col, SparseIndex_NotAssigned, numCols * sizeof(GPUSPARSE_INDEX_TYPE)));
    CUDA_CALL(cudaMemcpy(newCol2BlockId, cpuCol2BlockId, numCols * sizeof(GPUSPARSE_INDEX_TYPE), cudaMemcpyHostToDevice));

    int blocksPerGrid = CeilDiv(numCols, GridDim::maxThreadsPerBlock);
 
    // when useBlockId2Col==true, the original col2BlockId is copied to blockId2Col to avoid getting overwritten 
    // during the inplace aggregation of col2BlockId prior to this
    _adjustCol2BlockId<ElemType> << <blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream >> > (
        numRows,
        numCols,
        useBlockId2Col ? BlockId2ColOrRow() : ColOrRow2BlockId(),
        Data(),
        newCol2BlockId,
        pArray,
        newBlockId2Col);

    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), Buffer());

    SetBuffer(pArray, bufferSizeNeeded);
    SetSizeAllocated(numNZ);
    SetBlockSize(numBlocks);
}

// fetch the CSC-column/CSR-row offset array from the GPU to the CPU
// Returns a pointer that the caller must delete[].
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
void GPUSparseMatrix<ElemType>::MaskColumnsValue(const GPUMatrix<char>& columnsMask, ElemType val, size_t numColsPerMaskEntry)
{
    VerifyWritable(__FUNCTION__);

    if (GetNumCols() != (columnsMask.GetNumCols() * numColsPerMaskEntry))
        RuntimeError("Matrix number of columns must equal 'number of columns in column mask * numColsPerMaskEntry'.");

    if (val != 0)
        LogicError("MaskColumnsValue is not implmented for a non-zero mask for sparse matrices.");

    // We are already done, since the gaps already contain zero by definition.

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
        size_t n = columnsMask.GetNumCols();
        #pragma omp parallel for
        for (long j = 0; j < n; j++)
            for (long k = 0; k < numColsPerMaskEntry; ++k)
                if (maskedCols[j] == 0 && colVector[(j * numColsPerMaskEntry) + k + 1] != colVector[(j * numColsPerMaskEntry) + k])
                    RuntimeError("GPUSparseMatrix attempted to mask column %d, but it has %d elements in it.", (int)((j * numColsPerMaskEntry) + k), (int)(colVector[(j * numColsPerMaskEntry) + k + 1] - colVector[(j * numColsPerMaskEntry) + k]));
    }
    else
        NOT_IMPLEMENTED;
#endif
}

// assignment is deep
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
    ZeroValues(); // TODO: why is this necessary?
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
    UpdateCachedNzCount(a.NzCount());
}

//-------------------------------------------------------------------------
// main operations
//-------------------------------------------------------------------------

// unlike dense matrices, Reshape() is involved for sparse
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
        // BUGBUG? I suspect Data() here should be Buffer().
        CUDA_CALL(cudaMemcpy(pArray, Data_IThinkThisShouldBeBuffer(), GetSizeElemAllocated(), cudaMemcpyDeviceToDevice));

        GPUSPARSE_INDEX_TYPE* majorIndexInNewBuffer = (GPUSPARSE_INDEX_TYPE*) (pArray + GetSizeAllocated());
        GPUSPARSE_INDEX_TYPE* secondaryIndexInNewBuffer = majorIndexInNewBuffer + MajorIndexCount(numRows, numCols, GetSizeAllocated(), GetFormat());

        int blocksPerGrid = (int) ceil(1.0 * numCols / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        // update the indices to represent the reshaping operation (element values remain unchanged)
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

// Reserves space for numNZElemToReserve non-zero elements. Also verifies that the matrix is indeed [numRows x numCols].
// If keepExistingValues then the object is assumed already in valid state. This is currently only used for MultiplyAndAdd() for SBC format.
// If not keepExistingValues, the memory is 0-initialized.
template <class ElemType>
void GPUSparseMatrix<ElemType>::Allocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const bool growOnly, bool keepExistingValues)
{
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
        CUDA_CALL(cudaMemsetAsync(pArray, 0, bufferSizeNeeded, t_stream));
        if (Buffer() != nullptr)
        {
            if (keepExistingValues)
            {
                if (NzCount() > numNZElemToReserve || BufferSizeAllocated() > bufferSizeNeeded)
                    LogicError("Allocate: To keep values, m_nz should <= numNZElemToReserve.");

                // BUGBUG? I suspect Data() here should be Buffer().
                CUDA_CALL(cudaMemcpyAsync(pArray, Data_IThinkThisShouldBeBuffer(), GetSizeElemAllocated(), cudaMemcpyDeviceToDevice, t_stream));

                GPUSPARSE_INDEX_TYPE* majorIndexInNewBuffer = (GPUSPARSE_INDEX_TYPE*) (pArray + numNZElemToReserve);

                CUDA_CALL(cudaMemcpyAsync(majorIndexInNewBuffer, MajorIndexLocation(), MajorIndexSize(), cudaMemcpyDeviceToDevice, t_stream));

                GPUSPARSE_INDEX_TYPE* secondaryIndexInNewBuffer = majorIndexInNewBuffer + MajorIndexCount(numRows, numCols, numNZElemToReserve, GetFormat());
                CUDA_CALL(cudaMemcpyAsync(secondaryIndexInNewBuffer, SecondaryIndexLocation(), SecondaryIndexSize(), cudaMemcpyDeviceToDevice, t_stream));
            }
            TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), Buffer());
        }

        SetBuffer(pArray, bufferSizeNeeded);
        SetSizeAllocated(numNZElemToReserve);
        if (!keepExistingValues)
            UpdateCachedNzCount(0);
    }
    else
    {
        SetSizeAllocated(ElemCountFromBufferSize(numRows, numCols, GetFormat(), BufferSizeAllocated()));
        // if requested size is smaller, make sure we still initialize to 0 as if it had been reallocated
        if (!keepExistingValues)
            CUDA_CALL(cudaMemsetAsync(Buffer(), 0, BufferSizeAllocated(), t_stream));
        UpdateCachedNzCount(0, /*shouldVerify=*/false);
    }
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::RequireSizeAndAllocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve /*= 10000*/, const bool growOnly /*= true*/, bool keepExistingValues /*= false*/)
{
    RequireSizeAndAllocate(numRows, numCols, numNZElemToReserve, GetFormat(), growOnly, keepExistingValues);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::RequireSizeAndAllocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly, bool keepExistingValues)
{
    RequireSize(numRows, numCols, numNZElemToReserve, matrixFormat, growOnly); // (does nothing if type and numRows/numCols already match, irrespective of numNZElemToReserve)

    if (matrixFormat != GetFormat())
        LogicError("RequireSizeAndAllocate: matrixFormat not set?");

    // this test is redundant; we only short-circuit a comparison of dimensions
    //size_t bufferSizeNeeded = BufferSizeNeeded(numRows, numCols, numNZElemToReserve, matrixFormat);
    //bool reallocate = (BufferSizeAllocated() < bufferSizeNeeded || (!growOnly && BufferSizeAllocated() > bufferSizeNeeded));
    //
    //if (reallocate)
        Allocate(numRows, numCols, numNZElemToReserve, growOnly, keepExistingValues);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::RequireSize(const size_t numRows, const size_t numCols, const bool growOnly /*= true*/)
{
    RequireSize(numRows, numCols, GetFormat(), growOnly);
}

template <class ElemType>
void GPUSparseMatrix<ElemType>::RequireSize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly /*= true*/)
{
    if (GetFormat() != matrixFormat || GetNumRows() != numRows || GetNumCols() != numCols)
        Resize(numRows, numCols, numNZElemToReserve, matrixFormat, growOnly);
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
        Allocate(numRows, numCols, numNZElemToReserve, growOnly, /*keepExistingValues=*/false);
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
    //    1. We must clear the secondary column index.    --TODO: Why? It should be considered virgin memory when reused!
    //    2. Set the block size to 0.
    // These requirements can be deduced by the NzCount method.
    CUDA_CALL(cudaMemsetAsync(Buffer(), 0, BufferSizeAllocated(), t_stream));
    SetBlockSize(0);
    UpdateCachedNzCount(0, /*shouldVerify=*/false);
}

// copy features to GPU
// TODO: This function should be near-identical to SetMatrixFromCSCFormat(), but SetMatrixFromCSCFormat() has been updated. Merge these.
template <class ElemType>
void GPUSparseMatrix<ElemType>::SetMatrixFromCSRFormat(const GPUSPARSE_INDEX_TYPE* h_CSRRow, const GPUSPARSE_INDEX_TYPE* h_Col, const ElemType* h_Val,
                                                       const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice /*= false*/, const DEVICEID_TYPE devId /*= -1*/)
{
    VerifyWritable(__FUNCTION__);

    if (h_CSRRow == nullptr || h_Col == nullptr || h_Val == nullptr)
        LogicError("SetMatrixFromCSRFormat: nullptr passed in.");
    if (!IsOnDevice && nz != h_CSRRow[numRows] - h_CSRRow[0])
        LogicError("SetMatrixFromCSRFormat: wrong nz value passed in.");

    SetComputeDeviceId(PrepareDevice(devId));

    SetFormat(matrixFormatSparseCSR);
    RequireSizeAndAllocate(numRows, numCols, nz, true, false);

    cudaMemcpyKind kind = IsOnDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    // BUGBUG? I suspect Data() here should be Buffer().
    CUDA_CALL(cudaMemcpy(Data_IThinkThisShouldBeBuffer(), h_Val, nz * sizeof(ElemType), kind));

    if (sizeof(CPUSPARSE_INDEX_TYPE) == sizeof(GPUSPARSE_INDEX_TYPE))
    {
        // ColSize doesn't work since it requires NzCount() to be usable (RowSize doesn't, since it's the fixed, compressed,
        // dimension. Since NzCount is not available (because the sparse indices which is where the NzCount is computed from
        // haven't been copied in yet), we just tell it how many bytes to copy. That is, nz * sizeof(GPUSPARSE_INDEX_TYPE);
        CUDA_CALL(cudaMemcpy(RowLocation(), h_CSRRow, RowSize(), kind));
        CUDA_CALL(cudaMemcpy(ColLocation(), h_Col, nz * sizeof(GPUSPARSE_INDEX_TYPE), kind));
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
    UpdateCachedNzCount(nz, IsOnDevice); // (when coming from CPU, nz was already validated)
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
        // BUGBUG? I suspect Data() here should be Buffer(), and/or slice view offset should be 0.
        CUDA_CALL(cudaMemcpy(h_Val, Data_IThinkThisShouldBeBuffer(), GetSizeElemAllocated(), cudaMemcpyDeviceToHost));

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

// Set the matrix to the data given by the three arrays, copying the data to the GPU.
// this version is used from the reader
template <class ElemType>
void GPUSparseMatrix<ElemType>::SetMatrixFromCSCFormat(
    const CPUSPARSE_INDEX_TYPE* h_CSCCol, // [0..numCols-1] starting index into h_Row
    const CPUSPARSE_INDEX_TYPE* h_Row,    // [0..nz-1] row of value, order matches h_Val
    const ElemType* h_Val,                // [0..nz-1] values
    const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice /*= false*/, const DEVICEID_TYPE devId /*= -1*/, DataTransferer* transferer /*= nullptr*/)
{
    VerifyWritable(__FUNCTION__);

    if (h_CSCCol == nullptr || h_Row == nullptr || h_Val == nullptr)
        LogicError("SetMatrixFromCSCFormat: nullptr passed in.");
    if (!IsOnDevice && nz != h_CSCCol[numCols] - h_CSCCol[0])
        LogicError("SetMatrixFromCSCFormat: wrong nz value passed in.");
#if 0 // validate input indices
    if (!IsOnDevice)
    {
        for (size_t j = 0; j < numCols; j++)
        {
            if (h_CSCCol[j] < 0 || h_CSCCol[j] > nz)
                LogicError("SetMatrixFromCSCFormat: h_CSCCol[colIndex=%d] beyond nz=%d", (int)j, (int)nz);
            if (j > 0 && h_CSCCol[j] < h_CSCCol[j - 1])
                LogicError("SetMatrixFromCSCFormat: h_CSCCol[] not in ascending order, %d, %d", (int)h_CSCCol[j - 1], (int)h_CSCCol[j]);
        }
        for (size_t k = 0; k < nz; k++)
            if (h_Row[k] < 0 || h_Row[k] >= numRows)
                LogicError("SetMatrixFromCSCFormat: row index of nz element [%d] out of bounds (%d >= %d)", (int)k, (int)h_Row[k], (int)numRows);
    }
#endif

    SetComputeDeviceId(PrepareDevice(devId));
    SetFormat(matrixFormatSparseCSC);
    RequireSizeAndAllocate(numRows, numCols, nz, /*growOnly=*/true, /*keepExistingValues=*/false);

    if (transferer && IsOnDevice)
        RuntimeError("SetMatrixFromCSCFormat: Currently it is not supported to copy data asynchronously from device to device.");
    // m_nz doesn't exist anymore. How are we going to deal with the NzBytes, RowSize, and ColSize? Do it ourselves of course.

    // copy the non-zero elements
    cudaMemcpyKind kind = IsOnDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    if (transferer)
    {
        // TODO: All RequireSizeAndAllocate should be async and use a transferer.
        // Currently there are some memset operations that can be still executing on the default stream,
        // Here we have to wait for them to finish.
        transferer->RecordComputeStreamSyncPoint();
        transferer->WaitForSyncPointOnAssignStreamAsync();
        // BUGBUG? I suspect Data() here should be Buffer(), and/or slice view offset should be 0.
        transferer->CopyCPUToGPUAsync(h_Val, nz, sizeof(ElemType), Data_IThinkThisShouldBeBuffer());
    }
    else
        // BUGBUG? I suspect Data() here should be Buffer(), and/or slice view offset should be 0.
        CUDA_CALL(cudaMemcpy(Data_IThinkThisShouldBeBuffer(), h_Val, nz * sizeof(ElemType), kind));

    // copy the index arrays
    if (sizeof(CPUSPARSE_INDEX_TYPE) == sizeof(GPUSPARSE_INDEX_TYPE)) // note: this is true
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
    else // TODO: is this branch needed, or can it just throw a logic_error?
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

    // TODO: When coming from the CPU, we can check whether the data is one-hot; and pass that to UpdateCachedNZCount() as well.

    UpdateCachedNzCount(nz, IsOnDevice && !transferer); // (when coming from CPU, nz was already validated)
}

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

    static std::vector<GPUSPARSE_INDEX_TYPE> gpuBlockId2Col(numCols);
    static std::vector<GPUSPARSE_INDEX_TYPE> gpuCol2BlockId(numCols);

    std::fill(gpuBlockId2Col.begin(), gpuBlockId2Col.end(), SparseIndex_NotAssigned);
    std::fill(gpuCol2BlockId.begin(), gpuCol2BlockId.end(), SparseIndex_NotAssigned);

    #pragma omp parallel for
    for (int i = 0; i < numBlocks; ++i)
    {
        gpuBlockId2Col[i] = (GPUSPARSE_INDEX_TYPE)blockIds[i];
        gpuCol2BlockId[blockIds[i]] = i;
    }

    CUDA_CALL(cudaMemcpy(Data(), val, nz * sizeof(ElemType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(BlockId2ColOrRow(), &gpuBlockId2Col[0], numCols * sizeof(GPUSPARSE_INDEX_TYPE), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(ColOrRow2BlockId(), &gpuCol2BlockId[0], numCols * sizeof(GPUSPARSE_INDEX_TYPE), cudaMemcpyHostToDevice));
    InvalidateCachedNzCount(); // (for SBC, it is cheap to recover NzCount)
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
        h_CSCCol = new GPUSPARSE_INDEX_TYPE[GetNumRows() + 1];
        h_Row = new GPUSPARSE_INDEX_TYPE[nz];

        PrepareDevice();
        // BUGBUG? I suspect Data() here should be Buffer(), and/or slice view offset should be 0.
        CUDA_CALL(cudaMemcpy(h_Val, Data_IThinkThisShouldBeBuffer(), GetSizeElemAllocated(), cudaMemcpyDeviceToHost));

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
// This is e.g. used for the forward pass of an embedding (e = E w where w is one-hot).
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
        // this is the code branch for embedding from sparse input
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
// This is called from MultiplyAndWeightedAdd() for the forward pass of an embedding (e = E w where w is one-hot), with numChannels=1, no subsampling, no padding, not channelwise.
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
            if (beta == 0.0)
                c.SetValue((ElemType)0);
            else if (beta != 1.0)
                RuntimeError("Only support c += alpha * a operation");

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

// c[:,j] = alpha * v[j] * a[:,j] + beta * c[:,j]
// -> dense
template <class ElemType>
void GPUSparseMatrix<ElemType>::ColumnwiseScaleAndWeightedAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& v, ElemType beta, GPUMatrix<ElemType>& c)
{
    if (v.GetNumRows() != 1 && v.GetNumCols() != 1)
        InvalidArgument("the argument v must be a vector"); // v is a vector

    if (a.GetFormat() != matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    if (beta == 0)
    {
        c.RequireSize(a.GetNumRows(), a.GetNumCols());
        c.SetValue((ElemType)0);
    }
    else
        c.VerifySize(a.GetNumRows(), a.GetNumCols()); // Can't resize if beta != 0

    int blocksPerGrid = (int)ceil(1.0 * a.GetNumCols() / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _columnwiseScaleAndWeightedAdd4CSC<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
        alpha,
        a.Buffer(), a.ColLocation(), a.RowLocation(),
        v.Data(),
        beta,
        c.Data(),
        a.GetNumRows(), a.GetNumCols());
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
        c.UpdateCachedNzCount(a.NzCount());
    }
    else
    {
        CUDA_CALL(cudaMemset(c.Buffer(), 0, c.BufferSizeAllocated()));
        c.UpdateCachedNzCount(0);
    }
}

// dense X sparse = sparse
// This is the backward pass from hidden layer to feature weight.
// E.g. e = E * w -> grad_E = grad_e * w'   (where w = CSC one-hot).
// In the one-hot case, this adds grad_e(t) to column w_index(t) of E.
template <class ElemType>
/*static*/ void GPUSparseMatrix<ElemType>::MultiplyAndAdd(ElemType alpha, const GPUMatrix<ElemType>& lhs, const bool transposeA,
                                                          const GPUSparseMatrix<ElemType>& rhs, const bool transposeB, GPUSparseMatrix<ElemType>& c)
{
    c.VerifyWritable(__FUNCTION__);

    if (lhs.GetComputeDeviceId() != rhs.GetComputeDeviceId())
        RuntimeError("GPUSparseMatrix::MultiplyAndAdd: All matrices must be on the same GPU");

    int m = transposeA ? (int) lhs.GetNumCols() : (int) lhs.GetNumRows(); // output dimension
    int k = transposeA ? (int) lhs.GetNumRows() : (int) lhs.GetNumCols(); // inner dimension (sparse gradient: == number of samples)
    int l = transposeB ? (int) rhs.GetNumCols() : (int) rhs.GetNumRows(); // inner dimension (== k required)
    int n = transposeB ? (int) rhs.GetNumRows() : (int) rhs.GetNumCols(); // input dimension

    assert(m > 0 && k > 0 && l > 0 && n > 0);
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
        // This is the backward pass from hidden layer to feature weight.
        if (rhs.GetFormat() != matrixFormatSparseCSC)
            NOT_IMPLEMENTED;

        c.SetFormat(matrixFormatSparseBlockCol);

        lhs.PrepareDevice();

        int blocksPerGrid = 0;
        SyncGuard syncGuard;

        // based on the size of m_nz in rhs and numCols in the resulted matrix we use different approaches
        size_t rhs_nz = rhs.NzCount();

        // Block col storage format (target matrix):
        //  - GetBlockSize()               :                  number of non-zero columns
        //  - ColOrRow2BlockId()[colIndex] : [numCols]        storage index (=index into the compact matrix), or SparseIndex_Pending if not determined yet, or SparseIndex_NotAssigned if empty
        //  - BlockId2ColOrRow()[blockId]  : [GetBlockSize()] column index (=logical index into the matrix that this object represents)
        //                                                     This array is allocated as numCols, but only elements up to GetBlockSize()-1 are used.
        // The storage indices can be in any order (they are not sorted).
        size_t blockSizePrev = c.GetBlockSize(); // number of non-zero columns in target matrix. Compact storage contains this many columns.
        if (blockSizePrev == 0)
        {
            //fprintf(stderr, "MultiplyAndAdd: resetting to %d items\n", (int)n), fflush(stderr);
            // the first time, we allocate space for all entries
            // Initially, all columns are empty. As we keep adding matrix products into it, columns
            // flip from empty to non-empty (but never back to empty).
            // This resetting is done lazily. Reset() just resets the block size, and this code here picks up on it and finishes the initialization.
            // Note that this may be expensive, as we initialize the full dimension (which is large, otherwise we wouldn't be using sparse).
            // We could speed that up by maintaining a dirty range, and only resetting that. Reset() could create a "lazy reset" instruction.
            c.Resize(m, n, 0);
            // Note a small hack: cudaMemset() sets bytes, but we initialize 32-bit ints. Hence, all bytes in SparseIndex_NotAssigned must be identical (0xff).
            static_assert(SparseIndex_NotAssigned == -1, "SparseIndex_NotAssigned must be 0xffffffff");
            if (n > c.GetNumCols())
                LogicError("MultiplyAndAdd: wrong allocation (primary and secondary indices)?");
            CUDA_CALL(cudaMemsetAsync(c.ColOrRow2BlockId(), 0xff, sizeof(GPUSPARSE_INDEX_TYPE) * n, t_stream));
            // PERF BUGBUG: BlockId2ColOrRow()[*] does not need to be initialized actually, does it?
            CUDA_CALL(cudaMemsetAsync(c.BlockId2ColOrRow(), 0xff, sizeof(GPUSPARSE_INDEX_TYPE) * n, t_stream));
        }

        // temp buffer to transfer a single value
        size_t* pBlockSizeTempGpu = TracingGPUMemoryAllocator::Allocate<size_t>(lhs.GetComputeDeviceId(), 1);
        // (perf note: we could use a kernel to set the value, to avoid the GPU sync; but below we copy it back, which cannot be avoided)
        CUDA_CALL(cudaMemcpyAsync(pBlockSizeTempGpu, &blockSizePrev, sizeof(size_t), cudaMemcpyHostToDevice, t_stream));
        // TODO: Can we avoid the memory allocation here?? Just keep around a bunch of general-use buffers?

        // determine which columns are non-zero -> ColOrRow2BlockId()[colIndex]
        // Some columns may already be non-zero. Those already have a storage index.
        // Columns that were zero before but are no longer get SparseIndex_Pending.
        // This is driven by rhs.RowLocation(); that is, the array of row indices of non-zero elements.
        blocksPerGrid = (int) ceil(((double) rhs_nz) / GridDim::maxThreadsPerBlock);
        _findColsWithValues<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            /*in*/rhs.RowLocation(), /*in ref*/rhs.ColLocation()[0], /*out*/c.ColOrRow2BlockId(), rhs_nz);
        // RowLocation = base of nz row-index array, without potential slice-view offset. Kernel offsets it by ColLocation()[0], which is non-zero in case of a slice view.
        // Now ColOrRow2BlockId()[colIndex] contains an index or SparseIndex_Pending for all non-empty columns.

        // assign a storage index to any newly added columns
        blocksPerGrid = (int) ceil(((double) n) / GridDim::maxThreadsPerBlock);
        _determineBlockIds<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            c.BlockId2ColOrRow(), c.ColOrRow2BlockId(), n, pBlockSizeTempGpu);
        // Now all SparseIndex_Pending values in ColOrRow2BlockId()[colIndex] have been replaced,
        // and BlockId2ColOrRow()[storageIndex] values for those have been placed.
        // *pBlockSizeTempGpu has been incremented accordingly.
        // BlockId2ColOrRow()[storageIndex] is now valid up to [*pBlockSizeTempGpu-1].
        // Newly added columns at this point contain a storage index that is out of bounds w.r.t. the compact storage.

        // Retrieve the updated #non-zero columns (*pBlockSizeTempGpu).
        // setting the block size incurs a GPU sync
        // Note: In the case of one-hot, we know an upper bound. We could leverage that to avoid the round-trip/GPU sync.
        // TODO: We could count the number of non-zero rows when transferring from the CPU.
        //       Should we just keep the CPU-side data around? In the one-hot case? Then we can do the mapping CPU-side.
        //       We can then even keep a CPU-side buffer in the weight matrix, for this purpose.
        // Or:
        //       We could also operate with an upper bound, which gets updated asynchronously (just fire off the async copy).
        //       We would then allocate w.r.t. the upper bound (=current + #new samples). At some point in time, the true,
        //       smaller, value would arrive asynchronously. With proper state tracking, we could avoid to unnecessarily
        //       initialize newly aded zero-columns, beyond the upper bound.
        size_t blockSizeCurr;
        CUDA_CALL(cudaMemcpy(&blockSizeCurr, pBlockSizeTempGpu, sizeof(size_t), cudaMemcpyDeviceToHost));
        TracingGPUMemoryAllocator::Free<size_t>(lhs.GetComputeDeviceId(), pBlockSizeTempGpu);
        c.SetBlockSize(blockSizeCurr);
        // Now GetBlockSize(), ColOrRow2BlockId()[*], and BlockId2ColOrRow()[*] are up to date.
        if (blockSizeCurr > c.GetNumCols())
            LogicError("MultiplyAndAdd: wrong allocation (block size)?");

        // if new storage columns have been added, zero them out (after growing the compact storage if needed)
        if (blockSizeCurr < blockSizePrev)
            LogicError("MultiplyAndAdd: #non-zero columns became less??");
        if (blockSizeCurr > blockSizePrev)
        {
            //fprintf(stderr, "MultiplyAndAdd: growing #non-zero columns from %d to %d, for %d items\n", (int)blockSizePrev, (int)blockSizeCurr, (int)k), fflush(stderr);
            // zero-initialize new blocks that were just added to block storage
            size_t nnz = m * blockSizeCurr;
            c.RequireSizeAndAllocate(m, n, nnz, true, true); // we need to keep the col2blockid and blockid2col info when resizing.
            CUDA_CALL(cudaMemsetAsync(c.Buffer() + m * blockSizePrev, 0, sizeof(ElemType) * m * (blockSizeCurr - blockSizePrev), t_stream));
        }
        // Now allocation is up-to-date as well.

        // now perform the actual matrix product, adding into the compact storage
        // This only processes the non-zero columns, which are already determined and passed in via ColOrRow2BlockId().
        LONG64 N = (LONG64) lhs.GetNumElements(); // =m*l, here we process for each row in lhs and each column in rhs (==columns in lhs)
        blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock); // elements of lhs linearly distributed over cores
        if (c.m_sliceViewOffset != 0)
            InvalidArgument("MultiplyAndAdd: Sparse block column matrices cannot be sliced.");
        _denseMulSparseCSCTransposeToSparseBlockCol2<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, t_stream>>>(
            alpha,
            // lhs (in)
            /*lhsValues=*/    lhs.Data(), // this is dense
            /*numRowsLhs=*/   m,          // output dimension, height of lhs
            /*numColsRhs=*/   l,          // inner dimension. In the case of the sparse gradient, this is the number of samples.
            // rhs (in)
            /*rhsNZValues=*/  rhs.Buffer(),      // [nzIndex] rhs nz-element array base, without potential slice-view offset.
            /*rhsRows=*/      rhs.RowLocation(), // [nzIndex] rhs index array base, without potential slice-view offset.
            /*rhsCols=*/      rhs.ColLocation(), // [colIndex] first nzIndex for a given column, with potential slice-view offset. End nzIndex is that of the next column.
            // result (out)
            /*col2blockIds=*/ c.ColOrRow2BlockId(), // (in) [colIndex] storage index for each non-zero column
            /*resultValues=*/ c.Buffer());          // (in/out) [rowIndex, storageIndex] pointer to compact storage

        c.InvalidateCachedNzCount(); // (the cached nzCount value is not used for block-sparse; nzCount = GetBlockSize() * numRows)
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

// -> dense
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
// Updates a dense matrix.
// TODO: this should be const.
template <class ElemType>
void GPUSparseMatrix<ElemType>::NormalGrad(GPUMatrix<ElemType>& c, const ElemType momentum, ElemType unitGainFactor)
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
            unitGainFactor);
    }
    else
    {
        NOT_IMPLEMENTED;
    }
}

// -> dense
// TODO: this should be const
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

// -> dense
// TODO: This should be const
template <class ElemType>
void GPUSparseMatrix<ElemType>::FSAdagrad(
    GPUMatrix<ElemType>& c,
    GPUMatrix<ElemType>& functionValues,
    ElemType learnRatePerSample,
    ElemType momentum,
    ElemType adaWeight,
    ElemType adaMul,
    ElemType unitGainFactor)
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
        learnRatePerSample, momentum, adaWeight, adaMul, unitGainFactor);
}

// -> dense
// TODO: This should be const
template <class ElemType>
void GPUSparseMatrix<ElemType>::Adam(
    GPUMatrix<ElemType>& c,
    GPUMatrix<ElemType>& functionValues,
    ElemType learnRatePerSample,
    ElemType momentum,
    ElemType adaWeight,
    ElemType adaMul,
    ElemType epsilon,
    ElemType unitGainFactor,
    bool adamax)
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
        learnRatePerSample, momentum, adaWeight, adaMul, epsilon, unitGainFactor, adamax);
}

// -> dense
// TODO: This should be const
template <class ElemType>
ElemType GPUSparseMatrix<ElemType>::RmsProp(GPUMatrix<ElemType>& c,
    ElemType RMS_GAMMA,
    ElemType RMS_WGT_INC,
    ElemType RMS_WGT_MAX,
    ElemType RMS_WGT_DEC,
    ElemType RMS_WGT_MIN,
    const bool needAveMultiplier,
    const bool initialized)
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

    if (c.IsEmpty() || c.GetNumCols() < numColsNeeded || !initialized)
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

// -> dense
// TODO: This should be const
template <class ElemType>
void GPUSparseMatrix<ElemType>::AdaDelta(GPUMatrix<ElemType>&c, GPUMatrix<ElemType>&functionValues, ElemType learningRate, ElemType rho, ElemType epsilon)
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
    _adadelta4BlockSparseCol<ElemType> << <blocksPerGrid, GridDim::maxThreadsPerBlock >> >(
        n, Data(), ColOrRow2BlockId(), GetNumRows(),
        c.Data(), c.Data() + n, functionValues.Data(),
        learningRate, rho, epsilon);
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
    // BUGBUG? I suspect Data() here should be Buffer().
    if (cSize >= rowBufferRequired && c.Data_IThinkThisShouldBeBuffer() != nullptr && canReuseBuffer)
    {
        // BUGBUG? I suspect Data() here should be Buffer().
        csrRowPtrC = (GPUSPARSE_INDEX_TYPE*) c.Data_IThinkThisShouldBeBuffer();
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
    VerifyCachedNzCount(nnzC); // (to be sure)

    // if we allocated the buffer, free it here
    if (allocatedBuffer)
        TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(GetComputeDeviceId(), csrRowPtrC);
}

// Multiply - multiply one sparse matrix by another sparse matrix
// S1 - first sparse matrix
// transposeS1 - transpose first matrix?
// S2 - second sparse matrix
// transposeS2 - tanspose second matrix?
// c - result matrix
// NOTE: if c has enough space allocated, it will be reused, otherwise it will be freed and a new memory block used
template <class ElemType>
void GPUSparseMatrix<ElemType>::Multiply(const GPUSparseMatrix<ElemType>& S1, bool transposeS1, const GPUSparseMatrix<ElemType>& S2, bool transposeS2, 
                                         GPUSparseMatrix<ElemType>& c)
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
                                       // BUGBUG? I suspect Data() here should be Buffer().
                                       descrC, (float*) c.Data_IThinkThisShouldBeBuffer(), c.RowLocation(), c.ColLocation()));
    }
    else
    {
        CUSPARSE_CALL(cusparseDcsrgemm(cusparseHandle, operA, operB, m, n, k, descrA, nnzA, (const double*) S1.Buffer(), S1.RowLocation(), S1.ColLocation(),
                                       descrB, nnzB, (const double*) S2.Buffer(), S2.RowLocation(), S2.ColLocation(),
                                       // BUGBUG? I suspect Data() here should be Buffer().
                                       descrC, (double*) c.Data_IThinkThisShouldBeBuffer(), c.RowLocation(), c.ColLocation()));
    }
    cusparseDestroy(cusparseHandle);
    c.VerifyCachedNzCount(c.NzCount()); // (to be sure)
}

template <class ElemType>
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignProductOf(const GPUSparseMatrix<ElemType>& a, const bool transposeA, const GPUSparseMatrix<ElemType>& b, const bool transposeB)
{
    Multiply(a, transposeA, b, transposeB, *this);
    return *this;
}

// sparse op sparse -> sparse
template <class ElemType>
void GPUSparseMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& a, ElemType beta, const GPUSparseMatrix<ElemType>& b, GPUSparseMatrix<ElemType>& c)
{
    if (a.GetFormat() != matrixFormatSparseCSR || b.GetFormat() != matrixFormatSparseCSR )
    {
        NOT_IMPLEMENTED;
    }
    if (c.m_sob.get() == nullptr)
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
        // BUGBUG? I suspect Data() here should be Buffer().
        CUSPARSE_CALL(cusparseScsrgeam(cusparseHandle, m, n, reinterpret_cast<const float*>(&alpha), descrA, nnzA, reinterpret_cast<const float*>(a.Data_IThinkThisShouldBeBuffer()), a.RowLocation(), a.ColLocation(),
                                       reinterpret_cast<const float*>(&beta), descrB, nnzB, reinterpret_cast<const float*>(b.Data_IThinkThisShouldBeBuffer()), b.RowLocation(), b.ColLocation(), descrC, reinterpret_cast<float*>(c.Data_IThinkThisShouldBeBuffer()), c.RowLocation(), c.ColLocation()));
    }
    else
    {
        // BUGBUG? I suspect Data() here should be Buffer().
        CUSPARSE_CALL(cusparseDcsrgeam(cusparseHandle, m, n, reinterpret_cast<const double*>(&alpha), descrA, nnzA, reinterpret_cast<const double*>(a.Data_IThinkThisShouldBeBuffer()), a.RowLocation(), a.ColLocation(),
                                       reinterpret_cast<const double*>(&beta), descrB, nnzB, reinterpret_cast<const double*>(b.Data_IThinkThisShouldBeBuffer()), b.RowLocation(), b.ColLocation(), descrC, reinterpret_cast<double*>(c.Data_IThinkThisShouldBeBuffer()), c.RowLocation(), c.ColLocation()));
    }
    cusparseDestroy(cusparseHandle);
    c.VerifyCachedNzCount(c.NzCount()); // (to be sure)
}

// sparse op dense -> dense
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
    // BUGBUG? I suspect a.Data() here should be Buffer().
    _sparseCSRPlusDense<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(alpha, a.Data_IThinkThisShouldBeBuffer(), a.RowLocation(), a.ColLocation(), c.Data(), M);
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
            // BUGBUG? I suspect Data() here should be Buffer().
            CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const float*>(a.Data_IThinkThisShouldBeBuffer()), a.RowLocation(), a.ColLocation(), reinterpret_cast<float*>(cscValA), cscRowIndA, cscColPtrA, cpVals, idxBase));
        }
        else
        {
            // BUGBUG? I suspect Data() here should be Buffer().
            CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const double*>(a.Data_IThinkThisShouldBeBuffer()), a.RowLocation(), a.ColLocation(), reinterpret_cast<double*>(cscValA), cscRowIndA, cscColPtrA, cpVals, idxBase));
        }
    }
    else if (a.GetFormat() == matrixFormatSparseCSC)
    {
        // BUGBUG? I suspect Data() here should be Buffer().
        cscValA = (ElemType*) a.Data_IThinkThisShouldBeBuffer();
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

// sparse op dense -> dense
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
        a.Buffer(), a.RowLocation(), a.ColLocation(),
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
    // BUGBUG? I suspect a.Data() here should be Buffer().
    _sparseCSRElemMulDense<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(a.Data_IThinkThisShouldBeBuffer(), a.RowLocation(), a.ColLocation(), b.Data(), c.Data(), M);
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
                // BUGBUG? I suspect Data() here should be Buffer().
                CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const float*>(Data_IThinkThisShouldBeBuffer()), RowLocation(), ColLocation(),
                                               reinterpret_cast<float*>(c.Data_IThinkThisShouldBeBuffer()), c.ColLocation(), c.RowLocation(), cpVals, idxBase));
            }
            else
            {
                // BUGBUG? I suspect Data() here should be Buffer().
                CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, m, n, nnz, reinterpret_cast<const double*>(Data_IThinkThisShouldBeBuffer()), RowLocation(), ColLocation(),
                                               reinterpret_cast<double*>(c.Data_IThinkThisShouldBeBuffer()), c.ColLocation(), c.RowLocation(), cpVals, idxBase));
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
                // BUGBUG? I suspect Data() here should be Buffer().
                CUSPARSE_CALL(cusparseScsr2csc(cusparseHandle, n, m, nnz, reinterpret_cast<const float*>(this->Data_IThinkThisShouldBeBuffer()), this->ColLocation(), this->RowLocation(),
                                               reinterpret_cast<float*>(c.Data_IThinkThisShouldBeBuffer()), c.RowLocation(), c.ColLocation(), cpVals, idxBase));
            }
            else
            {
                // BUGBUG? I suspect Data() here should be Buffer().
                CUSPARSE_CALL(cusparseDcsr2csc(cusparseHandle, n, m, nnz, reinterpret_cast<const double*>(this->Data_IThinkThisShouldBeBuffer()), this->ColLocation(), this->RowLocation(),
                                               reinterpret_cast<double*>(c.Data_IThinkThisShouldBeBuffer()), c.RowLocation(), c.ColLocation(), cpVals, idxBase));
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
    UpdateCachedNzCount(nnz);
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
    slice.m_sliceViewOffset = m_sliceViewOffset + startColumn; // Just shift the compressed index location to the new startColumn - that's it!
    if (startColumn == 0 && numCols == GetNumCols() && HasCachedNzCount())
        slice.UpdateCachedNzCount(NzCount());

    return slice;
}

// -> dense
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
    {
        if ((startColumn != 0) || (numCols != GetNumCols()))
            NOT_IMPLEMENTED;

        return CopyToDenseMatrix(slice);
    }

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

template<class ElemTypePtr> // ElemType* or const ElemType*
struct CSCSlice
{
    ElemTypePtr m_buffer;          // pointer to first array
    CUDA_LONG m_elemSizeAllocated; // pointer delta between the 3 arrays
    CUDA_LONG m_firstColumn;       // start of slice
    CUDA_LONG m_numColumns;        // width of slice
    // note: bad alignment, got 4 extra padding bytes free :(
};

template <size_t N, class ElemType>
__global__ void _gatherMemcpyCSC(const CSCSlice<ElemType*> outputSlice, const FixedSizeParameterArray<N, CSCSlice<const ElemType*>> inputSlices)
{
    // output data arrays
    auto* outputBuffer = outputSlice.m_buffer;
    let outputElemSizeAllocated = outputSlice.m_elemSizeAllocated;
    auto* outputRowIndices    = (CUDA_LONG*)(outputBuffer + outputElemSizeAllocated);
    auto* outputColumnOffsets = outputRowIndices          + outputElemSizeAllocated;
    // adjust for starting point
    auto* outputData = outputBuffer;
    auto jo = outputSlice.m_firstColumn;
    outputColumnOffsets += jo;
    if (jo == 0) // upon first call, the very first entry has not been initialized yet
        *outputColumnOffsets = 0;
    else // otherwise position m_firstColumn must already have been written by previous launch
    {
        let firstColumnOffset = *outputColumnOffsets; // was written during last launch
        outputData       += firstColumnOffset;
        outputRowIndices += firstColumnOffset;
    }
    // ready to write.
    // loop over input slices
    for (CUDA_LONG i = 0; i < inputSlices.size(); i++)
    {
        // get input pointers for this slice
        let& inputSlice = inputSlices[i];
        let* inputData = inputSlice.m_buffer;
        let inputElemSizeAllocated = inputSlice.m_elemSizeAllocated;
        let* inputRowIndices    = (CUDA_LONG*)(inputData + inputElemSizeAllocated);
        let* inputColumnOffsets = inputRowIndices        + inputElemSizeAllocated;
        let j0 =      inputSlice.m_firstColumn;
        let j1 = j0 + inputSlice.m_numColumns;
        auto columnOffset = inputColumnOffsets[j0];
        inputData       += columnOffset;
        inputRowIndices += columnOffset;
        // write column offsets
        auto* endOutputData = outputData;
        for (CUDA_LONG j = j0 + 1; j <= j1; j++)
        {
            let endColumnOffset = inputColumnOffsets[j];
            endOutputData += endColumnOffset - columnOffset;
            *++outputColumnOffsets = endOutputData - outputBuffer;
            columnOffset = endColumnOffset;
        }
        // copy values and row indices
        while (outputData < endOutputData)
        {
            *outputData++       = *inputData++;
            *outputRowIndices++ = *inputRowIndices++;
        }
    }
}

template <size_t N, class ElemType>
static void GatherMemcpyCSC(const CSCSlice<ElemType*>& outputSlice, const MaxFixedSizeParameterArray<CSCSlice<const ElemType*>>& inputSliceBuffer)
{
    let& inputSliceArray = (const FixedSizeParameterArray<N, CSCSlice<const ElemType*>>&)inputSliceBuffer;
    SyncGuard syncGuard;
    _gatherMemcpyCSC<N, ElemType> <<<1, 1, 0, t_stream>>>(outputSlice, inputSliceArray);
}

// GatherBatch() batches many independent inputs into one output tensor
// Only supports CSC format. Matrix must already have the output shape and correct type.
// This current implementation is not efficient for data other than one-hot.
template <class ElemType>
void GPUSparseMatrix<ElemType>::GatherBatch(size_t numInputs, const std::function<const GPUSparseMatrix<ElemType>&(size_t)>& inputs)
{
    if (GetFormat() != MatrixFormat::matrixFormatSparseCSC)
        InvalidArgument("GatherBatch (sparse): Requires CSC format.");
    // TODO: NzCount() is two GPU syncs! We should cache this value CPU-side.
    if (NzCount() != 0)
        InvalidArgument("GatherBatch (sparse): The target matrix cannot have pre-existing non-zero values when being gathered into.");
    // determine necessary allocation
    PrepareDevice();
    let numRows = GetNumRows();
    size_t numCols = 0;
    size_t nz = 0; // TODO: use an upper bound, to avoid GPU sync
    for (size_t i = 0; i < numInputs; i++)
    {
        let& input = inputs(i);
        if (input.GetFormat() != MatrixFormat::matrixFormatSparseCSC)
            InvalidArgument("GatherBatch (sparse): Requires CSC format.");
        if (input.GetNumRows() != numRows)
            InvalidArgument("GatherBatch (sparse): All inputs must have the same number of rows as the output (%d).", (int)numRows);
        let inputCols = input.GetNumCols();
        nz += 1;//input.NzCount(); // TODO: double-check that this does not actually read data from the GPU, that caching works
        numCols += inputCols;
    }
    if (numCols != GetNumCols())
        InvalidArgument("GatherBatch: Total number of input columns (%d) must be equal to number of output columns (%d).",
                        (int)numCols, (int)GetNumCols());
    // allocate
    RequireSizeAndAllocate(numRows, numCols, nz, /*growOnly=*/true, /*keepExistingValues=*/false);
    // process all inputs
    MaxFixedSizeParameterArray<CSCSlice<const ElemType*>> inputSliceBuffer;
    static constexpr size_t capacity = MaxFixedSizeParameterArray<CSCSlice<const ElemType*>>::CAPACITY;
    m_sliceViewOffset = 0;
    CSCSlice<ElemType*> outputSlice =
    {
        Buffer(), (CUDA_LONG)GetSizeAllocated(), (CUDA_LONG)m_sliceViewOffset, /*numCols=*/0
    };
    for (size_t i = 0; i < numInputs; i++)
    {
        let& input = inputs(i);
        let inCols = input.GetNumCols();
        if (inCols == 0)
            continue;
        inputSliceBuffer.push_back(CSCSlice<const ElemType*>
        {
            input.Buffer(), (CUDA_LONG)input.GetSizeAllocated(), (CUDA_LONG)input.m_sliceViewOffset, (CUDA_LONG)inCols
        });
        outputSlice.m_numColumns += inCols;
        if (inputSliceBuffer.size() == inputSliceBuffer.capacity())
        {
            // flush
            GatherMemcpyCSC<capacity, ElemType>(outputSlice, inputSliceBuffer);
            inputSliceBuffer.clear();
            // advance the output range column pointer
            outputSlice.m_firstColumn += outputSlice.m_numColumns;
            outputSlice.m_numColumns = 0;
        }
    }
    let colsLeft = inputSliceBuffer.size();
    if      (colsLeft == 0) {}
    else if (colsLeft <= 1)        GatherMemcpyCSC<       1, ElemType>(outputSlice, inputSliceBuffer);
    else if (colsLeft <= 2)        GatherMemcpyCSC<       2, ElemType>(outputSlice, inputSliceBuffer);
    else if (colsLeft <= 4)        GatherMemcpyCSC<       4, ElemType>(outputSlice, inputSliceBuffer);
    else if (colsLeft <= 8)        GatherMemcpyCSC<       8, ElemType>(outputSlice, inputSliceBuffer);
    else if (colsLeft <= 16)       GatherMemcpyCSC<      16, ElemType>(outputSlice, inputSliceBuffer);
    else if (colsLeft <= 24)       GatherMemcpyCSC<      24, ElemType>(outputSlice, inputSliceBuffer);
    else if (colsLeft <= 32)       GatherMemcpyCSC<      32, ElemType>(outputSlice, inputSliceBuffer);
    else if (colsLeft <= 48)       GatherMemcpyCSC<      48, ElemType>(outputSlice, inputSliceBuffer);
    else if (colsLeft <= 64)       GatherMemcpyCSC<      64, ElemType>(outputSlice, inputSliceBuffer);
    else if (colsLeft <= 96)       GatherMemcpyCSC<      96, ElemType>(outputSlice, inputSliceBuffer);
    else if (colsLeft <= 128)      GatherMemcpyCSC<     128, ElemType>(outputSlice, inputSliceBuffer);
    else if (colsLeft <= capacity) GatherMemcpyCSC<capacity, ElemType>(outputSlice, inputSliceBuffer);
    else LogicError("GatherBatch: We should have flushed inside the loop, but somehow didn't??");
    InvalidateCachedNzCount();
}

// -> dense
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

// -> scalar
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

// -> scalar
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

// sqrt(sum all elements^2) -> scalar
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

// -> scalar
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

// -> scalar
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
GPUSparseMatrix<ElemType>& GPUSparseMatrix<ElemType>::AssignOneHot(const GPUMatrix<ElemType>& a, vector<size_t>& shape, size_t axis)
{
    if (a.IsEmpty())
        LogicError("AssignOneHot: Matrix a is empty."); // BUGBUG: Just handle this gracefully.

    if (GetFormat() != matrixFormatSparseCSC)
        LogicError("AssignOneHot: Matrix format is not supported.");

    if (axis >= shape.size())
        LogicError("AssignOneHot: axis is not correct");

    int item_size = 1;
    for (size_t i = 0; i < shape.size() && i < axis; i++)
        item_size *= (int)shape[i];

    int num_class = (int)shape[axis];

    auto nRows = item_size * num_class;
    auto nCols = a.GetNumElements() / item_size;
    if (((GetNumRows() != 0) && (GetNumRows() != nRows)) || ((GetNumCols() != 0) && (GetNumCols() != nCols)))
        LogicError("AssignOneHot: Target matrix size is not correct");

    this->RequireSizeAndAllocate(nRows, nCols, a.GetNumElements());
    this->PrepareDevice();

    ElemType* indices = a.Data();
    GPUSPARSE_INDEX_TYPE* secondaryIndices = SecondaryIndexLocation();
    GPUSPARSE_INDEX_TYPE* majorIndices = MajorIndexLocation();
    ElemType* targetData = NzValues();
    CUDA_LONG N = (CUDA_LONG)a.GetNumElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    _assignOneHotAsSparse<ElemType><<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(indices, 
                                                                                    secondaryIndices,
                                                                                    majorIndices,
                                                                                    targetData,
                                                                                    num_class,
                                                                                    item_size,
                                                                                    N);

    UpdateCachedNzCount(a.GetNumElements());
    return *this;
}

// determines the row index of the row with the largest value in a column
template<class ElemType>
__global__ void _assignCSCArgmaxTo(ElemType *outData, CUDA_LONG numCols,
    const ElemType* nzValues,                 // base of nz-value array
    const GPUSPARSE_INDEX_TYPE* nzRowIndices, // base of corresponding row-index array
    const GPUSPARSE_INDEX_TYPE* colOffsets)   // array of offsets into nz array (including slice-view offset)
{
    // each thread processes one column, in a serial loop which is fine since this is meant for use with one-hot data
    const CUDA_LONG j = blockIdx.x;      // index of the column to process
    auto beginNZIndex = colOffsets[j];   // nz elements of this columns have this index range in the nz arrays
    auto endNZIndex   = colOffsets[j+1];
    ElemType bestVal = 0;   // (dummy)
    auto bestRowIndex = -1; // result for empty rows
    for (auto nzIndex = beginNZIndex; nzIndex != endNZIndex; nzIndex++)
    {
        if (bestRowIndex == -1 || bestVal < nzValues[nzIndex])
        {
            bestVal      = nzValues[nzIndex];
            bestRowIndex = nzRowIndices[nzIndex];
        }
    }
    outData[j] = (ElemType)bestRowIndex;
}

template <class ElemType>
/*static*/ void GPUSparseMatrix<ElemType>::AssignColumnwiseArgmaxTo(GPUMatrix<ElemType>& lhs, const GPUSparseMatrix<ElemType>& rhs)
{
    if (rhs.GetFormat() != matrixFormatSparseCSC)
        LogicError("AssignColumnwiseHardmaxTo: Argument must be in CSC format.");

    // output is a row vector
    let numCols = rhs.GetNumCols();
    lhs.Resize(1, numCols);

    // one thread per column (it's simple enough)
    SyncGuard syncGuard;
    if (numCols > 0)
        _assignCSCArgmaxTo<ElemType> <<<numCols, 1, 0, t_stream>>> (
            lhs.Data(), lhs.GetNumCols(), // target
            rhs.Buffer(),       // [nzIndex] rhs nz-element array base, without potential slice-view offset.
            rhs.RowLocation(),  // [nzIndex] rhs index array base, without potential slice-view offset.
            rhs.ColLocation()); // [colIndex] first nzIndex for rhs given column, with potential slice-view offset. End nzIndex is that of the next column.
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

// This is a memcpy() with built-in type cast.
// outBuffer should be allocated to be >= size by the caller
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
template void GPUSparseMatrix<char>::AssignColumnSliceToDense(GPUMatrix<char>&, size_t, size_t) const;
template GPUMatrix<char> GPUSparseMatrix<char>::CopyColumnSliceToDense(size_t, size_t) const;
template GPUSparseMatrix<char>& GPUSparseMatrix<char>::operator=(GPUSparseMatrix<char>&&);
template void GPUSparseMatrix<char>::Reshape(const size_t, const size_t);
template void GPUSparseMatrix<char>::ScaleAndAdd(char, GPUSparseMatrix<char> const &, GPUMatrix<char> &);
template void GPUSparseMatrix<char>::ColumnwiseScaleAndWeightedAdd(char, const GPUSparseMatrix<char>&, const GPUMatrix<char>&, char, GPUMatrix<char>&);

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
template void GPUSparseMatrix<short>::AssignColumnSliceToDense(GPUMatrix<short>&, size_t, size_t) const;
template GPUMatrix<short> GPUSparseMatrix<short>::CopyColumnSliceToDense(size_t, size_t) const;
template GPUSparseMatrix<short>& GPUSparseMatrix<short>::operator=(GPUSparseMatrix<short>&&);
template void GPUSparseMatrix<short>::Reshape(const size_t, const size_t);
template void GPUSparseMatrix<short>::ScaleAndAdd(short, GPUSparseMatrix<short> const &, GPUMatrix<short> &);
template void GPUSparseMatrix<short>::ColumnwiseScaleAndWeightedAdd(short, const GPUSparseMatrix<short>&, const GPUMatrix<short>&, short, GPUMatrix<short>&);

// Support <int>
template GPUSparseMatrix<int>::GPUSparseMatrix(DEVICEID_TYPE, const MatrixFormat);
template GPUSparseMatrix<int>::~GPUSparseMatrix();
template void GPUSparseMatrix<int>::RequireSizeAndAllocate(const size_t, const size_t, const size_t, const bool, const bool);
template void GPUSparseMatrix<int>::Reset();

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
