//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Math.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Basics.h"
#include "File.h"
#include <assert.h>
#include <stdexcept>
#include <omp.h>
#include <math.h>
#include "CPUMatrix.h"
#include "CPUSparseMatrix.h"
#include <random>
#include <chrono>
#include <iostream>
#ifdef LEAKDETECT
#include <vld.h>
#endif

#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this

#ifdef USE_MKL
// requires MKL 10.0 and above
#include <mkl.h>
#else
#ifdef _MSC_VER
// Visual Studio doesn't define standard complex types properly
#define HAVE_LAPACK_CONFIG_H
#define LAPACK_COMPLEX_STRUCTURE
#endif
#include <cblas.h>
#include <lapacke.h>
#endif

// This is an example of an exported variable
//MATH_API int nMath=0;

// This is an example of an exported function.
//MATH_API int fnMath(void)
//{
//    return 42;
//}

// TODO: Move to CommonMatrix.h
#define IDX2C(i, j, ld) (((j) * (ld)) + (i)) // 0 based indexing

namespace Microsoft { namespace MSR { namespace CNTK {

#pragma region Helpful Enum Definitions

enum class MatrixOrder
{
    RowMajor = 101, // row-major arrays
    ColMajor = 102  // column-major arrays
};

enum class MatrixTranspose : char
{
    NoTrans   = 'N', // trans='N'
    Trans     = 'T', // trans='T'
    ConjTrans = 'C'  // trans='C'
};

enum class SymMatrixType : char
{
    Up           = 'U', // symmetric matrix is stored in the upper part
    Low          = 'L', // symmetric matrix is stored in thelower part
    Full         = 'F', // full populated
    NotSymmetric = 'N'  // not a symmetric matrix
};

enum class MatrixOpSide : char
{
    Left  = 'L', // left multiply
    Right = 'R', // right multiply
};

#pragma endregion Helpful Enum Definitions

#pragma region Constructors and Destructor

//-------------------------------------------------------------------------
// construction and conversion
//-------------------------------------------------------------------------

// should only be used by constructors.
template <class ElemType>
/*private*/ void CPUSparseMatrix<ElemType>::ZeroInit()
{
    Base::ZeroInit();
    SetComputeDeviceId(CPUDEVICE);

    SetCompIndexSize(0);
    SetColIdx(-1);
    SetBuffer(nullptr, 0, false);
    SetUnCompIndex(nullptr);
    SetCompIndex(nullptr);
    SetBlockSize(0);
    SetBlockIdShift(0);
    SetBlockIds(nullptr);
}

//should only be used by constructors.
template <class ElemType>
void CPUSparseMatrix<ElemType>::CheckInit(const MatrixFormat format)
{
    if (format != MatrixFormat::matrixFormatSparseCSC && format != MatrixFormat::matrixFormatSparseCSR && format != MatrixFormat::matrixFormatSparseBlockCol && format != MatrixFormat::matrixFormatSparseBlockRow)
    {
        LogicError("CPUSparseMatrix:  unsupported sparse matrix format");
    }
    SetFormat(format);
    ZeroInit();
}

template <class ElemType>
CPUSparseMatrix<ElemType>::CPUSparseMatrix(const MatrixFormat format)
{
    CheckInit(format);
}

template <class ElemType>
CPUSparseMatrix<ElemType>::CPUSparseMatrix(const MatrixFormat format, const size_t numRows, const size_t numCols, const size_t size)
{
    CheckInit(format);
    RequireSizeAndAllocate(numRows, numCols, size, true, false);
}

// copy constructor, deep copy
template <class ElemType>
CPUSparseMatrix<ElemType>::CPUSparseMatrix(const CPUSparseMatrix<ElemType>& deepCopyFrom)
{
    ZeroInit();
    if (!deepCopyFrom.IsEmpty())
        SetValue(deepCopyFrom);
}

// assignment operator, deep copy
template <class ElemType>
CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::operator=(const CPUSparseMatrix<ElemType>& deepCopyFrom)
{
    if (!deepCopyFrom.IsEmpty())
        SetValue(deepCopyFrom);
    return *this;
}

// move constructor, shallow copy
template <class ElemType>
CPUSparseMatrix<ElemType>::CPUSparseMatrix(CPUSparseMatrix<ElemType>&& moveFrom)
{
    Base::ShallowCopyFrom(moveFrom);
    // release the pointer from the source object so that the destructor won't release it twice
    moveFrom.ZeroValues();
}

//move assignment operator, shallow copy
template <class ElemType>
CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::operator=(CPUSparseMatrix<ElemType>&& moveFrom)
{
    if (this != &moveFrom)
    {
        Base::ShallowCopyFrom(moveFrom);
        // release the pointer from the source object so that the destructor won't release it twice
        moveFrom.ZeroValues();
    }
    return *this;
}

template <class ElemType>
CPUSparseMatrix<ElemType>::~CPUSparseMatrix()
{
    ZeroValues();
}

#pragma endregion Constructors and Destructor

#pragma region Basic Operators

// make sure call order in column wise for CSC and row wise for CSR
template <class ElemType>
void CPUSparseMatrix<ElemType>::SetValue(const size_t row, const size_t col, const ElemType v)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (GetFormat() != MatrixFormat::matrixFormatSparseCSC && GetFormat() != MatrixFormat::matrixFormatSparseCSR)
    {
        LogicError("CPUSparseMatrix:  unsupported SetValue() call.");
    }

    if ((GetFormat() == MatrixFormat::matrixFormatSparseCSC) && ((*this)(row, col) == v))
        return;

    let nz = NzCount();
    if (GetSizeAllocated() < nz + 1) // automatic resize
    {
        Allocate(m_numRows, m_numCols, nz + 100, true, true); // allocate 100 more elelemnts and keep existing values
    }

    if (row < 0 || row >= m_numRows)
    {
        LogicError("CPUSparseMatrix: SetValue() invalid row id");
    }

    if (col < 0 || col >= m_numCols)
    {
        LogicError("CPUSparseMatrix: SetValue() invalid column id");
    }

    size_t r = (GetFormat() == matrixFormatSparseCSC) ? row : col;
    size_t c = (GetFormat() == matrixFormatSparseCSC) ? col : row;

    Data()[nz] = v;
    MajorIndexLocation()[nz] = (CPUSPARSE_INDEX_TYPE) r;

    // consistency check
    if (nz > 0)
    {
        if (c == GetColIdx() && r <= MajorIndexLocation()[nz - 1])
        {
            LogicError("CPUSparseMatrix:  SetValue is not called properly");
        }
    }

    if (c != GetColIdx())
    {
        SecondaryIndexLocation()[c] = CPUSPARSE_INDEX_TYPE(nz);
        SetColIdx((int) c);
    }
    // Note we don't have m_nz anymore. In order for the change from m_nz to
    // NzCount to make sense, we need to propogate nz+1 to all col slices.
    for (size_t max = c + 1; max < m_numCols + 1; max++)
    {
        SecondaryIndexLocation()[max] = CPUSPARSE_INDEX_TYPE(nz + 1);
    }
}

// make sure call order in colume wise for CSC and row wise for CSR
template <class ElemType>
void CPUSparseMatrix<ElemType>::SetValue(const CPUSparseMatrix<ElemType>& v)
{
    SetFormat(v.GetFormat());

    RequireSizeAndAllocate(v.GetNumRows(), v.GetNumCols(), v.NzCount() ); // TODO: rename to *Bytes/*Count instead of vague *Size if possible
    let nz = v.NzCount();

    auto matrixFormat = v.GetFormat();
    if (((matrixFormat == matrixFormatSparseBlockCol) || (matrixFormat == matrixFormatSparseBlockRow)) && (v.GetBlockIdShift() > 0))
        NOT_IMPLEMENTED;

    if (nz > 0)
    {
        memcpy(NzValues(),    v.NzValues(),    v.NzSize());

        if ((matrixFormat == matrixFormatSparseCSC) || (matrixFormat == matrixFormatSparseCSR))
        {
            memcpy(RowLocation(), v.RowLocation(), v.RowSize());
            memcpy(ColLocation(), v.ColLocation(), v.ColSize());
        }
        else
        {
            memcpy(GetBlockIds(), v.GetBlockIds(), v.GetBlockSize() * sizeof(size_t)); // TODO: change block id from size_t to CPUSPARSE_INDEX_TYPE, and rename BlockSize to BlockCount
            SetBlockSize(v.GetBlockSize());
        }
    }
    if (v.m_sliceViewOffset > 0)
    {
        CPUSPARSE_INDEX_TYPE* loc = (GetFormat() == matrixFormatSparseCSC) ? ColLocation() : RowLocation();
        size_t len = (GetFormat() == matrixFormatSparseCSC) ? ColSize() : RowSize();
        CPUSPARSE_INDEX_TYPE offset = loc[0];
        for (size_t c = 0; c < len; c++)
            loc[c] -= offset;
    }
}

#if 0
template <class ElemType>
void CPUSparseMatrix<ElemType>::SetValue(const CPUMatrix<ElemType>& /*v*/)
{
    NOT_IMPLEMENTED;
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::SetValue(const GPUMatrix<ElemType>& /*v*/)
{
    NOT_IMPLEMENTED;
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::SetValue(const GPUSparseMatrix<ElemType>& /*v*/)
{
    NOT_IMPLEMENTED;
}
#endif

template <class ElemType>
void CPUSparseMatrix<ElemType>::MaskColumnsValue(const CPUMatrix<char>& columnsMask, ElemType val, size_t numColsPerMaskEntry)
{
    VerifyWritable(__func__);

    if (GetNumCols() != (columnsMask.GetNumCols() * numColsPerMaskEntry))
        RuntimeError("Matrix number of columns must equal 'number of columns in column mask * numColsPerMaskEntry'.");

    if (val != 0)
        LogicError("MaskColumnsValue is not implmented for a non-zero mask for sparse matrices.");

#ifdef _DEBUG
    if (GetFormat() == MatrixFormat::matrixFormatSparseCSC)
    {
        // Get the binary columns mask
        char* maskedCols = columnsMask.Data();

        // If we're CSC, we only need to verify that the columns to be zeroed are empty.
        GPUSPARSE_INDEX_TYPE* colVector = SecondaryIndexLocation();
        auto n = columnsMask.GetNumCols();
#pragma omp parallel for
        for (long j = 0; j < n; j++)
            for (long k = 0; k < numColsPerMaskEntry; ++k)
                if (maskedCols[j] == 0 && colVector[(j * numColsPerMaskEntry) + k + 1] != colVector[(j * numColsPerMaskEntry) + k])
                    LogicError("CPUSparseMatrix attempted to mask column %d, but it has %d elements in it.", (int)((j * numColsPerMaskEntry) + k), (int)(colVector[(j * numColsPerMaskEntry) + k + 1] - colVector[(j * numColsPerMaskEntry) + k]));
    }
    else
        NOT_IMPLEMENTED;
#endif
}

template <class ElemType>
CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::DoGatherColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUSparseMatrix<ElemType>& a, ElemType alpha)
{
    VerifyWritable(__func__);

    if ((a.GetFormat() != matrixFormatSparseCSC) || (GetFormat() != matrixFormatSparseCSC))
        NOT_IMPLEMENTED;

    if (idx.GetNumRows() != 1) // index is 1-dimensional only
        InvalidArgument("DoGatherColumnsOf: Map must be a row vector.");

    if (beta != 0)
        NOT_IMPLEMENTED;

    // Determine the number of non-zero elements
    size_t numCols = idx.GetNumCols();
    size_t numNonZeroElements = 0;
    // TODO: Does it make sense to parallelize this?
    for (long j = 0; j < numCols; j++)
    {
        auto jInF = idx(0, j); // this is the column we need to get
        if (std::isnan(jInF) || (jInF < 0))     // negative index means gap
            continue;
        size_t jIn = (size_t)jInF;

        auto start = a.SecondaryIndexLocation()[jIn];
        auto end = a.SecondaryIndexLocation()[jIn + 1];
        numNonZeroElements += (end - start);
    }

    if (beta == 0)
        RequireSizeAndAllocate(a.GetNumRows(), idx.GetNumCols(), numNonZeroElements); // output has same column format as a, but number of columns comes from idx

    size_t offset = SecondaryIndexLocation()[0];
    // TODO: Does it make sense to parallelize this?
    for (long j = 0; j < numCols; j++)
    {
        auto jInF = idx(0, j); // this is the column we need to get
        if (jInF >= 0)     // negative or nan index means gap, but we still need to update the CompIndex
        {
            size_t jIn = (size_t)jInF;

            auto start = a.SecondaryIndexLocation()[jIn];
            auto end = a.SecondaryIndexLocation()[jIn + 1];
            for (auto p = start; p < end; p++, offset++)
            {
                GetUnCompIndex()[offset] = a.GetUnCompIndex()[p];
                Buffer()[offset] = a.Buffer()[p] * alpha;
            }
        }
        SecondaryIndexLocation()[j + 1] = CPUSPARSE_INDEX_TYPE(offset);
    }

    return *this;
}

// *this[:,idx[j]] = a[:,j] * alpha + *this[:,idx[j]] * beta
template <class ElemType>
CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::DoScatterColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUSparseMatrix<ElemType>& a, ElemType alpha)
{
    VerifyWritable(__func__);

    if ((a.GetFormat() != matrixFormatSparseCSC) || (GetFormat() != matrixFormatSparseCSC))
        NOT_IMPLEMENTED;

    if (idx.GetNumRows() != 1) // index is 1-dimensional only
        InvalidArgument("DoScatterColumnsOf: Map must be a row vector.");

    if (beta != 0)
        NOT_IMPLEMENTED;

    if (NzCount() != 0)
        InvalidArgument("CPUSparseMatrix::DoScatterColumnsOf: The target matrix cannot have pre-existing non-zero values when being scattered into");

    size_t numNonZeroElements = a.NzCount();

    if (beta == 0)
        RequireSizeAndAllocate(GetNumRows(), GetNumCols(), numNonZeroElements);

    // Setup the Secondary index
    std::vector<int> columnElementCounts(GetNumCols(), 0);
    size_t numColsToWrite = idx.GetNumCols();
    for (long j = 0; j < numColsToWrite; j++)
    {
        auto jOutF = idx(0, j); // this is the column we need to write to
        if (std::isnan(jOutF) || (jOutF < 0))     // negative index means gap
            continue;
        size_t jOut = (size_t)jOutF;
        columnElementCounts[jOut] = a.SecondaryIndexLocation()[j + 1] - a.SecondaryIndexLocation()[j];
    }

    // TODO: Replace with std::exclusive_scan when we switch to C++17
    for (size_t i = 1; i <= GetNumCols(); ++i)
        SecondaryIndexLocation()[i] = SecondaryIndexLocation()[i - 1] + columnElementCounts[i - 1];
    
    size_t offset = a.SecondaryIndexLocation()[0];
    // TODO: Does it make sense to parallelize this?
    for (long j = 0; j < numColsToWrite; j++)
    {
        auto jOutF = idx(0, j); // this is the column we need to write to
        if (std::isnan(jOutF) || (jOutF < 0))     // negative index means gap
            continue;
        size_t jOut = (size_t)jOutF;

        auto start = SecondaryIndexLocation()[jOut];
        auto end = SecondaryIndexLocation()[jOut + 1];
        for (auto p = start; p < end; p++, offset++)
        {
            GetUnCompIndex()[p] = a.GetUnCompIndex()[offset];
            Buffer()[p] = a.Buffer()[offset] * alpha;
        }
    }

    return *this;
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::Print(const char* matrixName) const
{
    Print(matrixName, 0, 0, 0, 0);
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::Print(const char* matrixName, ptrdiff_t /*rowStart*/, ptrdiff_t /*rowEnd*/, ptrdiff_t /*colStart*/, ptrdiff_t /*colEnd*/) const
{
    if (this->GetFormat() != matrixFormatSparseCSC && this->GetFormat() != matrixFormatSparseCSR)
    {
        return;
        // NOT_IMPLEMENTED;
    }

    fprintf(stderr, "%s\n", matrixName);

    const ElemType* dataBuffer = NzValues();
    const size_t nz = MajorIndexCount();
    CPUSPARSE_INDEX_TYPE* unCompressedIndex = MajorIndexLocation();
    CPUSPARSE_INDEX_TYPE* compressedIndex = SecondaryIndexLocation();

    for (size_t i = 0, j = 0; i < nz; ++i)
    {
        if (i >= compressedIndex[j])
        {
            fprintf(stderr, "\n");
            j++;
        }
        fprintf(stderr, "%d:%.f ", unCompressedIndex[i], dataBuffer[i]);
    }
    fprintf(stderr, "\n");
}

template <class ElemType>
CPUSparseMatrix<ElemType> CPUSparseMatrix<ElemType>::ColumnSlice(size_t startColumn, size_t numCols) const
{
    if (startColumn + numCols > m_numCols)
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) m_numCols);

    if (GetFormat() != MatrixFormat::matrixFormatSparseCSC && GetFormat() != MatrixFormat::matrixFormatSparseBlockCol)
        NOT_IMPLEMENTED;

    CPUSparseMatrix<ElemType> slice(GetFormat());
    slice.ShallowCopyFrom(*this);

    if ((startColumn != 0) || (slice.m_numCols != numCols))
    {
        slice.m_numCols = numCols;
        if (GetFormat() == MatrixFormat::matrixFormatSparseCSC)
        {
            slice.m_sliceViewOffset = m_sliceViewOffset + startColumn;
        }
        else if (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol)
        {
            long long startColBlock = 0, endColBlock = 0;
            bool foundStart = false, foundEnd = false;
            for (size_t j = 0; j < GetBlockSize(); j++)
            {
                if (j > 0)
                {
                    assert(GetBlockIds()[j] > GetBlockIds()[j - 1]); // assume ids are increasing.Is this valid?
                }

                if (!foundStart && (long long)GetBlockIds()[j] - (long long)GetBlockIdShift() >= (long long)startColumn) // start column with values
                {
                    startColBlock = j;
                    foundStart = true;
                }
                else if ((long long)GetBlockIds()[j] - (long long)GetBlockIdShift() >= (long long)(startColumn + numCols)) // end column with values
                {
                    endColBlock = j;
                    foundEnd = true;
                    break;
                }
            }
            if (!foundStart)
            {
                startColBlock = (long long)GetBlockSize();
            }
            if (!foundEnd)
            {
                endColBlock = (long long)GetBlockSize();
            }

            slice.m_sliceViewOffset = startColBlock;

            slice.SetBlockIds((size_t*)GetBlockIds() + startColBlock); // the value stored in the block id is based on the original column numbers
            slice.SetBlockSize((size_t)max((long long)0, endColBlock - startColBlock));
            slice.SetBlockIdShift(GetBlockIdShift() + startColumn);
        }
    }

    return slice;
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::AssignColumnSliceToDense(CPUMatrix<ElemType>& slice, size_t startColumn, size_t numCols) const
{
    if (startColumn + numCols > m_numCols)
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) m_numCols);

    if ((GetFormat() != MatrixFormat::matrixFormatSparseCSC) && (GetFormat() != MatrixFormat::matrixFormatSparseBlockCol))
        NOT_IMPLEMENTED;

    // We can either error out or RequireSize. Because RequireSize will error out if it's not allowed, I think this makes more sense.
    slice.RequireSize(m_numRows, numCols);

    memset(slice.Data(), 0, sizeof(ElemType) * slice.GetNumElements());

    if (GetFormat() == MatrixFormat::matrixFormatSparseCSC)
    {
#pragma omp parallel for
        for (long j = 0; j < numCols; j++)
        {
            long start = (long)SecondaryIndexLocation()[startColumn + j];
            long end = (long)SecondaryIndexLocation()[startColumn + j + 1];

            for (long p = start; p < end; p++)
            {
                size_t i = GetUnCompIndex()[p];
                ElemType value = Buffer()[(size_t)p];
                slice(i, (size_t)j) = value;
            }
        }
    }
    else
    {
        CPUSparseMatrix<ElemType> sparseSlice = ColumnSlice(startColumn, numCols);
        size_t numColumnsWithNonZeroValues = sparseSlice.GetBlockSize();
#pragma omp parallel for
        for (long j = 0; j < numColumnsWithNonZeroValues; j++)
        {
            size_t i = sparseSlice.GetBlockIds()[j] - sparseSlice.GetBlockIdShift();
            size_t len = sparseSlice.GetNumRows();
            size_t start = j * len;
            for (size_t p = start; p < start + len; p++)
            {
                ElemType val = sparseSlice.Buffer()[p];
                slice(p - start, i) = val;
            }
        }
    }
}
template <class ElemType>
CPUMatrix<ElemType> CPUSparseMatrix<ElemType>::CopyColumnSliceToDense(size_t startColumn, size_t numCols) const
{
    CPUMatrix<ElemType> slice(m_numRows, numCols);

    AssignColumnSliceToDense(slice, startColumn, numCols);

    return slice;
}

template <class ElemType>
CPUMatrix<ElemType> CPUSparseMatrix<ElemType>::DiagonalToDense() const
{
    if (m_numRows != m_numCols)
        LogicError("DiagonalToDense can be called only for square matrix.");

    if (GetFormat() != MatrixFormat::matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    CPUMatrix<ElemType> diag(1, m_numCols);

#pragma omp parallel for
    for (long j = 0; j < m_numCols; j++)
    {
        long start = (long) SecondaryIndexLocation()[j];
        long end = (long) SecondaryIndexLocation()[j + 1];

        for (long p = start; p < end; p++)
        {
            size_t i = MajorIndexLocation()[p];

            if (i == (size_t) j)
            {
                diag(0, i) = Data()[(size_t) p];
            }
        }
    }

    return diag;
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::SetMatrixFromCSCFormat(const CPUSPARSE_INDEX_TYPE* h_CSCCol, const CPUSPARSE_INDEX_TYPE* h_Row, const ElemType* h_Val,
                                                       const size_t nz, const size_t numRows, const size_t numCols)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    SetFormat(matrixFormatSparseCSC);
    RequireSizeAndAllocate(numRows, numCols, nz, true, false);

    // Note: This is a casualty of the switch away from m_nz. RowSize and NzSize depend on ColLocation being correct for format SparseCSC. Thus we must
    // copy ColLocation before RowLocation and NzValues. That's ugly and error prone.
    memcpy(ColLocation(), h_CSCCol, sizeof(CPUSPARSE_INDEX_TYPE)*(numCols + 1));
    memcpy(RowLocation(), h_Row, sizeof(CPUSPARSE_INDEX_TYPE)*nz);
    memcpy(NzValues(), h_Val, sizeof(ElemType)*nz);
}

#if 0 // add it back with test
template <class ElemType>
void CPUSparseMatrix<ElemType>::SetMatrixFromSBCFormat(const size_t* blockIds, const ElemType* val, const size_t numBlocks, const size_t numRows, const size_t numCols)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    SetFormat(matrixFormatSparseBlockCol);
    Resize(numRows, numCols, numBlocks * numRows);
    SetBlockSize(numBlocks);

    memcpy(GetBlockIds(), blockIds, sizeof(size_t)*(numBlocks));
    memcpy(Data(), val, sizeof(ElemType)*numBlocks*numRows);
}
#endif

template <class ElemType>
ElemType* CPUSparseMatrix<ElemType>::Data()  const
{
    return (Buffer() + 
        ((GetFormat() == matrixFormatSparseCSC || GetFormat() == matrixFormatSparseCSR) ? GetCompIndex()[m_sliceViewOffset] : 0));
}

// WARNING: When memory is reallocated, existing information will be lost.
// TODO: add keepExistingValues (default to true) argument so that the existing values are kept even after reallocation
template <class ElemType>
void CPUSparseMatrix<ElemType>::Allocate(const size_t numRows, const size_t numCols, const size_t numNZElemRequested, const bool growOnly /*= true*/, bool keepExistingValues /*= true*/)
{
    if (m_numRows != numRows || m_numCols != numCols)
        LogicError("Error, calling allocate with dimensions (%d, %d), but the matrix has dimension (%d, %d).", (int)numRows, (int)numCols, (int)GetNumRows(), (int)GetNumCols());

    size_t numNZElemToReserve = max(numNZElemRequested, (size_t) 1);
    size_t newCompIndexSize;
    if (GetFormat() == MatrixFormat::matrixFormatSparseCSC)
        newCompIndexSize = numCols + 1;
    else if (GetFormat() == MatrixFormat::matrixFormatSparseCSR)
        newCompIndexSize = numRows + 1;
    else
        newCompIndexSize = (numCols > numRows ? numCols : numRows) + 1;

    bool reallocate = (GetSizeAllocated() < numNZElemToReserve || (GetSizeAllocated() > numNZElemToReserve && !growOnly) || GetCompIndexSize() < newCompIndexSize);

    if (reallocate)
    {
        if (GetFormat() == MatrixFormat::matrixFormatSparseCSC || GetFormat() == MatrixFormat::matrixFormatSparseCSR)
        {
            // The initialization of the following buffer is done by new []().
            auto* pArray      = new ElemType[numNZElemToReserve]();
            auto* unCompIndex = new CPUSPARSE_INDEX_TYPE[numNZElemToReserve]();
            auto* compIndex   = new CPUSPARSE_INDEX_TYPE[newCompIndexSize]();

            if (keepExistingValues && (NzCount() > numNZElemToReserve || GetCompIndexSize() > newCompIndexSize))
                LogicError("Allocate: To keep values m_nz should <= numNZElemToReserve and m_compIndexSize <= newCompIndexSize");

            if (keepExistingValues && NzCount() > 0)
            {
                assert(GetCompIndexSize() > 0 && NzCount() < numNZElemToReserve);
                memcpy(pArray, Data(), NzSize());
                memcpy(unCompIndex, GetUnCompIndex(), MajorIndexSize());
                memcpy(compIndex, GetCompIndex(), SecondaryIndexSize());
            }

            // TODO: This is super ugly. The internals of the storage object should be a shared_ptr.
            delete[] Buffer();
            delete[] GetUnCompIndex();
            delete[] GetCompIndex();

            SetBuffer(pArray, numNZElemToReserve, false);
            SetUnCompIndex(unCompIndex);
            SetCompIndex(compIndex);
        }
        else if (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol || GetFormat() == MatrixFormat::matrixFormatSparseBlockRow)
        {
            ElemType* blockVal = new ElemType[numNZElemToReserve];
            size_t* blockIds = new size_t[newCompIndexSize];

            if (keepExistingValues && (NzCount() > numNZElemToReserve || GetCompIndexSize() > newCompIndexSize))
                LogicError("Resize: To keep values m_nz should <= numNZElemToReserve and m_compIndexSize <= newCompIndexSize");

            if (keepExistingValues && GetSizeAllocated() > 0)
            {
                assert(GetCompIndexSize() > 0 && GetSizeAllocated() < numNZElemToReserve);
                memcpy(blockVal, Data(), NzSize());
                memcpy(blockIds, GetBlockIds(), sizeof(size_t) * GetCompIndexSize());
            }

            delete[] Buffer();
            delete[] GetBlockIds();

            SetBuffer(blockVal, numNZElemToReserve, false);
            SetBlockIds(blockIds);
        }

        SetSizeAllocated(numNZElemToReserve);
        SetCompIndexSize(newCompIndexSize);
    }
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::RequireSizeAndAllocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve /*= 10000*/, const bool growOnly /*= true*/, bool keepExistingValues /*= false*/)
{
    RequireSizeAndAllocate(numRows, numCols, numNZElemToReserve, GetFormat(), growOnly, keepExistingValues);
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::RequireSizeAndAllocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly /*= true*/, bool keepExistingValues /*= true*/)
{
    RequireSize(numRows, numCols, numNZElemToReserve, matrixFormat, growOnly);
    
    size_t newCompIndexSize = (numCols > numRows ? numCols : numRows) + 1;
    bool reallocate = (GetSizeAllocated() < numNZElemToReserve || (GetSizeAllocated() > numNZElemToReserve && !growOnly) || GetCompIndexSize() < newCompIndexSize);

    if (reallocate)
        Allocate(numRows, numCols, numNZElemToReserve, growOnly, keepExistingValues);
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::RequireSize(const size_t numRows, const size_t numCols, const bool growOnly /*= true*/)
{
    RequireSize(numRows, numCols, GetFormat(), growOnly);
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::RequireSize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly /*= true*/)
{
    if (GetFormat() != matrixFormat || GetNumRows() != numRows || GetNumCols() != numCols)
        Resize(numRows, numCols, numNZElemToReserve, matrixFormat, growOnly);
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve /*= 10000*/, const bool growOnly /*= true*/)
{
    Resize(numRows, numCols, numNZElemToReserve, GetFormat(), growOnly);
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly /*= true*/)
{
    VerifyResizable(__func__);

    m_sliceViewOffset = 0;
    m_numRows = numRows;
    m_numCols = numCols;
    SetNumStorageRows(numRows);
    SetNumStorageCols(numCols);
    SetFormat(matrixFormat);

    size_t newCompIndexSize = (numCols > numRows ? numCols : numRows) + 1;
    bool reallocate = (GetCompIndexSize() < newCompIndexSize);

    if (reallocate)
        Allocate(numRows, numCols, numNZElemToReserve, growOnly, false);
    else
        Reset();
}


// Reset matrix to 0.
template <class ElemType>
void CPUSparseMatrix<ElemType>::Reset()
{
    // This is equivalent to setting m_nz = 0; Note we can only do this for sparse CSC/CSR because CompIndexSize is overloaded.
    if (GetFormat() == MatrixFormat::matrixFormatSparseCSC || GetFormat() == MatrixFormat::matrixFormatSparseCSR)
        memset(GetCompIndex(), 0, sizeof(CPUSPARSE_INDEX_TYPE) * GetCompIndexSize());
    SetColIdx(-1);
    SetBlockSize(0);
    SetBlockIdShift(0);
}

// Implements product of one sparse and one dense matrix updating a third dense matrix. Input matrices are optionally transposed.
// NOTE: The only for using a class template instead of a function template was that I couldn't make the function template compile.
template <class ElemType, bool denseTimesSparse /* false means SparseTimesDense */, bool transposeA, bool transposeB>
class MultiplyDenseAndSparse{
public:
    // Note: Below the ordering of the matrix parameters 'sparse' and 'dense' does not imply the order of the matrices in the product which is instead controlled
    // by the value of the boolean template parameter 'denseTimesSparse'.
    static void MultiplyAndWeightedAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& sparse, const CPUMatrix<ElemType>& dense, ElemType beta, CPUMatrix<ElemType>& c)
    {
        const BaseMatrix<ElemType>* lhs = denseTimesSparse ? (const BaseMatrix<ElemType>*) &dense  : (const BaseMatrix<ElemType>*) &sparse;
        const BaseMatrix<ElemType>* rhs = denseTimesSparse ? (const BaseMatrix<ElemType>*) &sparse : (const BaseMatrix<ElemType>*) &dense;

        // C(m:n) is the product of matrices X * Y where we have the shapes X(m:k) and Y(l:n)
        size_t m = transposeA ? lhs->GetNumCols() : lhs->GetNumRows();
        size_t k = transposeA ? lhs->GetNumRows() : lhs->GetNumCols();
        size_t l = transposeB ? rhs->GetNumCols() : rhs->GetNumRows();
        size_t n = transposeB ? rhs->GetNumRows() : rhs->GetNumCols();

        if (k != l)
            InvalidArgument("CPUSparseMatrix::MultiplyAndWeightedAdd: The inner dimensions of a (= %lu) and b (= %lu) don't match.", k, l);

        // Determine the dimension of the outer index of the dense matrix.
        size_t outerDimensionDense;
        if      ( denseTimesSparse && !transposeA) outerDimensionDense = dense.GetNumRows();
        else if ( denseTimesSparse &&  transposeA) outerDimensionDense = dense.GetNumCols();
        else if (!denseTimesSparse && !transposeB) outerDimensionDense = dense.GetNumCols();
        else if (!denseTimesSparse &&  transposeB) outerDimensionDense = dense.GetNumRows();

        if (beta == 0)
            c.RequireSize(m, n);
        else
            c.VerifySize(m, n); // Can't resize if beta != 0

        if (beta == 0)
            memset(c.Data(), 0, sizeof(ElemType)* c.GetNumElements());
        else if (beta != 1)
        {
#pragma omp parallel for
            foreach_coord(i, j, c)
            {
                c(i, j) = beta * c(i, j);
            }
        }
        else /* beta == 1*/
            ; // We keep the previous value of c before adding the matrix product.

        // In case one factor in the matrix product is empty there is nothing to add to the output c so we can exit here.
        if (sparse.IsEmpty() || dense.IsEmpty())
            return;

        // TODO: Implement CSR as a transposition of b, like we do for GPU.
        if (sparse.GetFormat() != matrixFormatSparseCSC)
            NOT_IMPLEMENTED;

        // Up to here we have:
        // * checked that the matrices are compatible in size
        // * Initialized the output matrix c

        // Now do the actual multiplication.
        ElemType* valueBuffer = sparse.Buffer() + *sparse.SecondaryIndexLocation(); // Points to the value buffer of the current view (i.e. buffer containing values of non-zero elements).
        int* rowIndexBuffer = sparse.MajorIndexLocation();                          // Points to the index buffer of the current view (i.e. buffer containing indices of non-zero elements).
        size_t iNonzero = 0;                                                           // Number of nonzero elements handled so far for curent slice view.
        int numPreviosNonzero = sparse.SecondaryIndexLocation()[0];                 // Total number of nonzero values handled in previous slices.

        // Loop over columns of the sparse matrix
        for (size_t colSparse = 0; colSparse < sparse.GetNumCols(); colSparse++)
        {
            size_t numNonzeroInSparseCol = sparse.SecondaryIndexLocation()[colSparse + 1] - numPreviosNonzero;
            // Loop over the nonzero rows of the current column of the sparse matrix
            for (; iNonzero < numNonzeroInSparseCol; iNonzero++)
            {
                size_t rowSparse = rowIndexBuffer[iNonzero]; // RowLocation
                ElemType sparseVal = valueBuffer[iNonzero];

                // Determine the index of the 'outer' dimension of the sparse matrix and the common inner index.
                size_t outerIndexSparse;
                size_t innerIndex;
                // Below if-statements are evaluated at compile time.
                if      ( denseTimesSparse && !transposeB) { outerIndexSparse = colSparse; innerIndex = rowSparse; }
                else if ( denseTimesSparse &&  transposeB) { outerIndexSparse = rowSparse; innerIndex = colSparse; }
                else if (!denseTimesSparse && !transposeA) { outerIndexSparse = rowSparse; innerIndex = colSparse; }
                else if (!denseTimesSparse &&  transposeA) { outerIndexSparse = colSparse; innerIndex = rowSparse; }

                // Loop over the outer index of the dense matrix
                for (size_t outerIndexDense = 0; outerIndexDense < outerDimensionDense; outerIndexDense++)
                {
                    // Determine the row index of the dense input matrix.
                    // Below if-statements are evaluated at compile time.
                    ElemType denseVal;
                    if      ( denseTimesSparse && !transposeA) denseVal = dense(outerIndexDense,      innerIndex);
                    else if ( denseTimesSparse &&  transposeA) denseVal = dense(     innerIndex, outerIndexDense);
                    else if (!denseTimesSparse && !transposeB) denseVal = dense(     innerIndex, outerIndexDense);
                    else if (!denseTimesSparse &&  transposeB) denseVal = dense(outerIndexDense,      innerIndex);
                    

                    // Update matrix c.
                    if (denseTimesSparse)
                        c(outerIndexDense, outerIndexSparse) += alpha * denseVal * sparseVal;
                    else /*Sparse times dense */
                        c(outerIndexSparse, outerIndexDense) += alpha * denseVal * sparseVal;
                }
            }
        }
    }
};

// c = alpha * lhs * rhs + beta * c
// dense * sparse -> dense
template <class ElemType>
void CPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, const bool transposeA,
                                                       const CPUSparseMatrix<ElemType>& b, const bool transposeB, ElemType beta, CPUMatrix<ElemType>& c)
{
    // Mapping variables to compile time template parameters for efficiency
    if      ( transposeA &&  transposeB)
        MultiplyDenseAndSparse<ElemType, true /* dense times sparse */,  true /* transposeA */,  true  /*transposeB*/>::MultiplyAndWeightedAdd(alpha, b /*sparse*/, a /* dense */, beta, c /* matrix beeing updated */);
    else if ( transposeA && !transposeB)
        MultiplyDenseAndSparse<ElemType, true /* dense times sparse */,  true /* transposeA */, false  /*transposeB*/>::MultiplyAndWeightedAdd(alpha, b /*sparse*/, a /* dense */, beta, c /* matrix beeing updated */);
    else if (!transposeA &&  transposeB)
        MultiplyDenseAndSparse<ElemType, true /* dense times sparse */, false /* transposeA */,  true  /*transposeB*/>::MultiplyAndWeightedAdd(alpha, b /*sparse*/, a /* dense */, beta, c /* matrix beeing updated */);
    else if (!transposeA && !transposeB)
        MultiplyDenseAndSparse<ElemType, true /* dense times sparse */, false /* transposeA */, false  /*transposeB*/>::MultiplyAndWeightedAdd(alpha, b /*sparse*/, a /* dense */, beta, c /* matrix beeing updated */);
}

// c = alpha * lhs * rhs + beta * c
// sparse * dense -> dense
template <class ElemType>
void CPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& a, const bool transposeA,
    const CPUMatrix<ElemType>& b, const bool transposeB, ElemType beta, CPUMatrix<ElemType>& c)
{
    // Mapping variables to compile time template parameters for efficiency
    if (transposeA &&  transposeB)
        MultiplyDenseAndSparse<ElemType, false /* dense times sparse */,  true /* transposeA */,  true /*transposeB*/>::MultiplyAndWeightedAdd(alpha, a /*sparse*/, b /* dense */, beta, c /* matrix beeing updated */);
    else if (transposeA && !transposeB)
        MultiplyDenseAndSparse<ElemType, false /* dense times sparse */,  true /* transposeA */, false /*transposeB*/>::MultiplyAndWeightedAdd(alpha, a /*sparse*/, b /* dense */, beta, c /* matrix beeing updated */);
    else if (!transposeA &&  transposeB)
        MultiplyDenseAndSparse<ElemType, false /* dense times sparse */, false /* transposeA */,  true /*transposeB*/>::MultiplyAndWeightedAdd(alpha, a /*sparse*/, b /* dense */, beta, c /* matrix beeing updated */);
    else if (!transposeA && !transposeB)
        MultiplyDenseAndSparse<ElemType, false /* dense times sparse */, false /* transposeA */, false /*transposeB*/>::MultiplyAndWeightedAdd(alpha, a /*sparse*/, b /* dense */, beta, c /* matrix beeing updated */);
}

// c = alpha * lhs * rhs
// dense * sparse -> sparse
template <class ElemType>
void CPUSparseMatrix<ElemType>::MultiplyAndAdd(ElemType alpha, const CPUMatrix<ElemType>& lhs, const bool transposeA,
                                               const CPUSparseMatrix<ElemType>& rhs, const bool transposeB, CPUSparseMatrix<ElemType>& c)
{
    if (!c.OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    if (lhs.IsEmpty() || rhs.IsEmpty())
        LogicError("LeftMultiplyAndAdd:  one of the input matrix is empty.");

    size_t m = transposeA ? (int) lhs.GetNumCols() : (int) lhs.GetNumRows();
    size_t k = transposeA ? (int) lhs.GetNumRows() : (int) lhs.GetNumCols();
    size_t l = transposeB ? (int) rhs.GetNumCols() : (int) rhs.GetNumRows();
    size_t n = transposeB ? (int) rhs.GetNumRows() : (int) rhs.GetNumCols();

    assert(m > 0 && k > 0 && l > 0 && n > 0);
    m;
    n; // converting from size_t to int may cause overflow
    assert(k == l);
    if (k != l)
    {
        InvalidArgument("CPUSparseMatrix::MultiplyAndAdd: The inner dimensions of a (= %lu) and b (= %lu) don't match.", k, l);
    }

    if (!transposeA && !transposeB)
    {
        NOT_IMPLEMENTED;
    }
    else if (!transposeA && transposeB)
    {
        if (rhs.GetFormat() != matrixFormatSparseCSC)
            NOT_IMPLEMENTED;

        // allocate enough memory
        c.SetFormat(matrixFormatSparseBlockCol);
        size_t blockSizePrev = c.GetBlockSize();

        if (blockSizePrev == 0)
        {
            c.RequireSizeAndAllocate(m, n, 0, true); // allocate for blockIds
        }

        map<size_t, size_t> col2BlockId;
        for (size_t blockId = 0; blockId < blockSizePrev; blockId++)
        {
            col2BlockId[c.GetBlockIds()[blockId]] = blockId;
        }

        size_t blockSizeCurr = blockSizePrev;
        for (size_t rhsNz = 0; rhsNz < rhs.NzCount(); rhsNz++)
        {
            size_t resultCol = rhs.MajorIndexLocation()[rhsNz];
            if (col2BlockId.find(resultCol) == col2BlockId.end())
            {
                col2BlockId[resultCol] = blockSizeCurr;
                c.GetBlockIds()[blockSizeCurr] = resultCol;
                blockSizeCurr ++;
            }
        }

        if (blockSizeCurr > blockSizePrev)
        {
            c.RequireSizeAndAllocate(m, n, m * blockSizeCurr, true, true);
            c.SetBlockSize(blockSizeCurr);
            memset(c.Data() + m * blockSizePrev, 0, sizeof(ElemType) * m * (blockSizeCurr - blockSizePrev));
        }

        for (size_t rhsCol = 0; rhsCol < rhs.GetNumCols(); rhsCol++)
        {
            size_t start = rhs.SecondaryIndexLocation()[rhsCol];
            size_t end = rhs.SecondaryIndexLocation()[rhsCol + 1];

            for (size_t p = start; p < end; p++)
            {
                size_t rhsRow = rhs.MajorIndexLocation()[p];
                ElemType val = rhs.Buffer()[p];

                ElemType* results = c.Buffer() + col2BlockId[rhsRow] * m;
                #pragma omp parallel for
                for (int lhsRow = 0; lhsRow < (int)m; lhsRow++)
                {
                    results[lhsRow] += alpha * lhs((size_t)lhsRow, rhsCol) * val;
                }
            }
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

// dense += sparse
template <class ElemType>
void CPUSparseMatrix<ElemType>::ScaleAndAdd(const ElemType alpha, const CPUSparseMatrix<ElemType>& lhs, CPUMatrix<ElemType>& rhs)
{
    if (lhs.IsEmpty() || rhs.IsEmpty())
    {
        LogicError("ScaleAndAdd:  one of the input matrix is empty.");
    }

    if (lhs.GetNumRows() != rhs.GetNumRows() || lhs.GetNumCols() != rhs.GetNumCols())
    {
        InvalidArgument("CPUSparseMatrix::ScaleAndAdd: The dimensions of a and b must match.");
    }

    if (lhs.GetFormat() == MatrixFormat::matrixFormatSparseCSC || lhs.GetFormat() == MatrixFormat::matrixFormatSparseCSR)
    {
        size_t col_num = (lhs.GetFormat() == MatrixFormat::matrixFormatSparseCSC) ? lhs.GetNumCols() : lhs.GetNumRows();
        for (size_t j = 0; j < col_num; j++)
        {
            size_t start = lhs.SecondaryIndexLocation()[j];
            size_t end = lhs.SecondaryIndexLocation()[j + 1];
            for (size_t p = start; p < end; p++)
            {
                size_t i = lhs.MajorIndexLocation()[p];
                ElemType val = lhs.Buffer()[p];
                size_t r = (lhs.GetFormat() == MatrixFormat::matrixFormatSparseCSC) ? i : j;
                size_t c = (lhs.GetFormat() == MatrixFormat::matrixFormatSparseCSC) ? j : i;
                rhs(r, c) += alpha * val;
            }
        }
    }
    else if (lhs.GetFormat() == MatrixFormat::matrixFormatSparseBlockCol || lhs.GetFormat() == MatrixFormat::matrixFormatSparseBlockRow)
    {
        for (size_t j = 0; j < lhs.GetBlockSize(); j++)
        {
            size_t i = lhs.GetBlockIds()[j] - lhs.GetBlockIdShift();
            size_t len = (lhs.GetFormat() == MatrixFormat::matrixFormatSparseBlockCol) ? lhs.GetNumRows() : lhs.GetNumCols();
            size_t start = j * len;
            for (size_t p = start; p < start + len; p++)
            {
                ElemType val = lhs.Buffer()[p];

                size_t r = (lhs.GetFormat() == MatrixFormat::matrixFormatSparseBlockCol) ? (p - start) : i;
                size_t c = (lhs.GetFormat() == MatrixFormat::matrixFormatSparseBlockCol) ? i : (p - start);
                rhs(r, c) += alpha * val;
            }
        }
    }
    else
    {
        RuntimeError("CPUSparseMatrix:: ScaleAndAdd() Not implemented");
    }
}

template <class ElemType>
/*static*/ bool CPUSparseMatrix<ElemType>::AreEqual(const CPUSparseMatrix<ElemType>& a, const CPUSparseMatrix<ElemType>& b, const ElemType threshold)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AreEqual: one of the input matrices is empty.");

    if (a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
        return false;

    bool result = true;

#pragma omp parallel for
    foreach_coord (i, j, a)
    {
        if (abs(a(i, j) - b(i, j)) > threshold)
        {
            result = false;
            break;
        }
    }

    return result;
}

template<class ElemType>
void CPUSparseMatrix<ElemType>::InnerProduct(const CPUSparseMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, const bool isColWise)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProduct:  one of the input matrices is empty.");

    const int m = (int)a.GetNumRows();
    const int n = (int)a.GetNumCols();
    const int k = (int)b.GetNumRows();
    const int l = (int)b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("InnerProduct: Matrices a and b should have same dimension.");

    if (isColWise) // col-wise
    {
        c.RequireSize(1, n);

#pragma omp parallel for
        foreach_column(j, c)
        {
            ElemType sum = 0;
            for (CPUSPARSE_INDEX_TYPE iRow = a.ColLocation()[j]; iRow < a.ColLocation()[j+1]; ++iRow)
            {
                size_t row = a.RowLocation()[iRow];
                sum += a.Data()[iRow] * b(row, j);
            }
            c(0, j) = sum;
        }
    }
    else
    {
        c.RequireSize(m, 1);

#pragma omp parallel for
        foreach_row(i, c)
        {
            ElemType sum = 0;
            for(CPUSPARSE_INDEX_TYPE j = 0; j < n; ++j)
            {
                for (CPUSPARSE_INDEX_TYPE iRow = a.ColLocation()[j]; iRow < a.ColLocation()[j + 1]; ++iRow)
                {
                    if (a.RowLocation()[iRow] == i)
                    {
                        sum += a.Data()[iRow] * b(i, j);
                        break;
                    }
                }
            }
            c(i, 0) = sum;
        }
    }
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
void CPUSparseMatrix<ElemType>::NormalGrad(CPUMatrix<ElemType>& c, const ElemType momentum, bool unitGainMomentum)
{
    const auto unitGainFactor = ElemType(unitGainMomentum ? (1.0 - momentum) : 1.0);

    if (c.IsEmpty())
    {
        c.RequireSize(GetNumRows(), GetNumCols());
        c.SetValue(0.0);
    }
    // BUGBUG: dimension/ownbuffer check?

    if (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol || GetFormat() == MatrixFormat::matrixFormatSparseBlockRow)
    {
        const auto isSparseBlockCol = (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol);
        for (size_t j = 0; j < GetBlockSize(); j++)
        {
            size_t i = GetBlockIds()[j] - GetBlockIdShift();
            size_t len = (isSparseBlockCol) ? GetNumRows() : GetNumCols();
            size_t start = j * len;
            for (size_t p = start; p < start + len; p++)
            {
                ElemType val = Buffer()[p];
                size_t row = (isSparseBlockCol) ? (p - start) : i;
                size_t col = (isSparseBlockCol) ? i : (p - start);
                c(row, col) = unitGainFactor * val + momentum * c(row, col);
                Buffer()[p] = c(row, col);
            }
        }
    }
    else
    {
        RuntimeError("CPUSparseMatrix:: NormalGrad() only support block sparse format");
    }
}

// update smoothed gradients c and current gradients (this)
template <class ElemType>
ElemType CPUSparseMatrix<ElemType>::Adagrad(CPUMatrix<ElemType>& c, const bool needAveMultiplier)
{
    if (c.IsEmpty() || c.GetNumCols() != GetNumCols() || c.GetNumRows() != GetNumRows())
    {
        c.RequireSize(GetNumRows(), GetNumCols());
        c.SetValue(0.0);
    }
    // BUGBUG: dimension/ownbuffer check?

    ElemType aveMultiplier = 0;

    const ElemType floor = 1e-16f;
    if (GetFormat() == MatrixFormat::matrixFormatSparseCSC || GetFormat() == MatrixFormat::matrixFormatSparseCSR)
    {
        size_t col_num = (GetFormat() == MatrixFormat::matrixFormatSparseCSC) ? GetNumCols() : GetNumRows();
        for (size_t j = 0; j < col_num; j++)
        {
            size_t start = SecondaryIndexLocation()[j];
            size_t end = SecondaryIndexLocation()[j + 1];
            for (size_t p = start; p < end; p++)
            {
                size_t i = MajorIndexLocation()[p];
                ElemType val = Buffer()[p];

                size_t row = (GetFormat() == MatrixFormat::matrixFormatSparseCSC) ? i : j;
                size_t col = (GetFormat() == MatrixFormat::matrixFormatSparseCSC) ? j : i;
                ElemType adenorm = c(row, col);
                adenorm += val * val;
                ElemType a = sqrt(floor + adenorm);
                Buffer()[p] = val / a;
                c(row, col) = adenorm;

                if (needAveMultiplier)
                    aveMultiplier += 1 / a;
            }
        }
    }
    else if (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol || GetFormat() == MatrixFormat::matrixFormatSparseBlockRow)
    {
        size_t len = (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol) ? GetNumRows() : GetNumCols();
        size_t p = 0;
        for (long j = 0; j < GetBlockSize(); j++)
        {
            size_t colOrRow = GetBlockIds()[j] - GetBlockIdShift();
            for (long i = 0; i < len; i++, p++)
            {
                ElemType val = Buffer()[p];

                size_t row = (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol) ? i : colOrRow;
                size_t col = (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol) ? colOrRow : i;
                c(row, col) += val * val;
                ElemType a = sqrt(floor + c(row, col));
                Buffer()[p] /= a;

                if (needAveMultiplier)
                    aveMultiplier += 1 / a;
            }
        }
    }

    size_t nz = NzCount();
    if (needAveMultiplier && nz > 0)
        return aveMultiplier / nz;
    else
        return 1;
}

template <class ElemType>
void CPUSparseMatrix<ElemType>::AdaDelta(CPUMatrix<ElemType>& c, CPUMatrix<ElemType>& functionValues, ElemType rho, ElemType epsilon)
{
    size_t numColsNeeded = 2 * GetNumCols();

    if (IsEmpty() || (GetNumCols() < numColsNeeded))
    {
        c.RequireSize(GetNumRows(), numColsNeeded);
        c.SetValue(0.0);
    }

    if (GetNumRows() != GetNumRows() || GetNumCols() != numColsNeeded)
        LogicError("The matrix gradients does not have expected dimensions.");

    if (GetFormat() != MatrixFormat::matrixFormatSparseBlockCol)
        LogicError("Unsupported sparse format.");

    size_t n = GetNumElements();
    ElemType* grad = Data();
    ElemType* smoothAda = c.Data();
    ElemType* smoothX2 = c.Data() + n;
    ElemType* val = functionValues.Data();

#pragma omp parallel for
    // TODO: Unroll 4-times for better performance leveraging vectorization
    for (int j = 0; j < (int)GetBlockSize(); j++)
    {
        size_t i = GetBlockIds()[j] - GetBlockIdShift();
        size_t len = GetNumRows();
        size_t start = j * len;
        for (size_t p = start; p < start + len; p++)
        {
            ElemType g = grad[p];
            ElemType adaSqr = rho * smoothAda[i] + (1 - rho) * g * g;
            smoothAda[i] = adaSqr;
            ElemType x2 = smoothX2[i];
            ElemType deltaX = -sqrt(x2 + epsilon) / sqrt(adaSqr + epsilon) * g;
            smoothX2[i] = rho * smoothX2[i] + (1 - rho) * deltaX * deltaX;
            val[i] += deltaX;
        }
    }
}

template <class ElemType>
CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::InplaceTruncateTop(const ElemType threshold)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    long m = (long) this->NzCount();
    ElemType* nzValues = NzValues();

#pragma omp parallel for
    for (long i = 0; i < (m & ~3); i += 4) // four-way unrolling
    {
        if (nzValues[i] > threshold)
            nzValues[i] = threshold;

        if (nzValues[i + 1] > threshold)
            nzValues[i + 1] = threshold;

        if (nzValues[i + 2] > threshold)
            nzValues[i + 2] = threshold;

        if (nzValues[i + 3] > threshold)
            nzValues[i + 3] = threshold;
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        if (nzValues[i] > threshold)
            nzValues[i] = threshold;
    }

    return *this;
}

template <class ElemType>
CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::InplaceTruncateBottom(const ElemType threshold)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    long m = (long) this->NzCount();
    ElemType* nzValues = NzValues();

#pragma omp parallel for
    for (long i = 0; i < (m & ~3); i += 4) // four-way unrolling
    {
        if (nzValues[i] < threshold)
            nzValues[i] = threshold;

        if (nzValues[i + 1] < threshold)
            nzValues[i + 1] = threshold;

        if (nzValues[i + 2] < threshold)
            nzValues[i + 2] = threshold;

        if (nzValues[i + 3] < threshold)
            nzValues[i + 3] = threshold;
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        if (nzValues[i] < threshold)
            nzValues[i] = threshold;
    }

    return *this;
}

template <class ElemType>
CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::InplaceTruncate(const ElemType threshold)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    ElemType locThresholdPos = abs(threshold);
    ElemType locTHresholdNeg = -locThresholdPos;

    long m = (long) this->NzCount();
    ElemType* nzValues = NzValues();

#pragma omp parallel for
    for (long i = 0; i < (m & ~3); i += 4) // four-way unrolling
    {
        if (nzValues[i] > locThresholdPos)
            nzValues[i] = locThresholdPos;
        else if (nzValues[i] < locTHresholdNeg)
            nzValues[i] = locTHresholdNeg;

        if (nzValues[i + 1] > locThresholdPos)
            nzValues[i + 1] = locThresholdPos;
        else if (nzValues[i + 1] < locTHresholdNeg)
            nzValues[i + 1] = locTHresholdNeg;

        if (nzValues[i + 2] > locThresholdPos)
            nzValues[i + 2] = locThresholdPos;
        else if (nzValues[i + 2] < locTHresholdNeg)
            nzValues[i + 2] = locTHresholdNeg;

        if (nzValues[i + 3] > locThresholdPos)
            nzValues[i + 3] = locThresholdPos;
        else if (nzValues[i + 3] < locTHresholdNeg)
            nzValues[i + 3] = locTHresholdNeg;
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        if (nzValues[i] > locThresholdPos)
            nzValues[i] = locThresholdPos;
        else if (nzValues[i] < locTHresholdNeg)
            nzValues[i] = locTHresholdNeg;
    }

    return *this;
}

template <class ElemType>
CPUSparseMatrix<ElemType>& CPUSparseMatrix<ElemType>::InplaceSoftThreshold(const ElemType threshold)
{
    if (!OwnBuffer())
        LogicError("Cannot modify since the buffer is managed externally.");

    long m = (long) this->NzCount();
    ElemType* nzValues = NzValues();

#pragma omp parallel for
    for (long i = 0; i < (m & ~3); i += 4) // four-way unrolling
    {
        if (nzValues[i] > threshold)
            nzValues[i] -= threshold;
        else if (nzValues[i] < -threshold)
            nzValues[i] += threshold;
        else
            nzValues[i] = 0;

        if (nzValues[i + 1] > threshold)
            nzValues[i + 1] -= threshold;
        else if (nzValues[i + 1] < -threshold)
            nzValues[i + 1] += threshold;
        else
            nzValues[i + 1] = 0;

        if (nzValues[i + 2] > threshold)
            nzValues[i + 2] -= threshold;
        else if (nzValues[i + 2] < -threshold)
            nzValues[i + 2] += threshold;
        else
            nzValues[i + 2] = 0;

        if (nzValues[i + 3] > threshold)
            nzValues[i + 3] -= threshold;
        else if (nzValues[i + 3] < -threshold)
            nzValues[i + 3] += threshold;
        else
            nzValues[i + 3] = 0;
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        if (nzValues[i] > threshold)
            nzValues[i] -= threshold;
        else if (nzValues[i] < -threshold)
            nzValues[i] += threshold;
        else
            nzValues[i] = 0;
    }
    return *this;
}

template <class ElemType>
ElemType CPUSparseMatrix<ElemType>::FrobeniusNorm() const
{
    if (IsEmpty())
        return 0;

    ElemType v = 0; // TODO: do this in 'double'?

    long m = (long) NzCount();
    const ElemType* nzValues = NzValues();

//four-way unrolling
#pragma omp parallel for reduction(+ : v)
    for (long i = 0; i < (m & ~3); i += 4)
    {
        v += nzValues[i] * nzValues[i] + nzValues[i + 1] * nzValues[i + 1] + nzValues[i + 2] * nzValues[i + 2] + nzValues[i + 3] * nzValues[i + 3];
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        v += nzValues[i] * nzValues[i];
    }

    return sqrt(v);
}

//sum of all abs(elements)
template <class ElemType>
ElemType CPUSparseMatrix<ElemType>::SumOfAbsElements() const
{
    if (IsEmpty())
        return 0;

    if (sizeof(ElemType) == sizeof(double))
    {
        return (ElemType) cblas_dasum((int) this->NzCount(), reinterpret_cast<double*>(Data()), 1);
    }
    else
    {
#pragma warning(suppress : 4244)
        return cblas_sasum((int) this->NzCount(), reinterpret_cast<float*>(Data()), 1);
    }
}

//sum of all elements
template <class ElemType>
ElemType CPUSparseMatrix<ElemType>::SumOfElements() const
{
    if (IsEmpty())
        return 0;

    ElemType sum = 0; // TODO: Do this in 'double'?

    long m = (long) NzCount();
    const ElemType* nzValues = NzValues();

//four-way unrolling
#pragma omp parallel for reduction(+ : sum)
    for (long i = 0; i < (m & ~3); i += 4)
    {
        sum += nzValues[i] + nzValues[i + 1] + nzValues[i + 2] + nzValues[i + 3];
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        sum += nzValues[i];
    }

    return sum;
}

template <typename ElemType>
MATH_API File& operator>>(File& stream, CPUSparseMatrix<ElemType>& us)
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

    us.SetFormat((MatrixFormat) format);
    if (us.GetFormat() != matrixFormatSparseCSC && us.GetFormat() != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    us.RequireSizeAndAllocate(rownum, colnum, nz, true, false);

    if (nz > 0)
    {
        size_t compressedSize = (us.GetFormat() == matrixFormatSparseCSC) ? colnum + 1 : rownum + 1;
        ElemType* dataBuffer = us.NzValues();
        CPUSPARSE_INDEX_TYPE* unCompressedIndex = us.MajorIndexLocation();
        CPUSPARSE_INDEX_TYPE* compressedIndex = us.SecondaryIndexLocation();

        // read in the sparse matrix info
        for (size_t i = 0; i < nz; ++i)
        {
            stream >> dataBuffer[i];
        }
        for (size_t i = 0; i < nz; ++i)
        {
            stream >> unCompressedIndex[i];
        }
        for (size_t i = 0; i < compressedSize; ++i)
        {
            stream >> compressedIndex[i];
        }
    }
    stream.GetMarker(fileMarkerEndSection, std::wstring(L"EMAT"));

    return stream;
}

template MATH_API File& operator>>(File& stream, CPUSparseMatrix<float>& us);
template MATH_API File& operator>>(File& stream, CPUSparseMatrix<double>& us);

template <typename ElemType>
MATH_API File& operator<<(File& stream, const CPUSparseMatrix<ElemType>& us)
{
    if (us.GetFormat() != matrixFormatSparseCSC && us.GetFormat() != matrixFormatSparseCSR)
        NOT_IMPLEMENTED;

    stream.PutMarker(fileMarkerBeginSection, std::wstring(L"BMAT"));
    stream << sizeof(ElemType);
    stream << std::wstring(L"nnmatrix"); // Note this is needed for compatability, and could potentially be an empty string

    size_t nz, numRows, numCols;
    size_t compressedSize = us.SecondaryIndexCount();
    int format = us.GetFormat();

    stream << format << nz << numCols << numRows;

    if (nz > 0)
    {
        ElemType* dataBuffer = us.NzValues();
        CPUSPARSE_INDEX_TYPE* unCompressedIndex = us.MajorIndexLocation();
        CPUSPARSE_INDEX_TYPE* compressedIndex = us.SecondaryIndexLocation();

        for (size_t i = 0; i < nz; ++i)
        {
            stream << dataBuffer[i];
        }
        for (size_t i = 0; i < nz; ++i)
        {
            stream << unCompressedIndex[i];
        }
        for (size_t i = 0; i < compressedSize; ++i)
        {
            stream << compressedIndex[i];
        }
    }
    stream.PutMarker(fileMarkerEndSection, std::wstring(L"EMAT"));

    return stream;
}

template class CPUSparseMatrix<float>;
template class CPUSparseMatrix<double>;

// We use Matrix<char> as the backing store for QuantizedMatrix
// Let's explciitly instantiate the methods we need for that purpose
template CPUSparseMatrix<char>::CPUSparseMatrix(const MatrixFormat format, const size_t numRows, const size_t numCols, const size_t size);
template CPUSparseMatrix<char>::CPUSparseMatrix(MatrixFormat);
template CPUSparseMatrix<char>::CPUSparseMatrix(CPUSparseMatrix<char> const&);
template CPUSparseMatrix<char>::CPUSparseMatrix(CPUSparseMatrix<char>&&);
template CPUSparseMatrix<char>& CPUSparseMatrix<char>::operator=(CPUSparseMatrix<char>&& moveFrom);
template void CPUSparseMatrix<char>::SetValue(size_t, size_t, char);
//template void CPUSparseMatrix<char>::SetValue(CPUMatrix<char> const&);
//template void CPUSparseMatrix<char>::SetValue(GPUMatrix<char> const&);
template void CPUSparseMatrix<char>::SetValue(CPUSparseMatrix<char> const&);
//template void CPUSparseMatrix<char>::SetValue(GPUSparseMatrix<char> const&);
template char* CPUSparseMatrix<char>::Data() const;
template void CPUSparseMatrix<char>::Reset(void);
template void CPUSparseMatrix<char>::Resize(const size_t, const size_t, const size_t, const bool);
template void CPUSparseMatrix<char>::RequireSizeAndAllocate(const size_t, const size_t, const size_t, const bool, bool);
template void CPUSparseMatrix<char>::RequireSizeAndAllocate(const size_t, const size_t, const size_t, const MatrixFormat, const bool, bool);
template CPUSparseMatrix<char>::~CPUSparseMatrix();
template CPUSparseMatrix<char> CPUSparseMatrix<char>::ColumnSlice(size_t startColumn, size_t numCols) const;
template CPUMatrix<char> CPUSparseMatrix<char>::CopyColumnSliceToDense(size_t startColumn, size_t numCols) const;
template void CPUSparseMatrix<char>::AssignColumnSliceToDense(CPUMatrix<char>&, size_t startColumn, size_t numCols) const;
template CPUSparseMatrix<char>& CPUSparseMatrix<char>::operator=(const CPUSparseMatrix<char>& deepCopyFrom);
template void CPUSparseMatrix<char>::ScaleAndAdd(char, class Microsoft::MSR::CNTK::CPUSparseMatrix<char> const &, class Microsoft::MSR::CNTK::CPUMatrix<char> &);

// Support <short>
template CPUSparseMatrix<short>::CPUSparseMatrix(const MatrixFormat format, const size_t numRows, const size_t numCols, const size_t size);
template CPUSparseMatrix<short>::CPUSparseMatrix(MatrixFormat);
template CPUSparseMatrix<short>::CPUSparseMatrix(CPUSparseMatrix<short> const&);
template CPUSparseMatrix<short>::CPUSparseMatrix(CPUSparseMatrix<short>&&);
template CPUSparseMatrix<short>& CPUSparseMatrix<short>::operator=(CPUSparseMatrix<short>&& moveFrom);
template void CPUSparseMatrix<short>::SetValue(size_t, size_t, short);
//template void CPUSparseMatrix<short>::SetValue(CPUMatrix<short> const&);
//template void CPUSparseMatrix<short>::SetValue(GPUMatrix<short> const&);
template void CPUSparseMatrix<short>::SetValue(CPUSparseMatrix<short> const&);
//template void CPUSparseMatrix<short>::SetValue(GPUSparseMatrix<short> const&);
template short* CPUSparseMatrix<short>::Data() const;
template void CPUSparseMatrix<short>::Reset(void);
template void CPUSparseMatrix<short>::Resize(const size_t, const size_t, const size_t, const bool);
template void CPUSparseMatrix<short>::RequireSizeAndAllocate(const size_t, const size_t, const size_t, const bool, bool);
template void CPUSparseMatrix<short>::RequireSizeAndAllocate(const size_t, const size_t, const size_t, const MatrixFormat, const bool, bool);
template CPUSparseMatrix<short>::~CPUSparseMatrix();
template CPUSparseMatrix<short> CPUSparseMatrix<short>::ColumnSlice(size_t startColumn, size_t numCols) const;
template CPUMatrix<short> CPUSparseMatrix<short>::CopyColumnSliceToDense(size_t startColumn, size_t numCols) const;
template void CPUSparseMatrix<short>::AssignColumnSliceToDense(CPUMatrix<short>&, size_t startColumn, size_t numCols) const;
template CPUSparseMatrix<short>& CPUSparseMatrix<short>::operator=(const CPUSparseMatrix<short>& deepCopyFrom);
template void CPUSparseMatrix<short>::ScaleAndAdd(short, class Microsoft::MSR::CNTK::CPUSparseMatrix<short> const &, class Microsoft::MSR::CNTK::CPUMatrix<short> &);

template CPUSparseMatrix<int>::CPUSparseMatrix(const MatrixFormat, const size_t, const size_t, const size_t);
template CPUSparseMatrix<int>::~CPUSparseMatrix();

}}}
