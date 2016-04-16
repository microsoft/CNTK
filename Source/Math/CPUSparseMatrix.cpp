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

#ifdef USE_ACML
// use ACML as default.
// Download ACML 5.3.0 (e.g., acml5.3.0-ifort64.exe) or above
// from http://developer.amd.com/tools/cpu-development/amd-core-math-library-acml/acml-downloads-resources/
// Install the ifort64 variant (compiled with intel compiler) of the library
// Set Environment variable ACML_PATH to C:\AMD\acml5.3.0\ifort64_mp or the folder you installed acml
// to point to your folder for the include file and link library
#include <acml.h> // requires ACML 5.3.0 and above
#elif defined(USE_MKL)
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

#ifdef USE_ACML // MKL has one additional parameter for different matrix order
#define BLAS_COLMAJOR
#else
#define BLAS_COLMAJOR (int) MatrixOrder::ColMajor,
#endif

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

    RequireSizeAndAllocate(v.GetNumRows(), v.GetNumCols(), v.NzSize());
    let nz = v.NzCount();

    if (nz > 0)
    {
        memcpy(NzValues(),    v.NzValues(),    v.NzSize());
        memcpy(RowLocation(), v.RowLocation(), v.RowSize());
        memcpy(ColLocation(), v.ColLocation(), v.ColSize());
    }
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

    slice.m_numCols             = numCols;
    if (GetFormat() == MatrixFormat::matrixFormatSparseCSC)
    {
        slice.m_sliceViewOffset   = m_sliceViewOffset + startColumn;
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

            if (!foundStart && (long long) GetBlockIds()[j] - (long long) GetBlockIdShift() >= (long long) startColumn) // start column with values
            {
                startColBlock = j;
                foundStart = true;
            }
            else if ((long long) GetBlockIds()[j] - (long long) GetBlockIdShift() >= (long long) (startColumn + numCols)) // end column with values
            {
                endColBlock = j;
                foundEnd = true;
                break;
            }
        }
        if (!foundStart)
        {
            startColBlock = (long long) GetBlockSize();
        }
        if (!foundEnd)
        {
            endColBlock = (long long) GetBlockSize();
        }

        slice.m_sliceViewOffset = startColBlock;

        slice.SetBlockIds((size_t*)GetBlockIds() + startColBlock); // the value stored in the block id is based on the original column numbers
        slice.SetBlockSize((size_t) max((long long) 0, endColBlock - startColBlock));
        slice.SetBlockIdShift(GetBlockIdShift() + startColumn);
    }

    return slice;
}

template <class ElemType>
CPUMatrix<ElemType> CPUSparseMatrix<ElemType>::CopyColumnSliceToDense(size_t startColumn, size_t numCols) const
{
    if (startColumn + numCols > m_numCols)
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) m_numCols);

    if (GetFormat() != MatrixFormat::matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    CPUMatrix<ElemType> slice(m_numRows, numCols);

#pragma omp parallel for
    for (long j = 0; j < numCols; j++)
    {
        long start = (long) SecondaryIndexLocation()[startColumn + j];
        long end = (long)SecondaryIndexLocation()[startColumn + j + 1];

        for (long p = start; p < end; p++)
        {
            size_t i = GetUnCompIndex()[p];
            ElemType value = Buffer()[(size_t) p];
            slice(i, (size_t) j) = value;
        }
    }

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

template <class ElemType>
ElemType* CPUSparseMatrix<ElemType>::Data() const
{
    return Buffer() + GetCompIndex()[m_sliceViewOffset];
}

template <class ElemType>
ElemType* CPUSparseMatrix<ElemType>::Data() 
{
    return Buffer() + GetCompIndex()[m_sliceViewOffset];
}

// WARNING: When memory is reallocated, existing information will be lost.
// TODO: add keepExistingValues (default to true) argument so that the existing values are kept even after reallocation
template <class ElemType>
void CPUSparseMatrix<ElemType>::Allocate(const size_t numRows, const size_t numCols, const size_t numNZElemRequested, const bool growOnly /*= true*/, bool keepExistingValues /*= true*/)
{
    if (m_numRows != numRows || m_numCols != numCols)
        LogicError("Error, calling allocate with dimensions (%d, %d), but the matrix has dimension (%d, %d).", (int)numRows, (int)numCols, (int)GetNumRows(), (int)GetNumCols());

    size_t numNZElemToReserve = max(numNZElemRequested, (size_t) 1);
    size_t newCompIndexSize = (numCols > numRows ? numCols : numRows) + 1;
    bool reallocate = (GetSizeAllocated() < numNZElemToReserve || (GetSizeAllocated() > numNZElemToReserve && !growOnly) || GetCompIndexSize() < newCompIndexSize);

    if (reallocate)
    {
        if (GetFormat() == MatrixFormat::matrixFormatSparseCSC || GetFormat() == MatrixFormat::matrixFormatSparseCSR)
        {
            auto* pArray      = new ElemType[numNZElemToReserve]();
            auto* unCompIndex = new CPUSPARSE_INDEX_TYPE[numNZElemToReserve]();
            auto* compIndex   = new CPUSPARSE_INDEX_TYPE[newCompIndexSize]();

            if (keepExistingValues && (NzCount() > numNZElemToReserve || GetCompIndexSize() > newCompIndexSize))
                LogicError("Allocate: To keep values m_nz should <= numNZElemToReserve and m_compIndexSize <= newCompIndexSize");

			memset(pArray, 0, sizeof(ElemType) * numNZElemToReserve);
			memset(unCompIndex, 0, sizeof(CPUSPARSE_INDEX_TYPE) * numNZElemToReserve);
			memset(compIndex, 0, sizeof(CPUSPARSE_INDEX_TYPE) * newCompIndexSize);

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
    RequireSize(numRows, numCols, matrixFormat, growOnly);
    
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
void CPUSparseMatrix<ElemType>::RequireSize(const size_t numRows, const size_t numCols, const MatrixFormat matrixFormat, const bool growOnly /*= true*/)
{
    if (GetFormat() != matrixFormat || GetNumRows() != numRows || GetNumCols() != numCols)
        Resize(numRows, numCols, 0, matrixFormat, growOnly);
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

// c = alpha*op(lhs) * op(rhs) + beta*c
// dense x sparse = dense
template <class ElemType>
void CPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& lhs, const bool transposeA,
                                                       const CPUSparseMatrix<ElemType>& rhs, const bool transposeB, ElemType beta, CPUMatrix<ElemType>& c)
{
    if (lhs.IsEmpty() || rhs.IsEmpty())
        LogicError("MultiplyAndWeightedAdd:  one of the input matrix is empty.");

    int m = transposeA ? (int) lhs.GetNumCols() : (int) lhs.GetNumRows();
    int k = transposeA ? (int) lhs.GetNumRows() : (int) lhs.GetNumCols();
    int l = transposeB ? (int) rhs.GetNumCols() : (int) rhs.GetNumRows();
    int n = transposeB ? (int) rhs.GetNumRows() : (int) rhs.GetNumCols();

    assert(m > 0 && k > 0 && l > 0 && n > 0); // converting from size_t to int may cause overflow
    assert(k == l);
    if (k != l)
    {
        InvalidArgument("CPUSparseMatrix::MultiplyAndWeightedAdd: The inner dimensions of a and b must match.");
    }

    if (beta == 0)
        c.RequireSize(m, n);
    else
        c.VerifySize(m, n); // Can't resize if beta != 0


    if (beta == 0)
    {
        memset(c.Buffer(), 0, sizeof(ElemType) * c.GetNumElements());
    }
    else if (beta != 1)
    {
#pragma omp parallel for
        foreach_coord (i, j, c)
        {
            c(i, j) = beta * c(i, j);
        }
    }

    if (rhs.GetFormat() != matrixFormatSparseCSC)
        NOT_IMPLEMENTED;

    if (!transposeA && !transposeB)
    {
        for (size_t j = 0; j < rhs.GetNumCols(); j++)
        {
            size_t start = rhs.SecondaryIndexLocation()[j]; // ColLocation
            size_t end = rhs.SecondaryIndexLocation()[j + 1];
            for (size_t p = start; p < end; p++)
            {
                size_t i = rhs.MajorIndexLocation()[p]; // RowLocation
                ElemType val = rhs.Buffer()[p];

                for (size_t h = 0; h < lhs.GetNumRows(); h++)
                {
                    c(h, j) += alpha * lhs(h, i) * val;
                }
            }
        }
    }
    else if (!transposeA && transposeB)
    {
        for (size_t j = 0; j < rhs.GetNumCols(); j++)
        {
            size_t start = rhs.SecondaryIndexLocation()[j];
            size_t end = rhs.SecondaryIndexLocation()[j + 1];

            for (size_t p = start; p < end; p++)
            {
                size_t i = rhs.MajorIndexLocation()[p];
                ElemType val = rhs.Buffer()[p];
                for (size_t h = 0; h < lhs.GetNumRows(); h++)
                {
                    c(h, i) += alpha * lhs(h, j) * val;
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

// dense x sparse = sparse
// c = alpha * op(lhs) * op(rhs)
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
        InvalidArgument("CPUSparseMatrix::MultiplyAndAdd: The inner dimensions of a and b must match.");
    }

    c.Reset();

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
        c.RequireSizeAndAllocate(m, n, m * min(n, rhs.NzCount()), true, false);

        map<size_t, size_t> w2Id;
        for (size_t j = 0; j < rhs.GetNumCols(); j++)
        { // j ranges over batches
            size_t start = rhs.SecondaryIndexLocation()[j];
            size_t end = rhs.SecondaryIndexLocation()[j + 1];

            for (size_t p = start; p < end; p++)
            {
                size_t i = rhs.MajorIndexLocation()[p]; // i ranges over words
                ElemType val = rhs.Buffer()[p];  // 1 for(i, j)

                bool first = true;
                if (w2Id.find(i) == w2Id.end())
                {
                    size_t id = w2Id.size();
                    w2Id[i] = id;
                    c.GetBlockIds()[c.GetBlockSize()] = i;
                    c.SetBlockSize(c.GetBlockSize() + 1);
                }
                else
                {
                    first = false;
                }
                size_t pos = w2Id[i] * lhs.GetNumRows();
                for (size_t h = 0; h < lhs.GetNumRows(); h++)
                { // h range over hidden layer
                    if (first == true)
                    {
                        c.Buffer()[pos] = alpha * lhs(h, j) * val;
                    }
                    else
                    {
                        c.Buffer()[pos] += alpha * lhs(h, j) * val;
                    }
                    pos++;
                }
            }
        }
        if (c.GetBlockSize() * m > c.GetSizeAllocated())
        {
            LogicError("Sparse matrix is unexpectedly out of range.");
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

// normal update for smoothed gradients c and current gradients (this)
// TODO: comment seems wrong; cf. SGD.cpp: smoothedGradient.NormalGrad(gradientValues, functionValues,...)
template <class ElemType>
void CPUSparseMatrix<ElemType>::NormalGrad(CPUMatrix<ElemType>& c, const ElemType momentum)
{
    if (c.IsEmpty())
    {
        c.RequireSize(GetNumRows(), GetNumCols());
        c.SetValue(0.0);
    }
    // BUGBUG: dimension/ownbuffer check?

    if (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol || GetFormat() == MatrixFormat::matrixFormatSparseBlockRow)
    {
        for (size_t j = 0; j < GetBlockSize(); j++)
        {
            size_t i = GetBlockIds()[j] - GetBlockIdShift();
            size_t len = (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol) ? GetNumRows() : GetNumCols();
            size_t start = j * len;
            for (size_t p = start; p < start + len; p++)
            {
                ElemType val = Buffer()[p];
                size_t row = (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol) ? (p - start) : i;
                size_t col = (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol) ? i : (p - start);
                c(row, col) = (1 - momentum) * val + momentum * c(row, col);
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
#ifdef USE_ACML
        return (ElemType) dasum((int) this->NzCount(), reinterpret_cast<double*>(Data()), 1);
#else
        return (ElemType) cblas_dasum((int) this->NzCount(), reinterpret_cast<double*>(Data()), 1);
#endif
    }
    else
    {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
        return sasum((int) this->NzCount(), reinterpret_cast<float*>(Data()), 1);
#else
        return cblas_sasum((int) this->NzCount(), reinterpret_cast<float*>(Data()), 1);
#endif
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
template void CPUSparseMatrix<char>::SetValue(CPUSparseMatrix<char> const&);
template char* CPUSparseMatrix<char>::Data() const;
template char* CPUSparseMatrix<char>::Data();
template void CPUSparseMatrix<char>::Reset(void);
template void CPUSparseMatrix<char>::RequireSizeAndAllocate(const size_t, const size_t, const size_t, const bool, bool);
template void CPUSparseMatrix<char>::RequireSizeAndAllocate(const size_t, const size_t, const size_t, const MatrixFormat, const bool, bool);
template CPUSparseMatrix<char>::~CPUSparseMatrix();
template CPUSparseMatrix<char> CPUSparseMatrix<char>::ColumnSlice(size_t startColumn, size_t numCols) const;
template CPUMatrix<char> CPUSparseMatrix<char>::CopyColumnSliceToDense(size_t startColumn, size_t numCols) const;
template CPUSparseMatrix<char>& CPUSparseMatrix<char>::operator=(const CPUSparseMatrix<char>& deepCopyFrom);

template CPUSparseMatrix<int>::CPUSparseMatrix(const MatrixFormat, const size_t, const size_t, const size_t);
template CPUSparseMatrix<int>::~CPUSparseMatrix();

}}}
