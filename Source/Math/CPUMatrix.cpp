//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//

#include "stdafx.h"
#include "Basics.h"
#include "File.h"

#include "CPUMatrix.h"
#include "TensorOps.h"
#include <assert.h>
#include <stdexcept>
#include <omp.h>
#include <math.h>
#include <random>
#include <chrono>
#include <exception>
#include <thread>
#include <iostream>
#include <algorithm>
#ifdef _WIN32
#define NOMINMAX
#include "Windows.h"
#else
#include <cfloat>
#endif

#ifdef LEAKDETECT
#include <vld.h>
#endif

#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this
#pragma warning(disable : 4702) // unreachable code; triggered for unknown reasons

#ifdef USE_ACML
// Download ACML 5.3.1 (e.g., acml5.3.1-ifort64.exe) or above
// from http://developer.amd.com/tools/cpu-development/amd-core-math-library-acml/acml-downloads-resources/
// Install the ifort64_mp variant (compiled with intel compiler) of the library
// Set Environment variable ACML_PATH to C:\AMD\acml5.3.1\ifort64_mp or the folder you installed acml
// to point to your folder for the include file and link library
#include <acml.h> // requires ACML 5.3.1 and above
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

#ifdef USE_ACML // MKL has one additional parameter for different matrix order
#define BLAS_COLMAJOR
#else
#define BLAS_COLMAJOR (int) MatrixOrder::ColMajor,
#endif

#define SWAP(a, b)  \
    {               \
        (a) ^= (b); \
        (b) ^= (a); \
        (a) ^= (b); \
    }
#define IDX2C(i, j, ld) (((j) * (ld)) + (i)) // 0 based indexing
namespace Microsoft { namespace MSR { namespace CNTK {

int MATH_API TracingGPUMemoryAllocator::m_traceLevel = 0;

void TracingGPUMemoryAllocator::SetTraceLevel(int traceLevel)
{
    m_traceLevel = traceLevel;
}

bool TracingGPUMemoryAllocator::IsTraceEnabled()
{
    return (m_traceLevel > 0);
}

#pragma region Helpful Enum Definitions
enum class MatrixOrder
{
    RowMajor = 101, // row-major arrays
    ColMajor = 102  // column-major arrays
};

enum class MatrixTranspose : char
{
    NoTrans = 'N',  // trans='N'
    Trans = 'T',    // trans='T'
    ConjTrans = 'C' // trans='C'
};

enum class SymMatrixType : char
{
    Up = 'U',          // symmetric matrix is stored in the upper part
    Low = 'L',         // symmetric matrix is stored in thelower part
    Full = 'F',        // full populated
    NotSymmetric = 'N' // not a symmetric matrix
};

enum class MatrixOpSide : char
{
    Left = 'L',  // left multiply
    Right = 'R', // right multiply
};
#pragma endregion Helpful Enum Definitions

#pragma region Constructors and Destructor

//should only be used by constructors.
template <class ElemType>
void CPUMatrix<ElemType>::ZeroInit()
{
    m_computeDevice = CPUDEVICE;
    m_pArray = nullptr;
    m_numRows = 0;
    m_numCols = 0;
    m_elemSizeAllocated = 0;
    m_format = matrixFormatDense;
    m_externalBuffer = false;
}

template <class ElemType>
CPUMatrix<ElemType>::CPUMatrix()
{
    ZeroInit();
}

// helper to allocate an array of ElemType
// Use this instead of new[] to get NaN initialization for debugging.
template <class ElemType>
static ElemType* NewArray(size_t n)
{
    ElemType* p = new ElemType[n]();
#if 0 // _DEBUG
        ElemType nan = Matrix<ElemType>::MakeNan(__LINE__);
        for (size_t i = 0; i < n; i++)
            p[i] = nan;
#endif
    return p;
}

template <class ElemType>
CPUMatrix<ElemType>::CPUMatrix(const size_t numRows, const size_t numCols)
{
    ZeroInit();

    m_numRows = numRows;
    m_numCols = numCols;
    m_elemSizeAllocated = GetNumElements();

    if (m_elemSizeAllocated != 0)
        m_pArray = NewArray<ElemType>(m_elemSizeAllocated);
}

template <class ElemType>
CPUMatrix<ElemType>::CPUMatrix(const size_t numRows, const size_t numCols, ElemType* pArray, const size_t matrixFlags)
{
    ZeroInit();
    SetValue(numRows, numCols, pArray, matrixFlags);
}

//copy constructor, deep copy
template <class ElemType>
CPUMatrix<ElemType>::CPUMatrix(const CPUMatrix<ElemType>& deepCopyFrom)
{
    ZeroInit();
    if (!deepCopyFrom.IsEmpty())
        SetValue(deepCopyFrom);
}

//assignment operator, deep copy
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator=(const CPUMatrix<ElemType>& deepCopyFrom)
{
    Clear();
    if (!deepCopyFrom.IsEmpty())
        SetValue(deepCopyFrom);
    return *this;
}

//move constructor, shallow copy
template <class ElemType>
CPUMatrix<ElemType>::CPUMatrix(CPUMatrix<ElemType>&& moveFrom)
{
    m_computeDevice = moveFrom.m_computeDevice;
    m_numRows = moveFrom.m_numRows;
    m_numCols = moveFrom.m_numCols;
    m_elemSizeAllocated = moveFrom.m_elemSizeAllocated;
    m_pArray = moveFrom.m_pArray; // shallow copy the pointer
    m_format = moveFrom.m_format;
    m_externalBuffer = moveFrom.m_externalBuffer;
    // release the pointer from the source object so that the destructor won't release it twice
    moveFrom.ZeroInit();
}

//move assignment operator, shallow copy
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator=(CPUMatrix<ElemType>&& moveFrom)
{
    if (this != &moveFrom)
    {
        if (OwnBuffer() && m_pArray != nullptr)
            delete[] m_pArray; // always delete the data pointer since we will use the pointer from moveFrom

        m_computeDevice = moveFrom.m_computeDevice;
        m_numRows = moveFrom.m_numRows;
        m_numCols = moveFrom.m_numCols;
        m_elemSizeAllocated = moveFrom.m_elemSizeAllocated;
        m_pArray = moveFrom.m_pArray;
        m_format = moveFrom.m_format;
        m_externalBuffer = moveFrom.m_externalBuffer;

        // release the pointer from the source object so that the destructor won't release it twice
        moveFrom.ZeroInit();
    }
    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>::~CPUMatrix()
{
    Clear();
}

template <class ElemType>
void CPUMatrix<ElemType>::Clear()
{
    if (m_pArray != nullptr && OwnBuffer())
    {
        delete[] m_pArray;
        m_pArray = nullptr;
        m_elemSizeAllocated = 0;
    }
    BaseMatrix<ElemType>::Clear();

    ZeroInit();
}

#pragma endregion Constructors and Destructor

#pragma region Basic Operators

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::ColumnSlice(size_t startColumn, size_t numCols) const
{
    // if (numCols == 0)
    //    LogicError("The slice cannot have 0 columns.");

    if (startColumn + numCols > m_numCols)
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) m_numCols);

    CPUMatrix<ElemType> slice;

    slice.m_externalBuffer = true; // memory of a slice is managed externally.
    slice.m_numRows = m_numRows;
    slice.m_numCols = numCols;
    slice.m_elemSizeAllocated = slice.GetNumElements();
    slice.m_pArray = m_pArray + startColumn * m_numRows;
    slice.m_format = m_format;

    return slice;
}

// set this(:, 0:numCols-1) = fromMatrix(:, startColumn : startColumn+numCols-1)
// TODO: why not say *this = ColumnSlice()?
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignColumnSlice(const CPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols)
{
    // if (numCols == 0)
    //    LogicError("The slice cannot have 0 columns.");

    if (startColumn + numCols > fromMatrix.m_numCols)
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) fromMatrix.m_numCols);

    Clear();

    SetOwnBuffer(false); // memory of a slice is managed externally.
    m_numRows = fromMatrix.m_numRows;
    m_numCols = numCols;
    m_elemSizeAllocated = GetNumElements();
    m_pArray = fromMatrix.m_pArray + startColumn * m_numRows;

    return *this;
}

// set this(: , startColumn:startColumn+numCols-1)= fromMatrix;
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::SetColumnSlice(const CPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols)
{
    // if (numCols == 0)
    //    LogicError("The slice cannot have 0 columns.");
    if (startColumn + numCols > m_numCols)
        LogicError("The slice is out of range of the destination matrix.");
    if (numCols > fromMatrix.GetNumCols())
        InvalidArgument("The slice (%d) is out of range of the source matrix (%d).", (int) numCols, (int) fromMatrix.GetNumCols());
    if (m_numRows != fromMatrix.m_numRows)
        LogicError("The number of rows in source and destination matrices do not match");

    // SetOwnBuffer(false);
    memcpy(m_pArray + startColumn * m_numRows, fromMatrix.m_pArray, numCols * m_numRows * sizeof(ElemType));

    return *this;
}

template <class ElemType>
void CPUMatrix<ElemType>::CopyColumnsStrided(const CPUMatrix<ElemType>& fromMatrix, size_t numCols, size_t srcNumColsStride, size_t destNumColsStride)
{
    if ((((numCols - 1) * srcNumColsStride) + 1) > fromMatrix.m_numCols)
        LogicError("The numCols to copy and srcNumColsStride specified is out of range of the source matrix.");
    if ((((numCols - 1) * destNumColsStride) + 1) > m_numCols)
        LogicError("The numCols to copy and srcNumColsStride specified is out of range of the destination matrix.");
    if (m_numRows != fromMatrix.m_numRows)
        LogicError("The number of rows in source and destination matrices do not match");

    long n = (long) numCols, m = (long) m_numRows;

    auto& us = *this;

#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (size_t i = 0; i < (m & ~3); i += 4)
        {
            us(i, j * destNumColsStride) = fromMatrix(i, j * srcNumColsStride);
            us(i + 1, j * destNumColsStride) = fromMatrix(i + 1, j * srcNumColsStride);
            us(i + 2, j * destNumColsStride) = fromMatrix(i + 2, j * srcNumColsStride);
            us(i + 3, j * destNumColsStride) = fromMatrix(i + 3, j * srcNumColsStride);
        }

        // handle remaining
        for (size_t i = m & ~3; i < m; i++)
        {
            us(i, j * destNumColsStride) = fromMatrix(i, j * srcNumColsStride);
        }
    }
}

//for each column of a, we add all rows of a to this starting from startIndex
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignToRowSliceValuesOf(const CPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (a.GetNumRows() != numRows)
        LogicError("AddToRowSliceValuesOf: a.GetNumRows() != numRows.");

    if (startIndex + numRows > GetNumRows())
        LogicError("AddToRowSliceValuesOf: startIndex + numRows exceeds GetNumRows().");

    if (a.GetNumCols() != GetNumCols())
        LogicError("AddToRowSliceValuesOf: columns does not match.");

    long n = (long) a.GetNumCols(), m = (long) numRows;

    auto& us = *this;

#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (size_t i = 0, startRow = startIndex; i < (m & ~3); i += 4, startRow += 4)
        {
            us(startRow, j) = a(i, j);
            us(startRow + 1, j) = a(i + 1, j);
            us(startRow + 2, j) = a(i + 2, j);
            us(startRow + 3, j) = a(i + 3, j);
        }
        // handle remaining stuffs
        for (size_t i = m & ~3, startRow = startIndex + (m & ~3); i < m; i++, startRow++)
        {
            us(startRow, j) = a(i, j);
        }
    }

    return *this;
}

//for each column of a, we assign numRows starting from startIndex to this
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignRowSliceValuesOf(const CPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (startIndex + numRows > a.GetNumRows())
        LogicError("AssignRowSliceValuesOf: startIndex + numRows exceeds a.GetNumRows().");

    Resize(numRows, a.GetNumCols());

    long n = (long) a.GetNumCols(); // note: OpenMP requires loop indices to be long, not size_t
    long k = (long) a.GetNumRows();

#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // memory copy might be faster?
        memcpy(m_pArray + j * numRows, a.m_pArray + j * k + startIndex, sizeof(ElemType) * numRows);

        // //four-way unrolling
        // for (long i=0, startRow = startIndex; i<(m & ~3); i+=4, startRow+=4)
        // {
        //    us(i,j) = a(startRow,j);
        //    us(i+1,j) = a(startRow+1,j);
        //    us(i+2,j) = a(startRow+2,j);
        //    us(i+3,j) = a(startRow+3,j);
        // }
        // //handle remaining stuffs
        // for (long i=m & ~3, startRow = startIndex+(m & ~3); i<m; i++, startRow++)
        // {
        //    us(i,j) = a(startRow,j);
        // }
    }

    return *this;
}

//for the row slice of this starting from startIndex we add a to it.
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddToRowSliceValuesOf(const CPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (a.IsEmpty())
        LogicError("AddToRowSliceValuesOf: input matrix a is empty.");

    if (a.GetNumRows() != numRows)
        LogicError("AddToRowSliceValuesOf: a.GetNumRows() != numRows.");

    if (startIndex + numRows > GetNumRows())
        LogicError("AddToRowSliceValuesOf: startIndex + numRows exceeds GetNumRows().");

    if (a.GetNumCols() != GetNumCols())
        LogicError("AddToRowSliceValuesOf: columns does not match.");

    long n = (long) a.GetNumCols(), m = (long) numRows;

    auto& us = *this;

#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0, startRow = (long) startIndex; i < (m & ~3); i += 4, startRow += 4)
        {
            us(startRow, j) += a(i, j);
            us(startRow + 1, j) += a(i + 1, j);
            us(startRow + 2, j) += a(i + 2, j);
            us(startRow + 3, j) += a(i + 3, j);
        }
        // handle remaining stuffs
        for (long i = m & ~3, startRow = (long) startIndex + (m & ~3); i < m; i++, startRow++)
        {
            us(startRow, j) += a(i, j);
        }
    }

    return *this;
}

//for each column of this, we add row slice of a starting from startIndex
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddWithRowSliceValuesOf(const CPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (a.IsEmpty())
        LogicError("AddWithRowSliceValuesOf: input matrix a is empty.");

    if (GetNumRows() != numRows)
        LogicError("AddWithRowSliceValuesOf: GetNumRows() != numRows.");

    if (startIndex + numRows > a.GetNumRows())
        LogicError("AddWithRowSliceValuesOf: startIndex + numRows exceeds a.GetNumRows().");

    if (a.GetNumCols() != GetNumCols())
        LogicError("AddWithRowSliceValuesOf: columns does not match.");

    long n = (long) a.GetNumCols(), m = (long) numRows;

    auto& us = *this;

#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0, startRow = (long) startIndex; i < (m & ~3); i += 4, startRow += 4)
        {
            us(i, j) += a(startRow, j);
            us(i + 1, j) += a(startRow + 1, j);
            us(i + 2, j) += a(startRow + 2, j);
            us(i + 3, j) += a(startRow + 3, j);
        }
        // handle remaining stuffs
        for (long i = m & ~3, startRow = (long) startIndex + (m & ~3); i < m; i++, startRow++)
        {
            us(i, j) += a(startRow, j);
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::Diagonal() const
{
    if (m_numRows != m_numCols)
        LogicError("Diagonal can be called only for square matrix. (rows=%d, cols=%d)", (int) m_numRows, (int) m_numCols);

    CPUMatrix<ElemType> diag(1, m_numCols);

    auto& us = *this;

#pragma omp parallel for
    for (long i = 0; i < m_numRows; i++)
    {
        diag(0, (size_t) i) = us(i, i);
    }

    return diag;
}

template <class ElemType>
void CPUMatrix<ElemType>::MinusOneAt(CPUMatrix<ElemType>& c, const size_t position)
{
    if (position < c.GetNumElements())
        c.m_pArray[position] -= 1.0;
    else
        RuntimeError("MinusOneAt: position is out of CPU matrix size");
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignRepeatOf(const CPUMatrix<ElemType>& a, const size_t numRowRepeats, const size_t numColRepeats)
{
    if (this == &a)
        LogicError("AssignRepeatOf: a is the same as [this]. Does not support inplace repeat.");

    if (a.IsEmpty())
        LogicError("AssignRepeatOf: Matrix a is empty.");

    Resize(a.GetNumRows() * numRowRepeats, a.GetNumCols() * numColRepeats);
    long n = (long) a.GetNumCols(), m = (long) a.GetNumRows();
    auto& us = *this;

#pragma omp parallel for
    for (long q = 0; q < numColRepeats; q++)
    {
        for (long p = 0; p < numRowRepeats; p++)
        {
            long colOffset = q * n;

            for (long j = 0; j < n; j++, colOffset++)
            {
                long rowOffset = p * m;

                // four-way unrolling
                for (long i = 0; i < (m & ~3); i += 4, rowOffset += 4)
                {
                    us(rowOffset, colOffset) = a(i, j);
                    us(rowOffset + 1, colOffset) = a(i + 1, j);
                    us(rowOffset + 2, colOffset) = a(i + 2, j);
                    us(rowOffset + 3, colOffset) = a(i + 3, j);
                }
                // handle remaining stuffs
                for (long i = m & ~3; i < m; i++, rowOffset++)
                {
                    us(rowOffset, colOffset) = a(i, j);
                }
            }
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddToRowRepeatValuesOf(const CPUMatrix<ElemType>& a, const size_t numRepeats)
{
    if (a.IsEmpty())
        LogicError("AddToRowRepeatValuesOf: input matrix a is empty.");

    if (a.GetNumRows() != GetNumRows() * numRepeats)
        LogicError("AddToRowRepeatValuesOf: a.GetNumRows() != GetNumRows() * numRepeats.");

    long n = (long) a.GetNumCols(), m = (long) GetNumRows();

    auto& us = *this;

#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            for (long k = 0; k < numRepeats; k++)
            {
                us(i, j) += a(k * m + i, j);
                us(i + 1, j) += a(k * m + i + 1, j);
                us(i + 2, j) += a(k * m + i + 2, j);
                us(i + 3, j) += a(k * m + i + 3, j);
            }
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            for (long k = 0; k < numRepeats; k++)
            {
                us(i, j) += a(k * m + i, j);
            }
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignPositiveAndShiftedNegSample(const CPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber)
{
    a;
    posNumber;
    negNumber;
    shiftNumber;
    NOT_IMPLEMENTED;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddFoldedPositiveAndShiftedNegSample(const CPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber)
{
    a;
    posNumber;
    negNumber;
    shiftNumber;
    NOT_IMPLEMENTED;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::Transpose()
{
    if (IsEmpty())
        LogicError("Transpose: Matrix is empty.");

    CPUMatrix<ElemType> c;
    c.AssignTransposeOf(*this);
    return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignTransposeOf(const CPUMatrix<ElemType>& a)
{
    if (this == &a)
        LogicError("AssignTransposeOf: a is the same as [this]. Does not support inplace transpose.");

    if (a.IsEmpty())
        LogicError("AssignTransposeOf: Matrix a is empty.");

    Resize(a.GetNumCols(), a.GetNumRows());
    long n = (long) a.GetNumCols(), m = (long) a.GetNumRows();

    auto& us = *this;

#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(j, i) = a(i, j);
            us(j, i + 1) = a(i + 1, j);
            us(j, i + 2) = a(i + 2, j);
            us(j, i + 3) = a(i + 3, j);
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(j, i) = a(i, j);
        }
    }

    return *this;
}

// dst[i] = src[i] * alpha + dst[i] * beta
// scale a column vector and add it to another
// The usual special case: If beta = 0, then dst[] is not read, and may be uninitialized or NaN.
template <class ElemType>
static void ScaleAndAddColumn(ElemType beta, ElemType* dst, const ElemType* src, size_t numRows, ElemType alpha)
{
    if (alpha != 1) // rare case: just do the full thing
        for (size_t i = 0; i < numRows; i++)
            dst[i] = beta * dst[i] + alpha * src[i];
    else if (beta == 1) // used in backprop
        for (size_t i = 0; i < numRows; i++)
            dst[i] += src[i];
    else if (beta == 0) // plain assignment
        memcpy(dst, src, sizeof(ElemType) * numRows);
    else // alpha=1, arbitrary beta: also rare case
        for (size_t i = 0; i < numRows; i++)
            dst[i] = beta * dst[i] + src[i];
}

// *this[:,j] = a[:,m[j]] * alpha + *this[:,j] * beta
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::DoGatherColumnsOf(ElemType beta, const CPUMatrix<ElemType>& m, const CPUMatrix<ElemType>& a, ElemType alpha)
{
    if (m.GetNumRows() != 1) // index is 1-dimensional only
        InvalidArgument("DoGatherColumnsOf: Map must be a row vector.");

    if (beta)
        VerifySize(a.GetNumRows(), m.GetNumCols());
    else
        Resize(a.GetNumRows(), m.GetNumCols());

    auto& us = *this;
#pragma omp parallel for // TODO: Depending in circumstance, it may be more efficient to parallelize over rows.
    foreach_column(jOut, us)
    {
        auto jInF = m(0, jOut); // this is the column we need to get
        if (jInF < 0)           // negative index means gap
            continue;
        size_t jIn = (size_t)jInF;
        if (jIn >= a.GetNumCols())
            InvalidArgument("DoGatherColumnsOf: Map out of bounds.");
        ScaleAndAddColumn(beta, &us(0,jOut), &a(0,jIn), us.GetNumRows(), alpha);
    }

    return *this;
}

// *this[:,m[j]] = a[:,j] * alpha + *this[:,m[j]] * beta
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::DoScatterColumnsOf(ElemType beta, const CPUMatrix<ElemType>& m, const CPUMatrix<ElemType>& a, ElemType alpha)
{
    if (m.GetNumRows() != 1) // index is 1-dimensional only
        InvalidArgument("DoScatterColumnsOf: Map must be a row vector.");
    if (m.GetNumCols() != a.GetNumCols())
        InvalidArgument("DoScatterColumnsOf: Map must have width of input vector.");
    if (a.GetNumRows() != GetNumRows())
        InvalidArgument("DoScatterColumnsOf: Output must have same height as input vector.");

    auto& us = *this;

    // pre-scale with beta upfront
    // Scatter may add more than one source column to the same target, so we must pre-scale with beta, and then just keep adding.
    Scale(beta, us); // if beta is 0, then this will be a memset()

#pragma omp parallel for // TODO: Depending in circumstance, it may be more efficient to parallelize over rows.
    foreach_column(jIn, a)
    {
        auto jOutF = m(0, jIn); // this is the column we copy/add into
        if (jOutF < 0)          // negative index means gap
            continue;
        size_t jOut = (size_t)jOutF;
        if (jOut >= GetNumCols())
            InvalidArgument("DoGatherColumnsOf: Map out of bounds.");
        ScaleAndAddColumn(/*beta=*/(ElemType)1, &us(0, jOut), &a(0, jIn), us.GetNumRows(), alpha);
    }

    return *this;
}

template <class ElemType>
void CPUMatrix<ElemType>::SetValue(const ElemType v)
{
    if (IsEmpty())
        LogicError("SetValue: Matrix is empty.");
    bool isFinite = std::numeric_limits<ElemType>::is_integer || std::isfinite((double) v);
    if (isFinite && v == 0)
    {
        memset(m_pArray, 0, sizeof(ElemType) * GetNumElements());
    }
    else
    {
        long m = (long) GetNumElements();
        // 2-way thread parallelism is sufficient for the memory bound
        // operation of just setting the values of an array.
        const unsigned SETVALUE_NUM_THREADS = 2;
#pragma omp parallel for num_threads(SETVALUE_NUM_THREADS)
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            m_pArray[i] = v;
            m_pArray[i + 1] = v;
            m_pArray[i + 2] = v;
            m_pArray[i + 3] = v;
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            m_pArray[i] = v;
        }
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::MaskColumnsValue(const CPUMatrix<char>& columnsMask, ElemType val)
{
    if (GetNumCols() != columnsMask.GetNumCols())
        RuntimeError("Matrix and column mask must have equal number of columns");

    auto& us = *this;
    long n = (long) GetNumCols(), m = (long) GetNumRows();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        if (columnsMask(0, j) == 1)
            continue;

        // four-way unrolling
        for (size_t i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) = val;
            us(i + 1, j) = val;
            us(i + 2, j) = val;
            us(i + 3, j) = val;
        }

        // handle remaining
        for (size_t i = m & ~3; i < m; i++)
        {
            us(i, j) = val;
        }
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::SetColumn(const ElemType* colPointer, size_t j)
{
    if (IsEmpty())
        LogicError("SetColumn: Matrix is empty.");
    if (colPointer == NULL)
        return;

    auto& us = *this;
    long m = (long) GetNumRows();
#pragma omp parallel for
    // four-way unrolling
    for (long i = 0; i < (m & ~3); i += 4)
    {
        us(i, j) = colPointer[i];
        us(i + 1, j) = colPointer[i + 1];
        us(i + 2, j) = colPointer[i + 2];
        us(i + 3, j) = colPointer[i + 3];
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        us(i, j) = colPointer[i];
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::SetColumn(const ElemType val, size_t j)
{
    if (IsEmpty())
        LogicError("SetColumn: Matrix is empty.");

    auto& us = *this;
    long m = (long) GetNumRows();
#pragma omp parallel for
    // four-way unrolling
    for (long i = 0; i < (m & ~3); i += 4)
    {
        us(i, j) = val;
        us(i + 1, j) = val;
        us(i + 2, j) = val;
        us(i + 3, j) = val;
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        us(i, j) = val;
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::SetColumn(const CPUMatrix<ElemType>& valMat, size_t j)
{
    if (IsEmpty())
        LogicError("SetColumn: Matrix is empty.");
    assert(valMat.GetNumRows() == GetNumRows() && valMat.GetNumCols() == 1);

    auto& us = *this;
    long m = (long) GetNumRows();
#pragma omp parallel for
    // four-way unrolling
    for (long i = 0; i < (m & ~3); i += 4)
    {
        us(i, j) = valMat(i, 0);
        us(i + 1, j) = valMat(i + 1, 0);
        us(i + 2, j) = valMat(i + 2, 0);
        us(i + 3, j) = valMat(i + 3, 0);
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        us(i, j) = valMat(i, 0);
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::SetValue(const CPUMatrix<ElemType>& deepCopyFrom)
{
    if (this == &deepCopyFrom)
        return;

    Resize(deepCopyFrom.GetNumRows(), deepCopyFrom.GetNumCols());
    memcpy(m_pArray, deepCopyFrom.m_pArray, deepCopyFrom.GetNumElements() * sizeof(ElemType));
}

template <class ElemType>
void CPUMatrix<ElemType>::SetValue(const size_t numRows, const size_t numCols, ElemType* pArray, const size_t matrixFlags)
{
    if (pArray == nullptr)
        InvalidArgument("Invalid pArray.");

    m_format = matrixFormatDense;
    m_computeDevice = CPUDEVICE;

    // if it's externally managed, then populate the structure
    if (matrixFlags & matrixFlagDontOwnBuffer)
    {
        // free previous array allocation if any before overwriting
        if (m_pArray != nullptr)
            delete[] m_pArray;

        m_pArray = pArray;
        m_numRows = numRows;
        m_numCols = numCols;
        m_elemSizeAllocated = GetNumElements();
        m_externalBuffer = true;
    }
    else
    {
        Resize(numRows, numCols);

        if (!IsEmpty())
        {
            if (!(matrixFlags & matrixFormatRowMajor)) // compatible with internal structure
                memcpy(m_pArray, pArray, GetNumElements() * sizeof(ElemType));
            else // need to transpose
            {
                auto& us = *this;
                if (sizeof(ElemType) == sizeof(double))
                {
#pragma omp parallel for
                    foreach_column (j, us)
                    {
#ifdef USE_ACML
                        dcopy((int) numRows, reinterpret_cast<double*>(pArray + j), (int) numCols, reinterpret_cast<double*>(m_pArray + LocateColumn(j)), 1);
#else
                        cblas_dcopy((int) numRows, reinterpret_cast<double*>(pArray + j), (int) numCols, reinterpret_cast<double*>(m_pArray + LocateColumn(j)), 1);
#endif
                    }
                }
                else
                {
#pragma omp parallel for
                    foreach_column (j, us)
                    {
                        {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
                            scopy((int) numRows, reinterpret_cast<float*>(pArray + j), (int) numCols, reinterpret_cast<float*>(m_pArray + LocateColumn(j)), 1);
#else
                            cblas_scopy((int) numRows, reinterpret_cast<float*>(pArray + j), (int) numCols, reinterpret_cast<float*>(m_pArray + LocateColumn(j)), 1);
#endif
                        }
                    }
                }
            }
        }
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::SetDiagonalValue(const ElemType v)
{
    if (GetNumRows() != GetNumCols())
        LogicError("SetDiagonalValue: NumRows and NumCols do not agree.");

    auto& us = *this;
    long m = (long) GetNumRows();
#pragma omp parallel for
    // four-way unrolling
    for (long i = 0; i < (m & ~3); i += 4)
    {
        us(i, i) = v;
        us(i + 1, i + 1) = v;
        us(i + 2, i + 2) = v;
        us(i + 3, i + 3) = v;
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        us(i, i) = v;
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::SetDiagonalValue(const CPUMatrix<ElemType>& vector)
{
    if (IsEmpty() || vector.IsEmpty())
        LogicError("SetDiagonalValue: Matrix is empty.");

    if (GetNumRows() != GetNumCols())
        LogicError("SetDiagonalValue: NumRows and NumCols do not agree.");

    if (vector.GetNumRows() != 1 && vector.GetNumCols() != 1)
        LogicError("SetDiagonalValue: input vector must be a vector.");

    if (vector.GetNumElements() == 1) // reduce to simple form
        SetDiagonalValue(vector(0, 0));
    else if (vector.GetNumRows() != GetNumRows())
        LogicError("SetDiagonalValue: input vector's dimension does not agree with [this].");
    else
    {
        auto& us = *this;

        long m = (long) GetNumRows();
        if (vector.GetNumRows() == 1) // row vector
        {
#pragma omp parallel for
            // four-way unrolling
            for (long i = 0; i < (m & ~3); i += 4)
            {
                us(i, i) = vector(0, i);
                us(i + 1, i + 1) = vector(0, i + 1);
                us(i + 2, i + 2) = vector(0, i + 2);
                us(i + 3, i + 3) = vector(0, i + 3);
            }
            // handle remaining stuffs
            for (long i = m & ~3; i < m; i++)
            {
                us(i, i) = vector(0, i);
            }
        }
        else
        {
#pragma omp parallel for
            // four-way unrolling
            for (long i = 0; i < (m & ~3); i += 4)
            {
                us(i, i) = vector(i, 0);
                us(i + 1, i + 1) = vector(i + 1, 0);
                us(i + 2, i + 2) = vector(i + 2, 0);
                us(i + 3, i + 3) = vector(i + 3, 0);
            }
            // handle remaining stuffs
            for (long i = m & ~3; i < m; i++)
            {
                us(i, i) = vector(i, 0);
            }
        }
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::SetUniformRandomValue(const ElemType low, const ElemType high, unsigned long seed)
{
    if (IsEmpty())
        LogicError("SetUniformRandomValue: Matrix is empty.");

#ifdef _MSC_VER // TODO: check if available under GCC/Linux
    std::ranlux64_base_01 generator;
    generator.seed(seed == USE_TIME_BASED_SEED ? (unsigned long) time(NULL) : seed);
#else
    std::default_random_engine generator(seed);
#endif
    std::uniform_real_distribution<ElemType> r(low, high);

    long m = (long) GetNumElements();
    // four-way unrolling
    for (long i = 0; i < (m & ~3); i += 4)
    {
        m_pArray[i] = r(generator);
        m_pArray[i + 1] = r(generator);
        m_pArray[i + 2] = r(generator);
        m_pArray[i + 3] = r(generator);
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        m_pArray[i] = r(generator);
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::SetGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed)
{
    if (sigma <= 0)
        InvalidArgument("SetUniformRandomValue: sigma must be a positive value.");

    if (IsEmpty())
        LogicError("SetUniformRandomValue: Matrix is empty.");

    auto& us = *this;
#ifdef _MSC_VER // TODO: check if available under GCC/Linux
    std::ranlux64_base_01 generator;
    generator.seed(seed == USE_TIME_BASED_SEED ? (unsigned long) time(NULL) : seed);
#else
    std::default_random_engine generator(seed);
#endif
    std::normal_distribution<ElemType> r(mean, sigma);
    // #pragma omp parallel for   // is it thread safe?
    foreach_coord (i, j, us)
    {
        us(i, j) = r(generator);
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::AddGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed)
{
    if (sigma <= 0)
        InvalidArgument("SetUniformRandomValue: sigma must be a positive value.");

    if (IsEmpty())
        LogicError("SetUniformRandomValue: Matrix is empty.");

    auto& us = *this;
#ifdef _MSC_VER // TODO: check if available under GCC/Linux
    std::ranlux64_base_01 generator;
    generator.seed(seed == USE_TIME_BASED_SEED ? (unsigned long) time(NULL) : seed);
#else
    std::default_random_engine generator(seed);
#endif
    std::normal_distribution<ElemType> r(mean, sigma);

    long m = (long) GetNumRows(), n = (long) GetNumCols();
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) = r(generator);
            us(i + 1, j) = r(generator);
            us(i + 2, j) = r(generator);
            us(i + 3, j) = r(generator);
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) = r(generator);
        }
    }
}

//maskRate: percentage of values masked out (similar to dropout rate)
//scaleValue: which scale value to set to the left ones (unmasked items).
template <class ElemType>
void CPUMatrix<ElemType>::SetUniformRandomMask(const ElemType maskRate, const ElemType scaleValue, unsigned long seed)
{
    if (IsEmpty())
        LogicError("SetUniformRandomValue: Matrix is empty.");

    auto& us = *this;
#ifdef _MSC_VER // TODO: check if available under GCC/Linux
    std::ranlux64_base_01 generator;
    generator.seed(seed == USE_TIME_BASED_SEED ? (unsigned long) time(NULL) : seed);
#else
    std::default_random_engine generator(seed == USE_TIME_BASED_SEED ? (unsigned long) time(NULL) : seed);
#endif
    std::uniform_real_distribution<ElemType> r(0, 1);

    long m = (long) GetNumRows(), n = (long) GetNumCols();
    ElemType v;
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            v = r(generator);
            us(i, j) = v <= maskRate ? 0 : scaleValue;
            v = r(generator);
            us(i + 1, j) = v <= maskRate ? 0 : scaleValue;
            v = r(generator);
            us(i + 2, j) = v <= maskRate ? 0 : scaleValue;
            v = r(generator);
            us(i + 3, j) = v <= maskRate ? 0 : scaleValue;
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            v = r(generator);
            us(i, j) = v <= maskRate ? 0 : scaleValue;
        }
    }
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::Adagrad(CPUMatrix<ElemType>& gradients, const bool needAveMultiplier)
{
    ElemType aveMultiplier = 0;

    if (IsEmpty() || gradients.GetNumCols() != GetNumCols() || gradients.GetNumRows() != GetNumRows())
    {
        Resize(gradients.GetNumRows(), gradients.GetNumCols());
        SetValue(0.0);
    }

    assert(GetNumRows() == gradients.GetNumRows() && GetNumCols() == gradients.GetNumCols());

    ElemType *a = m_pArray, *d_v = gradients.m_pArray;
    size_t n = GetNumElements();

    const ElemType floor = 1e-16f;
    ElemType a0, a1, a2, a3;

    // disable omp here because aveMultiper needs to be added atomically. however, it seems the result is incorrect even if rmp atomic and amp critical are used.
    // #pragma omp parallel for
    for (long i = 0; i < (n & ~3); i += 4) // four-way unrolling
    {
        a[i] += d_v[i] * d_v[i];
        a[i + 1] += d_v[i + 1] * d_v[i + 1];
        a[i + 2] += d_v[i + 2] * d_v[i + 2];
        a[i + 3] += d_v[i + 3] * d_v[i + 3];

        a0 = sqrt(a[i] + floor);
        a1 = sqrt(a[i + 1] + floor);
        a2 = sqrt(a[i + 2] + floor);
        a3 = sqrt(a[i + 3] + floor);

        d_v[i] /= a0;
        d_v[i + 1] /= a1;
        d_v[i + 2] /= a2;
        d_v[i + 3] /= a3;

        if (needAveMultiplier)
        {
            aveMultiplier += 1 / a0 + 1 / a1 + 1 / a2 + 1 / a3;
        }
    }

    // get the last few elements if any
    for (long i = n & ~3; i < n; i++)
    {
        a[i] += d_v[i] * d_v[i];

        a0 = sqrt(a[i] + floor);
        d_v[i] /= a0;

        if (needAveMultiplier)
        {
            aveMultiplier += 1 / a0;
        }
    }

    if (needAveMultiplier && n > 0)
        return aveMultiplier / n;
    else
        return 1;
}

template <class ElemType>
void CPUMatrix<ElemType>::FSAdagrad(CPUMatrix<ElemType>& gradients,
                                    CPUMatrix<ElemType>& functionValues,
                                    ElemType learnRatePerSample,
                                    ElemType momentum,
                                    ElemType adaWeight,
                                    ElemType adaMul)
{
    size_t numColsNeeded = 2 * gradients.GetNumCols();

    if (IsEmpty() || (GetNumCols() < numColsNeeded))
    {
        Resize(gradients.GetNumRows(), numColsNeeded);
        SetValue(0.0);
    }

    assert((GetNumRows() == gradients.GetNumRows()) && (GetNumCols() == numColsNeeded));

    size_t n = gradients.GetNumElements();
    ElemType* grad = gradients.m_pArray;
    ElemType* smoothAda = m_pArray;
    ElemType* smoothMom = m_pArray + n;
    ElemType* val = functionValues.m_pArray;
#pragma omp parallel for
    // TODO: Unroll 4-times for better performance leveraging vectorization
    for (long i = 0; i < n; i++)
    {
        ElemType g = grad[i];
        ElemType adaSqr = adaWeight * smoothAda[i] + (1.0f - adaWeight) * g * g;
        smoothAda[i] = adaSqr;
        if (adaSqr != 0.0f)
        {
            ElemType ada = sqrt(adaSqr);
            ElemType w = adaMul * ((ElemType) 1.0 / ada);

            if (w > 10.0f)
                w = 10.0f;
            g *= w;
        }

        if (momentum > 0.0f)
        {
            g = momentum * smoothMom[i] + (1.0f - momentum) * g;
            smoothMom[i] = g;
        }

        g *= learnRatePerSample;
        val[i] -= g;
    }
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::RmsProp(CPUMatrix<ElemType>& gradients,
                                      ElemType RMS_GAMMA,
                                      ElemType RMS_WGT_INC,
                                      ElemType RMS_WGT_MAX,
                                      ElemType RMS_WGT_DEC,
                                      ElemType RMS_WGT_MIN,
                                      const bool needAveMultiplier)
{
    const ElemType floor = 1e-6f;

    size_t n = gradients.GetNumElements();
    ElemType* curr_grad = gradients.m_pArray;

    if (IsEmpty() || GetNumCols() < gradients.GetNumCols() * 3)
    {
        Resize(gradients.GetNumRows(), gradients.GetNumCols() * 3);
        SetValue(0.0);

        ElemType* avars = m_pArray;         // accumulated variances for RMS scaling
        ElemType* steps = m_pArray + 2 * n; // current step size

        // initialize moving average of gradient-squared
        for (long i = 0; i < n; i++)
            avars[i] = curr_grad[i] * curr_grad[i];

        // initialize starting step size
        for (long i = 0; i < n; i++)
            steps[i] = ElemType(0.02);
    }

    ElemType* avars = m_pArray;         // accumulated variances for RMS scaling
    ElemType* signs = m_pArray + n;     // sign of previous gradient
    ElemType* steps = m_pArray + 2 * n; // current step size

    assert(GetNumRows() == gradients.GetNumRows() && GetNumCols() == gradients.GetNumCols() * 3);

    ElemType ONE_MINUS_GAMMA = ElemType(1.0) - RMS_GAMMA;
    // int upd[] = {
    //    2,2,0,
    //    2,2,0,
    //    1,1,1,
    //    2,2,0,
    //    1,2,1,
    //    0,2,2,
    //    1,1,1,
    //    0,2,2,
    //    0,2,2,
    // };

    //      for (long i=0; i<n; i++)
    //      {
    //          avars[i] = RMS_GAMMA * avars[i] + ONE_MINUS_GAMMA * (curr_grad[i] * curr_grad[i]);
    //    // grad sign base 3: 0->neg, 1->zero, 2->pos
    //    const int grad_sign = 1 + (ElemType(0) < curr_grad[i]) - (curr_grad[i] < ElemType(0));

    //    // signs[i] contains three consecutive grad_sign
    //    signs[i]  = 3*(int(signs[i]) % 9) + grad_sign;

    //    switch(upd[int(signs[i])])
    //    {
    //    case 0:
    //        steps[i] = max(steps[i] * RMS_WGT_DEC, RMS_WGT_MIN);
    //        break;
    //    case 2:
    //        steps[i] = min(steps[i] * RMS_WGT_INC, RMS_WGT_MAX);
    //        break;
    //    }
    //    curr_grad[i] *= steps[i] / sqrt(avars[i] + floor);
    //      }

    ElemType aveMultiplier = 0, a;
    for (long i = 0; i < n; i++)
    {
        avars[i] = RMS_GAMMA * avars[i] + ONE_MINUS_GAMMA * (curr_grad[i] * curr_grad[i]);
        const int grad_sign = (ElemType(0) < curr_grad[i]) - (curr_grad[i] < ElemType(0));

        if (signs[i] * grad_sign > 0)
            steps[i] = std::min(steps[i] * RMS_WGT_INC, RMS_WGT_MAX);
        else
            steps[i] = std::max(steps[i] * RMS_WGT_DEC, RMS_WGT_MIN);

        a = steps[i] / sqrt(avars[i] + floor);
        curr_grad[i] *= a;
        signs[i] = (ElemType) grad_sign;

        if (needAveMultiplier)
            aveMultiplier += a;
    }

    if (needAveMultiplier)
        return aveMultiplier / n;
    else
        return 1;
}

template <class ElemType>
void CPUMatrix<ElemType>::Reshape(const size_t numRows, const size_t numCols)
{
    assert(numRows * numCols == GetNumElements());
    if (numRows * numCols != GetNumElements())
        InvalidArgument("Reshape: Total number of elements does not match.");

    m_numRows = numRows;
    m_numCols = numCols;
}

// Resize() -- change matrix size
// This function is cheap if the matrix size does not change.
// Current content is not preserved.
// BUGBUG: There is code that relies on zero initialization (without, we get subtle variations of output). That is wrong--we should initialize to QNaN and see where it fails.
// If growOnly is true, resize will not reallocate memory if the current memory is large enough (i.e., will not shrink).
// If this object does not own its memory then new memory cannot be allocated (one can still shrink and/or reshape).
template <class ElemType>
void CPUMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, bool growOnly /*=true*/)
{
    if (m_numRows == numRows && m_numCols == numCols)
        return;

    size_t numElements = numRows * numCols;
    if (numElements > m_elemSizeAllocated ||                 // grow allocation
        (!growOnly && (numElements != m_elemSizeAllocated))) // shrink allocation (not if 'growOnly')
    {
        // reallocate buffer
        ElemType* pArray = nullptr;
        if (numElements > 0)
        {
            if (!OwnBuffer())
                LogicError("Resize: Resizing an matrix you don't own is not supported.");
            pArray = NewArray<ElemType>(numElements);
        }
        // success: update the object
        if (OwnBuffer())
            delete[] m_pArray;
        else
            assert(pArray == nullptr); // (if !OwnBuffer we can still resize to 0)
        m_pArray = pArray;
        m_elemSizeAllocated = numElements;
    }

    // success
    m_numRows = numRows;
    m_numCols = numCols;
}

// allocated by the callee but should be deleted by the caller
// TODO: change to use STL vector instead
template <class ElemType>
ElemType* CPUMatrix<ElemType>::CopyToArray() const
{
    size_t numElements = GetNumElements();
    if (numElements != 0)
    {
        ElemType* arrayCopyTo = NewArray<ElemType>(numElements);
        memcpy(arrayCopyTo, m_pArray, sizeof(ElemType) * numElements);
        return arrayCopyTo;
    }
    else
    {
        return nullptr;
    }
}

//memory will be allocated by the callee if not enough but need to be deleted by the caller after it's done
//return number of elements copied
template <class ElemType>
size_t CPUMatrix<ElemType>::CopyToArray(ElemType*& arrayCopyTo, size_t& currentArraySize) const
{
    size_t numElements = GetNumElements();

    if (numElements > currentArraySize)
    {
        delete arrayCopyTo;
        arrayCopyTo = NewArray<ElemType>(numElements);
        currentArraySize = numElements;
    }

    if (numElements != 0)
    {
        memcpy(arrayCopyTo, m_pArray, sizeof(ElemType) * numElements);
    }

    return numElements;
}

template <typename ElemType>
void CPUMatrix<ElemType>::CopySection(size_t /*numRows*/, size_t /*numCols*/, ElemType* /*dst*/, size_t /*colStride*/) const
{
    // REVIEW alexeyk: currently not used by CPU, but implement when possible.
    RuntimeError("Not implemented.");
}

template <class ElemType>
inline size_t CPUMatrix<ElemType>::LocateColumn(const size_t col) const
{
    assert(col < m_numCols);
    return col * m_numRows; // matrix in column-wise storage
}

template <class ElemType>
inline size_t CPUMatrix<ElemType>::LocateElement(const size_t row, const size_t col) const
{
    assert(row < m_numRows);
    return LocateColumn(col) + row; // matrix in column-wise storage
}

#pragma endregion Basic Operators

#pragma region Member BLAS Functions

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator+=(ElemType alpha)
{
    return AssignSumOf(alpha, *this);
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator+(ElemType alpha) const
{
    CPUMatrix<ElemType> c(GetNumRows(), GetNumCols());
    c.AssignSumOf(alpha, *this);
    return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSumOf(const ElemType alpha, const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSumOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) = alpha + a(i, j);
            us(i + 1, j) = alpha + a(i + 1, j);
            us(i + 2, j) = alpha + a(i + 2, j);
            us(i + 3, j) = alpha + a(i + 3, j);
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) = alpha + a(i, j);
        }
    }

    return *this;
}

//if [this] and a have same dimension then [this]=[this]+a
//if a is a column vector, add to all columns of [this]
//if a is a row vector, add to all rows of [this]
//if a is a scalar, add it to all elements.
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator+=(const CPUMatrix<ElemType>& a)
{
    // if (a.GetNumElements() == 1)
    //    *this += a(0,0);
    // else
    ScaleAndAdd(1, a, *this);

    return *this;
}

//if [this] and a have same dimension then OUTPUT=[this]+a
//if a is a column vector, add to all columns of [this]
//if a is a row vector, add to all rows of [this]
template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator+(const CPUMatrix<ElemType>& a) const
{
    if (GetNumElements() == 1)
    {
        CPUMatrix<ElemType> c(a);
        c += (*this)(0, 0);
        return c;
    }
    else if (a.GetNumElements() == 1)
    {
        CPUMatrix<ElemType> c(*this);
        c += a(0, 0);
        return c;
    }
    else
    {
        CPUMatrix<ElemType> c(*this); // this implementation will introduce a copy overhead. but make resue of the code
        c += a;
        return c;
    }
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSumOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
    if (a.GetNumElements() == 1)
    {
        SetValue(b);
        (*this) += a;
    }
    else
    {
        SetValue(a);
        (*this) += b;
    }
    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator-=(ElemType alpha)
{
    return AssignDifferenceOf(*this, alpha);
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator-(ElemType alpha) const
{
    CPUMatrix<ElemType> c(GetNumRows(), GetNumCols());
    c.AssignDifferenceOf(*this, alpha);
    return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignDifferenceOf(const ElemType alpha, const CPUMatrix<ElemType>& a)
{
    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) = alpha - a(i, j);
            us(i + 1, j) = alpha - a(i + 1, j);
            us(i + 2, j) = alpha - a(i + 2, j);
            us(i + 3, j) = alpha - a(i + 3, j);
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) = alpha - a(i, j);
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignDifferenceOf(const CPUMatrix<ElemType>& a, const ElemType alpha)
{
    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) = a(i, j) - alpha;
            us(i + 1, j) = a(i + 1, j) - alpha;
            us(i + 2, j) = a(i + 2, j) - alpha;
            us(i + 3, j) = a(i + 3, j) - alpha;
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) = a(i, j) - alpha;
        }
    }
    return *this;
}

//if [this] and a have same dimension then [this]=[this]-a
//if a is a column vector, minus it from all columns of [this]
//if a is a row vector, minus it from all rows of [this]
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator-=(const CPUMatrix<ElemType>& a)
{
    ScaleAndAdd(-1, a, *this);

    return *this;
}

//if [this] and a have same dimension then output=[this]-a
//if a is a column vector, minus it from all columns of [this]
//if a is a row vector, minus it from all rows of [this]
template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator-(const CPUMatrix<ElemType>& a) const
{
    CPUMatrix<ElemType> c(*this); // this implementation will introduce a copy overhead. but make resue of the code
    c -= a;
    return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignDifferenceOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
    if (this != &a)
    {
        Resize(a.GetNumRows(), a.GetNumCols());
        SetValue(a);
    }
    (*this) -= b;
    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator*=(ElemType alpha)
{
    Scale(alpha, *this);
    return *this;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator*(ElemType alpha) const
{
    CPUMatrix<ElemType> c(GetNumRows(), GetNumCols());
    Scale(alpha, *this, c);
    return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignProductOf(const ElemType alpha, const CPUMatrix<ElemType>& a)
{
    Scale(alpha, a, *this);
    return *this;
}

// [this]=a*b
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignProductOf(const CPUMatrix<ElemType>& a, const bool transposeA, const CPUMatrix<ElemType>& b, const bool transposeB)
{
    if (a.GetNumElements() == 1)
    {
        if (transposeB)
            AssignTransposeOf(b);
        (*this) *= a(0, 0);
    }
    else if (b.GetNumElements() == 1)
    {
        if (transposeA)
            AssignTransposeOf(a);
        (*this) *= b(0, 0);
    }
    else
        Multiply(a, transposeA, b, transposeB, *this);

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator*(const CPUMatrix<ElemType>& a) const
{
    auto& us = *this;
    if (GetNumElements() == 1)
    {
        CPUMatrix<ElemType> c;
        c.AssignProductOf(us(0, 0), a);
        return c;
    }
    else if (a.GetNumElements() == 1)
    {
        CPUMatrix<ElemType> c;
        c.AssignProductOf(a(0, 0), us);
        return c;
    }
    else
    {
        CPUMatrix<ElemType> c;
        Multiply(*this, a, c);
        return c;
    }
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator/=(ElemType alpha)
{
    (*this) *= 1 / alpha;
    return (*this);
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator/(ElemType alpha) const
{
    return ((*this) * (1 / alpha));
}

//element-wise power
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::operator^=(ElemType alpha)
{
    auto& us = *this;
    ElementWisePower(alpha, us, us);
    return us;
}

//element-wise power
template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::operator^(ElemType alpha) const
{
    CPUMatrix<ElemType> c(GetNumRows(), GetNumCols());
    ElementWisePower(alpha, *this, c);
    return c;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementPowerOf(const CPUMatrix<ElemType>& a, const ElemType power)
{
    ElementWisePower(power, a, *this);
    return *this;
}

//[this]=[this] .* a (we cannot override operator .* in c++)
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::ElementMultiplyWith(const CPUMatrix<ElemType>& a)
{
    return AssignElementProductOf(*this, a);
}

//[this]=[this] .* a (we cannot override operator .* in c++)
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::ElementDivideBy(const CPUMatrix<ElemType>& a)
{
    return AssignElementDivisionOf(*this, a);
}

//[this]=a .* b
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("AssignElementProductOf: The input matrix dimensions do not match.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) = a(i, j) * b(i, j);
            us(i + 1, j) = a(i + 1, j) * b(i + 1, j);
            us(i + 2, j) = a(i + 2, j) * b(i + 2, j);
            us(i + 3, j) = a(i + 3, j) * b(i + 3, j);
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) = a(i, j) * b(i, j);
        }
    }
    return *this;
}

//[this] +=a .* b
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddElementProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AddElementProductOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("AddElementProductOf : The input matrix dimensions do not match.");

    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == GetNumCols()))
        InvalidArgument("AddElementProductOf : The input matrix dimensions do not match [this].");

    auto& us = *this;

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) += a(i, j) * b(i, j);
            us(i + 1, j) += a(i + 1, j) * b(i + 1, j);
            us(i + 2, j) += a(i + 2, j) * b(i + 2, j);
            us(i + 3, j) += a(i + 3, j) * b(i + 3, j);
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) += a(i, j) * b(i, j);
        }
    }

    return *this;
}

//[this]=a ./ b
// TODO: This clips the divisor by a small value. Is that really what one would want?
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementDivisionOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementDivisionOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("AssignElementDivisionOf : The input matrix dimensions do not match.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    ElemType smallValue = EPS_IN_INVERSE;

#pragma omp parallel for
    foreach_coord (i, j, us)
    {
        ElemType v = b(i, j);
        if (v >= 0 && v < smallValue)
            us(i, j) = a(i, j) / smallValue;
        else if (v < 0 && v > -smallValue)
            us(i, j) = a(i, j) / (-smallValue);
        else
            us(i, j) = a(i, j) / v;
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::ColumnElementMultiplyWith(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("ColumnElementMultiplyWith: Matrix is empty.");

    assert(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1);
    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
        InvalidArgument("ColumnElementMultiplyWith: The input matrix should be a col vector and match [this]'s rows.");

    auto& us = *this;

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) *= a(i, 0);
            us(i + 1, j) *= a(i + 1, 0);
            us(i + 2, j) *= a(i + 2, 0);
            us(i + 3, j) *= a(i + 3, 0);
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) *= a(i, 0);
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::RowElementMultiplyWith(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("RowElementMultiplyWith: Matrix is empty.");

    assert(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols());
    if (!(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols()))
        InvalidArgument("RowElementMultiplyWith: The input matrix should be a row vector and match [this]'s columns.");

    auto& us = *this;

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        ElemType v = a(0, j);
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) *= v;
            us(i + 1, j) *= v;
            us(i + 2, j) *= v;
            us(i + 3, j) *= v;
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) *= v;
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::RowElementDivideBy(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("RowElementDivideBy: Matrix is empty.");

    assert(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols());
    if (!(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols()))
        InvalidArgument("RowElementDivideBy: The input matrix should be a row vector and match [this]'s columns.");

    auto& us = *this;

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        ElemType v = a(0, j);
        if (v >= 0 && v < EPS_IN_INVERSE)
            v = EPS_IN_INVERSE;
        else if (v < 0 && v > -EPS_IN_INVERSE)
            v = (-EPS_IN_INVERSE);

        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) /= v;
            us(i + 1, j) /= v;
            us(i + 2, j) /= v;
            us(i + 3, j) /= v;
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) /= v;
        }
    }

    return *this;
}
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::ColumnElementDivideBy(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("ColumnElementDivideBy: Matrix is empty.");

    assert(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1);
    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
        InvalidArgument("ColumnElementDivideBy: The input matrix should be a col vector and match [this]'s rows.");

    auto& us = *this;

    long m = (long) GetNumRows(), n = (long) GetNumCols();

    ElemType smallValue = EPS_IN_INVERSE;
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        for (long i = 0; i < m; i++)
        {
            ElemType v = a(i, 0);
            if (v >= 0 && v < smallValue)
                us(i, j) /= smallValue;
            else if (v < 0 && v > -smallValue)
                us(i, j) /= (-smallValue);
            else
                us(i, j) /= v;
        }
    }

    return *this;
}

//[this]=1 ./ a
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::ElementInverse()
{
    return AssignElementInverseOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementInverseOf(const CPUMatrix<ElemType>& a)
{
    ElemType smallValue = EPS_IN_INVERSE;

    if (a.IsEmpty())
        LogicError("AssignElementInverseOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
    foreach_coord (i, j, us)
    {
        if (a(i, j) < 0 && a(i, j) > -smallValue)
            us(i, j) = 1 / (-smallValue);
        else if (a(i, j) >= 0 && a(i, j) < smallValue)
            us(i, j) = 1 / smallValue;
        else
            us(i, j) = 1 / a(i, j);
    }

    return *this;
}

//[this]=sigmoid([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceSigmoid()
{
    return AssignSigmoidOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSigmoidOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSigmoidOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
    foreach_coord (i, j, us)
    {
        if (a(i, j) >= 0)
            us(i, j) = 1 / (1 + exp(-a(i, j)));
        else
        {
            ElemType v = exp(a(i, j));
            us(i, j) = v / (1 + v);
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceLinearRectifierDerivative()
{
    return AssignLinearRectifierDerivativeOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignLinearRectifierDerivativeOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignLinearRectifierDerivativeOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) = a(i, j) > 0.0f ? 1.0f : 0.0f;
            us(i + 1, j) = a(i + 1, j) > 0.0f ? 1.0f : 0.0f;
            us(i + 2, j) = a(i + 2, j) > 0.0f ? 1.0f : 0.0f;
            us(i + 3, j) = a(i + 3, j) > 0.0f ? 1.0f : 0.0f;
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) = a(i, j) > 0.0f ? 1.0f : 0.0f;
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceSigmoidDerivative()
{
    return AssignSigmoidDerivativeOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSigmoidDerivativeOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSigmoidDerivativeOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            ElemType v = a(i, j);
            us(i, j) = v * (1 - v);

            ElemType v1 = a(i + 1, j);
            us(i + 1, j) = v1 * (1 - v1);

            ElemType v2 = a(i + 2, j);
            us(i + 2, j) = v2 * (1 - v2);

            ElemType v3 = a(i + 3, j);
            us(i + 3, j) = v3 * (1 - v3);
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            ElemType v = a(i, j);
            us(i, j) = v * (1 - v);
        }
    }

    return *this;
}

//[this]=tanh([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceTanh()
{
    return AssignTanhOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignTanhOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignTanhOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) = tanh(a(i, j));
            us(i + 1, j) = tanh(a(i + 1, j));
            us(i + 2, j) = tanh(a(i + 2, j));
            us(i + 3, j) = tanh(a(i + 3, j));
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) = tanh(a(i, j));
        }
    }

    return *this;
}

//[this]=softmax([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceLogSoftmax(const bool isColWise)
{
    return AssignLogSoftmaxOf(*this, isColWise);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignLogSoftmaxOf(const CPUMatrix<ElemType>& a, const bool isColWise)
{
    if (a.IsEmpty())
        LogicError("AssignLogSoftmaxOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    if (isColWise)
    {
#pragma omp parallel for
        foreach_column (j, a)
        {
            // we need to extract max before applying exp to avoid overflow
            ElemType maxV = a(0, j);
            foreach_row (i, a)
                maxV = std::max(maxV, a(i, j));

            ElemType sum = 0;
            foreach_row (i, a)
                sum += exp(us(i, j) = a(i, j) - maxV);
            sum = log(sum);
            foreach_row (i, us)
                us(i, j) -= sum;
        }
    }
    else
    {
#pragma omp parallel for
        foreach_row (i, a)
        {
            // we need to extract max before applying exp to avoid overflow
            ElemType maxV = a(i, 0);
            foreach_column (j, a)
                maxV = std::max(maxV, a(i, j));

            ElemType sum = 0;
            foreach_column (j, a)
                sum += exp(us(i, j) = a(i, j) - maxV);
            sum = log(sum);
            foreach_column (j, us)
                us(i, j) -= sum;
        }
    }

    return *this;
}

//[this]=hardmax([this])
//the max element is 1 else is 0
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceHardmax(const bool isColWise)
{
    return AssignHardmaxOf(*this, isColWise);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignHardmaxOf(const CPUMatrix<ElemType>& a, const bool isColWise)
{
    if (a.IsEmpty())
        LogicError("AssignHardmaxOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    if (isColWise)
    {
#pragma omp parallel for
        foreach_column (j, a)
        {
            // we need to extract max
            ElemType maxV = a(0, j);
            long maxI = 0;
            foreach_row (i, a)
            {
                if (maxV < a(i, j))
                {
                    maxV = a(i, j);
                    maxI = i;
                }
            }

            foreach_row (i, us)
                us(i, j) = (i == maxI) ? 1.0f : 0.0f;
        }
    }
    else
    {
#pragma omp parallel for
        foreach_row (i, a)
        {
            // we need to extract max
            ElemType maxV = a(i, 0);
            long maxJ = 0;
            foreach_column (j, a)
            {
                if (maxV < a(i, j))
                {
                    maxV = a(i, j);
                    maxJ = j;
                }
            }

            foreach_column (j, us)
                us(i, j) = (j == maxJ) ? 1.0f : 0.0f;
        }
    }

    return *this;
}

//[this]=sqrt([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceSqrt()
{
    return AssignSqrtOf(*this);
}

//to prevent negative values caused by floating operations, we force inputs to be >=0
//this may, however, hide problems in the caller.
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSqrtOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSqrtOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j)     = sqrt(max((ElemType)0, a(i, j)));
            us(i + 1, j) = sqrt(max((ElemType)0, a(i + 1, j)));
            us(i + 2, j) = sqrt(max((ElemType)0, a(i + 2, j)));
            us(i + 3, j) = sqrt(max((ElemType)0, a(i + 3, j)));
        }
        // remaining
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) = sqrt(max((ElemType)0, a(i, j)));
        }
    }

    return *this;
}

//[this]=exp([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceExp()
{
    return AssignExpOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignExpOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignExpOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) = exp(a(i, j));
            us(i + 1, j) = exp(a(i + 1, j));
            us(i + 2, j) = exp(a(i + 2, j));
            us(i + 3, j) = exp(a(i + 3, j));
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) = exp(a(i, j));
        }
    }

    return *this;
}

//[this]=exp([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceAbs()
{
    return AssignAbsOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignAbsOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignAbsOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            us(i, j) = abs(a(i, j));
            us(i + 1, j) = abs(a(i + 1, j));
            us(i + 2, j) = abs(a(i + 2, j));
            us(i + 3, j) = abs(a(i + 3, j));
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            us(i, j) = abs(a(i, j));
        }
    }

    return *this;
}

//[this]=log([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceLog()
{
    return AssignLogOf(*this);
}

//[this]=log([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceLog10()
{
    return AssignLog10Of(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignLogOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignLogOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
    foreach_coord (i, j, a)
    {
        const ElemType v = a(i, j);
        if (v < EPS_IN_LOG)
        {
            us(i, j) = LOG_OF_EPS_IN_LOG;
        }
        else
            us(i, j) = log(v);
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignLog10Of(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignLogOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
    foreach_coord (i, j, a)
    {
        const ElemType v = a(i, j);
        if (v <= 0)
            LogicError("AssignLogOf: Log can only applied to numbers larger than 0.");
        else if (v < EPS_IN_LOG)
        {
            us(i, j) = LOG10_OF_EPS_IN_LOG;
        }
        else
            us(i, j) = log10(v);
    }

    return *this;
}

//[this]=cos([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceCosine()
{
    return AssignCosineOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignCosineOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignCosineOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
    foreach_coord (i, j, a)
    {
        const ElemType v = a(i, j);
        us(i, j) = cos(v);
    }

    return *this;
}

//[this]=-sin([this]) element wise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceNegativeSine()
{
    return AssignNegativeSineOf(*this);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignNegativeSineOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignCosineOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
    foreach_coord (i, j, a)
    {
        const ElemType v = a(i, j);
        us(i, j) = -sin(v);
    }

    return *this;
}

//Threshold truncating: this[i] = max( this[i], threshold )
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceTruncateBottom(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("InplaceTruncateBottom: Matrix is empty.");

    auto& us = *this;

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            if (us(i, j) < threshold)
                us(i, j) = threshold;

            if (us(i + 1, j) < threshold)
                us(i + 1, j) = threshold;

            if (us(i + 2, j) < threshold)
                us(i + 2, j) = threshold;

            if (us(i + 3, j) < threshold)
                us(i + 3, j) = threshold;
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            if (us(i, j) < threshold)
                us(i, j) = threshold;
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceTruncate(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("InplaceTruncate: Matrix is empty.");

    auto& us = *this;
    ElemType locThresholdPos = abs(threshold);
    ElemType locTHresholdNeg = -locThresholdPos;

    long m = (long) GetNumRows(), n = (long) GetNumCols();
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        // four-way unrolling
        for (long i = 0; i < (m & ~3); i += 4)
        {
            if (us(i, j) > locThresholdPos)
                us(i, j) = locThresholdPos;
            else if (us(i, j) < locTHresholdNeg)
                us(i, j) = locTHresholdNeg;

            if (us(i + 1, j) > locThresholdPos)
                us(i + 1, j) = locThresholdPos;
            else if (us(i + 1, j) < locTHresholdNeg)
                us(i + 1, j) = locTHresholdNeg;

            if (us(i + 2, j) > locThresholdPos)
                us(i + 2, j) = locThresholdPos;
            else if (us(i + 2, j) < locTHresholdNeg)
                us(i + 2, j) = locTHresholdNeg;

            if (us(i + 3, j) > locThresholdPos)
                us(i + 3, j) = locThresholdPos;
            else if (us(i + 3, j) < locTHresholdNeg)
                us(i + 3, j) = locTHresholdNeg;
        }
        // handle remaining stuffs
        for (long i = m & ~3; i < m; i++)
        {
            if (us(i, j) > locThresholdPos)
                us(i, j) = locThresholdPos;
            else if (us(i, j) < locTHresholdNeg)
                us(i, j) = locTHresholdNeg;
        }
    }

    return *this;
}

//x= x-threshold if x>threshold, x+threshold if x<-threshold, 0 otherwise
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceSoftThreshold(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("InplaceTruncate: Matrix is empty.");

    long m = (long) GetNumElements();

#pragma omp parallel for
    for (long i = 0; i < (m & ~3); i += 4) // four-way unrolling
    {
        if (m_pArray[i] > threshold)
            m_pArray[i] -= threshold;
        else if (m_pArray[i] < -threshold)
            m_pArray[i] += threshold;
        else
            m_pArray[i] = 0;

        if (m_pArray[i + 1] > threshold)
            m_pArray[i + 1] -= threshold;
        else if (m_pArray[i + 1] < -threshold)
            m_pArray[i + 1] += threshold;
        else
            m_pArray[i + 1] = 0;

        if (m_pArray[i + 2] > threshold)
            m_pArray[i + 2] -= threshold;
        else if (m_pArray[i + 2] < -threshold)
            m_pArray[i + 2] += threshold;
        else
            m_pArray[i + 2] = 0;

        if (m_pArray[i + 3] > threshold)
            m_pArray[i + 3] -= threshold;
        else if (m_pArray[i + 3] < -threshold)
            m_pArray[i + 3] += threshold;
        else
            m_pArray[i + 3] = 0;
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        if (m_pArray[i] > threshold)
            m_pArray[i] -= threshold;
        else if (m_pArray[i] < -threshold)
            m_pArray[i] += threshold;
        else
            m_pArray[i] = 0;
    }

    return *this;
}

//Threshold truncating: this[i] = max( a[i], threshold )
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignTruncateBottomOf(const CPUMatrix<ElemType>& a, const ElemType threshold)
{
    if (a.IsEmpty())
        LogicError("AssignTruncateBottomOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
    foreach_coord (i, j, a)
    {
        if (a(i, j) < threshold)
            us(i, j) = threshold;
        else
            us(i, j) = a(i, j);
    }

    return *this;
}

//Threshold truncating: this[i] = min( this[i], threshold )
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::InplaceTruncateTop(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("InplaceTruncateTop: Matrix is empty.");

    auto& us = *this;

#pragma omp parallel for
    foreach_coord (i, j, us)
    {
        if (us(i, j) > threshold)
            us(i, j) = threshold;
    }

    return *this;
}

//Threshold truncating: this[i] = min( a[i], threshold )
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignTruncateTopOf(const CPUMatrix<ElemType>& a, const ElemType threshold)
{
    if (a.IsEmpty())
        LogicError("AssignTruncateTopOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
    foreach_coord (i, j, a)
    {
        if (a(i, j) > threshold)
            us(i, j) = threshold;
        else
            us(i, j) = a(i, j);
    }

    return *this;
}
//Threshold truncating: this[i] = 0 if abs(this[i]<threshold).

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::SetToZeroIfAbsLessThan(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("SetToZeroIfAbsLessThan: Matrix is empty.");

    auto& us = *this;

#pragma omp parallel for
    foreach_coord (i, j, us)
    {
        if (abs(us(i, j)) < threshold)
            us(i, j) = 0;
    }

    return *this;
}

//sum of all abs(elements)
template <class ElemType>
ElemType CPUMatrix<ElemType>::SumOfAbsElements() const
{
    if (IsEmpty())
        LogicError("SumOfAbsElements: Matrix is empty.");

    if (sizeof(ElemType) == sizeof(double))
    {
#ifdef USE_ACML
        return (ElemType) dasum((int) GetNumElements(), reinterpret_cast<double*>(m_pArray), 1);
#else
        return (ElemType) cblas_dasum((int) GetNumElements(), reinterpret_cast<double*>(m_pArray), 1);
#endif
    }
    else
    {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
        return sasum((int) GetNumElements(), reinterpret_cast<float*>(m_pArray), 1);
#else
        return cblas_sasum((int) GetNumElements(), reinterpret_cast<float*>(m_pArray), 1);
#endif
    }
}

//sum of all elements
template <class ElemType>
ElemType CPUMatrix<ElemType>::SumOfElements() const
{
    if (IsEmpty())
        LogicError("SumOfElements: Matrix is empty.");

    ElemType sum = 0;
    long m = (long) GetNumElements(); // note: OpenMP requires loop indices to be long, not size_t

//four-way unrolling
#pragma omp parallel for reduction(+ : sum)
    for (long i = 0; i < (m & ~3); i += 4)
    {
        sum += m_pArray[i] + m_pArray[i + 1] + m_pArray[i + 2] + m_pArray[i + 3];
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        sum += m_pArray[i];
    }

    return sum;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSumOfElements(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSumOfElements: Matrix a is empty.");

    auto& us = *this;
    us.Resize(1, 1);
    us(0, 0) = a.SumOfElements();

    return *this;
}

template <class ElemType>
bool CPUMatrix<ElemType>::IsEqualTo(const CPUMatrix<ElemType>& a, const ElemType threshold /*= 1e-8*/) const
{
    return AreEqual(*this, a, threshold);
}

template <class ElemType>
void CPUMatrix<ElemType>::VectorSum(const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c, const bool isColWise)
{
    if (a.IsEmpty())
        LogicError("VectorSum:  Input matrix a is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();

    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    if (isColWise) // col-wise
    {
        c.Resize(1, n);

#pragma omp parallel for
        foreach_column (j, a)
        {
            ElemType v = 0;
            foreach_row (i, a)
            {
#pragma omp atomic
                v += a(i, j);
            }
            c(0, j) = v;
        }
    }
    else
    {
        c.Resize(m, 1);

#pragma omp parallel for
        foreach_row (i, a)
        {
            ElemType v = 0;
            foreach_column (j, a)
            {
#pragma omp atomic
                v += a(i, j);
            }
            c(i, 0) = v;
        }
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::VectorNorm1(CPUMatrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorNorm1: Matrix is empty.");

    auto& us = *this;

    const int m = (int) us.GetNumRows();
    const int n = (int) us.GetNumCols();

    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    if (isColWise) // col-wise
    {
        c.Resize(1, n);

#pragma omp parallel for
        foreach_column (j, us)
        {
            ElemType v = 0;
            foreach_row (i, us)
            {
#pragma omp atomic
                v += abs(us(i, j));
            }
            c(0, j) = v;
        }
    }
    else
    {
        c.Resize(m, 1);

#pragma omp parallel for
        foreach_row (i, us)
        {
            ElemType v = 0;
            foreach_column (j, us)
            {
#pragma omp atomic
                v += abs(us(i, j));
            }
            c(i, 0) = v;
        }
    }
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignVectorNorm1Of(CPUMatrix<ElemType>& a, const bool isColWise)
{
    a.VectorNorm1(*this, isColWise);
    return *this;
}

template <class ElemType>
void CPUMatrix<ElemType>::VectorNorm2(CPUMatrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorNorm2: Matrix is empty.");

    auto& us = *this;

    const int m = (int) us.GetNumRows();
    const int n = (int) us.GetNumCols();

    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    if (isColWise) // col-wise
    {
        c.Resize(1, n);

        if (sizeof(ElemType) == sizeof(double))
        {
#pragma omp parallel for
            foreach_column (j, c)
            {
#ifdef USE_ACML
                c(0, j) = (ElemType) dnrm2(m, reinterpret_cast<double*>(us.m_pArray + us.LocateColumn(j)), 1);
#else
                c(0, j) = (ElemType) cblas_dnrm2(m, reinterpret_cast<double*>(us.m_pArray + us.LocateColumn(j)), 1);
#endif
            }
        }
        else
        {
#pragma omp parallel for
            foreach_column (j, c)
            {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
                c(0, j) = snrm2(m, reinterpret_cast<float*>(us.m_pArray + us.LocateColumn(j)), 1);
#else
                c(0, j) = cblas_snrm2(m, reinterpret_cast<float*>(us.m_pArray + us.LocateColumn(j)), 1);
#endif
            }
        }
    }
    else
    {
        c.Resize(m, 1);

        if (sizeof(ElemType) == sizeof(double))
        {
#pragma omp parallel for
            foreach_row (i, c)
            {
#ifdef USE_ACML
                c(i, 0) = dnrm2(n, reinterpret_cast<double*>(us.m_pArray + i), m);
#else
                c(i, 0) = cblas_dnrm2(n, reinterpret_cast<double*>(us.m_pArray + i), m);
#endif
            }
        }
        else
        {
#pragma omp parallel for
            foreach_row (i, c)
            {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
                c(i, 0) = snrm2(n, reinterpret_cast<float*>(us.m_pArray + i), m);
#else
                c(i, 0) = cblas_snrm2(n, reinterpret_cast<float*>(us.m_pArray + i), m);
#endif
            }
        }
    }
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignVectorNorm2Of(CPUMatrix<ElemType>& a, const bool isColWise)
{
    a.VectorNorm2(*this, isColWise);
    return *this;
}

template <class ElemType>
void CPUMatrix<ElemType>::VectorNormInf(CPUMatrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorNormInf: Matrix is empty.");

    auto& us = *this;

    const int m = (int) us.GetNumRows();
    const int n = (int) us.GetNumCols();

    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    if (isColWise) // col-wise
    {
        c.Resize(1, n);

        // #pragma omp parallel for
        foreach_column (j, us)
        {
            ElemType v = 0;
            foreach_row (i, us)
            {
                v = std::max(v, abs(us(i, j)));
            }
            c(0, j) = v;
        }
    }
    else
    {
        c.Resize(m, 1);

        // #pragma omp parallel for
        foreach_row (i, us)
        {
            ElemType v = 0;
            foreach_column (j, us)
            {
                v = std::max(v, abs(us(i, j)));
            }
            c(i, 0) = v;
        }
    }
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignVectorNormInfOf(CPUMatrix<ElemType>& a, const bool isColWise)
{
    a.VectorNormInf(*this, isColWise);
    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignInnerProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const bool isColWise)
{
    InnerProduct(a, b, *this, isColWise);
    return *this;
}

//column-wise crossproduct
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignKhatriRaoProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignKhatriRaoProductOf: Matrix is empty.");

    long cols = (long) a.GetNumCols();
    assert(cols == b.GetNumCols());
    if (cols != b.GetNumCols())
        InvalidArgument("a.GetNumCols() != b.GetNumCols()");

    long rowsA = (long) a.GetNumRows();
    long rowsB = (long) b.GetNumRows();
    Resize(rowsA * rowsB, cols);

#ifdef __INTEL_COMPILER // TODO: check this
#pragma simd statement
#endif
#pragma omp parallel for
    for (long k = 0; k < cols; k++)
    {
        long jj = 0;
        for (long j = 0; j < rowsB; j++)
        {
            for (long i = 0; i < rowsA; i++)
            {
                (*this)(jj++, k) = a(i, k) * b(j, k);
            }
        }
    }

    return *this;
}

//column-wise reshaped product. Used to compute KhatriRaoProduct Gradient
//   this = reshape each column of a from (K1xK2,1) to (K1, K2)
//   if each column of a is not transposed, each (K1, K2) times each column of b (K2, frames).
//   the output is a (K1, frames) matrix
//   if each column of a is tranposed, each (K1, K2)^T times each column of b(K1, frames) and output is (K2, frames)
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddColumnReshapeProductOf(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const bool transposeAColumn)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AddColumnReshapeProductOf: Matrix is empty.");

    long cols = (long) a.GetNumCols();
    assert(cols == b.GetNumCols());
    if (cols != b.GetNumCols())
        InvalidArgument("AddColumnReshapeProductOf: a.GetNumCols() != b.GetNumCols()");

    long rowsA = (long) a.GetNumRows();
    long rowsB = (long) b.GetNumRows();

    if (rowsA % rowsB != 0)
        InvalidArgument("AddColumnReshapeProductOf: number of rows in a should be multiples of that in b.");

    long rowsC = rowsA / rowsB;
    if (rowsC != GetNumRows() || cols != GetNumCols())
        InvalidArgument("AddColumnReshapeProductOf: This matrix does not have the right size.");

    auto& us = *this;

    if (transposeAColumn)
    {
        // find nrows and ncols of tbe reshaped a
        long nrows = rowsB;
        long ncols = rowsC;

#ifdef __INTEL_COMPILER // TODO: check this
#pragma simd statement
#endif
#pragma omp parallel for
        foreach_column (t, a)
        {
            size_t k = 0;
            for (size_t j = 0; j < ncols; j++) // row and col is transposed
            {
                ElemType v = 0;
                for (size_t i = 0; i < nrows; i++)
                {
                    v += a(k, t) * b(i, t);
                    k++;
                }
                us(j, t) += v;
            }
        }
    }
    else
    {
        size_t ncols = rowsB;
        size_t nrows = rowsC;

#ifdef __INTEL_COMPILER // TODO: check this
#pragma simd statement
#endif
#pragma omp parallel for
        foreach_column (t, a)
        {
            size_t k = 0;
            for (size_t j = 0; j < ncols; j++)
            {
                for (size_t i = 0; i < nrows; i++)
                {
                    us(i, t) += a(k, t) * b(j, t);
                    k++;
                }
            }
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddWithScaleOf(ElemType alpha, const CPUMatrix<ElemType>& a)
{
    ScaleAndAdd(alpha, a, *this);
    return *this;
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::FrobeniusNorm() const
{
    if (IsEmpty())
        LogicError("FrobeniusNorm: Matrix is empty.");

    ElemType v = 0;

    long m = (long) GetNumElements();

//four-way unrolling
#pragma omp parallel for reduction(+ : v)
    for (long i = 0; i < (m & ~3); i += 4)
    {
        v += m_pArray[i] * m_pArray[i] + m_pArray[i + 1] * m_pArray[i + 1] + m_pArray[i + 2] * m_pArray[i + 2] + m_pArray[i + 3] * m_pArray[i + 3];
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        v += m_pArray[i] * m_pArray[i];
    }

    return sqrt(v);
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignFrobeniusNormOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignFrobeniusNormOf: Matrix a is empty.");

    auto& us = *this;
    us.Resize(1, 1);
    us(0, 0) = a.FrobeniusNorm();

    return us;
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::MatrixNormInf() const
{
    if (IsEmpty())
        LogicError("MatrixNormInf: Matrix is empty.");

    auto& us = *this;

    ElemType v = 0;
#pragma omp parallel for
    foreach_coord (i, j, us)
    {
#pragma omp critical
        {
            v = std::max(v, abs(us(i, j)));
        }
    }
    return v;
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::MatrixNorm0() const
{
    if (IsEmpty())
        LogicError("MatrixNorm0: Matrix is empty.");

    auto& us = *this;

    ElemType v = 0;
#pragma omp parallel for
    foreach_coord (i, j, us)
    {
        if (us(i, j) != 0)
        {
#pragma omp critical
            {
                ++v;
            }
        }
    }
    return v;
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::MatrixNorm1() const
{
    if (IsEmpty())
        LogicError("MatrixNorm1: Matrix is empty.");

    auto& us = *this;

    ElemType sum = 0;
#pragma omp parallel for reduction(+ : sum)
    foreach_coord (i, j, us)
    {
        sum += abs(us(i, j));
    }
    return sum;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSignOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSignOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
    foreach_column (j, us)
    {
        foreach_row (i, us)
        {
            ElemType v = a(i, j);
            if (!std::isnan(v))
                us(i, j) = (v == (ElemType) 0 ? (ElemType) 0 : (v > 0 ? (ElemType) 1 : (ElemType)(-1)));
            else
                us(i, j) = v;
        }
    }

    return us;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddSignOf(const CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AddSignOf: Matrix a is empty.");

    auto& us = *this;
    if (this != &a)
        Resize(a.GetNumRows(), a.GetNumCols());

#pragma omp parallel for
    foreach_column (j, us)
    {
        foreach_row (i, us)
        {
            ElemType v = a(i, j);
            if (!std::isnan(v))
                us(i, j) += (v == (ElemType) 0 ? (ElemType) 0 : (v > 0 ? (ElemType) 1 : (ElemType)(-1)));
            else
                us(i, j) = v;
        }
    }

    return us;
}
//I decided to use CPUMatrix<ElemType>& maxIndexes instead of integer vector because the result may be used to do additional calculation
template <class ElemType>
void CPUMatrix<ElemType>::VectorMax(CPUMatrix<ElemType>& maxIndexes, CPUMatrix<ElemType>& maxValues, const bool isColWise, int topK) const
{
    if (IsEmpty())
        LogicError("VectorMax: Matrix is empty.");

    auto& us = *this;
    const int m = (int) GetNumRows();
    const int n = (int) GetNumCols();
    assert(topK <= m);

    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    if (isColWise) // col-wise
    {
        maxValues.Resize(topK, n);
        maxIndexes.Resize(topK, n);

        if (topK == 1)
        {
#pragma omp parallel for
            for (int j = 0; j < n; j++)
            {
                ElemType v = us(0, j);
                size_t index = 0;
                foreach_row (i, us)
                {
                    if (v < us(i, j))
                    {
                        index = i;
                        v = us(i, j);
                    }
                }
                maxValues(0, j) = v;
                maxIndexes(0, j) = (ElemType) index;
            }
        }
        else
        {
            std::vector<int> indices(m);
            int i = 0;
            std::generate(indices.begin(), indices.end(), [&i]
                          {
                              return i++;
                          });

            const ElemType* curVal = m_pArray;
            ElemType* curIdx = maxIndexes.m_pArray;
            ElemType* curMax = maxValues.m_pArray;
            for (int icol = 0; icol < n; icol++, curVal += m, curIdx += topK, curMax += topK)
            {
                // Partial sort, descending order.
                std::nth_element(indices.begin(), indices.begin() + topK, indices.end(),
                                 [curVal](const int& a, const int& b)
                                 {
                                     return curVal[a] > curVal[b];
                                 });
                // REVIEW alexeyk: the following produces warning (see SCL_SECURE_NO_WARNINGS) so use loop instead.
                // std::transform(indices.begin(), indices.begin() + topK, curIdx, [](const int& a) { return static_cast<ElemType>(a); });
                for (int i = 0; i < topK; i++)
                {
                    curIdx[i] = static_cast<ElemType>(indices[i]);
                    curMax[i] = curVal[indices[i]];
                }
            }
        }
    }
    else
    {
        if (topK > 1)
            RuntimeError("Row-wise TopK max is not supported.");

        maxValues.Resize(m, 1);
        maxIndexes.Resize(m, 1);

#pragma omp parallel for
        for (int i = 0; i < m; i++)
        {
            ElemType v = us(i, 0);
            size_t index = 0;
            foreach_column (j, us)
            {
                if (v < us(i, j))
                {
                    index = j;
                    v = us(i, j);
                }
            }
            maxValues(i, 0) = v;
            maxIndexes(i, 0) = (ElemType) index;
        }
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::VectorMin(CPUMatrix<ElemType>& minIndexes, CPUMatrix<ElemType>& minValues, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorMin: Matrix is empty.");

    auto& us = *this;
    const int m = (int) GetNumRows();
    const int n = (int) GetNumCols();

    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    if (isColWise) // col-wise
    {
        minValues.Resize(1, n);
        minIndexes.Resize(1, n);

#pragma omp parallel for
        for (int j = 0; j < n; j++)
        {
            ElemType v = us(0, j);
            size_t index = 0;
            foreach_row (i, us)
            {
                if (v > us(i, j))
                {
                    index = i;
                    v = us(i, j);
                }
            }
            minValues(0, j) = v;
            minIndexes(0, j) = (ElemType) index;
        }
    }
    else
    {
        minValues.Resize(m, 1);
        minIndexes.Resize(m, 1);

#pragma omp parallel for
        for (int i = 0; i < m; i++)
        {
            ElemType v = us(i, 0);
            size_t index = 0;
            foreach_column (j, us)
            {
                if (v > us(i, j))
                {
                    index = j;
                    v = us(i, j);
                }
            }
            minValues(i, 0) = v;
            minIndexes(i, 0) = (ElemType) index;
        }
    }
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignNumOfDiff(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, bool searchInCol)
{
    if (a.GetNumCols() != b.GetNumCols())
        throw std::invalid_argument("AssignNumOfDiff: a and b must have the same number of columns.");
    if (!searchInCol && a.GetNumRows() != b.GetNumRows())
        throw std::invalid_argument("AssignNumOfDiff: a and b must have the same number of rows.");

    ElemType n = 0;
    if (!searchInCol)
    {
        foreach_coord (i, j, a)
        {
            n += (a(i, j) != b(i, j));
        }
    }
    else
    {
        size_t crow = b.GetNumRows();
        const ElemType* curCol = b.m_pArray;
        for (size_t icol = 0; icol < a.GetNumCols(); icol++, curCol += crow)
        {
            auto res = std::find(curCol, curCol + crow, a(0, icol));
            if (res == curCol + crow)
                n++;
        }
    }

    Resize(1, 1); // result should be one element
    (*this)(0, 0) = n;

    return *this;
}

#pragma endregion Member BLAS Functions

#pragma region Other helper Functions

struct PrintRange
{
    // print from begin to skipBegin, then from skipEnd to end
    // skipBegin = end if no split
    size_t begin;
    size_t skipBegin;
    size_t skipEnd;
    size_t end;
    bool IsEmpty() const { return end <= begin; }

    // examples:
    //  * 3..10
    //  * -3..-3: include end-3..end and 0..3
    PrintRange(ptrdiff_t first, ptrdiff_t last, size_t total)
    {
        if (first >= 0 && last >= 0)
        {
            begin = (size_t)first;
            end = (size_t)last + 1;
            if (end > total)    // allow INT_MAX, meaning to end
                end = total;
            skipBegin = end;
            skipEnd = end;
        }
        else if (first < 0 && last < 0)
        {
            begin = 0;
            skipBegin = (size_t)(-last);
            skipEnd = (size_t)(total + first);
            if (skipEnd <= skipBegin)
                skipBegin = skipEnd = total;
            end = total;
        }
        else    // if other combinations are ever of interest then implement them here
            LogicError("Print: Bounds must be either both positive or both negative.");
    }
};

// use negative ranges to print corners, e.g. Print("name", -3, -3, -3, -3) will print the first 3 and last 3 rows/cols
template <class ElemType>
void CPUMatrix<ElemType>::Print(const char* matrixName, ptrdiff_t rowFirst, ptrdiff_t rowLast, ptrdiff_t colFirst, ptrdiff_t colLast) const
{
    fprintf(stderr, "\n###### ");
    if (matrixName != nullptr)
        fprintf(stderr, "%s ", matrixName);
    fprintf(stderr, "(%lu, %lu)", GetNumRows(), GetNumCols());
    if (rowFirst != 0 || colFirst != 0 || (size_t)(rowLast + 1) != GetNumRows() || (size_t)(colLast + 1) != GetNumCols())
        fprintf(stderr, " [%ld:%ld, %ld:%ld]", rowFirst, rowLast, colFirst, colLast);
    fprintf(stderr, " ######\n\n");

    if (IsEmpty())
    {
        fprintf(stderr, "(empty)\n");
        return;
    }

    PrintRange rowRange(rowFirst, rowLast, GetNumRows());
    PrintRange colRange(colFirst, colLast, GetNumCols());

    if (rowRange.IsEmpty() || colRange.IsEmpty())
    {
        fprintf(stderr, "(empty)\n");
        return;
    }

    const auto& us = *this;
    if (rowRange.begin > 0)
        fprintf(stderr, "...\n");
    for (size_t i = rowRange.begin; i < rowRange.end; i++)
    {
        if (i == rowRange.skipBegin)        // insert ... between the two blocks if any
        {
            fprintf(stderr, "...\n");
            i = rowRange.skipEnd;
        }
        if (colRange.begin > 0)             // ... at line start
            fprintf(stderr, "...\t");
        for (size_t j = colRange.begin; j < colRange.end; j++)
        {
            if (j == colRange.skipBegin)
            {
                fprintf(stderr, "...\t");
                j = colRange.skipEnd;
            }
            fprintf(stderr, "%.10f\t", us(i, j));
        }
        if (colRange.end < GetNumCols())    // ... at line end
            fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }
    if (rowRange.end < GetNumRows())
        fprintf(stderr, "...\n");
}

template <class ElemType>
void CPUMatrix<ElemType>::Print(const char* matrixName /*=nullptr*/) const
{
    Print(matrixName, 0, GetNumRows() - 1, 0, GetNumCols() - 1);
}

// file I/O
//matrixName is used to verify that correct matrix is read.
template <class ElemType>
void CPUMatrix<ElemType>::ReadFromFile(FILE*, const char* /*matrixName*/)
{
    RuntimeError("not implemented.");
}

//matrixName is used to verify that correct matrix is read.
template <class ElemType>
void CPUMatrix<ElemType>::WriteToFile(FILE*, const char* /*matrixName*/)
{
    RuntimeError("not implemented.");
}

//assume each column is an input sample. Each sample is stored in [channel, row, col]  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignPackedConvolutionInput(const CPUMatrix<ElemType>& inputSubBatch,
                                                                       const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                                       const size_t outputWidth, const size_t outputHeight, const size_t /*outputChannels*/,
                                                                       const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                                       const bool zeroPadding)
{
    assert(verticalSubsample <= kernelHeight && horizontalSubsample <= kernelWidth);

    const size_t packedInputRows = kernelWidth * kernelHeight * inputChannels;
    const size_t packedInputColsPerSample = outputWidth * outputHeight; // output size per channel
    const size_t inputDim = inputWidth * inputHeight * inputChannels;
    const size_t smallBatchSize = inputSubBatch.GetNumCols();
    const long inputHeightTimesChannel = (long) (inputHeight * inputChannels);
    Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
    if (zeroPadding)
        SetValue((ElemType) 0);

    const long halfKernelWidth = (long) kernelWidth / 2;
    const long halfKernelHeight = (long) kernelHeight / 2;

#pragma omp parallel for // each input element is copied to many places
    for (long sample = 0; sample < smallBatchSize; sample++)
    {
        for (long id = 0; id < inputDim; id++)
        {
            // IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * inputChannels)
            // IN_ELEM_COLPOS = sample

            const long y = id / inputHeightTimesChannel;   // inputCol
            const long nXC = id % inputHeightTimesChannel; // channel + inputRow*inputChannels
            const long x = nXC / (long) inputChannels;     // inputRow
            const long c = nXC % (long) inputChannels;     // channel

            long x0 = 0, y0 = 0, x1 = 0, y1 = 0;
            if (zeroPadding)
            {
                x0 = (long) max((ElemType)0, ceil((x - (ElemType)kernelHeight + 1.0f + halfKernelHeight) / (ElemType)verticalSubsample)); // row : first wrow in which x is in
                x1 = (long) (x + halfKernelHeight - x0 * verticalSubsample);                                                      // first posxInKernel
                y0 = (long) max((ElemType)0, ceil((y - (ElemType)kernelWidth + 1.0f + halfKernelWidth) / (ElemType)horizontalSubsample)); // col : first wcol in which y is in
                y1 = (long) (y + halfKernelWidth - y0 * horizontalSubsample);                                                     // first posyInKernel
            }
            else
            {
                x0 = (long) max((ElemType)0, ceil((x - (ElemType)kernelHeight + 1) / (ElemType)verticalSubsample));  // row : first wrow in which x is in
                x1 = (long) (x - x0 * verticalSubsample);                                                    // first posxInKernel
                y0 = (long) max((ElemType)0, ceil((y - (ElemType)kernelWidth + 1) / (ElemType)horizontalSubsample)); // col : first wcol in which y is in
                y1 = (long) (y - y0 * horizontalSubsample);                                                  // first posyInKernel
            }

            assert(x1 >= 0 && x1 < kernelHeight && y1 >= 0 && y1 < kernelWidth);

            // PACK_ELEM_ROWPOS(channel, posxInKernel, posyInKernel) = (channel * kernelWidth * kernelHeight + posxInKernel + posyInKernel * kernelHeight)
            // PACK_ELEM_COLPOS(sample, wrow, wcol) = (sample*packedInputColsPerSample + outputHeight*wcol + wrow

            ElemType currentInputValue = inputSubBatch(id, sample);
            long packColBase = (long) (sample * packedInputColsPerSample + y0 * outputHeight);
            for (long wcol = y0, posyInKernel = y1; wcol < (long) outputWidth && posyInKernel >= 0; wcol++, posyInKernel -= (long) horizontalSubsample)
            {
                long packRowBase = (long) (c * kernelWidth * kernelHeight + posyInKernel * kernelHeight);
                for (long wrow = x0, posxInKernel = x1; wrow < (long) outputHeight && posxInKernel >= 0; wrow++, posxInKernel -= (long) verticalSubsample)
                {
                    const long packRow = packRowBase + posxInKernel;
                    const long packCol = packColBase + wrow;
                    (*this)(packRow, packCol) = currentInputValue;
                }
                packColBase += (long) outputHeight;
            }
        }
    }

    return *this;
}
//assume each column is an input sample. Each sample is stored in [channel, row, col]  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::UnpackConvolutionInput(CPUMatrix<ElemType>& inputSubBatch,
                                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                                 const size_t outputWidth, const size_t outputHeight, const size_t /*outputChannels*/,
                                                                 const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                                 const bool zeroPadding) const
{
    assert(verticalSubsample <= kernelHeight && horizontalSubsample <= kernelWidth);

    const size_t packedInputColsPerSample = outputWidth * outputHeight; // output size per channel
    const size_t inputDim = inputWidth * inputHeight * inputChannels;
    const size_t smallBatchSize = inputSubBatch.GetNumCols();
    const long inputHeightTimesChannel = (long) (inputHeight * inputChannels);

    const long halfKernelWidth = (long) kernelWidth / 2;
    const long halfKernelHeight = (long) kernelHeight / 2;

#pragma omp parallel for // each input element is copied to many places
    for (long sample = 0; sample < smallBatchSize; sample++)
    {
        for (long id = 0; id < inputDim; id++)
        {
            // IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * inputChannels)
            // IN_ELEM_COLPOS = sample

            const long y = id / inputHeightTimesChannel;   // inputCol
            const long nXC = id % inputHeightTimesChannel; // channel + inputRow*inputChannels
            const long x = nXC / (long) inputChannels;     // inputRow
            const long c = nXC % (long) inputChannels;     // channel

            long x0 = 0, y0 = 0, x1 = 0, y1 = 0;
            if (zeroPadding)
            {
                x0 = (long) max((ElemType)0, ceil((x - (ElemType) kernelHeight + 1.0f + halfKernelHeight) / (ElemType) verticalSubsample)); // row : first wrow in which x is in
                x1 = (long) (x + halfKernelHeight - x0 * verticalSubsample);                                                      // first posxInKernel
                y0 = (long) max((ElemType)0, ceil((y - (ElemType) kernelWidth + 1.0f + halfKernelWidth) / (ElemType) horizontalSubsample)); // col : first wcol in which y is in
                y1 = (long) (y + halfKernelWidth - y0 * horizontalSubsample);                                                     // first posyInKernel
            }
            else
            {
                x0 = (long) max((ElemType)0, ceil((x - (ElemType) kernelHeight + 1) / (ElemType) verticalSubsample));  // row : first wrow in which x is in
                x1 = (long) (x - x0 * verticalSubsample);                                                    // first posxInKernel
                y0 = (long) max((ElemType)0, ceil((y - (ElemType) kernelWidth + 1) / (ElemType) horizontalSubsample)); // col : first wcol in which y is in
                y1 = (long) (y - y0 * horizontalSubsample);                                                  // first posyInKernel
            }

            assert(x1 >= 0 && x1 < kernelHeight && y1 >= 0 && y1 < kernelWidth);

            // PACK_ELEM_ROWPOS(channel, posxInKernel, posyInKernel) = (channel * kernelWidth * kernelHeight + posxInKernel + posyInKernel * kernelHeight)
            // PACK_ELEM_COLPOS(sample, wrow, wcol) = (sample*packedInputColsPerSample + outputHeight*wcol + wrow

            ElemType currentInputValue = inputSubBatch(id, sample);
            long packColBase = (long) (sample * packedInputColsPerSample + y0 * outputHeight);
            for (long wcol = y0, posyInKernel = y1; wcol < (long) outputWidth && posyInKernel >= 0; wcol++, posyInKernel -= (long) horizontalSubsample)
            {
                long packRowBase = (long) (c * kernelWidth * kernelHeight + posyInKernel * kernelHeight);
                for (long wrow = x0, posxInKernel = x1; wrow < (long) outputHeight && posxInKernel >= 0; wrow++, posxInKernel -= (long) verticalSubsample)
                {
                    const long packRow = packRowBase + posxInKernel;
                    const long packCol = packColBase + wrow;
                    currentInputValue += (*this)(packRow, packCol);
                }
                packColBase += (long) outputHeight;
            }
            inputSubBatch(id, sample) = currentInputValue;
        }
    }

    return inputSubBatch;
}

//assume each column is an input sample. Each sample is stored in  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignMaxPoolingResult(const CPUMatrix<ElemType>& inputBatch, const size_t channels,
                                                                 const size_t /*inputWidth*/, const size_t inputHeight, const size_t /*inputSizePerSample*/,
                                                                 const size_t /*outputWidth*/, const size_t outputHeight, const size_t outputSizePerSample,
                                                                 const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    const long inputHeightTimesChannel = (long) (inputHeight * channels);
    const long outputHeightTimesChannel = (long) (outputHeight * channels);
    const size_t batchSize = inputBatch.GetNumCols();
    Resize(outputSizePerSample, batchSize);

// IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
// IN_ELEM_COLPOS = sample

// OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
// OUT_ELEM_COLPOS = sample

#pragma omp parallel for
    for (long sample = 0; sample < (long) batchSize; sample++)
    {
        for (long outputIndexWithinSample = 0; outputIndexWithinSample < outputSizePerSample; outputIndexWithinSample++)
        {
            const long y = outputIndexWithinSample / outputHeightTimesChannel;   // wcol
            const long nXC = outputIndexWithinSample % outputHeightTimesChannel; // channel + wrow*channels
            const long x = (long) (nXC / channels);                              // wrow
            const long c = (long) (nXC % channels);                              // channel

            ElemType maxVal = -FLT_MAX;
            ElemType minVal = FLT_MAX;
            const long rowInWindowBase = (long) ((x * verticalSubsample + y * horizontalSubsample * inputHeight) * channels + c);
            for (long colInWindow = 0; colInWindow < windowWidth; colInWindow++)
            {
                long rowInInput = rowInWindowBase + colInWindow * inputHeightTimesChannel;
                for (long rowInWindow = 0; rowInWindow < windowHeight; rowInWindow++)
                {
                    const ElemType val = inputBatch(rowInInput, sample); // pf[rowInWindow*channels];
                    maxVal = std::max(maxVal, val);
                    minVal = std::min(minVal, val);
                    rowInInput += (long) channels;
                }
            }

            (*this)(outputIndexWithinSample, sample) = maxVal;
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddMaxPoolingGradient(const CPUMatrix<ElemType>& outputGradientBatch, const CPUMatrix<ElemType>& inputBatch, const CPUMatrix<ElemType>& outputBatch,
                                                                const size_t channels,
                                                                const size_t /*inputWidth*/, const size_t inputHeight, const size_t inputSizePerSample,
                                                                const size_t outputWidth, const size_t outputHeight, const size_t /*outputSizePerSample*/,
                                                                const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    size_t batchSize = inputBatch.GetNumCols();
    const long inputHeightTimesChannel = (long) (inputHeight * channels);
    const long outputHeightTimesChannel = (long) (outputHeight * channels);

// IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
// IN_ELEM_COLPOS = sample

// OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
// OUT_ELEM_COLPOS = sample

#pragma omp parallel for
    for (long sample = 0; sample < batchSize; sample++)
    {
        for (long inputIndexWithinSample = 0; inputIndexWithinSample < inputSizePerSample; inputIndexWithinSample++)
        {
            const long y = inputIndexWithinSample / inputHeightTimesChannel;   // col in input
            const long nXC = inputIndexWithinSample % inputHeightTimesChannel; // channel + row*chanels
            const long x = (long) (nXC / channels);                            // row in input
            const long c = (long) (nXC % channels);                            // channel

            long startOutX = (long) max((ElemType)0, ceil((x - (ElemType) windowHeight + 1) / (ElemType) verticalSubsample));          // inclusive start
            long endOutX = (long) ((x / verticalSubsample < outputHeight - 1) ? x / verticalSubsample : outputHeight - 1);   // inclusive end
            long startOutY = (long) max((ElemType)0, ceil((y - (ElemType) windowWidth + 1) / (ElemType) horizontalSubsample));         // inclusive start
            long endOutY = (long) ((y / horizontalSubsample < outputWidth - 1) ? y / horizontalSubsample : outputWidth - 1); // inclusive end

            ElemType inputValue = inputBatch(inputIndexWithinSample, sample);
            for (long outY = startOutY; outY <= endOutY; outY++)
            {
                for (long outX = startOutX; outX <= endOutX; outX++)
                {
                    long outputIndex = (long) (outY * outputHeightTimesChannel + outX * channels + c);
                    if (inputValue == outputBatch(outputIndex, sample))
                        (*this)(inputIndexWithinSample, sample) += outputGradientBatch(outputIndex, sample);
                }
            }
        }
    }

    return *this;
}
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignAveragePoolingResult(const CPUMatrix<ElemType>& inputBatch, const size_t channels,
                                                                     const size_t /*inputWidth*/, const size_t inputHeight, const size_t /*inputSizePerSample*/,
                                                                     const size_t /*outputWidth*/, const size_t outputHeight, const size_t outputSizePerSample,
                                                                     const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    const long inputHeightTimesChannel = (long) (inputHeight * channels);
    const long outputHeightTimesChannel = (long) (outputHeight * channels);
    const size_t batchSize = inputBatch.GetNumCols();
    const size_t windowSize = windowWidth * windowHeight;
    Resize(outputSizePerSample, batchSize);

// IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
// IN_ELEM_COLPOS = sample

// OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
// OUT_ELEM_COLPOS = sample

#pragma omp parallel for
    for (long sample = 0; sample < batchSize; sample++)
    {
        for (long outputIndexWithinSample = 0; outputIndexWithinSample < outputSizePerSample; outputIndexWithinSample++)
        {
            const long y = outputIndexWithinSample / outputHeightTimesChannel;   // wcol
            const long nXC = outputIndexWithinSample % outputHeightTimesChannel; // channel + wrow*channels
            const long x = (long) (nXC / channels);                              // wrow
            const long c = (long) (nXC % channels);                              // channel

            ElemType sum = 0;
            const long rowInWindowBase = (long) ((x * verticalSubsample + y * horizontalSubsample * inputHeight) * channels + c);
            for (long colInWindow = 0; colInWindow < windowWidth; colInWindow++)
            {
                long rowInInput = rowInWindowBase + colInWindow * inputHeightTimesChannel;
                for (long rowInWindow = 0; rowInWindow < windowHeight; rowInWindow++)
                {
                    sum += inputBatch(rowInInput, sample);
                    rowInInput += (long) channels;
                }
            }

            (*this)(outputIndexWithinSample, sample) = sum / windowSize;
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AddAveragePoolingGradient(const CPUMatrix<ElemType>& outputGradientBatch,
                                                                    const size_t channels,
                                                                    const size_t /*inputWidth*/, const size_t inputHeight, const size_t inputSizePerSample,
                                                                    const size_t outputWidth, const size_t outputHeight, const size_t /*outputSizePerSample*/,
                                                                    const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    size_t batchSize = outputGradientBatch.GetNumCols();
    const long inputHeightTimesChannel = (long) (inputHeight * channels);
    const long outputHeightTimesChannel = (long) (outputHeight * channels);
    const long windowSize = (long) (windowWidth * windowHeight);

// IN_ELEM_ROWPOS(channel, row, col) = (channel + (row + col * inputHeight) * channels)
// IN_ELEM_COLPOS = sample

// OUT_ELEM_ROWPOS(channel, wrow, wcol) = (channel + (wrow + wcol * outputHeight) * channels)
// OUT_ELEM_COLPOS = sample

#pragma omp parallel for
    for (long sample = 0; sample < batchSize; sample++)
    {
        for (long inputIndexWithinSample = 0; inputIndexWithinSample < inputSizePerSample; inputIndexWithinSample++)
        {
            const long y = inputIndexWithinSample / inputHeightTimesChannel;   // col in input
            const long nXC = inputIndexWithinSample % inputHeightTimesChannel; // channel + row*chanels
            const long x = nXC / (long) channels;                              // row in input
            const long c = nXC % (long) channels;                              // channel

            long startOutX = (long) max((ElemType)0, ceil((x - (ElemType) windowHeight + 1) / (ElemType) verticalSubsample));               // inclusive start
            long endOutX = (long) ((x / verticalSubsample < outputHeight - 1) ? x / (long) verticalSubsample : outputHeight - 1); // inclusive end
            long startOutY = (long) max((ElemType)0, ceil((y - (ElemType) windowWidth + 1) / (ElemType) horizontalSubsample));              // inclusive start
            long endOutY = (long) ((y / horizontalSubsample < outputWidth - 1) ? y / horizontalSubsample : outputWidth - 1);      // inclusive end

            for (long outY = startOutY; outY <= endOutY; outY++)
            {
                for (long outX = startOutX; outX <= endOutX; outX++)
                {
                    long outputIndex = outY * outputHeightTimesChannel + outX * (long) channels + c;
                    (*this)(inputIndexWithinSample, sample) += outputGradientBatch(outputIndex, sample) / windowSize;
                }
            }
        }
    }

    return *this;
}
#pragma endregion Other Helper Functions

template <class ElemType>
void CPUMatrix<ElemType>::NDConvolutionForward(const CPUMatrix<ElemType>& filter, const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIwht,
                                               const CPUMatrix<int>& mpRowRun, const CPUMatrix<int>& runs, CPUMatrix<ElemType>& output) const
{
    for (size_t sample = 0; sample < output.GetNumCols(); sample++)
    {
        for (size_t row = 0; row < output.GetNumRows(); row++)
        {
            int colBase = mpRowCol(row, 0);
            int ivBase = mpRowIwht(row, 0);
            assert(0 <= colBase && colBase < GetNumRows());

            ElemType sum = 0;
            int i0 = mpRowRun(row, 0);
            int skip = runs(i0++, 0);
            int size = runs(i0++, 0);
            int imask = i0 + size;
            for (int i = 0; i < size; i++)
            {
                if (runs(imask + i, 0) == 0)
                    continue;
                int dcol = runs(i0 + i, 0);
                assert(0 <= colBase + dcol && colBase + dcol < GetNumRows());
                sum += filter.BufferPointer()[ivBase + skip + i] * (*this)(colBase + dcol, sample);
            }
            output(row, sample) = sum;
        }
    }
}

#pragma region Static BLAS Functions

/// <summary>Matrix-matrix multiply with col-major matrices (a and b may be transposed): c = alpha * op(a) * op(b) + beta*c</summary>
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="transposeA">Whether matrix a is transposed</param>
/// <param name="b">Input matrix</param>
/// <param name="transposeB">Whether matrix b is transposed</param>
/// <param name="beta">Scalar</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void CPUMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, const bool transposeA, const CPUMatrix<ElemType>& b, const bool transposeB,
                                                 ElemType beta, CPUMatrix<ElemType>& c)
{
    if (a.IsEmpty() || b.IsEmpty())
        return;

    int m, n, k, l;
    int lda, ldb, ldc;
#ifdef USE_ACML
    char transA, transB;
#else
    CBLAS_TRANSPOSE mklTransA;
    CBLAS_TRANSPOSE mklTransB;
#endif

    if (transposeA)
    {
        m = (int) a.GetNumCols();
        k = (int) a.GetNumRows();
        lda = k;
#ifdef USE_ACML
        transA = (char) MatrixTranspose::Trans;
#else
        mklTransA = CBLAS_TRANSPOSE::CblasTrans;
#endif
    }
    else
    {
        m = (int) a.GetNumRows();
        k = (int) a.GetNumCols();
        lda = m;
#ifdef USE_ACML
        transA = (char) MatrixTranspose::NoTrans;
#else
        mklTransA = CBLAS_TRANSPOSE::CblasNoTrans;
#endif
    }

    if (transposeB)
    {
        l = (int) b.GetNumCols();
        n = (int) b.GetNumRows();
        ldb = n;
#ifdef USE_ACML
        transB = (char) MatrixTranspose::Trans;
#else
        mklTransB = CBLAS_TRANSPOSE::CblasTrans;
#endif
    }
    else
    {
        l = (int) b.GetNumRows();
        n = (int) b.GetNumCols();
        ldb = l;
#ifdef USE_ACML
        transB = (char) MatrixTranspose::NoTrans;
#else
        mklTransB = CBLAS_TRANSPOSE::CblasNoTrans;
#endif
    }

    assert(m > 0 && k > 0 && l > 0 && n > 0); // converting from size_t to int may cause overflow
    assert(k == l);
    if (k != l)
        InvalidArgument("CPUMatrix<ElemType>::MultiplyAndWeightedAdd : The inner dimensions of a and b must match.");

    if (beta == 0)
        c.Resize(m, n);
    else
        c.VerifySize(m, n); // Can't resize if beta != 0

    ldc = (int) c.GetNumRows();

    if (sizeof(ElemType) == sizeof(double))
    {
#ifdef USE_ACML
        dgemm(transA, transB, m, n, k, alpha, reinterpret_cast<double*>(a.m_pArray), lda, reinterpret_cast<double*>(b.m_pArray), ldb, beta, reinterpret_cast<double*>(c.m_pArray), ldc);
#else
        cblas_dgemm((CBLAS_ORDER) BLAS_COLMAJOR mklTransA, mklTransB, m, n, k, alpha, reinterpret_cast<double*>(a.m_pArray), lda, reinterpret_cast<double*>(b.m_pArray), ldb, beta, reinterpret_cast<double*>(c.m_pArray), ldc);
#endif
    }
    else
    {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
        sgemm(BLAS_COLMAJOR transA, transB, m, n, k, alpha, reinterpret_cast<float*>(a.m_pArray), lda, reinterpret_cast<float*>(b.m_pArray), ldb, beta, reinterpret_cast<float*>(c.m_pArray), ldc);
#else
        cblas_sgemm((CBLAS_ORDER) BLAS_COLMAJOR mklTransA, mklTransB, m, n, k, alpha, reinterpret_cast<float*>(a.m_pArray), lda, reinterpret_cast<float*>(b.m_pArray), ldb, beta, reinterpret_cast<float*>(c.m_pArray), ldc);
#endif
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::Multiply1x1AndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b,
                                                    ElemType beta, CPUMatrix<ElemType>& c)
{
    assert(a.GetNumElements() == 1); // a is a scalar

    ElemType f = alpha * a.Get00Element();
    if (beta == 0) // don't even read the memory if beta is 0
#pragma omp parallel for
        foreach_coord (i, j, c)
            c(i, j) = b(i, j) * f;
    else
#pragma omp parallel for
        foreach_coord (i, j, c)
            c(i, j) = b(i, j) * f + c(i, j) * beta;
}

/* compute singular value decomposition as
    A = U*SIGMA*VT
    W is used as temp working memory
    */
template <class ElemType>
void CPUMatrix<ElemType>::SVD(const CPUMatrix<ElemType>& A, CPUMatrix<ElemType>& SIGMA, CPUMatrix<ElemType>& U, CPUMatrix<ElemType>& VT, CPUMatrix<ElemType>& W)
{
    if (A.IsEmpty())
        LogicError("SVD:  input matrix is empty.");

    int info;
    int m, n, lda, ldu, ldvt;
    m = (int) A.GetNumRows();
    n = (int) A.GetNumCols();
    W.GetNumRows(); // W is used as temp working memory
    lda = m;
    ldu = m;
    ldvt = n;
    U.Resize(m, m);
    SIGMA.Resize(std::min(m, n), 1);
    VT.Resize(n, n);

    if (sizeof(ElemType) == sizeof(double))
    {
#ifdef USE_ACML
        dgesvd('A', 'A', (int) m, (int) n, reinterpret_cast<double*>(A.m_pArray), (int) lda, reinterpret_cast<double*>(SIGMA.m_pArray), reinterpret_cast<double*>(U.m_pArray), (int) ldu, reinterpret_cast<double*>(VT.m_pArray), (int) ldvt, &info);
#elif defined(USE_MKL)
        double wkopt;
        int lwork = -1;
        dgesvd("All", "All", &m, &n, reinterpret_cast<double*>(A.m_pArray), &lda, reinterpret_cast<double*>(SIGMA.m_pArray), reinterpret_cast<double*>(U.m_pArray), &ldu, reinterpret_cast<double*>(VT.m_pArray), &ldvt, &wkopt, &lwork, &info);
        lwork = (int) wkopt;
        W.Resize(lwork, 1);
        dgesvd("All", "All", &m, &n, reinterpret_cast<double*>(A.m_pArray), &lda, reinterpret_cast<double*>(SIGMA.m_pArray), reinterpret_cast<double*>(U.m_pArray), &ldu, reinterpret_cast<double*>(VT.m_pArray), &ldvt, reinterpret_cast<double*>(W.m_pArray), &lwork, &info);
#else
        std::vector<double> superb(std::max(std::min(m, n) - 1, 1));
        info = LAPACKE_dgesvd(BLAS_COLMAJOR 'A', 'A', (int) m, (int) n, reinterpret_cast<double*>(A.m_pArray), (int) lda, reinterpret_cast<double*>(SIGMA.m_pArray),
            reinterpret_cast<double*>(U.m_pArray), (int) ldu, reinterpret_cast<double*>(VT.m_pArray), (int) ldvt, &superb[0]);
#endif
    }
    else
    {
#ifdef USE_ACML
#pragma warning(suppress : 4244)
        sgesvd('A', 'A', (int) m, (int) n, reinterpret_cast<float*>(A.m_pArray), (int) lda, reinterpret_cast<float*>(SIGMA.m_pArray), reinterpret_cast<float*>(U.m_pArray), (int) ldu, reinterpret_cast<float*>(VT.m_pArray), (int) ldvt, &info);
#elif defined(USE_MKL)
        float wkopt;
        int lwork = -1;
        sgesvd("All", "All", &m, &n, reinterpret_cast<float*>(A.m_pArray), &lda, reinterpret_cast<float*>(SIGMA.m_pArray), reinterpret_cast<float*>(U.m_pArray), &ldu, reinterpret_cast<float*>(VT.m_pArray), &ldvt, &wkopt, &lwork, &info);
        lwork = (int) wkopt;
        W.Resize(lwork, 1);
        sgesvd("All", "All", &m, &n, reinterpret_cast<float*>(A.m_pArray), &lda, reinterpret_cast<float*>(SIGMA.m_pArray), reinterpret_cast<float*>(U.m_pArray), &ldu, reinterpret_cast<float*>(VT.m_pArray), &ldvt, reinterpret_cast<float*>(W.m_pArray), &lwork, &info);
#else
        std::vector<float> superb(std::max(std::min(m, n) - 1, 1));
        info = LAPACKE_sgesvd(BLAS_COLMAJOR 'A', 'A', (int) m, (int) n, reinterpret_cast<float*>(A.m_pArray), (int) lda, reinterpret_cast<float*>(SIGMA.m_pArray),
            reinterpret_cast<float*>(U.m_pArray), (int) ldu, reinterpret_cast<float*>(VT.m_pArray), (int) ldvt, &superb[0]);
#endif
    }

    if (info > 0)
    {
        RuntimeError("The algorithm computing SVD failed to converge.\n");
    }
}

/// <summary>Matrix-matrix multiply with col-major matrices (a and b may be transposed): c =  op(a) * op(b) + c</summary>
/// <param name="a">Input matrix</param>
/// <param name="transposeA">Whether matrix a is transposed</param>
/// <param name="b">Input matrix</param>
/// <param name="transposeB">Whether matrix b is transposed</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void CPUMatrix<ElemType>::MultiplyAndAdd(const CPUMatrix<ElemType>& a, const bool transposeA, const CPUMatrix<ElemType>& b, const bool transposeB,
                                         CPUMatrix<ElemType>& c)
{
    return CPUMatrix<ElemType>::MultiplyAndWeightedAdd(1.0, a, transposeA, b, transposeB, 1.0, c);
}
template <class ElemType>
void CPUMatrix<ElemType>::AssignSoftmaxSum(const CPUMatrix<ElemType>& softmax, CPUMatrix<ElemType>& c)
{
    ElemType log_likelihood = 0.0;
    size_t batch_size = this->GetNumCols();
#pragma omp parallel for reduction(+ : log_likelihood)
    for (int instance_id = 0; instance_id < batch_size; instance_id++)
    {
        int sample = (int) (*this)(0, instance_id);
        log_likelihood += softmax(instance_id, sample);
    }
    c(0, 0) = -log_likelihood;
}

template <class ElemType>
void CPUMatrix<ElemType>::AssignNCEUnnormalizedEval(const CPUMatrix<ElemType>& a,
                                                    const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& bias, CPUMatrix<ElemType>& c)
//this: samples+probs
// a:   hidden
// b:   embedding
// tmp:  softmax
//  c: loglikelihood
{
    ElemType log_likelihood = 0.0;
    size_t batch_size = this->GetNumCols();
#pragma omp parallel for reduction(+ : log_likelihood)
    for (int instance_id = 0; instance_id < batch_size; instance_id++)
    {
        int sample = -(int) (*this)(0, instance_id);
        ElemType score = bias(sample, 0);
        for (int dim = 0; dim < b.GetNumRows(); dim++)
            score += b(dim, sample) * a(dim, instance_id);
        log_likelihood += score;
    }
    c(0, 0) = -log_likelihood;
}

//samples+prob                         gradient           hidden               embedding          embedding/hidden
//a.m_CPUMatrix->AssignNCEDerivative(*tmp.m_CPUMatrix, *a.m_CPUMatrix, *b.m_CPUMatrix, inputIndex, *c.m_CPUMatrix);
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignNCEDerivative(const CPUMatrix<ElemType>& tmp, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t inputIndex, CPUMatrix<ElemType>& c)
{
    size_t sample_size = this->GetNumRows() / 2;
    size_t batch_size = this->GetNumCols();
    if (inputIndex == 1)
    {
#pragma omp parallel for
        for (int instance_id = 0; instance_id < batch_size; instance_id++)
            for (int sample_id = 0; sample_id < sample_size; sample_id++)
            {
                int sample = (int) (*this)(2 * sample_id, instance_id);
                for (int dim = 0; dim < b.GetNumRows(); dim++)
                    c(dim, instance_id) -= b(dim, sample) * tmp(sample_id, instance_id);
            }
    }
    else if (inputIndex == 2)
    {
        int i_blocks = omp_get_num_threads() * 16;
// Assume only one block in k direction.
// We don't need to explicitly block in the j direction.
#pragma omp parallel for
        for (int ib = 0; ib < i_blocks; ib++)
            for (int instance_id = 0; instance_id < batch_size; instance_id++)
                for (int sample_id = 0; sample_id < sample_size; sample_id++)
                {
                    int sample = (int) (*this)(2 * sample_id, instance_id);
                    if (sample % i_blocks == ib)
                        for (int dim = 0; dim < b.GetNumRows(); dim++)
                            c(dim, sample) -= a(dim, instance_id) * tmp(sample_id, instance_id);
                }
    }
    else
    {
        assert(inputIndex == 3);
        // Assume only one block in k direction.
        // We don't need to explicitly block in the j direction.
        for (int instance_id = 0; instance_id < batch_size; instance_id++)
            for (int sample_id = 0; sample_id < sample_size; sample_id++)
            {
                int sample = (int) (*this)(2 * sample_id, instance_id);
                c(0, sample) -= tmp(sample_id, instance_id);
            }
    }
    return *this;
}

template <class ElemType>
void CPUMatrix<ElemType>::AssignNoiseContrastiveEstimation(const CPUMatrix<ElemType>& a,
                                                           const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& bias, CPUMatrix<ElemType>& tmp, CPUMatrix<ElemType>& c)
//this: samples+probs
// a:   hidden
// b:   embedding
// tmp:  softmax
// c: loglikelihood
{
    double log_likelihood = 0.0;
    size_t sample_size = this->GetNumRows() / 2;
    size_t batch_size = this->GetNumCols();
    size_t num_noise_samples = sample_size - 1;
    double log_num_noise_samples = std::log(num_noise_samples);
#pragma omp parallel for reduction(+ : log_likelihood)
    for (int instance_id = 0; instance_id < batch_size; instance_id++)
        for (int sample_id = 0; sample_id < sample_size; sample_id++)
        {
            int sample = (int) (*this)(2 * sample_id, instance_id);
            double score = bias(0, sample);
            for (int dim = 0; dim < b.GetNumRows(); dim++)
                score += a(dim, instance_id) * b(dim, sample);
            double sample_prob = -(*this)(2 * sample_id + 1, instance_id);
            if (sample_id == 0)
                sample_prob = -sample_prob;
            double score_noise = log_num_noise_samples + sample_prob;
            double z = LogAdd(score, score_noise);
            double logprob = score - z;
            double logprob_noise = score_noise - z;
            tmp(sample_id, instance_id) = (ElemType) -std::exp(logprob);
            if (sample_id == 0)
                tmp(sample_id, instance_id) += 1;
            log_likelihood += sample_id == 0 ? logprob : logprob_noise;
        }
    c(0, 0) = (ElemType) -log_likelihood;
}

/// <summary>Matrix-matrix multiply with col-major matrices (a and b may be transposed): c =  op(a) * op(b)</summary>
/// <param name="a">Input matrix</param>
/// <param name="transposeA">Whether matrix a is transposed</param>
/// <param name="b">Input matrix</param>
/// <param name="transposeB">Whether matrix b is transposed</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void CPUMatrix<ElemType>::Multiply(const CPUMatrix<ElemType>& a, const bool transposeA, const CPUMatrix<ElemType>& b, const bool transposeB,
                                   CPUMatrix<ElemType>& c)
{
    return CPUMatrix<ElemType>::MultiplyAndWeightedAdd(1.0, a, transposeA, b, transposeB, 0.0, c);
}

/// <summary>Matrix-matrix multiply with col-major matrices (a and b are not transposed): c =  a * b</summary>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void CPUMatrix<ElemType>::Multiply(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
{
    return CPUMatrix<ElemType>::MultiplyAndWeightedAdd(1.0, a, false, b, false, 0.0, c);
}

/// <summary>Matrix-scalar multiply with col-major matrices: c = alpha * a + c</summary>
/// if a is a column vector, add to all columns of c
/// if a is a row vector, add to all rows of c
/// if a is a scalar, add to all rows of c
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void CPUMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c)
{
    if (a.IsEmpty() || c.IsEmpty())
        LogicError("ScaleAndAdd:  one of the input matrices is empty.");

    if (a.GetNumRows() != 1 && a.GetNumCols() != 1) // a is not a col or row vector
    {
        const int m = (int) a.GetNumRows();
        const int n = (int) a.GetNumCols();
        const int len = m * n;
        const int incx = 1;
        const int incy = 1;

        assert(m > 0 && n > 0 && len > 0); // converting from size_t to int may cause overflow
        assert((int) c.GetNumRows() == m && (int) c.GetNumCols() == n);
        if ((int) c.GetNumRows() != m || (int) c.GetNumCols() != n)
            InvalidArgument("Dimension of matrix c does not match dimension of matrix a.");

        if (sizeof(ElemType) == sizeof(double))
        {
#ifdef USE_ACML
            daxpy(len, alpha, reinterpret_cast<double*>(a.m_pArray), incx, reinterpret_cast<double*>(c.m_pArray), incy);
#else
            cblas_daxpy(len, alpha, reinterpret_cast<double*>(a.m_pArray), incx, reinterpret_cast<double*>(c.m_pArray), incy);
#endif
        }
        else
        {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
            saxpy(len, alpha, reinterpret_cast<float*>(a.m_pArray), incx, reinterpret_cast<float*>(c.m_pArray), incy);
#else
            cblas_saxpy(len, alpha, reinterpret_cast<float*>(a.m_pArray), incx, reinterpret_cast<float*>(c.m_pArray), incy);
#endif
        }
    }
    else if (a.GetNumElements() == 1) // scalar, add to all elements
    {
        ElemType v = alpha * a(0, 0);
        long m = (long) c.GetNumRows(), n = (long) c.GetNumCols();
#pragma omp parallel for
        for (long j = 0; j < n; j++)
        {
            // four-way unrolling
            for (long i = 0; i < (m & ~3); i += 4)
            {
                c(i, j) += v;
                c(i + 1, j) += v;
                c(i + 2, j) += v;
                c(i + 3, j) += v;
            }
            // handle remaining stuffs
            for (long i = m & ~3; i < m; i++)
            {
                c(i, j) += v;
            }
        }
    }
    else if (a.GetNumCols() == 1) // col vector, add it to all columns
    {
        int m = (int) c.GetNumRows();
        assert(m == (int) a.GetNumRows());
        if (m != (int) a.GetNumRows())
            InvalidArgument("To add column vector, rows should match.");

        if (sizeof(ElemType) == sizeof(double))
        {
#pragma omp parallel for
            foreach_column (j, c)
            {
#ifdef USE_ACML
                daxpy(m, alpha, reinterpret_cast<double*>(a.m_pArray), 1, reinterpret_cast<double*>(c.m_pArray + c.LocateColumn(j)), 1);
#else
                cblas_daxpy(m, alpha, reinterpret_cast<double*>(a.m_pArray), 1, reinterpret_cast<double*>(c.m_pArray + c.LocateColumn(j)), 1);
#endif
            }
        }
        else
        {
#pragma omp parallel for
            foreach_column (j, c)
            {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
                saxpy(m, alpha, reinterpret_cast<float*>(a.m_pArray), 1, reinterpret_cast<float*>(c.m_pArray + c.LocateColumn(j)), 1);
#else
                cblas_saxpy(m, alpha, reinterpret_cast<float*>(a.m_pArray), 1, reinterpret_cast<float*>(c.m_pArray + c.LocateColumn(j)), 1);
#endif
            }
        }
    }
    else // row vector, add it to all rows
    {
        int m = (int) c.GetNumRows();
        int n = (int) c.GetNumCols();
        assert(n == (int) a.GetNumCols());
        if (n != (int) a.GetNumCols())
            InvalidArgument("To add row vector, cols should match.");

        if (sizeof(ElemType) == sizeof(double))
        {
#pragma omp parallel for
            foreach_row (i, c)
            {
#ifdef USE_ACML
                daxpy(n, alpha, reinterpret_cast<double*>(a.m_pArray), 1, reinterpret_cast<double*>(c.m_pArray + i), m);
#else
                cblas_daxpy(n, alpha, reinterpret_cast<double*>(a.m_pArray), 1, reinterpret_cast<double*>(c.m_pArray + i), m);
#endif
            }
        }
        else
        {
#pragma omp parallel for
            foreach_row (i, c)
            {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
                saxpy(n, alpha, reinterpret_cast<float*>(a.m_pArray), 1, reinterpret_cast<float*>(c.m_pArray + i), m);
#else
                cblas_saxpy(n, alpha, reinterpret_cast<float*>(a.m_pArray), 1, reinterpret_cast<float*>(c.m_pArray + i), m);
#endif
            }
        }
    }
}
/// <summary>c += alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void CPUMatrix<ElemType>::AddScaledDifference(const ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
{
    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
           a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols());

    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
          a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols()))
    {
        InvalidArgument("AddScaledDifference:  a, b, and c must have same dimension.");
    }

    if (a.IsEmpty())
        LogicError("AddScaledDifference:  Input matrix a is empty.");

    long m = (long) c.GetNumElements();
#pragma omp parallel for
    // four-way unrolling
    for (long i = 0; i < (m & ~3); i += 4)
    {
        c.m_pArray[i] += alpha * (a.m_pArray[i] - b.m_pArray[i]);
        c.m_pArray[i + 1] += alpha * (a.m_pArray[i + 1] - b.m_pArray[i + 1]);
        c.m_pArray[i + 2] += alpha * (a.m_pArray[i + 2] - b.m_pArray[i + 2]);
        c.m_pArray[i + 3] += alpha * (a.m_pArray[i + 3] - b.m_pArray[i + 3]);
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        c.m_pArray[i] += alpha * (a.m_pArray[i] - b.m_pArray[i]);
    }
}

/// <summary> c = alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void CPUMatrix<ElemType>::AssignScaledDifference(const ElemType alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
{
    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());

    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
    {
        InvalidArgument("AssignScaledDifference:  a, b must have same dimension.");
    }

    if (a.IsEmpty())
        LogicError("AssignScaledDifference:  Input matrix a is empty.");

    if (&c != &a && &c != &b)
        c.Resize(a.GetNumRows(), a.GetNumCols());

    long m = (long) c.GetNumElements();
#pragma omp parallel for
    // four-way unrolling
    for (long i = 0; i < (m & ~3); i += 4)
    {
        c.m_pArray[i] = alpha * (a.m_pArray[i] - b.m_pArray[i]);
        c.m_pArray[i + 1] = alpha * (a.m_pArray[i + 1] - b.m_pArray[i + 1]);
        c.m_pArray[i + 2] = alpha * (a.m_pArray[i + 2] - b.m_pArray[i + 2]);
        c.m_pArray[i + 3] = alpha * (a.m_pArray[i + 3] - b.m_pArray[i + 3]);
    }
    // handle remaining stuffs
    for (long i = m & ~3; i < m; i++)
    {
        c.m_pArray[i] = alpha * (a.m_pArray[i] - b.m_pArray[i]);
    }
}

//c[ci,cj] += a[ai,aj]
template <class ElemType>
void CPUMatrix<ElemType>::AddElementToElement(const CPUMatrix<ElemType>& a, const size_t ai, const size_t aj, CPUMatrix<ElemType>& c, const size_t ci, const size_t cj)
{
    if (ai >= a.GetNumRows() || aj >= a.GetNumCols() ||
        ci >= c.GetNumRows() || cj >= c.GetNumCols())
        InvalidArgument("AddElementToElement:  index out of range.");

    c(ci, cj) += a(ai, aj);
}

////c[ci,cj] += a[ai,aj]
//template<class ElemType>
//void CPUMatrix<ElemType>::AddLogElementToElement(const CPUMatrix<ElemType>& a, const size_t ai, const size_t aj, CPUMatrix<ElemType>& c, const size_t ci, const size_t cj)
//{
//    if (ai >= a.GetNumRows() || aj >=a.GetNumCols() ||
//        ci >= c.GetNumRows() || cj >=c.GetNumCols())
//        InvalidArgument("AddElementToElement:  index out of range.");
//
//    ElemType v = a(ai,aj);
//    c(ci, cj) += ((v < EPS_IN_LOG) ? LOG_OF_EPS_IN_LOG : log(v));
//}

//c[ci,cj] = a[ai,aj]
template <class ElemType>
void CPUMatrix<ElemType>::AssignElementToElement(const CPUMatrix<ElemType>& a, const size_t ai, const size_t aj, CPUMatrix<ElemType>& c, const size_t ci, const size_t cj)
{
    if (ai >= a.GetNumRows() || aj >= a.GetNumCols() ||
        ci >= c.GetNumRows() || cj >= c.GetNumCols())
        InvalidArgument("AssignElementToElement:  index out of range.");

    c(ci, cj) = a(ai, aj);
}

/// <summary>c += alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">1X1 matrix</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void CPUMatrix<ElemType>::AddScaledDifference(const CPUMatrix<ElemType>& alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
{
    assert(alpha.GetNumElements() == 1);
    if (!(alpha.GetNumElements() == 1))
        InvalidArgument("AddScaledDifference:  alpha must be a 1X1 matrix.");

    AddScaledDifference(alpha(0, 0), a, b, c);
}

/// <summary> c = alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">1X1 matrix</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void CPUMatrix<ElemType>::AssignScaledDifference(const CPUMatrix<ElemType>& alpha, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
{
    assert(alpha.GetNumElements() == 1);
    if (!(alpha.GetNumElements() == 1))
        InvalidArgument("AddScaledDifference:  alpha must be a 1X1 matrix.");

    AssignScaledDifference(alpha(0, 0), a, b, c);
}
/// <summary>Matrix-scalar multiply with col-major matrices: c = alpha * a</summary>
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
/*static*/ void CPUMatrix<ElemType>::Scale(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c)
{
    if (a.IsEmpty())
        LogicError("Scale:  Input matrix a is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();

    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow
    c.Resize(m, n);

    if (alpha == 0)
    {
        memset(c.m_pArray, 0, sizeof(ElemType) * c.GetNumElements());
        return;
    }

    long size = (long) c.GetNumElements();
#pragma omp parallel for
    // four-way unrolling
    for (long i = 0; i < (size & ~3); i += 4)
    {
        c.m_pArray[i] = alpha * a.m_pArray[i];
        c.m_pArray[i + 1] = alpha * a.m_pArray[i + 1];
        c.m_pArray[i + 2] = alpha * a.m_pArray[i + 2];
        c.m_pArray[i + 3] = alpha * a.m_pArray[i + 3];
    }
    // remaining elements
    for (long i = size & ~3; i < size; i++)
    {
        c.m_pArray[i] = alpha * a.m_pArray[i];
    }
}

/// <summary>Matrix-scalar multiply with col-major matrices: a = alpha * a</summary>
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
template <class ElemType>
/*static*/ void CPUMatrix<ElemType>::Scale(ElemType alpha, CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("Scale:  Input matrix a is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int len = m * n;
    const int incx = 1;

    assert(m > 0 && n > 0 && len > 0); // converting from size_t to int may cause overflow

    if (alpha == 0 && incx == 1)
    {
        memset(a.m_pArray, 0, sizeof(ElemType) * len);
    }
    else if (sizeof(ElemType) == sizeof(double))
    {
#ifdef USE_ACML
        dscal(len, alpha, reinterpret_cast<double*>(a.m_pArray), incx); // TODO: Use overloads.
#else
        cblas_dscal(len, alpha, reinterpret_cast<double*>(a.m_pArray), incx);
#endif
    }
    else
    {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
        sscal(len, alpha, reinterpret_cast<float*>(a.m_pArray), incx);
#else
        cblas_sscal(len, alpha, reinterpret_cast<float*>(a.m_pArray), incx);
#endif
    }
}

/// <summary>Matrix multiply with col-major matrices: a = alpha[1,1] * a</summary>
/// <param name="alpha">1x1 matrix</param>
/// <param name="a">Input matrix</param>
template <class ElemType>
/*static*/ void CPUMatrix<ElemType>::Scale(CPUMatrix<ElemType> alpha, CPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("Scale:  Input matrix a is empty.");
    if (alpha.GetNumElements() != 1)
        LogicError("Matrix alpha must be 1x1");
    CPUMatrix<ElemType>::Scale(alpha(0, 0), a);
}

template <class ElemType>
void CPUMatrix<ElemType>::InnerProduct(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, const bool isColWise)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProduct:  one of the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int k = (int) b.GetNumRows();
    const int l = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("InnerProduct: Matrices a and b should have same dimension.");

    if ((isColWise && m == 1) || !isColWise && n == 1) // in this case it's equivalent to element-wise product
    {
        c.AssignElementProductOf(a, b);
    }
    else if (isColWise) // col-wise
    {
        c.Resize(1, n);

        if (sizeof(ElemType) == sizeof(double))
        {
#pragma omp parallel for
            foreach_column (j, c)
            {
#ifdef USE_ACML
                c(0, j) = (ElemType) ddot(m, reinterpret_cast<double*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<double*>(b.m_pArray + b.LocateColumn(j)), 1);
#else
                c(0, j) = (ElemType) cblas_ddot(m, reinterpret_cast<double*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<double*>(b.m_pArray + b.LocateColumn(j)), 1);
#endif
            }
        }
        else
        {
#pragma omp parallel for
            foreach_column (j, c)
            {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
                c(0, j) = (ElemType) sdot(m, reinterpret_cast<float*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<float*>(b.m_pArray + b.LocateColumn(j)), 1);
#else
                c(0, j) = (ElemType) cblas_sdot(m, reinterpret_cast<float*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<float*>(b.m_pArray + b.LocateColumn(j)), 1);
#endif
            }
        }
    }
    else
    {
        c.Resize(m, 1);

        if (sizeof(ElemType) == sizeof(double))
        {
#pragma omp parallel for
            foreach_row (i, c)
            {
#ifdef USE_ACML
                c(i, 0) = ddot(n, reinterpret_cast<double*>(a.m_pArray + i), m, reinterpret_cast<double*>(b.m_pArray + i), m);
#else
                c(i, 0) = cblas_ddot(n, reinterpret_cast<double*>(a.m_pArray + i), m, reinterpret_cast<double*>(b.m_pArray + i), m);
#endif
            }
        }
        else
        {
#pragma omp parallel for
            foreach_row (i, c)
            {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
                c(i, 0) = sdot(n, reinterpret_cast<float*>(a.m_pArray + i), m, reinterpret_cast<float*>(b.m_pArray + i), m);
#else
                c(i, 0) = cblas_sdot(n, reinterpret_cast<float*>(a.m_pArray + i), m, reinterpret_cast<float*>(b.m_pArray + i), m);
#endif
            }
        }
    }
}

// treat matrices as vectors. do vec(a)^T vec(b)
template <class ElemType>
ElemType CPUMatrix<ElemType>::InnerProductOfMatrices(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b)
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

    if (sizeof(ElemType) == sizeof(double))
    {
#ifdef USE_ACML
        return (ElemType) ddot((int) a.GetNumElements(), reinterpret_cast<double*>(a.m_pArray), 1, reinterpret_cast<double*>(b.m_pArray), 1);
#else
        return (ElemType) cblas_ddot((int) a.GetNumElements(), reinterpret_cast<double*>(a.m_pArray), 1, reinterpret_cast<double*>(b.m_pArray), 1);
#endif
    }
    else
    {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
        return (ElemType) sdot((int) a.GetNumElements(), reinterpret_cast<float*>(a.m_pArray), 1, reinterpret_cast<float*>(b.m_pArray), 1);
#else
        return (ElemType) cblas_sdot((int) a.GetNumElements(), reinterpret_cast<float*>(a.m_pArray), 1, reinterpret_cast<float*>(b.m_pArray), 1);
#endif
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::ElementWisePower(ElemType alpha, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& c)
{
    if (a.IsEmpty())
        LogicError("Scale:  The input matrix a is empty.");

    c.Resize(a.GetNumRows(), a.GetNumCols());

    if (alpha == 2)
    {
#pragma omp parallel for
        foreach_coord (i, j, c)
        {
            c(i, j) = a(i, j) * a(i, j);
        }
    }
    else if (alpha == 3)
    {
#pragma omp parallel for
        foreach_coord (i, j, c)
        {
            c(i, j) = a(i, j) * a(i, j) * a(i, j);
        }
    }
    else
    {
#pragma omp parallel for
        foreach_coord (i, j, c)
        {
            c(i, j) = pow(a(i, j), alpha);
        }
    }
}

template <class ElemType>
bool CPUMatrix<ElemType>::AreEqual(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const ElemType threshold /*= 1e-8*/)
{
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

// see Matrix<ElemType>::TensorShuffleScaleAndAdd() for comments
template <class ElemType>
void CPUMatrix<ElemType>::TensorShuffleScaleAndAdd(ElemType keepWeight, const CPUMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c)
{
    size_t N = D * S * M * K * T;
    const ElemType* pa = a.m_pArray;
    const ElemType* pb = b.m_pArray;
    ElemType* pc = c.m_pArray;
    // Note: This code is written to match a GPU implementation. It is not super-efficient on the CPU.
    for (size_t na = 0; na < N; na++) // loop over all elements
    {
        // recover the 5 indices from the loop counter
        size_t d = na % D;
        size_t s = (na / D) % S;
        size_t m = (na / D / S) % M;
        size_t k = (na / D / S / M) % K;
        size_t t = (na / D / S / M / K) % T;
        // compute index for the a and b/c tensors
        assert(na == (((t * K + k) * M + m) * S + s) * D + d); // input tensor of dimension (D x S x M x K x T)
        size_t nb = (((t * S + s) * M + m) * K + k) * D + d;   // output tensor of dimension (D x K x M x S x T): k/K and s/S swapped
        assert(nb < N);
        // perform the computation
        ElemType cval = keepWeight ? keepWeight * pb[nb] : 0; // if weight is 0 then don't bother to read memory (efficiency) or to multiply (NaN-safe)
        cval += scaleFactor * pa[na];
        pc[nb] = cval;
    }
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::Ones(const size_t rows, const size_t cols)
{
    CPUMatrix<ElemType> c(rows, cols); // will initialize to 0
    c.SetValue(1);
    return c;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::Zeros(const size_t rows, const size_t cols)
{
    CPUMatrix<ElemType> c(rows, cols); // will initialize to 0
    c.SetValue(0);
    return c;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::Eye(const size_t rows)
{
    CPUMatrix<ElemType> c(rows, rows); // will initialize to 0
    c.SetDiagonalValue(1);
    return c;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::RandomUniform(const size_t rows, const size_t cols, const ElemType low, const ElemType high, unsigned long seed)
{
    CPUMatrix<ElemType> c(rows, cols); // will initialize to 0
    c.SetUniformRandomValue(low, high, seed);
    return c;
}

template <class ElemType>
CPUMatrix<ElemType> CPUMatrix<ElemType>::RandomGaussian(const size_t rows, const size_t cols, const ElemType mean, const ElemType sigma, unsigned long seed)
{
    CPUMatrix<ElemType> c(rows, cols); // will initialize to 0
    c.SetGaussianRandomValue(mean, sigma, seed);
    return c;
}

template <class ElemType>
bool CPUMatrix<ElemType>::HasElement(const CPUMatrix<ElemType>& mat, const ElemType v)
{
    bool bHas = false;

    bool isvFinite = std::isfinite(v);
#pragma omp parallel for
    for (long j = 0; j < mat.GetNumElements(); j++)
    {
#pragma omp flush(bHas)
        if (!bHas)
        {
            ElemType cur = mat.m_pArray[j];
            if (isvFinite && std::isfinite(cur))
            {
                if (cur == v)
                    bHas = true;
            }
            else if (std::isnan(v) && std::isnan(cur))
                bHas = true;
            else if (std::isinf(v) && std::isinf(cur) && std::signbit(v) == std::signbit(cur))
                bHas = true;
        }
    }

    return bHas;
}

//        CPUMatrix<ElemType>& AssignElementProductOfWithShiftNeg(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t shift, size_t negnumber);
//[this]=a .* b
// here, a and b must be two row vectors of the same size, i.e. [1,m]
// the inputs are two rwo vectors
// the output is a matrix of size(neg+1, col)
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementProductOfWithShiftNeg(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t shift, size_t negnumber)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOfWithShiftNeg: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("AssignElementProductOfWithShiftNeg: The input matrix dimensions do not match.");

    if (a.GetNumRows() != 1)
        InvalidArgument("AssignElementProductOfWithShiftNeg: The input matrix must be a row vector.");

    auto& us = *this;
    if (this != &a)
    {
        Resize(negnumber + 1, a.GetNumCols());
        //            Resize(a.GetNumRows(), a.GetNumCols());
    }

    long m = (long) GetNumRows(), n = (long) GetNumCols(); // a and b are of size (1,n)
    // #pragma omp parallel for

    for (long j = 0; j < n; j++)
    {
        us(0, j) = a(0, j) * b(0, j);
    }
    for (long j = 0; j < n; j++)
    {
        for (long i = 1; i < m; i++)
        {
            us(i, j) = a(0, j) * b(0, (j + shift + i - 1) % n);
        }
    }

    return *this;
}

template <class ElemType>
void CPUMatrix<ElemType>::InnerProductWithShiftNeg(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, const bool isColWise, size_t shift, size_t negnumber)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProduct:  one of the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int k = (int) b.GetNumRows();
    const int l = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("InnerProduct: Matrices a and b should have same dimension.");

    if ((isColWise && m == 1) || !isColWise && n == 1) // in this case it's equivalent to element-wise product
    {
        InvalidArgument("InnerProduct: Both matrices should be normal ones, not vectors");
        //            c.AssignElementProductOf(a, b);
    }
    else if (isColWise) // col-wise
    {
        c.Resize(negnumber + 1, n); // this line ischanged

        if (sizeof(ElemType) == sizeof(double))
        {
            for (long j = 0; j < n; j++)
            {
#ifdef USE_ACML
                c(0, j) = (ElemType) ddot(m, reinterpret_cast<double*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<double*>(b.m_pArray + b.LocateColumn(j)), 1);
#else
                c(0, j) = (ElemType) cblas_ddot(m, reinterpret_cast<double*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<double*>(b.m_pArray + b.LocateColumn(j)), 1);
#endif
            }
            for (long j = 0; j < n; j++)
            {
                for (long i = 1; i < negnumber + 1; i++)
                {
#ifdef USE_ACML
                    c(i, j) = (ElemType) ddot(m, reinterpret_cast<double*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<double*>(b.m_pArray + b.LocateColumn((j + shift + i - 1) % n)), 1);
#else
                    c(i, j) = (ElemType) cblas_ddot(m, reinterpret_cast<double*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<double*>(b.m_pArray + b.LocateColumn((j + shift + i - 1) % n)), 1);
#endif
                }
            }
        }
        else
        {
            for (long j = 0; j < n; j++)
            {
#ifdef USE_ACML
                c(0, j) = (ElemType) sdot(m, reinterpret_cast<float*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<float*>(b.m_pArray + b.LocateColumn(j)), 1);
#else
                c(0, j) = (ElemType) cblas_sdot(m, reinterpret_cast<float*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<float*>(b.m_pArray + b.LocateColumn(j)), 1);
#endif
            }
            for (long j = 0; j < n; j++)
            {
                for (long i = 1; i < negnumber + 1; i++)
                {
#ifdef USE_ACML
                    c(i, j) = (ElemType) sdot(m, reinterpret_cast<float*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<float*>(b.m_pArray + b.LocateColumn((j + shift + i - 1) % n)), 1);
#else
                    c(i, j) = (ElemType) cblas_sdot(m, reinterpret_cast<float*>(a.m_pArray + a.LocateColumn(j)), 1, reinterpret_cast<float*>(b.m_pArray + b.LocateColumn((j + shift + i - 1) % n)), 1);
#endif
                }
            }
        }
    }
    else
    {
        InvalidArgument("InnerProduct: Rowwise is not supported yet");

        c.Resize(m, 1);

        if (sizeof(ElemType) == sizeof(double))
        {
#pragma omp parallel for
            foreach_row (i, c)
            {
#ifdef USE_ACML
                c(i, 0) = (ElemType) ddot(n, reinterpret_cast<double*>(a.m_pArray + i), m, reinterpret_cast<double*>(b.m_pArray + i), m);
#else
                c(i, 0) = (ElemType) cblas_ddot(n, reinterpret_cast<double*>(a.m_pArray + i), m, reinterpret_cast<double*>(b.m_pArray + i), m);
#endif
            }
        }
        else
        {
#pragma omp parallel for
            foreach_row (i, c)
            {
#pragma warning(suppress : 4244)
#ifdef USE_ACML
                c(i, 0) = sdot(n, reinterpret_cast<float*>(a.m_pArray + i), m, reinterpret_cast<float*>(b.m_pArray + i), m);
#else
                c(i, 0) = cblas_sdot(n, reinterpret_cast<float*>(a.m_pArray + i), m, reinterpret_cast<float*>(b.m_pArray + i), m);
#endif
            }
        }
    }
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::GetARowByIndex(const CPUMatrix<ElemType>& a, size_t index)
{
    if (a.IsEmpty())
        LogicError("GetARowByIndex:  the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();

    if (index < 0 || index >= m)
        LogicError("GetARowByIndex:  the row index is out of range.");

    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    auto& us = *this;
    this->Resize(1, n);
    for (long j = 0; j < n; j++)
    {
        us(0, j) = a(index, j);
    }

    return *this;
}

// input: a, a row vector
// input: b, a matrix. b.col == a.col
// input firstmatrixfixed: If true, keep a's order. Otherwise, keep b's order
// output: c, a matrix. c.size == b.size
/*
    Example, a = [a1 a2 a3]
    b = [b11 b12 b13;
    b21 b22 b23 ]

    if true:
    shift = 1

    then c = [a1*b12 a2*b13 a3*b11
    a1*b22 a2*b23 a3*b21]

    if shift = 2
    then c = [  a1*b13 a2*b11 a3*b12
    a1*b23 a2*b21 a3*b22]
    i.e. we do column-wise shift

    if false:
    shift = 1

    then c = [a2*b11 a3*b12 a1*b13
    a2*b21 a3*b22 a1*b23]

    shift = 2

    then c = [  a3*b11 a1*b12 a2*b13
    a3*b21 a1*b22 a2*b23]


    */
template <class ElemType>
void CPUMatrix<ElemType>::ConductRowElementMultiplyWithShift(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, size_t shift, bool bFirstmatrixfixed)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProduct:  one of the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int k = (int) b.GetNumRows();
    const int l = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == 1 && n == l);                 // converting from size_t to int may cause overflow
    if (m != 1 || n != l)
        InvalidArgument("InnerProduct: Matrices a and b should have same dimension.");

    c.Resize(k, l); // c must the the same size of b

    if (bFirstmatrixfixed)
    {
        for (long j = 0; j < l; j++)
        {
            for (long i = 0; i < k; i++)
            {
                c(i, j) = a(0, j) * b(i, (j + shift) % l);
            }
        }
    }
    else
    {
        for (long j = 0; j < l; j++)
        {
            for (long i = 0; i < k; i++)
            {
                c(i, j) = a(0, (j + shift) % l) * b(i, j);
            }
        }
    }
}

//        CPUMatrix<ElemType>& AssignElementProductOfWithShift(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t shift);
//[this]=a .* b
// here, a and b must be two row vectors of the same size, i.e. [1,m]. We will do element product with shift.
// inputs are 2 row vectors
// output is a row vector
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignElementProductOfWithShift(const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, size_t shift)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOfWithShiftNeg: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("AssignElementProductOfWithShiftNeg: The input matrix dimensions do not match.");

    if (a.GetNumRows() != 1)
        InvalidArgument("AssignElementProductOfWithShiftNeg: The input matrix must be a row vector.");

    auto& us = *this;
    if (this != &a)
    {
        Resize(1, a.GetNumCols());
        //            Resize(a.GetNumRows(), a.GetNumCols());
    }

    // long m = (long)GetNumRows(), n = (long)GetNumCols();  // a and b are of size (1,n)
    long n = (long) GetNumCols(); // a and b are of size (1,n)
#pragma omp parallel for
    for (long j = 0; j < n; j++)
    {
        us(0, j) = a(0, j) * b(0, (j + shift) % n);
    }
    return *this;
}

#pragma endregion Static BLAS Functions

// 'double' version of LogAdd
double LogAddD(double x, double y)
{
    return LogAdd(x, y);
}

template <class ElemType>
ElemType CPUMatrix<ElemType>::LogAddSumOfElements() const
{
    ElemType fAlpha = (ElemType) LZERO;
    for (int k = 0; k < GetNumElements(); k++)
        fAlpha = (ElemType) LogAddD(fAlpha, m_pArray[k]);
    return fAlpha;
}

template <class ElemType>
void CPUMatrix<ElemType>::RCRFBackwardCompute(const CPUMatrix<ElemType>& alpha, CPUMatrix<ElemType>& beta,
                                              const CPUMatrix<ElemType>& lbls,
                                              const CPUMatrix<ElemType>& pair_scores)
{
    int iNumPos = (int) lbls.GetNumCols();
    int iNumLab = (int) lbls.GetNumRows();

    int lastLbl = -1;
    for (int ik = 0; ik < lbls.GetNumRows(); ik++)
        if (lbls(ik, iNumPos - 1) != 0)
        {
            lastLbl = ik;
            break;
        }

    beta.Resize(iNumLab, iNumPos);

    for (int t = iNumPos - 1; t >= 0; t--)
    {
#pragma omp parallel for
        for (int k = 0; k < iNumLab; k++)
        {
            _rcrfBackwardCompute(t, k, alpha, beta, pair_scores);
        }
    }
};

/// the kernel function for RCRF backward computation
template <class ElemType>
void CPUMatrix<ElemType>::_rcrfBackwardCompute(size_t t, size_t k, const CPUMatrix<ElemType>& alpha,
                                               CPUMatrix<ElemType>& beta,
                                               const CPUMatrix<ElemType>& pair_scores)
{
    size_t iNumLab = alpha.GetNumRows();
    size_t iNumPos = alpha.GetNumCols();

    ElemType fSum;
    ElemType fTmp = (ElemType) LZERO;
    if (t == iNumPos - 1)
    {
        fSum = (ElemType) LZERO;
        for (int j = 0; j < iNumLab; j++)
        {
            fSum = (ElemType) LogAddD(fSum, alpha(j, t));
        }

        fTmp = alpha(k, t) - fSum;
        beta(k, t) = fTmp;
    }
    else
    {
        for (int j = 0; j < iNumLab; j++)
        {
            fSum = (ElemType) LZERO;
            for (int m = 0; m < iNumLab; m++)
            {
                fSum = (ElemType) LogAddD(fSum, alpha(m, t) + pair_scores(j, m));
            }

            fTmp = (ElemType) LogAddD(fTmp, beta(j, t + 1) + alpha(k, t) + pair_scores(j, k) - fSum);
        }
        beta(k, t) = fTmp;
    }
}

template <class ElemType>
void CPUMatrix<ElemType>::RCRFTransGrdCompute(const CPUMatrix<ElemType>& lbls,
                                              const CPUMatrix<ElemType>& alpha,
                                              const CPUMatrix<ElemType>& beta,
                                              const CPUMatrix<ElemType>& pair_scores,
                                              CPUMatrix<ElemType>& grd)
{
    int iNumPos = (int) alpha.GetNumCols();
    int iNumLab = (int) alpha.GetNumRows();

    int firstLbl = -1;
    for (int ik = 0; ik < lbls.GetNumRows(); ik++)
        if (lbls(ik, 0) != 0)
        {
            firstLbl = ik;
            break;
        }

    for (size_t tPos = 0; tPos < iNumPos; tPos++)
    {
        CPUMatrix<ElemType> b = beta.ColumnSlice(tPos, 1);
        CPUMatrix<ElemType> a;
        if (tPos > 0)
            a = alpha.ColumnSlice(tPos - 1, 1);

#pragma omp parallel for
        for (int i = 0; i < iNumLab; i++)
        {
            _rcrfTransGrdCompute(i, lbls, alpha, beta, pair_scores, grd, tPos);
        }

        // transition score
        int i = -1;
        if (tPos == 0)
            i = firstLbl;
        else
        {
            for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, tPos - 1) != 0)
                {
                    i = ik;
                    break;
                }
        }

        int j = -1;
        for (int ik = 0; ik < lbls.GetNumRows(); ik++)
        {
            if (lbls(ik, tPos) != 0)
            {
                j = ik;
                break;
            }
        }

        grd(j, i) -= 1.0;
    }
};

template <class ElemType>
void CPUMatrix<ElemType>::_rcrfTransGrdCompute(size_t i,
                                               const CPUMatrix<ElemType>& lbls,
                                               const CPUMatrix<ElemType>& alpha,
                                               const CPUMatrix<ElemType>& beta,
                                               const CPUMatrix<ElemType>& pair_scores,
                                               CPUMatrix<ElemType>& grd,
                                               const size_t tPos // position
                                               )
{
    int iNumLab = (int) alpha.GetNumRows();

    int firstLbl = -1;
    for (int ik = 0; ik < lbls.GetNumRows(); ik++)
        if (lbls(ik, 0) != 0)
        {
            firstLbl = ik;
            break;
        }

    CPUMatrix<ElemType> b = beta.ColumnSlice(tPos, 1);
    CPUMatrix<ElemType> a;
    if (tPos > 0)
        a = alpha.ColumnSlice(tPos - 1, 1);

    {
        ElemType fTmp = (ElemType) LZERO;
        for (int j = 0; j < iNumLab; j++)
        {
            if (tPos == 0)
            {
                if (i == firstLbl)
                {
                    fTmp = 0;
                }
                else
                {
                    fTmp = (ElemType) LZERO;
                }
            }
            else
            {
                fTmp = a(i, 0);
            }
            fTmp += pair_scores(j, i);

            ElemType fSum = (ElemType) LZERO;
            for (int k = 0; k < iNumLab; k++)
            {
                ElemType fTmp2;
                if (tPos == 0)
                {
                    if (k == firstLbl)
                    {
                        fTmp2 = 0;
                    }
                    else
                    {
                        fTmp2 = (ElemType) LZERO;
                    }
                }
                else
                {
                    fTmp2 = a(k, 0);
                }
                fSum = (ElemType) LogAddD(fSum, fTmp2 + pair_scores(j, k));
            }

            fTmp -= fSum;
            fTmp += b(j, 0);

            grd(j, i) += exp(fTmp);
        }
    }
};
template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::DropFrame(const CPUMatrix<ElemType>& label, const CPUMatrix<ElemType>& gamma, const ElemType& threshhold)
{
    auto& us = *this;
    if (us.GetNumCols() != gamma.GetNumCols() || us.GetNumRows() != gamma.GetNumRows())
        LogicError("DropFrame: target matrix is not in the same size as gamm matrix.");

#pragma omp parallel for
    foreach_column (j, label)
    {

        bool dropframe = false;
        foreach_row (i, label)
        {
            if (fabs(label(i, j) - 1.0f) < 0.1)
            {
                if (gamma(i, j) < threshhold)
                    dropframe = true;
                break;
            }
        }

        foreach_row (i, label)
        {
            us(i, j) = 0.0f;
        }
    }

    return *this;
}

template <class ElemType>
CPUMatrix<ElemType>& CPUMatrix<ElemType>::AssignSequenceError(const ElemType hsmoothingWeight, const CPUMatrix<ElemType>& label,
                                                              const CPUMatrix<ElemType>& dnnoutput, const CPUMatrix<ElemType>& gamma, ElemType alpha)
{
    auto& us = *this;
    foreach_coord (i, j, us)
        us(i, j) += alpha * (label(i, j) - (1 - hsmoothingWeight) * dnnoutput(i, j) - hsmoothingWeight * gamma(i, j));
    return *this;
}

// note: this function does not depend on the <ElemType> parameter
template <class ElemType>
int CPUMatrix<ElemType>::SetNumThreads(int numThreads)
{
    if (numThreads == 0) // use default
        return numThreads;

    int mthreads = (int) std::thread::hardware_concurrency();

    if (numThreads <= 0)
        numThreads = std::max(1, mthreads + numThreads);
    if (numThreads > mthreads)
        numThreads = mthreads;

#ifdef _OPENMP
    omp_set_num_threads(numThreads);
    numThreads = omp_get_max_threads();

#ifdef USE_ACML
    acmlsetnumthreads(numThreads);
#elif defined(USE_MKL)
    mkl_set_num_threads(numThreads);
#elif defined(USE_OPENBLAS)
    openblas_set_num_threads(numThreads);
#endif
#endif
    return numThreads;
}

// =======================================================================
// TensorView support
// =======================================================================

// To save time, this makes extensive use of templates and macros.

// -----------------------------------------------------------------------
// function to compute the value for a given output location (perform reduction if needed)
// -----------------------------------------------------------------------

// perform loop over reduction index m
// This function is declared inside a wrapper struct to allow partial specialization (m = -1).
template <class ElemType, typename OPFN, size_t N, int m>
struct TensorOpReduction
{
    // reduction case (non-reduction case is specialized)
    static inline ElemType Loop(array<ElemType*, N> pointers, const OPFN& opfn,
                                const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
    {
        array<ptrdiff_t, N - 1> strides;   // N-1 because last one is the result pointer, which is unused in reduction
        for (size_t i = 0; i < N - 1; i++) // N = a small constant, this will be unrolled
            strides[i] = reducingStrides[i][(size_t) m];
        double /*ElemType*/ aggregate = 0;
        for (size_t dim = reducingOpDims[(size_t) m]; dim-- > 0;)
        {
            // need to descend into one loop deeper
            aggregate += TensorOpReduction<ElemType, OPFN, N, m - 1>::Loop(pointers, opfn, reducingOpDims, reducingStrides);
            // advance the pointers
            for (size_t i = 0; i < N - 1; i++)
                pointers[i] += strides[i]; // note: last pointer (result) is unused and untouched here
        }
        return (ElemType) aggregate;
    }
};

// perform loop over reduction index m
// This is the specialized version for m = -1, which terminates the recursion.
template <class ElemType, typename OPFN, size_t N>
struct TensorOpReduction<ElemType, OPFN, N, -1>
{
    static inline ElemType Loop(array<ElemType*, N> pointers, const OPFN& opfn,
                                const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, N>&)
    {
        return opfn(pointers); // finally we are doing some work!!!
    }
};

// -----------------------------------------------------------------------
// perform loop over regular index k for N-nary operations (N counting the output)
// -----------------------------------------------------------------------

// perform loop over regular index k and reducing index m for N operands (counting the output)
template <class ElemType, typename OPFN, size_t N, bool vectorizable, int m, int k>
struct TensorOpIteration
{
    static inline void Loop(ElemType beta, array<ElemType*, N> pointers, ElemType alpha, const OPFN& opfn,
                            const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrides,
                            const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
    {
        // non-scalar case: still nested result loops left
        array<ptrdiff_t, N> strides;
        for (size_t i = 0; i < N; i++) // N = a small constant, this will be unrolled
            strides[i] = regularStrides[i][(size_t) k];
        for (size_t dim = regularOpDims[(size_t) k]; dim-- > 0;)
        {
            // need to descend into one loop deeper
            TensorOpIteration<ElemType, OPFN, N, vectorizable, m, k - 1>::Loop(beta, pointers, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
            // advance the pointers
            for (size_t i = 0; i < N; i++)
                pointers[i] += strides[i];
        }
    }
};

// Special version for innermost loop with strides all being 1 and no further reduction. Compiler can use SSE.
// This is a very common case, e.g. adding vectors or computing the Sigmoid.
template <class ElemType, typename OPFN>
struct TensorOpIteration<ElemType, OPFN, 3, true /*vectorizable*/, -1 /*no reduction*/, 0 /*innermost loop*/>
{
    static inline void Loop(ElemType beta, array<ElemType*, 3> pointers, ElemType alpha, const OPFN& opfn,
                            const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
                            const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides)
    {
        ElemType* pa = pointers[0];
        ElemType* pb = pointers[1];
        ElemType* pc = pointers[2];
        size_t K = regularOpDims[0];
        // special-case beta and alpha to allow the compiler to short-circuit it
        if (beta != 0)
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, 3, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(beta, array<ElemType*, 3>{pa + k, pb + k, pc + k}, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        else if (alpha != 1)
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, 3, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(0, array<ElemType*, 3>{pa + k, pb + k, pc + k}, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        else
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, 3, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(0, array<ElemType*, 3>{pa + k, pb + k, pc + k}, 1, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        // TODO: According to Amit, the VS compiler is not able to vectorize into lambdas. Solution: change the lambda to take an N, or to implement the loop inside (with 1 element by default).
        // TODO: The signedness of k (required for omp) causes an extra sign-extend.
        // TODO: OMP adds LOTS of overhead. Do we need a guard, a min size when to use it?
    }
};
// and unary
template <class ElemType, typename OPFN>
struct TensorOpIteration<ElemType, OPFN, 2, true /*vectorizable*/, -1 /*no reduction*/, 0 /*innermost loop*/>
{
    static inline void Loop(ElemType beta, array<ElemType*, 2> pointers, ElemType alpha, const OPFN& opfn,
                            const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                            const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
    {
        ElemType* pa = pointers[0];
        ElemType* pb = pointers[1];
        size_t K = regularOpDims[0];
        // special-case beta and alpha to allow the compiler to short-circuit it
        if (beta != 0)
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, 2, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(beta, array<ElemType*, 2>{pa + k, pb + k}, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        else if (alpha != 1)
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, 2, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(0, array<ElemType*, 2>{pa + k, pb + k}, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        else
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, 2, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(0, array<ElemType*, 2>{pa + k, pb + k}, 1, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    }
};

template <class ElemType, typename OPFN, size_t N, bool vectorizable, int m>
struct TensorOpIteration<ElemType, OPFN, N, vectorizable, m, -1>
{
    static inline void Loop(ElemType beta, array<ElemType*, N> pointers, ElemType alpha, const OPFN& opfn,
                            const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, N>&,
                            const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
    {
        // we are at element level for the result: perform the op (there may still be reduction)
        ElemType val = TensorOpReduction<ElemType, OPFN, N, m>::Loop(pointers, opfn, reducingOpDims, reducingStrides);
        // scale
        val *= alpha;
        // combine with previous value in target matrix, then write it out
        auto* pout = pointers.back();
        if (beta != 0)
            val += beta * *pout;
        // save
        *pout = val;
        return;
    }
};

// -----------------------------------------------------------------------
// map runtime parameters N to template parameters
// -----------------------------------------------------------------------

// tensor operation with k+1 dimensions (-1 means scalar)
template <class ElemType, typename OPFN, size_t N, int k>
static void TensorOpWithRegularLoop(ElemType beta, const array<ElemType*, N>& pointers, ElemType alpha, const OPFN& opfn,
                                    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrides,
                                    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
{
    size_t dims = reducingOpDims.size();
    switch (dims)
    {
    case 2:
        return TensorOpIteration<ElemType, OPFN, N, false /*vectorizable*/, 1, k>::Loop(beta, pointers, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 1:
        return TensorOpIteration<ElemType, OPFN, N, false /*vectorizable*/, 0, k>::Loop(beta, pointers, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 0:
    {
        // if all leading dimensions are 1, we can let the compiler do some unrolling
        bool leadingAllOne = true;
        for (size_t i = 0; i < N; i++)
            leadingAllOne &= k >= 0 && regularStrides[i][0] == 1;
        if (leadingAllOne) // special version that uses a hard-coded increment of 1 for all leading dimensions
            return TensorOpIteration<ElemType, OPFN, N, true /*vectorizable*/, -1, k>::Loop(beta, pointers, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        else
            return TensorOpIteration<ElemType, OPFN, N, false /*vectorizable*/, -1, k>::Loop(beta, pointers, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    }
    default:
        LogicError("TensorOp: %d non-flattened reduction dimensions are not supported.", (int) dims);
    }
}

// tensor operation, generalized in number of arguments, operation already provided as a lambda
// This function now expands into different k.
template <class ElemType, typename OPFN, size_t N>
static void TensorOpWithFn(ElemType beta, array<ElemType*, N> pointers, ElemType alpha, const OPFN& opfn,
                           const array<size_t, N>& offsets,
                           const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrides,
                           const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
{
    for (size_t i = 0; i < N; i++) // N = a small constant, this will be unrolled
        pointers[i] += offsets[i];
    size_t dims = regularOpDims.size();
    switch (dims)
    {
    case 4:
        return TensorOpWithRegularLoop<ElemType, OPFN, N, 3>(beta, pointers, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 3:
        return TensorOpWithRegularLoop<ElemType, OPFN, N, 2>(beta, pointers, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 2:
        return TensorOpWithRegularLoop<ElemType, OPFN, N, 1>(beta, pointers, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 1:
        return TensorOpWithRegularLoop<ElemType, OPFN, N, 0>(beta, pointers, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 0:
        return TensorOpWithRegularLoop<ElemType, OPFN, N, -1>(beta, pointers, alpha, opfn, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    default:
        LogicError("TensorOp: %d non-flattened input dimensions are not supported.", (int) dims);
    }
}

// -----------------------------------------------------------------------
// entry points from Matrix.cpp; also map op to a lambda
// -----------------------------------------------------------------------

// perform unary operation 'op' on a giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
// This maps 'op' to a lambda.
template <class ElemType>
void CPUMatrix<ElemType>::TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, ElemType alpha, ElementWiseOperator op,
                                   const array<size_t, 2>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
// TODO: Change the lambda to take a pointer and a number of elements, so that we can pass it 1 or 4 elements, in order for it to SSE-vectorize.
#define CaseUnaryTensorOp(oper)                                                        \
    case ElementWiseOperator::op##oper:                                                \
        return TensorOpWithFn(beta, pointers, alpha, [](const array<ElemType*, 2>& pp) \
                              {                                                        \
                                  return Op##oper((*(pp[0])));                         \
                              },                                                       \
                              offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides)

    array<ElemType*, 2> pointers = {a.m_pArray, m_pArray};
    switch (op)
    {
        ForAllUnaryOps(CaseUnaryTensorOp);
    default:
        LogicError("TensorUnaryOp: Unknown op code %d.", (int) op);
    }
}

// perform binary operation 'op' on a and b giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
// This maps 'op' to a lambda.
template <class ElemType>
void CPUMatrix<ElemType>::TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, ElemType alpha, ElementWiseOperator op,
                                   const array<size_t, 3>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides)
{
#define CaseBinaryTensorOp(oper)                                                       \
    case ElementWiseOperator::op##oper:                                                \
        return TensorOpWithFn(beta, pointers, alpha, [](const array<ElemType*, 3>& pp) \
                              {                                                        \
                                  return Op##oper((*(pp[0])), (*(pp[1])));             \
                              },                                                       \
                              offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides)

    array<ElemType*, 3> pointers = {a.m_pArray, b.m_pArray, m_pArray};
    switch (op)
    {
        ForAllBinaryOps(CaseBinaryTensorOp);
    default:
        LogicError("TensorBinaryOp: Unknown op code %d.", (int) op);
    }
}

// perform ternary operation 'op' on a, and c giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
// This maps 'op' to a lambda.
template <class ElemType>
void CPUMatrix<ElemType>::TensorOp(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& c, ElemType alpha, ElementWiseOperator op,
                                   const array<size_t, 4>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides)
{
#define CaseTernaryTensorOp(oper)                                                      \
    case ElementWiseOperator::op##oper:                                                \
        return TensorOpWithFn(beta, pointers, alpha, [](const array<ElemType*, 4>& pp) \
                              {                                                        \
                                  return Op##oper((*(pp[0])), (*(pp[1])), (*(pp[2]))); \
                              },                                                       \
                              offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides)

    array<ElemType*, 4> pointers = {a.m_pArray, b.m_pArray, c.m_pArray, m_pArray};
    switch (op)
    {
        ForAllTernaryOps(CaseTernaryTensorOp);
    default:
        LogicError("TensorTernaryOp: Unknown op code %d.", (int) op);
    }
}

// =======================================================================
// explicit instantiations
// =======================================================================
template class MATH_API CPUMatrix<float>;
template class MATH_API CPUMatrix<double>;

// We use Matrix<char> as the backing store for QuantizedMatrix
// Let's explicitly instantiate the methods we need for that purpose
template CPUMatrix<char>::CPUMatrix(const size_t numRows, const size_t numCols);
template CPUMatrix<char>::CPUMatrix(const size_t numRows, const size_t numCols, char* pArray, const size_t matrixFlags);
template CPUMatrix<char>::CPUMatrix();
template CPUMatrix<char>::CPUMatrix(CPUMatrix<char> const&);
template CPUMatrix<char>::CPUMatrix(CPUMatrix<char>&&);
template size_t CPUMatrix<char>::LocateElement(size_t, size_t) const;
template CPUMatrix<char>::~CPUMatrix();
template CPUMatrix<char> CPUMatrix<char>::ColumnSlice(size_t startColumn, size_t numCols) const;
template CPUMatrix<char>& CPUMatrix<char>::operator=(CPUMatrix<char>&&);
template void CPUMatrix<char>::SetValue(const char);
template void CPUMatrix<char>::SetValue(const size_t numRows, const size_t numCols, char* pArray, size_t matrixFlags);
template void CPUMatrix<char>::SetValue(CPUMatrix<char> const&);
template void CPUMatrix<char>::Resize(const size_t numRows, const size_t numCols, bool growOnly);

template CPUMatrix<int>::CPUMatrix(const size_t, const size_t, int*, const size_t);
template CPUMatrix<int>::~CPUMatrix();

} } }
