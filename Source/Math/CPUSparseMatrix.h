//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <stdio.h>
#include "CPUMatrix.h"
#include <map>
#include <unordered_map>

#ifdef _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#endif /* Linux - already defined in CPUMatrix.h */

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class MATH_API CPUSparseMatrix : public BaseMatrix<ElemType>
{
    typedef BaseMatrix<ElemType> Base;
    using Base::m_elemSizeAllocated;
    using Base::m_computeDevice;
    using Base::m_externalBuffer;
    using Base::m_format;
    using Base::m_numCols;
    using Base::m_numRows;
    using Base::m_nz;
    using Base::m_pArray; // without this, base members would require to use thi-> in GCC
    using Base::Clear;
    using Base::NzCount;

public:
    using Base::OwnBuffer;
    using Base::IsEmpty;

private:
    void ZeroInit();
    void CheckInit(const MatrixFormat format);
    void ReleaseMemory();

public:
    explicit CPUSparseMatrix(const MatrixFormat format);
    CPUSparseMatrix(const MatrixFormat format, const size_t numRows, const size_t numCols, const size_t size);
    CPUSparseMatrix(const CPUSparseMatrix<ElemType>& deepCopyFrom);                      // copy constructor, deep copy
    CPUSparseMatrix<ElemType>& operator=(const CPUSparseMatrix<ElemType>& deepCopyFrom); // assignment operator, deep copy
    CPUSparseMatrix(CPUSparseMatrix<ElemType>&& moveFrom);                               // move constructor, shallow copy
    CPUSparseMatrix<ElemType>& operator=(CPUSparseMatrix<ElemType>&& moveFrom);          // move assignment operator, shallow copy
    ~CPUSparseMatrix();

public:
    using Base::GetNumCols;
    using Base::GetNumRows;

    void SetValue(const size_t row, const size_t col, ElemType val);
    void SetValue(const CPUSparseMatrix<ElemType>& /*val*/);

    void ShiftBy(int /*numShift*/)
    {
        NOT_IMPLEMENTED;
    }

    size_t BufferSize() const
    {
        return m_elemSizeAllocated * sizeof(ElemType);
    }
    ElemType* BufferPointer() const;
    inline size_t GetNumElemAllocated() const
    {
        return m_elemSizeAllocated;
    }

    CPUSparseMatrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;
    CPUMatrix<ElemType> CopyColumnSliceToDense(size_t startColumn, size_t numCols) const;

    CPUMatrix<ElemType> DiagonalToDense() const;

    void SetGaussianRandomValue(const ElemType /*mean*/, const ElemType /*sigma*/, unsigned long /*seed*/)
    {
        NOT_IMPLEMENTED;
    }

    void SetMatrixFromCSCFormat(const CPUSPARSE_INDEX_TYPE* h_CSCCol, const CPUSPARSE_INDEX_TYPE* h_Row, const ElemType* h_Val,
                                const size_t nz, const size_t numRows, const size_t numCols);

    static void MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& lhs, const bool transposeA,
                                       const CPUSparseMatrix<ElemType>& rhs, const bool transposeB, ElemType beta, CPUMatrix<ElemType>& c);

    static void MultiplyAndAdd(ElemType alpha, const CPUMatrix<ElemType>& lhs, const bool transposeA,
                               const CPUSparseMatrix<ElemType>& rhs, const bool transposeB, CPUSparseMatrix<ElemType>& c);

    static void ScaleAndAdd(const ElemType alpha, const CPUSparseMatrix<ElemType>& lhs, CPUMatrix<ElemType>& c);

    static bool AreEqual(const CPUSparseMatrix<ElemType>& a, const CPUSparseMatrix<ElemType>& b, const ElemType threshold = 1e-8);

    // sum(vec(a).*vec(b))
    static ElemType InnerProductOfMatrices(const CPUSparseMatrix<ElemType>& /*a*/, const CPUMatrix<ElemType>& /*b*/)
    {
        NOT_IMPLEMENTED;
    }

    static void AddScaledDifference(const ElemType /*alpha*/, const CPUSparseMatrix<ElemType>& /*a*/, const CPUMatrix<ElemType>& /*b*/, CPUMatrix<ElemType>& /*c*/,
                                    bool /*bDefaultZero*/)
    {
        NOT_IMPLEMENTED;
    }
    static void AddScaledDifference(const ElemType /*alpha*/, const CPUMatrix<ElemType>& /*a*/, const CPUSparseMatrix<ElemType>& /*b*/, CPUMatrix<ElemType>& /*c*/,
                                    bool /*bDefaultZero*/)
    {
        NOT_IMPLEMENTED;
    }

    int GetComputeDeviceId() const
    {
        return -1;
    }

    void Resize(const size_t numRows, const size_t numCols, size_t numNZElemToReserve = 10000, const bool growOnly = true, bool keepExistingValues = false);
    void Reset();

    const ElemType operator()(const size_t row, const size_t col) const
    {
        if (col >= m_numCols || row >= m_numRows)
        {
            RuntimeError("Position outside matrix dimensions");
        }

        if (m_format == MatrixFormat::matrixFormatSparseCSC)
        {
            size_t start = m_compIndex[col];
            size_t end = m_compIndex[col + 1];
            for (size_t p = start; p < end; p++)
            {
                size_t i = m_unCompIndex[p];
                if (i == row)
                {
                    return m_pArray[p];
                }
            }

            return 0;
        }
        else
        {
            NOT_IMPLEMENTED;
        }
    }

public:
    void NormalGrad(CPUMatrix<ElemType>& c, const ElemType momentum);
    ElemType Adagrad(CPUMatrix<ElemType>& c, const bool needAveMultiplier);

public:
    CPUSparseMatrix<ElemType>& InplaceTruncateTop(const ElemType threshold);
    CPUSparseMatrix<ElemType>& InplaceTruncateBottom(const ElemType threshold);
    CPUSparseMatrix<ElemType>& InplaceTruncate(const ElemType threshold);
    CPUSparseMatrix<ElemType>& InplaceSoftThreshold(const ElemType threshold);

    ElemType FrobeniusNorm() const; // useful for comparing CPU and GPU results

    ElemType SumOfAbsElements() const; // sum of all abs(elements)
    ElemType SumOfElements() const;    // sum of all elements

public:
    void Print(const char* matrixName, ptrdiff_t rowStart, ptrdiff_t rowEnd, ptrdiff_t colStart, ptrdiff_t colEnd) const;
    void Print(const char* matrixName = NULL) const; // print whole matrix. can be expensive

public:
    const ElemType* NzValues() const
    {
        return m_nzValues;
    }
    inline ElemType* NzValues()
    {
        return m_nzValues;
    }
    size_t NzSize() const
    {
        return sizeof(ElemType) * m_nz;
    } // actual number of element bytes in use

    CPUSPARSE_INDEX_TYPE* MajorIndexLocation() const
    {
        return m_unCompIndex;
    } // this is the major index, row/col ids in CSC/CSR format
    size_t MajorIndexCount() const
    {
        return m_nz;
    }
    size_t MajorIndexSize() const
    {
        return sizeof(CPUSPARSE_INDEX_TYPE) * MajorIndexCount();
    } // actual number of major index bytes in use

    CPUSPARSE_INDEX_TYPE* SecondaryIndexLocation() const
    {
        return m_compIndex;
    } // this is the compressed index, col/row in CSC/CSR format
    size_t SecondaryIndexCount() const
    {
        if (m_format & matrixFormatCompressed)
        {
            size_t cnt = (m_format & matrixFormatRowMajor) ? m_numRows : m_numCols;
            if (cnt > 0)
                cnt++; // add an extra element on the end for the "max" value
            return cnt;
        }
        else
            return m_nz; // COO format
    }
    // get size for compressed index
    size_t SecondaryIndexSize() const
    {
        return (SecondaryIndexCount()) * sizeof(CPUSPARSE_INDEX_TYPE);
    }

    // the column and row locations will swap based on what format we are in. Full index always follows the data array
    CPUSPARSE_INDEX_TYPE* RowLocation() const
    {
        return (m_format & matrixFormatRowMajor) ? SecondaryIndexLocation() : MajorIndexLocation();
    }
    size_t RowSize() const
    {
        return (m_format & matrixFormatRowMajor) ? SecondaryIndexSize() : MajorIndexSize();
    }
    CPUSPARSE_INDEX_TYPE* ColLocation() const
    {
        return (m_format & matrixFormatRowMajor) ? MajorIndexLocation() : SecondaryIndexLocation();
    }
    size_t ColSize() const
    {
        return (m_format & matrixFormatRowMajor) ? MajorIndexSize() : SecondaryIndexSize();
    } // actual number of bytes in use

private:
    int m_colIdx; // used to SetValue()
    size_t m_compIndexSize;
    ElemType* m_nzValues;

    // non-zero values are stored in m_pArray
    CPUSPARSE_INDEX_TYPE* m_unCompIndex; // row/col ids in CSC/CSR format
    CPUSPARSE_INDEX_TYPE* m_compIndex;   // begin ids of col/row in CSC/CSR format

    size_t m_blockSize;    // block size
    size_t* m_blockIds;    // block ids
    size_t m_blockIdShift; // used to get efficient slice, actual col = blockIds[j] - m_blockIdShift

    CPUSparseMatrix* m_sliceOf; // if this is a slice, then this points to the owning matrix object that we sliced from
};

typedef CPUSparseMatrix<float> CPUSingleSparseMatrix;
typedef CPUSparseMatrix<double> CPUDoubleSparseMatrix;
} } }
