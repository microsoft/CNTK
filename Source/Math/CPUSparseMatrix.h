//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <stdio.h>
#include "CPUMatrix.h"
//#include "GPUMatrix.h"
//#include "GPUSparseMatrix.h"
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
    using Base::m_numRows;
    using Base::m_numCols;
    using Base::m_sliceViewOffset;
    using Base::SetBuffer;
    using Base::HasExternalBuffer;
    using Base::GetNumStorageRows;
    using Base::SetNumStorageRows;
    using Base::GetNumStorageCols;
    using Base::SetNumStorageCols;
    using Base::SetComputeDeviceId;
    using Base::SetSizeAllocated;
    using Base::GetSizeAllocated;
    using Base::GetCompIndex;
    using Base::SetCompIndex;
    using Base::GetUnCompIndex;
    using Base::SetUnCompIndex;
    using Base::GetCompIndexSize;
    using Base::SetCompIndexSize;
    using Base::GetColIdx;
    using Base::SetColIdx;
    using Base::GetBlockSize;
    using Base::SetBlockSize;
    using Base::GetBlockIds;
    using Base::SetBlockIds;
    using Base::GetBlockIdShift;
    using Base::SetBlockIdShift;
    using Base::ZeroInit;
    using Base::ZeroValues;
    using Base::m_sob;
    using Base::ShallowCopyFrom;
    using Base::VerifyResizable;
public:
    using Base::VerifyWritable;
    using Base::GetComputeDeviceId;
    using Base::Buffer;
    using Base::GetNumRows;
    using Base::GetNumCols;
    using Base::GetNumElements;
    using Base::OwnBuffer;
    using Base::GetFormat;
    using Base::SetFormat;
    using Base::IsEmpty;

private:
    void ZeroInit();
    void CheckInit(const MatrixFormat format);

public:
    explicit CPUSparseMatrix(const MatrixFormat format);
    CPUSparseMatrix(const MatrixFormat format, const size_t numRows, const size_t numCols, const size_t size);
    CPUSparseMatrix(const CPUSparseMatrix<ElemType>& deepCopyFrom);                      // copy constructor, deep copy
    CPUSparseMatrix<ElemType>& operator=(const CPUSparseMatrix<ElemType>& deepCopyFrom); // assignment operator, deep copy
    CPUSparseMatrix(CPUSparseMatrix<ElemType>&& moveFrom);                               // move constructor, shallow copy
    CPUSparseMatrix<ElemType>& operator=(CPUSparseMatrix<ElemType>&& moveFrom);          // move assignment operator, shallow copy
    ~CPUSparseMatrix();

public:

    void SetValue(const size_t row, const size_t col, ElemType val);
    //void SetValue(const CPUMatrix<ElemType>& /*val*/);
    //void SetValue(const GPUMatrix<ElemType>& /*val*/);
    void SetValue(const CPUSparseMatrix<ElemType>& /*val*/);
    //void SetValue(const GPUSparseMatrix<ElemType>& /*val*/);

    void MaskColumnsValue(const CPUMatrix<char>& columnsMask, ElemType val, size_t numColsPerMaskEntry);

    CPUSparseMatrix<ElemType>& AssignOneHot(const CPUMatrix<ElemType>& a, vector<size_t>& shape, size_t axis);

    CPUSparseMatrix<ElemType>& DoGatherColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUSparseMatrix<ElemType>& a, ElemType alpha);
    CPUSparseMatrix<ElemType>& DoScatterColumnsOf(ElemType beta, const CPUMatrix<ElemType>& idx, const CPUSparseMatrix<ElemType>& a, ElemType alpha);

    size_t BufferSize() const
    {
        return GetSizeAllocated() * sizeof(ElemType);
    }
    ElemType* Data() const;
    inline size_t GetNumElemAllocated() const
    {
        return GetSizeAllocated();
    }

    CPUSparseMatrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;
    CPUMatrix<ElemType> CopyColumnSliceToDense(size_t startColumn, size_t numCols) const;
    void AssignColumnSliceToDense(CPUMatrix<ElemType>& slice, size_t startColumn, size_t numCols) const;

    CPUMatrix<ElemType> DiagonalToDense() const;

    void SetGaussianRandomValue(const ElemType /*mean*/, const ElemType /*sigma*/, unsigned long /*seed*/)
    {
        NOT_IMPLEMENTED;
    }

    void SetMatrixFromCSCFormat(const CPUSPARSE_INDEX_TYPE* h_CSCCol, const CPUSPARSE_INDEX_TYPE* h_Row, const ElemType* h_Val,
                                const size_t nz, const size_t numRows, const size_t numCols);

    void SetMatrixFromSBCFormat(const size_t* blockIds, const ElemType* val, const size_t numBlocks, const size_t numRows, const size_t numCols);

    // Dense * Sparse -> Dense
    static void MultiplyAndWeightedAdd(ElemType alpha, const CPUMatrix<ElemType>& lhs, const bool transposeA,
                                       const CPUSparseMatrix<ElemType>& rhs, const bool transposeB, ElemType beta, CPUMatrix<ElemType>& c);

    // Sparse * Dense -> Dense
    static void MultiplyAndWeightedAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& lhs, const bool transposeA,
                                       const CPUMatrix<ElemType>& rhs, const bool transposeB, ElemType beta, CPUMatrix<ElemType>& c);

    // Dense * Sparse -> Sparse
    static void MultiplyAndAdd(ElemType alpha, const CPUMatrix<ElemType>& lhs, const bool transposeA,
                               const CPUSparseMatrix<ElemType>& rhs, const bool transposeB, CPUSparseMatrix<ElemType>& c);

    static void ColumnwiseScaleAndWeightedAdd(ElemType alpha, const CPUSparseMatrix<ElemType>& a, const CPUMatrix<ElemType>& v, ElemType beta, CPUMatrix<ElemType>& c);

    static void ScaleAndAdd(const ElemType alpha, const CPUSparseMatrix<ElemType>& lhs, CPUMatrix<ElemType>& c);

    static bool AreEqual(const CPUSparseMatrix<ElemType>& a, const CPUSparseMatrix<ElemType>& b, const ElemType threshold = 1e-8);

    // sum(vec(a).*vec(b))
    static ElemType InnerProductOfMatrices(const CPUSparseMatrix<ElemType>& /*a*/, const CPUMatrix<ElemType>& /*b*/)
    {
        NOT_IMPLEMENTED;
    }

    static void InnerProduct(const CPUSparseMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& c, const bool isColWise);

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


    // Allocate actually allocates the storage space for numNZElemToReserve elements. This is different than resizing, which changes the dimensions of the underlying matrix.
    // Unfortunately numRows/numCols need to be passed in in the case of various matrix formats (e.g., SparseCSC), because some of the dimensions allocated depend on the
    // dimensions of the matrix.
    void Allocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve = 10000, const bool growOnly = true, bool keepExistingValues = false); // matrix format will affect the size to allocate
    // RequireSizeAndAllocate is required by SpasreMatrix since resizing the dimensions and allocating storage are different operations. Since a Resize can entail changing
    // MatrixFormat, we offer an overload for that case.
    void RequireSizeAndAllocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly = true, bool keepExistingValues = true); // matrix format will affect the size to allocate
    // Otherwise we will just use the current MatrixFormat.
    void RequireSizeAndAllocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve = 10000, const bool growOnly = true, bool keepExistingValues = false);
    // Sparse matrix RequireSize is similar to dense matrix RequireSize in that it will only allocate the minimum amount of storage requried to successfully create the matrix.
    // This is required because some formats (e.g., SparseCSC) require the SecondaryIndexLocation to have valid data in order to compute m_nz. Otherwise this method would not
    // update the storage at all.
    void RequireSize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat format, const bool growOnly = true);
    void RequireSize(const size_t numRows, const size_t numCols, const MatrixFormat format, const bool growOnly = true)
    {
        return RequireSize(numRows, numCols, 0, format, growOnly);
    }
    // Allows RequireSize to be called without modifying the MatrixFormat.
    void RequireSize(const size_t numRows, const size_t numCols, const bool growOnly = true);
    // Resizes the dimensions of the underlying sparse matrix object. Since the caller may have a hint for m_nz, we allow that to be passed, but this is terrible design. In the
    // future if we want better separation between allocation and resizing, this interface should be updated to be less clunky.
    void Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly = true); // matrix format will affect the size to allocate
    void Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve = 10000, const bool growOnly = true);

    void Reset();

    const ElemType operator()(const size_t row, const size_t col) const
    {
        if (col >= m_numCols || row >= m_numRows)
        {
            RuntimeError("Position outside matrix dimensions");
        }

        if (GetFormat() == MatrixFormat::matrixFormatSparseCSC)
        {
            size_t start = SecondaryIndexLocation()[col];
            size_t end = SecondaryIndexLocation()[col + 1];
            for (size_t p = start; p < end; p++)
            {
                size_t i = MajorIndexLocation()[p];
                if (i == row)
                {
                    return ((ElemType*)Buffer())[p];
                }
            }

            return 0;
        }
        else if (GetFormat() == MatrixFormat::matrixFormatSparseBlockCol)
        {
            for (size_t blockId = 0; blockId < GetBlockSize(); blockId++)
            {
                size_t blockCol = GetBlockIds()[blockId] - GetBlockIdShift();
                if (blockCol == col)
                {
                    return ((ElemType*)Buffer())[blockId * GetNumRows() + row];
                }
            }
            return 0;
        }
        NOT_IMPLEMENTED;
    }

public:
    void NormalGrad(CPUMatrix<ElemType>& c, const ElemType momentum, bool unitGainMomentum = true);
    ElemType Adagrad(CPUMatrix<ElemType>& c, const bool needAveMultiplier);
    void AdaDelta(CPUMatrix<ElemType>& c, CPUMatrix<ElemType>& functionValues, ElemType rho, ElemType epsilon);

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
        return Data();
    }
    inline ElemType* NzValues()
    {
        return Data();
    }

    size_t NzCount() const
    {
        if (GetFormat() == matrixFormatSparseCSC)
            return GetCompIndex()[GetNumCols()] - GetCompIndex()[0];
        else if (GetFormat()== matrixFormatSparseCSR)
            return GetCompIndex()[GetNumRows()] - GetCompIndex()[0];
        else if (GetFormat() == matrixFormatSparseBlockCol)
            return GetBlockSize() * GetNumRows();
        else
            NOT_IMPLEMENTED;
    }

    size_t NzSize() const
    {
        return sizeof(ElemType) * NzCount();
    } // actual number of element bytes in use

    void SetBlockSize(size_t newBlockSize)
    {
        BaseMatrix<ElemType>::SetBlockSize(newBlockSize);
    }

    size_t GetBlockSize() const
    {
        return BaseMatrix<ElemType>::GetBlockSize();
    }

    size_t* BlockIdsLocation() const
    {
        if ((GetFormat() != matrixFormatSparseBlockCol) && (GetFormat() != matrixFormatSparseBlockRow))
            LogicError("CPUSparseMatrix::BlockIdsLocation is only applicable to sparse block formats");

        return GetBlockIds();
    }

    CPUSPARSE_INDEX_TYPE* MajorIndexLocation() const
    {
        return (GetUnCompIndex() + 
            ((GetFormat() == matrixFormatSparseCSC || GetFormat() == matrixFormatSparseCSR) ? GetCompIndex()[m_sliceViewOffset] : 0));
    } // this is the major index, row/col ids in CSC/CSR format

    size_t MajorIndexCount() const
    {
        return NzCount();
    }

    size_t MajorIndexSize() const
    {
        return sizeof(CPUSPARSE_INDEX_TYPE) * MajorIndexCount();
    } // actual number of major index bytes in use

    // Returns the start of the secondary index valid for the slice-view.
    // Secondary index provides the offset to the data buffer for the values.
    // E.g. for CSC the the first nonzero value of column k is Buffer(SecondaryIndexLocation[k])
    CPUSPARSE_INDEX_TYPE* SecondaryIndexLocation() const
    {
        return GetCompIndex() + m_sliceViewOffset;
    }
    
    size_t SecondaryIndexCount() const
    {
        if (GetFormat() & matrixFormatCompressed)
        {
            size_t cnt = (GetFormat() & matrixFormatRowMajor) ? m_numRows : m_numCols;
            if (cnt > 0)
                cnt++; // add an extra element on the end for the "max" value
            return cnt;
        }
        else
            return NzCount(); // COO format
    }
    // get size for compressed index
    size_t SecondaryIndexSize() const
    {
        return (SecondaryIndexCount()) * sizeof(CPUSPARSE_INDEX_TYPE);
    }

    // the column and row locations will swap based on what format we are in. Full index always follows the data array
    CPUSPARSE_INDEX_TYPE* RowLocation() const
    {
        return (GetFormat() & matrixFormatRowMajor) ? SecondaryIndexLocation() : MajorIndexLocation();
    }
    size_t RowSize() const
    {
        return (GetFormat() & matrixFormatRowMajor) ? SecondaryIndexSize() : MajorIndexSize();
    }
    CPUSPARSE_INDEX_TYPE* ColLocation() const
    {
        return (GetFormat() & matrixFormatRowMajor) ? MajorIndexLocation() : SecondaryIndexLocation();
    }
    size_t ColSize() const
    {
        return (GetFormat() & matrixFormatRowMajor) ? MajorIndexSize() : SecondaryIndexSize();
    } // actual number of bytes in use
};

typedef CPUSparseMatrix<float> CPUSingleSparseMatrix;
typedef CPUSparseMatrix<double> CPUDoubleSparseMatrix;
} } }
