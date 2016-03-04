//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "GPUMatrix.h"
#include "CPUSparseMatrix.h"
#include <functional>

namespace Microsoft { namespace MSR { namespace CNTK {

//GPU Sparse Matrix, using cuSPARSE library.
//By default we are assuming CSR representation
// NOTE m_elemSizeAllocated (in base matrix) means the number of non-zero elements we have allocated space
// We are packing the CSR format (pointed to by m_pArray) as follows:
// ElemType elements[m_elemSizeAllocated]
// int colIdx[m_elemSizeAllocated]
// int rowIdxStart[m_numRows+1]

template <class ElemType>
class MATH_API GPUSparseMatrix : public BaseMatrix<ElemType>
{
public:
    typedef BaseMatrix<ElemType> Base;
    using Base::m_numRows;
    using Base::m_numCols;
    using Base::m_pArray;
    using Base::m_elemSizeAllocated;
    using Base::m_sliceViewOffset;
    using Base::m_nz;
    using Base::m_format;
    using Base::m_computeDevice;
    using Base::m_externalBuffer;
    using Base::OwnBuffer;
    using Base::GetFormat;
    using Base::SetFormat;
    using Base::GetNumRows;
    using Base::GetNumCols;
    using Base::SetComputeDeviceId;
    using Base::SetNzCount;
    using Base::Clear;
    // without this, base members would require to use thi-> in GCC
public:
    using Base::IsEmpty;

    GPUSparseMatrix(const size_t numRows, const size_t numCols, const size_t numNZ, DEVICEID_TYPE computeDevice, const MatrixFormat matrixFormat = MatrixFormat::matrixFormatSparseCSR);

    explicit GPUSparseMatrix(DEVICEID_TYPE computeDevice, const MatrixFormat matrixFormat = MatrixFormat::matrixFormatSparseCSR);

    GPUSparseMatrix(const GPUSparseMatrix<ElemType>&);

    GPUSparseMatrix(const GPUMatrix<ElemType>&, const MatrixFormat matrixFormat = MatrixFormat::matrixFormatSparseCSR);

    // #ifndef __unix__
    GPUSparseMatrix(GPUSparseMatrix<ElemType>&&);
    // #endif    /* LINUX */

    ~GPUSparseMatrix();

public:
    void Reset();

public:
    // return col pointer, which is immediately following the non-zero element
    // in memory format is always in the following order:
    // Non-zero data elements, Full index locations, compressed index locations
    // In CSR row data is compressed, in CSC col data is compressed
    // Special Note: for the matrix may be a read-only column slice view of another
    // matrix (only supported for CSC format today) and hence the NzValues needs
    // to be offset accordingly.
    inline const ElemType* NzValues() const
    {
        return m_format != matrixFormatSparseCSC ? m_pArray : m_pArray + SecondaryIndexValueAt(0);
    }
    inline ElemType* NzValues()
    {
        return m_format != matrixFormatSparseCSC ? m_pArray : m_pArray + SecondaryIndexValueAt(0);
    }
    inline size_t NzSize() const
    {
        return sizeof(ElemType) * m_nz;
    } // actual number of element bytes in use
    inline size_t GetNumNZElements() const
    {
        return m_nz;
    }

    GPUSPARSE_INDEX_TYPE* MajorIndexLocation() const // row/col ids in CSC/CSR format, blockId2col/blockId2row in BlockCol/BlockRow format
    {
        return (GPUSPARSE_INDEX_TYPE*) (m_pArray + m_elemSizeAllocated);
    }

    GPUSPARSE_INDEX_TYPE* MajorIndexLocationWithSliceViewOffset() const
    {
        return (MajorIndexLocation() + (m_format == matrixFormatSparseCSC ? SecondaryIndexValueAt(0) : 0));
    }

    // TODO: Comment these methods more thoroughly, e.g., why it uses numNZ instead of m_elemSizeAllocated.
    size_t MajorIndexCount() const
    {
        return MajorIndexCount(m_numRows, m_numCols, m_nz, m_format);
    }
    size_t MajorIndexCount(const size_t numRows, const size_t numCols, const size_t numNZ, const MatrixFormat format) const
    {
        if (format == matrixFormatSparseBlockCol)
            return numCols;
        else if (format == matrixFormatSparseBlockRow)
            return numRows;
        else
            return numNZ;
    }
    size_t MajorIndexSize() const // actual number of major index bytes in use
    {
        return sizeof(GPUSPARSE_INDEX_TYPE) * MajorIndexCount();
    }

    GPUSPARSE_INDEX_TYPE* SecondaryIndexLocation() const // compressed index, col/row in CSC/CSR format, col2blockId/row2blockId in BlockCol/BlockRow format
    {
        if (m_format == matrixFormatSparseBlockCol)
            return MajorIndexLocation() + m_numCols;
        else if (m_format == matrixFormatSparseBlockRow)
            return MajorIndexLocation() + m_numRows;
        else
            return MajorIndexLocation() + m_elemSizeAllocated + m_sliceViewOffset;
        // return MajorIndexLocation() + m_elemSizeAllocated + m_sliceViewOffset;
    }
    size_t SecondaryIndexCount(const size_t numRows, const size_t numCols, const size_t numNZReserved, const MatrixFormat format) const
    {
        if (format == matrixFormatSparseBlockCol)
            return numCols;
        else if (format == matrixFormatSparseBlockRow)
            return numRows;
        else if (format == matrixFormatSparseCSC)
            return numCols + 1;
        else if (format == matrixFormatSparseCSR)
            return numRows + 1;
        else
            return numNZReserved; // COO format
    }

    size_t SecondaryIndexCount() const
    {
        return SecondaryIndexCount(m_numRows, m_numCols, m_elemSizeAllocated, m_format);
    }

    // get size for compressed index
    size_t SecondaryIndexSize() const
    {
        return (SecondaryIndexCount()) * sizeof(GPUSPARSE_INDEX_TYPE);
    }

    size_t BufferSizeNeeded(const size_t numRows, const size_t numCols, const size_t numNZ, const MatrixFormat format) const
    {
        return sizeof(ElemType) * numNZ + sizeof(GPUSPARSE_INDEX_TYPE) * (MajorIndexCount(numRows, numCols, numNZ, format) + SecondaryIndexCount(numRows, numCols, numNZ, format));
    }

    inline size_t BufferSizeAllocated() const
    {
        return m_totalBufferSizeAllocated;
    }
    inline ElemType* BufferPointer() const
    {
        return m_pArray;
    }
    inline size_t GetNumElemAllocated() const
    {
        return m_elemSizeAllocated;
    }
    inline size_t GetSizeElemAllocated() const
    {
        return sizeof(ElemType) * m_elemSizeAllocated;
    }

    // the column and row locations will swap based on what format we are in. Full index always follows the data array
    GPUSPARSE_INDEX_TYPE* RowLocation() const
    {
        // not a valid function for other formats
        assert(m_format == matrixFormatSparseCSC || m_format == matrixFormatSparseCSR);

        return (m_format & matrixFormatRowMajor) ? SecondaryIndexLocation() : MajorIndexLocation();
    }
    size_t RowSize() const // actual number of bytes in use
    {
        // not a valid function for other formats
        assert(m_format == matrixFormatSparseCSC || m_format == matrixFormatSparseCSR);

        return (m_format & matrixFormatRowMajor) ? SecondaryIndexSize() : MajorIndexSize();
    }
    GPUSPARSE_INDEX_TYPE* ColLocation() const
    {
        // not a valid function for other formats
        assert(m_format == matrixFormatSparseCSC || m_format == matrixFormatSparseCSR);

        return (m_format & matrixFormatRowMajor) ? MajorIndexLocation() : SecondaryIndexLocation();
    }
    size_t ColSize() const // actual number of bytes in use
    {
        // not a valid function for other formats
        assert(m_format == matrixFormatSparseCSC || m_format == matrixFormatSparseCSR);

        return (m_format & matrixFormatRowMajor) ? MajorIndexSize() : SecondaryIndexSize();
    }
    GPUSPARSE_INDEX_TYPE SecondaryIndexValueAt(size_t idx) const;
    GPUSPARSE_INDEX_TYPE* BlockId2ColOrRow() const
    {
        // not a valid function for other formats
        assert(m_format == matrixFormatSparseBlockCol || m_format == matrixFormatSparseBlockRow);
        return MajorIndexLocation();
    }
    GPUSPARSE_INDEX_TYPE* ColOrRow2BlockId() const
    {
        // not a valid function for other formats
        assert(m_format == matrixFormatSparseBlockCol || m_format == matrixFormatSparseBlockRow);
        return SecondaryIndexLocation();
    }
    void SetValue(const GPUSparseMatrix<ElemType>& deepCopyFrom);
    void SetValue(const CPUSparseMatrix<ElemType>& deepCopyFrom);
    void SetValue(const GPUMatrix<ElemType>& denseMatrix, const MatrixFormat matrixFormat);
    void SetValue(const GPUMatrix<ElemType>& denseMatrix);

    void Reshape(const size_t numRows, const size_t numCols);
    void ResizeAsAndCopyIndexFrom(const GPUSparseMatrix<ElemType>& a, const bool growOnly = true);
    void Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly = true, bool keepExistingValues = true); // matrix format will affect the size to allocate
    void Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve = 10000, const bool growOnly = true, bool keepExistingValues = false);

    GPUSparseMatrix<ElemType> Transpose() const;
    void InplaceTranspose();
    GPUSparseMatrix<ElemType>& AssignTransposeOf(const GPUSparseMatrix<ElemType>& a);

    GPUSparseMatrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;
    GPUMatrix<ElemType> CopyColumnSliceToDense(size_t startColumn, size_t numCols) const;

    GPUMatrix<ElemType> DiagonalToDense() const;

    GPUMatrix<ElemType> CopyToDenseMatrix() const;
    void CopyToDenseMatrix(GPUMatrix<ElemType>& denseMatrix) const;
    void CopyToCPUSparseMatrix(CPUSparseMatrix<ElemType>& cpuSparseMatrix) const;
    void ChangeDeviceTo(DEVICEID_TYPE toId);

    GPUSparseMatrix<ElemType>& operator=(const GPUSparseMatrix<ElemType>& deepCopy);
    // #ifndef __unix__
    GPUSparseMatrix<ElemType>& operator=(GPUSparseMatrix<ElemType>&& moveFrom);
    // #endif    /* LINUX */
    GPUSparseMatrix<ElemType> operator+(const GPUSparseMatrix<ElemType>& a) const;
    GPUSparseMatrix<ElemType> operator-(const GPUSparseMatrix<ElemType>& a) const;
    GPUSparseMatrix<ElemType>& operator^=(const ElemType alpha);     // element-wise power
    GPUSparseMatrix<ElemType> operator^(const ElemType alpha) const; // element-wise power
    GPUSparseMatrix<ElemType>& operator*=(const ElemType alpha);
    GPUSparseMatrix<ElemType> operator*(const ElemType alpha) const;
    GPUSparseMatrix<ElemType>& AssignElementPowerOf(const GPUSparseMatrix<ElemType>& a, const ElemType power);

    bool IsEqualTo(const GPUSparseMatrix<ElemType>& a, const ElemType threshold = 1e-8) const;
    bool IsEqualTo(const GPUMatrix<ElemType>& a, const ElemType threshold = 1e-8) const;

public:
    virtual DEVICEID_TYPE GetComputeDeviceId(void) const;

    // Sets sparse matrix in CSR format. this acts as deep copy
    void SetMatrixFromCSRFormat(const CPUSPARSE_INDEX_TYPE* h_CSRRow, const CPUSPARSE_INDEX_TYPE* h_Col, const ElemType* h_Val,
                                const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice = false, const DEVICEID_TYPE devId = -1);
    void SetMatrixFromCSCFormat(const CPUSPARSE_INDEX_TYPE* h_CSCCol, const CPUSPARSE_INDEX_TYPE* h_Row, const ElemType* h_Val,
                                const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice = false, const DEVICEID_TYPE devId = -1);

    // Gets sparse matrix in CSR format. this acts as deep copy. All passed pointers must be NULL. the function will allocate memory itself.
    void GetMatrixFromCSRFormat(CPUSPARSE_INDEX_TYPE*& h_CSRRow, CPUSPARSE_INDEX_TYPE*& h_Col, ElemType*& h_Val, size_t& numElemAllocated, size_t& nz, size_t& numRows, size_t& numCols) const;

    void GetMatrixFromCSCFormat(CPUSPARSE_INDEX_TYPE*& h_CSCCol, CPUSPARSE_INDEX_TYPE*& h_Row, ElemType*& h_Val, size_t& numElemAllocated, size_t& nz, size_t& numRows, size_t& numCols) const;

    void ConvertToSparseFormat(MatrixFormat newFormat);
    void ConvertToSparseFormat(MatrixFormat newFormat, GPUSparseMatrix<ElemType>& outMatrix) const;

    bool IsValid() const;

public:
    GPUSparseMatrix<ElemType>& ElementInverse();
    GPUSparseMatrix<ElemType>& AssignElementInverseOf(const GPUSparseMatrix<ElemType>& a);

    GPUSparseMatrix<ElemType>& InplaceLinearRectifierDerivative();
    GPUSparseMatrix<ElemType>& AssignLinearRectifierDerivativeOf(const GPUSparseMatrix<ElemType>& a);

    GPUSparseMatrix<ElemType>& InplaceSigmoid();
    GPUSparseMatrix<ElemType>& AssignSigmoidOf(const GPUSparseMatrix<ElemType>& a);

    GPUSparseMatrix<ElemType>& InplaceTanh();
    GPUSparseMatrix<ElemType>& AssignTanhOf(const GPUSparseMatrix<ElemType>& a);

    GPUSparseMatrix<ElemType>& InplaceSqrt();
    GPUSparseMatrix<ElemType>& AssignSqrtOf(const GPUSparseMatrix<ElemType>& a);

    GPUSparseMatrix<ElemType>& InplaceExp();
    GPUSparseMatrix<ElemType>& AssignExpOf(const GPUSparseMatrix<ElemType>& a);

    GPUSparseMatrix<ElemType>& InplaceLog();
    GPUSparseMatrix<ElemType>& AssignLogOf(const GPUSparseMatrix<ElemType>& a);

    GPUSparseMatrix<ElemType>& InplaceAbs();
    GPUSparseMatrix<ElemType>& AssignAbsOf(const GPUSparseMatrix<ElemType>& a);

    GPUSparseMatrix<ElemType>& InplaceTruncate(const ElemType threshold);
    GPUSparseMatrix<ElemType>& InplaceSoftThreshold(const ElemType threshold);

    GPUSparseMatrix<ElemType>& InplaceTruncateBottom(const ElemType threshold);
    GPUSparseMatrix<ElemType>& AssignTruncateBottomOf(const GPUSparseMatrix<ElemType>& a, const ElemType threshold);
    GPUSparseMatrix<ElemType>& InplaceTruncateTop(const ElemType threshold);
    GPUSparseMatrix<ElemType>& AssignTruncateTopOf(const GPUSparseMatrix<ElemType>& a, const ElemType threshold);

    GPUSparseMatrix<ElemType>& SetToZeroIfAbsLessThan(const ElemType threshold);

    ElemType SumOfElements() const;    // sum of all elements
    ElemType SumOfAbsElements() const; // sum of all abs(elements)
    ElemType FrobeniusNorm() const;
    ElemType MatrixNormInf() const;
    ElemType MatrixNorm1() const;
    ElemType MatrixNorm0() const
    {
        return (ElemType) GetNumNZElements();
    };

public:
    // Performs C = alpha ? op ( S ) ? D + beta ? C; Where S is sparse and D and C are dense
    static void MultiplyAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const bool transposeA, const GPUSparseMatrix<ElemType>& b,
                                       const bool transposeB, ElemType beta, GPUMatrix<ElemType>& c);
    static void MultiplyAndWeightedAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& S, const bool transposeS, const GPUMatrix<ElemType>& D,
                                       const bool transposeD, ElemType beta, GPUMatrix<ElemType>& C);
    static void MultiplyAndAdd(ElemType alpha, const GPUMatrix<ElemType>& lhs, const bool transposeA, const GPUSparseMatrix<ElemType>& rhs,
                               const bool transposeB, GPUSparseMatrix<ElemType>& c);
    static void ScaleAndAdd(const ElemType alpha, const GPUSparseMatrix<ElemType>& lhs, GPUMatrix<ElemType>& c);
    static void ConvolveAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& lhs, const bool transposeA, const GPUSparseMatrix<ElemType>& rhs,
                                       const bool transposeB, ElemType beta, GPUMatrix<ElemType>& c, size_t numChannels, size_t horizontalSubsample, bool padding, bool channelwise);
    static void TensorShuffleScaleAndAdd(ElemType keepWeight, const GPUSparseMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const GPUSparseMatrix<ElemType>& b, GPUSparseMatrix<ElemType>& c);

    void NormalGrad(GPUMatrix<ElemType>& c, const ElemType momentum);
    ElemType Adagrad(GPUMatrix<ElemType>& c, const bool needAveMultiplier);

    static void Multiply(const GPUSparseMatrix<ElemType>& S, const GPUMatrix<ElemType>& D, GPUMatrix<ElemType>& C);
    static void Multiply(const GPUMatrix<ElemType>& D, const GPUSparseMatrix<ElemType>& S, GPUMatrix<ElemType>& C);
    static void Multiply(const GPUSparseMatrix<ElemType>& S1, bool transposeS1, const GPUSparseMatrix<ElemType>& S2, bool transposeS2, GPUSparseMatrix<ElemType>& C);
    GPUSparseMatrix<ElemType>& AssignProductOf(const GPUSparseMatrix<ElemType>& a, const bool transposeA, const GPUSparseMatrix<ElemType>& b, const bool transposeB);

    static ElemType InnerProductOfMatrices(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
    static ElemType InnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b);
    static void ScaleAndAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& a, ElemType beta, const GPUSparseMatrix<ElemType>& b, GPUSparseMatrix<ElemType>& c);
    static void ScaleAndAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& a, ElemType beta, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
    static void ScaleAndAdd(ElemType alpha, const GPUMatrix<ElemType>& a, ElemType beta, const GPUSparseMatrix<ElemType>& b, GPUMatrix<ElemType>& c);
    static void Scale(ElemType alpha, GPUSparseMatrix<ElemType>& a);
    static void ElementWisePower(ElemType alpha, const GPUSparseMatrix<ElemType>& a, GPUSparseMatrix<ElemType>& c);
    static bool AreEqual(const GPUSparseMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b, const ElemType threshold = 1e-8);
    static bool AreEqual(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const ElemType threshold = 1e-8);
    static bool AreEqual(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b, const ElemType threshold = 1e-8);

    // For these two, I should also add a version which would return GPUSparseMatrix, since Dense.*Sparse =Sparse.*Dense=Sparse
    static GPUMatrix<ElemType> ElementProductOf(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
    static GPUMatrix<ElemType> ElementProductOf(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b);

public:
    // See: http://stackoverflow.com/questions/4660123/overloading-friend-operator-for-template-class/4661372#4661372
    template <class ElemTypeDummy>
    friend MATH_API File& operator>>(File& stream, GPUSparseMatrix<ElemTypeDummy>& us);
    template <class ElemTypeDummy>
    friend MATH_API File& operator<<(File& stream, const GPUSparseMatrix<ElemTypeDummy>& us);

private:
    void* ReserveTempHostBuffer(const size_t sizeInByte) const;
    template <class OutType, class InType>
    static void CopyBuffer(OutType* outBuffer, const InType* inBuffer, const size_t size);

private:
    void ZeroInit(const MatrixFormat matrixFormat, const DEVICEID_TYPE deviceId);

private:
    void performElementWiseFunction(const ElementWiseOperator kind, const GPUSparseMatrix<ElemType>& src);
    void DeepCopy(const GPUSparseMatrix<ElemType>& deepCopyFrom);
    void ReleaseMemory();
    void PrepareBuffer(const size_t numRows, const size_t numCols, const bool canReuseBuffer, std::function<size_t(GPUSPARSE_INDEX_TYPE* csrRowPtrC)> func);

    size_t ElemCountFromBufferSize(const size_t numRows, const size_t numCols, const MatrixFormat format, const size_t totalBufferSize) const;
    size_t ElemCountFromBufferSize() const;
    DEVICEID_TYPE PrepareDevice(const DEVICEID_TYPE deviceId = -1) const;
    size_t IdentifyRowsWithValues() const;

private:
    size_t m_totalBufferSizeAllocated;

    // used by the blockCol and blockRow format
    size_t m_blockSize;                      // block size
    mutable GPUSPARSE_INDEX_TYPE* m_rowToId; // the id showing the order row number is observed in the nnz values.

    mutable void* m_tempHostBuffer; // used to copy values.
    mutable size_t m_tempHostBufferSize;

    GPUSparseMatrix* m_sliceOf; // if this is a slice, then this points to the owning matrix object that we sliced from
};

}}}
