//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

//#include "CPUMatrix.h"
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
    using Base::m_sliceViewOffset;
    using Base::HasExternalBuffer;
    using Base::SetBuffer;
    using Base::GetNumStorageRows;
    using Base::SetNumStorageRows;
    using Base::GetNumStorageCols;
    using Base::SetNumStorageCols;
    using Base::SetComputeDeviceId;
    using Base::SetSizeAllocated;
    using Base::GetSizeAllocated;
    using Base::ZeroInit;
    using Base::ZeroValues;
    using Base::m_sob;
    using Base::ShallowCopyFrom;
    using Base::GetBlockSize;
    using Base::SetBlockSize;
    using Base::GetTempHostBuffer;
    using Base::SetTempHostBuffer;
    using Base::GetTempHostBufferSize;
    using Base::SetTempHostBufferSize;
    using Base::BufferSizeAllocated;
    using Base::GetTempDeviceBuffer;
    using Base::VerifyResizable;
    // without this, base members would require to use thi-> in GCC
public:
    using Base::VerifyWritable;
    using Base::ReserveTempDeviceBuffer;
    using Base::GetComputeDeviceId;
    using Base::Buffer;
    using Base::GetNumRows;
    using Base::GetNumCols;
    using Base::SetNumRows;
    using Base::SetNumCols;
    using Base::GetNumElements;
    using Base::OwnBuffer;
    using Base::GetFormat;
    using Base::SetFormat;
    using Base::IsEmpty;

private:
    // caching of NzCount(); see there for explanation
    static const GPUSPARSE_INDEX_TYPE INVALID_NZ_COUNT = -1;
    mutable GPUSPARSE_INDEX_TYPE m_cachedNzCount = INVALID_NZ_COUNT;

public:
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
    // TODO: Name this clearly. Does it include the m_sliceViewOffset or not?
    //       E.g., rename to CSxNzValueArray
    inline const ElemType* NzValues() const
    {
        return Data();
    }

    inline ElemType* NzValues()
    {
        return Data();
    }

private:
    // determine the NzCount from the GPU-side arrays
    // This is a GPU sync, and thus expensive. NzCount() caches this value.
    // TODO: Clarify by renaming: Does this apply to the entire buffer or the view?
    GPUSPARSE_INDEX_TYPE FetchNzCount() const
    {
        if (GetFormat() == matrixFormatSparseCSC)
            return SecondaryIndexValueAt(GetNumCols()) - SecondaryIndexValueAt(0);
        if (GetFormat() == matrixFormatSparseCSR)
            return SecondaryIndexValueAt(GetNumRows()) - SecondaryIndexValueAt(0);
        else if (GetFormat() == matrixFormatSparseBlockCol)
            return (int)(GetNumRows() * GetBlockSize());
        else
            NOT_IMPLEMENTED;
    }
public:
    // The non-zero count is used often in CPU-side preparations, and fetching it from the
    // GPU is expensive; so it is cached.
    GPUSPARSE_INDEX_TYPE NzCount() const
    {
        if (!HasCachedNzCount())
            m_cachedNzCount = FetchNzCount();
        return m_cachedNzCount;
    }
    // Call this after all GPU-side functions that may update the element set.
    void InvalidateCachedNzCount() const
    {
        m_cachedNzCount = INVALID_NZ_COUNT;
    }
    // Call this if you want to copy the value in case it is available.
    bool HasCachedNzCount() const
    {
        return m_cachedNzCount != INVALID_NZ_COUNT;
    }
    // Call this when you know the value CPU-side already.
    void UpdateCachedNzCount(GPUSPARSE_INDEX_TYPE nzCount, bool shouldVerify=true) const // m_cachedNzCount is mutable
    {
        m_cachedNzCount = nzCount;
        // for now do some safety check (this is a GPU barrier again though, so remove one day)
        if (shouldVerify) // (e.g. can't check if transfer was not synchronous)
            VerifyCachedNzCount(m_cachedNzCount); // (to be sure)
    }
    // and for diagnostics
    void VerifyCachedNzCount(GPUSPARSE_INDEX_TYPE nzCount) const
    {
#if 1   // this branch disables the verification
        nzCount;
#else   // verification is costly and completely undoes the benefit of caching
        if (FetchNzCount() != nzCount)
            LogicError("VerifyCachedNzCount: GPU-side NzCount unexpectedly changed/not synced??");
#endif
    }

    inline size_t NzBytes() const { return sizeof(ElemType) * NzCount(); } // actual number of element bytes in use

    inline size_t GetNumNZElements() const { return NzCount(); } // TODO: GetNumNZElements() and NzCount() are redundant. Remove one of the two.

    void ClearNzCount();

    // Terminology and store of sparse matrices:
    //  - nz array    [0..nzIndex..nzCount-1]: array of non-zero element values, all concatenated linearly in memory
    //  - major index [0..nzIndex..nzCount-1]: corresponding index of non-zero element, same layout as nz values. CSC: row index (RowLocation(), does not include slice-view offset).
    //  - secondary index [0..j..J-1] -> firstNzIndex: first nzIndex of the non-sparse dimension. CSC: j=colIndex. Secondary index = ColLocation(), *does* include slice-view offset.
    // The three arrays are concatenated in memory.
    // CSC:
    //   This stores the columns sparsely. Columns are indexable; individual elements are not. The memory layout is equivalent to:
    //     struct CSCLayout
    //     {
    //         ElemType nzArray[nzCount];                    // [nzIndex] nz array. --TODO: m_elemSizeAllocated == nzCount? Or can there be a gap, too?
    //         GPUSPARSE_INDEX_TYPE majorIndex[nzCount];     // [nzIndex] major index
    //         GPUSPARSE_INDEX_TYPE secondaryIndex[numCols]; // [colIndex] secondary index
    //     };
    //   In presence of a slice view, the nz array and major index are commonly the base of the entire array, while the secondary index is offset by the slice view offset.
    // SBC (sparseBlockCol):
    //   This stores a matrix with zero or non-zero columns.
    //     struct SBCLayout
    //     {
    //       typedef ElemType E;
    //       typedef GPUSPARSE_INDEX_TYPE I;
    //       E block_storage           [num_rows, num_blocks];                      // [rowIndex, block_id]
    //       E block_storage_reserved  [num_rows, num_allocated_blocks-num_blocks]; // [rowIndex, extra block_id]
    //       I block_id_to_col         [num_blocks];         // [block_id] -> col_index
    //       I block_id_to_col_padding [num_cols-num_blocks] // [extra block_id]
    //                       // TODO: is the padding region uninitialized  or SparseIndex_NotAssigned?
    //       I col_to_block_id         [num_cols];           // [col_index] -> block id, or SparseIndex_NotAssigned, or SparseIndex_Pending
    //       // where _ names are concepts, camelCase is actual members, and PascalCase are actual method names
    //     };
    //    Terms:
    //     - block_storage := a matrix formed by the non-zero columns
    //     - "(stored) block" := a non=zero column in block storage. We do not call those columns as to separate the concepts
    //     - col_index, "(nominal) column index", [0,num_cols) := index of column of the uncompressed matrix that this object represents
    //     - num_rows := number of rows
    //     - num_cols := number of columns in the uncompressed matrix that this object represents
    //     - block_id, "block index", [0,num_blocks) := index of a stored block
    //     - num_blocks := number of stored blocks == number of non-zero columns
    //     - block_storage_reserved  := additional reserved space for more stored blocks, can be grown like std::vector::reserve
    //     - block_id_to_col_padding := block_id_to_col and block_id_to_col_padding together are always num_cols long
    //    Synonyms used at places:
    //     - nzArray := refers to block_storage (plus -Reserved depending on context)
    //     - nzCount == size of block storage, in elements
    //               == num_blocks * num_rows
    //     - "major index" := col_to_block_id table
    //     - "secondary index" := block_id_to_col table (plus -Padding depending on context)
    //    Relevant class members:
    //     - m_pArray == address of entire data
    //                == &block_storage[0,0]
    //     - m_blockSize == size of block storage, in blocks
    //                   == number of non-zero columns stored in block storage
    //                   == num_blocks
    //     - m_elemSizeAllocated == size of block_storage plus reserved, in elements (not in bytes, not in columns)
    //                           == num_rows * num_allocated_blocks
    //                           == size of block_storage and block_storage_reserved, in elements
    //                              // block_id_to_col follows after this in memory, and then col_to_block_id
    //    Relevant methods:
    //     - Buffer() == m_pArray
    //     - GetSizeAllocated() == m_elemSizeAllocated
    //     - GetBlockSize() == m_blockSize == num_blocks
    //     - BlockId2ColOrRow() == &block_id_to_col[0]
    //                          == MajorIndexLocation() == Buffer() + GetSizeAllocated() == m_pArray + m_elemSizeAllocated
    //     - ColOrRow2BlockId() == &col_to_block_id[0]
    //                          == SecondaryIndexLocation() == MajorIndexLocation() + GetNumCols()
    //     - GetNumCols() == num_cols
    //     - GetNumRows() == num_rows

    // major index location:
    //  - CSC/CSR: nz value indices, matching layout of nz values
    //  - other: ...TODO
    // Use this function to get the array that gets indexed by the secondary index (=nz offsets), to make sure sliceViewOffset is correct.
    // The sparse matrix representation of CSC/CSR uses one large value array (m_pArray) with offsets to the Major/Secondary index location.
    // m_pArray [0:nz] are the nz elements, [nz:2*nz] is the major index location, and [2*nz:2*nz+numcols/rows] is the secondary
    // index location.
    // Note: This function does not include the slice-view offset, while in CPUSparseMatrix::MajorIndexLocation() does. Should clean this up.
    GPUSPARSE_INDEX_TYPE* MajorIndexLocation() const // row/col ids in CSC/CSR format, blockId2col/blockId2row in BlockCol/BlockRow format
    {
        return (GPUSPARSE_INDEX_TYPE*) (Buffer() + GetSizeAllocated());
    }

    // location of major index storage, including slice-view offset
    // Use this for copy operations.
    // Note: Data is already offset by the sliceViewOffset, so we can just add the allocated size to get the start of the MajorIndexLoc
    GPUSPARSE_INDEX_TYPE* MajorIndexLocationWithSliceViewOffset() const
    {
        return (GPUSPARSE_INDEX_TYPE*) (Data() + GetSizeAllocated());
    }

    // MajorIndexCount depends on the format.
    //     1. SparseBlockCol: numCols
    //     2. SparseBlockRow: numRows
    //     3. SparseCSC/CSR : nnz
    // Note that NzCount is the number of non-zero elements currently in use. GetSizeAllocated is the number
    //    of nz values that will fit in the current buffer.
    size_t MajorIndexCount() const
    {
        return MajorIndexCount(GetNumRows(), GetNumCols(), NzCount(), GetFormat());
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

    size_t ComputeMaxNZElemFromBufferSize(size_t numRows, size_t numCols, size_t bufferSize, MatrixFormat format)
    {
        if (format == matrixFormatSparseBlockCol)
            return ( bufferSize - 2 * sizeof(GPUSPARSE_INDEX_TYPE) * numCols) / sizeof(ElemType);
        else if (format == matrixFormatSparseBlockRow)
            return (bufferSize - 2 * sizeof(GPUSPARSE_INDEX_TYPE) * numRows) / sizeof(ElemType);
        else if (format == matrixFormatSparseCSC)
            return (bufferSize - sizeof(GPUSPARSE_INDEX_TYPE) * (numCols + 1)) / (sizeof(GPUSPARSE_INDEX_TYPE) + sizeof(ElemType));
        else if (format == matrixFormatSparseCSR)
            return (bufferSize - sizeof(GPUSPARSE_INDEX_TYPE) * (numRows + 1)) / (sizeof(GPUSPARSE_INDEX_TYPE) + sizeof(ElemType));
        else
            NOT_IMPLEMENTED;
    }

    // Since the m_sliceViewOffset affects Data and MajorIndexLocation differently than SecondaryIndexLocation, we compute it fully here.
    // BUGBUG: This function is semantically ill-defined. Does it return the buffer address, or the view that considers m_sliceViewOffset?
    //         TODO: Use different names for those two different purposes.
    GPUSPARSE_INDEX_TYPE* SecondaryIndexLocation() const // compressed index, col/row in CSC/CSR format, col2blockId/row2blockId in BlockCol/BlockRow format
    {
        if (GetFormat() == matrixFormatSparseBlockCol)
            return MajorIndexLocation() + GetNumCols();
        else if (GetFormat() == matrixFormatSparseBlockRow)
            return MajorIndexLocation() + GetNumRows();
        else // CSR or CSC
            return m_sliceViewOffset +
                   (GPUSPARSE_INDEX_TYPE*)((char*)Buffer() +
                                           (sizeof(ElemType) + sizeof(GPUSPARSE_INDEX_TYPE)) * GetSizeAllocated());
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
        return SecondaryIndexCount(GetNumRows(), GetNumCols(), GetSizeAllocated(), GetFormat());
    }

    // get size for compressed index
    size_t SecondaryIndexSize() const
    {
        return (SecondaryIndexCount()) * sizeof(GPUSPARSE_INDEX_TYPE);
    }

    size_t BufferSizeNeeded(const size_t numRows, const size_t numCols, const size_t numNZ, const MatrixFormat format) const // in bytes
    {
        return sizeof(ElemType) * numNZ + sizeof(GPUSPARSE_INDEX_TYPE) * (MajorIndexCount(numRows, numCols, numNZ, format) + SecondaryIndexCount(numRows, numCols, numNZ, format));
    }

    // Data() returns the address of the first non-zero element's value, after accounting for m_sliceViewOffset.
    // SecondaryIndexValueAt calls SecondaryIndexLocation which is already appropriately offset by m_sliceViewOffset
    inline ElemType* Data() const
    {
        return (Buffer() +
            ((GetFormat() == matrixFormatSparseCSC || GetFormat() == matrixFormatSparseCSR) ? SecondaryIndexValueAt(0)/*CSC or CSR*/ : 0/*others*/));
    }

    // helper function to discover problems with potentially incorrect use of Data()
    // Data() includes m_sliceViewOffset. This function is called where I reviewed the code and think that
    // the offset should not be included (Buffer() should be called instead). But I am not 100% sure.
    // In some cases, it does not matter because slice-view offset is always 0.
    // All those potentially incorrect Data() calls have been replaced with Data_IThinkThisShouldBeBuffer(),
    // which still calls Data() as before, but tests for m_sliceViewOffset==0 as a precondition.
    // Note also that this same issue may be present in CPUSparseMatrix. CPUSparseMatrix handles this a little different, so I am not touching that for now.
    inline ElemType* Data_IThinkThisShouldBeBuffer() const
    {
        if (GetFormat() != matrixFormatSparseCSC && GetFormat() != matrixFormatSparseCSR && m_sliceViewOffset != 0) // TODO: Is this actually true?
            LogicError("Data_IThinkThisShouldBeBuffer: m_sliceViewOffset cannot be used for sparse matrix types other than CSC and CSR.");
        if (m_sliceViewOffset != 0)
            LogicError("Data_IThinkThisShouldBeBuffer: I believe this is an incorrect use of Data().");
        return Data(); // this is the current behavior, which I think should be Buffer(). For m_sliceViewOffset == 0, they are the same.
    }

    inline size_t GetNumElemAllocated() const
    {
        return GetSizeAllocated();
    }

    inline size_t GetSizeElemAllocated() const
    {
        return sizeof(ElemType) * GetSizeAllocated();
    }

    // the column and row locations will swap based on what format we are in. Full index always follows the data array
    // TODO: Find a better name. CSxNzIndexArray()?
    //       Potential naming conventions:
    //        - buffer: points to the first stored element, but buffer may contain extra space for future growth
    //        - array: points to the first stored element, but semantically does not include extra space
    //          Is the distinction of array and buffer ever needed?
    //        - data: points to the first actual element  --TODO: get rid of this altogether if possible
    //       Note: A GPUSparseMatrix is always a view. The unsliced entity is the storage object. Maybe that should be reflected in the names?
    // BUGBUG: For CSR, this includes m_sliceViewOffset. It is not possible to abstract this way, while handling m_sliceViewOffset correclty.
    //         The solution is to make this less abstract and call it what it is. It should just be a wrapper around getting the index array for CSR or CSC.
    GPUSPARSE_INDEX_TYPE* RowLocation() const
    {
        // not a valid function for other formats
        assert(GetFormat() == matrixFormatSparseCSC || GetFormat() == matrixFormatSparseCSR);

        return (GetFormat() & matrixFormatRowMajor) ? SecondaryIndexLocation()/*CSR*/ : MajorIndexLocation()/*CSC*/;
    }

    // TODO: Match name and semantics to CSxNzIndexArray(). CSxNzIndexArrayByteSize()?
    //       Also avoid "size" as it could be #elements (as in STL) or size in bytes. Change to ByteSize().
    size_t RowSize() const // actual number of bytes in use
    {
        // not a valid function for other formats
        assert(GetFormat() == matrixFormatSparseCSC || GetFormat() == matrixFormatSparseCSR);

        return (GetFormat() & matrixFormatRowMajor) ? SecondaryIndexSize()/*CSR*/ : MajorIndexSize()/*CSC*/;
    }

    // BUGBUG: Don't abstract that far, cannot be correct w.r.t. m_sliceViewOffset.
    // TODO: Change name, e.g. CSxNzOffsetSlice()? Include "Slice" to indicate it includes m_sliceView offset
    GPUSPARSE_INDEX_TYPE* ColLocation() const
    {
        // not a valid function for other formats
        assert(GetFormat() == matrixFormatSparseCSC || GetFormat() == matrixFormatSparseCSR);

        return (GetFormat() & matrixFormatRowMajor) ? MajorIndexLocation()/*CSR*/ : SecondaryIndexLocation()/*CSC*/;
    }

    size_t ColSize() const // actual number of bytes in use
    {
        // not a valid function for other formats
        assert(GetFormat() == matrixFormatSparseCSC || GetFormat() == matrixFormatSparseCSR);

        return (GetFormat() & matrixFormatRowMajor) ? MajorIndexSize() : SecondaryIndexSize();
    }

    // TODO: rename that, e.g. CSxNzOffsetSliceAt()
    GPUSPARSE_INDEX_TYPE SecondaryIndexValueAt(size_t idx) const;

    GPUSPARSE_INDEX_TYPE* BlockId2ColOrRow() const
    {
        // not a valid function for other formats
        assert(GetFormat() == matrixFormatSparseBlockCol || GetFormat() == matrixFormatSparseBlockRow);
        return MajorIndexLocation();
    }

    GPUSPARSE_INDEX_TYPE* ColOrRow2BlockId() const
    {
        // not a valid function for other formats
        assert(GetFormat() == matrixFormatSparseBlockCol || GetFormat() == matrixFormatSparseBlockRow);
        return SecondaryIndexLocation();
    }

    //void SetValue(const CPUMatrix<ElemType>& denseMatrix);
    void SetValue(const GPUSparseMatrix<ElemType>& deepCopyFrom);
    void SetValue(const CPUSparseMatrix<ElemType>& deepCopyFrom);
    void SetValue(const GPUMatrix<ElemType>& denseMatrix, const MatrixFormat matrixFormat);
    void SetValue(const GPUMatrix<ElemType>& denseMatrix);

    void AdjustCol2BlockId(const GPUSPARSE_INDEX_TYPE* cpuCol2BlockId, size_t numBlocks, bool useBlockId2Col);

    GPUSPARSE_INDEX_TYPE* GetCondensedVector() const;
    void MaskColumnsValue(const GPUMatrix<char>& columnsMask, ElemType val, size_t numColsPerMaskEntry);

    void Reshape(const size_t numRows, const size_t numCols);
    void ResizeAsAndCopyIndexFrom(const GPUSparseMatrix<ElemType>& a, const bool growOnly = true);

    void Allocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const bool growOnly, bool keepExistingValues); // matrix format will affect the size to allocate
    void RequireSizeAndAllocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly, bool keepExistingValues);
    void RequireSizeAndAllocate(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve = 10000, const bool growOnly = true, bool keepExistingValues = false); // matrix format will affect the size to allocate
    void RequireSize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat format, const bool growOnly = true);
    void RequireSize(const size_t numRows, const size_t numCols, const MatrixFormat format, const bool growOnly = true)
    {
        return RequireSize(numRows, numCols, 0, format, growOnly);
    }
    void RequireSize(const size_t numRows, const size_t numCols, const bool growOnly = true);
    void Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, const MatrixFormat matrixFormat, const bool growOnly = true); // matrix format will affect the size to allocate
    void Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve = 10000, const bool growOnly = true);

    GPUSparseMatrix<ElemType> Transpose() const;
    void InplaceTranspose();
    GPUSparseMatrix<ElemType>& AssignTransposeOf(const GPUSparseMatrix<ElemType>& a);

    GPUSparseMatrix<ElemType> ColumnSlice(size_t startColumn, size_t numCols) const;
    GPUMatrix<ElemType> CopyColumnSliceToDense(size_t startColumn, size_t numCols) const;
    void AssignColumnSliceToDense(GPUMatrix<ElemType>& slice, size_t startColumn, size_t numCols) const;

    void GatherBatch(size_t numInputs, const std::function<const GPUSparseMatrix<ElemType>&(size_t)>& inputs);

    GPUMatrix<ElemType> DiagonalToDense() const;

    GPUMatrix<ElemType> CopyToDenseMatrix() const;
    size_t* TryCopyToArrayAsOneHot() const;
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

    // Sets sparse matrix in CSR format. this acts as deep copy
    void SetMatrixFromCSRFormat(const CPUSPARSE_INDEX_TYPE* h_CSRRow, const CPUSPARSE_INDEX_TYPE* h_Col, const ElemType* h_Val,
                                const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice = false, const DEVICEID_TYPE devId = -1);
    void SetMatrixFromCSCFormat(const CPUSPARSE_INDEX_TYPE* h_CSCCol, const CPUSPARSE_INDEX_TYPE* h_Row, const ElemType* h_Val,
        const size_t nz, const size_t numRows, const size_t numCols, const bool IsOnDevice = false, const DEVICEID_TYPE devId = -1, DataTransferer* transferer = nullptr);
    void SetMatrixFromSBCFormat(const size_t* blockIds, const ElemType* val, const size_t numBlocks, const size_t numRows, const size_t numCols);

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

    GPUSparseMatrix<ElemType>& AssignOneHot(const GPUMatrix<ElemType>& a, vector<size_t>& shape, size_t axis);
    static void AssignColumnwiseArgmaxTo(GPUMatrix<ElemType>& b, const GPUSparseMatrix<ElemType>& a);

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

    static void ColumnwiseScaleAndWeightedAdd(ElemType alpha, const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& v, ElemType beta, GPUMatrix<ElemType>& c);

    static void ScaleAndAdd(const ElemType alpha, const GPUSparseMatrix<ElemType>& lhs, GPUMatrix<ElemType>& c);
    static void ConvolveAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& lhs, const bool transposeA, const GPUSparseMatrix<ElemType>& rhs,
                                       const bool transposeB, ElemType beta, GPUMatrix<ElemType>& c, size_t numChannels, size_t horizontalSubsample, bool padding, bool channelwise);
    static void TensorShuffleScaleAndAdd(ElemType keepWeight, const GPUSparseMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const GPUSparseMatrix<ElemType>& b, GPUSparseMatrix<ElemType>& c);

    void NormalGrad(GPUMatrix<ElemType>& c, const ElemType momentum, ElemType unitGainFactor);
    ElemType Adagrad(GPUMatrix<ElemType>& c, const bool needAveMultiplier);
    void FSAdagrad(GPUMatrix<ElemType>& c, GPUMatrix<ElemType>& functionValues, ElemType learnRatePerSample, ElemType momentum, ElemType adaWeight, ElemType adaMul, ElemType unitGainFactor);
    ElemType RmsProp(GPUMatrix<ElemType>& c, ElemType RMS_GAMMA, ElemType RMS_WGT_INC, ElemType RMS_WGT_MAX, ElemType RMS_WGT_DEC, ElemType RMS_WGT_MIN, const bool needAveMultiplier, const bool initialized);
    void Adam(GPUMatrix<ElemType>& c, GPUMatrix<ElemType>& functionValues, ElemType learnRatePerSample, ElemType momentum, ElemType adaWeight, ElemType adaMul, ElemType epsilon, ElemType unitGainFactor, bool adamax);
    void AdaDelta(GPUMatrix<ElemType>&c, GPUMatrix<ElemType>&functionValues, ElemType learningRate, ElemType rho, ElemType epsilon);

    static void Multiply(const GPUSparseMatrix<ElemType>& S, const GPUMatrix<ElemType>& D, GPUMatrix<ElemType>& C);
    static void Multiply(const GPUMatrix<ElemType>& D, const GPUSparseMatrix<ElemType>& S, GPUMatrix<ElemType>& C);
    static void Multiply(const GPUSparseMatrix<ElemType>& S1, bool transposeS1, const GPUSparseMatrix<ElemType>& S2, bool transposeS2, GPUSparseMatrix<ElemType>& C);
    GPUSparseMatrix<ElemType>& AssignProductOf(const GPUSparseMatrix<ElemType>& a, const bool transposeA, const GPUSparseMatrix<ElemType>& b, const bool transposeB);

    static ElemType InnerProductOfMatrices(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b);
    static ElemType InnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUSparseMatrix<ElemType>& b);
    static void InnerProduct(const GPUSparseMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const bool isColWise);
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
    static void ConvertBuffer(OutType* outBuffer, const InType* inBuffer, const size_t size);

private:
    void ZeroInit(const MatrixFormat matrixFormat, const DEVICEID_TYPE deviceId);

private:
    void performElementWiseFunction(const ElementWiseOperator kind, const GPUSparseMatrix<ElemType>& src);
    void DeepCopy(const GPUSparseMatrix<ElemType>& deepCopyFrom);
    void PrepareBuffer(const size_t numRows, const size_t numCols, const bool canReuseBuffer, std::function<size_t(GPUSPARSE_INDEX_TYPE* csrRowPtrC)> func);

    size_t ElemCountFromBufferSize(const size_t numRows, const size_t numCols, const MatrixFormat format, const size_t totalBufferSize) const;
    size_t ElemCountFromBufferSize() const;
    DEVICEID_TYPE PrepareDevice(const DEVICEID_TYPE deviceId = -1) const;
    size_t IdentifyRowsWithValues() const;
};

}}}
