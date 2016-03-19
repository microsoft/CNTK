//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#ifdef _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#else // no DLLs on Linux
#define MATH_API
#endif

#include "Basics.h"
#include <string>
#include <stdint.h>
#include <memory>

#pragma warning( disable: 4251 )
typedef unsigned char byte;

#define DEVICEID_TYPE int
// and the following magic values
#define CPUDEVICE (DEVICEID_TYPE) - 1                 // device is the CPU
#define DEVICEID_NOTYETDETERMINED (DEVICEID_TYPE) - 3 // not yet set
#define DEVICEID_AUTO (DEVICEID_TYPE) - 4             // device should be picked automatically

#define EPS_IN_INVERSE 1e-30f    // 1e-37 is the only guaranteed precision
#define EPS_IN_LOG 1e-37f        // 1e-37 is the only guaranteed precision
#define LOG_OF_EPS_IN_LOG -85.1f // log(EPS_IN_LOG)
#define LOG10_OF_EPS_IN_LOG -37  // log_10(EPS_IN_LOG)
#define LZERO -10e10
#define MINLOGEXP -9.2103
#define LSMALL -0.5E10

#define GPUSPARSE_INDEX_TYPE int // cuSparse only supports int array indexes
#define CPUSPARSE_INDEX_TYPE int // to be consistent with cuSparse but limited the possible size of the matrix.

namespace Microsoft { namespace MSR { namespace CNTK {

class MATH_API TracingGPUMemoryAllocator
{
private:
    static int m_traceLevel;

public:
    static void SetTraceLevel(int traceLevel);
    static bool IsTraceEnabled();

    template <typename AllocatedElemType>
    static AllocatedElemType* Allocate(int deviceId, size_t numRows, size_t numCols);

    template <typename AllocatedElemType>
    static AllocatedElemType* Allocate(int deviceId, size_t numElements);

    template <typename AllocatedElemType>
    static void Free(int deviceId, AllocatedElemType* bufferPtr, bool ignoreCUDARetCode = false);

private:
    template <typename AllocatedElemType>
    static AllocatedElemType* AllocateNoTrace(int deviceId, size_t numElements);

    static std::pair<size_t, size_t> GetFreeAndTotalMemoryInMBs(int deviceId);
};

// -----------------------------------------------------------------------
// ElementWiseOperator -- This enum represents which function to apply.
// This is shared between all matrix types and tensors.
// -----------------------------------------------------------------------

enum ElementWiseOperator
{
    // nullary
    opConstOne,
    // unary (or binary with constant parameter)
    opCopy,
    opNegate,
    opNot,
    opAbs,
    opSigmoid,
    opTanh,
    opSqr,
    opSqrt,
    opExp,
    opLog,
    opLinearRectifier,
    opCosine,
    // unary ops for use by Matrix class only (there is no TensorView implementation)
    opSigmoidDerivative,
    opLinearRectifierDerivative,
    opNegativeSine,
    // binary
    opSum,
    opDifference,
    opElementwiseProduct,
    opElementwiseQuotient,
    opLogSum,
    opMax,
    opMin,
    opEQ,
    opNE,
    opGT,
    opLT,
    opGE,
    opLE,
    opAnd,
    opOr,
    opXor,
    opMaskNegative,
    opElementwiseProductWithSigmoidDerivativeFromOutput,
    opElementwiseProductWithTanhDerivativeFromOutput,
    opElementwiseProductWithLinearRectifierDerivativeFromOutput,
    opElementwiseProductWithLogDerivativeFromOutput,
    opElementwiseProductWithCosDerivative,
    opElementwiseProductWithAbsDerivative,
    opSqrOfDifference,
    // binary ops for indexing
    // opIndex,
    // ternary
    opCond /*a ? b : c*/,
    opClip /*clip a within interval b..c*/
    // Note: not all that's implemented in CNTK ComputationNodes has an opcode yet.
};

// helper to apply a C macro for all operations of each kind
#define ForAllNullaryOps(Macro) \
    Macro(ConstOne);

#define ForAllUnaryOps(Macro) \
    Macro(Copy);              \
    Macro(Negate);            \
    Macro(Not);               \
    Macro(Abs);               \
    Macro(Sigmoid);           \
    Macro(Tanh);              \
    Macro(Sqr);               \
    Macro(Sqrt);              \
    Macro(Exp);               \
    Macro(Log);               \
    Macro(LinearRectifier);   \
    Macro(Cosine);

#define ForAllBinaryOps(Macro)                                        \
    Macro(Sum);                                                       \
    Macro(Difference);                                                \
    Macro(ElementwiseProduct);                                        \
    Macro(ElementwiseQuotient);                                       \
    Macro(LogSum);                                                    \
    Macro(Max);                                                       \
    Macro(Min);                                                       \
    Macro(EQ);                                                        \
    Macro(NE);                                                        \
    Macro(GT);                                                        \
    Macro(LT);                                                        \
    Macro(GE);                                                        \
    Macro(LE);                                                        \
    Macro(And);                                                       \
    Macro(Or);                                                        \
    Macro(Xor);                                                       \
    Macro(MaskNegative);                                              \
    Macro(ElementwiseProductWithSigmoidDerivativeFromOutput);         \
    Macro(ElementwiseProductWithTanhDerivativeFromOutput);            \
    Macro(ElementwiseProductWithLinearRectifierDerivativeFromOutput); \
    Macro(ElementwiseProductWithLogDerivativeFromOutput);             \
    Macro(ElementwiseProductWithCosDerivative);                       \
    Macro(ElementwiseProductWithAbsDerivative);                       \
    Macro(SqrOfDifference);                                           \
    //Macro(Index);

#define ForAllTernaryOps(Macro) \
    Macro(Cond);                \
    Macro(Clip);

// -----------------------------------------------------------------------
// various enums to describe
// -----------------------------------------------------------------------

enum MatrixFlagBitPosition
{
    // TODO: remove all formats that are actually not supported
    bitPosRowMajor = 0,         // row major matrix
    bitPosSparse = 1,           // sparse matrix (COO if uncompressed)
    bitPosCompressed = 2,       // a compressed sparse format (CSC/CSR)
    bitPosDontOwnBuffer = 3,    // buffer is not owned by this matrix
    bitPosSetValueOnDevice = 4, // in a setValue situation, the copy from buffer is already on the device
};

enum MatrixFormat
{
    // TODO: remove all formats that are actually not supported
    matrixFormatDense = 0,                          // default is dense
    matrixFormatColMajor = 0,                       // default is column major
    matrixFormatRowMajor = 1 << bitPosRowMajor,     // row major matrix
    matrixFormatSparse = 1 << bitPosSparse,         // sparse matrix
    matrixFormatCompressed = 1 << bitPosCompressed, // a compressed sparse format (CSC/CSR/COO)
    matrixFormatDenseColMajor = matrixFormatDense + matrixFormatColMajor,
    matrixFormatDenseRowMajor = matrixFormatDense + matrixFormatRowMajor,
    matrixFormatSparseCSC = matrixFormatSparse + matrixFormatColMajor + matrixFormatCompressed,
    matrixFormatSparseCSR = matrixFormatSparse + matrixFormatRowMajor + matrixFormatCompressed,
    matrixFormatSparseOther = matrixFormatSparse + matrixFormatRowMajor,                   // currently used for CPU sparse format, will change to CSC/CSR eventually
    matrixFormatMask = matrixFormatRowMajor + matrixFormatSparse + matrixFormatCompressed, // mask that covers all the
    matrixFormatSparseBlockCol,                                                            // col block based sparse matrix
    matrixFormatSparseBlockRow,                                                            // row block based sparse matrix
};

// common matrix flags for use on all matrices
enum MatrixFlags
{
    // first bits of matrix flags are MatrixFormat
    matrixFlagNormal = 0,
    matrixFlagDontOwnBuffer = 1 << bitPosDontOwnBuffer,       // the matrix memory pointers are externally managed, don't allocate/free or attempt to copy to another location
    matrixFlagSetValueOnDevice = 1 << bitPosSetValueOnDevice, // SetValue() call has a buffer that is already on the device
};

// -----------------------------------------------------------------------
// BaseMatrixStorage -- base class for all matrix types (CPU, GPU) x (dense, sparse)
// -----------------------------------------------------------------------

template <class ElemType>
class BaseMatrixStorage : public enable_shared_from_this<BaseMatrixStorage<ElemType>>
{
public:

    BaseMatrixStorage() 
    {
        ZeroInit(matrixFormatDense, CPUDEVICE);
    }

    BaseMatrixStorage(MatrixFormat format, DEVICEID_TYPE computeDevice)
    {
        ZeroInit(format, computeDevice);
    }

    ~BaseMatrixStorage()
    {
        ReleaseMemory();
        m_numRows = 0;
        m_numCols = 0;
    }

    void ReleaseMemory()
    {
        if (m_computeDevice < 0)
        {
            delete[] m_pArray;
            m_pArray = nullptr;
            m_nzValues = nullptr;

            delete[] m_unCompIndex;
            m_unCompIndex = nullptr;

            delete[] m_compIndex;
            m_compIndex = nullptr;

            delete[] m_blockIds;
            m_blockIds = nullptr;
        }
        else
        {
#ifndef CPUONLY
            if (m_pArray != nullptr)
                TracingGPUMemoryAllocator::Free<ElemType>(m_computeDevice, m_pArray, true);
            m_pArray = nullptr;

            if (m_rowToId != nullptr)
				TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(m_computeDevice, m_rowToId, true);
            m_rowToId = nullptr;
#endif

            delete[](byte*) m_tempHostBuffer;
            m_tempHostBuffer = nullptr;
        }
        m_elemSizeAllocated = 0;
        m_totalBufferSizeAllocated = 0;
    }

    //MatrixFormat m_format;
    MatrixFormat GetFormat() const { return m_format; }
    void SetFormat(MatrixFormat format) { m_format = format; }

    //mutable DEVICEID_TYPE m_computeDevice; // current GPU device Id or CPUDEVICE
    virtual DEVICEID_TYPE GetComputeDeviceId() const { return m_computeDevice; }
    void SetComputeDeviceId(const DEVICEID_TYPE computeId) const { m_computeDevice = computeId; }

    //size_t m_numRows;
    size_t GetNumStorageRows() const { return m_numRows; }
    void SetNumStorageRows(size_t rows) { m_numRows = rows; }

    //size_t m_numCols;
    size_t GetNumStorageCols() const { return m_numCols; }
    void SetNumStorageCols(size_t cols) { m_numCols = cols; }

    //size_t m_elemSizeAllocated;
    size_t GetSizeAllocated() const { return m_elemSizeAllocated; }
    void SetSizeAllocated(size_t alloc) { m_elemSizeAllocated = alloc; }

    size_t GetNumStorageElements() const { return m_numRows * m_numCols; }
    bool IsEmpty() const { return m_numRows == 0 || m_numCols == 0; }

    //ElemType* m_pArray;
    ElemType* GetArray() { return m_pArray; }
    ElemType* GetArray() const { return m_pArray; }
    void SetArray(ElemType* parray) { m_pArray = parray; }

    //size_t m_totalBufferSizeAllocated;
    size_t BufferSizeAllocated() const { return m_totalBufferSizeAllocated; }
    void SetBufferSizeAllocated(size_t alloc) { m_totalBufferSizeAllocated = alloc; }
    
    //size_t m_blockSize;                      // block size
    size_t GetBlockSize() const { return m_blockSize; }
    void SetBlockSize(size_t blockSize) { m_blockSize = blockSize; }

    //mutable GPUSPARSE_INDEX_TYPE* m_rowToId; // the id showing the order row number is observed in the nnz values.
    GPUSPARSE_INDEX_TYPE* GetRowToId() { return m_rowToId; }
    const GPUSPARSE_INDEX_TYPE* GetRowToId() const { return m_rowToId; }
    void SetRowToId(GPUSPARSE_INDEX_TYPE* parray) { m_rowToId = parray; }

    //mutable void* m_tempHostBuffer; // used to copy values.
    void* GetTempHostBuffer() { return m_tempHostBuffer; }
    void* GetTempHostBuffer() const { return m_tempHostBuffer; }
    void SetTempHostBuffer(void* buffer) { m_tempHostBuffer = buffer; }
    void SetTempHostBuffer(void* buffer) const { m_tempHostBuffer = buffer; }

    //mutable size_t m_tempHostBufferSize;
    size_t GetTempHostBufferSize() { return m_tempHostBufferSize; }
    size_t GetTempHostBufferSize() const { return m_tempHostBufferSize; }
    void SetTempHostBufferSize(size_t bufferSize) { m_tempHostBufferSize = bufferSize; }
    void SetTempHostBufferSize(size_t bufferSize) const { m_tempHostBufferSize = bufferSize; }

    //int m_colIdx; // used to SetValue()
    int GetColIdx() const { return m_colIdx; }
    void SetColIdx(int idx) { m_colIdx = idx; }

    //size_t m_compIndexSize;
    size_t GetCompIndexSize() const { return m_compIndexSize; }
    void SetCompIndexSize(size_t indexSize) { m_compIndexSize = indexSize; }

    //ElemType* m_nzValues;
    ElemType* GetNzValues() { return m_nzValues; }
    void SetNzValues(ElemType* values) { m_nzValues = values; }

    //size_t* m_blockIds;    // block ids
    size_t* GetBlockIds() { return m_blockIds; }
    const size_t* GetBlockIds() const { return m_blockIds; }
    void SetBlockIds(size_t* blockIds) { m_blockIds = blockIds; }

    //size_t m_blockIdShift; // used to get efficient slice, actual col = blockIds[j] - m_blockIdShift
    size_t GetBlockIdShift() { return m_blockIdShift; }
    size_t GetBlockIdShift() const { return m_blockIdShift; }
    void SetBlockIdShift(size_t blockIdShift) { m_blockIdShift = blockIdShift; }

    //CPUSPARSE_INDEX_TYPE* m_unCompIndex; // row/col ids in CSC/CSR format
    CPUSPARSE_INDEX_TYPE* GetUnCompIndex() { return m_unCompIndex; }
    CPUSPARSE_INDEX_TYPE* GetUnCompIndex() const { return m_unCompIndex; }
    void SetUnCompIndex(CPUSPARSE_INDEX_TYPE* parray) { m_unCompIndex = parray; }
    
    //CPUSPARSE_INDEX_TYPE* m_compIndex;   // begin ids of col/row in CSC/CSR format
    CPUSPARSE_INDEX_TYPE* GetCompIndex() { return m_compIndex; }
    CPUSPARSE_INDEX_TYPE* GetCompIndex() const { return m_compIndex; }
    void SetCompIndex(CPUSPARSE_INDEX_TYPE* parray) { m_compIndex = parray; }

	void ZeroInit(const MatrixFormat matrixFormat = matrixFormatDense, const DEVICEID_TYPE computeDevice = -1)
    {
        m_format                   = matrixFormat;
        m_computeDevice            = computeDevice;
		m_numRows                  = 0;
		m_numCols                  = 0;
		m_elemSizeAllocated        = 0;
		m_pArray                   = nullptr;
		m_totalBufferSizeAllocated = 0;
		m_blockSize                = 0;                      // block size
		m_rowToId                  = nullptr; // the id showing the order row number is observed in the nnz values.
		m_tempHostBuffer           = nullptr; // used to copy values.
		m_tempHostBufferSize       = 0;
		m_colIdx                   = 0; // used to SetValue()
		m_compIndexSize            = 0;
		m_nzValues                 = nullptr;
		m_unCompIndex              = nullptr; // row/col ids in CSC/CSR format
		m_compIndex                = nullptr;   // begin ids of col/row in CSC/CSR format
		m_blockIds                 = nullptr;    // block ids
		m_blockIdShift             = 0; // used to get efficient slice, actual col = blockIds[j] - m_blockIdShift
    }

protected:


protected:
    MatrixFormat m_format;
    mutable DEVICEID_TYPE m_computeDevice; // current GPU device Id or CPUDEVICE

    size_t m_numRows;
    size_t m_numCols;
    size_t m_elemSizeAllocated;
    ElemType* m_pArray;

	// GPU copied
    size_t m_totalBufferSizeAllocated;

    // used by the blockCol and blockRow format
    size_t m_blockSize;                      // block size
    mutable GPUSPARSE_INDEX_TYPE* m_rowToId; // the id showing the order row number is observed in the nnz values.

    mutable void* m_tempHostBuffer; // used to copy values.
    mutable size_t m_tempHostBufferSize;

    //GPUSparseMatrix* m_sliceOf; // if this is a slice, then this points to the owning matrix object that we sliced from

	// CPU copied
    int m_colIdx; // used to SetValue()
    size_t m_compIndexSize;
    ElemType* m_nzValues;

    // non-zero values are stored in m_pArray
    CPUSPARSE_INDEX_TYPE* m_unCompIndex; // row/col ids in CSC/CSR format
    CPUSPARSE_INDEX_TYPE* m_compIndex;   // begin ids of col/row in CSC/CSR format

    //size_t m_blockSize;    // block size
    size_t* m_blockIds;    // block ids
    size_t m_blockIdShift; // used to get efficient slice, actual col = blockIds[j] - m_blockIdShift

    //CPUSparseMatrix* m_sliceOf; // if this is a slice, then this points to the owning matrix object that we sliced from
};

// -----------------------------------------------------------------------
// BaseMatrix -- base class for all matrix types (CPU, GPU) x (dense, sparse)
// -----------------------------------------------------------------------

template <class ElemType>
class MATH_API BaseMatrix
{
public:
    //MatrixFormat m_format;
    MatrixFormat GetFormat() const { return m_sob->GetFormat(); }
    void SetFormat(MatrixFormat format) { m_sob->SetFormat(format); }

    //mutable DEVICEID_TYPE m_computeDevice; // current GPU device Id or CPUDEVICE
    virtual DEVICEID_TYPE GetComputeDeviceId() const { return m_sob->GetComputeDeviceId(); }
    void SetComputeDeviceId(const DEVICEID_TYPE computeId) const { m_sob->SetComputeDeviceId(computeId); }

    //size_t m_numRows;
    size_t GetNumStorageRows() const { return m_sob->GetNumStorageRows(); }
    void SetNumStorageRows(size_t rows) { m_sob->SetNumStorageRows(rows); }

    //size_t m_numCols;
    size_t GetNumStorageCols() const { return m_sob->GetNumStorageCols(); }
    void SetNumStorageCols(size_t cols) { m_sob->SetNumStorageCols(cols); }

    //size_t m_elemSizeAllocated;
    size_t GetSizeAllocated() const { return m_sob->GetSizeAllocated(); }
    void SetSizeAllocated(size_t alloc) { m_sob->SetSizeAllocated(alloc); }

    //ElemType* m_pArray;
    ElemType* GetArray() { return m_sob->GetArray(); }
    ElemType* GetArray() const { return m_sob->GetArray(); }
    void SetArray(ElemType* parray) { m_sob->SetArray(parray); }

    //size_t m_totalBufferSizeAllocated;
    size_t BufferSizeAllocated() const { return m_sob->BufferSizeAllocated(); }
    void SetBufferSizeAllocated(size_t alloc) { m_sob->SetBufferSizeAllocated(alloc); }
    
    //size_t m_blockSize;                      // block size
    size_t GetBlockSize() const { return m_sob->GetBlockSize(); }
    void SetBlockSize(size_t blockSize) { m_sob->SetBlockSize(blockSize); }

    //mutable GPUSPARSE_INDEX_TYPE* m_rowToId; // the id showing the order row number is observed in the nnz values.
    GPUSPARSE_INDEX_TYPE* GetRowToId() { return m_sob->GetRowToId(); }
    const GPUSPARSE_INDEX_TYPE* GetRowToId() const { return m_sob->GetRowToId(); }
    void SetRowToId(GPUSPARSE_INDEX_TYPE* parray) { m_sob->SetRowToId(parray); }

    //mutable void* m_tempHostBuffer; // used to copy values.
    void* GetTempHostBuffer() { return m_sob->GetTempHostBuffer(); }
    void* GetTempHostBuffer() const { return m_sob->GetTempHostBuffer(); }
    void SetTempHostBuffer(void* buffer) { m_sob->SetTempHostBuffer(buffer); }
    void SetTempHostBuffer(void* buffer) const { m_sob->SetTempHostBuffer(buffer); }

    //mutable size_t m_tempHostBufferSize;
    size_t GetTempHostBufferSize() { return m_sob->GetTempHostBufferSize(); }
    size_t GetTempHostBufferSize() const { return m_sob->GetTempHostBufferSize(); }
    void SetTempHostBufferSize(size_t bufferSize) { m_sob->SetTempHostBufferSize(bufferSize); }
    void SetTempHostBufferSize(size_t bufferSize) const { m_sob->SetTempHostBufferSize(bufferSize); }

    //int m_colIdx; // used to SetValue()
    int GetColIdx() const { return m_sob->GetColIdx(); }
    void SetColIdx(int idx) { m_sob->SetColIdx(idx); }

    //size_t m_compIndexSize;
    size_t GetCompIndexSize() const { return m_sob->GetCompIndexSize(); }
    void SetCompIndexSize(size_t indexSize) { m_sob->SetCompIndexSize(indexSize); }

    //ElemType* m_nzValues;
    ElemType* GetNzValues() { return m_sob->GetNzValues(); }
    void SetNzValues(ElemType* values) { m_sob->SetNzValues(values); }

    //size_t* m_blockIds;    // block ids
    size_t* GetBlockIds() { return m_sob->GetBlockIds(); }
    const size_t* GetBlockIds() const { return m_sob->GetBlockIds(); }
    void SetBlockIds(size_t* blockIds) const { m_sob->SetBlockIds(blockIds); }

    //size_t m_blockIdShift; // used to get efficient slice, actual col = blockIds[j] - m_blockIdShift
    size_t GetBlockIdShift() { return m_sob->GetBlockIdShift(); }
    size_t GetBlockIdShift() const { return m_sob->GetBlockIdShift(); }
    void SetBlockIdShift(size_t blockIdShift) { m_sob->SetBlockIdShift(blockIdShift); }

    //CPUSPARSE_INDEX_TYPE* m_unCompIndex; // row/col ids in CSC/CSR format
    CPUSPARSE_INDEX_TYPE* GetUnCompIndex() { return m_sob->GetUnCompIndex(); }
    CPUSPARSE_INDEX_TYPE* GetUnCompIndex() const { return m_sob->GetUnCompIndex(); }
    void SetUnCompIndex(CPUSPARSE_INDEX_TYPE* parray) { m_sob->SetUnCompIndex(parray); }
    
    //CPUSPARSE_INDEX_TYPE* m_compIndex;   // begin ids of col/row in CSC/CSR format
    CPUSPARSE_INDEX_TYPE* GetCompIndex() { return m_sob->GetCompIndex(); }
    CPUSPARSE_INDEX_TYPE* GetCompIndex() const { return m_sob->GetCompIndex(); }
    void SetCompIndex(CPUSPARSE_INDEX_TYPE* parray) { m_sob->SetCompIndex(parray); }

    size_t GetNumRows() const { return m_numRows; }
    size_t GetNumCols() const { return m_numCols; }

    size_t GetNumElements() const { return m_numRows * m_numCols; }
    bool IsEmpty() const { return m_numRows == 0 || m_numCols == 0; }

    bool OwnBuffer() const { return !m_externalBuffer; }
    void SetOwnBuffer(bool own) { m_externalBuffer = !own; }

    size_t NzCount() const { return m_nz; }
    void SetNzCount(const size_t nz) { m_nz = nz; }

    bool VerifyResizable() { return (m_sob->GetNumStorageRows() == m_numRows && m_sob->GetNumStorageCols() == m_numCols); }
    bool VerifyWritable() { return (m_sob->GetNumStorageRows() == m_numRows && m_sob->GetNumStorageCols() == m_numCols); }

    void VerifySize(size_t rows, size_t cols)
    {
        if (rows != GetNumRows() || cols != GetNumCols())
            LogicError("VerifySize: expected matrix size %lu x %lu, but it is %lu x %lu",
                       rows, cols, GetNumRows(), GetNumCols());
    }

    BaseMatrix()
    {
        ZeroInit();
    }
    ~BaseMatrix()
    {
        ZeroValues();
    }


    void ZeroInit()
    {
        MatrixFormat defFmt = matrixFormatDense;
        DEVICEID_TYPE compDev = -1;
        if (m_sob != nullptr)
        {
            defFmt = m_sob->GetFormat();
            compDev = m_sob->GetComputeDeviceId();

        }
        ZeroInit(defFmt, compDev);
    }

    void ZeroValues()
    {
        m_numRows           = 0;
        m_numCols           = 0;
        m_sliceViewOffset   = 0;
        m_externalBuffer    = false;
        m_nz                = 0;
        m_sob               = nullptr;
    }
	void ZeroInit(const MatrixFormat matrixFormat, const DEVICEID_TYPE computeDevice )
    {
        ZeroValues();
		m_sob = make_shared<BaseMatrixStorage<ElemType>>(matrixFormat, computeDevice);
    }

protected:
    void Clear() {}

    void ZeroStorageInit() { m_sob->ZeroInit(); }
    void ReleaseStorageMemory() { m_sob->ReleaseMemory(); }

    // copy all metadata (but not content taht pArray points to)
    void ShallowCopyFrom(const BaseMatrix& other) 
    {
        *this = other;
    }

protected:

    size_t m_numRows;
    size_t m_numCols;
    //size_t m_elemSizeAllocated;
    size_t m_sliceViewOffset; // this is used to get a column slice view of a matrix in the Sparse CSC format  --TODO: move to sparse matrix implementations? Or common sparse base class?
    bool m_externalBuffer; // is the buffer used by this matrix,
    //ElemType* m_pArray;
    size_t m_nz;                           // Number of non-zero elements for sparse matrices (unused in other formats)

    shared_ptr<BaseMatrixStorage<ElemType>> m_sob;
};

}}}
