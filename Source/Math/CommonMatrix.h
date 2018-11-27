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
#include "basetypes.h"
#include <string>
#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <map>

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

// TODO: merge these two types
#define GPUSPARSE_INDEX_TYPE int // cuSparse only supports int array indexes
#define CPUSPARSE_INDEX_TYPE int // to be consistent with cuSparse but limited the possible size of the matrix.

// special markers in BlockId2ColOrRow()/ColOrRow2BlockId()
static const GPUSPARSE_INDEX_TYPE SparseIndex_NotAssigned = -1; // the index is not used, for col2BlockId it means the column has no corresponding block
static const GPUSPARSE_INDEX_TYPE SparseIndex_Pending = -2; // the index assignment is pending, a transitional state when counting new blocks when sparse += sparse * dense 

namespace Microsoft { namespace MSR { namespace CNTK {

MATH_API void SetMathLibTraceLevel(int traceLevel);
MATH_API int GetMathLibTraceLevel();

inline bool IsGpu(DEVICEID_TYPE deviceId)
{
    return deviceId > CPUDEVICE;
}

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

    // Let it be public method, the memory manager could check the totoal free memory and decide whether to physically
    // release all the cached memory.
    static std::pair<size_t, size_t> GetFreeAndTotalMemoryInMBs(int deviceId);

private:
    template <typename AllocatedElemType>
    static AllocatedElemType* AllocateNoTrace(int deviceId, size_t numElements);
};

// -----------------------------------------------------------------------
// ElementWiseOperator -- This enum represents which function to apply.
// This is shared between all matrix types and tensors.
// -----------------------------------------------------------------------

enum ElementWiseOperator
{
    // nullary
    opConstOne, opNone,
    // unary (or binary with constant parameter)
    opCopy,
    opNegate, opNot, opAbs, opFloor, opReciprocal,
    opSigmoid, opTanh, opAtanh, opSqr, opSqrt, opExp, opLog, opLinearRectifier,
    opCosine, opSin, opTan, opAcos, opAsin, opAtan, opCosh, opSinh, opAsinh, opExponentialLinearUnit, opStableSigmoid, opStraightThrough,
    // unary ops for use by Matrix class only (there is no TensorView implementation)
    opSigmoidDerivative, opLinearRectifierDerivative, opNegativeSine, opExponentialLinearUnitDerivative, opStableSigmoidDerivative, opStraightThroughDerivative,
    // binary
    opCopyIf, opCopyIfNot, opSum, opDifference, opElementwiseProduct, opElementwiseQuotient, opLogSum, opPow,
    opMax, opMin, opArgmax, opArgmin,
    opLess, opEqual, opGreater, opGreaterEqual, opNotEqual, opLessEqual, // Note: must obey this order: (sgn(a-b) == -1, 0, +1), (sgn(a-b) != -1, 0, +1)
    opAnd, opOr, opXor, opMaskNegative,
    opElementwiseProductWithSigmoidDerivativeFromOutput, opElementwiseProductWithTanhDerivativeFromOutput,
    opElementwiseProductWithLinearRectifierDerivativeFromOutput, opElementwiseProductWithLogDerivativeFromOutput,
    opElementwiseProductWithCosDerivative, opElementwiseProductWithSinDerivative, opElementwiseProductWithTanDerivative,
    opElementwiseProductWithAcosDerivative, opElementwiseProductWithAsinDerivative, opElementwiseProductWithAtanDerivative,
    opElementwiseProductWithCoshDerivative, opElementwiseProductWithSinhDerivative,
    opElementwiseProductWithAtanhDerivative, opElementwiseProductWithAsinhDerivative,
    opElementwiseProductWithAbsDerivative, opElementwiseProductWithSqrtDerivative,
    opElementwiseProductWithReciprocalDerivative, opSqrOfDifference,
    opElementwiseProductWithExponentialLinearUnitDerivativeFromOutput,
    opElementwiseProductWithStraightThroughDerivative,
    // binary ops for indexing
    // opIndex,
    // ternary
    opCond /*a ? b : c*/,
    opClip, /*clip a within interval b..c*/
    opElementwiseProductWithLogSumDerivative,
    opCopyIfEqual,
    opElementwiseProductWithExpOfDiff, /* a * exp(b - c) */
    opElementwiseProductWithQuotient, /* a * (b / c) */
    opElementwiseProductWithPowExponentDerivative, /* a * b * log(c) */
    opElementwiseProductWithPowBaseDerivative,  /* a * c * pow(b, c-1) */
    // Note: not all that's implemented in CNTK ComputationNodes has an opcode yet.
};

// helper to apply a C macro for all operations of each kind
#define ForAllNullaryOps(Macro) \
    Macro(ConstOne);

#define ForAllUnaryOps(Macro)     \
    Macro(Copy);                  \
    Macro(Negate);                \
    Macro(Not);                   \
    Macro(Abs);                   \
    Macro(Floor);                 \
    Macro(Reciprocal);            \
    Macro(Sigmoid);               \
    Macro(Tanh);                  \
    Macro(Atanh);                 \
    Macro(Sqr);                   \
    Macro(Sqrt);                  \
    Macro(Exp);                   \
    Macro(Log);                   \
    Macro(LinearRectifier);       \
    Macro(Cosine);                \
    Macro(Sin);                   \
    Macro(Tan);                   \
    Macro(Acos);                  \
    Macro(Asin);                  \
    Macro(Atan);                  \
    Macro(Cosh);                  \
    Macro(Sinh);                  \
    Macro(Asinh);                 \
    Macro(ExponentialLinearUnit); \
    Macro(StableSigmoid);         \
    Macro(StraightThrough);

#define ForAllBinaryOps(Macro)                                               \
    Macro(CopyIf);                                                           \
    Macro(CopyIfNot);                                                        \
    Macro(Sum);                                                              \
    Macro(Difference);                                                       \
    Macro(ElementwiseProduct);                                               \
    Macro(ElementwiseQuotient);                                              \
    Macro(LogSum);                                                           \
    Macro(Pow);                                                              \
    Macro(Max);                                                              \
    Macro(Min);                                                              \
    Macro(Equal);                                                            \
    Macro(NotEqual);                                                         \
    Macro(Greater);                                                          \
    Macro(Less);                                                             \
    Macro(GreaterEqual);                                                     \
    Macro(LessEqual);                                                        \
    Macro(And);                                                              \
    Macro(Or);                                                               \
    Macro(Xor);                                                              \
    Macro(MaskNegative);                                                     \
    Macro(ElementwiseProductWithSigmoidDerivativeFromOutput);                \
    Macro(ElementwiseProductWithTanhDerivativeFromOutput);                   \
    Macro(ElementwiseProductWithAtanhDerivative);                            \
    Macro(ElementwiseProductWithLinearRectifierDerivativeFromOutput);        \
    Macro(ElementwiseProductWithLogDerivativeFromOutput);                    \
    Macro(ElementwiseProductWithCosDerivative);                              \
    Macro(ElementwiseProductWithSinDerivative);                              \
    Macro(ElementwiseProductWithTanDerivative);                              \
    Macro(ElementwiseProductWithAcosDerivative);                             \
    Macro(ElementwiseProductWithAsinDerivative);                             \
    Macro(ElementwiseProductWithAtanDerivative);                             \
    Macro(ElementwiseProductWithCoshDerivative);                             \
    Macro(ElementwiseProductWithSinhDerivative);                             \
    Macro(ElementwiseProductWithAsinhDerivative);                            \
    Macro(ElementwiseProductWithAbsDerivative);                              \
    Macro(ElementwiseProductWithReciprocalDerivative);                       \
    Macro(ElementwiseProductWithSqrtDerivative);                             \
    Macro(SqrOfDifference);                                                  \
    Macro(ElementwiseProductWithExponentialLinearUnitDerivativeFromOutput);  \
    Macro(ElementwiseProductWithStraightThroughDerivative); 
    //Macro(Index);

#define ForAllTernaryOps(Macro)                         \
    Macro(Cond);                                        \
    Macro(CopyIfEqual);                                 \
    Macro(Clip);                                        \
    Macro(ElementwiseProductWithLogSumDerivative);      \
    Macro(ElementwiseProductWithExpOfDiff);             \
    Macro(ElementwiseProductWithQuotient);              \
    Macro(ElementwiseProductWithPowExponentDerivative); \
    Macro(ElementwiseProductWithPowBaseDerivative);

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
    template <class ElemType2> friend class BaseMatrix;

private:
    BaseMatrixStorage<ElemType>(const BaseMatrixStorage<ElemType>& ) = delete;
    BaseMatrixStorage<ElemType>& operator=(const BaseMatrixStorage<ElemType>& ) = delete;
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
        if (!m_externalBuffer)
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

                if (m_tempDeviceBuffer != nullptr)
                    TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(m_computeDevice, m_tempDeviceBuffer, true);
                m_tempDeviceBuffer = nullptr;
                m_tempDeviceBufferSize = 0;
#endif

                delete[](byte*) m_tempHostBuffer;
                m_tempHostBuffer = nullptr;
            }
            m_elemSizeAllocated = 0;
            m_totalBufferSizeAllocated = 0;
        }
    }

protected:
    MatrixFormat GetFormat() const { return m_format; }
    void SetFormat(MatrixFormat format) { m_format = format; }

    bool HasExternalBuffer() const { return m_externalBuffer; }

    DEVICEID_TYPE GetComputeDeviceId() const { return m_computeDevice; }
    void SetComputeDeviceId(const DEVICEID_TYPE computeId) const { m_computeDevice = computeId; }

    size_t GetNumStorageRows() const { return m_numRows; }
    void SetNumStorageRows(size_t rows) { m_numRows = rows; }

    size_t GetNumStorageCols() const { return m_numCols; }
    void SetNumStorageCols(size_t cols) { m_numCols = cols; }

    size_t GetSizeAllocated() const { return m_elemSizeAllocated; }
    void SetSizeAllocated(size_t alloc) { m_elemSizeAllocated = alloc; }

    size_t GetNumStorageElements() const { return m_numRows * m_numCols; }
    bool IsEmpty() const { return m_numRows == 0 || m_numCols == 0; }

    ElemType* Buffer() const { return m_pArray; }
    void SetBuffer(ElemType* pArray, size_t alloc, bool external = false) { m_pArray = pArray; m_totalBufferSizeAllocated = alloc; m_externalBuffer = external; }

    size_t BufferSizeAllocated() const { return m_totalBufferSizeAllocated; }
    
    size_t GetBlockSize() const { return m_blockSize; }
    void SetBlockSize(size_t blockSize) { m_blockSize = blockSize; }

    GPUSPARSE_INDEX_TYPE* GetTempDeviceBuffer() const { return m_tempDeviceBuffer; }
    void ReserveTempDeviceBuffer(const size_t minSize) const
    { 
        BaseMatrixStorage<ElemType>* nonConstThis = const_cast<BaseMatrixStorage<ElemType>*>(this);
        if (minSize > m_tempDeviceBufferSize)
        {
            TracingGPUMemoryAllocator::Free<GPUSPARSE_INDEX_TYPE>(GetComputeDeviceId(), nonConstThis->m_tempDeviceBuffer);
            nonConstThis->m_tempDeviceBuffer = TracingGPUMemoryAllocator::Allocate<GPUSPARSE_INDEX_TYPE>(GetComputeDeviceId(), minSize);
            nonConstThis->m_tempDeviceBufferSize = minSize;
        }
    }

    void* GetTempHostBuffer() const { return m_tempHostBuffer; }
    void SetTempHostBuffer(void* buffer) const { m_tempHostBuffer = buffer; }

    size_t GetTempHostBufferSize() const { return m_tempHostBufferSize; }
    void SetTempHostBufferSize(size_t bufferSize) const { m_tempHostBufferSize = bufferSize; }

    int GetColIdx() const { return m_colIdx; }
    void SetColIdx(int idx) { m_colIdx = idx; }

    size_t GetCompIndexSize() const { return m_compIndexSize; }
    void SetCompIndexSize(size_t indexSize) { m_compIndexSize = indexSize; }

    ElemType* GetNzValues() { return m_nzValues; }
    void SetNzValues(ElemType* values) { m_nzValues = values; }

    size_t* GetBlockIds() const { return m_blockIds; }
    void SetBlockIds(size_t* blockIds) { m_blockIds = blockIds; }

    size_t GetBlockIdShift() const { return m_blockIdShift; }
    void SetBlockIdShift(size_t blockIdShift) { m_blockIdShift = blockIdShift; }

    CPUSPARSE_INDEX_TYPE* GetUnCompIndex() const { return m_unCompIndex; }
    void SetUnCompIndex(CPUSPARSE_INDEX_TYPE* parray) { m_unCompIndex = parray; }
    
    CPUSPARSE_INDEX_TYPE* GetCompIndex() const { return m_compIndex; }
    void SetCompIndex(CPUSPARSE_INDEX_TYPE* parray) { m_compIndex = parray; }

    void ZeroInit(const MatrixFormat matrixFormat = matrixFormatDense, const DEVICEID_TYPE computeDevice = -1)
    {
        m_externalBuffer           = false;
        m_format                   = matrixFormat;
        m_computeDevice            = computeDevice;
        m_numRows                  = 0;
        m_numCols                  = 0;
        m_pArray                   = nullptr;
        m_elemSizeAllocated        = 0;
        m_totalBufferSizeAllocated = 0;
        m_blockSize                = 0; // block size
        m_tempDeviceBuffer         = nullptr;
        m_tempDeviceBufferSize     = 0;
        m_tempHostBuffer           = nullptr; // used to copy values.
        m_tempHostBufferSize       = 0;
        m_colIdx                   = 0; // used to SetValue()
        m_compIndexSize            = 0;
        m_nzValues                 = nullptr;
        m_unCompIndex              = nullptr; // row/col ids in CSC/CSR format
        m_compIndex                = nullptr; // begin ids of col/row in CSC/CSR format
        m_blockIds                 = nullptr; // block ids
        m_blockIdShift             = 0; // used to get efficient slice, actual col = blockIds[j] - m_blockIdShift
    }

protected:
    // **************************
    // Variables required by all matrices
    // **************************
    MatrixFormat m_format;
    mutable DEVICEID_TYPE m_computeDevice; // current GPU device Id or CPUDEVICE
    bool m_externalBuffer; // is the buffer used by this matrix,

    // m_numRows and m_numCols should be removed
    size_t m_numRows;
    size_t m_numCols;
    size_t m_elemSizeAllocated;
    ElemType* m_pArray;

    // **************************
    // GPUSparseMatrix variables
    // **************************

    size_t m_totalBufferSizeAllocated;

    // used by the blockCol and blockRow format
    size_t m_blockSize;                      // block size
    mutable GPUSPARSE_INDEX_TYPE* m_tempDeviceBuffer;
    mutable size_t m_tempDeviceBufferSize;

    mutable void* m_tempHostBuffer; // used to copy values.
    mutable size_t m_tempHostBufferSize;

    // **************************
    // CPUSparseMatrix variables
    // **************************

    int m_colIdx; // used to SetValue()
    size_t m_compIndexSize;
    ElemType* m_nzValues;

    // non-zero values are stored in m_pArray
    CPUSPARSE_INDEX_TYPE* m_unCompIndex; // row/col ids in CSC/CSR format
    CPUSPARSE_INDEX_TYPE* m_compIndex;   // begin ids of col/row in CSC/CSR format

    size_t* m_blockIds;    // block ids
    size_t m_blockIdShift; // used to get efficient slice, actual col = blockIds[j] - m_blockIdShift

};

// -----------------------------------------------------------------------
// BaseMatrix -- base class for all matrix types (CPU, GPU) x (dense, sparse)
// -----------------------------------------------------------------------

template <class ElemType>
class MATH_API BaseMatrix
{
protected:    
    // Default constructor. Copy/Move constructors might set doNotInitialize to true to avoid double initialization.
    BaseMatrix(bool doNotInitializeFields = false)
    {
        if (!doNotInitializeFields)
            ZeroInit();
    }

    virtual ~BaseMatrix()
    {
        ZeroValues();
    }
public:
    void VerifyResizable(const char* function) const 
    { 
        if (!m_sob.unique())
            LogicError("%s: Cannot resize the matrix because it is a view.", function);
        else if (m_sob->HasExternalBuffer())
            LogicError("%s: Cannot resize the matrix because it is externally owned.", function);
    }

    // same as VerifyResizable() except for the error message. Could be folded into one.
    void VerifyMigratable(const char* function) const
    {
        if (!m_sob.unique())
            LogicError("%s: Cannot migrate the matrix between devices because it is a view.", function);
        else if (m_sob->HasExternalBuffer())
            LogicError("%s: Cannot migrate the matrix between devices because it is externally owned.", function);
    }

    // This is needed for Sparse Matrices to ensure they can write to the matrix. Note: writing to slices is not currently supported
    void VerifyWritable(const char* function) const 
    {
        if (!(m_sob->GetNumStorageRows() == m_numRows && m_sob->GetNumStorageCols() == m_numCols))
        {
            LogicError("%s: Cannot write to the matrix because it is a slice.", function);
        }
    }

    bool IsView() const { return (GetNumRows() != m_sob->GetNumStorageRows() || GetNumCols() != m_sob->GetNumStorageCols() || m_sliceViewOffset != 0); }

    void VerifySize(const size_t rows, const size_t cols)
    {
        if (rows != GetNumRows() || cols != GetNumCols())
            LogicError("VerifySize: expected matrix size %lu x %lu, but it is %lu x %lu",
                       rows, cols, GetNumRows(), GetNumCols());
    }

    MatrixFormat GetFormat() const { return m_sob->GetFormat(); }

    bool OwnBuffer() const { return !HasExternalBuffer(); }

    bool IsEmpty() const { return m_numRows == 0 || m_numCols == 0; }

    size_t GetSizeAllocated() const { return m_sob->GetSizeAllocated(); }

    size_t BufferSizeAllocated() const { return m_sob->BufferSizeAllocated(); }

    size_t GetNumRows() const { return m_numRows; }
    size_t GetNumCols() const { return m_numCols; }
    //for non-squared matrix, the major diagonal size is defined by the row or col with smaller dimension
    size_t GetDiagSize() const { return GetNumRows() < GetNumCols() ? GetNumRows() : GetNumCols(); }

protected:

    void SetFormat(MatrixFormat format) { m_sob->SetFormat(format); }

    bool HasExternalBuffer() const { return m_sob->HasExternalBuffer(); }

    DEVICEID_TYPE GetComputeDeviceId() const { return m_sob->GetComputeDeviceId(); }
    void SetComputeDeviceId(const DEVICEID_TYPE computeId) const { m_sob->SetComputeDeviceId(computeId); }

    // TODO: Some of these accessors should be merged into single methods like SetBuffer. 
    size_t GetNumStorageRows() const { return m_sob->GetNumStorageRows(); }
    void SetNumStorageRows(size_t rows) { m_sob->SetNumStorageRows(rows); }

    size_t GetNumStorageCols() const { return m_sob->GetNumStorageCols(); }
    void SetNumStorageCols(size_t cols) { m_sob->SetNumStorageCols(cols); }

    void SetSizeAllocated(size_t alloc) { m_sob->SetSizeAllocated(alloc); }

    ElemType* Buffer() const { return m_sob->Buffer(); }
    void SetBuffer(ElemType* parray, size_t alloc, bool external = false) { m_sob->SetBuffer(parray, alloc, external); }

    
    size_t GetBlockSize() const { return m_sob->GetBlockSize(); }
    void SetBlockSize(size_t blockSize) { m_sob->SetBlockSize(blockSize); }

    GPUSPARSE_INDEX_TYPE* GetTempDeviceBuffer() const { return m_sob->GetTempDeviceBuffer(); }
    void ReserveTempDeviceBuffer(const size_t minSize) const { m_sob->ReserveTempDeviceBuffer(minSize); }

    void* GetTempHostBuffer() const { return m_sob->GetTempHostBuffer(); }
    void SetTempHostBuffer(void* buffer) const { m_sob->SetTempHostBuffer(buffer); };

    size_t GetTempHostBufferSize() const { return m_sob->GetTempHostBufferSize(); }
    void SetTempHostBufferSize(size_t bufferSize) const { m_sob->SetTempHostBufferSize(bufferSize); }

    int GetColIdx() const { return m_sob->GetColIdx(); }
    void SetColIdx(int idx) { m_sob->SetColIdx(idx); }

    size_t GetCompIndexSize() const { return m_sob->GetCompIndexSize(); }
    void SetCompIndexSize(size_t indexSize) { m_sob->SetCompIndexSize(indexSize); }

    ElemType* GetNzValues() { return m_sob->GetNzValues(); }
    void SetNzValues(ElemType* values) { m_sob->SetNzValues(values); }

    size_t* GetBlockIds() const { return m_sob->GetBlockIds(); }
    void SetBlockIds(size_t* blockIds) const { m_sob->SetBlockIds(blockIds); }

    size_t GetBlockIdShift() const { return m_sob->GetBlockIdShift(); }
    void SetBlockIdShift(size_t blockIdShift) { m_sob->SetBlockIdShift(blockIdShift); }

    CPUSPARSE_INDEX_TYPE* GetUnCompIndex() const { return m_sob->GetUnCompIndex(); }
    void SetUnCompIndex(CPUSPARSE_INDEX_TYPE* parray) { m_sob->SetUnCompIndex(parray); }
    
    CPUSPARSE_INDEX_TYPE* GetCompIndex() const { return m_sob->GetCompIndex(); }
    void SetCompIndex(CPUSPARSE_INDEX_TYPE* parray) { m_sob->SetCompIndex(parray); }

    void SetNumRows(size_t numRows) { m_numRows = numRows; }
    void SetNumCols(size_t numCols) { m_numCols = numCols; }

    size_t GetNumElements() const { return m_numRows * m_numCols; }


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
        m_sob               = nullptr;
    }
    void ZeroInit(const MatrixFormat matrixFormat, const DEVICEID_TYPE computeDevice )
    {
        ZeroValues();
        m_sob = make_shared<BaseMatrixStorage<ElemType>>(matrixFormat, computeDevice);
    }

protected:
    //void Clear() {}

    void ZeroStorageInit() { m_sob->ZeroInit(); }
    void ReleaseStorageMemory() { m_sob->ReleaseMemory(); }

    // copy all metadata (but not content that m_sob points to)
    void ShallowCopyFrom(const BaseMatrix& other) 
    {
        *this = other;
    }

protected:

    size_t m_numRows;
    size_t m_numCols;
    // TODO: m_sliceViewOffset has a different meaning in sparse (column offset) versus dense (byte offset to start of pointer). This should perhaps be fixed.
    size_t m_sliceViewOffset; // this is the slice view of a matrix
    // TODO: implement m_colStride
    size_t m_colStride;

    // Storage OBject containing the underlying data used by this matrix
    shared_ptr<BaseMatrixStorage<ElemType>> m_sob;
};

}}}
