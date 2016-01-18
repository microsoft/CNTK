//
// <copyright file="CommonMatrix.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
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

#define DEVICEID_TYPE int
// and the following magic values
#define CPUDEVICE (DEVICEID_TYPE) - 1                 // device is the CPU
#define DEVICEID_NOTYETDETERMINED (DEVICEID_TYPE) - 3 // not yet set
#define DEVICEID_AUTO (DEVICEID_TYPE) - 4             // device should be picked automatically
#define AUTOPLACEMATRIX (DEVICEID_TYPE) 1000          // used in parameters only

#define EPS_IN_INVERSE 1e-30f    // 1e-37 is the only guaranteed precision
#define EPS_IN_LOG 1e-37f        // 1e-37 is the only guaranteed precision
#define LOG_OF_EPS_IN_LOG -85.1f // log(EPS_IN_LOG)
#define LOG10_OF_EPS_IN_LOG -37  // log_10(EPS_IN_LOG)
#define LZERO -10e10
#define MINLOGEXP -9.2103
#define LSMALL -0.5E10

#define GPUSPARSE_INDEX_TYPE int //cuSparse only supports int array indexes
#define CPUSPARSE_INDEX_TYPE int //to be consistent with cuSparse but limited the possible size of the matrix.

MATH_API DEVICEID_TYPE EnforceOneGPUOnly(DEVICEID_TYPE requestedDeviceId);

namespace Microsoft { namespace MSR { namespace CNTK {

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
    // binary ops for indexing
    //opIndex,
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
//Macro(Index);

#define ForAllTernaryOps(Macro) \
    Macro(Cond);                \
    Macro(Clip);

// -----------------------------------------------------------------------
// various enums to describe
// -----------------------------------------------------------------------

enum MatrixFlagBitPosition
{
    bitPosRowMajor = 0,         // row major matrix
    bitPosSparse = 1,           // sparse matrix (COO if uncompressed)
    bitPosCompressed = 2,       // a compressed sparse format (CSC/CSR)
    bitPosDontOwnBuffer = 3,    // buffer is not owned by this matrix
    bitPosSetValueOnDevice = 4, // in a setValue situation, the copy from buffer is already on the device
};

enum MatrixFormat
{
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
    matrixFormatSparseBlockCol,                                                            //col block based sparse matrix
    matrixFormatSparseBlockRow,                                                            //row block based sparse matrix
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
// BaseMatrix -- base class for all matrix types (CPU, GPU) x (dense, sparse)
// -----------------------------------------------------------------------

template <class ElemType>
class BaseMatrix
{
public:
    MatrixFormat GetFormat() const
    {
        return m_format;
    }
    void SetFormat(MatrixFormat format)
    {
        m_format = format;
    }
    size_t GetNumRows() const
    {
        return m_numRows;
    }
    size_t GetNumCols() const
    {
        return m_numCols;
    }
    size_t GetNumElements() const
    {
        return m_numRows * m_numCols;
    }
    bool IsEmpty() const
    {
        return m_numRows == 0 || m_numCols == 0;
    }
    ElemType* GetArray()
    {
        return m_pArray;
    }
    void SetArray(ElemType* parray)
    {
        m_pArray = parray;
    }
    virtual DEVICEID_TYPE GetComputeDeviceId() const
    {
        return m_computeDevice;
    }
    void SetComputeDeviceId(const DEVICEID_TYPE computeId) const
    {
        m_computeDevice = computeId;
    }
    bool OwnBuffer() const
    {
        return !m_externalBuffer;
    }
    void SetOwnBuffer(bool own)
    {
        m_externalBuffer = !own;
    }
    wchar_t* GetMatrixName() const
    {
        return m_matrixName;
    }
    size_t NzCount() const
    {
        return m_nz;
    }
    void SetNzCount(const size_t nz)
    {
        m_nz = nz;
    }
    size_t GetSizeAllocated() const
    {
        return m_elemSizeAllocated;
    }
    void VerifySize(size_t rows, size_t cols)
    {
        if (rows != GetNumRows() || cols != GetNumCols())
            LogicError("VerifySize: expected matrix size %lu x %lu, but it is %lu x %lu",
                       rows, cols, GetNumRows(), GetNumCols());
    }
    void SetMatrixName(const wchar_t* s)
    {
        Clear();
        if (s != nullptr)
        {
            size_t n = wcslen(s);
            m_matrixName = new wchar_t[n + 1];
            wmemcpy(m_matrixName, s, n + 1);
        }
    }

    BaseMatrix()
    {
        m_numRows = m_numCols = m_elemSizeAllocated = 0;
        m_pArray = NULL;
        m_matrixName = NULL;
        m_format = matrixFormatDense;
        m_externalBuffer = false;
        m_nz = 0;
        m_computeDevice = CPUDEVICE;
    }
    ~BaseMatrix()
    {
        Clear();
    }

protected:
    void Clear()
    {
        if (m_matrixName != nullptr)
        {
            delete[] m_matrixName;
            m_matrixName = nullptr;
        }
    }

protected:
    size_t m_numRows;
    size_t m_numCols;
    size_t m_elemSizeAllocated;
    size_t m_sliceViewOffset; // this is used to get a column slice view of a matrix in the Sparse CSC format
    MatrixFormat m_format;
    bool m_externalBuffer; // is the buffer used by this matrix,
    ElemType* m_pArray;
    mutable DEVICEID_TYPE m_computeDevice; //current GPU device Id or CPUDEVICE
    size_t m_nz;                           //Number of non-zero elements for sparse matrices (unused in other formats)
    wchar_t* m_matrixName;
};
} } }
