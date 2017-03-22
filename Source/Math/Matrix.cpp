//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Matrix.cpp -- main CPP file that contains all Matrix functions exported by the Cntk.Math.dll
//
#include "stdafx.h"
#include "Basics.h"
#include "Matrix.h"
#include "CPUMatrix.h"
#include "CPUSparseMatrix.h"
#include "GPUMatrix.h"
#include "GPUSparseMatrix.h"
#include "File.h"
#include <assert.h>
#include <math.h>
#include "GPUWatcher.h" // bring in this class as well so that it gets exported from this DLL
#include <memory>
#include <atomic>
#include "Quantizers.h"
#ifndef CPUONLY
#define ANAMEFORLIB "Cntk.Math.Cuda-" ## CNTK_COMPONENT_VERSION ## ".lib"
#pragma comment(lib, ANAMEFORLIB) // built by MathCUDA project
#undef ANAMEFORLIB
#endif

#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this
#pragma warning(disable : 4239) // nonstandard extension; triggered by this pattern: "auto& second = transposeB ? b.m_GPUMatrix->Transpose() : *b.m_GPUMatrix;"
#pragma warning(disable : 4702) // unreachable code; triggered for unknown reasons

// Helper to dispath matrix calls to the 4 underlying matrix libraries (CPU,GPU) x (DENSE,SPARSE)
// 'MatrixPointerToCheck' determines where the operation takes place.
// 'MatrixPointerToSetFlag' is the output. If not null and its location is BOTH, we collapse it to one.
#pragma warning(disable : 4456) // declaration of curLocation hides previous local declaration
#define DISPATCH_MATRIX_ON_FLAG(MatrixPointerToCheck, MatrixPointerToSetFlag, CPUDense, GPUDense, CPUSparse, GPUSparse) \
    {                                                                                                                   \
        CurrentDataLocation curLocation = (MatrixPointerToCheck)->GetCurrentMatrixLocation();                           \
        if (curLocation == CurrentDataLocation::GPU || curLocation == CurrentDataLocation::BOTH)                        \
        {                                                                                                               \
            if ((MatrixPointerToCheck)->GetMatrixType() != MatrixType::SPARSE)                                          \
            {                                                                                                           \
                GPUDense;                                                                                               \
                if (MatrixPointerToSetFlag != nullptr)                                                                  \
                    ((Matrix*) MatrixPointerToSetFlag)->SetDataLocation(CurrentDataLocation::GPU, MatrixType::DENSE);   \
            }                                                                                                           \
            else                                                                                                        \
            {                                                                                                           \
                GPUSparse;                                                                                              \
                if (MatrixPointerToSetFlag != nullptr)                                                                  \
                    ((Matrix*) MatrixPointerToSetFlag)->SetDataLocation(CurrentDataLocation::GPU, MatrixType::SPARSE);  \
            }                                                                                                           \
        }                                                                                                               \
        else if (curLocation == CurrentDataLocation::CPU)                                                               \
        {                                                                                                               \
            if ((MatrixPointerToCheck)->GetMatrixType() != MatrixType::SPARSE)                                          \
            {                                                                                                           \
                CPUDense;                                                                                               \
                if (MatrixPointerToSetFlag != nullptr)                                                                  \
                    ((Matrix*) MatrixPointerToSetFlag)->SetDataLocation(CurrentDataLocation::CPU, MatrixType::DENSE);   \
            }                                                                                                           \
            else                                                                                                        \
            {                                                                                                           \
                CPUSparse;                                                                                              \
                if (MatrixPointerToSetFlag != nullptr)                                                                  \
                    ((Matrix*) MatrixPointerToSetFlag)->SetDataLocation(CurrentDataLocation::CPU, MatrixType::SPARSE);  \
            }                                                                                                           \
        }                                                                                                               \
        else                                                                                                            \
        {                                                                                                               \
            RuntimeError("Matrices do not exist in either CPU or GPU.");                                                \
        }                                                                                                               \
    }

// version of dispatch macro that prefers the CPU if the 'MatrixPointerToCheck' location is BOTH
#define DISPATCH_MATRIX_ON_FLAG_USECPU_4BOTH(MatrixPointerToCheck, MatrixPointerToSetFlag, CPUDense, GPUDense, CPUSparse, GPUSparse) \
    {                                                                                                                                \
        CurrentDataLocation curLocation = (MatrixPointerToCheck)->GetCurrentMatrixLocation();                                        \
        if (curLocation == CurrentDataLocation::GPU)                                                                                 \
        {                                                                                                                            \
            if ((MatrixPointerToCheck)->GetMatrixType() != MatrixType::SPARSE)                                                       \
            {                                                                                                                        \
                GPUDense;                                                                                                            \
                if (MatrixPointerToSetFlag != nullptr)                                                                               \
                    ((Matrix*) MatrixPointerToSetFlag)->SetDataLocation(CurrentDataLocation::GPU, MatrixType::DENSE);                \
            }                                                                                                                        \
            else                                                                                                                     \
            {                                                                                                                        \
                GPUSparse;                                                                                                           \
                if (MatrixPointerToSetFlag != nullptr)                                                                               \
                    ((Matrix*) MatrixPointerToSetFlag)->SetDataLocation(CurrentDataLocation::GPU, MatrixType::SPARSE);               \
            }                                                                                                                        \
        }                                                                                                                            \
        else if (curLocation == CurrentDataLocation::CPU || curLocation == CurrentDataLocation::BOTH)                                \
        {                                                                                                                            \
            if ((MatrixPointerToCheck)->GetMatrixType() != MatrixType::SPARSE)                                                       \
            {                                                                                                                        \
                CPUDense;                                                                                                            \
                if (MatrixPointerToSetFlag != nullptr)                                                                               \
                    ((Matrix*) MatrixPointerToSetFlag)->SetDataLocation(CurrentDataLocation::CPU, MatrixType::DENSE);                \
            }                                                                                                                        \
            else                                                                                                                     \
            {                                                                                                                        \
                CPUSparse;                                                                                                           \
                if (MatrixPointerToSetFlag != nullptr)                                                                               \
                    ((Matrix*) MatrixPointerToSetFlag)->SetDataLocation(CurrentDataLocation::CPU, MatrixType::SPARSE);               \
            }                                                                                                                        \
        }                                                                                                                            \
        else                                                                                                                         \
        {                                                                                                                            \
            RuntimeError("Matrices do not exist in either CPU or GPU.");                                                             \
        }                                                                                                                            \
    }

// version of helper macro that executes both CPU and GPU macros if 'matrixPointer' location is BOTH
#define DISPATCH_MATRIX_ON_FLAG_USEBOTH_4BOTH(matrixPointer, CPUDense, GPUDense, CPUSparse, GPUSparse)  \
    {                                                                                                   \
        auto curLocation = (matrixPointer)->GetCurrentMatrixLocation();                                 \
        auto curMatrixType = (matrixPointer)->GetMatrixType();                                          \
        if (curLocation == CurrentDataLocation::NONE)                                                   \
            LogicError("Matrices do not exist in either CPU or GPU.");                                  \
        if (curMatrixType == MatrixType::UNDETERMINED)                                                  \
            LogicError("Matrices must be SPARSE or DENSE.");                                            \
        if (curLocation != CurrentDataLocation::CPU) /*GPU or BOTH*/                                    \
        {                                                                                               \
            if (curMatrixType == MatrixType::DENSE)                                                     \
            {                                                                                           \
                GPUDense;                                                                               \
            }                                                                                           \
            else                                                                                        \
            {                                                                                           \
                GPUSparse;                                                                              \
            }                                                                                           \
        }                                                                                               \
        if (curLocation != CurrentDataLocation::GPU) /*CPU or BOTH*/                                    \
        {                                                                                               \
            if (curMatrixType == MatrixType::DENSE)                                                     \
            {                                                                                           \
                CPUDense;                                                                               \
            }                                                                                           \
            else                                                                                        \
            {                                                                                           \
                CPUSparse;                                                                              \
            }                                                                                           \
        }                                                                                               \
    }

namespace Microsoft { namespace MSR { namespace CNTK {

std::atomic<int> m_mathLibTraceLevel(0);

void SetMathLibTraceLevel(int traceLevel)
{
    m_mathLibTraceLevel.store(traceLevel);
}

int GetMathLibTraceLevel()
{
    return m_mathLibTraceLevel.load();
}

MatrixBase::~MatrixBase() { }

#pragma region Constructors, destructors and other static matrix builders


// TODO: Reformat DISPATCH... macros to the following form:
//          DISPATCH..(p1, p2,
//            { Cpu code },
//            { GPU code },
//            ...

// Initialize members 
template <class ElemType>
void Matrix<ElemType>::Init(DEVICEID_TYPE deviceId)
{
    ReleaseMemory();
    m_preferredDeviceId = deviceId;
    m_numTimesDeviceChanged = 0;
    m_numTimesMatrixTypeChanged = 0;
    m_devicesTransferedTo[1]    = m_devicesTransferedTo[0] = CPUDEVICE - 1; // (some value that is different from any valid value)
}

// shallow-copy all members
template <class ElemType>
void Matrix<ElemType>::ShallowCopyFrom(const Matrix<ElemType>& other)
{
    m_baseMatrix                = other.m_baseMatrix;
    m_GPUMatrix                 = other.m_GPUMatrix;
    m_CPUMatrix                 = other.m_CPUMatrix;
    m_GPUSparseMatrix           = other.m_GPUSparseMatrix;
    m_CPUSparseMatrix           = other.m_CPUSparseMatrix;

    m_matrixType                = other.m_matrixType;
    m_currentDataLocation       = other.m_currentDataLocation;

    m_preferredDeviceId         = other.m_preferredDeviceId;
    m_numTimesDeviceChanged     = other.m_numTimesDeviceChanged;
    m_numTimesMatrixTypeChanged = other.m_numTimesMatrixTypeChanged;
    m_devicesTransferedTo[0]    = other.m_devicesTransferedTo[0]; // TODO: spelling
    m_devicesTransferedTo[1]    = other.m_devicesTransferedTo[1];
}

// Call this function after an update operation has created/set/updated the respective pointers.
//  - location: BOTH|CPU|GPU
//     - pass BOTH only if object will be read from; it is not allowed to write to both and then call this function.
//     - if CPU/GPU and current is BOTH, then object was written to
// What gets updated:
//  - m_currentDataLocation: from function argument
//  - m_matrixType:          from function argument unless UNDETERMINED in which case m_matrixType remains unmodified
//  - m_baseMatrix:          to one of current values of m_[GC]PU{Sparse,}Matrix
// This function is heavily overloaded in its responsibility.
//  - first-time initialization, e.g. of a ColumnSlice (NONE->!NONE)
//  - after creating a temp copy for reading
//  - collapse temp copies after writing to one of them
//  - setting matrixType if not set yet
template <class ElemType>
void Matrix<ElemType>::SetDataLocation(CurrentDataLocation location, MatrixType type) const
{
    assert(location == CurrentDataLocation::CPU || location == CurrentDataLocation::GPU || location == CurrentDataLocation::BOTH);

    // if the object used to live on BOTH, this will collapse it to 'location' (unless we actually wrote into BOTH)
    // In that case, we do a sanity check here that the object is a singleton view,
    // since otherwise the collapsing would go unnoticed by the other views.
    // The cases to cover:
    //  - everything is allowed on a singleton view
    //     - if the current state is BOTH:
    //       -> The result was written to 'location' so we should collapse it to there.
    //  - multiple views: much is forbidden since we cannot notify the other views on which one was written to
    //     - CPU <-> GPU: FORBIDDEN
    //     - BOTH -> CPU or GPU: current state is BOTH: location says which side was written to
    //       -> FORBIDDEN to write into
    //     - CPU or GPU -> BOTH: current state is CPU or GPU
    //       and a view onto it is put into BOTH state
    //       -> OK but inefficent to read, since this is likely happening over again; but we cannot put all views into BOTH state
    //     - BOTH -> BOTH:
    //        - read case: OK
    //        - write case: forbidden to call this function in this way
    //     - NONE -> !NONE: FORBIDDEN
    if (m_currentDataLocation != location &&                  // it is attempted to change location
        m_currentDataLocation != CurrentDataLocation::NONE && // from a valid object (NONE means we are a fresh object from ColumnSlice())
        location != CurrentDataLocation::BOTH)                // and we are changing it not into a temporary copy for reading
    {
        // we get here if we wrote into this object that was BOTH but is no longer, or if we move between CPU and GPU
        // Both is forbidden on shared views since we cannot inform other views of this change.
        // We will now check any *valid* pointer will now be checked for uniqueness. There may be mismatching left-over pointers kept around in case they should be revived.
        if (m_matrixType == MatrixType::DENSE) // note: this checks the current type, not the new one passed in. Asssumption: this tells us which pointers are valid.
        {
            assert(m_currentDataLocation == CurrentDataLocation::GPU || m_CPUMatrix);
            assert(m_currentDataLocation == CurrentDataLocation::CPU || m_GPUMatrix);
            if (m_currentDataLocation != CurrentDataLocation::GPU) ((BaseMatrix<ElemType>*)m_CPUMatrix.get())->VerifyMigratable("SetDataLocation [CPUMatrix]");
            if (m_currentDataLocation != CurrentDataLocation::CPU) ((BaseMatrix<ElemType>*)m_GPUMatrix.get())->VerifyMigratable("SetDataLocation [GPUMatrix]");
        }
        else if (m_matrixType == MatrixType::SPARSE)
        {
            assert(m_currentDataLocation == CurrentDataLocation::GPU || m_CPUSparseMatrix);
            assert(m_currentDataLocation == CurrentDataLocation::CPU || m_GPUSparseMatrix);
            if (m_currentDataLocation != CurrentDataLocation::GPU) ((BaseMatrix<ElemType>*)m_CPUSparseMatrix.get())->VerifyMigratable("SetDataLocation [CPUSparseMatrix]");
            if (m_currentDataLocation != CurrentDataLocation::CPU) ((BaseMatrix<ElemType>*)m_GPUSparseMatrix.get())->VerifyMigratable("SetDataLocation [GPUSparseMatrix]");
        }
        // TODO: Why do we need these typecasts? (without it will fail with "cannot access private member declared in class 'Microsoft::MSR::CNTK::CPUMatrix<float>'")

        if (m_baseMatrix && !OwnBuffer()) // same arguments for externally owned matrices: Can read a temp but not write.
            LogicError("SetDataLocation: A non-owning object cannot be written to in BOTH state.");
    }
    // passed validation: we can now update the state

    m_currentDataLocation = location;

    // update the matrix type if passed in
    if (type != MatrixType::UNDETERMINED)
        m_matrixType = type;

    // set m_baseMatrix (if location is unchanged, this will not change the pointer)
    // Note: m_currentDataLocation may also be CurrentDataLocation::BOTH, in which case the base matrix will be GPU.
    if (m_matrixType == MatrixType::DENSE)
        m_baseMatrix = ((m_currentDataLocation == CurrentDataLocation::CPU) ? dynamic_cast<BaseMatrix<ElemType>*>(m_CPUMatrix.get()) : dynamic_cast<BaseMatrix<ElemType>*>(m_GPUMatrix.get()));
    else if (m_matrixType == MatrixType::SPARSE)
        m_baseMatrix = ((m_currentDataLocation == CurrentDataLocation::CPU) ? dynamic_cast<BaseMatrix<ElemType>*>(m_CPUSparseMatrix.get()) : dynamic_cast<BaseMatrix<ElemType>*>(m_GPUSparseMatrix.get()));
    // Note: Typecasts are necessary since C++ cannot figure out the common base type (probably due to shared_ptr).
    // sanity check
    if (!m_baseMatrix && m_matrixType != MatrixType::UNDETERMINED)
        LogicError("SetDataLocation: New m_baseMatrix must not be NULL.");
}

//this is a private constructor only used internally to initialize a blank matrix
template <class ElemType>
Matrix<ElemType>::Matrix(const MatrixFlags matrixFlags, const MatrixType matrixType, const MatrixFormat matrixFormat, DEVICEID_TYPE deviceID)
{
    Init(deviceID);

    if (!(matrixFlags & matrixFlagDontOwnBuffer))
        SwitchToMatrixType(matrixType, matrixFormat, false);
}

//this is a private constructor only used internally to initialize a blank matrix
template <class ElemType>
Matrix<ElemType>::Matrix(const MatrixFlags matrixFlags, const MatrixType matrixType, DEVICEID_TYPE deviceID)
{
    Init(deviceID);

    if (!(matrixFlags & matrixFlagDontOwnBuffer))
        SwitchToMatrixType(matrixType, matrixType == MatrixType::DENSE ? MatrixFormat::matrixFormatDense : MatrixFormat::matrixFormatSparseCSC, false);
}

//this is a private constructor only used internally to initialize a blank matrix
template <class ElemType>
Matrix<ElemType>::Matrix(const MatrixFlags matrixFlags, DEVICEID_TYPE deviceID)
{
    Init(deviceID);

    if (!(matrixFlags & matrixFlagDontOwnBuffer))
        SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, false);
}

template <class ElemType>
Matrix<ElemType>::Matrix(DEVICEID_TYPE deviceID)
{
    Init(deviceID);

    SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, false);
}

// constructor for Matrix class to wrap an externally managed BaseMatrix, indicated by the use of shared_ptr.
//    The appropriate destructor should be passed in by the caller.
// baseMatrix - base matrix for this element
// pArray - pointer to current data array, will replace existing pointer in baseMatrix if != NULL
// deviceId - deviceId where the pArray exists
#if 0
template <class ElemType>
Matrix<ElemType>::Matrix(shared_ptr<BaseMatrix<ElemType>> baseMatrix, ElemType* pArray, DEVICEID_TYPE deviceId) // constructor for setting Matrix from a base matrix
{
    Init(deviceId);

    if (baseMatrix->GetFormat() & matrixFormatSparse)
    {
        if (m_preferredDeviceId == CPUDEVICE)
        {
            m_CPUSparseMatrix = DownCast<CPUSparseMatrix<ElemType>>(baseMatrix);
            SetDataLocation(CPU, SPARSE);
        }
        else
        {
            m_GPUSparseMatrix = DownCast<GPUSparseMatrix<ElemType>>(baseMatrix);
            SetDataLocation(GPU, SPARSE);
        }
    }
    else
    {
        if (m_preferredDeviceId == CPUDEVICE)
        {
            m_CPUMatrix = DownCast<CPUMatrix<ElemType>>(baseMatrix);
            SetDataLocation(CPU, DENSE);
        }
        else
        {
            m_GPUMatrix = DownCast<GPUMatrix<ElemType>>(baseMatrix);
            SetDataLocation(GPU, DENSE);
        }
    }
    m_baseMatrix = baseMatrix;
    m_baseMatrix->SetBuffer(pArray,0);
}
#endif

template <class ElemType>
Matrix<ElemType>::Matrix(const size_t numRows, const size_t numCols, DEVICEID_TYPE deviceId, const MatrixType matrixType, const MatrixFormat matrixFormat, const size_t nnz)
{
    Init(deviceId);

    if (matrixType == MatrixType::SPARSE)
    {
        if (m_preferredDeviceId == CPUDEVICE)
        {
            m_CPUSparseMatrix = make_shared<CPUSparseMatrix<ElemType>>(matrixFormat, numRows, numCols, nnz);
            SetDataLocation(CPU, SPARSE);
        }
        else
        {
            m_GPUSparseMatrix = make_shared<GPUSparseMatrix<ElemType>>(numRows, numCols, nnz, m_preferredDeviceId, matrixFormat);
            SetDataLocation(GPU, SPARSE);
        }
    }
    else
    {
        if (matrixFormat != matrixFormatDense)
        {
            NOT_IMPLEMENTED;
        }

        if (m_preferredDeviceId == CPUDEVICE)
        {
            m_CPUMatrix = make_shared<CPUMatrix<ElemType>>(numRows, numCols);
            SetDataLocation(CPU, DENSE);
        }
        else
        {
            m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(numRows, numCols, m_preferredDeviceId);
            SetDataLocation(GPU, DENSE);
        }

        SetValue(0);
    }
}

template <class ElemType>
Matrix<ElemType>::Matrix(const size_t numRows, const size_t numCols, ElemType* pArray, DEVICEID_TYPE deviceId, const size_t matrixFlags, const size_t nnz)
{
    Init(deviceId);

    if (m_preferredDeviceId == CPUDEVICE)
    {
        if (matrixFlags & matrixFormatSparse)
        {
            // WARNING: matrixFlag is not passed in and externally managed array cannot be passed in
            m_CPUSparseMatrix = make_shared<CPUSparseMatrix<ElemType>>(matrixFormatSparseCSC, numRows, numCols, nnz);
            SetDataLocation(CPU, SPARSE);
        }
        else
        {
            m_CPUMatrix = make_shared<CPUMatrix<ElemType>>(numRows, numCols, pArray, matrixFlags);
            SetDataLocation(CPU, DENSE);
        }
    }
    else
    {
        if (matrixFlags & matrixFormatSparse)
        {
            // m_GPUSparseMatrix = new GPUSparseMatrix<ElemType>(numRows,numCols,nnz, pArray,matrixFlags,m_preferredDeviceId);
            m_GPUSparseMatrix = make_shared<GPUSparseMatrix<ElemType>>(m_preferredDeviceId, MatrixFormat(matrixFlags & MatrixFormat::matrixFormatMask));
            m_GPUSparseMatrix->RequireSizeAndAllocate(numRows, numCols, nnz, true, false);
            SetDataLocation(GPU, SPARSE);
        }
        else
        {
            m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(numRows, numCols, m_preferredDeviceId, pArray, matrixFlags);
            SetDataLocation(GPU, DENSE);
        }
    }

    // Why is this here??
    /*
    if (matrixFlagDontOwnBuffer & matrixFlags)
        m_baseMatrix->SetOwnBuffer(false);
        */
}

//copy constructor, deep copy
template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::DeepClone() const
{
    return Matrix<ElemType>(*this, GetDeviceId());
}

template <class ElemType>
Matrix<ElemType>::Matrix(const Matrix<ElemType>& deepCopyFrom, DEVICEID_TYPE deviceId)
{
    int origCopyFromDeviceId = deepCopyFrom.GetDeviceId();

    Init(deviceId); // will set m_preferredDeviceId

    deepCopyFrom._transferToDevice(m_preferredDeviceId, true);

    DISPATCH_MATRIX_ON_FLAG(&deepCopyFrom,
                            this,
                            m_CPUMatrix = make_shared<CPUMatrix<ElemType>>(*(deepCopyFrom.m_CPUMatrix)),
                            m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(*(deepCopyFrom.m_GPUMatrix)),
                            m_CPUSparseMatrix = make_shared<CPUSparseMatrix<ElemType>>(*(deepCopyFrom.m_CPUSparseMatrix)),
                            m_GPUSparseMatrix = make_shared<GPUSparseMatrix<ElemType>>(*(deepCopyFrom.m_GPUSparseMatrix)));

    // should we move back?
    deepCopyFrom._transferToDevice(origCopyFromDeviceId, true);

    m_preferredDeviceId = deepCopyFrom.m_preferredDeviceId;
}


//move constructor, shallow copy
template <class ElemType>
Matrix<ElemType>::Matrix(Matrix<ElemType>&& moveFrom)
{
    Init((DEVICEID_TYPE)moveFrom.GetDeviceId());

#if 1
    operator=(move(moveFrom));
#else
    DISPATCH_MATRIX_ON_FLAG(&moveFrom,
                            this,
                            m_CPUMatrix = new CPUMatrix<ElemType>(static_cast<CPUMatrix<ElemType>&&>(*(moveFrom.m_CPUMatrix))),
                            m_GPUMatrix = new GPUMatrix<ElemType>(static_cast<GPUMatrix<ElemType>&&>(*(moveFrom.m_GPUMatrix))),
                            m_CPUSparseMatrix = new CPUSparseMatrix<ElemType>(static_cast<CPUSparseMatrix<ElemType>&&>(*(moveFrom.m_CPUSparseMatrix))),
                            m_GPUSparseMatrix = new GPUSparseMatrix<ElemType>(static_cast<GPUSparseMatrix<ElemType>&&>(*(moveFrom.m_GPUSparseMatrix))));

    m_preferredDeviceId = moveFrom.m_preferredDeviceId;
#endif
}

//move assignment operator, shallow copy
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::operator=(Matrix<ElemType>&& moveFrom)
{
    if (this == &moveFrom)
        LogicError("Matrix: Move assignment into itself is forbidden.");
#if 1
    // shallow-copy all members
    ShallowCopyFrom(moveFrom);
    // virgin-init the source
    moveFrom.Init(CPUDEVICE);
#else
    m_preferredDeviceId = moveFrom.m_preferredDeviceId;

    DISPATCH_MATRIX_ON_FLAG(&moveFrom,
                            this,
                            if (m_CPUMatrix != nullptr) m_CPUMatrix->operator=(static_cast<CPUMatrix<ElemType>&&>(*(moveFrom.m_CPUMatrix)));
                            else m_CPUMatrix = new CPUMatrix<ElemType>(static_cast<CPUMatrix<ElemType>&&>(*(moveFrom.m_CPUMatrix))),

                            if (m_GPUMatrix != nullptr) m_GPUMatrix->operator=(static_cast<GPUMatrix<ElemType>&&>(*(moveFrom.m_GPUMatrix)));
                            else m_GPUMatrix = new GPUMatrix<ElemType>(static_cast<GPUMatrix<ElemType>&&>(*(moveFrom.m_GPUMatrix))),

                            if (m_CPUSparseMatrix != nullptr) m_CPUSparseMatrix->operator=(static_cast<CPUSparseMatrix<ElemType>&&>(*(moveFrom.m_CPUSparseMatrix)));
                            else m_CPUSparseMatrix = new CPUSparseMatrix<ElemType>(static_cast<CPUSparseMatrix<ElemType>&&>(*(moveFrom.m_CPUSparseMatrix))),

                            if (m_GPUSparseMatrix != nullptr) m_GPUSparseMatrix->operator=(static_cast<GPUSparseMatrix<ElemType>&&>(*(moveFrom.m_GPUSparseMatrix)));
                            else m_GPUSparseMatrix = new GPUSparseMatrix<ElemType>(static_cast<GPUSparseMatrix<ElemType>&&>(*(moveFrom.m_GPUSparseMatrix))));

#endif
    return *this;
}

template <class ElemType>
void Matrix<ElemType>::ReleaseMemory()
{
    m_baseMatrix = nullptr;
    // Perf: Avoid assignments to shared_ptr unless necessary. In certain versions of the standard library
    // they cause ref counting, and this piece of code is called often..
    if (m_GPUMatrix.get() != nullptr)
        m_GPUMatrix = nullptr;
    if (m_CPUMatrix.get() != nullptr)
        m_CPUMatrix = nullptr;
    if (m_GPUSparseMatrix.get() != nullptr)
        m_GPUSparseMatrix = nullptr;
    if (m_CPUSparseMatrix.get() != nullptr)
        m_CPUSparseMatrix = nullptr;
    m_matrixType = MatrixType::UNDETERMINED;
    m_currentDataLocation = CurrentDataLocation::NONE;
}

template <class ElemType>
Matrix<ElemType>::~Matrix(void)
{
    ReleaseMemory();
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::Ones(const size_t rows, const size_t cols, DEVICEID_TYPE deviceId)
{
    Matrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    c.SetValue(1);
    return c;
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::Zeros(const size_t rows, const size_t cols, DEVICEID_TYPE deviceId)
{
    Matrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    c.SetValue(0);
    return c;
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::Eye(const size_t rows, DEVICEID_TYPE deviceId)
{
    Matrix<ElemType> c(rows, rows, deviceId); // will initialize to 0
    c.SetDiagonalValue(1);
    return c;
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::RandomUniform(const size_t rows, const size_t cols, DEVICEID_TYPE deviceId, const ElemType low, const ElemType high, unsigned long seed)
{
    Matrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    c.SetUniformRandomValue(low, high, seed);
    return c;
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::RandomGaussian(const size_t rows, const size_t cols, DEVICEID_TYPE deviceId, const ElemType mean, const ElemType sigma, unsigned long seed)
{
    Matrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    c.SetGaussianRandomValue(mean, sigma, seed);
    return c;
}

template <class ElemType>
void Matrix<ElemType>::SetDevice(DEVICEID_TYPE deviceId)
{
    if (deviceId >= 0)
        GPUMatrix<ElemType>::SetDevice(deviceId);
}

template <class ElemType>
void Matrix<ElemType>::Read(File& stream)
{
    Matrix<ElemType>& M = *this;
    char type;
    stream >> type;
    if (type == 'd')
    {
        if (M.GetDeviceId() < 0)
        {
            if (!M.m_CPUMatrix)
                M.m_CPUMatrix = make_shared<CPUMatrix<ElemType>>();
            stream >> (*M.m_CPUMatrix);
            M.SetDataLocation(CPU, DENSE);
        }
        else
        {
            if (!M.m_GPUMatrix)
                M.m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(M.GetDeviceId());
            stream >> (*M.m_GPUMatrix);
            M.SetDataLocation(GPU, DENSE);
        }
    }
    else if (type == 's')
    {
        if (M.GetDeviceId() < 0)
        {
            NOT_IMPLEMENTED; // You might want to tranfer your matrix to GPU
        }
        else
        {
            if (M.m_GPUSparseMatrix == NULL)
                M.m_GPUSparseMatrix = make_shared<GPUSparseMatrix<ElemType>>(M.GetDeviceId());
            stream >> (*M.m_GPUSparseMatrix);
            M.SetDataLocation(GPU, SPARSE);
        }
    }
    else
        LogicError("Read: Input file corrupt (invalid matrix type field 0x%02d, should be 'f' or 'd').", type);
}

template <class ElemType>
void Matrix<ElemType>::Write(File& stream) const
{
    const Matrix<ElemType>& M = *this;
    if (M.GetMatrixType() == MatrixType::DENSE)
    {
        stream << 'd';
        if (M.GetDeviceId() < 0)
            stream << (*M.m_CPUMatrix);
        else
            stream << (*M.m_GPUMatrix);
    }
    else
    {
        stream << 's';
        if (M.GetDeviceId() < 0)
            NOT_IMPLEMENTED // stream<<(*M.m_CPUMatrix);
                else stream
                << (*M.m_GPUSparseMatrix);
    }
}

#pragma endregion Constructors, destructors and other static matrix builders

#pragma region Basic Operators

template <class ElemType>
size_t Matrix<ElemType>::BufferSize() const
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            return m_baseMatrix->GetSizeAllocated() * sizeof(ElemType),
                            return m_baseMatrix->GetSizeAllocated() * sizeof(ElemType),
                            return m_CPUSparseMatrix->BufferSize(),
                            return m_GPUSparseMatrix->BufferSizeAllocated());
}

// BUGBUG: This is ugly code. The outside world should not have access to the raw data pointers.
// if this is to be used, then at least it should also return a number of bytes as well.
template <class ElemType>
ElemType* Matrix<ElemType>::Data() const
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            return m_CPUMatrix->Data(),
                            return m_GPUMatrix->Data(),
                            return m_CPUSparseMatrix->Data(),
                            return m_GPUSparseMatrix->Data());
}

template <class ElemType>
ElemType* Matrix<ElemType>::CopyToArray() const
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            return m_CPUMatrix->CopyToArray(),
                            return m_GPUMatrix->CopyToArray(),
                            { CPUMatrix<ElemType> tmpDense(m_CPUSparseMatrix->GetNumRows(), m_CPUSparseMatrix->GetNumCols()); tmpDense.SetValue((ElemType)0); CPUSparseMatrix<ElemType>::ScaleAndAdd((ElemType)1, *m_CPUSparseMatrix, tmpDense); return tmpDense.CopyToArray(); },
                            return m_GPUSparseMatrix->CopyToDenseMatrix().CopyToArray());
}

//memory will be allocated by the callee if not enough but need to be deleted by the caller after it's done
//return number of elements copied
template <class ElemType>
size_t Matrix<ElemType>::CopyToArray(ElemType*& arrayCopyTo, size_t& currentArraySize) const
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            return m_CPUMatrix->CopyToArray(arrayCopyTo, currentArraySize),
                            return m_GPUMatrix->CopyToArray(arrayCopyTo, currentArraySize),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::CopySection(size_t numRows, size_t numCols, ElemType* dst, size_t colStride) const
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            m_CPUMatrix->CopySection(numRows, numCols, dst, colStride),
                            m_GPUMatrix->CopySection(numRows, numCols, dst, colStride),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

// BUGBUG: Some code checks before calling here whether one of the dimensions is 0.
//         This function must handle that case properly, that is, preserving the non-zero dimension.
template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::ColumnSlice(size_t startColumn, size_t numCols) const
{
    int devId = GetDeviceId();

    Matrix<ElemType> slice(matrixFlagDontOwnBuffer, (DEVICEID_TYPE) devId); // this already creates pointers

    slice.m_preferredDeviceId = m_preferredDeviceId;

    // create slices for the underlying object
    // Note: In case of data location == BOTH, this creates two objects just like in the source.
    if (GetMatrixType() == MatrixType::DENSE)
    {
        if (GetCurrentMatrixLocation() == CPU || GetCurrentMatrixLocation() == BOTH)
        {
            if (slice.m_CPUMatrix)
                slice.m_CPUMatrix->operator=(static_cast<CPUMatrix<ElemType>&&>(m_CPUMatrix->ColumnSlice(startColumn, numCols)));
            else
                slice.m_CPUMatrix = make_shared<CPUMatrix<ElemType>>(static_cast<CPUMatrix<ElemType>&&>(m_CPUMatrix->ColumnSlice(startColumn, numCols)));
        }
        if (GetCurrentMatrixLocation() == GPU || GetCurrentMatrixLocation() == BOTH)
        {
            if (slice.m_GPUMatrix)
                slice.m_GPUMatrix->operator=(static_cast<GPUMatrix<ElemType>&&>(m_GPUMatrix->ColumnSlice(startColumn, numCols)));
            else
                slice.m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(static_cast<GPUMatrix<ElemType>&&>(m_GPUMatrix->ColumnSlice(startColumn, numCols)));
        }
    }
    else if (GetMatrixType() == MatrixType::SPARSE)
    {
        if (GetCurrentMatrixLocation() == CPU || GetCurrentMatrixLocation() == BOTH)
        {
            if (slice.m_CPUSparseMatrix)
                slice.m_CPUSparseMatrix->operator=(static_cast<CPUSparseMatrix<ElemType>&&>(m_CPUSparseMatrix->ColumnSlice(startColumn, numCols)));
            else
                slice.m_CPUSparseMatrix = make_shared<CPUSparseMatrix<ElemType>>(static_cast<CPUSparseMatrix<ElemType>&&>(m_CPUSparseMatrix->ColumnSlice(startColumn, numCols)));
        }
        if (GetCurrentMatrixLocation() == GPU || GetCurrentMatrixLocation() == BOTH)
        {
            if (slice.m_GPUSparseMatrix)
                slice.m_GPUSparseMatrix->operator=(static_cast<GPUSparseMatrix<ElemType>&&>(m_GPUSparseMatrix->ColumnSlice(startColumn, numCols)));
            else
                slice.m_GPUSparseMatrix = make_shared<GPUSparseMatrix<ElemType>>(static_cast<GPUSparseMatrix<ElemType>&&>(m_GPUSparseMatrix->ColumnSlice(startColumn, numCols)));
        }
    }
    else
        LogicError("Undetermined matrix type");

    // update the slice's m_currentDataLocation, m_matrixType, and m_baseMatrix
    // This will work for CPU, GPU, and BOTH.
    slice.SetDataLocation(GetCurrentMatrixLocation(), GetMatrixType());

    return slice;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignColumnSlice(const Matrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols)
{
    ReleaseMemory();
    m_preferredDeviceId = fromMatrix.m_preferredDeviceId;

    DISPATCH_MATRIX_ON_FLAG(&fromMatrix,
                            this,
                            if (m_CPUMatrix) m_CPUMatrix->AssignColumnSlice(*fromMatrix.m_CPUMatrix, startColumn, numCols);
                            else m_CPUMatrix = make_shared<CPUMatrix<ElemType>>(fromMatrix.m_CPUMatrix->ColumnSlice(startColumn, numCols)),

                            if (m_GPUMatrix) m_GPUMatrix->AssignColumnSlice(*fromMatrix.m_GPUMatrix, startColumn, numCols);
                            else m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(fromMatrix.m_GPUMatrix->ColumnSlice(startColumn, numCols)),

                            NOT_IMPLEMENTED,

                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::SetColumnSlice(const Matrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols)
{
    assert(m_CPUMatrix || m_GPUMatrix);
    // must already been allocated

    DISPATCH_MATRIX_ON_FLAG(&fromMatrix,
                            this,
                            m_CPUMatrix->SetColumnSlice(*fromMatrix.m_CPUMatrix, startColumn, numCols),
                            m_GPUMatrix->SetColumnSlice(*fromMatrix.m_GPUMatrix, startColumn, numCols),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
void Matrix<ElemType>::CopyColumnsStrided(const Matrix<ElemType>& fromMatrix, size_t numCols, size_t srcNumColsStride, size_t destNumColsStride)
{
    assert(m_CPUMatrix || m_GPUMatrix);

    DISPATCH_MATRIX_ON_FLAG(&fromMatrix,
                            this,
                            m_CPUMatrix->CopyColumnsStrided(*fromMatrix.m_CPUMatrix, numCols, srcNumColsStride, destNumColsStride),
                            m_GPUMatrix->CopyColumnsStrided(*fromMatrix.m_GPUMatrix, numCols, srcNumColsStride, destNumColsStride),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::Diagonal() const
{
    int devId = GetDeviceId();

    Matrix<ElemType> diag(matrixFlagDontOwnBuffer, (DEVICEID_TYPE) devId);
    diag.m_preferredDeviceId = m_preferredDeviceId;

    AssignDiagonalValuesTo(diag);

    return diag;
}

template <class ElemType>
void Matrix<ElemType>::AssignDiagonalValuesTo(Matrix<ElemType>& diag) const
{
    int devId = GetDeviceId();
    DecideAndMoveToRightDevice(*this, diag);

    if (GetMatrixType() == MatrixType::DENSE)
    {
        if (devId == CPUDEVICE)
        {
            if (diag.m_CPUMatrix)
                diag.m_CPUMatrix->operator=(static_cast<CPUMatrix<ElemType>&&>(m_CPUMatrix->Diagonal()));
            else
                diag.m_CPUMatrix = make_shared<CPUMatrix<ElemType>>(static_cast<CPUMatrix<ElemType>&&>(m_CPUMatrix->Diagonal()));
            diag.SetDataLocation(CPU, DENSE);
        }
        else
        {
            if (diag.m_GPUMatrix)
                diag.m_GPUMatrix->operator=(static_cast<GPUMatrix<ElemType>&&>(m_GPUMatrix->Diagonal()));
            else
                diag.m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(static_cast<GPUMatrix<ElemType>&&>(m_GPUMatrix->Diagonal()));
            diag.SetDataLocation(GPU, DENSE);
        }
    }
    else if (GetMatrixType() == MatrixType::SPARSE)
    {
        // TODO: Implement optimized diagonal functions for sparse matrices. For now use the DiagonalToDense instead.
        if (devId == CPUDEVICE)
        {
            if (diag.m_CPUMatrix)
                diag.m_CPUMatrix->operator=(static_cast<CPUMatrix<ElemType>&&>(m_CPUSparseMatrix->DiagonalToDense()));
            else
                diag.m_CPUMatrix = make_shared<CPUMatrix<ElemType>>(static_cast<CPUMatrix<ElemType>&&>(m_CPUSparseMatrix->DiagonalToDense()));
            diag.SetDataLocation(CPU, DENSE);
        }
        else
        {
            if (diag.m_GPUMatrix)
                diag.m_GPUMatrix->operator=(static_cast<GPUMatrix<ElemType>&&>(m_GPUSparseMatrix->DiagonalToDense()));
            else
                diag.m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(static_cast<GPUMatrix<ElemType>&&>(m_GPUSparseMatrix->DiagonalToDense()));
            diag.SetDataLocation(GPU, DENSE);
        }
    }
    else
        LogicError("Undetermined matrix type");

}

// This function will change the matrix type between DENSE and SPARSE.
// WARNING: The correct implementation is to copy the matrix between DENSE and SPARSE
//         However, the conversion functions are not implemented yet and so it will always create
//         a new blank matrix and destroy all info in the original matrix if different matrix type is asked.
// In case of !keepValues, the matrix content will be undefined.
template <class ElemType>
void Matrix<ElemType>::SwitchToMatrixType(MatrixType newMatrixType, MatrixFormat newMatrixFormat, bool keepValues)
{
    // This check should be uncommented but unfortunately there are still places
    // this function is being called with incorrect "default" format value
    /*if (m_matrixType == newMatrixType && GetFormat() != newMatrixFormat)
            NOT_IMPLEMENTED;*/

    if (m_matrixType == newMatrixType)
        return;

    if (!m_baseMatrix)
        keepValues = false;

#define NUM_MATRIXTYPE_CHANGED_WARN 20
    m_numTimesMatrixTypeChanged++;

    if ((GetMathLibTraceLevel() > 0) && (m_numTimesMatrixTypeChanged == NUM_MATRIXTYPE_CHANGED_WARN))
        fprintf(stderr, "WARNING: The same matrix with dim [%lu, %lu] has been transferred between different devices for %d times.\n", (unsigned long) GetNumRows(), (unsigned long) GetNumCols(), NUM_MATRIXTYPE_CHANGED_WARN);

    if (GetDeviceId() < 0) // CPU
    {
        if (newMatrixType == MatrixType::SPARSE)
        {
            if (!m_baseMatrix)
                m_CPUSparseMatrix = make_shared<CPUSparseMatrix<ElemType>>(newMatrixFormat);
            else
                m_CPUSparseMatrix = make_shared<CPUSparseMatrix<ElemType>>(newMatrixFormat, GetNumRows(), GetNumCols(), 1);

            if (keepValues)
                CopyElementsFromDenseToSparse(*m_CPUMatrix, *m_CPUSparseMatrix);

            SetDataLocation(CPU, SPARSE);
            m_CPUMatrix = nullptr;
        }
        else if (newMatrixType == MatrixType::DENSE)
        {
            if (!m_baseMatrix)
                m_CPUMatrix = make_shared<CPUMatrix<ElemType>>();
            else
                m_CPUMatrix = make_shared<CPUMatrix<ElemType>>(GetNumRows(), GetNumCols());

            if (keepValues)
                m_CPUMatrix->SetValue(m_CPUSparseMatrix->CopyColumnSliceToDense(0, GetNumCols()));

            SetDataLocation(CPU, DENSE);
            m_CPUSparseMatrix = nullptr;
        }
        else
            LogicError("SwitchToMatrixType: Unexpected/invalid new matrix type");
    }
    else // GPU
    {
        if (newMatrixType == MatrixType::SPARSE)
        {
            if (!m_baseMatrix)
                m_GPUSparseMatrix = make_shared<GPUSparseMatrix<ElemType>>(GetDeviceId(), newMatrixFormat);
            else
                m_GPUSparseMatrix = make_shared<GPUSparseMatrix<ElemType>>(GetNumRows(), GetNumCols(), 0, GetDeviceId(), newMatrixFormat);

            if (keepValues)
                m_GPUSparseMatrix->SetValue(*m_GPUMatrix);

            SetDataLocation(GPU, SPARSE);
            m_GPUMatrix = nullptr;
        }
        else if (newMatrixType == MatrixType::DENSE)
        {
            if (!m_baseMatrix)
                m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(GetDeviceId());
            else
                m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(GetNumRows(), GetNumCols(), GetDeviceId());

            if (keepValues)
                m_GPUSparseMatrix->CopyToDenseMatrix(*m_GPUMatrix);

            SetDataLocation(GPU, DENSE);
            m_GPUSparseMatrix = nullptr;
        }
        else
            LogicError("SwitchToMatrixType: Unexpected/invalid new matrix type");
    }
}

template <class ElemType>
void Matrix<ElemType>::CopyElementsFromDenseToSparse(CPUMatrix<ElemType>& from, CPUSparseMatrix<ElemType>& dest)
{
    foreach_coord (row, col, from)
    {
        auto val = from(row, col);
        dest.SetValue(row, col, val);
    }
}

template <class ElemType>
ElemType Matrix<ElemType>::Get00Element() const
{
    DISPATCH_MATRIX_ON_FLAG(this, nullptr,
        { return m_CPUMatrix->Get00Element(); },
        { return m_GPUMatrix->Get00Element(); },
        { NOT_IMPLEMENTED; },
        { NOT_IMPLEMENTED; });
}

// const operator(,)
template <class ElemType>
const ElemType Matrix<ElemType>::operator()(const size_t row, const size_t col) const
{
    DISPATCH_MATRIX_ON_FLAG_USECPU_4BOTH(this, nullptr,
        { return m_CPUMatrix->operator()(row, col); },
        { _transferFromDeviceToDevice(GetDeviceId(), CPUDEVICE, false); return m_CPUMatrix->operator()(row, col); },
        { NOT_IMPLEMENTED; },
        { NOT_IMPLEMENTED; });
}

// non-const operator(,)
//WARNING: This function is very slow for GPUs since it requires copying values between CPUs and GPUs.
//In addition, if ColumnSlice is used after this function but before the values are copied back to GPU
//the operation will fail since the memory is not managed by the slice.
// If you don't need to modify the values, to call the const version above, or GetValue(row,col) which does that for you unambiguously.
// TODO: Can we remove this, and have users use SetValue() instead? To avoid this potential error?
template <class ElemType>
ElemType& Matrix<ElemType>::operator()(const size_t row, const size_t col)
{
    DISPATCH_MATRIX_ON_FLAG_USECPU_4BOTH(this, nullptr,
        { return m_CPUMatrix->operator()(row, col); },
        {
            _transferFromDeviceToDevice(GetDeviceId(), CPUDEVICE, false);
            SetDataLocation(CPU, DENSE);
            return m_CPUMatrix->operator()(row, col);
        },
        { NOT_IMPLEMENTED; },
        { NOT_IMPLEMENTED; });
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::Transpose()
{
    if (IsEmpty())
        LogicError("Transpose: Matrix is empty.");

    Matrix<ElemType> c(GetNumCols(), GetNumRows(), (DEVICEID_TYPE) GetDeviceId(), this->GetMatrixType(), this->GetFormat());
    c.AssignTransposeOf(*this);
    return c;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignTransposeOf(const Matrix<ElemType>& a)
{
    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a, this,
        { m_CPUMatrix->AssignTransposeOf(*a.m_CPUMatrix); },
        { m_GPUMatrix->AssignTransposeOf(*a.m_GPUMatrix); },
        { NOT_IMPLEMENTED; },
        { m_GPUSparseMatrix->AssignTransposeOf(*a.m_GPUSparseMatrix); });

    return *this;
}

// *this[:,j] = a[:,idx[j]] * alpha + *this[:,j] * beta
// idx has width of 'this' and contains values w.r.t. 'a'
// Invalid entries (gap columns) are denoted by idx(0,j) == -1.
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::DoGatherColumnsOf(ElemType beta, const Matrix<ElemType>& idx, const Matrix<ElemType>& a, ElemType alpha)
{
    DecideAndMoveToRightDevice(*this, idx, a); // TODO: only move target if beta != 0

    if (a.GetMatrixType() != this->GetMatrixType())
        RuntimeError("Matrix::DoGatherColumnsOf: The source and target matrices must have same storage type (SPARSE/DENSE).");

    DISPATCH_MATRIX_ON_FLAG(&a, this,
        { m_CPUMatrix->DoGatherColumnsOf(beta, *idx.m_CPUMatrix, *a.m_CPUMatrix, alpha); },
        { m_GPUMatrix->DoGatherColumnsOf(beta, *idx.m_GPUMatrix, *a.m_GPUMatrix, alpha); },
        { m_CPUSparseMatrix->DoGatherColumnsOf(beta, *idx.m_CPUMatrix, *a.m_CPUSparseMatrix, alpha); },
        { 
            // TODO replace by more performant version directly on GPU that does not require the round-trip over CPU.

            Matrix<ElemType> tempIdx(CPUDEVICE); tempIdx.AssignValuesOf(idx);

            CPUSparseMatrix<ElemType> tempA(a.GetFormat(), a.GetNumRows(), a.GetNumCols(), a.m_GPUSparseMatrix->GetNumNZElements());
            a.m_GPUSparseMatrix->CopyToCPUSparseMatrix(tempA);

            CPUSparseMatrix<ElemType> tempThis(m_GPUSparseMatrix->GetFormat(), m_GPUSparseMatrix->GetNumRows(), m_GPUSparseMatrix->GetNumCols(), m_GPUSparseMatrix->GetNumNZElements());
            m_GPUSparseMatrix->CopyToCPUSparseMatrix(tempThis);

            tempThis.DoGatherColumnsOf(beta, *tempIdx.m_CPUMatrix, tempA, alpha);
            m_GPUSparseMatrix->SetValue(tempThis);
        });

    return *this;
}

// *this[:,idx[j]] = a[:,j] * alpha + *this[:,idx[j]] * beta
// idx has width of 'a' and contains values w.r.t. 'this'
// Unlike gather, for scatter, 'this' must have been sized already.
// Invalid entries (gap columns) are denoted by idx(0,j) == -1.
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::DoScatterColumnsOf(ElemType beta, const Matrix<ElemType>& idx, const Matrix<ElemType>& a, ElemType alpha)
{
    DecideAndMoveToRightDevice(*this, idx, a); // TODO: only move target if beta != 0

    if (a.GetMatrixType() != this->GetMatrixType())
        RuntimeError("Matrix::DoScatterColumnsOf: The source and target matrices must have same storage type (SPARSE/DENSE).");

    DISPATCH_MATRIX_ON_FLAG(&a, this,
        { m_CPUMatrix->DoScatterColumnsOf(beta, *idx.m_CPUMatrix, *a.m_CPUMatrix, alpha); },
        { m_GPUMatrix->DoScatterColumnsOf(beta, *idx.m_GPUMatrix, *a.m_GPUMatrix, alpha); },
        { m_CPUSparseMatrix->DoScatterColumnsOf(beta, *idx.m_CPUMatrix, *a.m_CPUSparseMatrix, alpha); },
        { 
            // TODO replace by more performant version directly on GPU that does not require the round-trip over CPU.

            Matrix<ElemType> tempIdx(CPUDEVICE); tempIdx.AssignValuesOf(idx);

            CPUSparseMatrix<ElemType> tempA(a.GetFormat(), a.GetNumRows(), a.GetNumCols(), a.m_GPUSparseMatrix->GetNumNZElements());
            a.m_GPUSparseMatrix->CopyToCPUSparseMatrix(tempA);

            CPUSparseMatrix<ElemType> tempThis(m_GPUSparseMatrix->GetFormat(), m_GPUSparseMatrix->GetNumRows(), m_GPUSparseMatrix->GetNumCols(), m_GPUSparseMatrix->GetNumNZElements());
            m_GPUSparseMatrix->CopyToCPUSparseMatrix(tempThis);

            tempThis.DoScatterColumnsOf(beta, *tempIdx.m_CPUMatrix, tempA, alpha);
            m_GPUSparseMatrix->SetValue(tempThis);
        });

    return *this;
}

// set all elements of a matrix to a scalar value
// For sparse matrices, the only allowed value is 0.
template <class ElemType>
void Matrix<ElemType>::SetValue(const ElemType v)
{
    if (IsEmpty()) // if empty then we are done
        return;

    if (v == 0 && GetMatrixType() == MatrixType::SPARSE) // if sparse, setting it to 0 is special
    {
        Reset();
        return;
    }

    DISPATCH_MATRIX_ON_FLAG(this, this,
        { m_CPUMatrix->SetValue(v); },
        { m_GPUMatrix->SetValue(v); },
        { NOT_IMPLEMENTED; },
        { NOT_IMPLEMENTED; });
}

template <class ElemType>
void Matrix<ElemType>::SetValue(const DeviceBoundNumber<ElemType>& db_number)
{
    if (IsEmpty()) // if empty then we are done
        return;

    DISPATCH_MATRIX_ON_FLAG(this, this,
        { m_CPUMatrix->SetValue(*db_number.ExposePointer2Value()); },
        {
            if (GetDeviceId() != db_number.GetDeviceId())
            RuntimeError("Matrix and device bound number must be on the same device");
            m_GPUMatrix->SetValue(db_number.ExposePointer2Value());
        },
        { NOT_IMPLEMENTED; },
        { NOT_IMPLEMENTED; });
}

template <>
/*static*/ float Matrix<float>::MakeNan(size_t /*payload*/)
{
    return nanf("");
}
template <>
/*static*/ double Matrix<double>::MakeNan(size_t /*payload*/)
{
    return nan("");
}
template <>
/*static*/ char Matrix<char>::MakeNan(size_t)
{
    return 0;
} // (needed for completeness and to pass unit tests)
template <>
/*static*/ short Matrix<short>::MakeNan(size_t)
{
    return 0;
} // (needed for completeness and to pass unit tests)

template <class ElemType>
void Matrix<ElemType>::MaskColumnsValue(const Matrix<char>& columnsMask, ElemType val, size_t numColsPerMaskEntry)
{
    if (GetNumCols() != (columnsMask.GetNumCols() * numColsPerMaskEntry))
        RuntimeError("MaskColumnsValue: Matrix number of columns must equal [column mask * numColsPerMaskEntry].");

    if (GetCurrentMatrixLocation() == CPU && (columnsMask.GetCurrentMatrixLocation() == CPU || columnsMask.GetCurrentMatrixLocation() == BOTH))
        ; // OK
    else if (GetDeviceId() != columnsMask.GetDeviceId() && columnsMask.GetCurrentMatrixLocation() != BOTH)
        RuntimeError("MaskColumnsValue: Matrix and column mask must be on the same device.");

    DISPATCH_MATRIX_ON_FLAG(this, this,
        { m_CPUMatrix->MaskColumnsValue(*columnsMask.m_CPUMatrix, val, numColsPerMaskEntry); },
        { m_GPUMatrix->MaskColumnsValue(*columnsMask.m_GPUMatrix, val, numColsPerMaskEntry); },
        { m_CPUSparseMatrix->MaskColumnsValue(*columnsMask.m_CPUMatrix, val, numColsPerMaskEntry); },
        { m_GPUSparseMatrix->MaskColumnsValue(*columnsMask.m_GPUMatrix, val, numColsPerMaskEntry); });
}

template <class ElemType>
void Matrix<ElemType>::SetColumn(const ElemType* colPointer, size_t colInd)
{
    if (colPointer == nullptr)
        InvalidArgument("SetColumn: colPointer is null.");

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->SetColumn(colPointer, colInd),
                            m_GPUMatrix->SetColumn(colPointer, colInd),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::SetColumn(const ElemType val, size_t colInd)
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->SetColumn(val, colInd),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::SetColumn(const Matrix<ElemType>& colMat, size_t colInd)
{
    DecideAndMoveToRightDevice(*this, colMat);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->SetColumn(*colMat.m_CPUMatrix, colInd),
                            m_GPUMatrix->SetColumn(*colMat.m_GPUMatrix, colInd),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::SetValue(const Matrix<ElemType>& deepCopyFrom)
{
    if (this == &deepCopyFrom)
        return;

    m_preferredDeviceId = deepCopyFrom.m_preferredDeviceId;
    DecideAndMoveToRightDevice(deepCopyFrom, *this);
    SwitchToMatrixType(deepCopyFrom.GetMatrixType(), deepCopyFrom.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&deepCopyFrom, this,
        { m_CPUMatrix->SetValue(*deepCopyFrom.m_CPUMatrix); },
        { m_GPUMatrix->SetValue(*deepCopyFrom.m_GPUMatrix); },
        { m_CPUSparseMatrix->SetValue(*deepCopyFrom.m_CPUSparseMatrix); },
        { m_GPUSparseMatrix->SetValue(*deepCopyFrom.m_GPUSparseMatrix); });
}

template <class ElemType>
void Matrix<ElemType>::AssignValuesOf(const Matrix<ElemType>& deepCopyFrom)
{
    if (this == &deepCopyFrom)
        return;

    // TODO: do we need all these 'this->'?
    DISPATCH_MATRIX_ON_FLAG(this, this,
        { 
            // Set CPUMatrix from:
            DISPATCH_MATRIX_ON_FLAG(&deepCopyFrom, nullptr,
                { m_CPUMatrix->SetValue(*deepCopyFrom.m_CPUMatrix); },
                { this->Resize(deepCopyFrom.GetNumRows(), deepCopyFrom.GetNumCols()); deepCopyFrom.CopySection(deepCopyFrom.GetNumRows(), deepCopyFrom.GetNumCols(), m_CPUMatrix->Data(), this->GetNumRows()); },
                { deepCopyFrom.m_CPUSparseMatrix->AssignColumnSliceToDense(*m_CPUMatrix, 0, deepCopyFrom.GetNumCols()); },
                { CPUSparseMatrix<ElemType> tempCPUSparseMatrix(deepCopyFrom.GetFormat(), deepCopyFrom.GetNumRows(), deepCopyFrom.GetNumCols(), deepCopyFrom.m_GPUSparseMatrix->GetNumNZElements()); deepCopyFrom.m_GPUSparseMatrix->CopyToCPUSparseMatrix(tempCPUSparseMatrix); tempCPUSparseMatrix.AssignColumnSliceToDense(*m_CPUMatrix, 0, deepCopyFrom.GetNumCols()); });
        },
        { 
            // Set GPUMatrix from:
            DISPATCH_MATRIX_ON_FLAG(&deepCopyFrom, nullptr,
                { m_GPUMatrix->SetValue(deepCopyFrom.GetNumRows(), deepCopyFrom.GetNumCols(), this->GetDeviceId(), deepCopyFrom.m_CPUMatrix->Data()); },
                { m_GPUMatrix->SetValue(*deepCopyFrom.m_GPUMatrix); },
                {
                    CPUMatrix<ElemType> tempCPUDenseMatrix(deepCopyFrom.GetNumRows(), deepCopyFrom.GetNumCols());
                    deepCopyFrom.m_CPUSparseMatrix->AssignColumnSliceToDense(tempCPUDenseMatrix, 0, deepCopyFrom.GetNumCols());
                    m_GPUMatrix->SetValue(deepCopyFrom.GetNumRows(), deepCopyFrom.GetNumCols(), this->GetDeviceId(), tempCPUDenseMatrix.Data());
                },//{ m_GPUMatrix->SetValue(*deepCopyFrom.m_CPUSparseMatrix); },
                { deepCopyFrom.m_GPUSparseMatrix->AssignColumnSliceToDense(*m_GPUMatrix, 0, deepCopyFrom.GetNumCols()); });
        },
        { 
            // Set CPUSparseMatrix from:
            DISPATCH_MATRIX_ON_FLAG(&deepCopyFrom, nullptr,
                { auto matrixType = GetMatrixType(); auto matrixFormat = GetFormat(); *this = deepCopyFrom.DeepClone(); SwitchToMatrixType(matrixType, matrixFormat, true); },
                { LogicError("AssignValuesOf: Assigning a GPUMatrix to a CPUSparseMatrix is not yet implemented."); },//{ m_CPUSparseMatrix->SetValue(*deepCopyFrom.m_GPUMatrix); },
                { m_CPUSparseMatrix->SetValue(*deepCopyFrom.m_CPUSparseMatrix); },
                { LogicError("AssignValuesOf: Assigning a GPUSparseMatrix to a CPUSparseMatrix is not yet implemented."); });//{ m_CPUSparseMatrix->SetValue(*deepCopyFrom.m_GPUSparseMatrix); });
        },
        { 
            // Set GPUSparseMatrix from:
            DISPATCH_MATRIX_ON_FLAG(&deepCopyFrom, nullptr,
                { Matrix<ElemType> tempCPUSparseMatrix(deepCopyFrom.DeepClone()); tempCPUSparseMatrix.SwitchToMatrixType(GetMatrixType(), GetFormat(), true); m_GPUSparseMatrix->SetValue(*tempCPUSparseMatrix.m_CPUSparseMatrix); },
                { m_GPUSparseMatrix->SetValue(*deepCopyFrom.m_GPUMatrix); },
                { m_GPUSparseMatrix->SetValue(*deepCopyFrom.m_CPUSparseMatrix); },
                { m_GPUSparseMatrix->SetValue(*deepCopyFrom.m_GPUSparseMatrix); });
        });
}

// CastAssignValuesOf() -- assign a matrix with type conversion, needed for feeding 'float' data to 'double' inputs in V2
// This version is a stop-gap for debugging and testing. If any conversion is done, it will be slow.
// If this is ever used for something that needs performance, it should not be too hard (but labor) to implement this efficiently.
static void DoCastAssignValuesOf(Matrix<float>&  target, const Matrix<float>&  other) { target.AssignValuesOf(other); }
static void DoCastAssignValuesOf(Matrix<double>& target, const Matrix<double>& other) { target.AssignValuesOf(other); }
template<class ElemType>
static void CopyToVector(const Matrix<ElemType>& source, vector<ElemType>& sourceData)
{
    sourceData.resize(source.GetNumElements());
    ElemType* datap = sourceData.data();
    size_t datasz = sourceData.size();
    source.CopyToArray(datap, datasz);
    assert(datap == sourceData.data() && datasz == sourceData.size()); // (make sure it used my buffer; a somewhat awkward API)
}

template<>
void Matrix<int>::AssignValuesOf(const Matrix<int>&) { NOT_IMPLEMENTED; }
template<class ElemType, class ElemTypeOther>
static void DoCastAssignValuesOf(Matrix<ElemType>& target, const Matrix<ElemTypeOther>& source)
{
    target; source;
    // this is implemented in a rather tedious way:
    //  - copy to a CPU-side STL vector
    //  - type-cast
    //  - copy to target
    vector<ElemTypeOther> sourceData;
    if (source.GetMatrixType() == MatrixType::SPARSE) // if sparse then convert it over (V2 cannot read sparse data into dense input_variables)
    {
        Matrix<ElemTypeOther> temp(source.GetNumRows(), source.GetNumCols(), CPUDEVICE, DENSE);
        temp.AssignValuesOf(source);
        CopyToVector(temp, sourceData);
    }
    else
    {
        CopyToVector(source, sourceData);
    }
    // cast all values
    vector<ElemType> targetData(sourceData.size());
    transform(sourceData.begin(), sourceData.end(), targetData.begin(), [](ElemTypeOther v){ return (ElemType)v; });
    // set the target
    if (target.GetMatrixType() == MatrixType::SPARSE) // if target is sparse then we cannot assign from a vector directly, but we can from a matrix object
    {
        Matrix<ElemType> temp(source.GetNumRows(), source.GetNumCols(), targetData.data(), CPUDEVICE);
        target.AssignValuesOf(temp);
    }
    else
    {
        target.SetValue(source.GetNumRows(), source.GetNumCols(), target.GetDeviceId(), targetData.data());
    }
}

template <class ElemType>
void Matrix<ElemType>::CastAssignValuesOf(const MatrixBase& other) /*override*/ // allows for mixed assignment with conversion
{
    const Matrix<float> * otherf = dynamic_cast<const Matrix<float>*>(&other);
    if (otherf)
        return DoCastAssignValuesOf(*this, *otherf);
    const Matrix<double> * otherd = dynamic_cast<const Matrix<double>*>(&other);
    if (otherd)
        return DoCastAssignValuesOf(*this, *otherd);
    LogicError("CastAssignValuesOf: Only accepts float and double matrices.");
}

template<>
void Matrix<int>::SetValue(const size_t, const size_t, int, int*, const size_t, DataTransferer*) { NOT_IMPLEMENTED; }
template <class ElemType>
void Matrix<ElemType>::SetValue(const size_t numRows, const size_t numCols, int deviceId, ElemType* pArray, const size_t matrixFlags, DataTransferer* transferer)
{
    if (((numRows * numCols) > 0) && (pArray == nullptr))
        InvalidArgument("Invalid pArray.");

    // Only gpu matrix supports async data transfers, so data transferer passed only to gpu matrix.
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->SetValue(numRows, numCols, pArray, matrixFlags),
                            m_GPUMatrix->SetValue(numRows, numCols, deviceId, pArray, matrixFlags, transferer),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::SetValue(const size_t rIdx, const size_t cIdx, ElemType val)
{
    DISPATCH_MATRIX_ON_FLAG_USECPU_4BOTH(this,
                                         this,
                                         (*m_CPUMatrix)(rIdx, cIdx) = val,
                                         NOT_IMPLEMENTED,
                                         m_CPUSparseMatrix->SetValue(rIdx, cIdx, val),
                                         NOT_IMPLEMENTED);
}

// read features
template <class ElemType>
void Matrix<ElemType>::SetMatrixFromCSCFormat(const CPUSPARSE_INDEX_TYPE* h_CSCCol, const CPUSPARSE_INDEX_TYPE* h_Row, const ElemType* h_Val,
    const size_t nz, const size_t numRows, const size_t numCols, DataTransferer* transferer)
{
    // Note: The current implementation uses the xPUSparseMatrix as temporary space. This allows for memory sharing between calls. If
    // xPUSparseMatrix is a view, this code will cause an error during runtime stating that the view is not writable nor resizable.

    // Only gpu matrix supports async data transfers, so data transferer passed only to gpu matrix in case we do not need to reassign to dense.
    // When we have to reassign sparse to dense we cannot use async operation, because at the time when AssignColumnSliceToDense is called the
    // data should already be copied to destination.
    DISPATCH_MATRIX_ON_FLAG(this, this,
        {
            if (!m_CPUSparseMatrix) m_CPUSparseMatrix = make_shared<CPUSparseMatrix<ElemType>>(matrixFormatSparseCSC, numRows, numCols, nz);
            m_CPUSparseMatrix->SetMatrixFromCSCFormat(h_CSCCol, h_Row, h_Val, nz, numRows, numCols);
            m_CPUSparseMatrix->AssignColumnSliceToDense(*m_CPUMatrix, 0, numCols);
        },
        {
            if (!m_GPUSparseMatrix) m_GPUSparseMatrix = make_shared<GPUSparseMatrix<ElemType>>(numRows, numCols, nz, GetDeviceId(), matrixFormatSparseCSC);
            m_GPUSparseMatrix->SetMatrixFromCSCFormat(h_CSCCol, h_Row, h_Val, nz, numRows, numCols);
            m_GPUSparseMatrix->AssignColumnSliceToDense(*m_GPUMatrix, 0, numCols);
        },
        { m_CPUSparseMatrix->SetMatrixFromCSCFormat(h_CSCCol, h_Row, h_Val, nz, numRows, numCols); },
        { m_GPUSparseMatrix->SetMatrixFromCSCFormat(h_CSCCol, h_Row, h_Val, nz, numRows, numCols, false, -1, transferer); });
}

template <class ElemType>
void Matrix<ElemType>::SetDiagonalValue(const ElemType v)
{
    if (IsEmpty())
        LogicError("SetDiagonalValue: Matrix is empty.");

    if (GetNumRows() != GetNumCols())
        LogicError("SetDiagonalValue: NumRows and NumCols do not agree.");

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->SetDiagonalValue(v),
                            m_GPUMatrix->SetDiagonalValue(v),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::SetDiagonalValue(const Matrix<ElemType>& vector)
{
    if (GetNumRows() != GetNumCols())
        LogicError("SetDiagonalValue: NumRows and NumCols do not agree.");

    if (vector.GetNumRows() != 1 && vector.GetNumCols() != 1)
        LogicError("SetDiagonalValue: Input vector must be a vector.");

    if (vector.GetNumRows() * vector.GetNumCols() != GetNumRows())
        LogicError("SetDiagonalValue: Input vector must match matrix dimension.");

    if (IsEmpty())
        return;

    DecideAndMoveToRightDevice(*this, vector);

    if (vector.GetNumElements() == 1) // reduce to simple form
    {
        DISPATCH_MATRIX_ON_FLAG(&vector,
                                nullptr,
                                SetDiagonalValue(vector(0, 0)),
                                SetDiagonalValue(vector.m_GPUMatrix->Get00Element()), // BUGBUG: efficiency
                                SetDiagonalValue(vector(0, 0)),
                                SetDiagonalValue(vector.m_GPUMatrix->Get00Element()) // BUGBUG: efficiency
                                );
    }
    else if (vector.GetNumRows() != GetNumRows() && vector.GetNumCols() != GetNumRows())
        LogicError("SetDiagonalValue: input vector's dimension does not agree with [this].");
    else
    {
        // WARNING: we use this pointer to decide which function to call. However, vector may be stored in a different matrix type (DENSE, SPARSE)
        DISPATCH_MATRIX_ON_FLAG(this,
                                this,
                                assert(vector.m_CPUMatrix);
                                m_CPUMatrix->SetDiagonalValue(*vector.m_CPUMatrix),
                                assert(vector.m_GPUMatrix);
                                m_GPUMatrix->SetDiagonalValue(*vector.m_GPUMatrix),
                                NOT_IMPLEMENTED,
                                NOT_IMPLEMENTED);
    }
}

template <class ElemType>
void Matrix<ElemType>::SetUniformRandomValue(const ElemType low, const ElemType high, unsigned long seed)
{
    if (IsEmpty())
        return;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->SetUniformRandomValue(low, high, seed),
                            m_GPUMatrix->SetUniformRandomValue(low, high, seed),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::SetGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed)
{
    if (sigma <= 0)
        InvalidArgument("SetUniformRandomValue: sigma must be a positive value.");

    if (IsEmpty())
        return;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->SetGaussianRandomValue(mean, sigma, seed),
                            m_GPUMatrix->SetGaussianRandomValue(mean, sigma, seed),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::AddGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed)
{
    if (sigma <= 0)
        InvalidArgument("SetUniformRandomValue: sigma must be a positive value.");

    if (IsEmpty())
        return;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AddGaussianRandomValue(mean, sigma, seed),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

//maskRate: percentage of values masked out (similar to dropout rate)
//scaleValue: which scale value to set to the left ones (unmasked items).
template <class ElemType>
void Matrix<ElemType>::SetUniformRandomMask(const ElemType maskRate, const ElemType scaleValue, RNGHandle& rngHandle)
{
    if (IsEmpty())
        return;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->SetUniformRandomMask(maskRate, scaleValue, rngHandle),
                            m_GPUMatrix->SetUniformRandomMask(maskRate, scaleValue, rngHandle),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

// Vanilla SGD update. 
// Modifies "this" parameter matrix, on which this method is invoked.
template <class ElemType>
void Matrix<ElemType>::SGDUpdate(Matrix<ElemType>& gradients, ElemType learnRatePerSample)
{
    DecideAndMoveToRightDevice(gradients, *this);

    DISPATCH_MATRIX_ON_FLAG(&gradients, nullptr,
    { 
        // w_t = w_{t-1} - learnRatePerSample * g_{t-1},
        ScaleAndAdd(ElemType(-learnRatePerSample), gradients, *this);
    },
    { 
        // BUGBUG: cannot call ScaleAndAdd(ElemType(-learnRatePerSample), gradients, *this) here,
        // it produces different results from the scale and add below.
        // g'_{t-1} = learnRatePerSample * g_{t-1}
        // w_t = w_{t-1} - g'_{t-1}
        Scale(ElemType(learnRatePerSample), gradients);
        *this -= gradients;
    },
    { 
        ScaleAndAdd(ElemType(-learnRatePerSample), gradients, *this);
    },
    { 
        ScaleAndAdd(ElemType(-learnRatePerSample), gradients, *this);
    });
    
}

// SGD update with momentum.
// Modifies "this" parameter matrix, on which this method is invoked.
template <class ElemType>
void Matrix<ElemType>::MomentumSGDUpdate(Matrix<ElemType>& gradients,
                                         Matrix<ElemType>& smoothedGradients,
                                         ElemType learnRatePerSample,
                                         ElemType momentum,
                                         bool unitGainMomentum)
{
    DecideAndMoveToRightDevice(smoothedGradients, gradients, *this);

    const auto unitGainFactor = ElemType(unitGainMomentum ? (1.0 - momentum) : 1.0);

    DISPATCH_MATRIX_ON_FLAG(&gradients, nullptr,
        { 
            // Classic momentum (unitGainFactor == 1.0):
            // 1) sg_t = momentum * sg_{t-1} + learnRatePerSample * g_{t-1}
            // Unit-gain momentum (unitGainFactor == 1.0 - momentum):
            // 1) sg_t = momentum * sg_{t-1} + learnRatePerSample * (1.0 - momentum) * g_{t-1}
            // 2) w_t = w_{t-1} - sg_t
            ScaleAndAdd(unitGainFactor * learnRatePerSample, gradients, momentum, smoothedGradients);
            *this -= smoothedGradients;
        },
        { 
            ScaleAndAdd(unitGainFactor * learnRatePerSample, gradients, momentum, smoothedGradients);
            *this -= smoothedGradients;
        },
        { 
            // The sparse update is slightly different from the dense implementation above:
            // Classic momentum (unitGainFactor == 1.0):
            // 1) sg_t = momentum * sg_{t-1} + g_{t-1}
            // Unit-gain momentum (unitGainFactor == 1.0 - momentum):
            // 1) sg_t = momentum * sg_{t-1} + (1.0 - momentum) * g_{t-1}
            // 2) g'_{t-1} = sg_t
            // 3) w_t = w_{t-1} - learnRatePerSample * g'_{t-1}
            if (momentum != 0)
            {
                gradients.m_CPUSparseMatrix->NormalGrad(*smoothedGradients.m_CPUMatrix, momentum, unitGainMomentum);
            }
            ScaleAndAdd(-learnRatePerSample, gradients, *this);
        },
        { 
            if (momentum != 0)
            {
                gradients.m_GPUSparseMatrix->NormalGrad(*smoothedGradients.m_GPUMatrix, momentum, unitGainMomentum);
            }
            ScaleAndAdd(-learnRatePerSample, gradients, *this);
        });
}

// Nesterov accelerated SGD update.
// Modifies "this" parameter matrix, on which this method is invoked.
template <class ElemType>
void Matrix<ElemType>::NesterovAcceleratedMomentumSGDUpdate(Matrix<ElemType>& gradients,
                                                            Matrix<ElemType>& smoothedGradients,
                                                            ElemType learnRatePerSample,
                                                            ElemType momentum,
                                                            bool unitGainMomentum)
{
    DecideAndMoveToRightDevice(smoothedGradients, gradients, *this);

    const auto unitGainFactor = ElemType(unitGainMomentum ? (1.0 - momentum) : 1.0);

    DISPATCH_MATRIX_ON_FLAG(&gradients, nullptr,
        { /* CPU dense */
            // 1) sg_t = momentum * sg_{t-1} + learnRatePerSample * unitGainFactor * g_{t-1}
            // 2) w'_t = w_{t-1} - momentum * sg_t
            // 3) w_t = w'_t - learnRatePerSample * unitGainFactor * g_{t-1}
            // The end result:
            //  w_t = w_{t-1} - momentum^2 * sg_{t-1} - learnRatePerSample * unitGainFactor * (1 + momentum) * g_{t-1}
            //  sg_t = momentum * sg_{t-1} + learnRatePerSample * unitGainFactor * g_{t-1}
            ScaleAndAdd( unitGainFactor * learnRatePerSample, gradients, momentum, smoothedGradients);
            ScaleAndAdd(-momentum, smoothedGradients, *this);
            ScaleAndAdd(-unitGainFactor * learnRatePerSample, gradients, *this);
        },
        { /* GPU dense */
            ScaleAndAdd(unitGainFactor * learnRatePerSample, gradients, momentum, smoothedGradients);
            ScaleAndAdd(-momentum, smoothedGradients, *this);
            ScaleAndAdd(-unitGainFactor * learnRatePerSample, gradients, *this);
        },
        { /* CPU sparse */
            if (momentum != 0)
            {
                // Identical to the above, except that as a side effect "NormalGrad" modifies 
                // gradient values in place, so that gradientCache is needed to store the original values.
                Matrix<ElemType> gradientCache(gradients.GetDeviceId());
                gradientCache.AssignValuesOf(gradients);
                gradients.m_CPUSparseMatrix->NormalGrad(*smoothedGradients.m_CPUMatrix, momentum, unitGainMomentum);
                ScaleAndAdd(-momentum, smoothedGradients, *this);
                ScaleAndAdd(-unitGainFactor * learnRatePerSample, gradientCache, *this);
            }
        },
        { /* GPU sparse */
            if (momentum != 0)
            {
                Matrix<ElemType> gradientCache(gradients.GetDeviceId());
                gradientCache.AssignValuesOf(gradients);
                gradients.m_GPUSparseMatrix->NormalGrad(*smoothedGradients.m_GPUMatrix, momentum, unitGainMomentum);
                ScaleAndAdd(-momentum, smoothedGradients, *this);
                ScaleAndAdd(-unitGainFactor * learnRatePerSample, gradientCache, *this);
            }
        });
}

// both 'this' and gradients will be changed
template <class ElemType>
ElemType Matrix<ElemType>::Adagrad(Matrix<ElemType>& gradients, const bool needAveMultiplier)
{
    DecideAndMoveToRightDevice(*this, gradients);

    DISPATCH_MATRIX_ON_FLAG(&gradients, &gradients,
        { return m_CPUMatrix->Adagrad(*gradients.m_CPUMatrix, needAveMultiplier);       SetDataLocation(CPU); },
        { return m_GPUMatrix->Adagrad(*gradients.m_GPUMatrix, needAveMultiplier);       SetDataLocation(GPU); },
        { return gradients.m_CPUSparseMatrix->Adagrad(*m_CPUMatrix, needAveMultiplier); SetDataLocation(CPU); },
        { return gradients.m_GPUSparseMatrix->Adagrad(*m_GPUMatrix, needAveMultiplier); SetDataLocation(GPU); });
    // Note: Since both 'this' and gradients are changed, we must call SetDataLocation() on 'this' as well.
}

// FSAdaGrad update -- Frank's "fix" of AdaGrad, very similar to what became later known as Adam
// updates
//  - momentum accumulator
//  - var momentum accumulator
//  - denominator
// then
//  - the model itself
template <class ElemType>
void Matrix<ElemType>::FSAdagradUpdate(size_t mbSize,
                                       Matrix<ElemType>& gradients, Matrix<ElemType>& functionValues, double& smoothedCount,
                                       const double learnRatePerSample, const double targetAdagradAvDenom,
                                       const double meanMomentum, const double varMomentum, bool unitGainMomentum)
{
    // keep track on how many samples have been accumulated into the g^2 accumulator
    smoothedCount = varMomentum * smoothedCount + (1.0 - varMomentum) * mbSize;

    // update the numerator and then do the meanMomentum-based model update
    // Each AdaGrad-normalized gradient value is multiplied by the following, which
    //  - makes up for general scaling (targetAdagradAvDenom, a constant chosen by the user that should resemble the typical value range of gradients)
    //  - sqrt(1/#samples accumulated) to turn the sqr sum into an average
    let targetAdagradAvDenom_x_sqrtAdagradSqrFrames = (ElemType)(targetAdagradAvDenom * sqrt(smoothedCount));

    DISPATCH_MATRIX_ON_FLAG(&gradients, &gradients,
        { 
            m_CPUMatrix->FSAdagrad(*gradients.m_CPUMatrix, *functionValues.m_CPUMatrix, 
                                   (ElemType)learnRatePerSample, (ElemType)meanMomentum, (ElemType)varMomentum, 
                                   targetAdagradAvDenom_x_sqrtAdagradSqrFrames, unitGainMomentum); 
            SetDataLocation(CPU); 
        },
        {
            m_GPUMatrix->FSAdagrad(*gradients.m_GPUMatrix, *functionValues.m_GPUMatrix, 
                                   (ElemType)learnRatePerSample, (ElemType)meanMomentum, (ElemType)varMomentum, 
                                   targetAdagradAvDenom_x_sqrtAdagradSqrFrames, unitGainMomentum); 
            SetDataLocation(GPU); 
        },
        { NOT_IMPLEMENTED; },
        { gradients.m_GPUSparseMatrix->FSAdagrad(*m_GPUMatrix, *functionValues.m_GPUMatrix, (ElemType)learnRatePerSample, (ElemType)meanMomentum, (ElemType)varMomentum, targetAdagradAvDenom_x_sqrtAdagradSqrFrames, unitGainMomentum); SetDataLocation(GPU); });

    // Note: Since both 'this' and gradients are changed, we must call SetDataLocation() on 'this' as well.
}

///
// Implement the original adam algorithm according to the paper
// Ref: ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION, https://arxiv.org/pdf/1412.6980.pdf
///
template <class ElemType>
void Matrix<ElemType>::AdamUpdate(Matrix<ElemType>& gradients, Matrix<ElemType>& functionValues, double& smoothedCount,
    const double learnRatePerSample, const double meanMomentum, const double varMomentum, bool unitGainMomentum)
{
    smoothedCount++;
    // Bias correction
    let biasCorrection = (ElemType)(sqrt(1- pow(varMomentum, smoothedCount))/(1- pow(meanMomentum, smoothedCount)));

    DISPATCH_MATRIX_ON_FLAG(&gradients, &gradients,
    {
        m_CPUMatrix->Adam(*gradients.m_CPUMatrix, *functionValues.m_CPUMatrix,
        (ElemType)learnRatePerSample, (ElemType)meanMomentum, (ElemType)varMomentum,
        biasCorrection, unitGainMomentum);
        SetDataLocation(CPU);
    },
    {
        m_GPUMatrix->Adam(*gradients.m_GPUMatrix, *functionValues.m_GPUMatrix,
        (ElemType)learnRatePerSample, (ElemType)meanMomentum, (ElemType)varMomentum,
        biasCorrection, unitGainMomentum);
        SetDataLocation(GPU);
    },
    { NOT_IMPLEMENTED; },
    { gradients.m_GPUSparseMatrix->Adam(*m_GPUMatrix, *functionValues.m_GPUMatrix, 
        (ElemType)learnRatePerSample, (ElemType)meanMomentum, 
        (ElemType)varMomentum, biasCorrection, unitGainMomentum); 
        SetDataLocation(GPU); });

    // Note: Since both 'this' and gradients are changed, we must call SetDataLocation() on 'this' as well.
}

template <class ElemType>
ElemType Matrix<ElemType>::RmsProp(Matrix<ElemType>& gradients,
                                   ElemType RMS_GAMMA,
                                   ElemType RMS_WGT_INC,
                                   ElemType RMS_WGT_MAX,
                                   ElemType RMS_WGT_DEC,
                                   ElemType RMS_WGT_MIN,
                                   const bool needAveMultiplier)
{
    DecideAndMoveToRightDevice(*this, gradients);

    DISPATCH_MATRIX_ON_FLAG(&gradients, &gradients,
        { return m_CPUMatrix->RmsProp(*gradients.m_CPUMatrix, RMS_GAMMA, RMS_WGT_INC, RMS_WGT_MAX, RMS_WGT_DEC, RMS_WGT_MIN, needAveMultiplier); SetDataLocation(CPU); },
        { return m_GPUMatrix->RmsProp(*gradients.m_GPUMatrix, RMS_GAMMA, RMS_WGT_INC, RMS_WGT_MAX, RMS_WGT_DEC, RMS_WGT_MIN, needAveMultiplier); SetDataLocation(GPU); },
        { NOT_IMPLEMENTED; },
        { return gradients.m_GPUSparseMatrix->RmsProp(*m_GPUMatrix, RMS_GAMMA, RMS_WGT_INC, RMS_WGT_MAX, RMS_WGT_DEC, RMS_WGT_MIN, needAveMultiplier); SetDataLocation(GPU); });
    // Note: Since both 'this' and gradients are changed, we must call SetDataLocation() on 'this' as well.
}

template <class ElemType>
void Matrix<ElemType>::AdaDeltaUpdate(Matrix<ElemType>& gradients,
    Matrix<ElemType>& functionValues,
    ElemType rho, ElemType epsilon)
{
    DecideAndMoveToRightDevice(*this, gradients);

    DISPATCH_MATRIX_ON_FLAG(&gradients, &gradients,
    { return m_CPUMatrix->AdaDelta(*gradients.m_CPUMatrix, *functionValues.m_CPUMatrix, rho, epsilon); SetDataLocation(CPU); },
    { return m_GPUMatrix->AdaDelta(*gradients.m_GPUMatrix, *functionValues.m_GPUMatrix, rho, epsilon); SetDataLocation(GPU); },
    { return gradients.m_CPUSparseMatrix->AdaDelta(*m_CPUMatrix, *functionValues.m_CPUMatrix, rho, epsilon); SetDataLocation(CPU); },
    { return gradients.m_GPUSparseMatrix->AdaDelta(*m_GPUMatrix, *functionValues.m_GPUMatrix, rho, epsilon); SetDataLocation(GPU); });
}

template <class ElemType>
void Matrix<ElemType>::Reshape(const size_t numRows, const size_t numCols)
{
    if (numRows != GetNumRows() || numCols != GetNumCols())
    {
        DISPATCH_MATRIX_ON_FLAG(this, this,
            { m_CPUMatrix->Reshape(numRows, numCols); },
            { m_GPUMatrix->Reshape(numRows, numCols); },
            { NOT_IMPLEMENTED; },
            { m_GPUSparseMatrix->Reshape(numRows, numCols); });
    }
}

// Note: Resize() will leave the matrix content undefined.
// Note: Resize calls RequireSizeAndAllocate on the sparse versions in for performance reasons. If the external caller knows the nz, then we should set it.
template <class ElemType>
void Matrix<ElemType>::Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve /*=0*/, bool growOnly /*=true*/)
{
    // TODO: should this function test whether the size is changing, and skip if it isn't? We have at least one explicit test for this code calling this (recurrent node)
    DISPATCH_MATRIX_ON_FLAG_USEBOTH_4BOTH(this,
        { m_CPUMatrix->Resize(numRows, numCols, growOnly); },
        { m_GPUMatrix->Resize(numRows, numCols, growOnly); },
        { m_CPUSparseMatrix->RequireSizeAndAllocate(numRows, numCols, numNZElemToReserve, growOnly, false); },
        { m_GPUSparseMatrix->RequireSizeAndAllocate(numRows, numCols, numNZElemToReserve, growOnly, false); });
#ifdef _DEBUG
    if (GetMatrixType() != MatrixType::SPARSE)
        Invalidate(); // Fill the matrix with NaNs to detect using the content which is undefined. Unfortunately this won't work for sparse matrices.
#endif
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::RepMat(const Matrix<ElemType>& frmMat, const size_t rowRatio, const size_t colRatio)
{
    size_t nCols = frmMat.GetNumCols();
    size_t nRows = frmMat.GetNumRows();

    if (rowRatio > 1)
        RuntimeError("RepMat not yet supporting raw ratio larger than 1");
    size_t newCols = colRatio * nCols;

    Matrix<ElemType> c(nRows, newCols, frmMat.GetDeviceId());
    for (size_t i = 0; i < colRatio; i++)
    {
        c.ColumnSlice(i * nCols, nCols).AssignValuesOf(frmMat);
    }

    return c;
}

template <class ElemType>
size_t Matrix<ElemType>::GetAllocatedSize() const
{
    return m_baseMatrix->GetSizeAllocated();
}

// reset for sparse matrix. Semantically the same as setting all values to 0.
template <class ElemType>
void Matrix<ElemType>::Reset()
{
    DISPATCH_MATRIX_ON_FLAG_USEBOTH_4BOTH(this,
        { NOT_IMPLEMENTED; },
        { NOT_IMPLEMENTED; },
        { m_CPUSparseMatrix->Reset(); },
        { m_GPUSparseMatrix->Reset(); });
}

template <class ElemType>
size_t Matrix<ElemType>::GetNumRows() const
{
    return m_baseMatrix->GetNumRows();
}

template <class ElemType>
size_t Matrix<ElemType>::GetNumCols() const
{
    return m_baseMatrix->GetNumCols();
}

template <class ElemType>
size_t Matrix<ElemType>::GetNumElements() const
{
    return GetNumRows() * GetNumCols();
}

template <class ElemType>
bool Matrix<ElemType>::IsEmpty() const
{
    return m_baseMatrix->IsEmpty();
}

#pragma endregion Basic Operators

#pragma region Member BLAS Functions

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::operator+=(ElemType alpha)
{
    return AssignSumOf(alpha, *this);
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::operator+(ElemType alpha) const
{
    Matrix<ElemType> c(GetNumRows(), GetNumCols(), GetDeviceId());
    c.AssignSumOf(alpha, *this);
    return c;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignSumOf(const ElemType alpha, const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSumOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignSumOf(alpha, *a.m_CPUMatrix),
                            m_GPUMatrix->AssignSumOf(alpha, *a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//if [this] and a have same dimension then [this]=[this]+a
//if a is a column vector, add to all columns of [this]
//if a is a row vector, add to all rows of [this]
//if a is a scalar, add it to all elements.
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::operator+=(const Matrix<ElemType>& a)
{
    DecideAndMoveToRightDevice(*this, a);

    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->operator+=(*a.m_CPUMatrix),
                            m_GPUMatrix->operator+=(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//if [this] and a have same dimension then OUTPUT=[this]+a
//if a is a column vector, add to all columns of [this]
//if a is a row vector, add to all rows of [this]
template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::operator+(const Matrix<ElemType>& a) const
{
    if (GetNumElements() == 1)
    {
        Matrix<ElemType> c(a.DeepClone());

        DISPATCH_MATRIX_ON_FLAG(this,
                                &c,
                                c += (*this)(0, 0),
                                c += (m_GPUMatrix->Get00Element()), // BUGBUG: efficiency
                                c += (*this)(0, 0),
                                NOT_IMPLEMENTED);
        return c;
    }
    else if (a.GetNumElements() == 1)
    {
        Matrix<ElemType> c(this->DeepClone());

        DISPATCH_MATRIX_ON_FLAG(&a,
                                &c,
                                c += a(0, 0),
                                c += (a.m_GPUMatrix->Get00Element()), // BUGBUG: efficiency
                                c += a(0, 0),
                                NOT_IMPLEMENTED);
        return c;
    }
    else
    {
        Matrix<ElemType> c(this->DeepClone()); // this implementation will introduce a copy overhead. but make resue of the code
        c += a;
        return c;
    }
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignSumOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b)
{
    if (this == &a)
    {
        *this += b;
        return *this;
    }
    if (this == &b)
    {
        *this += a;
        return *this;
    }
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
Matrix<ElemType>& Matrix<ElemType>::operator-=(ElemType alpha)
{
    return AssignDifferenceOf(*this, alpha);
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::operator-(ElemType alpha) const
{
    Matrix<ElemType> c(GetNumRows(), GetNumCols(), GetDeviceId());
    c.AssignDifferenceOf(*this, alpha);
    return c;
}

//for each column of a, we assign numRows starting from startIndex to this
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignRowSliceValuesOf(const Matrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignRowSliceValuesOf(*a.m_CPUMatrix, startIndex, numRows),
                            m_GPUMatrix->AssignRowSliceValuesOf(*a.m_GPUMatrix, startIndex, numRows),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
    return *this;
}

//for each column of a, we assign all rows of a to this starting from startIndex
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignToRowSliceValuesOf(const Matrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    DecideAndMoveToRightDevice(*this, a);

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignToRowSliceValuesOf(*a.m_CPUMatrix, startIndex, numRows),
                            m_GPUMatrix->AssignToRowSliceValuesOf(*a.m_GPUMatrix, startIndex, numRows),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//for the row slice of this starting from startIndex we add a to it.
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AddToRowSliceValuesOf(const Matrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    DecideAndMoveToRightDevice(*this, a);

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AddToRowSliceValuesOf(*a.m_CPUMatrix, startIndex, numRows),
                            m_GPUMatrix->AddToRowSliceValuesOf(*a.m_GPUMatrix, startIndex, numRows),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//for each column of this, we add row slice of a starting from startIndex
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AddWithRowSliceValuesOf(const Matrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    DecideAndMoveToRightDevice(*this, a);

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AddWithRowSliceValuesOf(*a.m_CPUMatrix, startIndex, numRows),
                            m_GPUMatrix->AddWithRowSliceValuesOf(*a.m_GPUMatrix, startIndex, numRows),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignRepeatOf(const Matrix<ElemType>& a, const size_t numRowRepeats, const size_t numColRepeats)
{
    DecideAndMoveToRightDevice(*this, a);

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignRepeatOf(*a.m_CPUMatrix, numRowRepeats, numColRepeats),
                            m_GPUMatrix->AssignRepeatOf(*a.m_GPUMatrix, numRowRepeats, numColRepeats),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AddToRowRepeatValuesOf(const Matrix<ElemType>& a, const size_t numRepeats)
{
    DecideAndMoveToRightDevice(*this, a);

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AddToRowRepeatValuesOf(*a.m_CPUMatrix, numRepeats),
                            m_GPUMatrix->AddToRowRepeatValuesOf(*a.m_GPUMatrix, numRepeats),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//used in the DSSM model. The resulted *this is a [a.GetRows()*(negNumber+1), a.GetCols()] matrix
//each column contains posNumber of  positive samples (original) and negNumber negative samples generated by copying
//sample shifted by shiftNumber columns
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignPositiveAndShiftedNegSample(const Matrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber)
{
    DecideAndMoveToRightDevice(*this, a);

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignPositiveAndShiftedNegSample(*a.m_CPUMatrix, posNumber, negNumber, shiftNumber),
                            m_GPUMatrix->AssignPositiveAndShiftedNegSample(*a.m_GPUMatrix, posNumber, negNumber, shiftNumber),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//used in the DSSM model. *this = *this + positive and negative samples folded back to the right place
//each column of a contains posNumber of  positive samples (original) and negNumber negative samples generated by copying
//sample shifted by shiftNumber columns
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AddFoldedPositiveAndShiftedNegSample(const Matrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber)
{
    DecideAndMoveToRightDevice(*this, a);

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AddFoldedPositiveAndShiftedNegSample(*a.m_CPUMatrix, posNumber, negNumber, shiftNumber),
                            m_GPUMatrix->AddFoldedPositiveAndShiftedNegSample(*a.m_GPUMatrix, posNumber, negNumber, shiftNumber),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignDifferenceOf(const ElemType alpha, const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignDifferenceOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignDifferenceOf(alpha, *a.m_CPUMatrix),
                            m_GPUMatrix->AssignDifferenceOf(alpha, *a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignDifferenceOf(const Matrix<ElemType>& a, const ElemType alpha)
{
    if (a.IsEmpty())
        LogicError("AssignDifferenceOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignDifferenceOf(*a.m_CPUMatrix, alpha),
                            m_GPUMatrix->AssignDifferenceOf(*a.m_GPUMatrix, alpha),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//if [this] and a have same dimension then [this]=[this]-a
//if a is a column vector, minus it from all columns of [this]
//if a is a row vector, minus it from all rows of [this]
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::operator-=(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("Minus Operation: Matrix a is empty.");
    DecideAndMoveToRightDevice(*this, a);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this, 
                            *m_CPUMatrix -= *a.m_CPUMatrix,
                            *m_GPUMatrix -= *a.m_GPUMatrix,
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//if [this] and a have same dimension then output=[this]-a
//if a is a column vector, minus it from all columns of [this]
//if a is a row vector, minus it from all rows of [this]
template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::operator-(const Matrix<ElemType>& a) const
{
    Matrix<ElemType> c(this->DeepClone()); // this implementation will introduce a copy overhead. but make resue of the code
    ScaleAndAdd(-1, a, c);
    return c;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignDifferenceOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b)
{
    // if first arg broadcasts, we swap first and the flip the sign
    // This is because there is no equivalent to operator-=() that works the other way round.
    // TODO: We need ternary ops where the output storage is separate.
    if (a.GetNumRows() < b.GetNumRows() || a.GetNumCols() < b.GetNumCols())
    {
        if (a.GetNumRows() > b.GetNumRows() || a.GetNumCols() > b.GetNumCols())
            LogicError("AssignDifferenceOf: Invalid dimensions.");
        AssignDifferenceOf(b, a);
        *this *= -1;
        return *this;
    }
    if (this != &a)
        SetValue(a);
    (*this) -= b;
    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::operator*=(ElemType alpha)
{
    Scale(alpha, *this);
    return *this;
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::operator*(ElemType alpha) const
{
    Matrix<ElemType> c(GetNumRows(), GetNumCols(), (DEVICEID_TYPE) m_preferredDeviceId);
    Scale(alpha, *this, c);
    return c;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignProductOf(const ElemType alpha, const Matrix<ElemType>& a)
{
    Scale(alpha, a, *this);
    return *this;
}

// [this]=a*b
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignProductOf(const Matrix<ElemType>& a, const bool transposeA, const Matrix<ElemType>& b, const bool transposeB)
{
    if (a.GetNumElements() == 1)
    {
        if (transposeB)
            AssignTransposeOf(b);
        else
            this->SetValue(b);

        DISPATCH_MATRIX_ON_FLAG(this,
                                nullptr,
                                (*this) *= a(0, 0),
                                (*this) *= a.m_GPUMatrix->Get00Element(),
                                (*this) *= a(0, 0),
                                NOT_IMPLEMENTED);
    }
    else if (b.GetNumElements() == 1)
    {
        if (transposeA)
            AssignTransposeOf(a);
        else
            this->SetValue(a);

        DISPATCH_MATRIX_ON_FLAG(this,
                                nullptr,
                                (*this) *= b(0, 0),
                                (*this) *= b.m_GPUMatrix->Get00Element(),
                                (*this) *= b(0, 0),
                                NOT_IMPLEMENTED);
    }
    else
        Multiply(a, transposeA, b, transposeB, *this);

    return *this;
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::operator*(const Matrix<ElemType>& a) const
{
    if (GetNumElements() == 1)
    {
        Matrix<ElemType> c((DEVICEID_TYPE) a.GetPreferredDeviceId());

        DISPATCH_MATRIX_ON_FLAG(this,
                                nullptr,
                                c.AssignProductOf((*this)(0, 0), a),
                                c.AssignProductOf(m_GPUMatrix->Get00Element(), a), // BUGBUG: efficiency
                                c.AssignProductOf((*this)(0, 0), a),
                                NOT_IMPLEMENTED);

        return c;
    }
    else if (a.GetNumElements() == 1)
    {
        Matrix<ElemType> c((DEVICEID_TYPE) GetPreferredDeviceId());

        DISPATCH_MATRIX_ON_FLAG(&a,
                                nullptr,
                                c.AssignProductOf(a(0, 0), (*this)),
                                c.AssignProductOf(a.m_GPUMatrix->Get00Element(), (*this)), // BUGBUG: efficiency
                                c.AssignProductOf(a(0, 0), (*this)),
                                NOT_IMPLEMENTED);

        return c;
    }
    else
    {
        Matrix<ElemType> c(GetNumRows(), a.GetNumCols(), (DEVICEID_TYPE) GetPreferredDeviceId());
        Multiply(*this, a, c);
        return c;
    }
}

// [this]=a*b  where a is a 1x1 scalar
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::Assign1x1ProductOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b)
{
    Multiply1x1AndWeightedAdd(+1, a, b, 0.0f, *this);
    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::operator/=(ElemType alpha)
{
    (*this) *= 1 / alpha;
    return (*this);
}

template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::operator/(ElemType alpha) const
{
    return ((*this) * (1 / alpha));
}

//element-wise power
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::operator^=(ElemType alpha)
{
    auto& us = *this;
    ElementWisePower(alpha, us, us);
    return us;
}

//element-wise power
template <class ElemType>
Matrix<ElemType> Matrix<ElemType>::operator^(ElemType alpha) const
{
    Matrix<ElemType> c(GetNumRows(), GetNumCols(), (DEVICEID_TYPE) GetDeviceId());
    ElementWisePower(alpha, *this, c);
    return c;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignElementPowerOf(const Matrix<ElemType>& a, const ElemType power)
{
    ElementWisePower(power, a, *this);
    return *this;
}

//[this]=[this] .* a (we cannot override operator .* in c++)
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::ElementMultiplyWith(const Matrix<ElemType>& a)
{
    return AssignElementProductOf(*this, a);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::ElementDivideBy(const Matrix<ElemType>& a)
{
    return AssignElementDivisionOf(*this, a);
}

//[this]=a .* b
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignElementProductOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    DecideAndMoveToRightDevice(a, b, *this);
    if (!(a.GetMatrixType() == b.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignElementProductOf(*a.m_CPUMatrix, *b.m_CPUMatrix),
                            m_GPUMatrix->AssignElementProductOf(*a.m_GPUMatrix, *b.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AddElementProductOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AddElementProductOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match [this].");

    DecideAndMoveToRightDevice(*this, a, b);

    if (!(a.GetMatrixType() == b.GetMatrixType() && GetMatrixType() == b.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            m_CPUMatrix->AddElementProductOf(*a.m_CPUMatrix, *b.m_CPUMatrix),
                            m_GPUMatrix->AddElementProductOf(*a.m_GPUMatrix, *b.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//[this]=a ./ b
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignElementDivisionOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementDivisionOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    DecideAndMoveToRightDevice(a, b, *this);
    // WARNING: a and b must have same type
    if (!(a.GetMatrixType() == b.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignElementDivisionOf(*a.m_CPUMatrix, *b.m_CPUMatrix),
                            m_GPUMatrix->AssignElementDivisionOf(*a.m_GPUMatrix, *b.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::ColumnElementMultiplyWith(const Matrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("ColumnElementMultiplyWith: Matrix is empty.");

    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
        InvalidArgument("ColumnElementMultiplyWith: The input matrix should be a col vector and match [this]'s rows.");

    DecideAndMoveToRightDevice(*this, a);
    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->ColumnElementMultiplyWith(*a.m_CPUMatrix),
                            m_GPUMatrix->ColumnElementMultiplyWith(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::RowElementMultiplyWith(const Matrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("RowElementMultiplyWith: Matrix is empty.");

    if (!(a.GetNumCols() == GetNumCols() && a.GetNumRows() == 1))
        InvalidArgument("RowElementMultiplyWith: The input matrix should be a row vector and match [this]'s columns.");

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->RowElementMultiplyWith(*a.m_CPUMatrix),
                            m_GPUMatrix->RowElementMultiplyWith(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::RowElementDivideBy(const Matrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("RowElementDivideBy: Matrix is empty.");

    if (!(a.GetNumCols() == GetNumCols() && a.GetNumRows() == 1))
        InvalidArgument("RowElementDivideBy: The input matrix should be a row vector and match [this]'s columns.");

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->RowElementDivideBy(*a.m_CPUMatrix),
                            m_GPUMatrix->RowElementDivideBy(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::ColumnElementDivideBy(const Matrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("ColumnElementDivideBy: Matrix is empty.");

    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
        InvalidArgument("ColumnElementDivideBy: The input matrix should be a col vector and match [this]'s rows.");

    DecideAndMoveToRightDevice(*this, a);
    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->ColumnElementDivideBy(*a.m_CPUMatrix),
                            m_GPUMatrix->ColumnElementDivideBy(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//[this]=1 ./ a
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::ElementInverse()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->ElementInverse(),
                            m_GPUMatrix->ElementInverse(),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->ElementInverse());

    return (*this);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignElementInverseOf(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignElementInverseOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignElementInverseOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignElementInverseOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->AssignElementInverseOf(*a.m_GPUSparseMatrix));

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceSigmoid()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceSigmoid(),
                            m_GPUMatrix->InplaceSigmoid(),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->InplaceSigmoid());

    return (*this);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignSigmoidOf(const Matrix<ElemType>& a)
{
    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignSigmoidOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignSigmoidOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->AssignSigmoidOf(*a.m_GPUSparseMatrix));

    return *this;
}

//[this]=sigmoid([this]) element wise
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceLinearRectifierDerivative()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceLinearRectifierDerivative(),
                            m_GPUMatrix->InplaceLinearRectifierDerivative(),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->InplaceLinearRectifierDerivative());

    return (*this);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignLinearRectifierDerivativeOf(const Matrix<ElemType>& a)
{
    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignLinearRectifierDerivativeOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignLinearRectifierDerivativeOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->AssignLinearRectifierDerivativeOf(*a.m_GPUSparseMatrix));

    return *this;
}

//[this]=sigmoid([this]) element wise
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceSigmoidDerivative()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceSigmoidDerivative(),
                            m_GPUMatrix->InplaceSigmoidDerivative(),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return (*this);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignSigmoidDerivativeOf(const Matrix<ElemType>& a)
{
    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignSigmoidDerivativeOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignSigmoidDerivativeOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignNumOfDiff(const Matrix<ElemType>& a, const Matrix<ElemType>& b, bool searchInCol)
{
    DecideAndMoveToRightDevice(a, b, *this);
    // WARNING: a and b must have same type
    if (!(a.GetMatrixType() == b.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignNumOfDiff(*a.m_CPUMatrix, *b.m_CPUMatrix, searchInCol),
                            m_GPUMatrix->AssignNumOfDiff(*a.m_GPUMatrix, *b.m_GPUMatrix, searchInCol),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}
//[this]=tanh([this]) element wise
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceTanh()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceTanh(),
                            m_GPUMatrix->InplaceTanh(),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->InplaceTanh());

    return (*this);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignTanhOf(const Matrix<ElemType>& a)
{
    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignTanhOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignTanhOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->AssignTanhOf(*a.m_GPUSparseMatrix));

    return *this;
}

//[this]=softmax([this]) element wise
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceLogSoftmax(const bool isColWise)
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceLogSoftmax(isColWise),
                            m_GPUMatrix->InplaceLogSoftmax(isColWise),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignLogSoftmaxOf(const Matrix<ElemType>& a, const bool isColWise)
{
    if (a.IsEmpty())
        LogicError("AssignLogSoftmaxOf: Matrix a is empty.");
    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignLogSoftmaxOf(*a.m_CPUMatrix, isColWise),
                            m_GPUMatrix->AssignLogSoftmaxOf(*a.m_GPUMatrix, isColWise),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//[this]=softmax([this]) element wise
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceHardmax(const bool isColWise)
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceHardmax(isColWise),
                            m_GPUMatrix->InplaceHardmax(isColWise),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignHardmaxOf(const Matrix<ElemType>& a, const bool isColWise)
{
    if (a.IsEmpty())
        LogicError("AssignHardmaxOf: Matrix a is empty.");
    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignHardmaxOf(*a.m_CPUMatrix, isColWise),
                            m_GPUMatrix->AssignHardmaxOf(*a.m_GPUMatrix, isColWise),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceSqrt()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceSqrt(),
                            m_GPUMatrix->InplaceSqrt(),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->InplaceSqrt());

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignSqrtOf(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSqrtOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignSqrtOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignSqrtOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->AssignSqrtOf(*a.m_GPUSparseMatrix));

    return *this;
}

//[this]=exp([this]) element wise
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceExp()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceExp(),
                            m_GPUMatrix->InplaceExp(),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->InplaceExp());

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignExpOf(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignExpOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignExpOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignExpOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->AssignExpOf(*a.m_GPUSparseMatrix));

    return *this;
}

//[this]=exp([this]) element wise
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceAbs()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            m_CPUMatrix->InplaceAbs(),
                            m_GPUMatrix->InplaceAbs(),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->InplaceAbs());

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignAbsOf(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignAbsOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignAbsOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignAbsOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->AssignAbsOf(*a.m_GPUSparseMatrix));

    return *this;
}

//[this]=log([this]) element wise
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceLog()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceLog(),
                            m_GPUMatrix->InplaceLog(),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->InplaceLog());

    return *this;
}

//[this]=log([this]) element wise
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceLog10()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceLog10(),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignLogOf(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignLogOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignLogOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignLogOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->AssignLogOf(*a.m_GPUSparseMatrix));

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignLog10Of(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignLogOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignLog10Of(*a.m_CPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->AssignLogOf(*a.m_GPUSparseMatrix));

    return *this;
}

//[this]=cos([this]) element wise
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceCosine()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceCosine(),
                            m_GPUMatrix->InplaceCosine(),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignCosineOf(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignCosineOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignCosineOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignCosineOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//[this]= -sin([this]) element wise
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceNegativeSine()
{
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceNegativeSine(),
                            m_GPUMatrix->InplaceNegativeSine(),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignNegativeSineOf(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignNegativeSineOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignNegativeSineOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignNegativeSineOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceTruncate(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("InplaceTruncate: Matrix is empty.");

    if (sizeof(ElemType) == sizeof(float))
    {
        if (!isfinite((float) threshold))
            return *this;
    }
    else
    {
        if (!isfinite(threshold))
            return *this;
    }

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceTruncate(threshold),
                            m_GPUMatrix->InplaceTruncate(threshold),
                            m_CPUSparseMatrix->InplaceTruncate(threshold),
                            m_GPUSparseMatrix->InplaceTruncate(threshold));

    return *this;
}

template <class ElemType>
void Matrix<ElemType>::InplaceTranspose()
{
    if (IsEmpty())
        return;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->InplaceTranspose());
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceSoftThreshold(const ElemType threshold)
{
    assert(threshold >= 0);

    if (IsEmpty())
        LogicError("InplaceSoftThreshold: Matrix is empty.");

    if (threshold == 0)
        return *this;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceSoftThreshold(threshold),
                            m_GPUMatrix->InplaceSoftThreshold(threshold),
                            m_CPUSparseMatrix->InplaceSoftThreshold(threshold),
                            m_GPUSparseMatrix->InplaceSoftThreshold(threshold));

    return *this;
}
//Threshold truncating: this[i] = max( this[i], threshold )
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceTruncateBottom(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("InplaceTruncateBottom: Matrix is empty.");

    if (sizeof(ElemType) == sizeof(float))
    {
        if (!isfinite((float) threshold))
            return *this;
    }
    else
    {
        if (!isfinite(threshold))
            return *this;
    }

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceTruncateBottom(threshold),
                            m_GPUMatrix->InplaceTruncateBottom(threshold),
                            m_CPUSparseMatrix->InplaceTruncateBottom(threshold),
                            m_GPUSparseMatrix->InplaceTruncateBottom(threshold));

    return *this;
}

//Threshold truncating: this[i] = max( a[i], threshold )
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignTruncateBottomOf(const Matrix<ElemType>& a, const ElemType threshold)
{
    if (a.IsEmpty())
        LogicError("AssignTruncateBottomOf: Matrix a is empty.");

    if (sizeof(ElemType) == sizeof(float))
    {
        if (!isfinite((float) threshold))
        {
            this->SetValue(a);
            return *this;
        }
    }
    else
    {
        if (!isfinite(threshold))
        {
            this->SetValue(a);
            return *this;
        }
    }

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignTruncateBottomOf(*a.m_CPUMatrix, threshold),
                            m_GPUMatrix->AssignTruncateBottomOf(*a.m_GPUMatrix, threshold),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->AssignTruncateBottomOf(*a.m_GPUSparseMatrix, threshold));

    return *this;
}

//Threshold truncating: this[i] = min( this[i], threshold )
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::InplaceTruncateTop(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("InplaceTruncateTop: Matrix is empty.");

    if (sizeof(ElemType) == sizeof(float))
    {
        if (!isfinite((float) threshold))
            return *this;
    }
    else
    {
        if (!isfinite(threshold))
            return *this;
    }

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->InplaceTruncateTop(threshold),
                            m_GPUMatrix->InplaceTruncateTop(threshold),
                            m_CPUSparseMatrix->InplaceTruncateTop(threshold),
                            m_GPUSparseMatrix->InplaceTruncateTop(threshold));

    return *this;
}
//Threshold truncating: this[i] = min( a[i], threshold )
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignTruncateTopOf(const Matrix<ElemType>& a, const ElemType threshold)
{
    if (a.IsEmpty())
        LogicError("AssignTruncateTopOf: Matrix a is empty.");

    if (sizeof(ElemType) == sizeof(float))
    {
        if (!isfinite((float) threshold))
        {
            this->SetValue(a);
            return *this;
        }
    }
    else
    {
        if (!isfinite(threshold))
        {
            this->SetValue(a);
            return *this;
        }
    }

    DecideAndMoveToRightDevice(a, *this);
    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignTruncateTopOf(*a.m_CPUMatrix, threshold),
                            m_GPUMatrix->AssignTruncateTopOf(*a.m_GPUMatrix, threshold),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->AssignTruncateTopOf(*a.m_GPUSparseMatrix, threshold));

    return *this;
}

//Threshold truncating: this[i] = 0 if abs(this[i]<threshold).
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::SetToZeroIfAbsLessThan(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("SetToZeroIfAbsLessThan: Matrix is empty.");

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->SetToZeroIfAbsLessThan(threshold),
                            m_GPUMatrix->SetToZeroIfAbsLessThan(threshold),
                            NOT_IMPLEMENTED,
                            m_GPUSparseMatrix->SetToZeroIfAbsLessThan(threshold));

    return *this;
}

//sum of all elements
template <class ElemType>
ElemType Matrix<ElemType>::SumOfElements() const
{
    if (IsEmpty())
        LogicError("SumOfElements: Matrix is empty.");

    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            return m_CPUMatrix->SumOfElements(),
                            return m_GPUMatrix->SumOfElements(),
                            return m_CPUSparseMatrix->SumOfElements(),
                            return m_GPUSparseMatrix->SumOfElements());
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignSumOfElements(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSumOfElements: Matrix a is empty.");

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignSumOfElements(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignSumOfElements(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
DeviceBoundNumber<ElemType> Matrix<ElemType>::Sum_AsDeviceBoundNum() const
{
    DeviceBoundNumber<ElemType> result;

    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            ElemType* val = new ElemType;
                                * val = m_CPUMatrix->SumOfElements(); result.ShallowCopyFrom(val, -1); return result,
                                                                                                              return m_GPUMatrix->Sum_AsDeviceBoundNum(),
                                                                                                              NOT_IMPLEMENTED,
                                                                                                              NOT_IMPLEMENTED);
}

//sum of all elements
template <class ElemType>
ElemType Matrix<ElemType>::SumOfAbsElements() const
{
    if (IsEmpty())
        LogicError("SumOfAbsElements: Matrix is empty.");

    DISPATCH_MATRIX_ON_FLAG(this, nullptr,
                            { return m_CPUMatrix->SumOfAbsElements(); },
                            { return m_GPUMatrix->SumOfAbsElements(); },
                            { NOT_IMPLEMENTED; },
                            { return m_GPUSparseMatrix->SumOfAbsElements(); });
}

//sum of all elements
template <class ElemType>
ElemType Matrix<ElemType>::LogSumOfElements() const
{
    if (IsEmpty())
        LogicError("LogSumOfElements: Matrix is empty.");

    DISPATCH_MATRIX_ON_FLAG(this, nullptr,
                            { return m_CPUMatrix->LogSumOfElements(); },
                            { return m_GPUMatrix->LogSumOfElements(); },
                            { NOT_IMPLEMENTED},
                            { NOT_IMPLEMENTED });
}

template <class ElemType>
bool Matrix<ElemType>::IsValid() const
{
    if (m_currentDataLocation == CurrentDataLocation::GPU && GetMatrixType() == MatrixType::SPARSE)
    {
        return this->m_GPUSparseMatrix->IsValid();
    }
    else
    {
        NOT_IMPLEMENTED;
    }

    return false;
}

template <class ElemType>
bool Matrix<ElemType>::IsEqualTo(const Matrix<ElemType>& a, const ElemType threshold /*= 1e-8*/) const
{
    return AreEqual(*this, a, threshold);
}

template <class ElemType>
void Matrix<ElemType>::VectorSum(const Matrix<ElemType>& a, Matrix<ElemType>& c, const bool isColWise)
{
    DecideAndMoveToRightDevice(c, a);
    if (!(a.GetMatrixType() == c.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(&c,
                            &c,
                            CPUMatrix<ElemType>::VectorSum(*a.m_CPUMatrix, *c.m_CPUMatrix, isColWise),
                            GPUMatrix<ElemType>::VectorSum(*a.m_GPUMatrix, *c.m_GPUMatrix, isColWise),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::VectorNorm1(Matrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorNormInf: Matrix is empty.");

    DecideAndMoveToRightDevice(*this, c);
    c.SwitchToMatrixType(GetMatrixType(), GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            &c,
                            m_CPUMatrix->VectorNorm1(*c.m_CPUMatrix, isColWise),
                            m_GPUMatrix->VectorNorm1(*c.m_GPUMatrix, isColWise),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignVectorNorm1Of(Matrix<ElemType>& a, const bool isColWise)
{
    a.VectorNorm1(*this, isColWise);
    return *this;
}

template <class ElemType>
void Matrix<ElemType>::VectorNorm2(Matrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorNorm2: Matrix is empty.");

    DecideAndMoveToRightDevice(*this, c);
    c.SwitchToMatrixType(GetMatrixType(), GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            &c,
                            m_CPUMatrix->VectorNorm2(*c.m_CPUMatrix, isColWise),
                            m_GPUMatrix->VectorNorm2(*c.m_GPUMatrix, isColWise),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignVectorNorm2Of(Matrix<ElemType>& a, const bool isColWise)
{
    a.VectorNorm2(*this, isColWise);
    return *this;
}

template <class ElemType>
void Matrix<ElemType>::VectorNormInf(Matrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorNormInf: Matrix is empty.");

    DecideAndMoveToRightDevice(*this, c);
    c.SwitchToMatrixType(GetMatrixType(), GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            &c,
                            m_CPUMatrix->VectorNormInf(*c.m_CPUMatrix, isColWise),
                            m_GPUMatrix->VectorNormInf(*c.m_GPUMatrix, isColWise),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignVectorNormInfOf(Matrix<ElemType>& a, const bool isColWise)
{
    a.VectorNormInf(*this, isColWise);
    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignInnerProductOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const bool isColWise)
{
    InnerProduct(a, b, *this, isColWise);
    return *this;
}

//column-wise crossproduct
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignKhatriRaoProductOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignKhatriRaoProductOf: Matrix is empty.");

    assert(a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("AssignKhatriRaoProductOf: The input matrix dimensions do not match.");

    DecideAndMoveToRightDevice(a, b, *this);
    // WARNING: a and b must have same type
    if (!(a.GetMatrixType() == b.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignKhatriRaoProductOf(*a.m_CPUMatrix, *b.m_CPUMatrix),
                            m_GPUMatrix->AssignKhatriRaoProductOf(*a.m_GPUMatrix, *b.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//column-wise reshaped product. Used to compute KhatriRaoProduct Gradient
//   this = reshape each column of a from (K1xK2,1) to (K1, K2)
//   if each column of a is not transposed, each (K1, K2) times each column of b (K2, frames).
//   the output is a (K1, frames) matrix
//   if each column of a is tranposed, each (K1, K2)^T times each column of b(K1, frames) and output is (K2, frames)
//column-wise crossproduct
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AddColumnReshapeProductOf(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const bool transposeAColumn)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AddColumnReshapeProductOf: Matrix is empty.");

    assert(a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("AddColumnReshapeProductOf: The input matrix dimensions do not match.");

    DecideAndMoveToRightDevice(*this, a, b);
    // WARNING: a and b must have same type
    if (!(a.GetMatrixType() == b.GetMatrixType() && GetMatrixType() == b.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AddColumnReshapeProductOf(*a.m_CPUMatrix, *b.m_CPUMatrix, transposeAColumn),
                            m_GPUMatrix->AddColumnReshapeProductOf(*a.m_GPUMatrix, *b.m_GPUMatrix, transposeAColumn),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AddWithScaleOf(ElemType alpha, const Matrix<ElemType>& a)
{
    ScaleAndAdd(alpha, a, *this);
    return *this;
}

template <class ElemType>
ElemType Matrix<ElemType>::FrobeniusNorm() const
{
    if (IsEmpty())
        LogicError("FrobeniusNorm: Matrix is empty.");

    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            return m_CPUMatrix->FrobeniusNorm(),
                            return m_GPUMatrix->FrobeniusNorm(),
                            return m_CPUSparseMatrix->FrobeniusNorm(),
                            return m_GPUSparseMatrix->FrobeniusNorm());
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignFrobeniusNormOf(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignFrobeniusNormOf: Matrix a is empty.");

    Resize(1, 1);

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignFrobeniusNormOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignFrobeniusNormOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
ElemType Matrix<ElemType>::MatrixNormInf() const
{
    if (IsEmpty())
        LogicError("MatrixNormInf: Matrix is empty.");

    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            return m_CPUMatrix->MatrixNormInf(),
                            return m_GPUMatrix->MatrixNormInf(),
                            NOT_IMPLEMENTED,
                            return m_GPUSparseMatrix->MatrixNormInf());
}

template <class ElemType>
ElemType Matrix<ElemType>::MatrixNorm1() const
{
    if (IsEmpty())
        LogicError("MatrixNorm1: Matrix is empty.");

    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            return m_CPUMatrix->MatrixNorm1(),
                            return m_GPUMatrix->MatrixNorm1(),
                            NOT_IMPLEMENTED,
                            return m_GPUSparseMatrix->MatrixNorm1());
}

template <class ElemType>
ElemType Matrix<ElemType>::MatrixNorm0() const
{
    if (IsEmpty())
        LogicError("MatrixNorm0: Matrix is empty.");

    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            return m_CPUMatrix->MatrixNorm0(),
                            return m_GPUMatrix->MatrixNorm0(),
                            NOT_IMPLEMENTED,
                            return m_GPUSparseMatrix->MatrixNorm0());
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignSignOf(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSignOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AssignSignOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AssignSignOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AddSignOf(const Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AddSignOf: Matrix a is empty.");

    DecideAndMoveToRightDevice(a, *this);
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(&a,
                            this,
                            m_CPUMatrix->AddSignOf(*a.m_CPUMatrix),
                            m_GPUMatrix->AddSignOf(*a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

// I decided to use Matrix<ElemType>& maxIndices instead of integer vector because the result may be used to do additional calculation
template <class ElemType>
void Matrix<ElemType>::VectorMax(Matrix<ElemType>& maxIndices, Matrix<ElemType>& maxValues, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorMax: Matrix is empty.");

    DecideAndMoveToRightDevice(*this, maxIndices, maxValues);
    maxIndices.SwitchToMatrixType(GetMatrixType(), GetFormat(), false);
    maxValues.SwitchToMatrixType(GetMatrixType(), GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this, &maxValues,
        { m_CPUMatrix->VectorMax(*maxIndices.m_CPUMatrix, *maxValues.m_CPUMatrix, isColWise); maxIndices.SetDataLocation(CPU, DENSE); },
        { m_GPUMatrix->VectorMax(*maxIndices.m_GPUMatrix, *maxValues.m_GPUMatrix, isColWise); maxIndices.SetDataLocation(GPU, DENSE); },
        { NOT_IMPLEMENTED; },
        { NOT_IMPLEMENTED; });
    // Note: must SetDataLocation() also on maxIndices, since both maxValues and maxIndices are written.
}

template <class ElemType>
void Matrix<ElemType>::VectorMax(Matrix<ElemType>& maxIndices, Matrix<ElemType>& maxValues, const bool isColWise, int topK) const
{
    if (IsEmpty())
        LogicError("VectorMax: Matrix is empty.");

    DecideAndMoveToRightDevice(*this, maxIndices, maxValues);
    maxIndices.SwitchToMatrixType(GetMatrixType(), GetFormat(), false);
    maxValues.SwitchToMatrixType(GetMatrixType(), GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this, &maxValues,
        { m_CPUMatrix->VectorMax(*maxIndices.m_CPUMatrix, *maxValues.m_CPUMatrix, isColWise, topK); maxIndices.SetDataLocation(CPU, DENSE); },
        { m_GPUMatrix->VectorMax(*maxIndices.m_GPUMatrix, *maxValues.m_GPUMatrix, isColWise, topK); maxIndices.SetDataLocation(GPU, DENSE); },
        { NOT_IMPLEMENTED; },
        { NOT_IMPLEMENTED; });
}

template <class ElemType>
void Matrix<ElemType>::VectorMin(Matrix<ElemType>& minIndices, Matrix<ElemType>& minValues, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorMin: Matrix is empty.");

    DecideAndMoveToRightDevice(*this, minIndices, minValues);
    minIndices.SwitchToMatrixType(GetMatrixType(), GetFormat(), false);
    minValues.SwitchToMatrixType(GetMatrixType(), GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this, &minValues,
        { m_CPUMatrix->VectorMin(*minIndices.m_CPUMatrix, *minValues.m_CPUMatrix, isColWise); minIndices.SetDataLocation(CPU, DENSE); },
        { m_GPUMatrix->VectorMin(*minIndices.m_GPUMatrix, *minValues.m_GPUMatrix, isColWise); minIndices.SetDataLocation(GPU, DENSE); },
        { NOT_IMPLEMENTED; },
        { NOT_IMPLEMENTED; });
}

#pragma endregion Member BLAS Functions

#pragma region Other helper Functions

template <class ElemType>
int Matrix<ElemType>::GetDeviceId() const
{
    if (m_currentDataLocation == CurrentDataLocation::NONE)
        return m_preferredDeviceId;

    DISPATCH_MATRIX_ON_FLAG(this, nullptr,
        { return CPUDEVICE; },
        { return m_GPUMatrix->GetComputeDeviceId(); },
        { return CPUDEVICE; },
        { return m_GPUSparseMatrix->GetComputeDeviceId(); });
}

template <class ElemType>
MatrixType Matrix<ElemType>::GetMatrixType() const
{
    return m_matrixType;
}

template <class ElemType>
MatrixFormat Matrix<ElemType>::GetFormat() const
{
    return m_baseMatrix->GetFormat();
}

// TODO: Comment why we need a second ElemType.
// TODO: Move the shared core functions to the front of this source file.
// BUGBUG: This performs a copy operation even for the output matrix that gets overwritten right away.
//         We should (1) define which is the output and (2) whether it will be completely overwritten (so we won't actually copy it).
// bring two matrices onto the same device
// If different and prefered devices are the same, move to preferred device.
// Otherwise GPU takes precedence over CPU, and if both are GPU move to a's device.
// The inputs are only distinguished in that a's GPU takes precedence over b's in case they differ.
// TODO: This is called somewhat inconsistently, sometimes with a=*this, sometimes with b=*this.
template <class ElemType>
template <class ElemType2>
void Matrix<ElemType>::DecideAndMoveToRightDevice(const Matrix<ElemType>& a, const Matrix<ElemType2>& b)
{
    int deviceIdA = a.GetDeviceId(), deviceIdB = b.GetDeviceId();
    if (deviceIdA == deviceIdB)
        return;

    if (!a.OwnBuffer() && b.OwnBuffer())
        b._transferToDevice(deviceIdA);
    else if (a.OwnBuffer() && !b.OwnBuffer())
        a._transferToDevice(deviceIdB);
    else
    {
        int preferredDeviceIdA = a.GetPreferredDeviceId(), preferredDeviceIdB = b.GetPreferredDeviceId();

        if (preferredDeviceIdA == preferredDeviceIdB) // both prefer the same device: move to preferred
        {
            a._transferToDevice(preferredDeviceIdA);
            b._transferToDevice(preferredDeviceIdA);
        }
        else if (deviceIdA != CPUDEVICE) // one of them lives on GPU: use that
        {
            b._transferToDevice(deviceIdA);
        }
        else
        {
            a._transferToDevice(deviceIdB);
        }
    }
}

// same but for 3 matrices
// If b and c are both on the same GPU then a will be forced to go there; otherwise a's GPU takes precedence, then b's.
template <class ElemType>
void Matrix<ElemType>::DecideAndMoveToRightDevice(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const Matrix<ElemType>& c)
{
    int deviceIdA = a.GetDeviceId(), deviceIdB = b.GetDeviceId(), deviceIdC = c.GetDeviceId();
    if (deviceIdA == deviceIdB && deviceIdA == deviceIdC)
        return;

    int preferredDeviceIdA = a.GetPreferredDeviceId(), preferredDeviceIdB = b.GetPreferredDeviceId(), preferredDeviceIdC = c.GetPreferredDeviceId();

    if (preferredDeviceIdA == preferredDeviceIdB && preferredDeviceIdA == preferredDeviceIdC)
    {
        a._transferToDevice(preferredDeviceIdA);
        b._transferToDevice(preferredDeviceIdA);
        c._transferToDevice(preferredDeviceIdA);
    }
    else if (deviceIdB == deviceIdC && deviceIdB != CPUDEVICE) // TODO: why not the other two combinations?
    {
        a._transferToDevice(deviceIdB); // 'a' is outvoted
    }
    else if (deviceIdA != CPUDEVICE) // one of them lives on GPU: use that
    {
        b._transferToDevice(deviceIdA);
        c._transferToDevice(deviceIdA);
    }
    else if (deviceIdB != CPUDEVICE)
    {
        a._transferToDevice(deviceIdB);
        c._transferToDevice(deviceIdB);
    }
    else
    {
        a._transferToDevice(deviceIdC);
        b._transferToDevice(deviceIdC);
    }
}

// same but for 4 matrices
template <class ElemType>
void Matrix<ElemType>::DecideAndMoveToRightDevice(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const Matrix<ElemType>& c, const Matrix<ElemType>& d)
{
    // this function is only called for one operator, so for now we keep it simple
    DecideAndMoveToRightDevice(a, b, c);
    d._transferToDevice(a.GetDeviceId()); // BUGBUG: Is this correct in case a,b,c share the same preferredDevice?
}

template <class ElemType>
void Matrix<ElemType>::_transferToDevice(int to_id, bool isBeingMoved /*= true*/, bool emptyTransfer /* = false*/) const
{
    int from_id = GetDeviceId();
    if (to_id == from_id) // nothing to do
        return;

    if (OwnBuffer())
        _transferFromDeviceToDevice(from_id, to_id, isBeingMoved, emptyTransfer);
    else
        RuntimeError("Cannot move externally owned matrices to the preferred device.");
}

// this function performs data transfer and updates data location, but not the device that is stored with it
template <class ElemType>
void Matrix<ElemType>::_transferFromDeviceToDevice(int from_id, int to_id, bool isBeingMoved /*= true*/, bool emptyTransfer /* = false*/) const
{
    if (from_id < 0)
        from_id = CPUDEVICE;
    if (to_id < 0)
        to_id = CPUDEVICE;

    if (from_id == to_id)
    {
        if (from_id != GetDeviceId())
            RuntimeError("Trying to transfer matrix from device to the same device while the matrix does not live in the from device.");

        return;
    }

    // warn about device change
#define NUM_DEVICE_CHANGED_WARN 20
    if (m_numTimesDeviceChanged <= NUM_DEVICE_CHANGED_WARN &&
        (!emptyTransfer || (from_id >= 0 && to_id >= 0)))
    {
        m_numTimesDeviceChanged++;
        if (m_devicesTransferedTo[0] < CPUDEVICE)
            m_devicesTransferedTo[0] = to_id;
        else if (m_devicesTransferedTo[0] != to_id)
            m_devicesTransferedTo[1] = to_id;
    }
    if ((GetMathLibTraceLevel() > 0) && (m_numTimesDeviceChanged == NUM_DEVICE_CHANGED_WARN && m_devicesTransferedTo[1] >= CPUDEVICE))
        fprintf(stderr, "WARNING: The same matrix with dim [%lu, %lu] has been transferred between different devices for %d times.\n", (unsigned long) GetNumRows(), (unsigned long) GetNumCols(), NUM_DEVICE_CHANGED_WARN);

    // do the transfer
    if (m_matrixType == MatrixType::SPARSE)
    {
        if (from_id == CPUDEVICE) // from CPU to GPU
        {
            if (!m_CPUSparseMatrix)
                LogicError("Can't move from CPU because I'm not there!");

            if (emptyTransfer)
            {
                if (m_GPUSparseMatrix && m_GPUSparseMatrix->GetComputeDeviceId() == to_id)
                    m_GPUSparseMatrix->Resize(m_CPUSparseMatrix->GetNumRows(), m_CPUSparseMatrix->GetNumCols(), m_CPUSparseMatrix->NzCount());
                else
                    m_GPUSparseMatrix = make_shared<GPUSparseMatrix<ElemType>>(m_CPUSparseMatrix->GetNumRows(), m_CPUSparseMatrix->GetNumCols(), m_CPUSparseMatrix->NzCount(), to_id, m_CPUSparseMatrix->GetFormat());
            }
            else
            {
                if (!m_GPUSparseMatrix || m_GPUSparseMatrix->GetComputeDeviceId() != to_id)
                    m_GPUSparseMatrix = make_shared<GPUSparseMatrix<ElemType>>(to_id);
                m_GPUSparseMatrix->SetValue(*m_CPUSparseMatrix);
            }

            if (isBeingMoved)
            {
                SetDataLocation(GPU, SPARSE);
                m_CPUSparseMatrix = nullptr;
            }
            else
            {
                SetDataLocation(BOTH, SPARSE);
            }
        }
        else // from GPU
        {
            if (!m_GPUSparseMatrix || m_GPUSparseMatrix->GetComputeDeviceId() != from_id)
                LogicError("This matrix isn't on this (or any?) GPU");

            if (to_id < 0) // to CPU
            {
                if (!m_CPUSparseMatrix)
                    m_CPUSparseMatrix = make_shared<CPUSparseMatrix<ElemType>>(m_GPUSparseMatrix->GetFormat());

                if (emptyTransfer)
                    m_CPUSparseMatrix->Resize(m_GPUSparseMatrix->GetNumRows(), m_GPUSparseMatrix->GetNumCols(), m_GPUSparseMatrix->NzCount(), true);
                else
                    m_GPUSparseMatrix->CopyToCPUSparseMatrix(*m_CPUSparseMatrix);

                if (isBeingMoved)
                {
                    SetDataLocation(CPU, SPARSE);
                    m_GPUSparseMatrix = nullptr;
                }
                else
                {
                    SetDataLocation(BOTH, SPARSE);
                }
            }
            else // to another GPU
            {
                m_GPUSparseMatrix->ChangeDeviceTo(to_id);
            }
        }
    }
    else
// #pragma omp critical // causes a build error on gcc; not clear why this is here
    {
        if (from_id == CPUDEVICE) // from CPU to GPU
        {
            if (!m_CPUMatrix)
                LogicError("Can't move from CPU because I'm not there!");
            if (emptyTransfer)
            {
                if (m_GPUMatrix && m_GPUMatrix->GetComputeDeviceId() == to_id)
                    m_GPUMatrix->Resize(m_CPUMatrix->GetNumRows(), m_CPUMatrix->GetNumCols());
                else
                    m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(m_CPUMatrix->GetNumRows(), m_CPUMatrix->GetNumCols(), to_id);
            }
            else
            {
                if (m_GPUMatrix && m_GPUMatrix->GetComputeDeviceId() == to_id)
                    m_GPUMatrix->SetValue(m_CPUMatrix->GetNumRows(), m_CPUMatrix->GetNumCols(), to_id, m_CPUMatrix->Data());
                else
                    m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(m_CPUMatrix->GetNumRows(), m_CPUMatrix->GetNumCols(), to_id, m_CPUMatrix->Data());
            }
            if (isBeingMoved)
            {
                SetDataLocation(GPU, DENSE);
                m_CPUMatrix = nullptr;
            }
            else
                SetDataLocation(BOTH, DENSE);
        }
        else // from GPU
        {
            if (!m_GPUMatrix || m_GPUMatrix->GetComputeDeviceId() != from_id)
                LogicError("This matrix isn't on this (or any?) GPU");

            if (to_id < 0) // to CPU
            {
                if (emptyTransfer)
                {
                    if (m_CPUMatrix)
                        m_CPUMatrix->Resize(m_GPUMatrix->GetNumRows(), m_GPUMatrix->GetNumCols());
                    else
                        m_CPUMatrix = make_shared<CPUMatrix<ElemType>>(m_GPUMatrix->GetNumRows(), m_GPUMatrix->GetNumCols());
                }
                else
                {
                    ElemType* arr = m_GPUMatrix->CopyToArray(); // TODO: unnecessary allocation/copy; why not make this a vector that we move over as an rvalue ref?
                    if (m_CPUMatrix)
                        m_CPUMatrix->SetValue(m_GPUMatrix->GetNumRows(), m_GPUMatrix->GetNumCols(), arr);
                    else
                        m_CPUMatrix = make_shared<CPUMatrix<ElemType>>(m_GPUMatrix->GetNumRows(), m_GPUMatrix->GetNumCols(), arr, matrixFlagNormal);

                    delete[] arr;
                }

                if (isBeingMoved)
                {
                    SetDataLocation(CPU, DENSE);
                    m_GPUMatrix = nullptr;
                }
                else
                {
                    SetDataLocation(BOTH, DENSE);
                }
            }
            else // to another GPU
            {
                m_GPUMatrix->ChangeDeviceTo(to_id);
            }
        }
    } // and of omp critical section
}

template <class ElemType>
void Matrix<ElemType>::TransferFromDeviceToDevice(int from_id, int to_id, bool isBeingMoved, bool emptyTransfer/* = false*/, bool updatePreferredDevice/* = true*/) const
{
    _transferFromDeviceToDevice(from_id, to_id, isBeingMoved, emptyTransfer);
    if (updatePreferredDevice)
        m_preferredDeviceId = GetDeviceId();
}
template <class ElemType>
void Matrix<ElemType>::TransferToDeviceIfNotThere(int to_id, bool isBeingMoved/*false: may leave in BOTH state*/, bool emptyTransfer/* = false*/, bool updatePreferredDevice/* = true*/) const
{
    int from_id = GetDeviceId();

    if (from_id == to_id)                     // already at the right place
        return;

    if (GetCurrentMatrixLocation() == BOTH && // if currently in BOTH state
        !isBeingMoved &&                      // and leaving in BOTH state is OK
        (from_id < 0 || to_id < 0))           // and this is not about changing GPUs
    {
        return;                               // then we are good
    }

    TransferFromDeviceToDevice(from_id, to_id, isBeingMoved, emptyTransfer, updatePreferredDevice);
}

template <class ElemType>
void Matrix<ElemType>::Print(const char* matrixName, ptrdiff_t rowStart, ptrdiff_t rowEnd, ptrdiff_t colStart, ptrdiff_t colEnd) const
{
    DEVICEID_TYPE orgdevice = GetDeviceId();

    DISPATCH_MATRIX_ON_FLAG(this,
                            nullptr,
                            // CPU:
                            m_CPUMatrix->Print(matrixName, rowStart, rowEnd, colStart, colEnd),
                            // GPU;
                            {
                                _transferToDevice(CPUDEVICE, false, false);
                                m_CPUMatrix->Print(matrixName, rowStart, rowEnd, colStart, colEnd);
                                _transferToDevice(orgdevice, false, false);
                            },
                            // CPU, sparse:
                            m_CPUSparseMatrix->Print(matrixName),
                            // GPU, sparse:
                            {
                                _transferToDevice(CPUDEVICE, false, false);
                                m_CPUSparseMatrix->Print(matrixName);
                                _transferToDevice(orgdevice, false, false);
                            });
}

template <class ElemType>
void Matrix<ElemType>::Print(const char* matrixName /*=nullptr*/) const
{
    Print(matrixName, 0, GetNumRows() - 1, 0, GetNumCols() - 1);
}

//helpfer function used for convolution neural network
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignPackedConvolutionInput(const Matrix<ElemType>& inputSubBatch,
                                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                                 const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                                 const bool zeroPadding)
{
    DecideAndMoveToRightDevice(inputSubBatch, *this);
    SwitchToMatrixType(inputSubBatch.GetMatrixType(), inputSubBatch.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&inputSubBatch,
                            this,
                            m_CPUMatrix->AssignPackedConvolutionInput(*(inputSubBatch.m_CPUMatrix),
                                                                      inputWidth, inputHeight, inputChannels,
                                                                      outputWidth, outputHeight, outputChannels,
                                                                      kernelWidth, kernelHeight, horizontalSubsample, verticalSubsample,
                                                                      zeroPadding),
                            m_GPUMatrix->AssignPackedConvolutionInput(*(inputSubBatch.m_GPUMatrix),
                                                                      inputWidth, inputHeight, inputChannels,
                                                                      outputWidth, outputHeight, outputChannels,
                                                                      kernelWidth, kernelHeight, horizontalSubsample, verticalSubsample,
                                                                      zeroPadding),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

//helpfer function used for convolution neural network
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::UnpackConvolutionInput(Matrix<ElemType>& inputSubBatch,
                                                           const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                           const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                           const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                           const bool zeroPadding) const
{
    DecideAndMoveToRightDevice(*this, inputSubBatch);
    inputSubBatch.SwitchToMatrixType(GetMatrixType(), inputSubBatch.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            &inputSubBatch,
                            m_CPUMatrix->UnpackConvolutionInput(*(inputSubBatch.m_CPUMatrix),
                                                                inputWidth, inputHeight, inputChannels,
                                                                outputWidth, outputHeight, outputChannels,
                                                                kernelWidth, kernelHeight, horizontalSubsample, verticalSubsample,
                                                                zeroPadding),
                            m_GPUMatrix->UnpackConvolutionInput(*(inputSubBatch.m_GPUMatrix),
                                                                inputWidth, inputHeight, inputChannels,
                                                                outputWidth, outputHeight, outputChannels,
                                                                kernelWidth, kernelHeight, horizontalSubsample, verticalSubsample,
                                                                zeroPadding),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return inputSubBatch;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignMaxPoolingResult(const Matrix<ElemType>& inputBatch, const size_t channels,
                                                           const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                           const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                           const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    DecideAndMoveToRightDevice(inputBatch, *this);
    SwitchToMatrixType(inputBatch.GetMatrixType(), inputBatch.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&inputBatch,
                            this,
                            m_CPUMatrix->AssignMaxPoolingResult(*(inputBatch.m_CPUMatrix), channels,
                                                                inputWidth, inputHeight, inputSizePerSample,
                                                                outputWidth, outputHeight, outputSizePerSample,
                                                                windowWidth, windowHeight, horizontalSubsample, verticalSubsample),
                            m_GPUMatrix->AssignMaxPoolingResult(*(inputBatch.m_GPUMatrix), channels,
                                                                inputWidth, inputHeight, inputSizePerSample,
                                                                outputWidth, outputHeight, outputSizePerSample,
                                                                windowWidth, windowHeight, horizontalSubsample, verticalSubsample),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AddMaxPoolingGradient(const Matrix<ElemType>& outputGradientBatch, const Matrix<ElemType>& inputBatch, const Matrix<ElemType>& outputBatch,
                                                          const size_t channels,
                                                          const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                          const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                          const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    DecideAndMoveToRightDevice(*this, outputGradientBatch, inputBatch);
    outputBatch._transferToDevice(GetDeviceId());

    if (!(GetMatrixType() == outputGradientBatch.GetMatrixType() && GetMatrixType() == inputBatch.GetMatrixType() && GetMatrixType() == outputBatch.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AddMaxPoolingGradient(*(outputGradientBatch.m_CPUMatrix), *(inputBatch.m_CPUMatrix), *(outputBatch.m_CPUMatrix), channels,
                                                               inputWidth, inputHeight, inputSizePerSample,
                                                               outputWidth, outputHeight, outputSizePerSample,
                                                               windowWidth, windowHeight, horizontalSubsample, verticalSubsample),
                            m_GPUMatrix->AddMaxPoolingGradient(*(outputGradientBatch.m_GPUMatrix), *(inputBatch.m_GPUMatrix), *(outputBatch.m_GPUMatrix), channels,
                                                               inputWidth, inputHeight, inputSizePerSample,
                                                               outputWidth, outputHeight, outputSizePerSample,
                                                               windowWidth, windowHeight, horizontalSubsample, verticalSubsample);
                            ,
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignAveragePoolingResult(const Matrix<ElemType>& inputBatch, const size_t channels,
                                                               const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                               const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                               const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    DecideAndMoveToRightDevice(inputBatch, *this);
    SwitchToMatrixType(inputBatch.GetMatrixType(), inputBatch.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&inputBatch,
                            this,
                            m_CPUMatrix->AssignAveragePoolingResult(*(inputBatch.m_CPUMatrix), channels,
                                                                    inputWidth, inputHeight, inputSizePerSample,
                                                                    outputWidth, outputHeight, outputSizePerSample,
                                                                    windowWidth, windowHeight, horizontalSubsample, verticalSubsample),
                            m_GPUMatrix->AssignAveragePoolingResult(*(inputBatch.m_GPUMatrix), channels,
                                                                    inputWidth, inputHeight, inputSizePerSample,
                                                                    outputWidth, outputHeight, outputSizePerSample,
                                                                    windowWidth, windowHeight, horizontalSubsample, verticalSubsample),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignSoftmaxSum(const Matrix<ElemType>& a, const Matrix<ElemType>& softmax)
{
    Resize(1, 1);
    if (GetDeviceId() < 0)
        a.m_CPUMatrix->AssignSoftmaxSum(*softmax.m_CPUMatrix, *m_CPUMatrix);
    else
        a.m_GPUMatrix->AssignSoftmaxSum(*softmax.m_GPUMatrix, *m_GPUMatrix);
    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignNceUnnormalizedEval(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const Matrix<ElemType>& c, const Matrix<ElemType>& bias)
{
    // if (a.GetMatrixType() != MatrixType::SPARSE)
    //    NOT_IMPLEMENTED;

    Resize(1, 1);
    if (GetDeviceId() < 0)
        a.m_CPUMatrix->AssignNCEUnnormalizedEval(*b.m_CPUMatrix, *c.m_CPUMatrix, *bias.m_CPUMatrix, *m_CPUMatrix);
    else
        a.m_GPUMatrix->AssignNCEUnnormalizedEval(*b.m_GPUMatrix, *c.m_GPUMatrix, *m_GPUMatrix);
    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignNoiseContrastiveEstimation(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const Matrix<ElemType>& c, const Matrix<ElemType>& bias, Matrix<ElemType>& tmp)
{
    if (a.IsEmpty() || b.IsEmpty() || c.IsEmpty())
        LogicError("AssignNoiseContrastiveEstimation: one of the input matrices is empty.");

    if (a.GetDeviceId() != b.GetDeviceId() || b.GetDeviceId() != c.GetDeviceId() || c.GetDeviceId() != GetDeviceId())
        NOT_IMPLEMENTED;

    Resize(1, 1);

    if (GetDeviceId() < 0)
    {
        size_t sampleCount = a.m_CPUMatrix->GetNumElements() / a.m_CPUMatrix->GetNumRows();
        tmp.Resize(a.GetNumRows() / 2, sampleCount);
        a.m_CPUMatrix->AssignNoiseContrastiveEstimation(*b.m_CPUMatrix, *c.m_CPUMatrix,
                                                        *bias.m_CPUMatrix, *tmp.m_CPUMatrix, *m_CPUMatrix);
    }
    else
    {
        size_t sampleCount = a.m_GPUMatrix->GetNumElements() / a.m_GPUMatrix->GetNumRows();
        tmp.Resize(a.GetNumRows() / 2, sampleCount);
        a.m_GPUMatrix->AssignNoiseContrastiveEstimation(*b.m_GPUMatrix, *c.m_GPUMatrix,
                                                        *bias.m_GPUMatrix, sampleCount, *tmp.m_GPUMatrix, *m_GPUMatrix);
    }
    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignNCEDerivative(const Matrix<ElemType>& tmp, const Matrix<ElemType>& a, const Matrix<ElemType>& b, const Matrix<ElemType>& c, size_t inputIndex)
{
    if (a.IsEmpty() || b.IsEmpty() || c.IsEmpty())
        LogicError("AssignNoiseContrastiveEstimation: one of the input matrices is empty.");

    if (a.GetDeviceId() != b.GetDeviceId() || b.GetDeviceId() != c.GetDeviceId() || c.GetDeviceId() != GetDeviceId())
        NOT_IMPLEMENTED;

    assert(tmp.GetNumRows() == a.GetNumRows() / 2);
    if (GetDeviceId() < 0)
    {
        // samples                         gradient          hidden          embedding                   embedding/hidden
        a.m_CPUMatrix->AssignNCEDerivative(*tmp.m_CPUMatrix, *b.m_CPUMatrix, *c.m_CPUMatrix, inputIndex, *m_CPUMatrix);
    }
    else
    {
        a.m_GPUMatrix->AssignNCEDerivative(*tmp.m_GPUMatrix, *b.m_GPUMatrix, *c.m_GPUMatrix, inputIndex, *m_GPUMatrix);
    }
    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AddAveragePoolingGradient(const Matrix<ElemType>& outputGradientBatch,
                                                              const size_t channels,
                                                              const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                              const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                              const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    DecideAndMoveToRightDevice(*this, outputGradientBatch);
    if (!(GetMatrixType() == outputGradientBatch.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AddAveragePoolingGradient(*(outputGradientBatch.m_CPUMatrix), channels,
                                                                   inputWidth, inputHeight, inputSizePerSample,
                                                                   outputWidth, outputHeight, outputSizePerSample,
                                                                   windowWidth, windowHeight, horizontalSubsample, verticalSubsample),
                            m_GPUMatrix->AddAveragePoolingGradient(*(outputGradientBatch.m_GPUMatrix), channels,
                                                                   inputWidth, inputHeight, inputSizePerSample,
                                                                   outputWidth, outputHeight, outputSizePerSample,
                                                                   windowWidth, windowHeight, horizontalSubsample, verticalSubsample),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

#pragma endregion Other Helper Functions

template <class ElemType>
void Matrix<ElemType>::ConvolutionForward(const Matrix<ElemType>& kernel, const Matrix<int>& mpRowCol, const Matrix<int>& mpRowIwht,
                                          const Matrix<int>& mpRowRun, const Matrix<int>& runs, Matrix<ElemType>& output) const
{
    assert(mpRowCol.GetNumCols() == 1);
    assert(mpRowIwht.GetNumCols() == 1);
    assert(mpRowRun.GetNumCols() == 1);
    assert(runs.GetNumCols() == 1);

    DecideAndMoveToRightDevice(*this, output);

    // REVIEW alexeyk: add sparse version.
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->ConvolutionForward(*(kernel.m_CPUMatrix), *(mpRowCol.m_CPUMatrix), *(mpRowIwht.m_CPUMatrix),
                                                              *(mpRowRun.m_CPUMatrix), *(runs.m_CPUMatrix), *(output.m_CPUMatrix)),
                            m_GPUMatrix->ConvolutionForward(*(kernel.m_GPUMatrix), *(mpRowCol.m_GPUMatrix), *(mpRowIwht.m_GPUMatrix),
                                                             *(mpRowRun.m_GPUMatrix), *(runs.m_GPUMatrix), *(output.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::ConvolutionBackwardData(const Matrix<ElemType>& kernel, const Matrix<int>& mpRowCol, const Matrix<int>& mpRowIwht,
                                               const Matrix<int>& mpRowRun, const Matrix<int>& runs, Matrix<ElemType>& grad) const
{
    assert(mpRowCol.GetNumCols() == 1);
    assert(mpRowIwht.GetNumCols() == 1);
    assert(mpRowRun.GetNumCols() == 1);
    assert(runs.GetNumCols() == 1);

    DecideAndMoveToRightDevice(*this, grad);

    // REVIEW alexeyk: add sparse version.
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->ConvolutionBackwardData(*(kernel.m_CPUMatrix), *(mpRowCol.m_CPUMatrix), *(mpRowIwht.m_CPUMatrix),
                                                                   *(mpRowRun.m_CPUMatrix), *(runs.m_CPUMatrix), *(grad.m_CPUMatrix)),
                            m_GPUMatrix->ConvolutionBackwardData(*(kernel.m_GPUMatrix), *(mpRowCol.m_GPUMatrix), *(mpRowIwht.m_GPUMatrix),
                                                                   *(mpRowRun.m_GPUMatrix), *(runs.m_GPUMatrix), *(grad.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::ConvolutionBackwardKernel(const Matrix<ElemType>& in, const Matrix<int>& mpRowCol, const Matrix<int>& mpRowIwht,
                                                 const Matrix<int>& mpRowRun, const Matrix<int>& runs, Matrix<ElemType>& kernelGrad) const
{
    assert(mpRowCol.GetNumCols() == 1);
    assert(mpRowIwht.GetNumCols() == 1);
    assert(mpRowRun.GetNumCols() == 1);
    assert(runs.GetNumCols() == 1);

    DecideAndMoveToRightDevice(*this, kernelGrad);

    // REVIEW alexeyk: add sparse version.
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->ConvolutionBackwardKernel(*(in.m_CPUMatrix), *(mpRowCol.m_CPUMatrix), *(mpRowIwht.m_CPUMatrix),
                                                                     *(mpRowRun.m_CPUMatrix), *(runs.m_CPUMatrix), *(kernelGrad.m_CPUMatrix)),
                            m_GPUMatrix->ConvolutionBackwardKernel(*(in.m_GPUMatrix), *(mpRowCol.m_GPUMatrix), *(mpRowIwht.m_GPUMatrix),
                                                                     *(mpRowRun.m_GPUMatrix), *(runs.m_GPUMatrix), *(kernelGrad.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::UnrollConvolutionInput(size_t unrollCols, size_t mapOutSize, const Matrix<int>& mpRowCol,
                                              const Matrix<int>& mpRowRun, const Matrix<int>& runs, Matrix<ElemType>& output) const
{
    assert(mpRowCol.GetNumCols() == 1);
    assert(mpRowRun.GetNumCols() == 1);
    assert(runs.GetNumCols() == 1);

    DecideAndMoveToRightDevice(*this, output);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->UnrollConvolutionInput(unrollCols, mapOutSize, *(mpRowCol.m_CPUMatrix),
                                                                *(mpRowRun.m_CPUMatrix), *(runs.m_CPUMatrix), *(output.m_CPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::UnrollConvolutionOutput(size_t unrollCols, size_t mapInCount, size_t mapOutCount, const Matrix<int>& mpRowCol,
                                               const Matrix<int>& mpRowRun, const Matrix<int>& runs, Matrix<ElemType>& output) const
{
    assert(mpRowCol.GetNumCols() == 1);
    assert(mpRowRun.GetNumCols() == 1);
    assert(runs.GetNumCols() == 1);

    DecideAndMoveToRightDevice(*this, output);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->UnrollConvolutionOutput(unrollCols, mapInCount, mapOutCount, *(mpRowCol.m_CPUMatrix),
                                                                 *(mpRowRun.m_CPUMatrix), *(runs.m_CPUMatrix), *(output.m_CPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::UnrollConvolutionInputForKernelBackprop(size_t mapOutSize, const Matrix<int>& mpRowCol,
                                                               const Matrix<int>& mpRowRun, const Matrix<int>& runs, Matrix<ElemType>& output) const
{
    assert(mpRowCol.GetNumCols() == 1);
    assert(mpRowRun.GetNumCols() == 1);
    assert(runs.GetNumCols() == 1);

    DecideAndMoveToRightDevice(*this, output);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->UnrollConvolutionInputForKernelBackprop(mapOutSize, *(mpRowCol.m_CPUMatrix),
                                                                                 *(mpRowRun.m_CPUMatrix), *(runs.m_CPUMatrix), *(output.m_CPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::MaxPoolingForward(const Matrix<int>& mpRowCol, const Matrix<int>& mpRowIndices, const Matrix<int>& indices, Matrix<ElemType>& output) const
{
    assert(mpRowCol.GetNumCols() == 1);
    assert(mpRowIndices.GetNumCols() == 1);
    assert(indices.GetNumCols() == 1);

    DecideAndMoveToRightDevice(*this, output);

    // REVIEW alexeyk: add sparse version.
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->MaxPoolingForward(*(mpRowCol.m_CPUMatrix), *(mpRowIndices.m_CPUMatrix), *(indices.m_CPUMatrix), *(output.m_CPUMatrix)),
                            m_GPUMatrix->MaxPoolingForward(*(mpRowCol.m_GPUMatrix), *(mpRowIndices.m_GPUMatrix), *(indices.m_GPUMatrix), *(output.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::MaxPoolingBackward(const Matrix<ElemType>& out, const Matrix<ElemType>& in,
                                          const Matrix<int>& mpRowCol, const Matrix<int>& mpRowIndices, const Matrix<int>& indices,
                                          Matrix<ElemType>& grad) const
{
    assert(mpRowCol.GetNumCols() == 1);
    assert(mpRowIndices.GetNumCols() == 1);
    assert(indices.GetNumCols() == 1);

    DecideAndMoveToRightDevice(*this, grad);

    // REVIEW alexeyk: add sparse version.
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->MaxPoolingBackward(*(out.m_CPUMatrix), *(in.m_CPUMatrix),
                                                              *(mpRowCol.m_CPUMatrix), *(mpRowIndices.m_CPUMatrix), *(indices.m_CPUMatrix),
                                                              *(grad.m_CPUMatrix)),
                            m_GPUMatrix->MaxPoolingBackward(*(out.m_GPUMatrix), *(in.m_GPUMatrix),
                                                              *(mpRowCol.m_GPUMatrix), *(mpRowIndices.m_GPUMatrix), *(indices.m_GPUMatrix),
                                                              *(grad.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::ROIPoolingForward(const size_t numRois, const size_t numImg, const size_t channels, const size_t width, const size_t height,
                                         const size_t pooledWidth, const size_t pooledHeight, const Matrix<ElemType>& roiData, Matrix<ElemType>& output, 
                                         Matrix<ElemType>& argmax) const
{
    DecideAndMoveToRightDevice(*this, output);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->ROIPoolingForward(numRois, numImg, channels, width, height, pooledWidth, pooledHeight, *(roiData.m_CPUMatrix), *(output.m_CPUMatrix), *(argmax.m_CPUMatrix)),
                            m_GPUMatrix->ROIPoolingForward(numRois, numImg, channels, width, height, pooledWidth, pooledHeight, *(roiData.m_GPUMatrix), *(output.m_GPUMatrix), *(argmax.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::ROIPoolingBackward(const size_t numRois, const size_t numImg, const size_t channels, const size_t width, const size_t height,
                                          const size_t pooledWidth, const size_t pooledHeight, const Matrix<ElemType>& roiData, Matrix<ElemType>& grad, 
                                          Matrix<ElemType>& argmax) const
{
    DecideAndMoveToRightDevice(*this, grad);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->ROIPoolingBackward(numRois, numImg, channels, width, height, pooledWidth, pooledHeight, *(roiData.m_CPUMatrix), *(grad.m_CPUMatrix), *(argmax.m_CPUMatrix)),
                            m_GPUMatrix->ROIPoolingBackward(numRois, numImg, channels, width, height, pooledWidth, pooledHeight, *(roiData.m_GPUMatrix), *(grad.m_GPUMatrix), *(argmax.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::MaxUnpooling(const Matrix<int>& mpRowCol, const Matrix<int>& mpRowIndices, const Matrix<int>& indices, const Matrix<ElemType>& poolInput, Matrix<ElemType>& input) const
{
    assert(mpRowCol.GetNumCols() == 1);
    assert(mpRowIndices.GetNumCols() == 1);
    assert(indices.GetNumCols() == 1);

    DecideAndMoveToRightDevice(*this, input);

    // REVIEW alexeyk: setting values to zero may cause inconsistency when negative values are unpooled.
    // To see why, let's assume we have just one input with negative value and output of, for example, 2x2.
    // As a result of unpooling, there will be 3 zero values and one negative. If we now apply max pooling
    // operation to the output then we get 0 as the output, not the original negative value.
    // In practice this will not happen as pooling layers usually go right after ReLU layer.
    input.SetValue(0);

    // REVIEW alexeyk: add sparse version.
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->MaxUnpooling(*(mpRowCol.m_CPUMatrix), *(mpRowIndices.m_CPUMatrix), *(indices.m_CPUMatrix), *(poolInput.m_CPUMatrix), *(input.m_CPUMatrix)),
                            m_GPUMatrix->MaxUnpooling(*(mpRowCol.m_GPUMatrix), *(mpRowIndices.m_GPUMatrix), *(indices.m_GPUMatrix), *(poolInput.m_GPUMatrix), *(input.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::AveragePoolingForward(const Matrix<int>& mpRowCol, const Matrix<int>& mpRowIndices, const Matrix<int>& indices, Matrix<ElemType>& output, const bool poolPadMode) const
{
    assert(mpRowCol.GetNumCols() == 1);
    assert(mpRowIndices.GetNumCols() == 1);
    assert(indices.GetNumCols() == 1);

    DecideAndMoveToRightDevice(*this, output);

    // REVIEW alexeyk: add sparse version.
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AveragePoolingForward(*(mpRowCol.m_CPUMatrix), *(mpRowIndices.m_CPUMatrix), *(indices.m_CPUMatrix), *(output.m_CPUMatrix), poolPadMode),
                            m_GPUMatrix->AveragePoolingForward(*(mpRowCol.m_GPUMatrix), *(mpRowIndices.m_GPUMatrix), *(indices.m_GPUMatrix), *(output.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::AveragePoolingBackward(const Matrix<int>& mpRowCol, const Matrix<int>& mpRowIndices, const Matrix<int>& indices, Matrix<ElemType>& grad, const bool poolPadMode) const
{
    assert(mpRowCol.GetNumCols() == 1);
    assert(mpRowIndices.GetNumCols() == 1);
    assert(indices.GetNumCols() == 1);

    DecideAndMoveToRightDevice(*this, grad);

    // REVIEW alexeyk: add sparse version.
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AveragePoolingBackward(*(mpRowCol.m_CPUMatrix), *(mpRowIndices.m_CPUMatrix), *(indices.m_CPUMatrix), *(grad.m_CPUMatrix), poolPadMode),
                            m_GPUMatrix->AveragePoolingBackward(*(mpRowCol.m_GPUMatrix), *(mpRowIndices.m_GPUMatrix), *(indices.m_GPUMatrix), *(grad.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::BatchNormalizationForward(const Matrix<ElemType>& scale, const Matrix<ElemType>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, 
                                                 Matrix<ElemType>& runMean, Matrix<ElemType>& runVariance, Matrix<ElemType>& out, double epsilon,
                                                 Matrix<ElemType>& saveMean, Matrix<ElemType>& saveInvStdDev) const
{
    DecideAndMoveToRightDevice(*this, out);

    // REVIEW alexeyk: add sparse version.
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->BatchNormalizationForward(*(scale.m_CPUMatrix), *(bias.m_CPUMatrix), inferenceOnly, expAvgFactor, blendFactor,
                                                                   *(runMean.m_CPUMatrix), *(runVariance.m_CPUMatrix),
                                                                   *(out.m_CPUMatrix), epsilon, *(saveMean.m_CPUMatrix), *(saveInvStdDev.m_CPUMatrix)),
                            m_GPUMatrix->BatchNormalizationForward(*(scale.m_GPUMatrix), *(bias.m_GPUMatrix), inferenceOnly, expAvgFactor, blendFactor,
                                                                   *(runMean.m_GPUMatrix), *(runVariance.m_GPUMatrix),
                                                                   *(out.m_GPUMatrix), epsilon, *(saveMean.m_GPUMatrix), *(saveInvStdDev.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::BatchNormalizationBackward(const Matrix<ElemType>& in, Matrix<ElemType>& grad, const Matrix<ElemType>& scale, double blendFactor,
                                                  const Matrix<ElemType>& saveMean, const Matrix<ElemType>& saveInvStdDev,
                                                  Matrix<ElemType>& scaleGrad, Matrix<ElemType>& biasGrad) const
{
    DecideAndMoveToRightDevice(*this, grad);

    // REVIEW alexeyk: add sparse version.
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->BatchNormalizationBackward(*(in.m_CPUMatrix), *(grad.m_CPUMatrix), *(scale.m_CPUMatrix), blendFactor,
                                                                    *(saveMean.m_CPUMatrix), *(saveInvStdDev.m_CPUMatrix),
                                                                    *(scaleGrad.m_CPUMatrix), *(biasGrad.m_CPUMatrix)),
                            m_GPUMatrix->BatchNormalizationBackward(*(in.m_GPUMatrix), *(grad.m_GPUMatrix), *(scale.m_GPUMatrix), blendFactor,
                                                                    *(saveMean.m_GPUMatrix), *(saveInvStdDev.m_GPUMatrix),
                                                                    *(scaleGrad.m_GPUMatrix), *(biasGrad.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::RNNForward(const Matrix<ElemType> &inputX, const Matrix<ElemType> &paramW, size_t xDim, size_t yDim, const vector<size_t>& numSequencesForFrame, const RnnAttributes& rnnAttributes, Matrix<ElemType>& reserve, Matrix<ElemType>& workspace)
{
    DecideAndMoveToRightDevice(*this, inputX, paramW);
    // move reserve/workspace to the consensus device
    reserve._transferToDevice(GetDeviceId());
    workspace._transferToDevice(GetDeviceId());

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            NOT_IMPLEMENTED,
                            m_GPUMatrix->RNNForward(*(inputX.m_GPUMatrix), *(paramW.m_GPUMatrix), xDim, yDim, numSequencesForFrame, rnnAttributes, *(reserve.m_GPUMatrix), *(workspace.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::RNNBackwardData(const Matrix<ElemType>& outputDY, const Matrix<ElemType>& paramW, Matrix<ElemType>& outputDX, const RnnAttributes& rnnAttributes, Matrix<ElemType>& reserve, Matrix<ElemType>& workspace)
{
    DecideAndMoveToRightDevice(*this, outputDY, paramW, outputDX);
    // move reserve/workspace to the consensus device
    reserve._transferToDevice(GetDeviceId());
    workspace._transferToDevice(GetDeviceId());
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            NOT_IMPLEMENTED,
                            m_GPUMatrix->RNNBackwardData(*(outputDY.m_GPUMatrix), *(paramW.m_GPUMatrix), *(outputDX.m_GPUMatrix), rnnAttributes, *(reserve.m_GPUMatrix), *(workspace.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::RNNBackwardWeights(const Matrix<ElemType>& inputX, const Matrix<ElemType>& outputY, Matrix<ElemType>& dw, const RnnAttributes& rnnAttributes, Matrix<ElemType>& reserve, Matrix<ElemType>& workspace)
{
    DecideAndMoveToRightDevice(*this, inputX, outputY, dw);
    // move reserve/workspace to the consensus device
    reserve._transferToDevice(GetDeviceId());
    workspace._transferToDevice(GetDeviceId());
    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            NOT_IMPLEMENTED,
                            m_GPUMatrix->RNNBackwardWeights(*(inputX.m_GPUMatrix), *(outputY.m_GPUMatrix), *(dw.m_GPUMatrix), rnnAttributes, *(reserve.m_GPUMatrix), *(workspace.m_GPUMatrix)),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

#pragma region Static BLAS Functions

template <class ElemType>
void Matrix<ElemType>::SVD(const Matrix<ElemType>& A, Matrix<ElemType>& SIGMA, Matrix<ElemType>& U, Matrix<ElemType>& VT, Matrix<ElemType>& W)
{
    if (A.IsEmpty())
        LogicError("SVD:  the input matrix is empty.");

    DecideAndMoveToRightDevice(A, SIGMA, U);
    VT._transferToDevice(A.GetDeviceId());
    W._transferToDevice(A.GetDeviceId());

    SIGMA.SwitchToMatrixType(A.GetMatrixType(), A.GetFormat(), false);
    U.SwitchToMatrixType(A.GetMatrixType(), A.GetFormat(), false);
    VT.SwitchToMatrixType(A.GetMatrixType(), A.GetFormat(), false);
    W.SwitchToMatrixType(A.GetMatrixType(), A.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&A, nullptr,
        {
            Matrix<ElemType> tA = A.DeepClone();
            CPUMatrix<ElemType>::SVD(*tA.m_CPUMatrix, *SIGMA.m_CPUMatrix, *U.m_CPUMatrix, *VT.m_CPUMatrix, *W.m_CPUMatrix);
            SIGMA.SetDataLocation(CPU);
            U.SetDataLocation(CPU);
            VT.SetDataLocation(CPU);
            W.SetDataLocation(CPU);
            // need to SetDataLocation() on all matrices we write to
        },
        { NOT_IMPLEMENTED; },
        { NOT_IMPLEMENTED; },
        { NOT_IMPLEMENTED; });
}

/// <summary>Matrix-matrix multiply with col-major matrices (a and b may be transposed): c = alpha * op(a) * op(b) + beta*c</summary>
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="transposeA">Whether matrix a is transposed</param>
/// <param name="b">Input matrix</param>
/// <param name="transposeB">Whether matrix b is transposed</param>
/// <param name="beta">Scalar</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void Matrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const Matrix<ElemType>& a, const bool transposeA, const Matrix<ElemType>& b, const bool transposeB,
                                              ElemType beta, Matrix<ElemType>& c, shared_ptr<QuantizedMultiplier<ElemType>> pQuantizedMultiplier)
{
    DecideAndMoveToRightDevice(a, b, c);

    if (c.GetDeviceId() < 0) // CPU
    {
        if (a.GetMatrixType() == MatrixType::SPARSE) // CPU, SPARSE * ANY -> ANY
        {
            if (b.GetMatrixType() == MatrixType::DENSE      && c.GetMatrixType() == MatrixType::DENSE) // CPU, SPARSE * DENSE  -> DENSE
            {
                CPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(alpha, *a.m_CPUSparseMatrix, transposeA, *b.m_CPUMatrix, transposeB, beta, *c.m_CPUMatrix);
                c.SetDataLocation(CPU, DENSE);
            }
            else if (b.GetMatrixType() == MatrixType::SPARSE && c.GetMatrixType() == MatrixType::DENSE) // CPU, SPARSE * SPARSE -> DENSE
            {
                NOT_IMPLEMENTED;
            }
            else if (b.GetMatrixType() == MatrixType::DENSE  && c.GetMatrixType() == MatrixType::SPARSE)// CPU, SPARSE * DENSE  -> SPARSE
            {
                NOT_IMPLEMENTED;
            }
            else if (b.GetMatrixType() == MatrixType::SPARSE && c.GetMatrixType() == MatrixType::SPARSE)// CPU, SPARSE * SPARSE -> SPARSE
            {
                NOT_IMPLEMENTED;
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }
        else  // CPU, DENSE  * ANY -> ANY
        {
            if (b.GetMatrixType() == MatrixType::SPARSE) // CPU, DENSE * SPARSE -> ANY
            {
                if (c.GetMatrixType() == MatrixType::DENSE) // CPU, DENSE * SPARSE -> DENSE
                {
                    CPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(alpha, *a.m_CPUMatrix, transposeA, *b.m_CPUSparseMatrix, transposeB, beta, *c.m_CPUMatrix);
                    c.SetDataLocation(CPU, DENSE);
                }
                else if (c.GetMatrixType() == MatrixType::SPARSE) // CPU, DENSE * SPARSE -> SPARSE
                {
                    if (beta != 0 && beta != 1)
                    {
                        NOT_IMPLEMENTED;
                    }
                    else
                    {
                        if (beta == 0)
                        {
                            c.Reset();
                        }
                        CPUSparseMatrix<ElemType>::MultiplyAndAdd(alpha, *a.m_CPUMatrix, transposeA, *b.m_CPUSparseMatrix, transposeB, *c.m_CPUSparseMatrix);
                    }
                    c.SetDataLocation(CPU, SPARSE);
                }
                else
                    NOT_IMPLEMENTED; // CPU, DENSE * SPARSE -> UNDETERMINED ?
            }
            else // CPU, DENSE * DENSE -> DENSE (matrix c enforced to be DENSE)
            {
                c.SwitchToMatrixType(MatrixType::DENSE, matrixFormatDense, false);
                CPUMatrix<ElemType>::MultiplyAndWeightedAdd(alpha, *a.m_CPUMatrix, transposeA, *b.m_CPUMatrix, transposeB, beta, *c.m_CPUMatrix, pQuantizedMultiplier);
                c.SetDataLocation(CPU, DENSE);
            }
        }
    }
    else // GPU operations
    {
        if (a.m_matrixType == MatrixType::DENSE && b.m_matrixType == MatrixType::DENSE && c.m_matrixType == MatrixType::DENSE) // GPU, DENSE * DENSE -> DENSE
        {
            GPUMatrix<ElemType>::MultiplyAndWeightedAdd(alpha, *a.m_GPUMatrix, transposeA, *b.m_GPUMatrix, transposeB, beta, *c.m_GPUMatrix);
            c.SetDataLocation(GPU, DENSE);
        }
        else if (a.m_matrixType == MatrixType::SPARSE && b.m_matrixType == MatrixType::DENSE && c.m_matrixType == MatrixType::DENSE) // GPU, SPARSE * DENSE -> DENSE
        {
            GPUMatrix<ElemType> second = transposeB ? b.m_GPUMatrix->Transpose() : *b.m_GPUMatrix;
            GPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(alpha, *a.m_GPUSparseMatrix, transposeA, second, false, beta, *c.m_GPUMatrix);
            c.SetDataLocation(GPU, DENSE);
        }
        else if (a.m_matrixType == MatrixType::DENSE && b.m_matrixType == MatrixType::SPARSE && c.m_matrixType == MatrixType::DENSE) // GPU, DENSE * SPARSE -> DENSE
        {
            GPUSparseMatrix<ElemType>::MultiplyAndWeightedAdd(alpha, *a.m_GPUMatrix, transposeA, *b.m_GPUSparseMatrix, transposeB, beta, *c.m_GPUMatrix);
            c.SetDataLocation(GPU, DENSE);
        }
        else if (a.m_matrixType == MatrixType::DENSE && b.m_matrixType == MatrixType::SPARSE && c.m_matrixType == MatrixType::SPARSE) // GPU, DENSE * SPARSE -> SPARSE
        {
            if (beta != 0 && beta != 1)
            {
                NOT_IMPLEMENTED;
            }
            else
            {
                if (beta == 0)
                {
                    c.Reset();
                }
                GPUSparseMatrix<ElemType>::MultiplyAndAdd(alpha, *a.m_GPUMatrix, transposeA, *b.m_GPUSparseMatrix, transposeB, *c.m_GPUSparseMatrix);
            }
            c.SetDataLocation(GPU, SPARSE);
        }
        else if (a.m_matrixType == MatrixType::SPARSE && b.m_matrixType == MatrixType::SPARSE && c.m_matrixType == MatrixType::SPARSE) // GPU, SPARSE * SPARSE -> SPARSE
        {
            GPUSparseMatrix<ElemType> firstDummy = alpha == 1 ? *a.m_GPUSparseMatrix : (*a.m_GPUSparseMatrix) * alpha;
            GPUSparseMatrix<ElemType>& first = firstDummy; // By Malcolm.. gcc doesn't support auto
            if (beta == 0)
            {
                GPUSparseMatrix<ElemType>::Multiply(first, transposeA, *b.m_GPUSparseMatrix, transposeB, *c.m_GPUSparseMatrix);
                c.SetDataLocation(GPU, SPARSE);
            }
            else
            {
                GPUSparseMatrix<ElemType> tmp(b.m_GPUSparseMatrix->GetComputeDeviceId());
                GPUSparseMatrix<ElemType>::Multiply(first, transposeA, *b.m_GPUSparseMatrix, transposeB, tmp);
                *c.m_GPUSparseMatrix = tmp + (*c.m_GPUSparseMatrix) * beta;
                c.SetDataLocation(GPU, SPARSE);
            }
        }
        else if (a.m_matrixType == MatrixType::DENSE && b.m_matrixType == MatrixType::DENSE && c.m_matrixType == MatrixType::SPARSE) // GPU, DENSE * DENSE -> SPARSE
        {
            GPUMatrix<ElemType> tmp(a.m_GPUMatrix->GetComputeDeviceId());
            GPUMatrix<ElemType>::MultiplyAndWeightedAdd(alpha, *a.m_GPUMatrix, transposeA, *b.m_GPUMatrix, transposeB, (ElemType)0.0, tmp);
            if (beta != 0)
            {
                GPUSparseMatrix<ElemType> tmpSparse(a.m_GPUMatrix->GetComputeDeviceId());
                tmpSparse.SetValue(tmp);
                *c.m_GPUSparseMatrix = tmpSparse + (*c.m_GPUSparseMatrix) * beta;
            }
            else
            {
                c.m_GPUSparseMatrix->SetValue(tmp);
            }
            c.SetDataLocation(GPU, SPARSE);
        }
        else if (a.m_matrixType == MatrixType::SPARSE && b.m_matrixType == MatrixType::SPARSE && c.m_matrixType == MatrixType::DENSE) // GPU, SPARSE * SPARSE -> DENSE
        {
            NOT_IMPLEMENTED;
        }
        else if (a.m_matrixType == MatrixType::SPARSE && b.m_matrixType == MatrixType::DENSE && c.m_matrixType == MatrixType::SPARSE) // GPU, SPARSE * DENSE -> SPARSE
        {
            NOT_IMPLEMENTED;
        }
        else // No combination left
        {
            NOT_IMPLEMENTED;
        }
    }
}

template <class ElemType>
/*static*/ void Matrix<ElemType>::Multiply1x1AndWeightedAdd(ElemType alpha, const Matrix<ElemType>& a, const Matrix<ElemType>& b, ElemType beta, Matrix<ElemType>& c)
{
    // special case: a is a 1x1 matrix
    // The only alternative is to Get00Elements(), which makes things inefficient.
    if (a.GetNumElements() != 1)
        InvalidArgument("Multiply1x1AndWeightedAdd: first arg must be a scalar.");

    DISPATCH_MATRIX_ON_FLAG(&c,
                            nullptr,
                            CPUMatrix<ElemType>::Multiply1x1AndWeightedAdd(alpha, *a.m_CPUMatrix, *b.m_CPUMatrix, beta, *c.m_CPUMatrix),
                            GPUMatrix<ElemType>::Multiply1x1AndWeightedAdd(alpha, *a.m_GPUMatrix, *b.m_GPUMatrix, beta, *c.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

/// <summary>Matrix-matrix multiply with col-major matrices (a and b may be transposed): c =  op(a) * op(b) + c</summary>
/// <param name="a">Input matrix</param>
/// <param name="transposeA">Whether matrix a is transposed</param>
/// <param name="b">Input matrix</param>
/// <param name="transposeB">Whether matrix b is transposed</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void Matrix<ElemType>::MultiplyAndAdd(const Matrix<ElemType>& a, const bool transposeA, const Matrix<ElemType>& b, const bool transposeB,
                                      Matrix<ElemType>& c)
{
    return Matrix<ElemType>::MultiplyAndWeightedAdd(1.0, a, transposeA, b, transposeB, 1.0, c);
}

/// <summary>Matrix-matrix multiply with col-major matrices (a and b may be transposed): c =  op(a) * op(b)</summary>
/// <param name="a">Input matrix</param>
/// <param name="transposeA">Whether matrix a is transposed</param>
/// <param name="b">Input matrix</param>
/// <param name="transposeB">Whether matrix b is transposed</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void Matrix<ElemType>::Multiply(const Matrix<ElemType>& a, const bool transposeA, const Matrix<ElemType>& b, const bool transposeB,
                                Matrix<ElemType>& c)
{
    return Matrix<ElemType>::MultiplyAndWeightedAdd(1.0, a, transposeA, b, transposeB, 0.0, c);
}

/// <summary>Matrix-matrix multiply with col-major matrices (a and b are not transposed): c =  a * b</summary>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void Matrix<ElemType>::Multiply(const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c)
{
    return Matrix<ElemType>::MultiplyAndWeightedAdd(1.0, a, false, b, false, 0.0, c);
}

/// <summary>1-D Convolution with col-major matrices (a and b may be transposed): c = alpha * op(a) * op(b) + beta*c. MultiplyAndWeightedAdd is just a special case of this.</summary>
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="transposeA">Whether matrix a is transposed</param>
/// <param name="b">Input matrix</param>
/// <param name="transposeB">Whether matrix b is transposed</param>
/// <param name="beta">Scalar</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void Matrix<ElemType>::ConvolveAndWeightedAdd(ElemType alpha, const Matrix<ElemType>& a, const bool transposeA, const Matrix<ElemType>& b, const bool transposeB,
                                              ElemType beta, Matrix<ElemType>& c, size_t numChannels, size_t horizontalSubsample, bool padding, bool channelwise)
{
    DecideAndMoveToRightDevice(a, b, c);

    if (c.GetDeviceId() >= 0 /*GPU*/ && a.GetMatrixType() == MatrixType::DENSE && b.GetMatrixType() == MatrixType::SPARSE && c.GetMatrixType() == MatrixType::DENSE)
    {
        GPUSparseMatrix<ElemType>::ConvolveAndWeightedAdd(alpha, *a.m_GPUMatrix, transposeA, *b.m_GPUSparseMatrix, transposeB, beta, *c.m_GPUMatrix, numChannels, horizontalSubsample, padding, channelwise);
    }
    else
    {
        NOT_IMPLEMENTED;
    }
}

/// <summary>Matrix-scalar multiply with col-major matrices: c = alpha * a + c</summary>
/// if a is a column vector, add to all columns of c
/// if a is a row vector, add to all rows of c
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
/*static*/ void Matrix<ElemType>::ScaleAndAdd(ElemType alpha, const Matrix<ElemType>& a, Matrix<ElemType>& c)
{
    if (a.IsEmpty() || c.IsEmpty())
        LogicError("ScaleAndAdd:  one of the input matrices is empty.");

    DecideAndMoveToRightDevice(c, a);

    if (a.GetMatrixType() == c.GetMatrixType())
    {
        DISPATCH_MATRIX_ON_FLAG(&c, &c,
            { CPUMatrix<ElemType>::ScaleAndAdd(alpha, *a.m_CPUMatrix, *c.m_CPUMatrix); },
            { GPUMatrix<ElemType>::ScaleAndAdd(alpha, *a.m_GPUMatrix, *c.m_GPUMatrix); },
            { NOT_IMPLEMENTED; },
            { GPUSparseMatrix<ElemType> b = move(*c.m_GPUSparseMatrix); GPUSparseMatrix<ElemType>::ScaleAndAdd(alpha, *a.m_GPUSparseMatrix, 1, b, *c.m_GPUSparseMatrix); });
    }
    else
    {
        DISPATCH_MATRIX_ON_FLAG(&c, nullptr,
            {
                CPUSparseMatrix<ElemType>::ScaleAndAdd(alpha, *a.m_CPUSparseMatrix, *c.m_CPUMatrix);
                c.SetDataLocation(CPU);
            },
            {
                if (a.m_GPUSparseMatrix->GetFormat() == MatrixFormat::matrixFormatSparseCSC)
                    GPUSparseMatrix<ElemType>::ScaleAndAdd(alpha, *a.m_GPUSparseMatrix, 1, *c.m_GPUMatrix, *c.m_GPUMatrix);
                else // new GPU sparse matrix code
                    GPUSparseMatrix<ElemType>::ScaleAndAdd(alpha, *a.m_GPUSparseMatrix, *c.m_GPUMatrix);
                c.SetDataLocation(GPU);
            },
            { NOT_IMPLEMENTED; },
            {
                c.m_GPUMatrix = make_shared<GPUMatrix<ElemType>>(c.m_GPUSparseMatrix->CopyToDenseMatrix());
                GPUSparseMatrix<ElemType>::ScaleAndAdd(alpha, *a.m_GPUMatrix, 1, *c.m_GPUSparseMatrix, *c.m_GPUMatrix);
                c.SetDataLocation(GPU, DENSE);
                c.m_GPUSparseMatrix = nullptr;
            });
    }
}

/// <summary>Matrix-scalar multiply with col-major matrices: c = alpha * a + beta * c</summary>
/// if a is a column vector, add to all columns of c
/// if a is a row vector, add to all rows of c
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="beta">Scalar</param>
/// <param name="c">Resulting matrix, caller is responsible for allocating this</param>
template <class ElemType>
/*static*/ void Matrix<ElemType>::ScaleAndAdd(ElemType alpha, const Matrix<ElemType>& a, ElemType beta, Matrix<ElemType>& c)
{
    if (beta == 1)
        ScaleAndAdd(alpha, a, c);
    else if (beta == 0)
        Scale(alpha, a, c);
    else
    {
        ScaleAndAdd(alpha / beta, a, c); // c1=alpha/beta * a + c
        Scale(beta, c);                  // c/beta * beta
        // TODO: two lines above should be changed as follows:
        // Scale(beta, c);                  // c1 = c * beta
        // ScaleAndAdd(alpha, a, c); // c=alpha * a + c1 = alpha * a + beta * c
    }
}

// tensor swapping and addition: c <- keepWeight * b + scaleFactor * swap_dimensions(a, S, K)
// where
//  - a is interpreted as a tensor of dimension (D x S x M x K x T)         // column-major, as usual
//  - b and c as a tensor of dimension          (D x K x M x S x T)   // note: K and S swapped
// The main point of this function is to reshuffle a tensor w.r.t. two dimensions that get swapped in memory,
// but for gradients, we will need to add, hence the keepWeight.
// Notes:
//  - c and b may be the same (in-place operation is expressly allowed).
//  - D, M, and/or T may be 1. For example, D == M == T == 1 implements a 2D matrix transpose from (S x K) to (K x S).
//  - If keepWeight == 0, then b will just get overwritten (straight assignment, b may be uninitialized or contain NaNs).
//  - The original matrix dimensions are ignored except that sizes must match (rows x cols == D x S x M x K x T).
//    For diagnostics purposes, this function also enforces the rows % D == 0 and cols % T == 0, but this is not a functional requirement and can be removed if that helps.
//  - Dense matrices only.
// TODO: Handle these cases:
//  - no swapping happening  --just do a block copy
//  - swapping can be implemented by cuDNN  --do so
template <class ElemType>
/*static*/ void Matrix<ElemType>::TensorShuffleScaleAndAdd(ElemType keepWeight, const Matrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const Matrix<ElemType>& b, Matrix<ElemType>& c)
{
    if (a.GetNumElements() != c.GetNumElements() || b.GetNumElements() != c.GetNumElements()) // allocations must match (but not dimensions, since we reinterpret the dimensions anyway)
        InvalidArgument("TensorShuffleScaleAndAdd: a, b, and c must have same number of elements.");
    if (c.IsEmpty()) // operating on empty minibatch slices is perfectly cromulent
        return;

    // sanity checks for current use cases--these are not strictly necessary and can be deleted
    if (a.GetNumRows() % D != 0 || b.GetNumRows() % D != 0 || c.GetNumRows() % D != 0)
        InvalidArgument("TensorShuffleScaleAndAdd: a, b, and c are meant to have a row dimension that is a multiple of D.");
    if (a.GetNumCols() % T != 0 || b.GetNumCols() % T != 0 || c.GetNumCols() % T != 0)
        InvalidArgument("TensorShuffleScaleAndAdd: a, b, and c are meant to have a column dimension that is a multiple of T.");

    DecideAndMoveToRightDevice(a, b, c);

    DISPATCH_MATRIX_ON_FLAG(&c,
                            nullptr,
                            CPUMatrix<ElemType>::TensorShuffleScaleAndAdd(keepWeight, *a.m_CPUMatrix, D, S, M, K, T, scaleFactor, *b.m_CPUMatrix, *c.m_CPUMatrix),
                            GPUMatrix<ElemType>::TensorShuffleScaleAndAdd(keepWeight, *a.m_GPUMatrix, D, S, M, K, T, scaleFactor, *b.m_GPUMatrix, *c.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            GPUSparseMatrix<ElemType>::TensorShuffleScaleAndAdd(keepWeight, *a.m_GPUSparseMatrix, D, S, M, K, T, scaleFactor, *b.m_GPUSparseMatrix, *c.m_GPUSparseMatrix));
}

/// <summary>c += alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void Matrix<ElemType>::AddScaledDifference(const ElemType alpha, const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c)
{
    DecideAndMoveToRightDevice(c, a, b);
    if (!(a.GetMatrixType() == b.GetMatrixType() && a.GetMatrixType() == c.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(&c,
                            &c,
                            CPUMatrix<ElemType>::AddScaledDifference(alpha, *a.m_CPUMatrix, *b.m_CPUMatrix, *c.m_CPUMatrix),
                            GPUMatrix<ElemType>::AddScaledDifference(alpha, *a.m_GPUMatrix, *b.m_GPUMatrix, *c.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

/// <summary> c = alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void Matrix<ElemType>::AssignScaledDifference(const ElemType alpha, const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c)
{
    DecideAndMoveToRightDevice(a, b, c);

    if (!(a.GetMatrixType() == b.GetMatrixType()))
        NOT_IMPLEMENTED;

    c.SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&c,
                            &c,
                            CPUMatrix<ElemType>::AssignScaledDifference(alpha, *a.m_CPUMatrix, *b.m_CPUMatrix, *c.m_CPUMatrix),
                            GPUMatrix<ElemType>::AssignScaledDifference(alpha, *a.m_GPUMatrix, *b.m_GPUMatrix, *c.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

/// <summary>c += alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void Matrix<ElemType>::AddScaledDifference(const Matrix<ElemType>& alpha, const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c)
{
    DecideAndMoveToRightDevice(c, a, b);
    alpha._transferToDevice(c.GetDeviceId());

    if (!(a.GetMatrixType() == b.GetMatrixType() && a.GetMatrixType() == c.GetMatrixType() && a.GetMatrixType() == alpha.GetMatrixType()))
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(&c,
                            &c,
                            CPUMatrix<ElemType>::AddScaledDifference(*alpha.m_CPUMatrix, *a.m_CPUMatrix, *b.m_CPUMatrix, *c.m_CPUMatrix),
                            GPUMatrix<ElemType>::AddScaledDifference(*alpha.m_GPUMatrix, *a.m_GPUMatrix, *b.m_GPUMatrix, *c.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

/// <summary> c = alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void Matrix<ElemType>::AssignScaledDifference(const Matrix<ElemType>& alpha, const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c)
{
    DecideAndMoveToRightDevice(a, b, alpha);
    c._transferToDevice(a.GetDeviceId());

    if (!(a.GetMatrixType() == b.GetMatrixType() && a.GetMatrixType() == alpha.GetMatrixType()))
        NOT_IMPLEMENTED;

    c.SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&c,
                            nullptr,
                            CPUMatrix<ElemType>::AssignScaledDifference(*alpha.m_CPUMatrix, *a.m_CPUMatrix, *b.m_CPUMatrix, *c.m_CPUMatrix),
                            GPUMatrix<ElemType>::AssignScaledDifference(*alpha.m_GPUMatrix, *a.m_GPUMatrix, *b.m_GPUMatrix, *c.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

//c[ci,cj] += a[ai,aj]
template <class ElemType>
void Matrix<ElemType>::AddElementToElement(const Matrix<ElemType>& a, const size_t ai, const size_t aj, Matrix<ElemType>& c, const size_t ci, const size_t cj)
{
    DecideAndMoveToRightDevice(c, a);

    if (c.GetMatrixType() != a.GetMatrixType())
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(&c,
                            &c,
                            CPUMatrix<ElemType>::AddElementToElement(1, *a.m_CPUMatrix, ai, aj, *c.m_CPUMatrix, ci, cj),
                            GPUMatrix<ElemType>::AddElementToElement(1, *a.m_GPUMatrix, ai, aj, *c.m_GPUMatrix, ci, cj),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

//c[ci,cj] = a[ai,aj]
template <class ElemType>
void Matrix<ElemType>::AssignElementToElement(const Matrix<ElemType>& a, const size_t ai, const size_t aj, Matrix<ElemType>& c, const size_t ci, const size_t cj)
{
    DecideAndMoveToRightDevice(c, a);

    if (c.GetMatrixType() != a.GetMatrixType())
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(&c,
                            &c,
                            CPUMatrix<ElemType>::AddElementToElement(0, *a.m_CPUMatrix, ai, aj, *c.m_CPUMatrix, ci, cj),
                            GPUMatrix<ElemType>::AddElementToElement(0, *a.m_GPUMatrix, ai, aj, *c.m_GPUMatrix, ci, cj),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

//for each column of this, we add row slice of a starting from startIndex
template <class ElemType>
void Matrix<ElemType>::MinusOneAt(Matrix<ElemType>& a, const size_t position)
{
    DISPATCH_MATRIX_ON_FLAG(&a,
                            &a,
                            CPUMatrix<ElemType>::MinusOneAt(*a.m_CPUMatrix, position),
                            GPUMatrix<ElemType>::MinusOneAt(*a.m_GPUMatrix, position),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

/// <summary>Matrix-scalar multiply with col-major matrices: c = alpha * a</summary>
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void Matrix<ElemType>::Scale(ElemType alpha, const Matrix<ElemType>& a, Matrix<ElemType>& c)
{
    DecideAndMoveToRightDevice(c, a);

    c.SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    if (alpha == 0)
    {
        c.Resize(a);
        c.SetValue(0); // this is a little faster, and also does not propagate NaNs, which we'd expect from 'beta' parameters
        return;
    }
    else
        DISPATCH_MATRIX_ON_FLAG(&c,
                                &c,
                                CPUMatrix<ElemType>::Scale(alpha, *a.m_CPUMatrix, *c.m_CPUMatrix),
                                GPUMatrix<ElemType>::Scale(alpha, *a.m_GPUMatrix, *c.m_GPUMatrix),
                                NOT_IMPLEMENTED, * c.m_GPUSparseMatrix = (*a.m_GPUSparseMatrix) * alpha);
}

/// <summary>Matrix-scalar multiply with col-major matrices: a = alpha * a</summary>
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
template <class ElemType>
void Matrix<ElemType>::Scale(ElemType alpha, Matrix<ElemType>& a)
{
    if (alpha == 0)
        a.SetValue(0); // this is a little faster, and also does not propagate NaNs, which we'd expect from 'beta' parameters
    else if (a.IsEmpty())
        return;
    else
        DISPATCH_MATRIX_ON_FLAG(&a,
                                &a,
                                CPUMatrix<ElemType>::Scale(alpha, *a.m_CPUMatrix),
                                GPUMatrix<ElemType>::Scale(alpha, *a.m_GPUMatrix),
                                NOT_IMPLEMENTED,
                                GPUSparseMatrix<ElemType>::Scale(alpha, *a.m_GPUSparseMatrix));
}

/// <summary>Matrix scalar matrix multiply with col-major matrices: a = alpha[0,0] * a</summary>
/// <param name="alpha">1x1 matrix</param>
/// <param name="a">Input matrix</param>
template <class ElemType>
void Matrix<ElemType>::Scale(const Matrix<ElemType>& alpha, Matrix<ElemType>& a)
{
    if (a.IsEmpty())
        return;

    DecideAndMoveToRightDevice(a, alpha);

    if (a.GetMatrixType() != alpha.GetMatrixType())
        NOT_IMPLEMENTED;

    DISPATCH_MATRIX_ON_FLAG(&a,
                            nullptr,
                            CPUMatrix<ElemType>::Scale(*alpha.m_CPUMatrix, *a.m_CPUMatrix),
                            GPUMatrix<ElemType>::Scale(*alpha.m_GPUMatrix, *a.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::InnerProduct(const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c, const bool isColWise)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProduct:  one of the input matrix is empty.");

    DecideAndMoveToRightDevice(a, b, c);

    // TODO: consider swapping the arguments in this case
    if (b.GetMatrixType() != DENSE) // only support a being sparse/dense. Both b and c should be dense
        NOT_IMPLEMENTED;

    c.SwitchToMatrixType(b.GetMatrixType(), b.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&a,
                            &a,
                            CPUMatrix<ElemType>::InnerProduct(*a.m_CPUMatrix, *b.m_CPUMatrix, *c.m_CPUMatrix, isColWise),
                            GPUMatrix<ElemType>::InnerProduct(*a.m_GPUMatrix, *b.m_GPUMatrix, *c.m_GPUMatrix, isColWise),
                            CPUSparseMatrix<ElemType>::InnerProduct(*a.m_CPUSparseMatrix, *b.m_CPUMatrix, *c.m_CPUMatrix, isColWise),
                            GPUSparseMatrix<ElemType>::InnerProduct(*a.m_GPUSparseMatrix, *b.m_GPUMatrix, *c.m_GPUMatrix, isColWise));
}

template <class ElemType>
ElemType Matrix<ElemType>::InnerProductOfMatrices(const Matrix<ElemType>& a, const Matrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProductOfMatrices:  one of the input matrices is empty.");

    DecideAndMoveToRightDevice(a, b);

    if (a.GetMatrixType() == b.GetMatrixType())
    {
        DISPATCH_MATRIX_ON_FLAG(&a,
                                nullptr,
                                return CPUMatrix<ElemType>::InnerProductOfMatrices(*a.m_CPUMatrix, *b.m_CPUMatrix),
                                return GPUMatrix<ElemType>::InnerProductOfMatrices(*a.m_GPUMatrix, *b.m_GPUMatrix),
                                NOT_IMPLEMENTED,
                                NOT_IMPLEMENTED);
    }
    else
    {
        DISPATCH_MATRIX_ON_FLAG(&a,
                                nullptr,
                                NOT_IMPLEMENTED,
                                return GPUSparseMatrix<ElemType>::InnerProductOfMatrices(*a.m_GPUMatrix, *b.m_GPUSparseMatrix),
                                NOT_IMPLEMENTED,
                                return GPUSparseMatrix<ElemType>::InnerProductOfMatrices(*a.m_GPUSparseMatrix, *b.m_GPUMatrix));
    }
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignInnerProductOfMatrices(const Matrix<ElemType>& a, const Matrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProductOfMatrices:  one of the input matrices is empty.");

    Resize(1, 1);

    DecideAndMoveToRightDevice(a, b, *this);

    if (a.GetMatrixType() == b.GetMatrixType())
    {
        SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

        DISPATCH_MATRIX_ON_FLAG(&a,
                                this,
                                m_CPUMatrix->SetValue(CPUMatrix<ElemType>::InnerProductOfMatrices(*a.m_CPUMatrix, *b.m_CPUMatrix)),
                                m_GPUMatrix->AssignInnerProductOfMatrices(*a.m_GPUMatrix, *b.m_GPUMatrix),
                                NOT_IMPLEMENTED,
                                NOT_IMPLEMENTED);
    }
    else
    {
        NOT_IMPLEMENTED;
    }

    return *this;
}

template <class ElemType>
void Matrix<ElemType>::ElementWisePower(ElemType alpha, const Matrix<ElemType>& a, Matrix<ElemType>& c)
{
    if (a.IsEmpty())
        return;

    DecideAndMoveToRightDevice(a, c);
    c.SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&c,
                            nullptr,
                            CPUMatrix<ElemType>::ElementWisePower(alpha, *a.m_CPUMatrix, *c.m_CPUMatrix),
                            GPUMatrix<ElemType>::ElementWisePower(alpha, *a.m_GPUMatrix, *c.m_GPUMatrix),
                            NOT_IMPLEMENTED,
                            GPUSparseMatrix<ElemType>::ElementWisePower(alpha, *a.m_GPUSparseMatrix, *c.m_GPUSparseMatrix));
}

template <class ElemType>
bool Matrix<ElemType>::AreEqual(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const ElemType threshold /*= 1e-8*/)
{
    if (a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
        return false;

    DecideAndMoveToRightDevice(a, b);

    if (a.GetMatrixType() == b.GetMatrixType())
    {
        DISPATCH_MATRIX_ON_FLAG(&a,
                                nullptr,
                                return CPUMatrix<ElemType>::AreEqual(*a.m_CPUMatrix, *b.m_CPUMatrix, threshold),
                                return GPUMatrix<ElemType>::AreEqual(*a.m_GPUMatrix, *b.m_GPUMatrix, threshold),
                                return CPUSparseMatrix<ElemType>::AreEqual(*a.m_CPUSparseMatrix, *b.m_CPUSparseMatrix, threshold),
                                return GPUSparseMatrix<ElemType>::AreEqual(*a.m_GPUSparseMatrix, *b.m_GPUSparseMatrix, threshold));
    }
    else
    {
        DISPATCH_MATRIX_ON_FLAG(&a,
                                nullptr,
                                NOT_IMPLEMENTED;
                                return false,
                                       return GPUSparseMatrix<ElemType>::AreEqual(*a.m_GPUMatrix, *b.m_GPUSparseMatrix, threshold),
                                       NOT_IMPLEMENTED;
                                return false,
                                       return GPUSparseMatrix<ElemType>::AreEqual(*a.m_GPUSparseMatrix, *b.m_GPUMatrix, threshold));
    }
}

template <class ElemType>
bool Matrix<ElemType>::HasElement(const Matrix<ElemType>& a, const ElemType value)
{
    if (a.IsEmpty())
        return false;

    DISPATCH_MATRIX_ON_FLAG(&a,
                            &a,
                            return CPUMatrix<ElemType>::HasElement(*a.m_CPUMatrix, value),
                            return GPUMatrix<ElemType>::HasElement(*a.m_GPUMatrix, value),
                            NOT_IMPLEMENTED;
                            return false,
                                   NOT_IMPLEMENTED;
                            return false);
}

// diagnostics helper to check if matrix has a NaN
// This is very slow.
template <class ElemType>
bool Matrix<ElemType>::HasNan(const char* name) const
{
    // Not implemented for sparse matrices.
    // Return false as a workaround to at
    // least evaluate the dense matrices.
    if (m_matrixType == MatrixType::SPARSE)
        return false;

    if (IsEmpty())
        return false;

    // if GPU then first detect NaN there, will be faster
    if (GetDeviceId() != CPUDEVICE)
    {
        Matrix<ElemType> sum(GetDeviceId());
        sum.AssignSumOfElements(*this);
        auto x = sum.Get00Element();
        if (!std::isnan(x))
            return false;
    }

    // const auto & us = *this;
    const Matrix<ElemType>& us = *this;

    foreach_coord (i, j, us)
        if (std::isnan(us(i, j)))
        {
            fprintf(stderr, "HasNan: NaN detected at %s (%ld,%ld) in (%d,%d) matrix\n", name, i, j, (int) GetNumRows(), (int) GetNumCols());
            return true;
        }
    return false;
}
#define CheckNan(m) m.HasNan(#m)

// another diagnostics helper to check if matrix has a NaN
// This is used at load and save time. This test is slow.

template <class ElemType>
size_t Matrix<ElemType>::CountNanInf() const
{
    const auto& us = *this;
    size_t n = 0; // number of NaNs/INF found
    foreach_coord (i, j, us)
    {
        auto val = us(i, j);
        if (std::isnan(val) || !std::isfinite(val))
            n++;
    }
    return n;
}

// TODO: these are scalar operations--why are they in Matrix?
template <class ElemType>
ElemType Matrix<ElemType>::Exp10(ElemType num)
{
    return (ElemType) exp(num * 2.302585093);
}

template <class ElemType>
ElemType Matrix<ElemType>::Mod(ElemType x, ElemType y)
{
    assert(y > 0);
    if (y <= 0)
        LogicError("y is smaller than zero");

    return x - y * floor(x / y);
}

// TODO: use static LogAdd() as defined in TensorOps.h
//       Not doing this currently because that one uses ElemType for all ops, while this one uses double inside. Must compare before making this change.
template <class ElemType>
ElemType Matrix<ElemType>::LogAdd(ElemType x, ElemType y)
{
    ElemType temp, diff, z;

    if (x < y)
    {
        temp = x;
        x = y;
        y = temp; // TODO: ::swap(x,y)?
    }
    diff = y - x;
    if (diff < MINLOGEXP)
    {
        return (ElemType)((x < LSMALL) ? LZERO : x);
    }
    else
    {
        z = exp(diff);
        return (ElemType)(x + log(1.0 + z));
    }
}

//Matrix<ElemType>& Matrix<ElemType>::Shift(const Matrix<ElemType>& a, size_t shift)
//[this]= (a right shift by n), padded with zeros
// shift left, shift needs to be negative value
// shift right, shift needs to be positive value
// BUGBUG: Leaves uninitialized values in the opened-up columns.
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::Shift(const Matrix<ElemType>& a, int shift)
{
    if (a.IsEmpty())
        LogicError("Shift: Matrix is empty.");
    else
        LogicError("Shift: BUGBUG This function currently leaves uninitialized values. Fix the code or contact fseide@microsoft.com.");

    auto& us = *this;
    if (this != &a)
    {
        Resize(a.GetNumRows(), a.GetNumCols());
    }

    long n = (long) GetNumCols();

    if (shift >= 0 && shift < n)
        us.ColumnSlice(shift, n - shift).AssignValuesOf(a.ColumnSlice(0, n - shift));
    if (shift < 0 && shift > -n)
        us.ColumnSlice(0, n + shift).AssignValuesOf(a.ColumnSlice(-shift, n + shift));
    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignElementProductOfWithShiftNeg(const Matrix<ElemType>& a, const Matrix<ElemType>& b, size_t shift, size_t negnumber)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOfWithShiftNeg: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    if (a.GetNumRows() != 1)
        InvalidArgument("AssignElementProductOfWithShiftNeg: The input matrix must be a row vector.");

    DecideAndMoveToRightDevice(a, b, *this);
    if (!(a.GetMatrixType() == b.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignElementProductOfWithShiftNeg(*a.m_CPUMatrix, *b.m_CPUMatrix, shift, negnumber),
                            m_GPUMatrix->AssignElementProductOfWithShiftNeg(*a.m_GPUMatrix, *b.m_GPUMatrix, shift, negnumber),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
    return *this;
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignInnerProductOfWithShiftNeg(const Matrix<ElemType>& a, const Matrix<ElemType>& b, const bool isColWise, size_t shift, size_t negnumber)
{
    InnerProductWithShiftNeg(a, b, *this, isColWise, shift, negnumber);
    return *this;
}

template <class ElemType>
void Matrix<ElemType>::InnerProductWithShiftNeg(const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c, const bool isColWise, size_t shift, size_t negnumber)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProduct:  one of the input matrix is empty.");

    DecideAndMoveToRightDevice(a, b, c);

    if (a.GetMatrixType() != b.GetMatrixType())
        NOT_IMPLEMENTED;

    c.SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&c,
                            &c,
                            CPUMatrix<ElemType>::InnerProductWithShiftNeg(*a.m_CPUMatrix, *b.m_CPUMatrix, *c.m_CPUMatrix, isColWise, shift, negnumber),
                            GPUMatrix<ElemType>::InnerProductWithShiftNeg(*a.m_GPUMatrix, *b.m_GPUMatrix, *c.m_GPUMatrix, shift, negnumber),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::GetARowByIndex(const Matrix<ElemType>& a, size_t index)
{
    if (a.IsEmpty())
        LogicError("GetARowByIndex: Matrix is empty.");

    // WARNING: a and this must have same type
    if (!(GetMatrixType() == a.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->GetARowByIndex(*a.m_CPUMatrix, index),
                            m_GPUMatrix->GetARowByIndex(*a.m_GPUMatrix, index),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

template <class ElemType>
void Matrix<ElemType>::ConductRowElementMultiplyWithShift(const Matrix<ElemType>& a, const Matrix<ElemType>& b, Matrix<ElemType>& c, size_t shift, bool bFirstmatrixfixed)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProduct:  one of the input matrix is empty.");

    DecideAndMoveToRightDevice(a, b, c);

    if (a.GetMatrixType() != b.GetMatrixType())
        NOT_IMPLEMENTED;

    c.SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&c,
                            &c,
                            CPUMatrix<ElemType>::ConductRowElementMultiplyWithShift(*a.m_CPUMatrix, *b.m_CPUMatrix, *c.m_CPUMatrix, shift, bFirstmatrixfixed),
                            GPUMatrix<ElemType>::ConductRowElementMultiplyWithShift(*a.m_GPUMatrix, *b.m_GPUMatrix, *c.m_GPUMatrix, shift, bFirstmatrixfixed),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignElementProductOfWithShift(const Matrix<ElemType>& a, const Matrix<ElemType>& b, size_t shift)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOfWithShift: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    if (a.GetNumRows() != 1)
        InvalidArgument("AssignElementProductOfWithShiftNeg: The input matrix must be a row vector.");

    DecideAndMoveToRightDevice(a, b, *this);
    if (!(a.GetMatrixType() == b.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(a.GetMatrixType(), a.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignElementProductOfWithShift(*a.m_CPUMatrix, *b.m_CPUMatrix, shift),
                            m_GPUMatrix->AssignElementProductOfWithShift(*a.m_GPUMatrix, *b.m_GPUMatrix, shift),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
    return *this;
}

template <class ElemType>
void Matrix<ElemType>::RCRFBackwardCompute(const Matrix<ElemType>& alpha, Matrix<ElemType>& beta,
                                           Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls,
                                           const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const int shift)
{
    DecideAndMoveToRightDevice(alpha, beta);
    functionValues._transferToDevice(alpha.GetDeviceId());
    beta._transferToDevice(alpha.GetDeviceId());

    DISPATCH_MATRIX_ON_FLAG(&alpha,
                            &beta,
                            CPUMatrix<ElemType>::RCRFBackwardCompute(
                                *alpha.m_CPUMatrix,
                                *beta.m_CPUMatrix,
                                *lbls.m_CPUMatrix,
                                *pair_scores.m_CPUMatrix),
                            GPUMatrix<ElemType>::RCRFBackwardCompute(
                                *alpha.m_GPUMatrix,
                                *beta.m_GPUMatrix,
                                *lbls.m_GPUMatrix,
                                *pos_scores.m_GPUMatrix,
                                *pair_scores.m_GPUMatrix, shift),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::RCRFTransGrdCompute(const Matrix<ElemType>& lbls,
                                           const Matrix<ElemType>& alpha,
                                           const Matrix<ElemType>& beta,
                                           const Matrix<ElemType>& pair_scores,
                                           Matrix<ElemType>& grd,
                                           const int startLbl,
                                           const int shift)
{
    DecideAndMoveToRightDevice(alpha, grd);
    grd._transferToDevice(alpha.GetDeviceId());

    DISPATCH_MATRIX_ON_FLAG(&alpha,
                            &grd,
                            CPUMatrix<ElemType>::RCRFTransGrdCompute(
                                *lbls.m_CPUMatrix,
                                *alpha.m_CPUMatrix,
                                *beta.m_CPUMatrix,
                                *pair_scores.m_CPUMatrix,
                                *grd.m_CPUMatrix),
                            GPUMatrix<ElemType>::RCRFTransGrdCompute(
                                *lbls.m_GPUMatrix,
                                *alpha.m_GPUMatrix,
                                *beta.m_GPUMatrix,
                                *pair_scores.m_GPUMatrix,
                                *grd.m_GPUMatrix,
                                startLbl,
                                shift),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::DropFrame(const Matrix<ElemType>& label, const Matrix<ElemType>& gamma, const ElemType& threshhold)
{
    DecideAndMoveToRightDevice(*this, label, gamma);

    if (label.GetNumCols() != gamma.GetNumCols() || label.GetNumRows() != gamma.GetNumRows())
        LogicError("DropFrame: label matrix is not in the same size as gamm matrix.");
    SwitchToMatrixType(label.GetMatrixType(), label.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->DropFrame(*label.m_CPUMatrix, *gamma.m_CPUMatrix, threshhold),
                            m_GPUMatrix->DropFrame(*label.m_GPUMatrix, *gamma.m_GPUMatrix, threshhold),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);

    return *this;
}

/// <summary> c = alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignSequenceError(const ElemType hsmoothingWeight, const Matrix<ElemType>& label,
                                                        const Matrix<ElemType>& dnnoutput, const Matrix<ElemType>& gamma, ElemType alpha)
{
    DecideAndMoveToRightDevice(label, dnnoutput, gamma);

    if (!(label.GetMatrixType() == gamma.GetMatrixType()))
        NOT_IMPLEMENTED;

    SwitchToMatrixType(label.GetMatrixType(), label.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->AssignSequenceError(hsmoothingWeight, *label.m_CPUMatrix, *dnnoutput.m_CPUMatrix, *gamma.m_CPUMatrix, alpha),
                            m_GPUMatrix->AssignSequenceError(hsmoothingWeight, *label.m_GPUMatrix, *dnnoutput.m_GPUMatrix, *gamma.m_GPUMatrix, alpha),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
    return *this;
}

// Calculate CTC score
// prob (input): the posterior output from the network
// alpha, beta (output): alpha and beta for forward-backward calculation. 
// phoneSeq (input): phone ID sequence for each utterance in this minibatch, each col is one utterance 
// phoneBound (input): phone boundary (frame index) of each phone for each utterance in this minibatch, each col is one utterance 
// totalScore (output): total CTC score
// uttToChanInd (input):  map from utterance ID to minibatch channel ID. We need this because each channel may contain more than one utterance.
// uttBeginFrame(input): the positon of the first frame of each utterance in the minibatch channel. We need this because each channel may contain more than one utterance.
// uttFrameNum (input): the frame number of each utterance. The size of this vector =  the number of all utterances in this minibatch
// uttPhoneNum (input): the phone number of each utterance. The size of this vector =  the number of all utterances in this minibatch
// numParallelSequences (input): num of parallel sequences
// mbsize (input): the maximum channel frame number
// blankTokenId (input): id of the CTC blank token
// delayConstraint -- label output delay constraint introduced during training that allows to have shorter delay during inference. This using the original time information to enforce that CTC tokens only get aligned within a time margin.
//      Setting this parameter smaller will result in shorted delay between label output during decoding, yet may hurt accuracy.
//      delayConstraint=-1 means no constraint
template<class ElemType>
Matrix<ElemType>& Matrix<ElemType>::AssignCTCScore(const Matrix<ElemType>& prob, Matrix<ElemType>& alpha, Matrix<ElemType>& beta,
    const Matrix<ElemType>& phoneSeq, const Matrix<ElemType>& phoneBound, ElemType &totalScore, const std::vector<size_t> & uttToChanInd,
    const std::vector<size_t> & uttBeginFrame, const std::vector<size_t> & uttFrameNum, const std::vector<size_t> & uttPhoneNum,
    const size_t numParallelSequences, const size_t mbsize, const size_t blankTokenId, const int delayConstraint, const bool isColWise)
{
    DecideAndMoveToRightDevice(prob, *this);
    alpha.Resize(phoneSeq.GetNumRows(), prob.GetNumCols());
    beta.Resize(phoneSeq.GetNumRows(), prob.GetNumCols());
    Resize(prob.GetNumRows(), prob.GetNumCols());

    alpha.SetValue(LZERO);
    beta.SetValue(LZERO);
    SetValue(LZERO);
    SwitchToMatrixType(prob.GetMatrixType(), prob.GetFormat(), false);

    DISPATCH_MATRIX_ON_FLAG(&prob,
        this,
        this->m_CPUMatrix->AssignCTCScore(*prob.m_CPUMatrix, *alpha.m_CPUMatrix, *beta.m_CPUMatrix, *phoneSeq.m_CPUMatrix, *phoneBound.m_CPUMatrix, totalScore,
            uttToChanInd, uttBeginFrame, uttFrameNum, uttPhoneNum, numParallelSequences, mbsize, blankTokenId, delayConstraint, isColWise),
        this->m_GPUMatrix->AssignCTCScore(*prob.m_GPUMatrix, *alpha.m_GPUMatrix, *beta.m_GPUMatrix, *phoneSeq.m_GPUMatrix, *phoneBound.m_GPUMatrix, totalScore,
            uttToChanInd, uttBeginFrame, uttFrameNum, uttPhoneNum, numParallelSequences, mbsize, blankTokenId, delayConstraint, isColWise),
        NOT_IMPLEMENTED,
        NOT_IMPLEMENTED
    );

    return *this;
}

#pragma endregion Static BLAS Functions

// TensorView currently does not interface with sparse matrices. For now, we just catch this and throw.
template <class ElemType>
static bool VerifyIsDense(const Matrix<ElemType>& a)
{
    if (a.GetMatrixType() != DENSE)
        RuntimeError("TensorOp: Tensor operations are currently not supported for sparse matrices.");
    return true;
}

template <class ElemType>
void Matrix<ElemType>::TensorOp(ElemType beta, const Matrix<ElemType>& a, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                const array<size_t, 2>& offsets,
                                const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                                const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
    VerifyIsDense(*this) && VerifyIsDense(a);

    DecideAndMoveToRightDevice(*this, a);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->TensorOp(beta, *a.m_CPUMatrix, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides),
                            m_GPUMatrix->TensorOp(beta, *a.m_GPUMatrix, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::TensorOp(ElemType beta, const Matrix<ElemType>& a, const Matrix<ElemType>& b, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                const array<size_t, 3>& offsets,
                                const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
                                const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides)
{
    VerifyIsDense(*this) && VerifyIsDense(a) && VerifyIsDense(b);

    DecideAndMoveToRightDevice(*this, a, b);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->TensorOp(beta, *a.m_CPUMatrix, *b.m_CPUMatrix, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides),
                            m_GPUMatrix->TensorOp(beta, *a.m_GPUMatrix, *b.m_GPUMatrix, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}

template <class ElemType>
void Matrix<ElemType>::TensorOp(ElemType beta, const Matrix<ElemType>& a, const Matrix<ElemType>& b, const Matrix<ElemType>& c, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                const array<size_t, 4>& offsets,
                                const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
                                const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides)
{
    VerifyIsDense(*this) && VerifyIsDense(a) && VerifyIsDense(b) && VerifyIsDense(c);

    DecideAndMoveToRightDevice(*this, a, b, c);

    DISPATCH_MATRIX_ON_FLAG(this,
                            this,
                            m_CPUMatrix->TensorOp(beta, *a.m_CPUMatrix, *b.m_CPUMatrix, *c.m_CPUMatrix, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides),
                            m_GPUMatrix->TensorOp(beta, *a.m_GPUMatrix, *b.m_GPUMatrix, *c.m_GPUMatrix, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides),
                            NOT_IMPLEMENTED,
                            NOT_IMPLEMENTED);
}
template <class ElemType>
void Matrix<ElemType>::TensorArgOp(const Matrix<ElemType>& a, ElementWiseOperator reductionOp,
                                   const array<size_t, 2>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
    VerifyIsDense(*this) && VerifyIsDense(a);

    DecideAndMoveToRightDevice(*this, a);

    DISPATCH_MATRIX_ON_FLAG(this,
        this,
        m_CPUMatrix->TensorArgOp(*a.m_CPUMatrix, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides),
        m_GPUMatrix->TensorArgOp(*a.m_GPUMatrix, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides),
        NOT_IMPLEMENTED,
        NOT_IMPLEMENTED);
}

//template class Matrix<short>;
template class Matrix<float>;
template class Matrix<double>;

// We use Matrix<char> as the backing store for QuantizedMatrix, and also as a flag matrix.
// Let's explicitly instantiate the methods we need for that purpose
template Matrix<char>::Matrix(DEVICEID_TYPE);
template Matrix<char>::Matrix(Matrix<char>&&);
template Matrix<char>::Matrix(const size_t numRows, const size_t numCols, DEVICEID_TYPE deviceId, const MatrixType matrixType, const MatrixFormat matrixFormat, const size_t nnz);
template Matrix<char>::Matrix(const size_t numRows, const size_t numCols, char* pArray, DEVICEID_TYPE deviceId, const size_t matrixFlags, const size_t nnz);
template Matrix<char>::~Matrix();
template Matrix<char>& Matrix<char>::operator=(Matrix<char>&& moveFrom);
template char* Matrix<char>::Data() const;
template int Matrix<char>::GetDeviceId() const;
template size_t Matrix<char>::GetNumElements() const;
template Matrix<char> Matrix<char>::ColumnSlice(size_t startColumn, size_t numCols) const;
template void Matrix<char>::_transferToDevice(int id_to, bool isBeingMoved, bool emptyTransfer) const;
template void Matrix<char>::TransferToDeviceIfNotThere(int id_to, bool isBeingMoved, bool emptyTransfer, bool updatePreferredDevice) const;
template size_t Matrix<char>::GetNumRows() const;
template size_t Matrix<char>::GetNumCols() const;
template void Matrix<char>::SetValue(const char);
template void Matrix<char>::SetValue(size_t numRows, const size_t numCols, int deviceId, char* pArray, size_t matrixFlags, DataTransferer* transferer);
//template void Matrix<char>::SetValue(const Matrix<char>&, MatrixFormat);
template void Matrix<char>::SetValue(const Matrix<char>&);
template void Matrix<char>::AssignValuesOf(const Matrix<char>&);
template void Matrix<char>::CastAssignValuesOf(const MatrixBase& other);
template bool Matrix<char>::IsEmpty() const;
template void Matrix<char>::Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, bool growOnly);
template void Matrix<char>::Reshape(const size_t, const size_t);
template char* Matrix<char>::CopyToArray(void) const;

// Matrix<short> methods
template Matrix<short>::Matrix(DEVICEID_TYPE);
template Matrix<short>::Matrix(Matrix<short>&&);
template Matrix<short>::Matrix(const size_t numRows, const size_t numCols, DEVICEID_TYPE deviceId, const MatrixType matrixType, const MatrixFormat matrixFormat, const size_t nnz);
template Matrix<short>::Matrix(const size_t numRows, const size_t numCols, short* pArray, DEVICEID_TYPE deviceId, const size_t matrixFlags, const size_t nnz);
template Matrix<short>::~Matrix();
template Matrix<short>& Matrix<short>::operator=(Matrix<short>&& moveFrom);
template short* Matrix<short>::Data() const;
template int Matrix<short>::GetDeviceId() const;
template size_t Matrix<short>::GetNumElements() const;
template Matrix<short> Matrix<short>::ColumnSlice(size_t startColumn, size_t numCols) const;
template void Matrix<short>::_transferToDevice(int id_to, bool isBeingMoved, bool emptyTransfer) const;
template void Matrix<short>::TransferToDeviceIfNotThere(int id_to, bool isBeingMoved, bool emptyTransfer, bool updatePreferredDevice) const;
template size_t Matrix<short>::GetNumRows() const;
template size_t Matrix<short>::GetNumCols() const;
template void Matrix<short>::SetValue(const short);
template void Matrix<short>::SetValue(size_t numRows, const size_t numCols, int deviceId, short* pArray, size_t matrixFlags, DataTransferer* transferer);
//template void Matrix<short>::SetValue(const Matrix<short>&, MatrixFormat);
template void Matrix<short>::SetValue(const Matrix<short>&);
template void Matrix<short>::AssignValuesOf(const Matrix<short>&);
template void Matrix<short>::CastAssignValuesOf(const MatrixBase& other);
template bool Matrix<short>::IsEmpty() const;
template void Matrix<short>::Resize(const size_t numRows, const size_t numCols, const size_t numNZElemToReserve, bool growOnly);
template void Matrix<short>::Reshape(const size_t, const size_t);
template short* Matrix<short>::CopyToArray(void) const;

template Matrix<int>::Matrix(const size_t, const size_t, int*, DEVICEID_TYPE, const size_t, const size_t);

}}}
