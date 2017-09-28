#pragma once

#include "ColumnQuantizer.h"
#include "QuantizedMatrix.h"

#ifdef _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#else // no DLLs on Linux
#define MATH_API
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class MATH_API MatrixQuantizerImpl
{
public:
    static MatrixQuantizerImpl<ElemType>* Create(int deviceId, bool useAsync);
    virtual ~MatrixQuantizerImpl()
    {
    }

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(MatrixQuantizerImpl);

    virtual void QuantizeAsync(const Matrix<ElemType>& inMatrix, const Matrix<ElemType>& inResidual, QuantizedMatrix<ElemType>& outQMatrix, Matrix<ElemType>& outResidual, bool zeroThresholdFor1Bit) = 0;
    virtual void WaitQuantizeAsyncDone() = 0;

    virtual void UnquantizeAsync(QuantizedMatrix<ElemType>& inQMatrix, Matrix<ElemType>& outMatrix, bool add = false) = 0;
    virtual void WaitUnquantizeAsyncDone() = 0;

    int GetDeviceId() const
    {
        return m_deviceId;
    }

protected:
    MatrixQuantizerImpl(int deviceId)
        : m_deviceId(deviceId)
    {
    }

private:
    int m_deviceId;
};

// This type records and synchronizes events on the main
// matrix computation work stream
class MATH_API MatrixComputeStreamEvent
{
public:
    static MatrixComputeStreamEvent* Create(int deviceId);
    virtual ~MatrixComputeStreamEvent();

    virtual void SynchronizeEvent();

    template <typename ElemType>
    void SynchronizeQuantizationComputeStreamWithEvent();

    template <typename ElemType>
    void SynchronizeDataTransferFetchStreamWithEvent();

protected:
    MatrixComputeStreamEvent(int deviceId);

protected:
    int m_deviceId;
};
} } }
