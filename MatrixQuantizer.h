//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under custom Microsoft Research License Terms for
// 1-bit Stochastic Gradient Descent.
// See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ColumnQuantizer.h"
#include "QuantizedMatrix.h"
#include "MatrixQuantizerImpl.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This type does the quantization on a matrix
// This is a technique to reduce the cost of communicating
// the gradient matrices during aggregation across all nodes in
// data-parallel SGD training, at the end of each minibatch.
// Refer this paper http://research.microsoft.com/apps/pubs/?id=230137
// for details.
template <class ElemType>
class MatrixQuantizer final
{
public:
    MatrixQuantizer(size_t numRows, size_t numCols, int deviceId, bool useAsync)
    {
        m_residual.reset(new Matrix<ElemType>(numRows, numCols, deviceId, DENSE));
        m_quantizerImpl.reset(MatrixQuantizerImpl<ElemType>::Create(deviceId, useAsync));
    }

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(MatrixQuantizer);

    void QuantizeAsync(const Matrix<ElemType>& inMatrix, QuantizedMatrix<ElemType>& outQMatrix, bool zeroThresholdFor1Bit)
    {
        m_quantizerImpl->QuantizeAsync(inMatrix, *m_residual, outQMatrix, *m_residual, zeroThresholdFor1Bit);
    }

    void WaitQuantizeAsyncDone()
    {
        m_quantizerImpl->WaitQuantizeAsyncDone();
    }

    void UnquantizeAsync(QuantizedMatrix<ElemType>& inQMatrix, Matrix<ElemType>& outMatrix, bool add = false)
    {
        m_quantizerImpl->UnquantizeAsync(inQMatrix, outMatrix, add);
    }

    void WaitUnquantizeAsyncDone()
    {
        m_quantizerImpl->WaitUnquantizeAsyncDone();
    }

    int GetDeviceId() const
    {
        return m_residual->GetDeviceId();
    }

    void ResetResidue()
    {
        m_residual->SetValue(0.0);
    }

    const Matrix<ElemType>& GetResidualMatrix() const
    {
        return *m_residual;
    }

private:
    std::unique_ptr<MatrixQuantizerImpl<ElemType>> m_quantizerImpl;

    // the residual matrix
    std::unique_ptr<Matrix<ElemType>> m_residual;
};

} } }
