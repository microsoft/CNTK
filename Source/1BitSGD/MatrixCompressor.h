//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ColumnQuantizer.h"
#include "QuantizedMatrix.h"
#include "MatrixQuantizerImpl.h"
#include "c_allreduce_ring.h"

namespace Microsoft {namespace MSR {namespace CNTK {
// TopK
class MatrixCompressorBase
{};

template <class ElemType>
class MatrixCompressor final : public MatrixCompressorBase
{
public:
    MatrixCompressor(int deviceId, bool useAsync)
    {
        m_quantizerImpl.reset(MatrixQuantizerImpl<ElemType>::Create(deviceId, useAsync));
    }

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(MatrixCompressor);

    void TopKAsync(const Matrix<ElemType>& inMatrix, const Matrix<ElemType>& inResidual, struct stream &sendbuf, Matrix<ElemType>& outResidual, int topK)
    {
        m_quantizerImpl->TopKAsync(inMatrix, inResidual, sendbuf, outResidual, topK);
    }

    void WaitTopKAsyncDone()
    {
        m_quantizerImpl->WaitTopKAsyncDone();
    }

    void UnTopKAsync(struct stream &recvbuf, Matrix<ElemType>& outMatrix)
    {
        m_quantizerImpl->UnTopKAsync(recvbuf, outMatrix);
    }

    void WaitUnTopKAsyncDone()
    {
        m_quantizerImpl->WaitUnTopKAsyncDone();
    }

    void AllReduce(const struct stream *sendbuf, struct stream *recvbuf, unsigned dim)
    {
        c_allreduce_ring<unsigned, ElemType>(sendbuf, recvbuf, dim);
    }

    int GetDeviceId() const
    {
        return m_quantizerImpl->GetDeviceId();
    }

private:
    std::unique_ptr<MatrixQuantizerImpl<ElemType>> m_quantizerImpl;
};

} } }
