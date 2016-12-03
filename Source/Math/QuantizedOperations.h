//
// Copyright (c) Microsoft. All rights resized.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once
#include "Quantizers.h"

namespace Microsoft { namespace MSR { namespace CNTK {


// Quantized product of two dense matrices A and B, where each matrix has its own quantizer.
// This class handles quantization of both matrices, product and de-quantization of the result.
// Other implementations should inherit from this class or extract common methods to the base class and inherit from the base.
template <class ElemType>
class QuantizedMultiplier
{
    // Quantizers for matrices A and B
    shared_ptr<QuantizerBase<ElemType, short>> m_pQuantizerA;
    shared_ptr<QuantizerBase<ElemType, short>> m_pQuantizerB;

    // Placeholders for quantized matrices A and B
    vector<short> m_pMatA, m_pMatB;

    // Whether matrices A and B are constant (i.e. weights)
    // If the matrix is constant, the size of the underlying container for quatized values will be preserved for
    // the lifespan of the object
    bool m_isAConstant;
    bool m_isBConstant;

    bool m_firstPass;

public: 
    QuantizedMultiplier(shared_ptr<QuantizerBase<ElemType, short>> pQuantizerA, bool isAConstant, shared_ptr<QuantizerBase<ElemType, short>> pQuantizerB, bool isBConstant) :
        m_pQuantizerA(pQuantizerA), m_pQuantizerB(pQuantizerB), m_isAConstant(isAConstant), m_isBConstant(isBConstant), m_firstPass(true)
    {
        if (isAConstant && isBConstant)
            LogicError("Quantized multiplication is applied to two constant matrices -- it is highly inefficient. Better approach is to replace the operation with the resulting matrix.");
    };
    QuantizedMultiplier(shared_ptr<QuantizerBase<ElemType, short>> pQuantizerA, shared_ptr<QuantizerBase<ElemType, short>> pQuantizerB) :
        QuantizedMultiplier(pQuantizerA, false, pQuantizerB, false)
    {
    };

    // A[m,k]*B[k,n] = C[m,n]
    void Multiply(int m, int n, int k, ElemType* A, ElemType* B, ElemType* C)
    {
        // Quantize
        if (!m_isAConstant || m_firstPass)
        {
            m_pMatA.resize(m*k);
            ArrayRef<short> refMatA(m_pMatA.data(), m_pMatA.size());
            m_pQuantizerA->Quantize(ArrayRef<ElemType>(A, m_pMatA.size()), refMatA);
        }
        
        if (!m_isBConstant || m_firstPass)
        {
            m_pMatB.resize(n*k);
            ArrayRef<short> refMatB(m_pMatB.data(), m_pMatB.size());
            m_pQuantizerB->Quantize(ArrayRef<ElemType>(B, m_pMatB.size()), refMatB);
        }

        m_firstPass = false;

        // Do multiply
        // Naive inefficient product, just for demonstation
        // TODO: replace with an efficient version, e.g. IPG, block multiplier, Eigen, gemmlowp, etc.
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < n; j++)
            {
                int dotProduct=0;
                for (size_t l = 0; l < k; l++)
                {
                    // CNTK is using column-major storage
                    dotProduct += m_pMatA[i + l*m] * m_pMatB[l + k*j];
                }
                C[i + j*m] = (ElemType)dotProduct;
            }

        // De-quantize
        int mn = m*n;
        m_pQuantizerB->Dequantize(C, C, mn);
        m_pQuantizerA->Dequantize(C, C, mn);
    }

    void SetIsAConstant(bool v) { m_isAConstant = v; }
    void SetIsBConstant(bool v) { m_isBConstant = v; }
};

}}}