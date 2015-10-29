//
// <copyright file="CuDnnConvolutionEngine.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "ConvolutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class CuDnnConvolutionEngineFactory : public ConvolutionEngineFactory<ElemType>
    {
    public:
        CuDnnConvolutionEngineFactory(DEVICEID_TYPE deviceId)
            : ConvolutionEngineFactory<ElemType>(deviceId)
        {
        }

    public:
        Tensor4DPtr CreateTensor(size_t w, size_t h, size_t c, size_t n) override;
        FilterPtr CreateFilter(size_t w, size_t h, size_t c, size_t k) override;
        ConvDescPtr CreateConvDescriptor(const Tensor4D& inT, const Filter& filterT, 
            size_t wStride, size_t hStride, bool padding) override;
        PoolDescPtr CreatePoolDescriptor(PoolDesc::PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad) override;

        ConvEnginePtr CreateConvEngine(size_t maxTempMemSizeInSamples) override;
        PoolEnginePtr CreatePoolEngine() override;

        static bool IsSupported();
    };
}}}
