//
// <copyright file="CuDnnConvolutionEngine.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "ConvolutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class CuDnnConvolutionEngineFactory : public ConvolutionEngineFactory<ElemType>
{
public:
    using Base = ConvolutionEngineFactory<ElemType>;
    using typename Base::Tensor4D;
    using typename Base::Tensor4DPtr;
    using typename Base::Filter;
    using typename Base::FilterPtr;
    using typename Base::ConvDesc;
    using typename Base::ConvDescPtr;
    using typename Base::PoolDesc;
    using typename Base::PoolDescPtr;

    using typename Base::ConvEnginePtr;
    using typename Base::PoolEnginePtr;

public:
    Tensor4DPtr CreateTensor(size_t w, size_t h, size_t c, size_t n) override;
    FilterPtr CreateFilter(size_t w, size_t h, size_t c, size_t k) override;
    ConvDescPtr CreateConvDescriptor(const Tensor4D& inT, const Filter& filterT,
                                     size_t wStride, size_t hStride, bool padding) override;
    PoolDescPtr CreatePoolDescriptor(typename PoolDesc::PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad) override;

    ConvEnginePtr CreateConvEngine(DEVICEID_TYPE deviceId, size_t maxTempMemSizeInSamples) override;
    PoolEnginePtr CreatePoolEngine(DEVICEID_TYPE deviceId) override;

    static bool IsSupported(DEVICEID_TYPE deviceId);
};
} } }
