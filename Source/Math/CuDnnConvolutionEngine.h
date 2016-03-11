//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ConvolutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class CuDnnConvolutionEngineFactory
{
public:
    static std::unique_ptr<ConvolutionEngine<ElemType>> Create(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, 
                                                               ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples,
                                                               PoolKind poolKind);
    static bool IsSupported(DEVICEID_TYPE deviceId);
};

//template <class ElemType>
//class CuDnnConvolutionEngineFactory : public ConvolutionEngineFactory<ElemType>
//{
//public:
//    using Base = ConvolutionEngineFactory<ElemType>;
//    using typename Base::Tensor4D;
//    using typename Base::Tensor4DPtr;
//    using typename Base::Filter;
//    using typename Base::FilterPtr;
//    using typename Base::ConvDesc;
//    using typename Base::ConvDescPtr;
//    using typename Base::PoolDesc;
//    using typename Base::PoolDescPtr;
//
//    using typename Base::ConvEnginePtr;
//    using typename Base::PoolEnginePtr;
//
//public:
//    Tensor4DPtr CreateTensor(size_t w, size_t h, size_t c, size_t n) override;
//    FilterPtr CreateFilter(size_t w, size_t h, size_t c, size_t k) override;
//    ConvDescPtr CreateConvDescriptor(const Tensor4D& inT, const Filter& filterT,
//                                     size_t wStride, size_t hStride, bool padding) override;
//    PoolDescPtr CreatePoolDescriptor(typename PoolDesc::PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad) override;
//
//    ConvEnginePtr CreateConvEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, BatchNormImpl bnImpl) override;
//    PoolEnginePtr CreatePoolEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout) override;
//
//    static bool IsSupported(DEVICEID_TYPE deviceId);
//};
//

// REVIEW alexeyk: wrong place. It is currently used only in unit tests but I can't add it there because of the build issues.
// Timer that can be used to measure CUDA calls. 
// Uses CUDA event and will synchronize(!) the stream when Stop is called.
class MATH_API CudaTimer
{
public:
    CudaTimer(): m_start(nullptr), m_stop(nullptr)
    {
    }
    ~CudaTimer();
    void Start();
    void Stop();
    float Elapsed();

    DISABLE_COPY_AND_MOVE(CudaTimer);
private:
    void* m_start;
    void* m_stop;
};
} } }
