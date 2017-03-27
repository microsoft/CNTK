//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ConvolutionEngine.h"
#include "BatchNormalizationEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class CuDnnConvolutionEngineFactory
{
public:
    static std::unique_ptr<ConvolutionEngine<ElemType>> Create(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId,
                                                               ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples,
                                                               PoolKind poolKind, bool forceDeterministicAlgorithms, bool poolPadMode);
    static bool IsSupported(DEVICEID_TYPE deviceId, ConvolveGeometryPtr geometry, PoolKind poolKind);
};

template <class ElemType>
class CuDnnBatchNormEngineFactory
{
public:
    static std::unique_ptr<BatchNormEngine<ElemType>> Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                                                             bool spatial, ImageLayoutKind imageLayout);
};

// REVIEW alexeyk: wrong place? It is currently used only in unit tests but I can't add it there because of the build issues.
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
