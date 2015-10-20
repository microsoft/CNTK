//
// <copyright file="CuDnnConvolutionEngine.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#ifdef    _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#else    // no DLLs on Linux
#define    MATH_API 
#endif

#include "ConvolutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class CuDnnConvolutionEngineImpl;

template<class ElemType>
class CuDnnConvolutionEngine : public ConvolutionEngine<ElemType>
{
public:
    CuDnnConvolutionEngine(DEVICEID_TYPE deviceId, size_t maxTempMemSizeInSamples);

public:
    void Forward(const Tensor4D& inT, const Mat& in, const Tensor4D& filterT, const Mat& filter, const ConvolutionOptions& convOpt,
        const Tensor4D& outT, Mat& out) override;

    Tensor4DPtr CreateTensor(size_t w, size_t h, size_t c, size_t n) override;

    static bool IsSupported();
private:
    // Using pimpl to hide cuDNN objects. CuDnnConvolutionEngine.h is included in other projects not aware of cuDNN.
    std::unique_ptr<CuDnnConvolutionEngineImpl> m_impl;
};

}}}