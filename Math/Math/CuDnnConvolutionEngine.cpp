//
// <copyright file="CuDnnConvolutionEngine.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "CuDnnConvolutionEngine.h"
#include "GPUMatrix.h"
#ifdef USE_CUDNN
#include <cudnn.h>

template<> static const char* CudaErrString(cudnnStatus_t x)
{
    return cudnnGetErrorString(x);
}
#define CUDNN_CALL(expr)     (CudaCall((expr), #expr, "cuDNN", CUDNN_STATUS_SUCCESS))

#endif

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
bool CuDnnConvolutionEngine<ElemType>::IsSupported()
{
// REVIEW alexeyk: compile-time for now, make runtime, config-driven.
#ifdef USE_CUDNN
    return true;
#else
    return false;
#endif
}

#ifdef USE_CUDNN

class CuDnnTensor4D : public ConvolutionTensor4D
{
public:
    CuDnnTensor4D(size_t w, size_t h, size_t c, size_t n)
        : ConvolutionTensor4D(w, h, c, n)
    {
    }
};

class CuDnnConvolutionEngineImpl
{
public:
    using Tensor4D = ConvolutionTensor4D;
    using Tensor4DPtr = std::unique_ptr<Tensor4D>;

    CuDnnConvolutionEngineImpl(size_t maxTempMemSizeInSamples)
        : m_maxTempMemSizeInSamples(maxTempMemSizeInSamples)
    {
        CUDNN_CALL(cudnnCreate(&m_cudnn));
    }

public:
    void Forward(const Tensor4D& inT, const void* in, const Tensor4D& filterT, const void* filter, const ConvolutionOptions& convOpt,
            const Tensor4D& outT, void* out)
    {
        UNUSED(inT);
        UNUSED(in);
        UNUSED(filterT);
        UNUSED(filter);
        UNUSED(convOpt);
        UNUSED(outT);
        UNUSED(out);
    }

    Tensor4DPtr CreateConvTensor(size_t w, size_t h, size_t c, size_t n)
    {
        return std::make_unique<CuDnnTensor4D>(w, h, c, n);
    }

private:
    size_t m_maxTempMemSizeInSamples;
    cudnnHandle_t m_cudnn;
};

#else

class CuDnnConvolutionEngineImpl
{
public:
    using Tensor4D = ConvolutionTensor4D;
    using Tensor4DPtr = std::unique_ptr<Tensor4D>;

    CuDnnConvolutionEngineImpl(size_t) { }

public:
    void Forward(const Tensor4D&, const void*, const Tensor4D&, const void*, const ConvolutionOptions&, const Tensor4D&, void*) {}
    
    Tensor4DPtr CreateConvTensor(size_t, size_t, size_t, size_t)
    {
        return std::make_unique<Tensor4D>();
    }

};

#endif

template<class ElemType>
CuDnnConvolutionEngine<ElemType>::CuDnnConvolutionEngine(DEVICEID_TYPE /*deviceId*/, size_t maxTempMemSizeInSamples)
    : m_impl(std::make_unique<CuDnnConvolutionEngineImpl>(maxTempMemSizeInSamples))
{
}

template<class ElemType>
void CuDnnConvolutionEngine<ElemType>::Forward(const Tensor4D& inT, const Mat& in, const Tensor4D& filterT, const Mat& filter, const ConvolutionOptions& convOpt,
    const Tensor4D& outT, Mat& out)
{
    m_impl->Forward(inT, in.BufferPointer(), filterT, filter.BufferPointer(), convOpt, outT, out.BufferPointer());
}

template<class ElemType>
typename CuDnnConvolutionEngine<ElemType>::Tensor4DPtr CuDnnConvolutionEngine<ElemType>::CreateConvTensor(size_t w, size_t h, size_t c, size_t n)
{
    return m_impl->CreateConvTensor(w, h, c, n);
}

template class CuDnnConvolutionEngine<float>;
template class CuDnnConvolutionEngine<double>;

}}}
