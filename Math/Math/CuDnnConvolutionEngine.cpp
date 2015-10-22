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

// REVIEW alexeyk: this is the format used originally by CNTK. Consider changing it to NCHW as NHWC currently does not support FFT-based convolutions.
#define TENSOR_FORMAT CUDNN_TENSOR_NHWC
// CNTK default implementation uses CHWN format which is converted during model loading to NCHW.
#define FILTER_FORMAT CUDNN_TENSOR_NCHW
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
        CuDnnTensor4D(size_t w, size_t h, size_t c, size_t n, cudnnDataType_t dataType)
            : ConvolutionTensor4D(w, h, c, n), m_tensor(nullptr)
        {
            CUDNN_CALL(cudnnCreateTensorDescriptor(&m_tensor));
            CUDNN_CALL(cudnnSetTensor4dDescriptor(m_tensor, TENSOR_FORMAT, dataType, 
                static_cast<int>(n), static_cast<int>(c), static_cast<int>(h), static_cast<int>(w)));
        }
    public:
        operator cudnnTensorDescriptor_t() const { return m_tensor; }

        ~CuDnnTensor4D()
        {
            if (m_tensor != nullptr)
            {
                cudnnDestroyTensorDescriptor(m_tensor);
                m_tensor = nullptr;
            }
        }
        // REVIEW alexeyk: implement move ctor/assignment.
    private:
        cudnnTensorDescriptor_t m_tensor;
    };

    class CuDnnFilter : public ConvolutionFilter
    {
    public:
        CuDnnFilter(size_t w, size_t h, size_t c, size_t k, cudnnDataType_t dataType)
            : ConvolutionFilter(w, h, c, k), m_filter(nullptr)
        {
            CUDNN_CALL(cudnnCreateFilterDescriptor(&m_filter));
            CUDNN_CALL(cudnnSetFilter4dDescriptor_v4(m_filter, dataType, FILTER_FORMAT,
                static_cast<int>(k), static_cast<int>(c), static_cast<int>(h), static_cast<int>(w)));
        }
    public:
        operator cudnnFilterDescriptor_t() const { return m_filter; }

        ~CuDnnFilter()
        {
            if (m_filter != nullptr)
            {
                cudnnDestroyFilterDescriptor(m_filter);
                m_filter = nullptr;
            }
        }
        // REVIEW alexeyk: implement move ctor/assignment.
    private:
        cudnnFilterDescriptor_t m_filter;
    };

    class CuDnnConvolutionDescriptor : public ConvolutionDescriptor
    {
    public:
        CuDnnConvolutionDescriptor(const ConvolutionTensor4D& inT, const ConvolutionFilter& filterT, 
            size_t wStride, size_t hStride, bool padding)
            : ConvolutionDescriptor(inT, filterT, wStride, hStride, padding), m_conv(nullptr)
        {
            CUDNN_CALL(cudnnCreateConvolutionDescriptor(&m_conv));
            CUDNN_CALL(cudnnSetConvolution2dDescriptor(m_conv,
                0, 0, static_cast<int>(hStride), static_cast<int>(wStride),
                1, 1, CUDNN_CROSS_CORRELATION));
        }
    public:
        operator cudnnConvolutionDescriptor_t() const { return m_conv; }

        ~CuDnnConvolutionDescriptor()
        {
            if (m_conv != nullptr)
            {
                cudnnDestroyConvolutionDescriptor(m_conv);
                m_conv = nullptr;
            }
        }
        // REVIEW alexeyk: implement move ctor/assignment.
    private:
        cudnnConvolutionDescriptor_t m_conv;
    };

    class CuDnnConvolutionEngineImpl
    {
    public:
        using Tensor4D = ConvolutionTensor4D;
        using Tensor4DPtr = std::unique_ptr<Tensor4D>;
        using Filter = ConvolutionFilter;
        using FilterPtr = std::unique_ptr<ConvolutionFilter>;
        using ConvDesc = ConvolutionDescriptor;
        using ConvDescPtr = std::unique_ptr<ConvolutionDescriptor>;

        CuDnnConvolutionEngineImpl(size_t maxTempMemSizeInSamples)
            : m_maxTempMemSizeInSamples(maxTempMemSizeInSamples), m_cudnn(nullptr)
        {
            CUDNN_CALL(cudnnCreate(&m_cudnn));
        }

        ~CuDnnConvolutionEngineImpl()
        {
            if (m_cudnn != nullptr)
            {
                cudnnDestroy(m_cudnn);
                m_cudnn = nullptr;
            }
        }
    public:
        void Forward(const Tensor4D& inT, const void* alpha, const void* in, const Filter& filterT, const void* filter, const ConvDesc& convDesc,
            const void* beta, const Tensor4D& outT, void* out)
        {
            CUDNN_CALL(cudnnConvolutionForward(m_cudnn, alpha, t(inT), in, f(filterT), filter, cd(convDesc), CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                nullptr, 0, beta, t(outT), out));
        }

        template <typename ElemType>
        Tensor4DPtr CreateTensor(size_t w, size_t h, size_t c, size_t n)
        {
            static_assert(false, "cuDNN engine currently supports only single and double precision tensors.");
        }
        template <>
        Tensor4DPtr CreateTensor<float>(size_t w, size_t h, size_t c, size_t n)
        {
            return std::make_unique<CuDnnTensor4D>(w, h, c, n, CUDNN_DATA_FLOAT);
        }
        template <>
        Tensor4DPtr CreateTensor<double>(size_t w, size_t h, size_t c, size_t n)
        {
            return std::make_unique<CuDnnTensor4D>(w, h, c, n, CUDNN_DATA_DOUBLE);
        }

        template <typename ElemType>
        FilterPtr CreateFilter(size_t w, size_t h, size_t c, size_t k)
        {
            static_assert(false, "cuDNN engine currently supports only single and double precision filters.");
        }
        template <>
        FilterPtr CreateFilter<float>(size_t w, size_t h, size_t c, size_t k)
        {
            return std::make_unique<CuDnnFilter>(w, h, c, k, CUDNN_DATA_FLOAT);
        }
        template <>
        FilterPtr CreateFilter<double>(size_t w, size_t h, size_t c, size_t k)
        {
            return std::make_unique<CuDnnFilter>(w, h, c, k, CUDNN_DATA_DOUBLE);
        }

        ConvDescPtr CreateConvDescriptor(const Tensor4D& inT, const Filter& filterT, 
            size_t wStride, size_t hStride, bool padding)
        {
            return std::make_unique<CuDnnConvolutionDescriptor>(inT, filterT, wStride, hStride, padding);
        }

    private:
        template <typename CuDnnT, typename Out, typename In>
        Out As(In& src)
        {
            // Do dynamic_cast only in debug builds and static_cast in release builds.
            assert(dynamic_cast<CuDnnT*>(&src) != nullptr);
            return static_cast<CuDnnT&>(src);
        }
        const cudnnTensorDescriptor_t t(const Tensor4D& src)
        {
            return As<const CuDnnTensor4D, const cudnnTensorDescriptor_t>(src);
        }
        cudnnTensorDescriptor_t t(Tensor4D& src)
        {
            return As<const CuDnnTensor4D, const cudnnTensorDescriptor_t>(src);
        }
        const cudnnFilterDescriptor_t f(const ConvolutionFilter& src)
        {
            return As<const CuDnnFilter, const cudnnFilterDescriptor_t>(src);
        }
        const cudnnConvolutionDescriptor_t cd(const ConvolutionDescriptor& src)
        {
            return As<const CuDnnConvolutionDescriptor, const cudnnConvolutionDescriptor_t>(src);
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
        using Filter = ConvolutionFilter;
        using FilterPtr = std::unique_ptr<ConvolutionFilter>;
        using ConvDesc = ConvolutionDescriptor;
        using ConvDescPtr = std::unique_ptr<ConvolutionDescriptor>;

        CuDnnConvolutionEngineImpl(size_t) { }

    public:
        void Forward(const Tensor4D&, const void*, const Filter&, const void*, const ConvDesc&, const Tensor4D&, void*) {}

        template <typename ElemType>
        Tensor4DPtr CreateConvTensor(size_t, size_t, size_t, size_t)
        {
            return std::make_unique<Tensor4D>();
        }

        template <typename ElemType>
        FilterPtr CreateFilter(size_t, size_t, size_t, size_t)
        {
            return std::make_unique<Filter>();
        }

        ConvDescPtr CreateConvDescriptor(const Tensor4D& inT, const Filter& filterT, 
            size_t, size_t, bool)
        {
            return std::make_unique<ConvDesc>(inT, filterT);
        }
    };

#endif

    template<class ElemType>
    CuDnnConvolutionEngine<ElemType>::CuDnnConvolutionEngine(DEVICEID_TYPE /*deviceId*/, size_t maxTempMemSizeInSamples)
        : m_impl(std::make_unique<CuDnnConvolutionEngineImpl>(maxTempMemSizeInSamples))
    {
    }

    template<class ElemType>
    void CuDnnConvolutionEngine<ElemType>::Forward(const Tensor4D& inT, const Mat& in, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
        const Tensor4D& outT, Mat& out)
    {
        const ElemType zero = static_cast<ElemType>(0);
        const ElemType one = static_cast<ElemType>(1);
        m_impl->Forward(inT, &one, in.BufferPointer(), filterT, filter.BufferPointer(), convDesc, &zero, outT, out.BufferPointer());
    }

    template<class ElemType>
    typename CuDnnConvolutionEngine<ElemType>::Tensor4DPtr CuDnnConvolutionEngine<ElemType>::CreateTensor(size_t w, size_t h, size_t c, size_t n)
    {
        return m_impl->CreateTensor<ElemType>(w, h, c, n);
    }

    template<class ElemType>
    typename CuDnnConvolutionEngine<ElemType>::FilterPtr CuDnnConvolutionEngine<ElemType>::CreateFilter(size_t w, size_t h, size_t c, size_t k)
    {
        return m_impl->CreateFilter<ElemType>(w, h, c, k);
    }

    template<class ElemType>
    typename CuDnnConvolutionEngine<ElemType>::ConvDescPtr typename CuDnnConvolutionEngine<ElemType>::CreateConvDescriptor(const Tensor4D& inT, const Filter& filterT, 
        size_t wStride, size_t hStride, bool padding)
    {
        return m_impl->CreateConvDescriptor(inT, filterT, wStride, hStride, padding);
    }

    template class CuDnnConvolutionEngine<float>;
    template class CuDnnConvolutionEngine<double>;
}}}
