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
// CNTK default implementation uses CHWN format which is converted in runtime to NHWC. Filter format must be the same as tensor as cuDNN back data/filter
// routines currently support only such configuration.
#define FILTER_FORMAT CUDNN_TENSOR_NHWC
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
            : ConvolutionTensor4D(w, h, c, n), m_dataType(dataType), m_tensor(nullptr)
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

        void setN(size_t newN) override
        {
            ConvolutionTensor4D::setN(newN);
            CUDNN_CALL(cudnnSetTensor4dDescriptor(m_tensor, TENSOR_FORMAT, m_dataType,
                static_cast<int>(n()), static_cast<int>(c()), static_cast<int>(h()), static_cast<int>(w())));
        }
    private:
        cudnnDataType_t m_dataType;
        cudnnTensorDescriptor_t m_tensor;
    };

    class CuDnnFilter : public ConvolutionFilter
    {
    public:
        CuDnnFilter(size_t w, size_t h, size_t c, size_t k, cudnnDataType_t dataType)
            : ConvolutionFilter(w, h, c, k), m_filter(nullptr), m_inF(nullptr), m_outF(nullptr)
        {
            CUDNN_CALL(cudnnCreateFilterDescriptor(&m_filter));
            CUDNN_CALL(cudnnSetFilter4dDescriptor_v4(m_filter, dataType, FILTER_FORMAT,
                static_cast<int>(k), static_cast<int>(c), static_cast<int>(h), static_cast<int>(w)));

            // Create tensors needed to convert filter. Should be removed in future.
            CUDNN_CALL(cudnnCreateTensorDescriptor(&m_inF));
            CUDNN_CALL(cudnnCreateTensorDescriptor(&m_outF));
            // CNTK legacy code uses filters in CHWN format. This format is not currently supported by cuDNN.
            int dims[] = { static_cast<int>(c), static_cast<int>(h), static_cast<int>(w), static_cast<int>(k) };
            int inStride[] =  { static_cast<int>(h * w * k), static_cast<int>(w * k), static_cast<int>(k), 1 };
            CUDNN_CALL(cudnnSetTensorNdDescriptor(m_inF, dataType, 4, dims, inStride));
            // Create output tensor which is a transpose of input tensor so the output becomes NHWC-format tensor supported by cuDNN.
            // Note that only strides change, tensor dimensions must be the same.
            int outStride[] =  { 1, static_cast<int>(w * c), static_cast<int>(c), static_cast<int>(c * h * w) };
            CUDNN_CALL(cudnnSetTensorNdDescriptor(m_outF, dataType, 4, dims, outStride));
        }
    public:
        operator cudnnFilterDescriptor_t() const { return m_filter; }

        cudnnTensorDescriptor_t InTensor() const { return m_inF; };
        cudnnTensorDescriptor_t OutTensor() const { return m_outF; };

        ~CuDnnFilter()
        {
            if (m_filter != nullptr)
            {
                cudnnDestroyFilterDescriptor(m_filter);
                m_filter = nullptr;
            }
            if (m_inF != nullptr)
            {
                cudnnDestroyTensorDescriptor(m_inF);
                m_inF = nullptr;
            }
            if (m_outF != nullptr)
            {
                cudnnDestroyTensorDescriptor(m_outF);
                m_outF = nullptr;
            }
        }
    private:
        cudnnFilterDescriptor_t m_filter;
        // REVIEW alexeyk: tensors and temp storage for filter conversion from CHWN to NHWC format, remove.
        cudnnTensorDescriptor_t m_inF;
        cudnnTensorDescriptor_t m_outF;
    };
    
    class CuDnnConvolutionDescriptor : public ConvolutionDescriptor
    {
    public:
        CuDnnConvolutionDescriptor(size_t wStride, size_t hStride, size_t wPad, size_t hPad)
            : ConvolutionDescriptor(wStride, hStride, wPad == 0 && hPad == 0), m_conv(nullptr)
        {
            CUDNN_CALL(cudnnCreateConvolutionDescriptor(&m_conv));
            CUDNN_CALL(cudnnSetConvolution2dDescriptor(m_conv,
                static_cast<int>(hPad), static_cast<int>(wPad),
                static_cast<int>(hStride), static_cast<int>(wStride),
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
    private:
        cudnnConvolutionDescriptor_t m_conv;
    };

    template <typename ElemType>
    class CuDnnConvolutionEngineImpl
    {
    public:
        using Tensor4D = ConvolutionTensor4D;
        using Tensor4DPtr = std::unique_ptr<Tensor4D>;
        using Filter = ConvolutionFilter;
        using FilterPtr = std::unique_ptr<ConvolutionFilter>;
        using ConvDesc = ConvolutionDescriptor;
        using ConvDescPtr = std::unique_ptr<ConvolutionDescriptor>;

        CuDnnConvolutionEngineImpl(DEVICEID_TYPE deviceId, size_t maxTempMemSizeInSamples)
            : m_maxTempMemSizeInSamples(maxTempMemSizeInSamples), m_cudnn(nullptr), m_temp(deviceId)
        {
            CUDNN_CALL(cudnnCreate(&m_cudnn));
            CUDNN_CALL(cudnnSetStream(m_cudnn, GetStream()));
            m_fwdAlgo.status = CUDNN_STATUS_NOT_INITIALIZED;
            m_backDataAlgo.status = CUDNN_STATUS_NOT_INITIALIZED;
            m_backFiltAlgo.status = CUDNN_STATUS_NOT_INITIALIZED;
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
        void Forward(ElemType alpha, const Tensor4D& inT, const ElemType* in, const Filter& filterT, const ElemType* filter, const ConvDesc& convDesc,
            ElemType beta, const Tensor4D& outT, ElemType* out)
        {
            auto& filtT = f(filterT);
            // Convert filter to NHWC format.
            m_temp.Resize(filtT.k() * filtT.c() * filtT.h() * filtT.w(), 1);
            CUDNN_CALL(cudnnTransformTensor(m_cudnn, &One, filtT.InTensor(), filter, &Zero, filtT.OutTensor(), m_temp.BufferPointer()));
            // Find best algo and allocate temp buffer, if needed.
            FindBestForwardAlgo(t(inT), filtT, cd(convDesc), t(outT));
            if (m_fwdAlgo.memory > 0)
                m_temp.Resize((m_fwdAlgo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
            // Perform forward convolution operation.
            CUDNN_CALL(cudnnConvolutionForward(m_cudnn, &alpha, t(inT), in, filtT, m_temp.BufferPointer(), cd(convDesc), m_fwdAlgo.algo,
                m_temp.BufferPointer(), m_fwdAlgo.memory, &beta, t(outT), out));
        }

        void BackwardData(ElemType alpha, const Tensor4D& srcGradT, const ElemType* srcGrad, const Filter& filterT, const ElemType* filter, const ConvDesc& convDesc,
            ElemType beta, const Tensor4D& gradT, ElemType* grad)
        {
            auto& filtT = f(filterT);
            // Convert filter to NHWC format.
            m_temp.Resize(filtT.k() * filtT.c() * filtT.h() * filtT.w(), 1);
            CUDNN_CALL(cudnnTransformTensor(m_cudnn, &One, filtT.InTensor(), filter, &Zero, filtT.OutTensor(), m_temp.BufferPointer()));
            // Find best algo and allocate temp buffer, if needed.
            FindBestBackwardDataAlgo(filtT, t(srcGradT), cd(convDesc), t(gradT));
            if (m_backDataAlgo.memory > 0)
                m_temp.Resize((m_backDataAlgo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
            // Compute gradients with respect to the output tensor (data).
            CUDNN_CALL(cudnnConvolutionBackwardData(m_cudnn, &alpha, filtT, m_temp.BufferPointer(), t(srcGradT), srcGrad, cd(convDesc), m_backDataAlgo.algo,
                m_temp.BufferPointer(), m_backDataAlgo.memory, &beta, t(gradT), grad));
        }

        void BackwardFilter(ElemType alpha, const Tensor4D& srcGradT, const ElemType* srcGrad, const Tensor4D& inT, const ElemType* in, const ConvDesc& convDesc,
            ElemType beta, const Filter& filterT, ElemType* filter)
        {
            auto& filtT = f(filterT);
            // Convert filter to NHWC format.
            m_temp.Resize(filtT.k() * filtT.c() * filtT.h() * filtT.w(), 1);
            CUDNN_CALL(cudnnTransformTensor(m_cudnn, &One, filtT.InTensor(), filter, &Zero, filtT.OutTensor(), m_temp.BufferPointer()));
            // Find best algo and allocate temp buffer, if needed.
            FindBestBackwardFilterAlgo(t(inT), t(srcGradT), cd(convDesc), filtT);
            if (m_backFiltAlgo.memory > 0)
                m_temp.Resize((m_backFiltAlgo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
            // Compute gradients with respect to the output tensor (data).
            CUDNN_CALL(cudnnConvolutionBackwardFilter(m_cudnn, &alpha, t(inT), in, t(srcGradT), srcGrad, cd(convDesc), m_backFiltAlgo.algo,
                m_temp.BufferPointer(), m_backFiltAlgo.memory, &beta, filtT, m_temp.BufferPointer()));
            // Convert filter back to CWHN format.
            CUDNN_CALL(cudnnTransformTensor(m_cudnn, &One, filtT.OutTensor(), m_temp.BufferPointer(), &Zero, filtT.InTensor(), filter));
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

        ConvDescPtr CreateConvDescriptor(const Tensor4D& /*inT*/, const Filter& filterT, 
            size_t wStride, size_t hStride, bool padding)
        {
            size_t wPad = padding ? filterT.w() / 2 : 0;
            size_t hPad = padding ? filterT.h() / 2 : 0;
            return std::make_unique<CuDnnConvolutionDescriptor>(wStride, hStride, wPad, hPad);
        }

    private:
        template <typename CuDnnT, typename In>
        CuDnnT& As(In& src)
        {
            // Do dynamic_cast only in debug builds and static_cast in release builds.
            assert(dynamic_cast<CuDnnT*>(&src) != nullptr);
            return static_cast<CuDnnT&>(src);
        }
        const CuDnnTensor4D& t(const Tensor4D& src)
        {
            return As<const CuDnnTensor4D>(src);
        }
        const CuDnnFilter& f(const ConvolutionFilter& src)
        {
            return As<const CuDnnFilter>(src);
        }
        const CuDnnConvolutionDescriptor& cd(const ConvolutionDescriptor& src)
        {
            return As<const CuDnnConvolutionDescriptor>(src);
        }

        void FindBestForwardAlgo(const CuDnnTensor4D& inT, const CuDnnFilter& filtT, const CuDnnConvolutionDescriptor& convDesc, const CuDnnTensor4D& outT)
        {
            if (m_fwdAlgo.status == CUDNN_STATUS_SUCCESS)
                return;
            const int MaxAlgoCount = 10;
            int calgo = 0;
            cudnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount];
            CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(m_cudnn, inT, filtT, convDesc, outT, MaxAlgoCount, &calgo, algoPerf));
            assert(calgo > 0);
            size_t maxMem = m_maxTempMemSizeInSamples == 0 ? (std::numeric_limits<size_t>::max)() : inT.w() * inT.h() * inT.c() * m_maxTempMemSizeInSamples * sizeof(ElemType);
            auto res = std::find_if(algoPerf, algoPerf + calgo, 
                [=](const cudnnConvolutionFwdAlgoPerf_t& cur)
                { 
                    return cur.status == CUDNN_STATUS_SUCCESS && cur.memory <= maxMem;
                }
            );
            if (res == algoPerf + calgo)
                RuntimeError("cuDNN could not find suitable algorithm for cudnnConvolutionForward.");
            m_fwdAlgo = *res;
        }

        void FindBestBackwardDataAlgo(const CuDnnFilter& filtT, const CuDnnTensor4D& srcGradT, const CuDnnConvolutionDescriptor& convDesc, const CuDnnTensor4D& gradT)
        {
            if (m_backDataAlgo.status == CUDNN_STATUS_SUCCESS)
                return;
            const int MaxAlgoCount = 10;
            int calgo = 0;
            cudnnConvolutionBwdDataAlgoPerf_t algoPerf[MaxAlgoCount];
            CUDNN_CALL(cudnnFindConvolutionBackwardDataAlgorithm(m_cudnn, filtT, srcGradT, convDesc, gradT, MaxAlgoCount, &calgo, algoPerf));
            assert(calgo > 0);
            size_t maxMem = m_maxTempMemSizeInSamples == 0 ? (std::numeric_limits<size_t>::max)() : gradT.w() * gradT.h() * gradT.c() * m_maxTempMemSizeInSamples * sizeof(ElemType);
            auto res = std::find_if(algoPerf, algoPerf + calgo, 
                [=](const cudnnConvolutionBwdDataAlgoPerf_t& cur)
                { 
                    return cur.status == CUDNN_STATUS_SUCCESS && cur.memory <= maxMem;
                }
            );
            if (res == algoPerf + calgo)
                RuntimeError("cuDNN could not find suitable algorithm for cudnnConvolutionBackwardData.");
            m_backDataAlgo = *res;
        }

        void FindBestBackwardFilterAlgo(const CuDnnTensor4D& inT, const CuDnnTensor4D& srcGradT, const CuDnnConvolutionDescriptor& convDesc, const CuDnnFilter& filtT)
        {
            if (m_backFiltAlgo.status == CUDNN_STATUS_SUCCESS)
                return;
            const int MaxAlgoCount = 10;
            int calgo = 0;
            cudnnConvolutionBwdFilterAlgoPerf_t algoPerf[MaxAlgoCount];
            CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(m_cudnn, inT, srcGradT, convDesc, filtT, MaxAlgoCount, &calgo, algoPerf));
            assert(calgo > 0);
            size_t maxMem = m_maxTempMemSizeInSamples == 0 ? (std::numeric_limits<size_t>::max)() : inT.w() * inT.h() * inT.c() * m_maxTempMemSizeInSamples * sizeof(ElemType);
            auto res = std::find_if(algoPerf, algoPerf + calgo, 
                [=](const cudnnConvolutionBwdFilterAlgoPerf_t& cur)
                { 
                    return cur.status == CUDNN_STATUS_SUCCESS && cur.memory <= maxMem;
                }
            );
            if (res == algoPerf + calgo)
                RuntimeError("cuDNN could not find suitable algorithm for cudnnConvolutionBackwardFilter.");
            m_backFiltAlgo = *res;
        }

    private:
        static const ElemType Zero;
        static const ElemType One;

        // REVIEW alexeyk: currently limit is set once in ctor though in CNTK it can be, theoretically, changed in runtime.
        size_t m_maxTempMemSizeInSamples;
        cudnnHandle_t m_cudnn;
        GPUMatrix<ElemType> m_temp;
        cudnnConvolutionFwdAlgoPerf_t m_fwdAlgo;
        cudnnConvolutionBwdDataAlgoPerf_t m_backDataAlgo;
        cudnnConvolutionBwdFilterAlgoPerf_t m_backFiltAlgo;
    };
    template<> const float CuDnnConvolutionEngineImpl<float>::One = 1;
    template<> const double CuDnnConvolutionEngineImpl<double>::One = 1;
    template<> const float CuDnnConvolutionEngineImpl<float>::Zero = 0;
    template<> const double CuDnnConvolutionEngineImpl<double>::Zero = 0;

#else

    template <typename ElemType>
    class CuDnnConvolutionEngineImpl
    {
    public:
        using Tensor4D = ConvolutionTensor4D;
        using Tensor4DPtr = std::unique_ptr<Tensor4D>;
        using Filter = ConvolutionFilter;
        using FilterPtr = std::unique_ptr<ConvolutionFilter>;
        using ConvDesc = ConvolutionDescriptor;
        using ConvDescPtr = std::unique_ptr<ConvolutionDescriptor>;

        CuDnnConvolutionEngineImpl(DEVICEID_TYPE, size_t) { }

    public:
        void Forward(ElemType, const Tensor4D&, const ElemType*, const Filter&, const ElemType*, const ConvDesc&,
            ElemType, const Tensor4D&, ElemType*)
        {
            RuntimeError("The code is compiled without USE_CUDNN macro.");
        }

        void BackwardData(ElemType, const Tensor4D&, const ElemType*, const Filter&, const ElemType*, const ConvDesc&,
            ElemType, const Tensor4D&, ElemType*)
        {
            RuntimeError("The code is compiled without USE_CUDNN macro.");
        }

        void BackwardFilter(ElemType, const Tensor4D&, const ElemType*, const Tensor4D&, const ElemType*, const ConvDesc&,
            ElemType, const Filter&, ElemType*)
        {
            RuntimeError("The code is compiled without USE_CUDNN macro.");
        }

        template <typename ElemType>
        Tensor4DPtr CreateTensor(size_t, size_t, size_t, size_t)
        {
            return std::make_unique<Tensor4D>();
        }

        template <typename ElemType>
        FilterPtr CreateFilter(size_t, size_t, size_t, size_t)
        {
            return std::make_unique<Filter>();
        }

        ConvDescPtr CreateConvDescriptor(const Tensor4D&, const Filter&, size_t, size_t, bool)
        {
            return std::make_unique<ConvDesc>();
        }
    };

#endif

    template<class ElemType>
    CuDnnConvolutionEngine<ElemType>::CuDnnConvolutionEngine(DEVICEID_TYPE deviceId, size_t maxTempMemSizeInSamples)
        : m_impl(std::make_unique<CuDnnConvolutionEngineImpl<ElemType>>(deviceId, maxTempMemSizeInSamples))
    {
    }

    template<class ElemType>
    void CuDnnConvolutionEngine<ElemType>::Forward(const Tensor4D& inT, const Mat& in, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
        const Tensor4D& outT, Mat& out)
    {
        m_impl->Forward(1, inT, in.BufferPointer(), filterT, filter.BufferPointer(), convDesc, 0, outT, out.BufferPointer());
    }

    template<class ElemType>
    void CuDnnConvolutionEngine<ElemType>::BackwardData(const Tensor4D& srcGradT, const Mat& srcGrad, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
        const Tensor4D& gradT, Mat& grad)
    {
        m_impl->BackwardData(1, srcGradT, srcGrad.BufferPointer(), filterT, filter.BufferPointer(), convDesc, 0, gradT, grad.BufferPointer());
    }

    template<class ElemType>
    void CuDnnConvolutionEngine<ElemType>::BackwardFilter(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& inT, const Mat& in,
        const ConvDesc& convDesc, const Filter& filterT, Mat& filter, bool allowReuse)
    {
        // REVIEW alexeyk: currently unused.
        UNUSED(allowReuse);
        m_impl->BackwardFilter(1, srcGradT, srcGrad.BufferPointer(), inT, in.BufferPointer(), convDesc, 0, filterT, filter.BufferPointer());
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
