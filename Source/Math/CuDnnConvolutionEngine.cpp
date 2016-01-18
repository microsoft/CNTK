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

template <>
const char* CudaErrString<cudnnStatus_t>(cudnnStatus_t x)
{
    return cudnnGetErrorString(x);
}

// A note on the formats: CNTK originally used NHWC for input/output tensors and CHWN for filters.
// Such formats have very limited support in cuDNN and not used in other frameworks.
// CNTK with cuDNN by default uses NCHW formats for both inputs/outputs and filters.
#define TENSOR_FORMAT CUDNN_TENSOR_NCHW
#define FILTER_FORMAT CUDNN_TENSOR_NCHW
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
bool CuDnnConvolutionEngineFactory<ElemType>::IsSupported(DEVICEID_TYPE deviceId)
{
// REVIEW alexeyk: compile-time for now, make runtime, config-driven.
#ifdef USE_CUDNN
    cudaDeviceProp props = {0};
    return cudaGetDeviceProperties(&props, deviceId) == cudaSuccess && props.major >= 3;
#else
    UNUSED(deviceId);
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
    operator cudnnTensorDescriptor_t() const
    {
        return m_tensor;
    }

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
        : ConvolutionFilter(w, h, c, k), m_filter(nullptr)
    {
        CUDNN_CALL(cudnnCreateFilterDescriptor(&m_filter));
        CUDNN_CALL(cudnnSetFilter4dDescriptor_v4(m_filter, dataType, FILTER_FORMAT,
                                                 static_cast<int>(k), static_cast<int>(c), static_cast<int>(h), static_cast<int>(w)));
    }

public:
    operator cudnnFilterDescriptor_t() const
    {
        return m_filter;
    }

    ~CuDnnFilter()
    {
        if (m_filter != nullptr)
        {
            cudnnDestroyFilterDescriptor(m_filter);
            m_filter = nullptr;
        }
    }

private:
    cudnnFilterDescriptor_t m_filter;
};

class CuDnnConvolutionDescriptor : public ConvolutionDescriptor
{
public:
    CuDnnConvolutionDescriptor(size_t wStride, size_t hStride, size_t wPad, size_t hPad)
        : ConvolutionDescriptor(wStride, hStride, wPad > 0 || hPad > 0), m_conv(nullptr)
    {
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&m_conv));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(m_conv,
                                                   static_cast<int>(hPad), static_cast<int>(wPad),
                                                   static_cast<int>(hStride), static_cast<int>(wStride),
                                                   1, 1, CUDNN_CROSS_CORRELATION));
    }

public:
    operator cudnnConvolutionDescriptor_t() const
    {
        return m_conv;
    }

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

class CuDnnPoolingDescriptor : public PoolingDescriptor
{
public:
    CuDnnPoolingDescriptor(PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad)
        : PoolingDescriptor(kind, w, h, wStride, hStride, wPad, hPad), m_pool(nullptr)
    {
        assert(kind == PoolKind::Max || kind == PoolKind::Average);

        CUDNN_CALL(cudnnCreatePoolingDescriptor(&m_pool));
        CUDNN_CALL(cudnnSetPooling2dDescriptor(m_pool,
                                               kind == PoolKind::Max ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                               static_cast<int>(h), static_cast<int>(w),
                                               static_cast<int>(hPad), static_cast<int>(wPad),
                                               static_cast<int>(hStride), static_cast<int>(wStride)));
    }

public:
    operator cudnnPoolingDescriptor_t() const
    {
        return m_pool;
    }

    ~CuDnnPoolingDescriptor()
    {
        if (m_pool != nullptr)
        {
            cudnnDestroyPoolingDescriptor(m_pool);
            m_pool = nullptr;
        }
    }

private:
    cudnnPoolingDescriptor_t m_pool;
};

template <typename CuDnnT, typename In>
static CuDnnT& As(In& src)
{
    // Do dynamic_cast only in debug builds and static_cast in release builds.
    assert(dynamic_cast<CuDnnT*>(&src) != nullptr);
    return static_cast<CuDnnT&>(src);
}
static const CuDnnTensor4D& t(const ConvolutionTensor4D& src)
{
    return As<const CuDnnTensor4D>(src);
}
static const CuDnnFilter& f(const ConvolutionFilter& src)
{
    return As<const CuDnnFilter>(src);
}
static const CuDnnConvolutionDescriptor& cd(const ConvolutionDescriptor& src)
{
    return As<const CuDnnConvolutionDescriptor>(src);
}
static const CuDnnPoolingDescriptor& p(const PoolingDescriptor& src)
{
    return As<const CuDnnPoolingDescriptor>(src);
}
template <typename ElemType>
static ElemType* ptr(Matrix<ElemType>& src)
{
    return src.BufferPointer();
}
template <typename ElemType>
static const ElemType* ptr(const Matrix<ElemType>& src)
{
    return src.BufferPointer();
}

template <typename ElemType>
struct Consts
{
    static const ElemType Zero;
    static const ElemType One;
};
template <>
const float Consts<float>::One = 1;
template <>
const double Consts<double>::One = 1;
template <>
const float Consts<float>::Zero = 0;
template <>
const double Consts<double>::Zero = 0;

template <typename ElemType>
class CuDnnConvolutionEngine : public ConvolutionEngine<ElemType>
{
public:
    using Base = ConvolutionEngine<ElemType>;
    using typename Base::Mat;
    using typename Base::Tensor4D;
    using typename Base::Filter;
    using typename Base::ConvDesc;

    CuDnnConvolutionEngine(size_t maxTempMemSizeInSamples)
        : m_maxTempMemSizeInSamples(maxTempMemSizeInSamples), m_cudnn(nullptr), m_curMBSize(0)
    {
        CUDNN_CALL(cudnnCreate(&m_cudnn));
        CUDNN_CALL(cudnnSetStream(m_cudnn, GetStream()));
        m_fwdAlgo.status = CUDNN_STATUS_NOT_INITIALIZED;
        m_backDataAlgo.status = CUDNN_STATUS_NOT_INITIALIZED;
        m_backFiltAlgo.status = CUDNN_STATUS_NOT_INITIALIZED;
    }

    ~CuDnnConvolutionEngine()
    {
        if (m_cudnn != nullptr)
        {
            cudnnDestroy(m_cudnn);
            m_cudnn = nullptr;
        }
    }

public:
    void Forward(const Tensor4D& inT, const Mat& in, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
                 const Tensor4D& outT, Mat& out, Mat& workspace) override
    {
        assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
        assert(inT.n() == in.GetNumCols());
        assert(filterT.k() == filter.GetNumRows());
        assert(filterT.w() * filterT.h() * filterT.c() == filter.GetNumCols());
        assert(inT.c() == filterT.c());
        assert(outT.c() == filterT.k());

        // Find best algo and allocate temp buffer, if needed.
        FindBestForwardAlgo(t(inT), f(filterT), cd(convDesc), t(outT));
        if (m_fwdAlgo.memory > 0)
            workspace.Resize((m_fwdAlgo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
        // Perform forward convolution operation.
        CUDNN_CALL(cudnnConvolutionForward(m_cudnn, &C::One, t(inT), ptr(in), f(filterT), ptr(filter), cd(convDesc), m_fwdAlgo.algo,
                                           ptr(workspace), m_fwdAlgo.memory, &C::Zero, t(outT), ptr(out)));
    }

    void BackwardData(const Tensor4D& srcGradT, const Mat& srcGrad, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
                      const Tensor4D& gradT, Mat& grad, Mat& workspace) override
    {
        assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
        assert(srcGradT.n() == srcGrad.GetNumCols());
        assert(filterT.k() == filter.GetNumRows());
        assert(filterT.w() * filterT.h() * filterT.c() == filter.GetNumCols());
        assert(srcGradT.c() == filterT.k());
        assert(gradT.c() == filterT.c());
        assert(gradT.w() * gradT.h() * gradT.c() == grad.GetNumRows());
        assert(gradT.n() == grad.GetNumCols());

        // Find best algo and allocate temp buffer, if needed.
        FindBestBackwardDataAlgo(f(filterT), t(srcGradT), cd(convDesc), t(gradT));
        if (m_backDataAlgo.memory > 0)
            workspace.Resize((m_backDataAlgo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
        // Compute gradients with respect to the output tensor (data).
        CUDNN_CALL(cudnnConvolutionBackwardData(m_cudnn, &C::One, f(filterT), ptr(filter), t(srcGradT), ptr(srcGrad), cd(convDesc), m_backDataAlgo.algo,
                                                ptr(workspace), m_backDataAlgo.memory, &C::One, t(gradT), ptr(grad)));
    }

    void BackwardFilter(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& inT, const Mat& in, const ConvDesc& convDesc,
                        const Filter& filterT, Mat& filter, bool /*allowReuse*/, Mat& workspace) override
    {
        assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
        assert(srcGradT.n() == srcGrad.GetNumCols());
        assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
        assert(inT.n() == in.GetNumCols());
        assert(srcGradT.c() == filterT.k());
        assert(inT.c() == filterT.c());
        assert(filterT.k() == filter.GetNumRows());
        assert(filterT.w() * filterT.h() * filterT.c() == filter.GetNumCols());

        // Find best algo and allocate temp buffer, if needed.
        FindBestBackwardFilterAlgo(t(inT), t(srcGradT), cd(convDesc), f(filterT));
        if (m_backFiltAlgo.memory > 0)
            workspace.Resize((m_backFiltAlgo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
        // Compute gradients with respect to the output tensor (data).
        CUDNN_CALL(cudnnConvolutionBackwardFilter(m_cudnn, &C::One, t(inT), ptr(in), t(srcGradT), ptr(srcGrad), cd(convDesc), m_backFiltAlgo.algo,
                                                  ptr(workspace), m_backFiltAlgo.memory, &C::One, f(filterT), ptr(filter)));
    }

    void AddBias(const Tensor4D& outT, const Mat& out, const Tensor4D& biasT, const Mat& bias, Mat& dst) override
    {
        assert(biasT.c() == outT.c());
        assert(biasT.w() == 1);
        assert(biasT.h() == 1);
        assert(biasT.n() == 1);
        assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
        assert(outT.n() == out.GetNumCols());

        CUDNN_CALL(cudnnAddTensor(m_cudnn, &C::One, t(outT), ptr(out), &C::Zero, t(outT), ptr(dst)));
        CUDNN_CALL(cudnnAddTensor(m_cudnn, &C::One, t(biasT), ptr(bias), &C::One, t(outT), ptr(dst)));
    }

    void BackwardBias(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& biasT, Mat& biasGrad) override
    {
        assert(biasT.c() == srcGradT.c());
        assert(biasT.w() == 1);
        assert(biasT.h() == 1);
        assert(biasT.n() == 1);

        CUDNN_CALL(cudnnConvolutionBackwardBias(m_cudnn, &C::One, t(srcGradT), ptr(srcGrad), &C::One, t(biasT), ptr(biasGrad)));
    }

    void NormalizeBatch(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
                        bool spatial, double expAvgFactor, Mat& runMean, Mat& runInvStdDev, Mat& out, Mat& saveMean, Mat& saveInvStdDev) override
    {
        const size_t crowIn = inT.w() * inT.h() * inT.c();
        UNUSED(crowIn); // crowIn used only in asserts.
        if (spatial)
        {
            assert(scaleBiasT.c() == inT.c());
            assert(scaleBiasT.w() == 1);
            assert(scaleBiasT.h() == 1);
            assert(runMean.GetNumRows() == inT.c());
            assert(runMean.GetNumCols() == 1);
            assert(runInvStdDev.GetNumRows() == inT.c());
            assert(runInvStdDev.GetNumCols() == 1);
        }
        else
        {
            assert(scaleBiasT.c() == inT.c());
            assert(scaleBiasT.w() == inT.w());
            assert(scaleBiasT.h() == inT.h());
            assert(runMean.GetNumRows() == crowIn);
            assert(runMean.GetNumCols() == 1);
            assert(runInvStdDev.GetNumRows() == crowIn);
            assert(runInvStdDev.GetNumCols() == 1);
        }
        assert(scaleBiasT.n() == 1);
        assert(crowIn == in.GetNumRows());
        assert(inT.n() == in.GetNumCols());
        assert(saveMean.GetNumElements() >= runMean.GetNumElements());
        assert(saveInvStdDev.GetNumElements() >= runInvStdDev.GetNumElements());

        cudnnBatchNormMode_t mode = spatial ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
        CUDNN_CALL(cudnnBatchNormalizationForwardTraining(m_cudnn, mode, &C::One, &C::Zero, t(inT), ptr(in), t(inT), ptr(out),
                                                          t(scaleBiasT), ptr(scale), ptr(bias), expAvgFactor, ptr(runMean), ptr(runInvStdDev), CUDNN_BN_MIN_EPSILON, ptr(saveMean), ptr(saveInvStdDev)));
    }

    void NormalizeBatchInference(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
                                 bool spatial, const Mat& runMean, const Mat& runInvStdDev, Mat& out) override
    {
        const size_t crowIn = inT.w() * inT.h() * inT.c();

        if (spatial)
        {
            assert(scaleBiasT.c() == inT.c());
            assert(scaleBiasT.w() == 1);
            assert(scaleBiasT.h() == 1);
            assert(scaleBiasT.c() == runMean.GetNumRows());
            assert(scaleBiasT.c() == runInvStdDev.GetNumRows());
        }
        else
        {
            assert(scaleBiasT.c() == inT.c());
            assert(scaleBiasT.w() == inT.w());
            assert(scaleBiasT.h() == inT.h());
            assert(crowIn == runMean.GetNumRows());
            assert(crowIn == runInvStdDev.GetNumRows());
        }
        assert(scaleBiasT.n() == 1);
        assert(crowIn == in.GetNumRows());
        assert(inT.n() == in.GetNumCols());
        assert(runMean.GetNumCols() == 1);
        assert(runInvStdDev.GetNumCols() == 1);
        UNUSED(crowIn);

        cudnnBatchNormMode_t mode = spatial ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
        CUDNN_CALL(cudnnBatchNormalizationForwardInference(m_cudnn, mode, &C::One, &C::Zero, t(inT), ptr(in), t(inT), ptr(out),
                                                           t(scaleBiasT), ptr(scale), ptr(bias), ptr(runMean), ptr(runInvStdDev), CUDNN_BN_MIN_EPSILON));
    }

    void BackwardNormalizeBatch(const Tensor4D& inT, const Mat& in, const Mat& srcGrad, Mat& grad,
                                const Tensor4D& scaleBiasT, const Mat& scale, bool spatial, const Mat& saveMean, const Mat& saveInvStdDev,
                                Mat& scaleGrad, Mat& biasGrad) override
    {
        if (spatial)
        {
            assert(scaleBiasT.c() == inT.c());
            assert(scaleBiasT.w() == 1);
            assert(scaleBiasT.h() == 1);
        }
        else
        {
            assert(scaleBiasT.c() == inT.c());
            assert(scaleBiasT.w() == inT.w());
            assert(scaleBiasT.h() == inT.h());
        }
        assert(scaleBiasT.n() == 1);
        const size_t crowIn = inT.w() * inT.h() * inT.c();
        assert(crowIn == in.GetNumRows());
        assert(inT.n() == in.GetNumCols());
        assert(saveMean.GetNumElements() >= scale.GetNumElements());
        assert(saveInvStdDev.GetNumElements() >= scale.GetNumElements());
        assert(scaleGrad.GetNumRows() == scale.GetNumRows());
        assert(scaleGrad.GetNumCols() == scale.GetNumCols());
        assert(biasGrad.GetNumRows() == scale.GetNumRows());
        assert(biasGrad.GetNumCols() == scale.GetNumCols());
        UNUSED(crowIn);

        cudnnBatchNormMode_t mode = spatial ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
        CUDNN_CALL(cudnnBatchNormalizationBackward(m_cudnn, mode, &C::One, &C::One, t(inT), ptr(in), t(inT), ptr(srcGrad), t(inT), ptr(grad),
                                                   t(scaleBiasT), ptr(scale), ptr(scaleGrad), ptr(biasGrad), CUDNN_BN_MIN_EPSILON, ptr(saveMean), ptr(saveInvStdDev)));
    }

private:
    void FindBestForwardAlgo(const CuDnnTensor4D& inT, const CuDnnFilter& filtT, const CuDnnConvolutionDescriptor& convDesc, const CuDnnTensor4D& outT)
    {
        // Need to re-run auto-tuner in case batch size has been changed.
        // We assume no other dimensions of tensors can change so we don't check it.
        // REVIEW alexeyk: is this a safe assumption? Can convolution configuration change in runtime?
        if (m_fwdAlgo.status == CUDNN_STATUS_SUCCESS && inT.n() == m_curMBSize && outT.n() == m_curMBSize)
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
                                });
        if (res == algoPerf + calgo)
            RuntimeError("cuDNN could not find suitable algorithm for cudnnConvolutionForward.");
        m_curMBSize = inT.n();
        m_fwdAlgo = *res;
    }

    void FindBestBackwardDataAlgo(const CuDnnFilter& filtT, const CuDnnTensor4D& srcGradT, const CuDnnConvolutionDescriptor& convDesc, const CuDnnTensor4D& gradT)
    {
        if (m_backDataAlgo.status == CUDNN_STATUS_SUCCESS && srcGradT.n() == m_curMBSize && gradT.n() == m_curMBSize)
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
                                });
        if (res == algoPerf + calgo)
            RuntimeError("cuDNN could not find suitable algorithm for cudnnConvolutionBackwardData.");
        m_curMBSize = srcGradT.n();
        m_backDataAlgo = *res;
    }

    void FindBestBackwardFilterAlgo(const CuDnnTensor4D& inT, const CuDnnTensor4D& srcGradT, const CuDnnConvolutionDescriptor& convDesc, const CuDnnFilter& filtT)
    {
        if (m_backFiltAlgo.status == CUDNN_STATUS_SUCCESS && inT.n() == m_curMBSize && srcGradT.n() == m_curMBSize)
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
                                });
        if (res == algoPerf + calgo)
            RuntimeError("cuDNN could not find suitable algorithm for cudnnConvolutionBackwardFilter.");
        m_curMBSize = inT.n();
        m_backFiltAlgo = *res;
    }

private:
    using C = Consts<ElemType>;

    // REVIEW alexeyk: currently limit is set once in ctor though in CNTK it can be, theoretically, changed in runtime.
    size_t m_maxTempMemSizeInSamples;
    cudnnHandle_t m_cudnn;
    // Current mini-batch size, needed for re-computing statistics in auto-tuner.
    size_t m_curMBSize;
    cudnnConvolutionFwdAlgoPerf_t m_fwdAlgo;
    cudnnConvolutionBwdDataAlgoPerf_t m_backDataAlgo;
    cudnnConvolutionBwdFilterAlgoPerf_t m_backFiltAlgo;
};

template <class ElemType>
class CuDnnPoolingEngine : public PoolingEngine<ElemType>
{
public:
    using Base = PoolingEngine<ElemType>;
    using typename Base::Tensor4D;
    using typename Base::PoolDesc;
    using typename Base::Mat;

public:
    CuDnnPoolingEngine()
        : m_cudnn(nullptr)
    {
        CUDNN_CALL(cudnnCreate(&m_cudnn));
        CUDNN_CALL(cudnnSetStream(m_cudnn, GetStream()));
    }

    ~CuDnnPoolingEngine()
    {
        if (m_cudnn != nullptr)
        {
            cudnnDestroy(m_cudnn);
            m_cudnn = nullptr;
        }
    }

public:
    void Forward(const Tensor4D& inT, const Mat& in, const PoolDesc& poolDesc, const Tensor4D& outT, Mat& out) override
    {
        assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
        assert(inT.n() == in.GetNumCols());
        assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
        assert(outT.n() == out.GetNumCols());

        CUDNN_CALL(cudnnPoolingForward(m_cudnn, p(poolDesc), &C::One, t(inT), ptr(in), &C::Zero, t(outT), ptr(out)));
    }

    void Backward(const Tensor4D& outT, const Mat& out, const Mat& srcGrad, const PoolDesc& poolDesc, const Tensor4D& inT, const Mat& in, Mat& grad) override
    {
        assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
        assert(outT.n() == out.GetNumCols());
        assert(out.GetNumRows() == srcGrad.GetNumRows());
        assert(out.GetNumCols() == srcGrad.GetNumCols());
        assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
        assert(inT.n() == in.GetNumCols());
        assert(in.GetNumRows() == grad.GetNumRows());
        assert(in.GetNumCols() == grad.GetNumCols());

        CUDNN_CALL(cudnnPoolingBackward(m_cudnn, p(poolDesc), &C::One, t(outT), ptr(out), t(outT), ptr(srcGrad),
                                        t(inT), ptr(in), &C::One, t(inT), ptr(grad)));
    }

private:
    using C = Consts<ElemType>;

    cudnnHandle_t m_cudnn;
};

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::Tensor4DPtr CuDnnConvolutionEngineFactory<ElemType>::CreateTensor(size_t w, size_t h, size_t c, size_t n)
{
    // REVIEW alexeyk: assert fires in GCC but not in VC++.
    //static_assert(false, "cuDNN engine currently supports only single and double precision tensors.");
}
template <>
typename CuDnnConvolutionEngineFactory<float>::Tensor4DPtr CuDnnConvolutionEngineFactory<float>::CreateTensor(size_t w, size_t h, size_t c, size_t n)
{
    return std::make_unique<CuDnnTensor4D>(w, h, c, n, CUDNN_DATA_FLOAT);
}
template <>
typename CuDnnConvolutionEngineFactory<double>::Tensor4DPtr CuDnnConvolutionEngineFactory<double>::CreateTensor(size_t w, size_t h, size_t c, size_t n)
{
    return std::make_unique<CuDnnTensor4D>(w, h, c, n, CUDNN_DATA_DOUBLE);
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::FilterPtr CuDnnConvolutionEngineFactory<ElemType>::CreateFilter(size_t w, size_t h, size_t c, size_t k)
{
    // REVIEW alexeyk: assert fires in GCC but not in VC++.
    //static_assert(false, "cuDNN engine currently supports only single and double precision filters.");
}
template <>
typename CuDnnConvolutionEngineFactory<float>::FilterPtr CuDnnConvolutionEngineFactory<float>::CreateFilter(size_t w, size_t h, size_t c, size_t k)
{
    return std::make_unique<CuDnnFilter>(w, h, c, k, CUDNN_DATA_FLOAT);
}
template <>
typename CuDnnConvolutionEngineFactory<double>::FilterPtr CuDnnConvolutionEngineFactory<double>::CreateFilter(size_t w, size_t h, size_t c, size_t k)
{
    return std::make_unique<CuDnnFilter>(w, h, c, k, CUDNN_DATA_DOUBLE);
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::ConvDescPtr CuDnnConvolutionEngineFactory<ElemType>::CreateConvDescriptor(
    const Tensor4D& /*inT*/, const Filter& filterT, size_t wStride, size_t hStride, bool padding)
{
    size_t wPad = padding ? filterT.w() / 2 : 0;
    size_t hPad = padding ? filterT.h() / 2 : 0;
    return std::make_unique<CuDnnConvolutionDescriptor>(wStride, hStride, wPad, hPad);
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::PoolDescPtr CuDnnConvolutionEngineFactory<ElemType>::CreatePoolDescriptor(
    typename PoolDesc::PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad)
{
    return std::make_unique<CuDnnPoolingDescriptor>(kind, w, h, wStride, hStride, wPad, hPad);
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::ConvEnginePtr CuDnnConvolutionEngineFactory<ElemType>::CreateConvEngine(
    DEVICEID_TYPE /*deviceId*/, size_t maxTempMemSizeInSamples)
{
    return std::make_unique<CuDnnConvolutionEngine<ElemType>>(maxTempMemSizeInSamples);
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::PoolEnginePtr CuDnnConvolutionEngineFactory<ElemType>::CreatePoolEngine(
    DEVICEID_TYPE /*deviceId*/)
{
    return std::make_unique<CuDnnPoolingEngine<ElemType>>();
}

#else

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::Tensor4DPtr CuDnnConvolutionEngineFactory<ElemType>::CreateTensor(size_t, size_t, size_t, size_t)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::FilterPtr CuDnnConvolutionEngineFactory<ElemType>::CreateFilter(size_t, size_t, size_t, size_t)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::ConvDescPtr CuDnnConvolutionEngineFactory<ElemType>::CreateConvDescriptor(
    const Tensor4D&, const Filter&, size_t, size_t, bool)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::PoolDescPtr CuDnnConvolutionEngineFactory<ElemType>::CreatePoolDescriptor(
    typename PoolDesc::PoolKind, size_t, size_t, size_t, size_t, size_t, size_t)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::ConvEnginePtr CuDnnConvolutionEngineFactory<ElemType>::CreateConvEngine(DEVICEID_TYPE, size_t)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::PoolEnginePtr CuDnnConvolutionEngineFactory<ElemType>::CreatePoolEngine(DEVICEID_TYPE)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

#endif

template class CuDnnConvolutionEngineFactory<float>;
template class CuDnnConvolutionEngineFactory<double>;
} } }
