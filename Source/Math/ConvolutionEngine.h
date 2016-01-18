//
// <copyright file="ConvolutionEngine.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

// REVIEW alexeyk: this seems to be repeated all over the CNTKMathDll.
#ifdef _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#else // no DLLs on Linux
#define MATH_API
#endif

#include "Matrix.h"
#include "TensorShape.h" // for ImageLayoutKind

namespace Microsoft { namespace MSR { namespace CNTK {

// REVIEW alexeyk: this is a temp class until we have generic tensor suport in CNTK.
class ConvolutionTensor4D
{
public:
    size_t w() const
    {
        return m_w;
    }
    size_t h() const
    {
        return m_h;
    }
    size_t c() const
    {
        return m_c;
    }
    size_t n() const
    {
        return m_n;
    }
    virtual void setN(size_t n)
    {
        m_n = n;
    }

public:
    ConvolutionTensor4D(size_t w = 1, size_t h = 1, size_t c = 1, size_t n = 1)
    {
        m_w = w;
        m_h = h;
        m_c = c;
        m_n = n;
    }

public:
    virtual ~ConvolutionTensor4D() = default;
    // Deleting copy ctor/assignment as derived objects may contain non-copyable state.
    ConvolutionTensor4D(const ConvolutionTensor4D&) = delete;
    ConvolutionTensor4D& operator=(const ConvolutionTensor4D&) = delete;
    // REVIEW alexeyk: Have to implement move ctor explicitly as VS2013 does not support default move ctors.
    //ConvolutionTensor4D(ConvolutionTensor4D&&);
    //ConvolutionTensor4D& operator=(ConvolutionTensor4D&&);

private:
    size_t m_w;
    size_t m_h;
    size_t m_c;
    size_t m_n;
};

class ConvolutionFilter
{
public:
    size_t w() const
    {
        return m_w;
    }
    size_t h() const
    {
        return m_h;
    }
    size_t c() const
    {
        return m_c;
    }
    size_t k() const
    {
        return m_k;
    }

public:
    ConvolutionFilter(size_t w = 1, size_t h = 1, size_t c = 1, size_t k = 1)
    {
        m_w = w;
        m_h = h;
        m_c = c;
        m_k = k;
    }

public:
    virtual ~ConvolutionFilter() = default;

    // Deleting copy ctor/assignment as derived objects may contain non-copyable state.
    ConvolutionFilter(const ConvolutionFilter&) = delete;
    ConvolutionFilter& operator=(const ConvolutionFilter&) = delete;

private:
    size_t m_w;
    size_t m_h;
    size_t m_c;
    size_t m_k;
};

// ConvolutionDescriptor describes properties specific to convolution application.
class ConvolutionDescriptor
{
public:
    // Horizontal stride (in w-dimension).
    size_t wStride() const
    {
        return m_wStride;
    }
    // Vertical stride (in h-dimension).
    size_t hStride() const
    {
        return m_hStride;
    }
    bool padding() const
    {
        return m_padding;
    }

public:
    ConvolutionDescriptor(size_t wStride = 1, size_t hStride = 1, bool padding = false)
    {
        m_wStride = wStride;
        m_hStride = hStride;
        m_padding = padding;
    }

public:
    virtual ~ConvolutionDescriptor() = default;
    // Deleting copy ctor/assignment as derived objects may contain non-copyable state.
    ConvolutionDescriptor(const ConvolutionDescriptor&) = delete;
    ConvolutionDescriptor& operator=(const ConvolutionDescriptor&) = delete;

private:
    size_t m_wStride;
    size_t m_hStride;
    bool m_padding;
};

// PoolingDescriptor describes properties specific to convolution application.
class PoolingDescriptor
{
public:
    enum class PoolKind
    {
        Max,
        Average
    };

    PoolKind kind() const
    {
        return m_kind;
    }
    // Pooling window size.
    size_t w() const
    {
        return m_w;
    }
    size_t h() const
    {
        return m_h;
    }
    // Horizontal stride (in w-dimension).
    size_t wStride() const
    {
        return m_wStride;
    }
    // Vertical stride (in h-dimension).
    size_t hStride() const
    {
        return m_hStride;
    }
    // Horizontal pad (in w-dimension).
    size_t wPad() const
    {
        return m_wPad;
    }
    // Vertical pad (in h-dimension).
    size_t hPad() const
    {
        return m_hPad;
    }

public:
    PoolingDescriptor(PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad)
    {
        m_kind = kind;
        m_w = w;
        m_h = h;
        m_wStride = wStride;
        m_hStride = hStride;
        m_wPad = wPad;
        m_hPad = hPad;
    }

public:
    virtual ~PoolingDescriptor() = default;
    // Deleting copy ctor/assignment as derived objects may contain non-copyable state.
    PoolingDescriptor(const PoolingDescriptor&) = delete;
    PoolingDescriptor& operator=(const PoolingDescriptor&) = delete;

private:
    PoolKind m_kind;
    size_t m_w;
    size_t m_h;
    size_t m_wStride;
    size_t m_hStride;
    size_t m_wPad;
    size_t m_hPad;
};

template <class ElemType>
class MATH_API ConvolutionEngine
{
public:
    using Tensor4D = ConvolutionTensor4D;
    using Filter = ConvolutionFilter;
    using ConvDesc = ConvolutionDescriptor;
    using Mat = Matrix<ElemType>;

public:
    ConvolutionEngine() = default;
    virtual ~ConvolutionEngine() = default;

    virtual void Forward(const Tensor4D& inT, const Mat& in, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
                         const Tensor4D& outT, Mat& out, Mat& workspace) = 0;

    virtual void BackwardData(const Tensor4D& srcGradT, const Mat& srcGrad, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
                              const Tensor4D& gradT, Mat& grad, Mat& workspace) = 0;

    virtual void BackwardFilter(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& inT, const Mat& in, const ConvDesc& convDesc,
                                const Filter& filterT, Mat& filter, bool allowReuse, Mat& workspace) = 0;

    virtual void AddBias(const Tensor4D& outT, const Mat& out, const Tensor4D& biasT, const Mat& bias, Mat& dst) = 0;
    virtual void BackwardBias(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& biasT, Mat& biasGrad) = 0;

    virtual void NormalizeBatch(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
                                bool spatial, double expAvgFactor, Mat& runMean, Mat& runInvStdDev, Mat& out, Mat& saveMean, Mat& saveInvStdDev) = 0;

    virtual void NormalizeBatchInference(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
                                         bool spatial, const Mat& runMean, const Mat& runInvStdDev, Mat& out) = 0;

    virtual void BackwardNormalizeBatch(const Tensor4D& inT, const Mat& in, const Mat& srcGrad, Mat& grad,
                                        const Tensor4D& scaleBiasT, const Mat& scale, bool spatial, const Mat& saveMean, const Mat& saveInvStdDev,
                                        Mat& scaleGrad, Mat& biasGrad) = 0;

public:
    ConvolutionEngine(const ConvolutionEngine&) = delete;
    ConvolutionEngine& operator=(const ConvolutionEngine&) = delete;
    ConvolutionEngine(ConvolutionEngine&&) = delete;
    ConvolutionEngine& operator=(ConvolutionEngine&&) = delete;
};

template <class ElemType>
class MATH_API PoolingEngine
{
public:
    using Tensor4D = ConvolutionTensor4D;
    using PoolDesc = PoolingDescriptor;
    using Mat = Matrix<ElemType>;

public:
    PoolingEngine() = default;
    virtual ~PoolingEngine() = default;

    virtual void Forward(const Tensor4D& inT, const Mat& in, const PoolDesc& poolDesc, const Tensor4D& outT, Mat& out) = 0;
    virtual void Backward(const Tensor4D& outT, const Mat& out, const Mat& srcGrad, const PoolDesc& poolDesc, const Tensor4D& inT, const Mat& in, Mat& grad) = 0;

public:
    PoolingEngine(const PoolingEngine&) = delete;
    PoolingEngine& operator=(const PoolingEngine&) = delete;
    PoolingEngine(PoolingEngine&&) = delete;
    PoolingEngine& operator=(PoolingEngine&&) = delete;
};

template <class ElemType>
class MATH_API ConvolutionEngineFactory
{
public:
    using Tensor4D = ConvolutionTensor4D;
    using Tensor4DPtr = std::unique_ptr<Tensor4D>;
    using Filter = ConvolutionFilter;
    using FilterPtr = std::unique_ptr<ConvolutionFilter>;
    using ConvDesc = ConvolutionDescriptor;
    using ConvDescPtr = std::unique_ptr<ConvolutionDescriptor>;
    using PoolDesc = PoolingDescriptor;
    using PoolDescPtr = std::unique_ptr<PoolingDescriptor>;

    using ConvEnginePtr = std::unique_ptr<ConvolutionEngine<ElemType>>;
    using PoolEnginePtr = std::unique_ptr<PoolingEngine<ElemType>>;

public:
    ConvolutionEngineFactory() = default;
    virtual ~ConvolutionEngineFactory() = default;

    virtual Tensor4DPtr CreateTensor(size_t w, size_t h, size_t c, size_t n) = 0;
    virtual FilterPtr CreateFilter(size_t w, size_t h, size_t c, size_t k) = 0;
    virtual ConvDescPtr CreateConvDescriptor(const Tensor4D& inT, const Filter& filterT,
                                             size_t wStride, size_t hStride, bool padding) = 0;
    virtual PoolDescPtr CreatePoolDescriptor(PoolDesc::PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad) = 0;
    //virtual Tensor4DPtr CreateLrnDescriptor() = 0;

    virtual ConvEnginePtr CreateConvEngine(DEVICEID_TYPE deviceId, size_t maxTempMemSizeInSamples) = 0;
    virtual PoolEnginePtr CreatePoolEngine(DEVICEID_TYPE deviceId) = 0;

    enum class EngineType
    {
        Auto,
        CuDnn,
        Legacy
    };
    static std::unique_ptr<ConvolutionEngineFactory<ElemType>> Create(DEVICEID_TYPE deviceId, EngineType engType, ImageLayoutKind imageLayoutKind);

public:
    ConvolutionEngineFactory(const ConvolutionEngineFactory&) = delete;
    ConvolutionEngineFactory& operator=(const ConvolutionEngineFactory&) = delete;
    ConvolutionEngineFactory(ConvolutionEngineFactory&&) = delete;
    ConvolutionEngineFactory& operator=(ConvolutionEngineFactory&&) = delete;
};
} } }
