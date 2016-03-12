//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Matrix.h"
#include "TensorShape.h" // for ImageLayoutKind
#include "ConvolveGeometry.h"

namespace Microsoft { namespace MSR { namespace CNTK {

//-------------------------------------------------------------
// Convolution and pooling engine interface.
//-------------------------------------------------------------
enum class ConvolutionEngineKind
{
    None      = 0,
    Reference = 1,
    CuDnn     = 1 << 1,
    Legacy    = 1 << 2,

    All     = Reference | CuDnn | Legacy
};

enum class PoolKind
{
    None,
    Max,
    Average
};

#pragma warning(push)
#pragma warning(disable : 4251)

template <class ElemType>
class MATH_API ConvolutionEngine
{
public:
    using Mat = Matrix<ElemType>;

public:
    virtual ~ConvolutionEngine() = default;

    void Forward(const Mat& in, const Mat& filter, Mat& out, Mat& workspace);

    void BackwardData(const Mat& srcGrad, const Mat& filter, Mat& grad, Mat& workspace);

    void BackwardFilter(const Mat& srcGrad, const Mat& in, Mat& filterGrad, bool allowReuse, Mat& workspace);

    void ForwardPooling(const Mat& in, Mat& out);

    void BackwardPooling(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad);

    static std::unique_ptr<ConvolutionEngine<ElemType>> Create(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout,
                                                               size_t maxTempMemSizeInSamples, PoolKind poolKind = PoolKind::None, ConvolutionEngineKind enabledEngines = ConvolutionEngineKind::All);

    DISABLE_COPY_AND_MOVE(ConvolutionEngine);

protected:
    ConvolutionEngine(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind)
        : m_geometry(geometry), m_deviceId(deviceId), m_imageLayout(imageLayout), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples), m_poolKind(poolKind)
    {
        assert(m_geometry != nullptr);
    }

    virtual void EnsureCompatible() = 0;

    virtual void EnsureConvolutionInitialized() = 0;

    virtual void ForwardCore(const Mat& in, const Mat& filter, Mat& out, Mat& workspace) = 0;

    virtual void BackwardDataCore(const Mat& srcGrad, const Mat& filter, Mat& grad, Mat& workspace) = 0;

    virtual void BackwardFilterCore(const Mat& srcGrad, const Mat& in, Mat& filterGrad, bool allowReuse, Mat& workspace) = 0;

    virtual void EnsurePoolingInitialized() = 0;

    virtual void ForwardPoolingCore(const Mat& in, Mat& out) = 0;

    virtual void BackwardPoolingCore(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad) = 0;

protected:
    ConvolveGeometryPtr m_geometry;
    DEVICEID_TYPE m_deviceId;
    ImageLayoutKind m_imageLayout;
    size_t m_maxTempMemSizeInSamples;
    PoolKind m_poolKind;
};

#pragma warning(pop)

//template <class ElemType>
//class MATH_API ConvolutionEngine
//{
//public:
//    using Tensor4D = ConvolutionTensor4D;
//    using Filter = ConvolutionFilter;
//    using ConvDesc = ConvolutionDescriptor;
//    using Mat = Matrix<ElemType>;
//
//public:
//    ConvolutionEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout)
//        : m_deviceId(deviceId), m_imageLayout(imageLayout)
//    {
//    }
//    virtual ~ConvolutionEngine() = default;
//
//    void Forward(const Tensor4D& inT, const Mat& in, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
//                 const Tensor4D& outT, Mat& out, Mat& workspace);
//
//    void BackwardData(const Tensor4D& srcGradT, const Mat& srcGrad, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
//                      const Tensor4D& gradT, Mat& grad, Mat& workspace);
//
//    void BackwardFilter(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& inT, const Mat& in, const ConvDesc& convDesc,
//                        const Filter& filterT, Mat& filter, bool allowReuse, Mat& workspace);
//
//    void NormalizeBatch(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
//                        bool spatial, double expAvgFactor, Mat& runMean, Mat& runInvStdDev, Mat& out,
//                        double epsilon, Mat& saveMean, Mat& saveInvStdDev);
//
//    void NormalizeBatchInference(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
//                                 bool spatial, const Mat& runMean, const Mat& runInvStdDev, Mat& out);
//
//    void BackwardNormalizeBatch(const Tensor4D& inT, const Mat& in, const Mat& srcGrad, Mat& grad,
//                                const Tensor4D& scaleBiasT, const Mat& scale, bool spatial, const Mat& saveMean, const Mat& saveInvStdDev,
//                                Mat& scaleGrad, Mat& biasGrad);
//
//    DISABLE_COPY_AND_MOVE(ConvolutionEngine);
//
//protected:
//    virtual void EnsureCompatible() = 0;
//
//    virtual void ForwardCore(const Tensor4D& inT, const Mat& in, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
//                             const Tensor4D& outT, Mat& out, Mat& workspace) = 0;
//
//    virtual void BackwardDataCore(const Tensor4D& srcGradT, const Mat& srcGrad, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
//                                  const Tensor4D& gradT, Mat& grad, Mat& workspace) = 0;
//
//    virtual void BackwardFilterCore(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& inT, const Mat& in, const ConvDesc& convDesc,
//                                    const Filter& filterT, Mat& filter, bool allowReuse, Mat& workspace) = 0;
//
//    virtual void EnsureCompatibleBatchNorm(bool spatial) = 0;
//
//    virtual void NormalizeBatchCore(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
//                                    bool spatial, double expAvgFactor, Mat& runMean, Mat& runInvStdDev, Mat& out,
//                                    double epsilon, Mat& saveMean, Mat& saveInvStdDev) = 0;
//
//    // REVIEW alexeyk: roll into NormalizeBatchCore.
//    virtual void NormalizeBatchInferenceCore(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
//                                             bool spatial, const Mat& runMean, const Mat& runInvStdDev, Mat& out) = 0;
//
//    virtual void BackwardNormalizeBatchCore(const Tensor4D& inT, const Mat& in, const Mat& srcGrad, Mat& grad,
//                                            const Tensor4D& scaleBiasT, const Mat& scale, bool spatial, const Mat& saveMean, const Mat& saveInvStdDev,
//                                            Mat& scaleGrad, Mat& biasGrad) = 0;
//
//protected:
//    DEVICEID_TYPE m_deviceId;
//    ImageLayoutKind m_imageLayout;
//};
//
//template <class ElemType>
//class MATH_API PoolingEngine
//{
//public:
//    using Tensor4D = ConvolutionTensor4D;
//    using PoolDesc = PoolingDescriptor;
//    using Mat = Matrix<ElemType>;
//
//public:
//    PoolingEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout)
//        : m_deviceId(deviceId), m_imageLayout(imageLayout)
//    {
//    }
//    virtual ~PoolingEngine() = default;
//
//    void Forward(const Tensor4D& inT, const Mat& in, const PoolDesc& poolDesc, const Tensor4D& outT, Mat& out);
//    void Backward(const Tensor4D& outT, const Mat& out, const Mat& srcGrad, const PoolDesc& poolDesc, const Tensor4D& inT, const Mat& in, Mat& grad);
//
//    DISABLE_COPY_AND_MOVE(PoolingEngine);
//
//protected:
//    virtual void EnsureCompatible() = 0;
//    virtual void ForwardCore(const Tensor4D& inT, const Mat& in, const PoolDesc& poolDesc, const Tensor4D& outT, Mat& out) = 0;
//    virtual void BackwardCore(const Tensor4D& outT, const Mat& out, const Mat& srcGrad, const PoolDesc& poolDesc, const Tensor4D& inT, const Mat& in, Mat& grad) = 0;
//
//protected:
//    DEVICEID_TYPE m_deviceId;
//    ImageLayoutKind m_imageLayout;
//};
//
//// REVIEW alexeyk: this is a temporary hack until we find a better place for the BatchNorm engine(s).
//enum class BatchNormImpl
//{
//    CuDnn,
//    Cntk
//};
//
//template <class ElemType>
//class MATH_API ConvolutionEngineFactory
//{
//public:
//    using Tensor4D = ConvolutionTensor4D;
//    using Tensor4DPtr = std::unique_ptr<Tensor4D>;
//    using Filter = ConvolutionFilter;
//    using FilterPtr = std::unique_ptr<ConvolutionFilter>;
//    using ConvDesc = ConvolutionDescriptor;
//    using ConvDescPtr = std::unique_ptr<ConvolutionDescriptor>;
//    using PoolDesc = PoolingDescriptor;
//    using PoolDescPtr = std::unique_ptr<PoolingDescriptor>;
//
//    using ConvEnginePtr = std::unique_ptr<ConvolutionEngine<ElemType>>;
//    using PoolEnginePtr = std::unique_ptr<PoolingEngine<ElemType>>;
//
//public:
//    ConvolutionEngineFactory() = default;
//    virtual ~ConvolutionEngineFactory() = default;
//
//    virtual Tensor4DPtr CreateTensor(size_t w, size_t h, size_t c, size_t n) = 0;
//    virtual FilterPtr CreateFilter(size_t w, size_t h, size_t c, size_t k) = 0;
//    virtual ConvDescPtr CreateConvDescriptor(const Tensor4D& inT, const Filter& filterT,
//                                             size_t wStride, size_t hStride, bool padding) = 0;
//    virtual PoolDescPtr CreatePoolDescriptor(PoolDesc::PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad) = 0;
//    // virtual Tensor4DPtr CreateLrnDescriptor() = 0;
//
//    virtual ConvEnginePtr CreateConvEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, BatchNormImpl bnImpl) = 0;
//    virtual PoolEnginePtr CreatePoolEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout) = 0;
//
//    enum class EngineType
//    {
//        Auto,
//        CuDnn,
//        Legacy
//    };
//    static std::unique_ptr<ConvolutionEngineFactory<ElemType>> Create(DEVICEID_TYPE deviceId, EngineType engType, ImageLayoutKind imageLayoutKind);
//
//    DISABLE_COPY_AND_MOVE(ConvolutionEngineFactory);
//};
} } }
