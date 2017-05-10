//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Matrix.h"
#include "TensorShape.h" // for ImageLayoutKind
#include "ConvolveGeometry.h"
#include "StringUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

//-------------------------------------------------------------
// Convolution and pooling engine interface.
//-------------------------------------------------------------
enum class ConvolutionEngineKind
{
    None      = 0,
    Reference = 1,      // Reference, lookup-based implementation. Very slow but works for any convo configuration.
    CuDnn     = 1 << 1, // cuDNN, works only for 2D/3D convos with full sharing.
    Legacy    = 1 << 2, // Legacy, for backwards compatibility. REVIEW alexeyk: implement sparse version and remove Legacy altogether.
    Gemm      = 1 << 3, // Uses convolution unrolling+GEMM technique. Works only for convos with full sharing.

    All       = Reference | CuDnn | Legacy | Gemm
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

    void Forward(const Mat& in, const Mat& kernel, Mat& out, Mat& workspace);

    void BackwardData(const Mat& srcGrad, const Mat& kernel, Mat& grad, bool accumulateGradient, Mat& workspace);

    void BackwardKernel(const Mat& srcGrad, const Mat& in, Mat& kernelGrad, bool accumulateGradient, bool allowReuse, Mat& workspace);

    void ForwardPooling(const Mat& in, Mat& out);

    void BackwardPooling(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad);

    void MaxUnpooling(const Mat& out, const Mat& poolIn, Mat& in);

    std::shared_ptr<const ConvolveGeometry> Geometry() const { return m_geometry; }

    static std::unique_ptr<ConvolutionEngine<ElemType>> Create(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, 
                                                               ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind = PoolKind::None,
                                                               ConvolutionEngineKind enabledEngines = ConvolutionEngineKind::All,
                                                               std::wstring logPrefix = L"", bool forceDeterministicAlgorithms = false, bool poolIncludePad = false);

    DISABLE_COPY_AND_MOVE(ConvolutionEngine);

    // REVIEW alexeyk: This is not enough as there should be invalidation of auto-tuner state in cuDNN engine. Fine for now if it works.
    void SetmMaxTempMemSizeInSamples(const size_t maxTempMemSizeInSamples)
    {
        m_maxTempMemSizeInSamples = maxTempMemSizeInSamples;
    }

    virtual bool ImplementsGradientOverwriteOptimization() const { return false; }

protected:
    ConvolutionEngine(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind, bool poolIncludePad = false)
        : m_geometry(geometry), m_deviceId(deviceId), m_imageLayout(imageLayout), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples), m_poolKind(poolKind), m_poolIncludePad(poolIncludePad)
    {
        assert(m_geometry != nullptr);
    }

    virtual void EnsureCompatible() = 0;

    virtual void EnsureConvolutionInitialized() = 0;

    virtual void ForwardCore(const Mat& in, const Mat& kernel, Mat& out, Mat& workspace) = 0;

    virtual void BackwardDataCore(const Mat& srcGrad, const Mat& kernel, Mat& grad, bool accumulateGradient, Mat& workspace) = 0;

    virtual void BackwardKernelCore(const Mat& srcGrad, const Mat& in, Mat& kernelGrad, bool accumulateGradient, bool allowReuse, Mat& workspace) = 0;

    virtual void EnsurePoolingInitialized() = 0;

    virtual void ForwardPoolingCore(const Mat& in, Mat& out) = 0;

    virtual void BackwardPoolingCore(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad) = 0;

    virtual void MaxUnpoolingCore(const Mat& out, const Mat& poolIn, Mat& in) = 0;

protected:
    ConvolveGeometryPtr m_geometry;
    DEVICEID_TYPE m_deviceId;
    ImageLayoutKind m_imageLayout;
    size_t m_maxTempMemSizeInSamples;
    PoolKind m_poolKind;
    bool m_poolIncludePad;
};

#pragma warning(pop)

static inline PoolKind PoolKindFrom(const wstring& s)
{
    if (s.empty() || AreEqualIgnoreCase(s, L"none"))
        return PoolKind::None;
    if (AreEqualIgnoreCase(s, L"max"))
        return PoolKind::Max;
    if (AreEqualIgnoreCase(s, L"average"))
        return PoolKind::Average;
    InvalidArgument("Unknown pooling kind: '%ls'. Supported values: 'none', 'max', 'average'.", s.c_str());
}

} } }
