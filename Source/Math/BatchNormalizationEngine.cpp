//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "BatchNormalizationEngine.h"
#include "CuDnnFactories.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void BatchNormEngine<ElemType>::Forward(const Mat& in, const Mat& scale, const Mat& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, Mat& runMean, Mat& runVariance,
                                        Mat& out, double epsilon, Mat& savedMean, Mat& savedInvStdDev)
{
    assert(in.GetNumRows() == m_inOutT.GetNumElements());
    assert(out.GetNumRows() == m_inOutT.GetNumElements());
    assert(in.GetNumCols() == out.GetNumCols());
    assert(std::isfinite(expAvgFactor) && (0 <= expAvgFactor && expAvgFactor <= 1));
    assert(std::isfinite(blendFactor) && (0 <= blendFactor && blendFactor <= 1));
    // In inference mode, must only use running statistics
    assert(!inferenceOnly || ((expAvgFactor == 0.0) && (blendFactor == 1.0)));
    assert(std::isfinite(epsilon) && epsilon > 0);
    if (!m_spatial)
    {
        assert(m_inOutT.GetNumElements() == scale.GetNumRows());
        assert(m_inOutT.GetNumElements() == bias.GetNumRows());
        assert(m_inOutT.GetNumElements() == runMean.GetNumRows());
        assert(m_inOutT.GetNumElements() == runVariance.GetNumRows());
    }
    else
    {
        assert((m_inOutT.GetNumElements() % scale.GetNumRows()) == 0);
        assert((m_inOutT.GetNumElements() % bias.GetNumRows()) == 0);
        assert((m_inOutT.GetNumElements() % runMean.GetNumRows()) == 0);
        assert((m_inOutT.GetNumElements() % runVariance.GetNumRows()) == 0);
    }
    assert(scale.GetNumCols() == 1);
    assert(bias.GetNumCols() == 1);
    assert(runMean.GetNumCols() == 1);
    assert(runVariance.GetNumCols() == 1);

    EnsureCompatible();
    ForwardCore(in, scale, bias, inferenceOnly, expAvgFactor, blendFactor, runMean, runVariance, out, epsilon, savedMean, savedInvStdDev);

    if (!inferenceOnly)
    {
        assert(!savedMean.IsEmpty());
        assert(!savedInvStdDev.IsEmpty());
        if (!m_spatial)
        {
            assert(m_inOutT.GetNumElements() == savedMean.GetNumRows());
            assert(m_inOutT.GetNumElements() == savedInvStdDev.GetNumRows());
        }
        else
        {
            assert((m_inOutT.GetNumElements() % savedMean.GetNumRows()) == 0);
            assert((m_inOutT.GetNumElements() % savedInvStdDev.GetNumRows()) == 0);
        }
        assert(savedMean.GetNumCols() == 1);
        assert(savedInvStdDev.GetNumCols() == 1);
    }
}

template <class ElemType>
void BatchNormEngine<ElemType>::Backward(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, double blendFactor,
                                         const Mat& savedMean, const Mat& savedInvStdDev, Mat& scaleGrad, Mat& biasGrad, bool accumulateDataGrad)
{
    assert(!savedMean.IsEmpty());
    assert(!savedInvStdDev.IsEmpty());
    EnsureCompatible();
    BackwardCore(in, srcGrad, grad, scale, blendFactor, savedMean, savedInvStdDev, scaleGrad, biasGrad, accumulateDataGrad);
}

template <class ElemType>
class CntkBatchNormEngine : public BatchNormEngine<ElemType>
{
public:
    using Base = BatchNormEngine<ElemType>;
    using typename Base::Mat;

public:
    CntkBatchNormEngine(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                        bool spatial, ImageLayoutKind imageLayout)
                        : Base(deviceId, inOutT, spatial, imageLayout)
    {
    }

protected:
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_inOutT;
    using Base::m_spatial;

    void EnsureCompatible() override
    {
        if (m_spatial && m_imageLayout == ImageLayoutKind::HWC)
            InvalidArgument("CNTK batch normalization supports only cudnn(CHW) layout.");
    }

    void ForwardCore(const Mat& in, const Mat& scale, const Mat& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, Mat& runMean, Mat& runVariance,
                     Mat& out, double epsilon, Mat& savedMean, Mat& savedInvStdDev) override
    {
        in.BatchNormalizationForward(scale, bias, inferenceOnly, expAvgFactor, blendFactor, runMean, runVariance, out, epsilon, savedMean, savedInvStdDev);
    }

    void BackwardCore(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, double blendFactor, const Mat& savedMean, const Mat& savedInvStdDev,
                      Mat& scaleGrad, Mat& biasGrad, bool accumulateDataGrad) override
    {
        if (!accumulateDataGrad)
            grad.SetValue((ElemType)0);

        srcGrad.BatchNormalizationBackward(in, grad, scale, blendFactor, savedMean, savedInvStdDev, scaleGrad, biasGrad);
    }
};

template class CntkBatchNormEngine<float>;
template class CntkBatchNormEngine<double>;

template <typename T> bool HasFlag(T src, T testFlag)
{
    return ((int)src & (int)testFlag) != 0;
}

template <class ElemType>
std::unique_ptr<BatchNormEngine<ElemType>> BatchNormEngine<ElemType>::Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                                                                             bool spatial, ImageLayoutKind imageLayout,
                                                                             BatchNormEngineKind enabledEngines)
{
    // Use CNTK as default batch norm engine.
    if (HasFlag(enabledEngines, BatchNormEngineKind::Cntk))
    {
        if (GetMathLibTraceLevel() > 0)
            fprintf(stderr, "Using CNTK batch normalization engine.\n");

        return std::make_unique<CntkBatchNormEngine<ElemType>>(deviceId, inOutT, spatial, imageLayout);
    }

    if (HasFlag(enabledEngines, BatchNormEngineKind::CuDnn))
    {
        if (GetMathLibTraceLevel() > 0)
            fprintf(stderr, "Using cuDNN batch normalization engine.\n");

        return CuDnnBatchNormEngineFactory<ElemType>::Create(deviceId, inOutT, spatial, imageLayout);
    }

    RuntimeError("Could not find appropriate batch normalization engine.");
}

template class BatchNormEngine<float>;
template class BatchNormEngine<double>;

}}}
