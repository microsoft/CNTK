//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "BatchNormalizationEngine.h"
#include "CuDnnFactories.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void BatchNormEngine<ElemType>::Forward(const Mat& in, const Mat& scale, const Mat& bias, double expAvgFactor, Mat& runMean, Mat& runInvStdDev,
                                        Mat& out, double epsilon, Mat& saveMean, Mat& saveInvStdDev)
{
    assert(in.GetNumRows() == m_inOutT.GetNumElements());
    assert(out.GetNumRows() == m_inOutT.GetNumElements());
    assert(in.GetNumCols() == out.GetNumCols());
    assert(std::isfinite(epsilon) && epsilon > 0);
    assert(std::isfinite(expAvgFactor) && (0 < expAvgFactor && expAvgFactor <= 1));
    if (!m_spatial)
    {
        assert(m_inOutT.GetNumElements() == scale.GetNumRows());
        assert(m_inOutT.GetNumElements() == bias.GetNumRows());
        assert(m_inOutT.GetNumElements() == runMean.GetNumRows());
        assert(m_inOutT.GetNumElements() == runInvStdDev.GetNumRows());
        assert(m_inOutT.GetNumElements() == saveMean.GetNumRows());
        assert(m_inOutT.GetNumElements() == saveInvStdDev.GetNumRows());
    }
    else
    {
        assert((m_inOutT.GetNumElements() % scale.GetNumRows()) == 0);
        assert((m_inOutT.GetNumElements() % bias.GetNumRows()) == 0);
        assert((m_inOutT.GetNumElements() % runMean.GetNumRows()) == 0);
        assert((m_inOutT.GetNumElements() % runInvStdDev.GetNumRows()) == 0);
        assert((m_inOutT.GetNumElements() % saveMean.GetNumRows()) == 0);
        assert((m_inOutT.GetNumElements() % saveInvStdDev.GetNumRows()) == 0);
    }
    assert(scale.GetNumCols() == 1);
    assert(bias.GetNumCols() == 1);
    assert(runMean.GetNumCols() == 1);
    assert(runInvStdDev.GetNumCols() == 1);
    assert(saveMean.GetNumCols() == 1);
    assert(saveInvStdDev.GetNumCols() == 1);

    EnsureCompatible();
    ForwardCore(in, scale, bias, expAvgFactor, runMean, runInvStdDev, out, epsilon, saveMean, saveInvStdDev);
}

template <class ElemType>
void BatchNormEngine<ElemType>::ForwardInference(const Mat& in, const Mat& scale, const Mat& bias,
                                                 const Mat& runMean, const Mat& runInvStdDev, Mat& out)
{
    EnsureCompatible();
    ForwardInferenceCore(in, scale, bias, runMean, runInvStdDev, out);
}

template <class ElemType>
void BatchNormEngine<ElemType>::Backward(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, 
                                         const Mat& saveMean, const Mat& saveInvStdDev, Mat& scaleGrad, Mat& biasGrad)
{
    EnsureCompatible();
    BackwardCore(in, srcGrad, grad, scale, saveMean, saveInvStdDev, scaleGrad, biasGrad);
}

template class BatchNormEngine<float>;
template class BatchNormEngine<double>;

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

    void ForwardCore(const Mat& in, const Mat& scale, const Mat& bias, double expAvgFactor, Mat& runMean, Mat& runInvStdDev,
                     Mat& out, double epsilon, Mat& saveMean, Mat& saveInvStdDev) override
    {
        in.BatchNormalizationForward(scale, bias, expAvgFactor, runMean, runInvStdDev, out, epsilon, saveMean, saveInvStdDev);
    }

    void ForwardInferenceCore(const Mat& in, const Mat& scale, const Mat& bias, const Mat& runMean, const Mat& runInvStdDev, Mat& out) override
    {
        in.BatchNormalizationForwardInference(scale, bias, runMean, runInvStdDev, out);
    }

    void BackwardCore(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, const Mat& saveMean, const Mat& saveInvStdDev,
                      Mat& scaleGrad, Mat& biasGrad) override
    {
        srcGrad.BatchNormalizationBackward(in, grad, scale, saveMean, saveInvStdDev, scaleGrad, biasGrad);
    }
};

template class CntkBatchNormEngine<float>;
template class CntkBatchNormEngine<double>;

template <typename T>
bool HasFlag(T src, T testFlag)
{
    return ((int)src & (int)testFlag) != 0;
}

template <class ElemType>
std::unique_ptr<BatchNormEngine<ElemType>> BatchNormEngine<ElemType>::Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                                                                             bool spatial, ImageLayoutKind imageLayout,
                                                                             BatchNormEngineKind enabledEngines = BatchNormEngineKind::All)
{
    // Use CNTK as default batch norm engine.
    if (HasFlag(enabledEngines, BatchNormEngineKind::Cntk))
    {
        fprintf(stderr, "\nUsing CNTK batch normalization engine.\n");
        return std::make_unique<CntkBatchNormEngine<ElemType>>(deviceId, inOutT, spatial, imageLayout);
    }

    if (HasFlag(enabledEngines, BatchNormEngineKind::CuDnn))
    {
        fprintf(stderr, "\nUsing cuDNN batch normalization engine.\n");
        return CuDnnBatchNormEngineFactory<ElemType>::Create(deviceId, inOutT, spatial, imageLayout);
    }

    RuntimeError("Could not find appropriate batch normalization engine.");
}

} } }
