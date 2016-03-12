//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "BatchNormalizationEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void BatchNormEngine<ElemType>::Forward(const Mat& in, const Mat& scale, const Mat& bias, double expAvgFactor, Mat& runMean, Mat& runInvStdDev,
                                        Mat& out, double epsilon, Mat& saveMean, Mat& saveInvStdDev)
{
    EnsureCompatible();
    ForwardCore(in, scale, bias, expAvgFactor, runMean, runInvStdDev, out, epsilon, saveMean, saveInvStdDev);
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
                        const TensorShape& scaleBiasT, bool spatial, ImageLayoutKind imageLayout)
                        : Base(deviceId, inOutT, scaleBiasT, spatial, imageLayout)
    {
    }

protected:
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_inOutT;
    using Base::m_scaleBiasT;
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
};

template class CntkBatchNormEngine<float>;
template class CntkBatchNormEngine<double>;

template <class ElemType>
std::unique_ptr<BatchNormEngine<ElemType>> BatchNormEngine<ElemType>::Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                                                                             const TensorShape& scaleBiasT, bool spatial, ImageLayoutKind imageLayout,
                                                                             BatchNormEngineKind enabledEngines = BatchNormEngineKind::All)
{
    if (spatial && imageLayout == ImageLayoutKind::HWC)
        InvalidArgument("Batch normalization is not supported for legacy(HWC) layout. Please use cudnn(CHW) layout instead.");

    auto isEnabled = [=](BatchNormEngineKind eng) { return ((int)enabledEngines & (int)eng) != 0; };
    // Use CNTK as default batch norm engine.
    if (isEnabled(BatchNormEngineKind::Cntk))
    {
        fprintf(stderr, "Using CNTK batch normalization engine.\n");
        return std::make_unique<CntkBatchNormEngine<ElemType>>(deviceId, inOutT, scaleBiasT, spatial, imageLayout);
    }

    if (isEnabled(BatchNormEngineKind::CuDnn))
    {
        fprintf(stderr, "Using cuDNN batch normalization engine.\n");
        return std::make_unique<CntkBatchNormEngine<ElemType>>(deviceId, inOutT, scaleBiasT, spatial, imageLayout);
    }

    RuntimeError("Failed to find appropriate batch normalization engine.");
}

} } }
