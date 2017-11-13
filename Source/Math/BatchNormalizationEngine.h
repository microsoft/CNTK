//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Matrix.h"
#include "TensorShape.h" // for ImageLayoutKind

namespace Microsoft { namespace MSR { namespace CNTK {

//-------------------------------------------------------------
// Batch normalization engine interface.
//-------------------------------------------------------------
enum class BatchNormEngineKind
{
    None  = 0,
    Cntk  = 1,
    CuDnn = 1 << 1,

    All  = Cntk  | CuDnn
};

#pragma warning(push)
#pragma warning(disable : 4251)

template <class InoutType, class StatType = InoutType>
class MATH_API BatchNormEngine
{
public:
    using InoutMat = Matrix<InoutType>;
    using StatMat = Matrix<StatType>;

public:
    virtual ~BatchNormEngine() {};

    void Forward(const InoutMat& in, const StatMat& scale, const StatMat& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, StatMat& runMean, StatMat& runVariance,
                 InoutMat& out, double epsilon, StatMat& saveMean, StatMat& saveInvStdDev);

    void Backward(const InoutMat& in, const InoutMat& srcGrad, InoutMat& grad, const StatMat& scale, double blendFactor, const StatMat& saveMean, const StatMat& saveInvStdDev,
                  StatMat& scaleGrad, StatMat& biasGrad, bool accumulateDataGrad);

    static std::unique_ptr<BatchNormEngine<InoutType, StatType>> Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                                                             bool spatial, ImageLayoutKind imageLayout,
                                                             BatchNormEngineKind enabledEngines = BatchNormEngineKind::All);

    DISABLE_COPY_AND_MOVE(BatchNormEngine);

protected:
    BatchNormEngine(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                    bool spatial, ImageLayoutKind imageLayout)
                    : m_deviceId(deviceId), m_inOutT(inOutT), m_spatial(spatial), m_imageLayout(imageLayout)
    {
    }

    virtual void EnsureCompatible() = 0;

    // saveMean/saveInvStdDev return the actual mean/stddev used for normalization, except for blendFactor=1, these are unused and untouched
    virtual void ForwardCore(const InoutMat& in, const StatMat& scale, const StatMat& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, StatMat& runMean, StatMat& runVariance,
                 InoutMat& out, double epsilon, StatMat& saveMean, StatMat& saveInvStdDev) = 0;

    virtual void BackwardCore(const InoutMat& in, const InoutMat& srcGrad, InoutMat& grad, const StatMat& scale, double blendFactor, const StatMat& saveMean, const StatMat& saveInvStdDev,
                  StatMat& scaleGrad, StatMat& biasGrad, bool accumulateDataGrad) = 0;

protected:
    DEVICEID_TYPE m_deviceId;
    TensorShape m_inOutT;
    bool m_spatial;
    ImageLayoutKind m_imageLayout;
};

#pragma warning(pop)

}}}
