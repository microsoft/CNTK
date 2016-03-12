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

template <class ElemType>
class MATH_API BatchNormEngine
{
public:
    using Mat = Matrix<ElemType>;

public:
    virtual ~BatchNormEngine() = default;

    void Forward(const Mat& in, const Mat& scale, const Mat& bias, double expAvgFactor, Mat& runMean, Mat& runInvStdDev,
                 Mat& out, double epsilon, Mat& saveMean, Mat& saveInvStdDev);

    //void ForwardInference(const Mat& in, const Mat& scale, const Mat& bias, const Mat& runMean, const Mat& runInvStdDev,
    //                      Mat& out);

    //void Backward(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, const Mat& saveMean, const Mat& saveInvStdDev,
    //              Mat& scaleGrad, Mat& biasGrad);

    static std::unique_ptr<BatchNormEngine<ElemType>> Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                                                             const TensorShape& scaleBiasT, bool spatial, ImageLayoutKind imageLayout,
                                                             BatchNormEngineKind enabledEngines = BatchNormEngineKind::All);

    DISABLE_COPY_AND_MOVE(BatchNormEngine);

protected:
    BatchNormEngine(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                    const TensorShape& scaleBiasT, bool spatial, ImageLayoutKind imageLayout)
                    : m_deviceId(deviceId), m_inOutT(inOutT), m_scaleBiasT(scaleBiasT), m_spatial(spatial), m_imageLayout(imageLayout)
    {
    }

    virtual void EnsureCompatible() = 0;

    virtual void ForwardCore(const Mat& in, const Mat& scale, const Mat& bias, double expAvgFactor, Mat& runMean, Mat& runInvStdDev,
                 Mat& out, double epsilon, Mat& saveMean, Mat& saveInvStdDev) = 0;

protected:
    DEVICEID_TYPE m_deviceId;
    TensorShape m_inOutT;
    TensorShape m_scaleBiasT;
    bool m_spatial;
    ImageLayoutKind m_imageLayout;
};

#pragma warning(pop)

} } }
