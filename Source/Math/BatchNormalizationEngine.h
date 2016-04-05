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

    void Forward(const Mat& in, const Mat& scale, const Mat& bias, double expAvgFactor, double blendFactor, Mat& runMean, Mat& runInvStdDev,
                 Mat& out, double epsilon, Mat& saveMean, Mat& saveInvStdDev);

    void Backward(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, const Mat& saveMean, const Mat& saveInvStdDev,
                  Mat& scaleGrad, Mat& biasGrad);

    static std::unique_ptr<BatchNormEngine<ElemType>> Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
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

    virtual void ForwardCore(const Mat& in, const Mat& scale, const Mat& bias, double expAvgFactor, double blendFactor, Mat& runMean, Mat& runInvStdDev,
                 Mat& out, double epsilon, Mat& saveMean, Mat& saveInvStdDev) = 0;

    virtual void BackwardCore(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, const Mat& saveMean, const Mat& saveInvStdDev,
                  Mat& scaleGrad, Mat& biasGrad) = 0;

protected:
    DEVICEID_TYPE m_deviceId;
    TensorShape m_inOutT;
    bool m_spatial;
    ImageLayoutKind m_imageLayout;
};

#pragma warning(pop)

} } }
