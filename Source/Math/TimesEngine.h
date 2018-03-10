//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Matrix.h"
#include "TensorShape.h" // for ImageLayoutKind

namespace Microsoft { namespace MSR { namespace CNTK {

enum class TimesKind
{
    None  = 0,
    MKLDNN = 1 << 1,
    All  = MKLDNN
};


template <class ElemType>
class MATH_API TimesEngine
{
public:
    using Mat = Matrix<ElemType>;

public:
    virtual ~TimesEngine() {};

    void Forward(const Mat& in, const Mat& weight, Mat& out);

    void BackwardWeight(const Mat& in, const Mat& grad, Mat& gradWeight, bool accumulateGradient, Mat& workspace);

    void BackwardData(const Mat& grad, const Mat& weight, Mat& gradData, bool accumulateGradient, Mat& workspace);

    static std::unique_ptr<TimesEngine<ElemType>> Create(DEVICEID_TYPE deviceId,
                  SmallVector<size_t>& dimA, SmallVector<size_t>& dimB,
                  TimesKind enabledEngines = TimesKind::All);

    DISABLE_COPY_AND_MOVE(TimesEngine);

protected:
    TimesEngine(DEVICEID_TYPE deviceId, SmallVector<size_t>& dimA, SmallVector<size_t>& dimB)
                    : m_deviceId(deviceId)
    {
      UNUSED(dimA);
      UNUSED(dimB);
    }

    virtual void EnsureCompatible() = 0;

    virtual void ForwardCore(const Mat& in, const Mat& weight, Mat& out) = 0;

    virtual void BackwardWeightCore(const Mat& in, const Mat& grad, Mat& gradWeight, bool accumulateGradient, Mat& workspace) = 0;

    virtual void BackwardDataCore(const Mat& grad, const Mat& weight, Mat& gradData, bool accumulateGradient, Mat& workspace) = 0;

protected:
    DEVICEID_TYPE m_deviceId;
};


}}}
