//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Matrix.h"
#include "TensorShape.h" // for ImageLayoutKind

namespace Microsoft
{
namespace MSR
{
namespace CNTK
{

enum class EltWiseEngineKind
{
    None = 0,
    MKLDNN = 1 << 1,
    All = MKLDNN
};
enum class UnaryEltWiseKind
{
    RELU
};
template <class ElemType>
class MATH_API UnaryEltWiseEngine
{
public:
    using Mat = Matrix<ElemType>;

public:
    virtual ~UnaryEltWiseEngine(){};

    void Forward(const Mat& in, Mat& out, bool inferenceOnly);

    void Backward(const Mat& in, const Mat& srcGrad, Mat& grad);

    static std::unique_ptr<UnaryEltWiseEngine<ElemType>>
    Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT, ImageLayoutKind imageLayout, UnaryEltWiseKind kind,
           EltWiseEngineKind enabledEngines = EltWiseEngineKind::All);

    DISABLE_COPY_AND_MOVE(UnaryEltWiseEngine);

protected:
    UnaryEltWiseEngine(DEVICEID_TYPE deviceId, const TensorShape& inOutT, ImageLayoutKind imageLayout)
        : m_deviceId(deviceId), m_inOutT(inOutT), m_imageLayout(imageLayout)
    {
    }

    virtual void EnsureCompatible() = 0;

    virtual void ForwardCore(const Mat& in, Mat& out, bool inferenceOnly) = 0;

    virtual void BackwardCore(const Mat& in, const Mat& srcGrad, Mat& grad) = 0;

protected:
    DEVICEID_TYPE m_deviceId;
    TensorShape m_inOutT;
    ImageLayoutKind m_imageLayout;
};

} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
