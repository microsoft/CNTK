//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "EltWiseEngine.h"
#pragma warning(disable : 4661)
#ifdef USE_MKLDNN
#include "./mkldnn/mkldnn_relu-inl.h"
#endif

namespace Microsoft
{
namespace MSR
{
namespace CNTK
{

template <class ElemType>
void UnaryEltWiseEngine<ElemType>::Forward(const Mat& in, Mat& out, bool inferenceOnly)
{
    EnsureCompatible();
    ForwardCore(in, out, inferenceOnly);
}

template <class ElemType>
void UnaryEltWiseEngine<ElemType>::Backward(const Mat& in, const Mat& srcGrad, Mat& grad)
{
    EnsureCompatible();
    BackwardCore(in, srcGrad, grad);
}

template <typename T>
bool HasFlag(T src, T testFlag)
{
    return ((int) src & (int) testFlag) != 0;
}

#ifdef USE_MKLDNN
template <class ElemType>
class MklDnnUnaryEltWiseEngine : public UnaryEltWiseEngine<ElemType>
{
public:
    size_t m_prevBatchSize = 0;
    using Base = UnaryEltWiseEngine<ElemType>;
    using typename Base::Mat;
    MKLDNNReluOp<ElemType>* m_relu;

public:
    MklDnnUnaryEltWiseEngine(DEVICEID_TYPE deviceId, const TensorShape& inOutT, ImageLayoutKind imageLayout)
        : Base(deviceId, inOutT, imageLayout), m_relu(NULL)
    {
    }
    ~MklDnnUnaryEltWiseEngine()
    {
        if (m_relu != NULL)
            delete m_relu;
    }

protected:
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_inOutT;

    void EnsureCompatible() override
    {
        if (m_imageLayout == ImageLayoutKind::HWC)
            InvalidArgument("CNTK batch normalization supports only (CHW) layout.");
        // MKL DNN do not only support 3D tensor
        if (m_inOutT.GetRank() != 3)
            InvalidArgument("MKLDNN batch normalization supports only 3D tensor.");
    }

    void ForwardCore(const Mat& in, Mat& out, bool inferenceOnly) override
    {
        size_t batchSize = in.GetNumCols();
        if (m_prevBatchSize == 0)
            m_prevBatchSize = batchSize;
        bool samBatchSize = batchSize == m_prevBatchSize;
        if (!samBatchSize && m_relu != NULL)
        {
            delete m_relu;
            m_relu = NULL;
            m_prevBatchSize = batchSize;
        }
        if (m_relu == NULL)
        {
            m_relu = new MKLDNNReluOp<ElemType>(m_inOutT, m_imageLayout);
        }
        m_relu->Forward(in, out, inferenceOnly);
    }

    void BackwardCore(const Mat& in, const Mat& srcGrad, Mat& grad) override
    {
        if (m_relu == NULL)
        {
            m_relu = new MKLDNNReluOp<ElemType>(m_inOutT, m_imageLayout);
        }
        // Todo:accumulateDataGrad
        m_relu->Backward(in, srcGrad, grad);
    }

public:
    static bool IsSupported(DEVICEID_TYPE deviceId, const TensorShape& inOutT)
    {
        if (inOutT.GetRank() != 3)
            return false;
        // MKL DNN do not support double
        const std::type_info& ti1 = typeid(ElemType);
        const std::type_info& ti2 = typeid(float);
        if (ti1.hash_code() != ti2.hash_code())
        {
            return false;
        }
        return deviceId < 0;
    }
};

template class MklDnnUnaryEltWiseEngine<float>;
template class MklDnnUnaryEltWiseEngine<double>;
#endif
template <class ElemType>
std::unique_ptr<UnaryEltWiseEngine<ElemType>>
UnaryEltWiseEngine<ElemType>::Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT, ImageLayoutKind imageLayout,
                                     UnaryEltWiseKind kind, EltWiseEngineKind enabledEngines)
{
    if (UnaryEltWiseKind::RELU == kind)
    {
#ifdef USE_MKLDNN
        if (HasFlag(enabledEngines, EltWiseEngineKind::MKLDNN) &&
            MklDnnUnaryEltWiseEngine<ElemType>::IsSupported(deviceId, inOutT))
        {
            if (GetMathLibTraceLevel() > 0)
                fprintf(stderr, "Using CNTK MKL DNN Rectified Linear engine.\n");

            return std::make_unique<MklDnnUnaryEltWiseEngine<ElemType>>(deviceId, inOutT, imageLayout);
        }
#else
        UNUSED(enabledEngines);
        UNUSED(imageLayout);
        UNUSED(kind);
        UNUSED(deviceId);
        UNUSED(inOutT);
#endif
        if (GetMathLibTraceLevel() > 0)
            fprintf(stderr, "Could not find appropriate Rectified Linear engine.\n");
    }
    return nullptr;
}
template <>
std::unique_ptr<UnaryEltWiseEngine<half>>
UnaryEltWiseEngine<half>::Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT, ImageLayoutKind imageLayout,
                                 UnaryEltWiseKind kind, EltWiseEngineKind enabledEngines)
{
    UNUSED(deviceId);
    UNUSED(inOutT);
    UNUSED(imageLayout);
    UNUSED(kind);
    UNUSED(enabledEngines);
    return nullptr;
}
template class UnaryEltWiseEngine<float>;
template class UnaryEltWiseEngine<double>;
template class UnaryEltWiseEngine<half>;
} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
