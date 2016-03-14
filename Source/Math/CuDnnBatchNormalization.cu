//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CuDnnFactories.h"
#include "BatchNormalizationEngine.h"
#include "CuDnnCommon.h"
#include "GPUMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class CuDnnBatchNormEngine : public BatchNormEngine<ElemType>
{
public:
    using Base = BatchNormEngine<ElemType>;
    using typename Base::Mat;

public:
    CuDnnBatchNormEngine(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                        const TensorShape& scaleBiasT, bool spatial, ImageLayoutKind imageLayout)
                        : Base(deviceId, inOutT, scaleBiasT, spatial, imageLayout),
                        m_cudnn(CuDnn::Instance()),
                        m_inOutCuDnnT(inOutT, CuDnnTensor::GetDataType<ElemType>()),
                        m_scaleBiasCuDnnT(scaleBiasT, CuDnnTensor::GetDataType<ElemType>())
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
            InvalidArgument("cuDNN batch normalization supports only cudnn(CHW) layout.");
    }

    void ForwardCore(const Mat& in, const Mat& scale, const Mat& bias, double expAvgFactor, Mat& runMean, Mat& runInvStdDev,
                     Mat& out, double epsilon, Mat& saveMean, Mat& saveInvStdDev) override
    {
        m_inOutCuDnnT.UpdateBatchSize(in.GetNumCols());
        cudnnBatchNormMode_t mode = m_spatial ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
        // cuDNN will fail with BAD_PARAM if epsilon < CUDNN_BN_MIN_EPSILON.
        epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);
        CUDNN_CALL(cudnnBatchNormalizationForwardTraining(*m_cudnn, mode, &C::One, &C::Zero, m_inOutCuDnnT, ptr(in), 
            m_inOutCuDnnT, ptr(out), m_scaleBiasCuDnnT, ptr(scale), ptr(bias), expAvgFactor, ptr(runMean), ptr(runInvStdDev),
            epsilon, ptr(saveMean), ptr(saveInvStdDev)));
    }

private:
    template <typename ElemType>
    static ElemType* ptr(Matrix<ElemType>& src)
    {
        return src.BufferPointer();
    }
    template <typename ElemType>
    static const ElemType* ptr(const Matrix<ElemType>& src)
    {
        return src.BufferPointer();
    }

private:
    using C = Consts<ElemType>;

    CuDnn::ptr_t m_cudnn;
    CuDnnTensor m_inOutCuDnnT;
    CuDnnTensor m_scaleBiasCuDnnT;
};

template class CuDnnBatchNormEngine<float>;
template class CuDnnBatchNormEngine<double>;

template <typename ElemType>
std::unique_ptr<BatchNormEngine<ElemType>> CuDnnBatchNormEngineFactory<ElemType>::Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                                                                                         const TensorShape& scaleBiasT, bool spatial,
                                                                                         ImageLayoutKind imageLayout)
{
    return std::make_unique<CuDnnBatchNormEngine<ElemType>>(deviceId, inOutT, scaleBiasT, spatial, imageLayout);
}

template class CuDnnBatchNormEngineFactory<float>;
template class CuDnnBatchNormEngineFactory<double>;

} } }
