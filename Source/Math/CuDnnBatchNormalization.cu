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
                        bool spatial, ImageLayoutKind imageLayout)
                        : Base(deviceId, inOutT, spatial, imageLayout),
                        m_cudnn(CuDnn::Instance()),
                        m_inOutCuDnnT(GetInOutTensor(inOutT), CuDnnTensor::GetDataType<ElemType>()),
                        m_scaleBiasCuDnnT(GetScaleBiasTensor(inOutT, spatial), CuDnnTensor::GetDataType<ElemType>())
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
            InvalidArgument("cuDNN batch normalization supports only cudnn(CHW) layout.");
        if (m_inOutT.GetRank() > 4)
            InvalidArgument("cuDNN batch normalization supports tensors of max 4 dimensions.");
    }

    void ForwardCore(const Mat& in, const Mat& scale, const Mat& bias, double expAvgFactor, double blendFactor, Mat& runMean, Mat& runInvStdDev,
                     Mat& out, double epsilon, Mat& saveMean, Mat& saveInvStdDev) override
    {
        // REVIEW alexeyk: there might be a way to do this in cuDNN.
        if (blendFactor != 0 && (blendFactor != 1 || expAvgFactor > 0))
            InvalidArgument("cuDNN batch normalization engine currently supports blendTimeConstant of 0 or 1 only.");

        m_inOutCuDnnT.UpdateBatchSize(in.GetNumCols());
        cudnnBatchNormMode_t mode = m_spatial ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
        // cuDNN will fail with BAD_PARAM if epsilon < CUDNN_BN_MIN_EPSILON.
        epsilon = max(epsilon, CUDNN_BN_MIN_EPSILON);
        // expAvgFactor == 0 && blendFactor == 1 means we are in eval mode.
        if (expAvgFactor == 0 && blendFactor == 1)
        {
            CUDNN_CALL(cudnnBatchNormalizationForwardInference(*m_cudnn, mode, &C::One, &C::Zero, m_inOutCuDnnT, ptr(in), m_inOutCuDnnT, ptr(out),
                m_scaleBiasCuDnnT, ptr(scale), ptr(bias), ptr(runMean), ptr(runInvStdDev), epsilon));
        }
        else
        {
            CUDNN_CALL(cudnnBatchNormalizationForwardTraining(*m_cudnn, mode, &C::One, &C::Zero, m_inOutCuDnnT, ptr(in),
                m_inOutCuDnnT, ptr(out), m_scaleBiasCuDnnT, ptr(scale), ptr(bias), expAvgFactor, ptr(runMean), ptr(runInvStdDev),
                epsilon, ptr(saveMean), ptr(saveInvStdDev)));
        }
    }

    void BackwardCore(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, const Mat& saveMean, const Mat& saveInvStdDev,
                      Mat& scaleGrad, Mat& biasGrad) override
    {
        m_inOutCuDnnT.UpdateBatchSize(srcGrad.GetNumCols());
        cudnnBatchNormMode_t mode = m_spatial ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
        // REVIEW alexeyk: remove once Philly is upgraded to prod version. Also change betaParamDiff to 1 and update CNTK BN engine.
#if CUDNN_PATCHLEVEL >= 7
        CUDNN_CALL(cudnnBatchNormalizationBackward(*m_cudnn, mode, &C::One, &C::One, &C::One, &C::Zero, m_inOutCuDnnT, ptr(in), m_inOutCuDnnT, ptr(srcGrad), m_inOutCuDnnT, ptr(grad),
            m_scaleBiasCuDnnT, ptr(scale), ptr(scaleGrad), ptr(biasGrad), CUDNN_BN_MIN_EPSILON, ptr(saveMean), ptr(saveInvStdDev)));
#else
        CUDNN_CALL(cudnnBatchNormalizationBackward(*m_cudnn, mode, &C::One, &C::One, m_inOutCuDnnT, ptr(in), m_inOutCuDnnT, ptr(srcGrad), m_inOutCuDnnT, ptr(grad),
            m_scaleBiasCuDnnT, ptr(scale), ptr(scaleGrad), ptr(biasGrad), CUDNN_BN_MIN_EPSILON, ptr(saveMean), ptr(saveInvStdDev)));
#endif
    }

private:
    static ElemType* ptr(Mat& src)
    {
        return src.Data();
    }
    static const ElemType* ptr(const Mat& src)
    {
        return src.Data();
    }

    static TensorShape GetInOutTensor(const TensorShape& inOutT)
    {
        // cuDNN supports only 3D and 4D tensors (in cuDNN docs it's 4D and 5D dues to N dimension)
        // even for non-spatial inputs so expand the tensor if needed.
        if (inOutT.GetRank() > 2)
            return inOutT;
        SmallVector<size_t> v(std::max(inOutT.GetRank(), (size_t)3), 1);
        for (size_t i = 0; i < inOutT.GetRank(); i++)
            v[i] = inOutT[i];
        return TensorShape(v);
    }

    static TensorShape GetScaleBiasTensor(const TensorShape& inOutT, bool spatial)
    {
        if (!spatial)
            return GetInOutTensor(inOutT);

        const auto& t = GetInOutTensor(inOutT);
        SmallVector<size_t> v(t.GetRank(), 1);
        v[v.size() - 1] = t[t.GetRank() - 1];
        return TensorShape(v);
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
                                                                                         bool spatial, ImageLayoutKind imageLayout)
{
    return std::make_unique<CuDnnBatchNormEngine<ElemType>>(deviceId, inOutT, spatial, imageLayout);
}

template class CuDnnBatchNormEngineFactory<float>;
template class CuDnnBatchNormEngineFactory<double>;

CudaTimer::~CudaTimer()
{
    // TODO: Should not throw if std::uncaught_exception()
    if (m_start != nullptr)
        CUDA_CALL(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(m_start)));
    if (m_stop != nullptr)
        CUDA_CALL(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(m_stop)));
}
void CudaTimer::Start()
{
    cudaEvent_t start;
    cudaEvent_t stop;
    if (m_start != nullptr)
        CUDA_CALL(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(m_start)));
    if (m_stop != nullptr)
        CUDA_CALL(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(m_stop)));
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    m_start = start;
    m_stop = stop;
    CUDA_CALL(cudaEventRecord(start, GetStream()));
}
void CudaTimer::Stop()
{
    CUDA_CALL(cudaEventRecord(reinterpret_cast<cudaEvent_t>(m_stop), GetStream()));
    CUDA_CALL(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(m_stop)));
}
float CudaTimer::Elapsed()
{
    float ms;
    CUDA_CALL(cudaEventElapsedTime(&ms, reinterpret_cast<cudaEvent_t>(m_start), reinterpret_cast<cudaEvent_t>(m_stop)));
    return ms;
}

} } }
