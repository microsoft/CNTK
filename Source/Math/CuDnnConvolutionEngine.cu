//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CuDnnFactories.h"
#include "GPUMatrix.h"
#include <typeinfo>
#include <typeindex>
#include "CuDnnCommon.h"

template <>
const char* CudaErrString<cudnnStatus_t>(cudnnStatus_t x)
{
    return cudnnGetErrorString(x);
}

// A note on the formats: CNTK originally used NHWC for input/output tensors and CHWN for filters.
// Such formats have very limited support in cuDNN and not used in other frameworks.
// CNTK with cuDNN by default uses NCHW formats for both inputs/outputs and filters.
#define TENSOR_FORMAT CUDNN_TENSOR_NCHW
#define FILTER_FORMAT CUDNN_TENSOR_NCHW

namespace Microsoft { namespace MSR { namespace CNTK {

static bool IsGpu(DEVICEID_TYPE deviceId)
{
    return deviceId >= 0;
}

class CuDnnFilter
{
public:
    CuDnnFilter(const ConvolveGeometry& geometry, cudnnDataType_t dataType)
        : m_filter(nullptr)
    {
        CUDNN_CALL(cudnnCreateFilterDescriptor(&m_filter));
        // Set cuDNN filter dimensions. cuDNN uses row-major format while TensorShape - column-major
        // so conversion is required.
        const auto& filt = geometry.KernelShape();
        size_t mapCount = geometry.GetMapCount(geometry.InputShape().GetRank() - 1);
        if (mapCount != geometry.MapCount().GetNumElements())
            InvalidArgument("cuDNN does not support map tensor of this configuration.");
        SmallVector<int> dims(filt.GetRank() + 1);
        for (int i = 0; i < filt.GetRank(); i++)
            dims[dims.size() - 1 - i] = (int)filt[i];
        // Set map count(aka K) dimension.
        dims[0] = (int)mapCount;
        CUDNN_CALL(cudnnSetFilterNdDescriptor_v4(m_filter, dataType, FILTER_FORMAT, (int)dims.size(), dims.data()));
    }

    ~CuDnnFilter()
    {
        if (m_filter != nullptr)
        {
            cudnnDestroyFilterDescriptor(m_filter);
            m_filter = nullptr;
        }
    }

    operator cudnnFilterDescriptor_t() const
    {
        return m_filter;
    }

    DISABLE_COPY_AND_MOVE(CuDnnFilter);

private:
    cudnnFilterDescriptor_t m_filter;
};

class CuDnnConv
{
public:
    CuDnnConv(const ConvolveGeometry& geometry, cudnnDataType_t dataType)
        : m_conv(nullptr)
    {
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&m_conv));
        // Set cuDNN convolution parameters. cuDNN uses row-major format while TensorShape - column-major
        // so conversion is required. Also, for 2D convolutions (which have 3D tensor shapes)
        // cuDNN uses 2D descriptors while for 3D convolutions - 3D so we need to ignore
        // rightmost dimension in ConvolveGeometry tensors.
        SmallVector<int> stride(geometry.InputShape().GetRank() - 1);
        SmallVector<int> pad(stride.size());
        for (int i = 0; i < stride.size(); i++)
        {
            stride[stride.size() - 1 - i] = (int)geometry.GetStride(i);
            pad[stride.size() - 1 - i] = geometry.GetLowerPad(i);
        }
        SmallVector<int> upscale(stride.size(), 1);
        CUDNN_CALL(cudnnSetConvolutionNdDescriptor(m_conv, (int)stride.size(), pad.data(),
                                                   stride.data(), upscale.data(),
                                                   CUDNN_CROSS_CORRELATION, dataType));
    }

    ~CuDnnConv()
    {
        if (m_conv != nullptr)
        {
            cudnnDestroyConvolutionDescriptor(m_conv);
            m_conv = nullptr;
        }
    }

    operator cudnnConvolutionDescriptor_t() const
    {
        return m_conv;
    }

    DISABLE_COPY_AND_MOVE(CuDnnConv);

private:
    cudnnConvolutionDescriptor_t m_conv;
};

class CuDnnPool
{
public:
    CuDnnPool(const ConvolveGeometry& geometry, PoolKind kind)
        : m_pool(nullptr)
    {
        assert(kind == PoolKind::Max || kind == PoolKind::Average);

        CUDNN_CALL(cudnnCreatePoolingDescriptor(&m_pool));
        // Set cuDNN pooling parameters. cuDNN uses row-major format while TensorShape - column-major
        // so conversion is required. Same as in convolution descriptor, cuDNN uses 2D descriptors
        // for 3D inputs.
        SmallVector<int> dims(geometry.InputShape().GetRank() - 1);
        SmallVector<int> stride(dims.size());
        SmallVector<int> pad(stride.size());
        int j = (int)dims.size() - 1;
        for (int i = 0; i < stride.size(); i++, j--)
        {
            dims[j] = (int)geometry.KernelShape()[i];
            stride[j] = (int)geometry.GetStride(i);
            pad[j] = geometry.GetLowerPad(i);
        }

        // Must use CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING to get the same results as in reference engine.
        CUDNN_CALL(cudnnSetPoolingNdDescriptor(m_pool,
                                               kind == PoolKind::Max ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                               (int)dims.size(), dims.data(), pad.data(), stride.data()));
    }

    ~CuDnnPool()
    {
        if (m_pool != nullptr)
        {
            cudnnDestroyPoolingDescriptor(m_pool);
            m_pool = nullptr;
        }
    }

    operator cudnnPoolingDescriptor_t() const
    {
        return m_pool;
    }

    DISABLE_COPY_AND_MOVE(CuDnnPool);

private:
    cudnnPoolingDescriptor_t m_pool;
};

template <class ElemType>
class CuDnnConvolutionEngine : public ConvolutionEngine<ElemType>
{
public:
    using Base = ConvolutionEngine<ElemType>;
    using typename Base::Mat;

public:
    CuDnnConvolutionEngine(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout,
                           size_t maxTempMemSizeInSamples, PoolKind poolKind)
                           : Base(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind),
                           m_cudnn(CuDnn::Instance()),
                           m_dataType(CuDnnTensor::GetDataType<ElemType>()),
                           m_inT(geometry->InputShape(), m_dataType),
                           m_outT(geometry->OutputShape(), m_dataType)
    {
    }

protected:
    using Base::m_geometry;
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_maxTempMemSizeInSamples;

    void EnsureCompatible() override
    {
        if (m_imageLayout != ImageLayoutKind::CHW)
            RuntimeError("cuDNN convolution engine supports only CHW/cudnn layout.");
        if (!IsGpu(m_deviceId))
            RuntimeError("cuDNN convolution engine supports GPU devices only.");
    }

    void EnsureConvolutionInitialized() override
    {
        if (m_filterT == nullptr)
        {
            m_filterT = std::make_unique<CuDnnFilter>(*m_geometry, m_dataType), 
            m_conv = std::make_unique<CuDnnConv>(*m_geometry, m_dataType);
        }
    }

    void ForwardCore(const Mat& in, const Mat& filter, Mat& out, Mat& workspace) override
    {
        size_t batchSize = in.GetNumCols();
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [this](int& calgo, cudnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
        {
            return cudnnFindConvolutionForwardAlgorithm(*m_cudnn, m_inT, *m_filterT, *m_conv, m_outT, MaxAlgoCount, &calgo, algoPerf);
        };
        auto staticFinder = [this](cudnnConvolutionFwdAlgo_t& algo) -> cudnnStatus_t
        {
            return cudnnGetConvolutionForwardAlgorithm(*m_cudnn, m_inT, *m_filterT, *m_conv, m_outT, CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, &algo);
        };
        FindBestAlgo(batchSize, m_fwdAlgo, finder, staticFinder);
        if (m_fwdAlgo.Algo.memory > 0)
            workspace.Resize((m_fwdAlgo.Algo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
        // Perform forward convolution operation.
        auto err = cudnnConvolutionForward(*m_cudnn, &C::One, m_inT, ptr(in), *m_filterT, ptr(filter), *m_conv,
                                           m_fwdAlgo.Algo.algo, ptr(workspace), m_fwdAlgo.Algo.memory, &C::Zero, m_outT, ptr(out));
        // There might be a case where cuDNN fails due to workspace being too small, try using no-workspace algo instead.
        // REVIEW alexeyk: NVIDIA is currently reviewing this issue.
        if (CUDNN_STATUS_INVALID_VALUE == err && m_fwdAlgo.Algo.memory > 0)
        {
            auto err2 = cudnnConvolutionForward(*m_cudnn, &C::One, m_inT, ptr(in), *m_filterT, ptr(filter), *m_conv,
                                                m_fwdAlgo.NoWorkspaceAlgo, nullptr, 0, &C::Zero, m_outT, ptr(out));
            // Update original error in case of success.
            if (CUDNN_STATUS_SUCCESS == err2)
                err = CUDNN_STATUS_SUCCESS;
        }
        CUDNN_CALL(err);
    }

    void BackwardDataCore(const Mat& srcGrad, const Mat& filter, Mat& grad, Mat& workspace) override
    {
        size_t batchSize = srcGrad.GetNumCols();
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [this](int& calgo, cudnnConvolutionBwdDataAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
        {
            return cudnnFindConvolutionBackwardDataAlgorithm(*m_cudnn, *m_filterT, m_outT, *m_conv, m_inT, MaxAlgoCount, &calgo, algoPerf);
        };
        auto staticFinder = [this](cudnnConvolutionBwdDataAlgo_t& algo) -> cudnnStatus_t
        {
            return cudnnGetConvolutionBackwardDataAlgorithm(*m_cudnn, *m_filterT, m_outT, *m_conv, m_inT, CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, 0, &algo);
        };
        FindBestAlgo(batchSize, m_backDataAlgo, finder, staticFinder);
        if (m_backDataAlgo.Algo.memory > 0)
            workspace.Resize((m_backDataAlgo.Algo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
        // Compute gradients with respect to the output tensor (data).
        CUDNN_CALL(cudnnConvolutionBackwardData(*m_cudnn, &C::One, *m_filterT, ptr(filter), m_outT, ptr(srcGrad), *m_conv, m_backDataAlgo.Algo.algo,
                                                ptr(workspace), m_backDataAlgo.Algo.memory, &C::One, m_inT, ptr(grad)));
    }

    void BackwardFilterCore(const Mat& srcGrad, const Mat& in, Mat& filterGrad, bool /*allowReuse*/, Mat& workspace) override
    {
        size_t batchSize = in.GetNumCols();
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [this](int& calgo, cudnnConvolutionBwdFilterAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
        {
            return cudnnFindConvolutionBackwardFilterAlgorithm(*m_cudnn, m_inT, m_outT, *m_conv, *m_filterT, MaxAlgoCount, &calgo, algoPerf);
        };
        auto staticFinder = [this](cudnnConvolutionBwdFilterAlgo_t& algo) -> cudnnStatus_t
        {
            return cudnnGetConvolutionBackwardFilterAlgorithm(*m_cudnn, m_inT, m_outT, *m_conv, *m_filterT, CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, &algo);
        };
        FindBestAlgo(batchSize, m_backFiltAlgo, finder, staticFinder);
        if (m_backFiltAlgo.Algo.memory > 0)
            workspace.Resize((m_backFiltAlgo.Algo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
        // Compute gradients with respect to the output tensor (data).
        CUDNN_CALL(cudnnConvolutionBackwardFilter(*m_cudnn, &C::One, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, m_backFiltAlgo.Algo.algo,
                                                  ptr(workspace), m_backFiltAlgo.Algo.memory, &C::One, *m_filterT, ptr(filterGrad)));
    }

    void EnsurePoolingInitialized() override
    {
        if (m_pool == nullptr)
            m_pool = std::make_unique<CuDnnPool>(*m_geometry, m_poolKind);
    }

    void ForwardPoolingCore(const Mat& in, Mat& out) override
    {
        size_t batchSize = in.GetNumCols();
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);
        CUDNN_CALL(cudnnPoolingForward(*m_cudnn, *(m_pool), &C::One, m_inT, ptr(in), &C::Zero, m_outT, ptr(out)));
    }

    void BackwardPoolingCore(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad) override
    {
        size_t batchSize = in.GetNumCols();
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);
        CUDNN_CALL(cudnnPoolingBackward(*m_cudnn, *(m_pool), &C::One, m_outT, ptr(out), m_outT, ptr(srcGrad),
                                        m_inT, ptr(in), &C::One, m_inT, ptr(grad)));
    }

private:
    using C = Consts<ElemType>;

    static const int MaxAlgoCount = 10;

    template <typename TAlgo, typename TFinder, typename TStaticFinder>
    void FindBestAlgo(size_t batchSize, TAlgo& algo, TFinder finder, TStaticFinder staticFinder)
    {
        if (!algo.NeedAutotuning(batchSize))
            return;
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);
        using CuDnnAlgoT = decltype(TAlgo::Algo);
        CuDnnAlgoT algoPerf[MaxAlgoCount];
        int calgo = 0;
        cudnnStatus_t err = finder(calgo, algoPerf);
        // Alloc failed - usually means cuDNN runtime auto-tuner could not allocate workspace.
        // In such case, use static auto-tuner with no workspace.
        if (err == CUDNN_STATUS_ALLOC_FAILED)
        {
            decltype(CuDnnAlgoT::algo) noMemAlgo;
            CUDNN_CALL(staticFinder(noMemAlgo));
            algo.CurMBSize = batchSize;
            algo.Algo = algoPerf[0];
            algo.Algo.algo = noMemAlgo;
            algo.Algo.memory = 0;
            algo.Algo.status = CUDNN_STATUS_SUCCESS;
            algo.NoWorkspaceAlgo = noMemAlgo;
            return;
        }
        CUDNN_CALL(err);
        assert(calgo > 0);
        size_t inputSampleSize = m_geometry->InputShape().GetNumElements();
        size_t maxMem = m_maxTempMemSizeInSamples == 0 ? (std::numeric_limits<size_t>::max)() : inputSampleSize * m_maxTempMemSizeInSamples * sizeof(ElemType);
        // Find best (fastest) algorithm which satisfies workspace requirements.
        auto res = std::find_if(algoPerf, algoPerf + calgo,
            [=](const CuDnnAlgoT& cur)
            {
                return cur.status == CUDNN_STATUS_SUCCESS && cur.memory <= maxMem;
            });
        if (res == algoPerf + calgo)
            RuntimeError("cuDNN could not find suitable algorithm for the current convolution configuration.");
        algo.CurMBSize = batchSize;
        algo.Algo = *res;
        // Find fastest algorithm that does NOT require workspace. It is used as a fallback algo in Forward function.
        res = std::find_if(algoPerf, algoPerf + calgo,
            [](const CuDnnAlgoT& cur)
            {
                return cur.status == CUDNN_STATUS_SUCCESS && cur.memory == 0;
            });
        if (res == algoPerf + calgo)
        {
            // In theory, this should never happen.
            RuntimeError("cuDNN could not find no-workspace algorithm for the current convolution configuration.");
        }
        else
            algo.NoWorkspaceAlgo = (*res).algo;
    }

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
    template <typename T>
    struct ConvAlgoInfo
    {
        using CuDnnAlgoT = decltype(T::algo);

        ConvAlgoInfo()
            : CurMBSize(0)
        {
            Algo.status = CUDNN_STATUS_NOT_INITIALIZED;
            NoWorkspaceAlgo = (CuDnnAlgoT)-1;
        }
        // Current mini-batch size, needed for re-computing statistics in auto-tuner.
        size_t CurMBSize;
        T Algo;
        CuDnnAlgoT NoWorkspaceAlgo;

        bool NeedAutotuning(size_t batchSize)
        {
            // Need to re-run auto-tuner in case minibatch size is increased.
            // If minibatch size is decreased we assume that previously selected algorithm requires less or the same amount of workspace.
            // This is done to avoid re-running auto-tuner every time in case minibatch size changes frequently (e.g. when distributed reading is enabled).
            // REVIEW alexeyk: potentially, this might cause some perf issues if better (faster) algo can be selected for a smaller mininbatch.
            // We also need to reset auto-tuning status at the beginning of each epoch but ComputationNode currently does not provide such notification.
            // We assume no other dimensions of tensors can change so we don't check it.
            // REVIEW alexeyk: review once we get response from NVIDIA.
            return (Algo.status != CUDNN_STATUS_SUCCESS || batchSize > CurMBSize);
        }
    };

    CuDnn::ptr_t m_cudnn;
    cudnnDataType_t m_dataType;
    CuDnnTensor m_inT;
    CuDnnTensor m_outT;
    // Convolution specific.
    std::unique_ptr<CuDnnFilter> m_filterT;
    std::unique_ptr<CuDnnConv> m_conv;
    // Pooling specific.
    std::unique_ptr<CuDnnPool> m_pool;

    ConvAlgoInfo<cudnnConvolutionFwdAlgoPerf_t> m_fwdAlgo;
    ConvAlgoInfo<cudnnConvolutionBwdDataAlgoPerf_t> m_backDataAlgo;
    ConvAlgoInfo<cudnnConvolutionBwdFilterAlgoPerf_t> m_backFiltAlgo;
};

template <class ElemType>
std::unique_ptr<ConvolutionEngine<ElemType>> CuDnnConvolutionEngineFactory<ElemType>::Create(ConvolveGeometryPtr geometry,
                                                                                             DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout,
                                                                                             size_t maxTempMemSizeInSamples, PoolKind poolKind)
{
    return std::make_unique<CuDnnConvolutionEngine<ElemType>>(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind);
}

template <class ElemType>
bool CuDnnConvolutionEngineFactory<ElemType>::IsSupported(ConvolveGeometryPtr geometry, PoolKind poolKind)
{
    // REVIEW alexeyk: IsSupported check should be performed by cuDNN itself. Is there a good way to do that?

    const auto& input = geometry->InputShape();
    const auto& filter = geometry->KernelShape();
    const auto& sharing = geometry->Sharing();
    const auto& mapCount = geometry->MapCount();
    // cuDNN supports 2D and 3D convolutions at the moment with full sharing.
    // In case map count size > 1, then it should have all ones except last dimension.
    // If pooling is requested, then cuDNN supports only 2D/3D inputs and 2D pooling filters.
    return (input.GetRank() <= 4 &&
            std::find(begin(sharing), end(sharing), false) == sharing.end() &&
            mapCount.GetNumElements() == mapCount[mapCount.GetRank() - 1] &&
            (poolKind == PoolKind::None || 
             input.GetRank() <= 3 && (filter.GetRank() < 3 || filter[2] == 1)));
}

template class CuDnnConvolutionEngineFactory<float>;
template class CuDnnConvolutionEngineFactory<double>;

} } }
