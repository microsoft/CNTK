//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CuDnnConvolutionEngine.h"
#include "GPUMatrix.h"
#ifdef USE_CUDNN
#include <typeinfo>
#include <typeindex>
#include <cudnn.h>

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
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

#ifdef USE_CUDNN

static bool IsGpu(DEVICEID_TYPE deviceId)
{
    return deviceId >= 0;
}

class CuDnnTensor
{
public:
    CuDnnTensor(const TensorShape& src, cudnnDataType_t dataType)
        : m_tensor(nullptr)
    {
        CUDNN_CALL(cudnnCreateTensorDescriptor(&m_tensor));
        // Set cuDNN tensor dimensions. cuDNN uses row-major format while TensorShape - column-major
        // so conversion is required. N dimension will be set to 1.
        const auto& stridesSrc = src.GetStrides();
        SmallVector<int> dims(src.GetRank() + 1);
        SmallVector<int> strides(stridesSrc.size() + 1);
        assert(dims.size() == strides.size());
        for (int i = 0; i < src.GetRank(); i++)
        {
            dims[dims.size() - 1 - i] = (int)src[i];
            strides[dims.size() - 1 - i] = (int)stridesSrc[i];
        }
        // Set "minibatch"(aka N) dimension.
        dims[0] = 1;
        strides[0] = strides[1] * dims[1];
        CUDNN_CALL(cudnnSetTensorNdDescriptor(m_tensor, dataType, (int)dims.size(), dims.data(), strides.data()));
    }

    void UpdateBatchSize(size_t batchSize)
    {
        // Currently cuDNN supports only 2D and 3D convlutions anyway (so max 5D tensors).
        const int MaxDims = 5;
        int dims[MaxDims];
        int strides[MaxDims];
        int nbDims = 0;
        cudnnDataType_t dataType;
        // According to NVIDIA, Get/Set functions are very fast so it's safe to call them in a loop.
        CUDNN_CALL(cudnnGetTensorNdDescriptor(m_tensor, MaxDims, &dataType, &nbDims, dims, strides));
        assert(nbDims <= MaxDims);
        dims[0] = (int)batchSize;
        CUDNN_CALL(cudnnSetTensorNdDescriptor(m_tensor, dataType, nbDims, dims, strides));
    }

    ~CuDnnTensor()
    {
        if (m_tensor != nullptr)
        {
            cudnnDestroyTensorDescriptor(m_tensor);
            m_tensor = nullptr;
        }
    }

    operator cudnnTensorDescriptor_t() const
    {
        return m_tensor;
    }

    DISABLE_COPY_AND_MOVE(CuDnnTensor);

private:
    cudnnTensorDescriptor_t m_tensor;
};

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

template <typename ElemType>
struct Consts
{
    static const ElemType Zero;
    static const ElemType One;
};
template <>
const float Consts<float>::One = 1;
template <>
const double Consts<double>::One = 1;
template <>
const float Consts<float>::Zero = 0;
template <>
const double Consts<double>::Zero = 0;

template <class ElemType>
class CuDnnConvolutionEngine : public ConvolutionEngine<ElemType>
{
public:
    using Base = ConvolutionEngine<ElemType>;
    using typename Base::Mat;

public:
    CuDnnConvolutionEngine(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind)
        : Base(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind),
        m_cudnn(nullptr), m_dataType(GetDataType()), 
        m_inT(geometry->InputShape(), m_dataType), m_outT(geometry->OutputShape(), m_dataType)
    {
        CUDNN_CALL(cudnnCreate(&m_cudnn));
        CUDNN_CALL(cudnnSetStream(m_cudnn, GetStream()));
    }

    ~CuDnnConvolutionEngine()
    {
        if (m_cudnn != nullptr)
        {
            cudnnDestroy(m_cudnn);
            m_cudnn = nullptr;
        }
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
            return cudnnFindConvolutionForwardAlgorithm(m_cudnn, m_inT, *m_filterT, *m_conv, m_outT, MaxAlgoCount, &calgo, algoPerf);
        };
        FindBestAlgo(batchSize, m_fwdAlgo, finder);
        if (m_fwdAlgo.Algo.memory > 0)
            workspace.Resize((m_fwdAlgo.Algo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
        // Perform forward convolution operation.
        auto err = cudnnConvolutionForward(m_cudnn, &C::One, m_inT, ptr(in), *m_filterT, ptr(filter), *m_conv,
                                           m_fwdAlgo.Algo.algo, ptr(workspace), m_fwdAlgo.Algo.memory, &C::Zero, m_outT, ptr(out));
        // There might be a case where cuDNN fails due to workspace being too small, try using no-workspace algo instead.
        // REVIEW alexeyk: NVIDIA is currently reviewing this issue.
        if (CUDNN_STATUS_INVALID_VALUE == err && m_fwdAlgo.Algo.memory > 0)
        {
            auto err2 = cudnnConvolutionForward(m_cudnn, &C::One, m_inT, ptr(in), *m_filterT, ptr(filter), *m_conv,
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
            return cudnnFindConvolutionBackwardDataAlgorithm(m_cudnn, *m_filterT, m_outT, *m_conv, m_inT, MaxAlgoCount, &calgo, algoPerf);
        };
        FindBestAlgo(batchSize, m_backDataAlgo, finder);
        if (m_backDataAlgo.Algo.memory > 0)
            workspace.Resize((m_backDataAlgo.Algo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
        // Compute gradients with respect to the output tensor (data).
        CUDNN_CALL(cudnnConvolutionBackwardData(m_cudnn, &C::One, *m_filterT, ptr(filter), m_outT, ptr(srcGrad), *m_conv, m_backDataAlgo.Algo.algo,
                                                ptr(workspace), m_backDataAlgo.Algo.memory, &C::One, m_inT, ptr(grad)));
    }

    void BackwardFilterCore(const Mat& srcGrad, const Mat& in, Mat& filterGrad, bool /*allowReuse*/, Mat& workspace) override
    {
        size_t batchSize = in.GetNumCols();
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [this](int& calgo, cudnnConvolutionBwdFilterAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
        {
            return cudnnFindConvolutionBackwardFilterAlgorithm(m_cudnn, m_inT, m_outT, *m_conv, *m_filterT, MaxAlgoCount, &calgo, algoPerf);
        };
        FindBestAlgo(batchSize, m_backFiltAlgo, finder);
        if (m_backFiltAlgo.Algo.memory > 0)
            workspace.Resize((m_backFiltAlgo.Algo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
        // Compute gradients with respect to the output tensor (data).
        CUDNN_CALL(cudnnConvolutionBackwardFilter(m_cudnn, &C::One, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, m_backFiltAlgo.Algo.algo,
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
        CUDNN_CALL(cudnnPoolingForward(m_cudnn, *(m_pool), &C::One, m_inT, ptr(in), &C::Zero, m_outT, ptr(out)));
    }

    void BackwardPoolingCore(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad) override
    {
        size_t batchSize = in.GetNumCols();
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);
        CUDNN_CALL(cudnnPoolingBackward(m_cudnn, *(m_pool), &C::One, m_outT, ptr(out), m_outT, ptr(srcGrad),
                                        m_inT, ptr(in), &C::One, m_inT, ptr(grad)));
    }

private:
    using C = Consts<ElemType>;

    static const int MaxAlgoCount = 10;

    static cudnnDataType_t GetDataType()
    {
        if (typeid(ElemType) == typeid(float))
            return CUDNN_DATA_FLOAT;
        else if (typeid(ElemType) == typeid(double))
            return CUDNN_DATA_DOUBLE;
        else
            InvalidArgument("cuDNN engine currently supports only single and double precision data types.");
    }

    template <typename TAlgo, typename TFinder>
    void FindBestAlgo(size_t batchSize, TAlgo& algo, TFinder finder)
    {
        if (!algo.NeedAutotuning(batchSize))
            return;
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);
        using CuDnnAlgoT = decltype(TAlgo::Algo);
        CuDnnAlgoT algoPerf[MaxAlgoCount];
        int calgo = 0;
        CUDNN_CALL(finder(calgo, algoPerf));
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

    // REVIEW alexeyk: is it safe to make m_cudnn static?
    cudnnHandle_t m_cudnn;
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
std::unique_ptr<ConvolutionEngine<ElemType>> CuDnnConvolutionEngineFactory<ElemType>::Create(
    ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind)
{
    return std::make_unique<CuDnnConvolutionEngine<ElemType>>(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind);
}

//class CuDnnTensor4D : public ConvolutionTensor4D
//{
//public:
//    CuDnnTensor4D(size_t w, size_t h, size_t c, size_t n, cudnnDataType_t dataType)
//        : ConvolutionTensor4D(w, h, c, n), m_dataType(dataType), m_tensor(nullptr)
//    {
//        CUDNN_CALL(cudnnCreateTensorDescriptor(&m_tensor));
//        CUDNN_CALL(cudnnSetTensor4dDescriptor(m_tensor, TENSOR_FORMAT, dataType,
//                                              static_cast<int>(n), static_cast<int>(c), static_cast<int>(h), static_cast<int>(w)));
//    }
//
//public:
//    operator cudnnTensorDescriptor_t() const
//    {
//        return m_tensor;
//    }
//
//    ~CuDnnTensor4D() noexcept
//    {
//        if (m_tensor != nullptr)
//        {
//            cudnnDestroyTensorDescriptor(m_tensor);
//            m_tensor = nullptr;
//        }
//    }
//
//    void setN(size_t newN) override
//    {
//        ConvolutionTensor4D::setN(newN);
//        CUDNN_CALL(cudnnSetTensor4dDescriptor(m_tensor, TENSOR_FORMAT, m_dataType,
//                                              static_cast<int>(n()), static_cast<int>(c()), static_cast<int>(h()), static_cast<int>(w())));
//    }
//
//private:
//    cudnnDataType_t m_dataType;
//    cudnnTensorDescriptor_t m_tensor;
//};
//
//class CuDnnFilter : public ConvolutionFilter
//{
//public:
//    CuDnnFilter(size_t w, size_t h, size_t c, size_t k, cudnnDataType_t dataType)
//        : ConvolutionFilter(w, h, c, k), m_filter(nullptr)
//    {
//        CUDNN_CALL(cudnnCreateFilterDescriptor(&m_filter));
//        CUDNN_CALL(cudnnSetFilter4dDescriptor_v4(m_filter, dataType, FILTER_FORMAT,
//                                                 static_cast<int>(k), static_cast<int>(c), static_cast<int>(h), static_cast<int>(w)));
//    }
//
//public:
//    operator cudnnFilterDescriptor_t() const
//    {
//        return m_filter;
//    }
//
//    ~CuDnnFilter() noexcept
//    {
//        if (m_filter != nullptr)
//        {
//            cudnnDestroyFilterDescriptor(m_filter);
//            m_filter = nullptr;
//        }
//    }
//
//private:
//    cudnnFilterDescriptor_t m_filter;
//};
//
//class CuDnnConvolutionDescriptor : public ConvolutionDescriptor
//{
//public:
//    CuDnnConvolutionDescriptor(size_t wStride, size_t hStride, size_t wPad, size_t hPad)
//        : ConvolutionDescriptor(wStride, hStride, wPad > 0 || hPad > 0), m_conv(nullptr)
//    {
//        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&m_conv));
//        CUDNN_CALL(cudnnSetConvolution2dDescriptor(m_conv,
//                                                   static_cast<int>(hPad), static_cast<int>(wPad),
//                                                   static_cast<int>(hStride), static_cast<int>(wStride),
//                                                   1, 1, CUDNN_CROSS_CORRELATION));
//    }
//
//public:
//    operator cudnnConvolutionDescriptor_t() const
//    {
//        return m_conv;
//    }
//
//    ~CuDnnConvolutionDescriptor() noexcept
//    {
//        if (m_conv != nullptr)
//        {
//            cudnnDestroyConvolutionDescriptor(m_conv);
//            m_conv = nullptr;
//        }
//    }
//
//private:
//    cudnnConvolutionDescriptor_t m_conv;
//};
//
//class CuDnnPoolingDescriptor : public PoolingDescriptor
//{
//public:
//    CuDnnPoolingDescriptor(PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad)
//        : PoolingDescriptor(kind, w, h, wStride, hStride, wPad, hPad), m_pool(nullptr)
//    {
//        assert(kind == PoolKind::Max || kind == PoolKind::Average);
//
//        CUDNN_CALL(cudnnCreatePoolingDescriptor(&m_pool));
//        CUDNN_CALL(cudnnSetPooling2dDescriptor(m_pool,
//                                               kind == PoolKind::Max ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
//                                               static_cast<int>(h), static_cast<int>(w),
//                                               static_cast<int>(hPad), static_cast<int>(wPad),
//                                               static_cast<int>(hStride), static_cast<int>(wStride)));
//    }
//
//public:
//    operator cudnnPoolingDescriptor_t() const
//    {
//        return m_pool;
//    }
//
//    ~CuDnnPoolingDescriptor() noexcept
//    {
//        if (m_pool != nullptr)
//        {
//            cudnnDestroyPoolingDescriptor(m_pool);
//            m_pool = nullptr;
//        }
//    }
//
//private:
//    cudnnPoolingDescriptor_t m_pool;
//};
//
//template <typename CuDnnT, typename In>
//static CuDnnT& As(In& src)
//{
//    // Do dynamic_cast only in debug builds and static_cast in release builds.
//    assert(dynamic_cast<CuDnnT*>(&src) != nullptr);
//    return static_cast<CuDnnT&>(src);
//}
//static const CuDnnTensor4D& t(const ConvolutionTensor4D& src)
//{
//    return As<const CuDnnTensor4D>(src);
//}
//static const CuDnnFilter& f(const ConvolutionFilter& src)
//{
//    return As<const CuDnnFilter>(src);
//}
//static const CuDnnConvolutionDescriptor& cd(const ConvolutionDescriptor& src)
//{
//    return As<const CuDnnConvolutionDescriptor>(src);
//}
//static const CuDnnPoolingDescriptor& p(const PoolingDescriptor& src)
//{
//    return As<const CuDnnPoolingDescriptor>(src);
//}
//template <typename ElemType>
//static ElemType* ptr(Matrix<ElemType>& src)
//{
//    return src.BufferPointer();
//}
//template <typename ElemType>
//static const ElemType* ptr(const Matrix<ElemType>& src)
//{
//    return src.BufferPointer();
//}
//
//template <typename ElemType>
//struct Consts
//{
//    static const ElemType Zero;
//    static const ElemType One;
//};
//template <>
//const float Consts<float>::One = 1;
//template <>
//const double Consts<double>::One = 1;
//template <>
//const float Consts<float>::Zero = 0;
//template <>
//const double Consts<double>::Zero = 0;
//
//template <typename ElemType>
//class CuDnnConvolutionEngine : public ConvolutionEngine<ElemType>
//{
//public:
//    using Base = ConvolutionEngine<ElemType>;
//    using typename Base::Mat;
//    using typename Base::Tensor4D;
//    using typename Base::Filter;
//    using typename Base::ConvDesc;
//
//    CuDnnConvolutionEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, BatchNormImpl bnImpl)
//        : Base(deviceId, imageLayout), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples), m_bnImpl(bnImpl), m_stream(GetStream()), m_cudnn(nullptr)
//    {
//        CUDNN_CALL(cudnnCreate(&m_cudnn));
//        CUDNN_CALL(cudnnSetStream(m_cudnn, m_stream));
//    }
//
//    ~CuDnnConvolutionEngine()
//    {
//        if (m_cudnn != nullptr)
//        {
//            cudnnDestroy(m_cudnn);
//            m_cudnn = nullptr;
//        }
//    }
//
//protected:
//    using Base::m_deviceId;
//    using Base::m_imageLayout;
//
//    void EnsureCompatible() override
//    {
//        if (m_imageLayout != ImageLayoutKind::CHW)
//            RuntimeError("cuDNN convolution engine supports only CHW/cudnn layout.");
//        if (!IsGpu(m_deviceId))
//            RuntimeError("cuDNN convolution engine supports GPU devices only.");
//    }
//
//    void ForwardCore(const Tensor4D& inT, const Mat& in, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
//                     const Tensor4D& outT, Mat& out, Mat& workspace) override
//    {
//        // Find best algo and allocate temp buffer, if needed.
//        auto finder = [&](int& calgo, cudnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
//        {
//            return cudnnFindConvolutionForwardAlgorithm(m_cudnn, t(inT), f(filterT), cd(convDesc), t(outT), MaxAlgoCount, &calgo, algoPerf);
//        };
//        FindBestAlgo(t(inT), m_fwdAlgo, finder);
//        if (m_fwdAlgo.Algo.memory > 0)
//            workspace.Resize((m_fwdAlgo.Algo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
//        // Perform forward convolution operation.
//        auto err = cudnnConvolutionForward(m_cudnn, &C::One, t(inT), ptr(in), f(filterT), ptr(filter), cd(convDesc),
//                                           m_fwdAlgo.Algo.algo, ptr(workspace), m_fwdAlgo.Algo.memory, &C::Zero, t(outT), ptr(out));
//        // There might be a case where cuDNN fails due to workspace being too small, try using no-workspace algo instead.
//        // REVIEW alexeyk: NVIDIA is currently reviewing this issue.
//        if (CUDNN_STATUS_INVALID_VALUE == err && m_fwdAlgo.Algo.memory > 0)
//        {
//            auto err2 = cudnnConvolutionForward(m_cudnn, &C::One, t(inT), ptr(in), f(filterT), ptr(filter), cd(convDesc),
//                                                m_fwdAlgo.NoWorkspaceAlgo, nullptr, 0, &C::Zero, t(outT), ptr(out));
//            // Update original error in case of success.
//            if (CUDNN_STATUS_SUCCESS == err2)
//                err = CUDNN_STATUS_SUCCESS;
//        }
//        CUDNN_CALL(err);
//    }
//
//    void BackwardDataCore(const Tensor4D& srcGradT, const Mat& srcGrad, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
//                          const Tensor4D& gradT, Mat& grad, Mat& workspace) override
//    {
//        // Find best algo and allocate temp buffer, if needed.
//        auto finder = [&](int& calgo, cudnnConvolutionBwdDataAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
//        {
//            return cudnnFindConvolutionBackwardDataAlgorithm(m_cudnn, f(filterT), t(srcGradT), cd(convDesc), t(gradT), MaxAlgoCount, &calgo, algoPerf);
//        };
//        FindBestAlgo(t(srcGradT), m_backDataAlgo, finder);
//        if (m_backDataAlgo.Algo.memory > 0)
//            workspace.Resize((m_backDataAlgo.Algo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
//        // Compute gradients with respect to the output tensor (data).
//        CUDNN_CALL(cudnnConvolutionBackwardData(m_cudnn, &C::One, f(filterT), ptr(filter), t(srcGradT), ptr(srcGrad), cd(convDesc), m_backDataAlgo.Algo.algo,
//                                                ptr(workspace), m_backDataAlgo.Algo.memory, &C::One, t(gradT), ptr(grad)));
//    }
//
//    void BackwardFilterCore(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& inT, const Mat& in, const ConvDesc& convDesc,
//                            const Filter& filterT, Mat& filter, bool /*allowReuse*/, Mat& workspace) override
//    {
//        // Find best algo and allocate temp buffer, if needed.
//        auto finder = [&](int& calgo, cudnnConvolutionBwdFilterAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
//        {
//            return cudnnFindConvolutionBackwardFilterAlgorithm(m_cudnn, t(inT), t(srcGradT), cd(convDesc), f(filterT), MaxAlgoCount, &calgo, algoPerf);
//        };
//        FindBestAlgo(t(inT), m_backFiltAlgo, finder);
//        if (m_backFiltAlgo.Algo.memory > 0)
//            workspace.Resize((m_backFiltAlgo.Algo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
//        // Compute gradients with respect to the output tensor (data).
//        CUDNN_CALL(cudnnConvolutionBackwardFilter(m_cudnn, &C::One, t(inT), ptr(in), t(srcGradT), ptr(srcGrad), cd(convDesc), m_backFiltAlgo.Algo.algo,
//                                                  ptr(workspace), m_backFiltAlgo.Algo.memory, &C::One, f(filterT), ptr(filter)));
//    }
//
//    void EnsureCompatibleBatchNorm(bool spatial) override
//    {
//        if (!IsGpu(m_deviceId))
//            InvalidArgument("cuDNN engine does not support batch normalization on CPUs.");
//        if (spatial && m_imageLayout != ImageLayoutKind::CHW)
//            InvalidArgument("cuDNN engine batch normalization currently supports only CHW data layout for convolutional nodes.");
//    }
//
//    void NormalizeBatchCore(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
//                            bool spatial, double expAvgFactor, Mat& runMean, Mat& runInvStdDev, Mat& out,
//                            double epsilon, Mat& saveMean, Mat& saveInvStdDev) override
//    {
//        if (m_bnImpl == BatchNormImpl::CuDnn)
//        {
//            cudnnBatchNormMode_t mode = spatial ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
//            // cuDNN will fail with BAD_PARAM if epsilon < CUDNN_BN_MIN_EPSILON.
//            epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);
//            CUDNN_CALL(cudnnBatchNormalizationForwardTraining(m_cudnn, mode, &C::One, &C::Zero, t(inT), ptr(in), t(inT), ptr(out),
//                t(scaleBiasT), ptr(scale), ptr(bias), expAvgFactor, ptr(runMean), ptr(runInvStdDev), 
//                epsilon, ptr(saveMean), ptr(saveInvStdDev)));
//        }
//        else if (m_bnImpl == BatchNormImpl::Cntk)
//        {
//            epsilon = std::max(epsilon, 1e-9);
//            CUDA_CALL(BatchNormalizationForwardTraining(inT, spatial, ptr(in), ptr(out), ptr(scale), ptr(bias),
//                                                        expAvgFactor, ptr(runMean), ptr(runInvStdDev),
//                                                        epsilon, ptr(saveMean), ptr(saveInvStdDev), m_stream));
//        }
//        else
//            RuntimeError("Provided batch norm implementation (%d) is not supported.", m_bnImpl);
//    }
//
//    void NormalizeBatchInferenceCore(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
//                                     bool spatial, const Mat& runMean, const Mat& runInvStdDev, Mat& out) override
//    {
//        if (m_bnImpl == BatchNormImpl::CuDnn)
//        {
//            cudnnBatchNormMode_t mode = spatial ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
//            CUDNN_CALL(cudnnBatchNormalizationForwardInference(m_cudnn, mode, &C::One, &C::Zero, t(inT), ptr(in), t(inT), ptr(out),
//                                                               t(scaleBiasT), ptr(scale), ptr(bias), ptr(runMean), ptr(runInvStdDev), CUDNN_BN_MIN_EPSILON));
//        }
//        else if (m_bnImpl == BatchNormImpl::Cntk)
//        {
//            CUDA_CALL(BatchNormalizationForwardInference(inT, spatial, ptr(in), ptr(out), ptr(scale), ptr(bias),
//                                                         ptr(runMean), ptr(runInvStdDev), m_stream));
//        }
//        else
//            RuntimeError("Provided batch norm implementation (%d) is not supported.", m_bnImpl);
//    }
//
//    void BackwardNormalizeBatchCore(const Tensor4D& inT, const Mat& in, const Mat& srcGrad, Mat& grad,
//                                    const Tensor4D& scaleBiasT, const Mat& scale, bool spatial, const Mat& saveMean, const Mat& saveInvStdDev,
//                                    Mat& scaleGrad, Mat& biasGrad) override
//    {
//        if (m_bnImpl == BatchNormImpl::CuDnn)
//        {
//            cudnnBatchNormMode_t mode = spatial ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
//// REVIEW alexeyk: remove once Philly is upgraded to prod version.
//#if CUDNN_PATCHLEVEL >= 7
//            CUDNN_CALL(cudnnBatchNormalizationBackward(m_cudnn, mode, &C::One, &C::One, &C::One, &C::One, t(inT), ptr(in), t(inT), ptr(srcGrad), t(inT), ptr(grad),
//                                                       t(scaleBiasT), ptr(scale), ptr(scaleGrad), ptr(biasGrad), CUDNN_BN_MIN_EPSILON, ptr(saveMean), ptr(saveInvStdDev)));
//#else
//            CUDNN_CALL(cudnnBatchNormalizationBackward(m_cudnn, mode, &C::One, &C::One, t(inT), ptr(in), t(inT), ptr(srcGrad), t(inT), ptr(grad),
//                t(scaleBiasT), ptr(scale), ptr(scaleGrad), ptr(biasGrad), CUDNN_BN_MIN_EPSILON, ptr(saveMean), ptr(saveInvStdDev)));
//#endif
//
//        }
//        else if (m_bnImpl == BatchNormImpl::Cntk)
//        {
//            CUDA_CALL(BatchNormalizationBackward(inT, spatial, ptr(in), ptr(srcGrad), ptr(grad), ptr(scale), ptr(scaleGrad), ptr(biasGrad),
//                                                 ptr(saveMean), ptr(saveInvStdDev), m_stream));
//        }
//        else
//            RuntimeError("Provided batch norm implementation (%d) is not supported.", m_bnImpl);
//    }
//
//private:
//    static const int MaxAlgoCount = 10;
//
//    template <typename TAlgo, typename TFinder>
//    void FindBestAlgo(const CuDnnTensor4D& t, TAlgo& algo, TFinder finder)
//    {
//        if (!algo.NeedAutotuning(t))
//            return;
//        using CuDnnAlgoT = decltype(TAlgo::Algo);
//        CuDnnAlgoT algoPerf[MaxAlgoCount];
//        int calgo = 0;
//        CUDNN_CALL(finder(calgo, algoPerf));
//        assert(calgo > 0);
//        size_t maxMem = m_maxTempMemSizeInSamples == 0 ? (std::numeric_limits<size_t>::max)() : t.w() * t.h() * t.c() * m_maxTempMemSizeInSamples * sizeof(ElemType);
//        auto res = std::find_if(algoPerf, algoPerf + calgo,
//            [=](const CuDnnAlgoT& cur)
//            {
//                return cur.status == CUDNN_STATUS_SUCCESS && cur.memory <= maxMem;
//            });
//        if (res == algoPerf + calgo)
//            RuntimeError("cuDNN could not find suitable algorithm for the current convolution configuration.");
//        algo.CurMBSize = t.n();
//        algo.Algo = *res;
//        res = std::find_if(algoPerf, algoPerf + calgo,
//            [](const CuDnnAlgoT& cur)
//            {
//                return cur.status == CUDNN_STATUS_SUCCESS && cur.memory == 0;
//            });
//        if (res == algoPerf + calgo)
//        {
//            // In theory, this should never happen.
//            RuntimeError("cuDNN could not find no-workspace algorithm for the current convolution configuration.");
//        }
//        else
//            algo.NoWorkspaceAlgo = (*res).algo;
//    }
//
//private:
//    template <typename T>
//    struct ConvAlgoInfo
//    {
//        using CuDnnAlgoT = decltype(T::algo);
//
//        ConvAlgoInfo()
//            : CurMBSize(0)
//        {
//            Algo.status = CUDNN_STATUS_NOT_INITIALIZED;
//            NoWorkspaceAlgo = (CuDnnAlgoT)-1;
//        }
//        // Current mini-batch size, needed for re-computing statistics in auto-tuner.
//        size_t CurMBSize;
//        T Algo;
//        CuDnnAlgoT NoWorkspaceAlgo;
//
//        bool NeedAutotuning(const CuDnnTensor4D& t)
//        {
//            // Need to re-run auto-tuner in case minibatch size is increased.
//            // If minibatch size is decreased we assume that previously selected algorithm requires less or the same amount of workspace.
//            // This is done to avoid re-running auto-tuner every time in case minibatch size changes frequently (e.g. when distributed reading is enabled).
//            // REVIEW alexeyk: potentially, this might cause some perf issues if better (faster) algo can be selected for a smaller mininbatch.
//            // We also need to reset auto-tuning status at the beginning of each epoch but ComputationNode currently does not provide such notification.
//            // We assume no other dimensions of tensors can change so we don't check it.
//            // REVIEW alexeyk: review once we get response from NVIDIA.
//            return (Algo.status != CUDNN_STATUS_SUCCESS || t.n() > CurMBSize);
//        }
//    };
//
//    using C = Consts<ElemType>;
//
//    // REVIEW alexeyk: currently limit is set once in ctor though in CNTK it can be, theoretically, changed in runtime.
//    size_t m_maxTempMemSizeInSamples;
//    BatchNormImpl m_bnImpl;
//    cudnnHandle_t m_cudnn;
//    cudaStream_t m_stream;
//    ConvAlgoInfo<cudnnConvolutionFwdAlgoPerf_t> m_fwdAlgo;
//    ConvAlgoInfo<cudnnConvolutionBwdDataAlgoPerf_t> m_backDataAlgo;
//    ConvAlgoInfo<cudnnConvolutionBwdFilterAlgoPerf_t> m_backFiltAlgo;
//};
//
//template <class ElemType>
//class CuDnnPoolingEngine : public PoolingEngine<ElemType>
//{
//public:
//    using Base = PoolingEngine<ElemType>;
//    using typename Base::Tensor4D;
//    using typename Base::PoolDesc;
//    using typename Base::Mat;
//
//public:
//    CuDnnPoolingEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout)
//        : Base(deviceId, imageLayout), m_cudnn(nullptr)
//    {
//        CUDNN_CALL(cudnnCreate(&m_cudnn));
//        CUDNN_CALL(cudnnSetStream(m_cudnn, GetStream()));
//    }
//
//    ~CuDnnPoolingEngine()
//    {
//        if (m_cudnn != nullptr)
//        {
//            cudnnDestroy(m_cudnn);
//            m_cudnn = nullptr;
//        }
//    }
//
//protected:
//    using Base::m_deviceId;
//    using Base::m_imageLayout;
//
//    void EnsureCompatible() override
//    {
//        if (m_imageLayout != ImageLayoutKind::CHW)
//            RuntimeError("cuDNN pooling engine supports only CHW/cudnn layout.");
//        if (!IsGpu(m_deviceId))
//            RuntimeError("cuDNN pooling engine supports GPU devices only.");
//    }
//
//    void ForwardCore(const Tensor4D& inT, const Mat& in, const PoolDesc& poolDesc, const Tensor4D& outT, Mat& out) override
//    {
//        CUDNN_CALL(cudnnPoolingForward(m_cudnn, p(poolDesc), &C::One, t(inT), ptr(in), &C::Zero, t(outT), ptr(out)));
//    }
//
//    void BackwardCore(const Tensor4D& outT, const Mat& out, const Mat& srcGrad, const PoolDesc& poolDesc, const Tensor4D& inT, const Mat& in, Mat& grad) override
//    {
//        CUDNN_CALL(cudnnPoolingBackward(m_cudnn, p(poolDesc), &C::One, t(outT), ptr(out), t(outT), ptr(srcGrad),
//                                        t(inT), ptr(in), &C::One, t(inT), ptr(grad)));
//    }
//
//private:
//    using C = Consts<ElemType>;
//
//    cudnnHandle_t m_cudnn;
//};
//
//template <class ElemType>
//typename CuDnnConvolutionEngineFactory<ElemType>::Tensor4DPtr CuDnnConvolutionEngineFactory<ElemType>::CreateTensor(size_t w, size_t h, size_t c, size_t n)
//{
//    // REVIEW alexeyk: assert fires in GCC but not in VC++.
//    // static_assert(false, "cuDNN engine currently supports only single and double precision tensors.");
//    RuntimeError("Not implemented.");
//}
//template <>
//typename CuDnnConvolutionEngineFactory<float>::Tensor4DPtr CuDnnConvolutionEngineFactory<float>::CreateTensor(size_t w, size_t h, size_t c, size_t n)
//{
//    return std::make_unique<CuDnnTensor4D>(w, h, c, n, CUDNN_DATA_FLOAT);
//}
//template <>
//typename CuDnnConvolutionEngineFactory<double>::Tensor4DPtr CuDnnConvolutionEngineFactory<double>::CreateTensor(size_t w, size_t h, size_t c, size_t n)
//{
//    return std::make_unique<CuDnnTensor4D>(w, h, c, n, CUDNN_DATA_DOUBLE);
//}
//
//template <class ElemType>
//typename CuDnnConvolutionEngineFactory<ElemType>::FilterPtr CuDnnConvolutionEngineFactory<ElemType>::CreateFilter(size_t w, size_t h, size_t c, size_t k)
//{
//    // REVIEW alexeyk: assert fires in GCC but not in VC++.
//    // static_assert(false, "cuDNN engine currently supports only single and double precision filters.");
//    RuntimeError("Not implemented.");
//}
//template <>
//typename CuDnnConvolutionEngineFactory<float>::FilterPtr CuDnnConvolutionEngineFactory<float>::CreateFilter(size_t w, size_t h, size_t c, size_t k)
//{
//    return std::make_unique<CuDnnFilter>(w, h, c, k, CUDNN_DATA_FLOAT);
//}
//template <>
//typename CuDnnConvolutionEngineFactory<double>::FilterPtr CuDnnConvolutionEngineFactory<double>::CreateFilter(size_t w, size_t h, size_t c, size_t k)
//{
//    return std::make_unique<CuDnnFilter>(w, h, c, k, CUDNN_DATA_DOUBLE);
//}
//
//template <class ElemType>
//typename CuDnnConvolutionEngineFactory<ElemType>::ConvDescPtr CuDnnConvolutionEngineFactory<ElemType>::CreateConvDescriptor(
//    const Tensor4D& /*inT*/, const Filter& filterT, size_t wStride, size_t hStride, bool padding)
//{
//    size_t wPad = padding ? filterT.w() / 2 : 0;
//    size_t hPad = padding ? filterT.h() / 2 : 0;
//    return std::make_unique<CuDnnConvolutionDescriptor>(wStride, hStride, wPad, hPad);
//}
//
//template <class ElemType>
//typename CuDnnConvolutionEngineFactory<ElemType>::PoolDescPtr CuDnnConvolutionEngineFactory<ElemType>::CreatePoolDescriptor(
//    typename PoolDesc::PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad)
//{
//    return std::make_unique<CuDnnPoolingDescriptor>(kind, w, h, wStride, hStride, wPad, hPad);
//}
//
//template <class ElemType>
//typename CuDnnConvolutionEngineFactory<ElemType>::ConvEnginePtr CuDnnConvolutionEngineFactory<ElemType>::CreateConvEngine(
//    DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, BatchNormImpl bnImpl)
//{
//    return std::make_unique<CuDnnConvolutionEngine<ElemType>>(deviceId, imageLayout, maxTempMemSizeInSamples, bnImpl);
//}
//
//template <class ElemType>
//typename CuDnnConvolutionEngineFactory<ElemType>::PoolEnginePtr CuDnnConvolutionEngineFactory<ElemType>::CreatePoolEngine(
//    DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout)
//{
//    return std::make_unique<CuDnnPoolingEngine<ElemType>>(deviceId, imageLayout);
//}
//
#else

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::Tensor4DPtr CuDnnConvolutionEngineFactory<ElemType>::CreateTensor(size_t, size_t, size_t, size_t)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::FilterPtr CuDnnConvolutionEngineFactory<ElemType>::CreateFilter(size_t, size_t, size_t, size_t)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::ConvDescPtr CuDnnConvolutionEngineFactory<ElemType>::CreateConvDescriptor(
    const Tensor4D&, const Filter&, size_t, size_t, bool)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::PoolDescPtr CuDnnConvolutionEngineFactory<ElemType>::CreatePoolDescriptor(
    typename PoolDesc::PoolKind, size_t, size_t, size_t, size_t, size_t, size_t)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::ConvEnginePtr CuDnnConvolutionEngineFactory<ElemType>::CreateConvEngine(DEVICEID_TYPE, ImageLayoutKind, size_t, BatchNormImpl)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

template <class ElemType>
typename CuDnnConvolutionEngineFactory<ElemType>::PoolEnginePtr CuDnnConvolutionEngineFactory<ElemType>::CreatePoolEngine(DEVICEID_TYPE, ImageLayoutKind)
{
    RuntimeError("The code is compiled without USE_CUDNN macro.");
}

#endif

// REVIEW alexeyk: remove #ifdef once cuDNN becomes mandatory dependency.
template <class ElemType>
bool CuDnnConvolutionEngineFactory<ElemType>::IsSupported(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, PoolKind poolKind)
{
#ifdef USE_CUDNN
    cudaDeviceProp props = {0};
    if (cudaGetDeviceProperties(&props, deviceId) != cudaSuccess || props.major < 3)
        return false;

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
#else
    UNUSED(deviceId);
    return false;
#endif
}

template class CuDnnConvolutionEngineFactory<float>;
template class CuDnnConvolutionEngineFactory<double>;

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
