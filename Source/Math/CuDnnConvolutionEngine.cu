//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

// A note on the formats: CNTK originally used NHWC for input/output tensors and CHWN for kernels.
// Such formats have very limited support in cuDNN and not used in other frameworks.
// CNTK with cuDNN by default uses NCHW formats for both inputs/outputs and kernels.
#define TENSOR_FORMAT CUDNN_TENSOR_NCHW
#define FILTER_FORMAT CUDNN_TENSOR_NCHW

namespace Microsoft { namespace MSR { namespace CNTK {

static bool IsGpu(DEVICEID_TYPE deviceId)
{
    return deviceId >= 0;
}

class CuDnnKernel
{
public:
    CuDnnKernel(const ConvolveGeometry& geometry, cudnnDataType_t dataType)
        : m_kernel(nullptr)
    {
        CUDNN_CALL(cudnnCreateFilterDescriptor(&m_kernel));
        // Set cuDNN kernel dimensions. cuDNN uses row-major format while TensorShape - column-major
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
        CUDNN_CALL(cudnnSetFilterNdDescriptor_v4(m_kernel, dataType, FILTER_FORMAT, (int)dims.size(), dims.data()));
    }

    ~CuDnnKernel()
    {
        if (m_kernel != nullptr)
        {
            cudnnDestroyFilterDescriptor(m_kernel);
            m_kernel = nullptr;
        }
    }

    operator cudnnFilterDescriptor_t() const
    {
        return m_kernel;
    }

    DISABLE_COPY_AND_MOVE(CuDnnKernel);

private:
    cudnnFilterDescriptor_t m_kernel;
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
    CuDnnPool(const ConvolveGeometry& geometry, PoolKind kind, bool forceDeterministicAlgorithms, bool poolIncludePad)
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
        cudnnPoolingMode_t poolMode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        if (poolIncludePad)
            poolMode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        // Must use CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING to get the same results as in reference engine.
        CUDNN_CALL(cudnnSetPoolingNdDescriptor(m_pool,
                                               kind == PoolKind::Max && !forceDeterministicAlgorithms ? CUDNN_POOLING_MAX : poolMode,
                                               CUDNN_PROPAGATE_NAN,
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

enum class AutotuningState : int
{
    Init = 0,          // initial state
    PendingTuning = 1, // memory of all nodes have been allocated, it's safe to do tuning now
    Running = 2        // done tuning, no long performing auto-tuning, code is running normally
};

template <class ElemType>
class CuDnnConvolutionEngine : public ConvolutionEngine<ElemType>
{
public:
    using Base = ConvolutionEngine<ElemType>;
    using typename Base::Mat;

public:
    CuDnnConvolutionEngine(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout,
                           size_t maxTempMemSizeInSamples, PoolKind poolKind, bool forceDeterministicAlgorithms, bool poolIncludePad)
        : Base(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, poolIncludePad),
          m_cudnn(CuDnn::Instance()),
          m_dataType(CuDnnTensor::GetDataType<ElemType>()),
          m_inT(geometry->InputShape(), m_dataType),
          m_outT(geometry->OutputShape(), m_dataType),
          m_forceDeterministicAlgorithms(forceDeterministicAlgorithms)
    {
    }

    virtual bool ImplementsGradientOverwriteOptimization() const override { return true; }

protected:
    using Base::m_geometry;
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_maxTempMemSizeInSamples;
    using Base::m_poolKind;
    using Base::m_poolIncludePad;

    void EnsureCompatible() override
    {
        if (m_imageLayout != ImageLayoutKind::CHW)
            RuntimeError("cuDNN convolution engine supports only CHW/cudnn layout.");
        if (!IsGpu(m_deviceId))
            RuntimeError("cuDNN convolution engine supports GPU devices only.");
    }

    void EnsureConvolutionInitialized() override
    {
        if (m_kernelT == nullptr)
        {
            m_kernelT = std::make_unique<CuDnnKernel>(*m_geometry, m_dataType);
            m_conv = std::make_unique<CuDnnConv>(*m_geometry, m_dataType);
        }
    }

    void ForwardCore(const Mat& in, const Mat& kernel, Mat& out, Mat& workspace) override
    {
        size_t batchSize = in.GetNumCols();
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [&,this](int& calgo, cudnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
        {
            auto result = cudnnFindConvolutionForwardAlgorithmEx(*m_cudnn, m_inT, ptr(in), *m_kernelT, ptr(kernel), *m_conv, m_outT, ptr(out), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
            if (m_forceDeterministicAlgorithms)
            {
                auto found = std::find_if(algoPerf, algoPerf + calgo,
                                          [](const cudnnConvolutionFwdAlgoPerf_t& a) { return a.algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM && a.status == CUDNN_STATUS_SUCCESS; });
                if (found == algoPerf + calgo)
                    RuntimeError("cuDNN could not find a deterministic algorithm. Set 'forceDeterministicAlgorithms=false' in your configuration.");
                calgo = 1;
                algoPerf[0] = *found;
            }
            return result;
        };
        // Find max Memory needed while running static finder. Workaround for cudnnFind fail. Number of algo is constant as in cudnn 5.1
        auto staticFinder = [&,this](cudnnConvolutionFwdAlgo_t& algo, bool noMem) -> cudnnStatus_t
        {
            if(!noMem)
                return cudnnGetConvolutionForwardAlgorithm(*m_cudnn, m_inT, *m_kernelT, *m_conv, m_outT, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, workspace.BufferSize(), &algo);
            size_t tmpSize;
            for(int i = 0; i < 8; i++)  // hard coded 8 algorithms in cuDNN cudnnConvolutionFwdAlgo_t. Wish there is a COUNT in cudnnConvolutionFwdAlgo_t.
            {
                cudnnStatus_t err = cudnnGetConvolutionForwardWorkspaceSize(*m_cudnn, m_inT, *m_kernelT, *m_conv, m_outT, (cudnnConvolutionFwdAlgo_t)i, &tmpSize);
                if(err == CUDNN_STATUS_SUCCESS && m_fwdAlgo.AlgoWorkspaceSize < tmpSize)
                    m_fwdAlgo.AlgoWorkspaceSize = tmpSize;
            }
            return cudnnGetConvolutionForwardAlgorithm(*m_cudnn, m_inT, *m_kernelT, *m_conv, m_outT, CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, &algo);
        };
        FindBestAlgo(batchSize, m_fwdAlgo, finder, staticFinder, workspace);
        // Perform forward convolution operation.
        CUDNN_CALL(cudnnConvolutionForward(*m_cudnn, &C::One, m_inT, ptr(in), *m_kernelT, ptr(kernel), *m_conv, m_fwdAlgo.selectedAlgo, ptr(workspace), workspace.BufferSize(), &C::Zero, m_outT, ptr(out)));
    }

    void BackwardDataCore(const Mat& srcGrad, const Mat& kernel, Mat& grad, bool accumulateGradient, Mat& workspace) override
    {
        size_t batchSize = srcGrad.GetNumCols();
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [&,this](int& calgo, cudnnConvolutionBwdDataAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
        {
            cudnnStatus_t result;
            if (accumulateGradient)
            {
                // cudnnFindConvolutionBackwardDataAlgorithmEx will overwrite the output buffer, thus we create a temporary buffer here
                // note this memory allocation might fail, so use try...catch for safety 
                auto gradReplace = Matrix<ElemType>((grad.BufferSize() + sizeof(ElemType) - 1)/sizeof(ElemType), 1, m_deviceId);
                result = cudnnFindConvolutionBackwardDataAlgorithmEx(*m_cudnn, *m_kernelT, ptr(kernel), m_outT, ptr(srcGrad), *m_conv, m_inT, ptr(gradReplace), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
                gradReplace.ReleaseMemory();
            }
            else
                result = cudnnFindConvolutionBackwardDataAlgorithmEx(*m_cudnn, *m_kernelT, ptr(kernel), m_outT, ptr(srcGrad), *m_conv, m_inT, ptr(grad), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
            if (m_forceDeterministicAlgorithms)
            {
                auto found = std::find_if(algoPerf, algoPerf + calgo,
                                          [](const cudnnConvolutionBwdDataAlgoPerf_t& a)  { return a.algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 && a.status == CUDNN_STATUS_SUCCESS; });
                if (found == algoPerf + calgo)
                    RuntimeError("cuDNN could not find a deterministic algorithm. Set 'forceDeterministicAlgorithms=false' in your configuration.");
                calgo = 1;
                algoPerf[0] = *found;
            }
            return result;
        };
        // Find max Memory needed while running static finder. Workaround for cudnnFind fail. Number of algo is constant as in cudnn 5.1
        auto staticFinder = [&,this](cudnnConvolutionBwdDataAlgo_t& algo, bool noMem) -> cudnnStatus_t
        {
            if(!noMem)
                return cudnnGetConvolutionBackwardDataAlgorithm(*m_cudnn, *m_kernelT, m_outT, *m_conv, m_inT, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, workspace.BufferSize(), &algo);
            size_t tmpSize;
            for(int i = 0; i < 6; i++)
            {
                cudnnStatus_t err = cudnnGetConvolutionBackwardDataWorkspaceSize(*m_cudnn, *m_kernelT, m_outT, *m_conv, m_inT, (cudnnConvolutionBwdDataAlgo_t)i, &tmpSize);
                if(err == CUDNN_STATUS_SUCCESS && m_backDataAlgo.AlgoWorkspaceSize < tmpSize)
                    m_backDataAlgo.AlgoWorkspaceSize = tmpSize;
            }
            return cudnnGetConvolutionBackwardDataAlgorithm(*m_cudnn, *m_kernelT, m_outT, *m_conv, m_inT, CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, 0, &algo);
        };
        FindBestAlgo(batchSize, m_backDataAlgo, finder, staticFinder, workspace);
        // Compute gradients with respect to the output tensor (data).
        CUDNN_CALL(cudnnConvolutionBackwardData(*m_cudnn, &C::One, *m_kernelT, ptr(kernel), m_outT, ptr(srcGrad), *m_conv, m_backDataAlgo.selectedAlgo, ptr(workspace), workspace.BufferSize(), accumulateGradient ? &C::One : &C::Zero, m_inT, ptr(grad)));
    }

    void BackwardKernelCore(const Mat& srcGrad, const Mat& in, Mat& kernelGrad, bool accumulateGradient, bool /*allowReuse*/, Mat& workspace) override
    {
        size_t batchSize = in.GetNumCols();
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [&,this](int& calgo, cudnnConvolutionBwdFilterAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
        {
            cudnnStatus_t result;
            if (accumulateGradient)
            {
                // cudnnFindConvolutionBackwardFilterAlgorithmEx will overwrite the output buffer, thus we create a temporary buffer here
                // note this memory allocation might fail, so use try...catch for safety 
                auto kernelGradReplace = Matrix<ElemType>((kernelGrad.BufferSize() + sizeof(ElemType) - 1)/sizeof(ElemType), 1, m_deviceId);
                result = cudnnFindConvolutionBackwardFilterAlgorithmEx(*m_cudnn, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, *m_kernelT, ptr(kernelGradReplace), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
                kernelGradReplace.ReleaseMemory();
            }
            else
                result = cudnnFindConvolutionBackwardFilterAlgorithmEx(*m_cudnn, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, *m_kernelT, ptr(kernelGrad), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
            if (m_forceDeterministicAlgorithms)
            {
                auto found = std::find_if(algoPerf, algoPerf + calgo,
                                          [](const cudnnConvolutionBwdFilterAlgoPerf_t& a)  { return a.algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 && a.status == CUDNN_STATUS_SUCCESS; });
                if (found == algoPerf + calgo)
                    RuntimeError("cuDNN could not find a deterministic algorithm. Set 'forceDeterministicAlgorithms=false' in your configuration.");
                calgo = 1;
                algoPerf[0] = *found;
            }
            return result;
        };
        // Find max Memory needed while running static finder. Workaround for cudnnFind fail. Number of algo is constant as in cudnn 5.1
        auto staticFinder = [&,this](cudnnConvolutionBwdFilterAlgo_t& algo, bool noMem) -> cudnnStatus_t
        {
            if(!noMem)
                return cudnnGetConvolutionBackwardFilterAlgorithm(*m_cudnn, m_inT, m_outT, *m_conv, *m_kernelT, CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, workspace.BufferSize(), &algo);
            size_t tmpSize;
            for(int i = 0; i < 5; i++)
            {
                cudnnStatus_t err = cudnnGetConvolutionBackwardFilterWorkspaceSize(*m_cudnn, m_inT, m_outT, *m_conv, *m_kernelT, (cudnnConvolutionBwdFilterAlgo_t)i, &tmpSize);
                if(err == CUDNN_STATUS_SUCCESS && m_backFiltAlgo.AlgoWorkspaceSize < tmpSize)
                    m_backFiltAlgo.AlgoWorkspaceSize = tmpSize;
            }
            return cudnnGetConvolutionBackwardFilterAlgorithm(*m_cudnn, m_inT, m_outT, *m_conv, *m_kernelT, CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, &algo);
        };
        FindBestAlgo(batchSize, m_backFiltAlgo, finder, staticFinder, workspace);
        // Compute gradients with respect to the output tensor (data).
        CUDNN_CALL(cudnnConvolutionBackwardFilter(*m_cudnn, &C::One, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, m_backFiltAlgo.selectedAlgo, ptr(workspace), workspace.BufferSize(), accumulateGradient ? &C::One : &C::Zero, *m_kernelT, ptr(kernelGrad)));
    }

    void EnsurePoolingInitialized() override
    {
        if (m_pool == nullptr)
            m_pool = std::make_unique<CuDnnPool>(*m_geometry, m_poolKind, m_forceDeterministicAlgorithms, m_poolIncludePad);
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

    void MaxUnpoolingCore(const Mat& out, const Mat& poolIn, Mat& in) override
    {
        UNUSED(out);
        UNUSED(poolIn);
        UNUSED(in);
        // Not implemented but potentially can make a fallback to reference engine.
        LogicError("MaxUnpooling is not implemented for cuDNN engine.");
    }

private:
    using C = Consts<ElemType>;

    static const int MaxAlgoCount = 10;

    template <typename TAlgo, typename TFinder, typename TStaticFinder>
    void FindBestAlgo(size_t batchSize, TAlgo& algo, TFinder finder, TStaticFinder staticFinder, Mat& workspace)
    {
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);

        // keep running if nothing changes
        if ((!algo.NeedAutotuning(batchSize)) && (workspace.BufferSize() >= algo.AlgoWorkspaceSize))
            return;

        // if batchsize changes again when just finish init, go back to init again
        if (algo.autotuningState == AutotuningState::PendingTuning && batchSize > algo.MBSizeForCurrentAlgo)
            algo.autotuningState = AutotuningState::Init;

        // batchSize is bigger than the one when initialize current workspace, need free up space and go back to init
        if (algo.autotuningState == AutotuningState::Running && batchSize > algo.maxMBSizeSeen)
        {
            algo.autotuningState = AutotuningState::Init;
            workspace.Resize(0,0,0,false);
            algo.AlgoWorkspaceSize = 0;
            algo.MBSizeForCurrentWorkspace = 0;
        } // batchSize changes but smaller than MBSizeForCurrentWorkspace, just need to re-do tuning
        else if (algo.autotuningState == AutotuningState::Running)
            algo.autotuningState = AutotuningState::PendingTuning;

        // in initState, where memory allocation for nodes are not completed, we only run the algorithm with no workspace
        // In such case, use static auto-tuner with no workspace and get m_MaxWorkspaceSize needed for findEx
        if (algo.autotuningState == AutotuningState::Init)
        {
            CUDNN_CALL(staticFinder(algo.selectedAlgo, true));
            algo.maxMBSizeSeen = batchSize;
            algo.MBSizeForCurrentAlgo = batchSize;
            algo.autotuningState = AutotuningState::PendingTuning;
            return;
        }

        // we allocate workspace and find algorithm if batchSize is higher than ever seen
        if (algo.MBSizeForCurrentWorkspace == 0)    // no workspace memory has been allocated for this node
        {
            size_t curSize = workspace.BufferSize();

            try
            {   // first try allocate as much to run FindEX, this may fail when accumulate is on
                size_t free, total, resizeTo = 0;
                CUDA_CALL(cudaMemGetInfo(&free, &total));
                free += workspace.BufferSize();
                // We reserve 2% of the total GPU memory because CuDNN seem to behave erroneously when there is no memory left
                if(free > (total/50))
                    resizeTo = free - (total/50) + sizeof(ElemType);
                // We don't need memory more than MAX
                if(resizeTo > algo.AlgoWorkspaceSize)
                    resizeTo = algo.AlgoWorkspaceSize;
                if(resizeTo > 0)
                    workspace.Resize((resizeTo + sizeof(ElemType) - 1) / sizeof(ElemType), 1);     // resize the workspace so that we can run the finder
                algo.MBSizeForCurrentWorkspace = batchSize;

                // Pending State now, let's do a find and get algorithm Perfs
                typename TAlgo::typeT algoPerf[MaxAlgoCount];
                int calgo = 0;

                CUDNN_CALL(finder(calgo, algoPerf));
                assert(calgo > 0);

                // To control memory usage. This flag seems not working and also no one uses it
                size_t inputSampleSize = m_geometry->InputShape().GetNumElements();
                size_t maxMem = m_maxTempMemSizeInSamples == 0 ? (std::numeric_limits<size_t>::max)() : inputSampleSize * m_maxTempMemSizeInSamples * sizeof(ElemType);

                // Find best (fastest) algorithm which satisfies workspace memory requirements.
                auto res = std::find_if(algoPerf, algoPerf + calgo,
                                        [=](const typename TAlgo::typeT& cur) { return cur.status == CUDNN_STATUS_SUCCESS && cur.memory <= maxMem; });
                if (res == algoPerf + calgo)
                    RuntimeError("During auto-tuning, cuDNN could not find suitable algorithm for the current convolution configuration.");
                algo.MBSizeForCurrentAlgo = batchSize;
                algo.selectedAlgo = (*res).algo;
                algo.maxAlgo = algo.selectedAlgo;
                algo.autotuningState = AutotuningState::Running;
                algo.AlgoWorkspaceSize = (*res).memory;
                if (algo.AlgoWorkspaceSize < curSize)   // need to shrink the workspace
                    workspace.Resize((curSize + sizeof(ElemType) - 1) / sizeof(ElemType), 1, 0, false);
                else
                    workspace.Resize((algo.AlgoWorkspaceSize + sizeof(ElemType) - 1) / sizeof(ElemType), 1, 0, false);
            } 
            catch (...) 
            {   // when it fails, it means accumulate is on, and allocation of temporary buffer failed. We resize to curSize and try again
                fprintf(stderr, "Retrying with reduced workspace memory for convolution\n"); 
                workspace.Resize((curSize + sizeof(ElemType) - 1) / sizeof(ElemType), 1, 0, false);
                try
                {
                    typename TAlgo::typeT algoPerf[MaxAlgoCount];
                    int calgo = 0;

                    CUDNN_CALL(finder(calgo, algoPerf));
                    assert(calgo > 0);

                    // To control memory usage. This flag seems not working and also no one uses it
                    size_t inputSampleSize = m_geometry->InputShape().GetNumElements();
                    size_t maxMem = m_maxTempMemSizeInSamples == 0 ? (std::numeric_limits<size_t>::max)() : inputSampleSize * m_maxTempMemSizeInSamples * sizeof(ElemType);

                    // Find best (fastest) algorithm which satisfies workspace memory requirements.
                    auto res = std::find_if(algoPerf, algoPerf + calgo,
                                            [=](const typename TAlgo::typeT& cur) { return cur.status == CUDNN_STATUS_SUCCESS && cur.memory <= maxMem; });
                    if (res == algoPerf + calgo)
                        RuntimeError("During auto-tuning, cuDNN could not find suitable algorithm for the current convolution configuration.");
                    algo.MBSizeForCurrentAlgo = batchSize;
                    algo.selectedAlgo = (*res).algo;
                    algo.maxAlgo = algo.selectedAlgo;
                    algo.autotuningState = AutotuningState::Running;
                    algo.AlgoWorkspaceSize = (*res).memory;
                } 
                catch (...) 
                {   // fails again, let's fall back to cudnnGet
                    fprintf(stderr, "Fall back to use static finder to get the algorithm for convolution\n");
                    CUDNN_CALL(staticFinder(algo.selectedAlgo, false));
                    algo.MBSizeForCurrentAlgo = batchSize;
                    algo.maxAlgo = algo.selectedAlgo;
                    algo.autotuningState = AutotuningState::Running;
                    algo.AlgoWorkspaceSize = curSize;
                }
            }
        }
        else if (batchSize == algo.MBSizeForCurrentWorkspace && workspace.BufferSize() >= algo.AlgoWorkspaceSize) // Use stored algo when batchsize go back to max. Likely happen when last batch in epoch lacking data
        {
            algo.selectedAlgo = algo.maxAlgo;
            algo.MBSizeForCurrentAlgo = batchSize;
            algo.autotuningState = AutotuningState::Running;
        }
        else if (m_forceDeterministicAlgorithms || m_maxTempMemSizeInSamples > 0)   // Need to do tunning if want deterministic/memlimit algorithm, there are better ways to do this
        {
            // Pending State now, let's do a find and get algorithm Perfs
            typename TAlgo::typeT algoPerf[MaxAlgoCount];
            int calgo = 0;
            CUDNN_CALL(finder(calgo, algoPerf));
            assert(calgo > 0);

            // To control memory usage. Need to investigate if still needed
            size_t inputSampleSize = m_geometry->InputShape().GetNumElements();
            size_t maxMem = m_maxTempMemSizeInSamples == 0 ? (std::numeric_limits<size_t>::max)() : inputSampleSize * m_maxTempMemSizeInSamples * sizeof(ElemType);

            // Find best (fastest) algorithm which satisfies workspace memory requirements.
            auto res = std::find_if(algoPerf, algoPerf + calgo,
                                    [=](const typename TAlgo::typeT& cur) { return cur.status == CUDNN_STATUS_SUCCESS && cur.memory <= maxMem; });
            if (res == algoPerf + calgo)
                RuntimeError("During auto-tuning, cuDNN could not find suitable algorithm for the current convolution configuration.");

            algo.MBSizeForCurrentAlgo = batchSize;
            algo.selectedAlgo = (*res).algo;
            algo.autotuningState = AutotuningState::Running;
        }
        else    // use fast method to get algorithm when batchsize get smaller. Avoid severe slowdown when batchsize change frequently
        {
            CUDNN_CALL(staticFinder(algo.selectedAlgo, false));
            algo.MBSizeForCurrentAlgo = batchSize;
            algo.autotuningState = AutotuningState::Running;
        }
        return;
    }

    static ElemType* ptr(Mat& src)
    {
        return src.Data();
    }
    static const ElemType* ptr(const Mat& src)
    {
        return src.Data();
    }

private:
    template <typename T>
    struct ConvAlgoInfo
    {
        typedef T typeT;
        ConvAlgoInfo()
            : MBSizeForCurrentAlgo(0), MBSizeForCurrentWorkspace(0), maxMBSizeSeen(0),autotuningState(AutotuningState::Init), AlgoWorkspaceSize(0)
        {
        }
        // Current mini-batch size, needed for re-computing statistics in auto-tuner.
        size_t maxMBSizeSeen;
        size_t MBSizeForCurrentAlgo;
        size_t MBSizeForCurrentWorkspace;
        size_t AlgoWorkspaceSize;
        AutotuningState autotuningState;
        decltype(T::algo) selectedAlgo;
        decltype(T::algo) maxAlgo;

        bool NeedAutotuning(size_t batchSize)
        {
            // We assume no other dimensions of tensors can change so we don't check it.
            // REVIEW alexeyk: review once we get response from NVIDIA.
            // NVIDIA response:
            // It is not safe to assume that previously selected algorithm requires less or the same amount of workspace when minibatch size decrease
            // Need to re-run auto-tuner everytime minibatch size grow.
            // Use faster(may not be optimal) method to get algorithm when batchsize decrease
            // Should remain reasonable performance when minibatch size changes frequently (e.g. distributed reading).
            return (autotuningState != AutotuningState::Running || batchSize != MBSizeForCurrentAlgo);
        }
    };

    CuDnn::ptr_t m_cudnn;
    cudnnDataType_t m_dataType;
    CuDnnTensor m_inT;
    CuDnnTensor m_outT;
    // Convolution specific.
    std::unique_ptr<CuDnnKernel> m_kernelT;
    std::unique_ptr<CuDnnConv> m_conv;
    // Pooling specific.
    std::unique_ptr<CuDnnPool> m_pool;

    ConvAlgoInfo<cudnnConvolutionFwdAlgoPerf_t> m_fwdAlgo;
    ConvAlgoInfo<cudnnConvolutionBwdDataAlgoPerf_t> m_backDataAlgo;
    ConvAlgoInfo<cudnnConvolutionBwdFilterAlgoPerf_t> m_backFiltAlgo;

    // Flag indicating whether only deterministic algorithms should be used.
    bool m_forceDeterministicAlgorithms;
};

template <class ElemType>
std::unique_ptr<ConvolutionEngine<ElemType>> CuDnnConvolutionEngineFactory<ElemType>::Create(ConvolveGeometryPtr geometry,
                                                                                             DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout,
                                                                                             size_t maxTempMemSizeInSamples, PoolKind poolKind,
                                                                                             bool forceDeterministicAlgorithms, bool poolIncludePad)
{
    return std::make_unique<CuDnnConvolutionEngine<ElemType>>(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, forceDeterministicAlgorithms, poolIncludePad);
}

template <class ElemType>
bool CuDnnConvolutionEngineFactory<ElemType>::IsSupported(DEVICEID_TYPE deviceId, ConvolveGeometryPtr geometry, PoolKind poolKind)
{
    // REVIEW alexeyk: IsSupported check should be performed by cuDNN itself. Is there a good way to do that?

    cudaDeviceProp props = {0};
    // Note that cudaGetDeviceProperties also sets CUDA last error so need to check/clear both.
    if (deviceId < 0 || (cudaGetDeviceProperties(&props, deviceId) | cudaGetLastError()) != cudaSuccess || props.major < 3)
        return false;

    const auto& input = geometry->InputShape();
    const auto& kernel = geometry->KernelShape();
    const auto& sharing = geometry->Sharing();
    const auto& mapCount = geometry->MapCount();

    const auto& inputRank = input.GetRank();
    const auto& kernelRank = kernel.GetRank();
    const auto& mapRank = mapCount.GetRank();
    // cuDNN supports 2D and 3D convolutions at the moment with full sharing.
    // In case map count size > 1, then it should have all ones except last dimension.
    // If pooling is requested, then cuDNN supports only 2D/3D inputs and 2D pooling kernels.
    bool retVal = (inputRank <= 4 &&
                   std::find(begin(sharing), end(sharing), false) == sharing.end() &&
                   mapCount.GetNumElements() == mapCount[mapRank - 1] &&
                   (poolKind == PoolKind::None ||
                   inputRank <= 3 && (kernelRank < 3 || kernel[2] == 1)));

    return retVal;

    // TODO: This currently either causes a CUDA timeout or slows the whole machine down to a crawl (GPU).
    // cuDNN as of version 8.0 does not handle asymmetric padding for convolution correctly. We need to detect asymmetric
    // padding due to auto-padding and choose the reference convolution implementation instead
    //if (poolKind == PoolKind::None)     // only for convolution, pooling seems fine
    //{
    //    for (int i = 0; i < kernelRank; i++)
    //    {
    //        if (geometry->GetAutoPad(i))
    //            retVal = retVal && (kernel[i] % 2 != 0);  // make sure kernel size is odd
    //        else
    //            retVal = retVal && (geometry->GetLowerPad(i) == geometry->GetUpperPad(i));   // lower pad is same as upper pad
    //    }
    //}
    //return retVal;
}

template class CuDnnConvolutionEngineFactory<float>;
template class CuDnnConvolutionEngineFactory<double>;

} } }
