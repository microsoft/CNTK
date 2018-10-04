//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "ConvolutionEngine.h"
#include "CuDnnFactories.h"
#include "MklDnnCommon.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void ConvolutionEngine<ElemType>::Forward(const Mat& in, const Mat& kernel, Mat& out, Mat& workspace)
{
    const auto& g = *m_geometry;
    assert(g.InputShape().GetNumElements() == in.GetNumRows());
    assert(g.OutputShape().GetNumElements() == out.GetNumRows());
    size_t batchSize = in.GetNumCols();
    assert(batchSize == out.GetNumCols());
    // REVIEW alexeyk: add shape-aware asserts?
    assert(g.KernelShape().GetNumElements() * g.KernelCount() == kernel.GetNumElements());
#ifdef NDEBUG
    UNUSED(g);
    UNUSED(batchSize);
#endif

    EnsureCompatible();
    EnsureConvolutionInitialized();
    ForwardCore(in, kernel, out, workspace);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::BackwardData(const Mat& srcGrad, const Mat& kernel, Mat& grad, bool accumulateGradient, Mat& workspace)
{
    const auto& g = *m_geometry;
    assert(g.InputShape().GetNumElements() == grad.GetNumRows());
    assert(g.OutputShape().GetNumElements() == srcGrad.GetNumRows());
    size_t batchSize = srcGrad.GetNumCols();
    assert(batchSize == grad.GetNumCols());
    assert(g.KernelShape().GetNumElements() * g.KernelCount() == kernel.GetNumElements());
#ifdef NDEBUG
    UNUSED(g);
    UNUSED(batchSize);
#endif

    EnsureCompatible();
    EnsureConvolutionInitialized();
    BackwardDataCore(srcGrad, kernel, grad, accumulateGradient, workspace);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::BackwardKernel(const Mat& srcGrad, const Mat& in, Mat& kernel, bool accumulateGradient, bool allowReuse, Mat& workspace)
{
    const auto& g = *m_geometry;
    assert(g.InputShape().GetNumElements() == in.GetNumRows());
    assert(g.OutputShape().GetNumElements() == srcGrad.GetNumRows());
    size_t batchSize = in.GetNumCols();
    assert(batchSize == srcGrad.GetNumCols());
    assert(g.KernelShape().GetNumElements() * g.KernelCount() == kernel.GetNumElements());
#ifdef NDEBUG
    UNUSED(g);
    UNUSED(batchSize);
#endif

    EnsureCompatible();
    EnsureConvolutionInitialized();
    BackwardKernelCore(srcGrad, in, kernel, accumulateGradient, allowReuse, workspace);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::ForwardPooling(const Mat& in, Mat& out)
{
    const auto& g = *m_geometry;
    assert(g.InputShape().GetNumElements() == in.GetNumRows());
    assert(g.OutputShape().GetNumElements() == out.GetNumRows());
    size_t batchSize = in.GetNumCols();
    assert(batchSize == out.GetNumCols());
#ifdef NDEBUG
    UNUSED(g);
    UNUSED(batchSize);
#endif

    EnsureCompatible();
    EnsurePoolingInitialized();
    ForwardPoolingCore(in, out);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::BackwardPooling(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad, bool accumulateGradient)
{
    const auto& g = *m_geometry;
    assert(g.InputShape().GetNumElements() == grad.GetNumRows());
    assert(g.InputShape().GetNumElements() == in.GetNumRows());
    assert(g.OutputShape().GetNumElements() == srcGrad.GetNumRows());
    assert(g.OutputShape().GetNumElements() == out.GetNumRows());
    size_t batchSize = out.GetNumCols();
    assert(batchSize == srcGrad.GetNumCols());
    assert(batchSize == in.GetNumCols());
    assert(batchSize == grad.GetNumCols());
#ifdef NDEBUG
    UNUSED(g);
    UNUSED(batchSize);
#endif

    EnsureCompatible();
    EnsurePoolingInitialized();
    BackwardPoolingCore(out, srcGrad, in, grad, accumulateGradient);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::MaxUnpooling(const Mat& out, const Mat& poolIn, Mat& in)
{
    const auto& g = *m_geometry;
    assert(g.InputShape().GetNumElements() == in.GetNumRows());
    assert(g.InputShape().GetNumElements() == poolIn.GetNumRows());
    assert(g.OutputShape().GetNumElements() == out.GetNumRows());
    size_t batchSize = in.GetNumCols();
    assert(batchSize == out.GetNumCols());
    assert(batchSize == poolIn.GetNumCols());
#ifdef NDEBUG
    UNUSED(g);
    UNUSED(batchSize);
#endif

    EnsureCompatible();
    EnsurePoolingInitialized();
    MaxUnpoolingCore(out, poolIn, in);
}

//------------------------------------------------------------------
// Reference convolution engine implementation.
// This engine supports arbitrary convolution geometry but does not provide efficient implementation.
// Its main purpose is to serve as a baseline for optmized engines (e.g. cuDNN) that
// usually implement only a subset of a general convolution geometry.
//------------------------------------------------------------------
template <class ElemType>
class ReferenceConvolutionEngine : public ConvolutionEngine<ElemType>
{
public:
    using Base = ConvolutionEngine<ElemType>;
    using typename Base::Mat;

public:
    ReferenceConvolutionEngine(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind, bool poolIncludePad)
        : Base(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, poolIncludePad),
        m_isConvGeometryComputed(geometry->ComputeConvGeometryExplicit()), // IMP NOTE: m_isConvGeometryComputed MUST be initialized before m_mpRowCol here in this list.
        m_mpRowCol(geometry->MpRowCol().size(), 1, const_cast<int*>(geometry->MpRowCol().data()), deviceId, IsGpu(deviceId) ? matrixFlagNormal : matrixFlagDontOwnBuffer)
    {
        assert(m_isConvGeometryComputed);
    }

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
            RuntimeError("Reference convolution engine supports only CHW/cudnn layout.");
    }

    void EnsureConvolutionInitialized() override
    {
        if (m_mpRowIwht == nullptr)
        {
            auto flags = IsGpu(m_deviceId) ? matrixFlagNormal : matrixFlagDontOwnBuffer;
            m_mpRowIwht = std::make_unique<Matrix<int>>(m_geometry->MpRowIwht().size(), 1,
                                                        const_cast<int*>(m_geometry->MpRowIwht().data()), m_deviceId, flags);
            m_mpRowRun = std::make_unique<Matrix<int>>(m_geometry->MpRowRun().size(), 1,
                                                       const_cast<int*>(m_geometry->MpRowRun().data()), m_deviceId, flags);
            m_runs = std::make_unique<Matrix<int>>(m_geometry->Runs().size(), 1,
                                                   const_cast<int*>(m_geometry->Runs().data()), m_deviceId, flags);
        }
    }

    void ForwardCore(const Mat& in, const Mat& kernel, Mat& out, Mat& /*workspace*/) override
    {
        in.ConvolutionForward(kernel, m_mpRowCol, *m_mpRowIwht, *m_mpRowRun, *m_runs, out);
    }

    void BackwardDataCore(const Mat& srcGrad, const Mat& kernel, Mat& grad, bool /*accumulateGradient*/, Mat& /*workspace*/) override
    {
        srcGrad.ConvolutionBackwardData(kernel, m_mpRowCol, *m_mpRowIwht, *m_mpRowRun, *m_runs, grad);
    }

    void BackwardKernelCore(const Mat& srcGrad, const Mat& in, Mat& kernelGrad, bool /*accumulateGradient*/, bool /*allowReuse*/, Mat& /*workspace*/) override
    {
        srcGrad.ConvolutionBackwardKernel(in, m_mpRowCol, *m_mpRowIwht, *m_mpRowRun, *m_runs, kernelGrad);
    }

    void EnsurePoolingInitialized() override
    {
        if (m_indices == nullptr)
        {
            auto flags = IsGpu(m_deviceId) ? matrixFlagNormal : matrixFlagDontOwnBuffer;
            m_mpRowIndices = std::make_unique<Matrix<int>>(m_geometry->MpRowIndices().size(), 1,
                                                           const_cast<int*>(m_geometry->MpRowIndices().data()), m_deviceId, flags);
            m_indices = std::make_unique<Matrix<int>>(m_geometry->Indices().size(), 1,
                                                      const_cast<int*>(m_geometry->Indices().data()), m_deviceId, flags);
        }
    }

    void ForwardPoolingCore(const Mat& in, Mat& out) override
    {
        if (m_poolKind == PoolKind::Max)
        {
            in.MaxPoolingForward(m_mpRowCol, *m_mpRowIndices, *m_indices, out);
        }
        else if (m_poolKind == PoolKind::Average)
        {
            in.AveragePoolingForward(m_mpRowCol, *m_mpRowIndices, *m_indices, out, m_poolIncludePad);
        }
        else
            InvalidArgument("Pooling type %d is not supported.", (int)m_poolKind);

    }

    void BackwardPoolingCore(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad, bool accumulateGradient) override
    {
        if (m_poolKind == PoolKind::Max)
        {
            srcGrad.MaxPoolingBackward(out, in, m_mpRowCol, *m_mpRowIndices, *m_indices, grad, accumulateGradient);
        }
        else if (m_poolKind == PoolKind::Average)
        {
            srcGrad.AveragePoolingBackward(m_mpRowCol, *m_mpRowIndices, *m_indices, grad, m_poolIncludePad, accumulateGradient);
        }
        else
            InvalidArgument("Pooling type %d is not supported.", (int)m_poolKind);
    }

    void MaxUnpoolingCore(const Mat& out, const Mat& poolIn, Mat& in) override
    {
        out.MaxUnpooling(m_mpRowCol, *m_mpRowIndices, *m_indices, poolIn, in);
    }

protected:
    using IntMatPtr = std::unique_ptr<Matrix<int>>;
    // IMP NOTE: Make sure that in the declaration below m_isConvGeometryComputed is declared
    // before m_mpRowCol. This ordering is required to ensure the right order of initialization
    // in the initializer list in the ctor (above) of this class.
    bool m_isConvGeometryComputed;  
    Matrix<int> m_mpRowCol;
    // Convolution-specific maps.
    IntMatPtr m_mpRowIwht;
    IntMatPtr m_mpRowRun;
    IntMatPtr m_runs;
    // Pooling-specific maps.
    IntMatPtr m_mpRowIndices;
    IntMatPtr m_indices;
};

//------------------------------------------------------------------
// Legacy convolution engine implementation.
//------------------------------------------------------------------
template <class ElemType>
class LegacyConvolutionEngine : public ConvolutionEngine<ElemType>
{
public:
    using Base = ConvolutionEngine<ElemType>;
    using typename Base::Mat;

public:
    LegacyConvolutionEngine(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind, bool poolIncludePad)
        : Base(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, poolIncludePad),
        m_inT(m_geometry->InputShape(), ImageLayoutKind::CHW), m_outT(m_geometry->OutputShape(), ImageLayoutKind::CHW),
        m_kernelT(m_geometry->KernelShape(), ImageLayoutKind::CHW), m_strideT(m_geometry->Stride(), ImageLayoutKind::CHW)
    {
        m_padding = m_geometry->AutoPad()[0];
    }

protected:
    using Base::m_geometry;
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_maxTempMemSizeInSamples;
    using Base::m_poolKind;
    using Base::m_poolIncludePad;

    void EnsureCompatible() override
    {
        if (m_imageLayout != ImageLayoutKind::HWC)
            RuntimeError("Legacy convolution engine supports only HWC/legacy layout.");
    }

    void EnsureConvolutionInitialized() override
    {
    }

    void ForwardCore(const Mat& in, const Mat& kernel, Mat& out, Mat& workspace) override
    {
        size_t batchSize = in.GetNumCols();
        size_t packedInputRows = m_kernelT.w() * m_kernelT.h() * m_kernelT.c();
        size_t packedInputColsPerSample = m_outT.w() * m_outT.h();
        size_t outputSizePerChannel = packedInputColsPerSample;
        // size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
        // size_t inputDim = inT.w() * inT.h() * inT.c();  // size of each input sample

        size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

        assert(kernel.GetNumCols() == packedInputRows && kernel.GetNumRows() == m_outT.c());
        UNUSED(packedInputRows);

        // GPU and 1-dimensional image
        m_gpuSparseOpt = (m_kernelT.h() == 1 &&
                          in.GetCurrentMatrixLocation() == CurrentDataLocation::GPU &&
                          m_strideT.w() == 1 &&
                          !m_padding &&
                          in.GetMatrixType() == MatrixType::SPARSE);
        m_gpuSparse1D = (m_gpuSparseOpt && m_inT.h() == 1);

        out.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, false);

        // Reshaping is only necessary if we are going to use the unpacking trick
        if (m_gpuSparseOpt)
            out.Reshape(m_outT.c() * m_outT.w(), m_outT.h() * batchSize);
        else
            out.Reshape(m_outT.c(), outputSizePerChannel * batchSize);

        size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples);
        size_t numSubBatches = (batchSize + subBatchSize - 1) / subBatchSize;

        for (size_t i = 0; i < numSubBatches; i++)
        {
            size_t startSampleId = i * subBatchSize;
            size_t endSampleId = min(batchSize, startSampleId + subBatchSize);
            size_t smallBatchSize = endSampleId - startSampleId;
            Mat inputSubBatch(in.GetDeviceId());

            // We optimize for three different scenarios here by handling them slightly differently.
            // [Scenario 1] Dense: Unroll using AssignPackedConvolutionInput and multiply.
            // [Scenario 2] Sparse 1-D convolution on GPU: for text scenarios we have a specific kernel.
            // [Scenario 3] Sparse all others: convert to dense. Temporary work-around - allocating/de-allocating memory is costly!
            if (in.GetMatrixType() == MatrixType::DENSE || m_gpuSparse1D)
                inputSubBatch = in.ColumnSlice(startSampleId, smallBatchSize);
            else
                inputSubBatch.SetValue(in.ColumnSlice(startSampleId, smallBatchSize));

            if (m_gpuSparseOpt)
            {
                if (m_kernelT.w() * m_inT.c() != kernel.GetNumCols())
                    LogicError("Kernel width and weight matrix dimensions don't match.");

                inputSubBatch.Reshape(m_inT.c() * m_inT.w(), m_inT.h() * smallBatchSize);
                Mat outputSubBatch = out.ColumnSlice(startSampleId, m_outT.h() * smallBatchSize);
                Mat::ConvolveAndWeightedAdd(1, kernel, false, inputSubBatch, false, 0, outputSubBatch,
                                            static_cast<int>(m_inT.c()), m_strideT.w(), m_padding, true);
            }
            else
            {
                inputSubBatch.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, true);
                workspace.AssignPackedConvolutionInput(inputSubBatch,
                                                       m_inT.w(), m_inT.h(), m_inT.c(),
                                                       m_outT.w(), m_outT.h(), m_outT.c(),
                                                       m_kernelT.w(), m_kernelT.h(), m_strideT.w(), m_strideT.h(),
                                                       m_padding);

                Mat outputSubBatch = out.ColumnSlice(outputSizePerChannel * startSampleId, outputSizePerChannel * smallBatchSize);

                // workspace.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                // BUGBUG: This ^^ destroys the content of the matrix. Also it seems not to change the size. Does it? Should this be a Reshape()?
                Mat::Multiply(kernel, false, workspace, false, outputSubBatch);
            }
        }

        out.Reshape(m_outT.c() * outputSizePerChannel, batchSize); // each sample becomes a column

        assert(m_outT.w() * m_outT.h() * m_outT.c() == out.GetNumRows());
        assert(batchSize == out.GetNumCols());
    }

    void BackwardDataCore(const Mat& srcGrad, const Mat& kernel, Mat& grad, bool /*accumulateGradient*/, Mat& workspace) override
    {
        size_t batchSize = srcGrad.GetNumCols();
        size_t packedInputRows = m_kernelT.w() * m_kernelT.h() * m_kernelT.c();
        size_t packedInputColsPerSample = m_outT.w() * m_outT.h();
        size_t outputSizePerChannel = packedInputColsPerSample;
        // size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
        // size_t inputDim = m_inT.w() * m_inT.h() * m_inT.c();  // size of each input sample

        size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

        // Create slice which is the same as full matrix so we can reshape it.
        Matrix<ElemType> srcGradTmp = srcGrad.ColumnSlice(0, srcGrad.GetNumCols());
        srcGradTmp.Reshape(m_outT.c(), outputSizePerChannel * batchSize); // reshape to match the longernal operation

        size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples);
        size_t numSubBatches = (batchSize + subBatchSize - 1) / subBatchSize;

        for (size_t i = 0; i < numSubBatches; i++)
        {
            size_t startSampleId = i * subBatchSize;
            size_t endSampleId = min(batchSize, startSampleId + subBatchSize);
            size_t smallBatchSize = endSampleId - startSampleId;

            workspace.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
            Matrix<ElemType> outputGradientSubBatch = srcGradTmp.ColumnSlice(startSampleId * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
            Matrix<ElemType>::Multiply(kernel, true, outputGradientSubBatch, false, workspace);

            Matrix<ElemType> inputGradientSubBatch = grad.ColumnSlice(startSampleId, smallBatchSize);
            workspace.UnpackConvolutionInput(inputGradientSubBatch,
                                             m_inT.w(), m_inT.h(), m_inT.c(),
                                             m_outT.w(), m_outT.h(), m_outT.c(),
                                             m_kernelT.w(), m_kernelT.h(), m_strideT.w(), m_strideT.h(),
                                             m_padding);
        }

        assert(m_outT.w() * m_outT.h() * m_outT.c() == srcGrad.GetNumRows());
        assert(batchSize == srcGrad.GetNumCols());
    }

    void BackwardKernelCore(const Mat& srcGrad, const Mat& in, Mat& kernelGrad, bool /*accumulateGradient*/, bool allowReuse, Mat& workspace) override
    {
        size_t batchSize = in.GetNumCols();
        size_t packedInputRows = m_kernelT.w() * m_kernelT.h() * m_kernelT.c();
        size_t packedInputColsPerSample = m_outT.w() * m_outT.h();
        size_t outputSizePerChannel = packedInputColsPerSample;
        // size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
        // size_t inputDim = m_inputImageLayout.width * m_inputImageLayout.height * m_inputImageLayout.channels;  // size of each input sample

        size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

        // const Matrix<ElemType> & weightMatrix = input0;
        // inputGradientValues.Resize(weightMatrix.GetNumRows(), weightMatrix.GetNumCols()); // should have been resized when preparing gradient computation

        // Create slice which is the same as full matrix so we can reshape it.
        Matrix<ElemType> srcGradTmp = srcGrad.ColumnSlice(0, srcGrad.GetNumCols());
        srcGradTmp.Reshape(m_outT.c(), outputSizePerChannel * batchSize); // reshape to match the longernal operation

        size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples);
        size_t numSubBatches = (batchSize + subBatchSize - 1) / subBatchSize;

        if (numSubBatches == 1 && allowReuse && !m_gpuSparseOpt) // reuse packed input from evaluation step if it's not changed by either subbatch or recurrent steps.
            // REVIEW alexeyk: the following makes an assumption that data in workspace was filled by Forward call and remained unchanged. Find way to enforce/verify that.
            Matrix<ElemType>::MultiplyAndAdd(srcGradTmp, false, workspace, true, kernelGrad);
        else
        {
            for (size_t i = 0; i < numSubBatches; i++)
            {
                size_t startSampleID = i * subBatchSize;
                size_t endSampleID = min(batchSize, startSampleID + subBatchSize);
                size_t smallBatchSize = endSampleID - startSampleID;
                Matrix<ElemType> outputGradientSubBatch = srcGradTmp.ColumnSlice(startSampleID * outputSizePerChannel, smallBatchSize * outputSizePerChannel);

                // We optimize for three different scenarios here by handling them slightly differently.
                // [Scenario 1] Dense: Unroll using AssignPackedConvolutionInput and multiply.
                // [Scenario 2] Sparse 1-D convolution on GPU: for text scenarios we have a specific kernel.
                // [Scenario 3] Sparse all others: convert to dense. Temporary work-around - allocating/de-allocating memory is costly!
                if (m_gpuSparseOpt)
                {
                    Matrix<ElemType> inputSubBatch(in.GetDeviceId());
                    inputSubBatch.SetValue(in.ColumnSlice(startSampleID, smallBatchSize));
                    inputSubBatch.Reshape(m_inT.c(), smallBatchSize * m_inT.w() * m_inT.h());
                    Matrix<ElemType> inputSubBatchSparseReordered(inputSubBatch.GetNumCols(), inputSubBatch.GetNumRows(), inputSubBatch.GetDeviceId(), MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC);
                    Matrix<ElemType>::TensorShuffleScaleAndAdd(0.0f, inputSubBatch.Transpose(), 1, m_inT.w(), 1, smallBatchSize * m_inT.h(), m_inT.c(), 1.0f, inputSubBatchSparseReordered, inputSubBatchSparseReordered);

                    Matrix<ElemType> outputGradientSubBatchReordered = Matrix<ElemType>::Zeros(smallBatchSize * m_outT.h() * m_outT.w(), m_outT.c(), outputGradientSubBatch.GetDeviceId());
                    Matrix<ElemType>::TensorShuffleScaleAndAdd(0.0f, outputGradientSubBatch.Transpose(), 1, m_outT.w(), 1, smallBatchSize * m_outT.h(), m_outT.c(), 1.0f, outputGradientSubBatchReordered, outputGradientSubBatchReordered);

                    kernelGrad.Reshape(m_outT.c() * m_kernelT.w(), m_inT.c());
                    Matrix<ElemType>::ConvolveAndWeightedAdd(1, outputGradientSubBatchReordered, true, inputSubBatchSparseReordered, false, 1, kernelGrad, smallBatchSize * m_inT.h(), m_strideT.w(), m_padding, false);
                    kernelGrad.Reshape(m_outT.c(), m_inT.c() * m_kernelT.w());
                }
                else
                {
                    workspace.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                    Matrix<ElemType> inputSubBatch = in.ColumnSlice(startSampleID, smallBatchSize);
                    inputSubBatch.SwitchToMatrixType(MatrixType::DENSE, inputSubBatch.GetFormat(), true);
                    workspace.AssignPackedConvolutionInput(inputSubBatch,
                                                           m_inT.w(), m_inT.h(), m_inT.c(),
                                                           m_outT.w(), m_outT.h(), m_outT.c(),
                                                           m_kernelT.w(), m_kernelT.h(), m_strideT.w(), m_strideT.h(),
                                                           m_padding);

                    Matrix<ElemType>::MultiplyAndAdd(outputGradientSubBatch, false, workspace, true, kernelGrad);
                }
            }
        }

        assert(m_outT.w() * m_outT.h() * m_outT.c() == srcGrad.GetNumRows());
        assert(batchSize == srcGrad.GetNumCols());
    }

    void EnsurePoolingInitialized() override
    {
    }

    void ForwardPoolingCore(const Mat& in, Mat& out) override
    {
        if (m_poolKind == PoolKind::Max)
        {
            out.AssignMaxPoolingResult(in, m_inT.c(), m_inT.w(), m_inT.h(), m_inT.w() * m_inT.h() * m_inT.c(),
                                       m_outT.w(), m_outT.h(), m_outT.w() * m_outT.h() * m_outT.c(),
                                       m_kernelT.w(), m_kernelT.h(), m_strideT.w(), m_strideT.h());
        }
        else if (m_poolKind == PoolKind::Average)
        {
            out.AssignAveragePoolingResult(in, m_inT.c(), m_inT.w(), m_inT.h(), m_inT.w() * m_inT.h() * m_inT.c(),
                                           m_outT.w(), m_outT.h(), m_outT.w() * m_outT.h() * m_outT.c(),
                                           m_kernelT.w(), m_kernelT.h(), m_strideT.w(), m_strideT.h());
        }
        else
            InvalidArgument("Pooling type %d is not supported.", (int)m_poolKind);
    }

    void BackwardPoolingCore(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad, bool accumulateGradient) override
    {
        if (!accumulateGradient)
            grad.SetValue(0);

        if (m_poolKind == PoolKind::Max)
        {
            grad.AddMaxPoolingGradient(srcGrad, in, out,
                                       m_inT.c(), m_inT.w(), m_inT.h(), m_inT.w() * m_inT.h() * m_inT.c(),
                                       m_outT.w(), m_outT.h(), m_outT.w() * m_outT.h() * m_outT.c(),
                                       m_kernelT.w(), m_kernelT.h(), m_strideT.w(), m_strideT.h());
        }
        else if (m_poolKind == PoolKind::Average)
        {
            grad.AddAveragePoolingGradient(srcGrad, m_inT.c(), m_inT.w(), m_inT.h(), m_inT.w() * m_inT.h() * m_inT.c(),
                                           m_outT.w(), m_outT.h(), m_outT.w() * m_outT.h() * m_outT.c(),
                                           m_kernelT.w(), m_kernelT.h(), m_strideT.w(), m_strideT.h());
        }
        else
            InvalidArgument("Pooling type %d is not supported.", (int)m_poolKind);
    }

    void MaxUnpoolingCore(const Mat& out, const Mat& poolIn, Mat& in) override
    {
        UNUSED(out);
        UNUSED(poolIn);
        UNUSED(in);
        // Not implemented but potentially can make a fallback to reference engine.
        LogicError("MaxUnpooling is not implemented for legacy engine.");
    }

private:
    ImageDimensions m_inT;
    ImageDimensions m_outT;
    ImageDimensions m_kernelT;
    ImageDimensions m_strideT;
    bool m_padding;

    bool m_gpuSparseOpt;
    bool m_gpuSparse1D;
};

//------------------------------------------------------------------
// GEMM convolution engine implementation.
// This engine supports arbitrary convolution configuration with full
// sharing and implemented using unroll + GEMM technique
// (High performance convolutional neural networks for document processing; Chellapilla, Puri, Simard)
// Uses reference engine for pooling operations.
//------------------------------------------------------------------
template <class ElemType>
class GemmConvolutionEngine : public ReferenceConvolutionEngine<ElemType>
{
public:
    using Base = ReferenceConvolutionEngine<ElemType>;
    using typename Base::Mat;

public:
    GemmConvolutionEngine(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind, bool poolIncludePad)
        : Base(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, poolIncludePad)
    {
    }

protected:
    using typename Base::IntMatPtr;

    using Base::m_geometry;
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_maxTempMemSizeInSamples;
    using Base::m_poolIncludePad;

    using Base::m_mpRowCol;
    using Base::m_mpRowIwht;
    using Base::m_mpRowRun;
    using Base::m_runs;

    void EnsureCompatible() override
    {
        if (m_imageLayout != ImageLayoutKind::CHW)
            LogicError("GEMM convolution engine supports only CHW/cudnn layout.");
        if (IsGpu(m_deviceId))
            LogicError("GEMM convolution engine currently supports only CPU device.");
    }

    // A note on notation used in the documentation for the next 3 functions:
    // for simplicity we use cuDNN-style notation for 2D convolutions (though this engine supports arbitrary convolution configuration)
    // where N - is the number of samples in a batch, C, H, W are number of channels, height and width of the input respectively.
    // For the output we use K as the number of output feature maps and H', W' as height and width of the output.
    // We also use column-major notation everywhere (as opposed to cuDNN which uses row-major) to follow CNTK rules.
    // For kernels we use X, Y, Z to represent width, height and depth. This engine requires Z == C which is
    // not a significant restriction as tensors of higher dimensions (+1) can be used to describe the same convolution configuration.
    // Example: [WHC x N] - is a matrix of WHC rows by N columns and represents a convolution input
    // where each column is a sample that has layout of WHC, so W dimension stride is 1.
    //
    // The forward method consists of 3 parts:
    // 1. Unrolling convolution input (in) into a matrix: [WHC x N] -> [XYC x NW'H']
    //    Using this format allows to perform convolution for the whole minibatch as a single GEMM operation
    //    which is not possible with WHCN format. Alternatively, CWHN format (used in legacy engine) could be used
    //    but this would require both unrolling the input and transforming the weight matrix.
    // 2. Performing matrix multiplication of unrolled input with weight matrix:
    //    [XYC x NW'H']^T * [XYC x K] -> [NW'H' x K]
    // 3. Reshape and transpose result: [NW'H' x K] -> [N x W'H'K]^T -> [W'H'K x N]
    //    In case minibatch size == 1 this step is not required and step 2 writes results directly to output (out).
    void ForwardCore(const Mat& in, const Mat& kernel, Mat& out, Mat& workspace) override
    {
#ifdef USE_MKL2017DNN
        if (ForwardCoreMKL(in, kernel, out)) return;
#endif

        size_t batchSize = in.GetNumCols();
        size_t subBatchSize = m_maxTempMemSizeInSamples == 0 ? batchSize : min(batchSize, m_maxTempMemSizeInSamples);

        size_t mapCount = m_geometry->GetMapCount(m_geometry->InputShape().GetRank() - 1);
        size_t mapOutSize = m_geometry->OutputShape().GetNumElements() / mapCount;
        size_t unrollRows = mapOutSize * subBatchSize;
        size_t unrollCols = m_geometry->KernelShape().GetNumElements();
        // Reserve space for unrolled inputs and, if needed, intermediate outputs.
        // Intermediate outputs will be transposed to final outputs after GEMM operation.
        // Transpose is not required if subBatchSize == 1.
        workspace.Resize(unrollRows, unrollCols + (subBatchSize > 1 ? mapCount : 0));

        for (size_t start = 0; start < batchSize; start += subBatchSize)
        {
            size_t curBatchSize = min(subBatchSize, batchSize - start);
            auto inputSlice = in.ColumnSlice(start, curBatchSize);
            auto unrolledInput = workspace.ColumnSlice(0, unrollCols);
            if (curBatchSize != subBatchSize)
            {
                unrolledInput.Reshape(mapOutSize, subBatchSize * unrollCols);
                unrolledInput = unrolledInput.ColumnSlice(0, curBatchSize * unrollCols);
            }
            // Need to reshape (soft transpose) as matrices are column-major.
            unrolledInput.Reshape(unrollCols, mapOutSize * curBatchSize);

            // Unroll inputs.
            unrolledInput.SetValue(0);
            inputSlice.UnrollConvolutionInput(unrollCols, mapOutSize, m_mpRowCol, *m_mpRowRun, *m_runs, unrolledInput);

            // cudnn layout uses row-major kernel weight matrix.
            auto kern = kernel.ColumnSlice(0, kernel.GetNumCols());
            kern.Reshape(unrollCols, kernel.GetNumElements()/unrollCols);

            // Perform matrix multiplication of unrolled inputs with weights.
            // If there is just one sample in the sub-batch then compute result directly to the output matrix.
            if (curBatchSize == 1)
            {
                auto outSlice = out.ColumnSlice(start, 1);
                outSlice.Reshape(mapOutSize, mapCount);
                Mat::Multiply(unrolledInput, true, kern, false, outSlice);
            }
            else
            {
                auto outTempSlice = workspace.ColumnSlice(unrollCols, mapCount);
                if (curBatchSize != subBatchSize)
                {
                    outTempSlice.Reshape(mapOutSize, subBatchSize * mapCount);
                    outTempSlice = outTempSlice.ColumnSlice(0, curBatchSize * mapCount);
                    outTempSlice.Reshape(mapOutSize * curBatchSize, mapCount);
                }
                Mat::Multiply(unrolledInput, true, kern, false, outTempSlice);
                outTempSlice.Reshape(curBatchSize, mapOutSize * mapCount);
                auto outSlice = out.ColumnSlice(start, curBatchSize);
                outSlice.AssignTransposeOf(outTempSlice);
            }
        }
    }

    // The backward data method works by representing this operation as a "reverse" convolution
    // in case kernel's last dimension is equal to input dimension. Gradients matrix (grad) becomes
    // an output of such reverse convolution.
    // There are 4 steps:
    // 1. Transpose and reshape kernel weights: [XYC x K]^T -> [K x XYC] -> [KXY x C]
    // 2. Unroll convolution output (here source gradients, srcGrad):
    //    [W'H'K' x N] -> [KXY x NWH]
    // 3. Performing matrix multiplication of unrolled scrGrad with transposed weights:
    //    [KXY x NWH]^T * [KXY x C] -> [NWH x C]
    // 4. Reshape and transpose outputs (grad): [NWH x C] -> [N x WHC]^T -> [WHC x N]
    //    In case minibatch size == 1 this step is not required and step 3 writes results directly to output (grad).
    void BackwardDataCore(const Mat& srcGrad, const Mat& kernel, Mat& grad, bool accumulateGradient, Mat& workspace) override
    {
#ifdef USE_MKL2017DNN
        if (BackwardDataMKL(srcGrad, kernel, grad, accumulateGradient, workspace)) return;
#else
        UNUSED(accumulateGradient);
#endif

        size_t batchSize = srcGrad.GetNumCols();
        size_t subBatchSize = m_maxTempMemSizeInSamples == 0 ? batchSize : min(batchSize, m_maxTempMemSizeInSamples);

        const auto& inT = m_geometry->InputShape();
        const auto& kernT = m_geometry->KernelShape();

        size_t dimCount = inT.GetRank();
        assert(kernT[dimCount - 1] == inT[dimCount - 1]);
        if (kernT[dimCount - 1] != inT[dimCount - 1])
        {
            RuntimeError("GEMM convolution engine does not support this convolution configuration. "
                         "It is possible to make GEMM engine work with this configuration by defining "
                         "input/output/kernel using tensors of higher(+1) dimension. Geometry: %s", ((string)*m_geometry).c_str());
        }

        size_t mapInCount  = kernT[dimCount - 1];
        size_t mapOutCount = m_geometry->GetMapCount(dimCount - 1);
        size_t mapInSize   = inT.GetNumElements() / mapInCount;

        size_t unrollRows = mapInSize * subBatchSize;
        size_t unrollCols = kernel.GetNumElements() / mapInCount;

        // Reserve space for:
        // 1. Transposed kernel weights.
        // 2. Unrolled source gradients.
        // 3. Intermediate gradients (optional).
        // Intermediate outputs will be transposed to final outputs after GEMM operation.
        // Transpose is not required if subBatchSize == 1.
        size_t kernCols = kernel.GetNumElements();
        workspace.Resize(1, kernCols + unrollRows * (unrollCols + (subBatchSize > 1 ? mapInCount : 0)));

        auto kern = kernel.ColumnSlice(0, kernel.GetNumCols());
        size_t kernTCols = kernT.GetNumElements();
        // cudnn layout uses row-major kernel weight matrix.
        kern.Reshape(kernTCols, kernCols/kernTCols);
        // Now transpose and reshape to [KXY x C].
        auto kernTran = workspace.ColumnSlice(0, kernCols);
        // Reshape to transpose shape, AssignTransposeOf requires that.
        kernTran.Reshape(kern.GetNumCols(), kern.GetNumRows());
        kernTran.AssignTransposeOf(kern);
        kern = kernTran.ColumnSlice(0, kernTran.GetNumCols());
        // Reshape to final shape.
        kern.Reshape(unrollCols, mapInCount);

        for (size_t start = 0; start < batchSize; start += subBatchSize)
        {
            size_t curBatchSize = min(subBatchSize, batchSize - start);
            auto srcGradSlice = srcGrad.ColumnSlice(start, curBatchSize);
            auto unrolledSrcGrad = workspace.ColumnSlice(kernCols, unrollRows * unrollCols);
            if (curBatchSize != subBatchSize)
                unrolledSrcGrad = unrolledSrcGrad.ColumnSlice(0, mapInSize * curBatchSize * unrollCols);
            // Need to reshape (soft transpose) as matrices are column-major.
            unrolledSrcGrad.Reshape(unrollCols, mapInSize * curBatchSize);

            // Unroll outputs (source gradients).
            unrolledSrcGrad.SetValue(0);
            srcGradSlice.UnrollConvolutionOutput(unrollCols, mapInCount, mapOutCount, m_mpRowCol, *m_mpRowRun, *m_runs, unrolledSrcGrad);

            // Perform matrix multiplication of unrolled outputs with weights.
            // If there is just one sample in the sub-batch then compute result directly to the output matrix.
            if (curBatchSize == 1)
            {
                auto gradSlice = grad.ColumnSlice(start, 1);
                gradSlice.Reshape(mapInSize, mapInCount);
                Mat::MultiplyAndAdd(unrolledSrcGrad, true, kern, false, gradSlice);
            }
            else
            {
                // Need to transpose existing destination gradients first so we can add new values to them.
                auto gradTempSlice = workspace.ColumnSlice(kernCols + unrollRows * unrollCols, unrollRows * mapInCount);
                if (curBatchSize != subBatchSize)
                    gradTempSlice = gradTempSlice.ColumnSlice(0, mapInSize * curBatchSize * mapInCount);
                gradTempSlice.Reshape(curBatchSize, mapInSize * mapInCount);
                auto gradSlice = grad.ColumnSlice(start, curBatchSize);
                gradTempSlice.AssignTransposeOf(gradSlice);
                gradTempSlice.Reshape(mapInSize * curBatchSize, mapInCount);
                // Multiply unrolled srcGrad with weights and add to grad.
                Mat::MultiplyAndAdd(unrolledSrcGrad, true, kern, false, gradTempSlice);
                // Reshape and transpose grads back to original form.
                gradTempSlice.Reshape(curBatchSize, mapInSize * mapInCount);
                gradSlice.AssignTransposeOf(gradTempSlice);
            }
        }
    }

    // The backward kernel method consists of 3 parts:
    // 1. Transpose and reshape convolution output matrix (srcGrad) into [NW'H' x K] layout.
    //    This step is not needed if current minibatch size == 1 and srcGrad are used instead.
    // 2. Unrolling convolution input (in) into a matrix of [NW'H' x WHC] layout.
    // 3. Performing matrix multiplication of unrolled input with transposed output:
    //    [NW'H' x WHC]^T * [NW'H' x K] -> [WHC x K] - kernel gradients.
    void BackwardKernelCore(const Mat& srcGrad, const Mat& in, Mat& kernelGrad, bool accumulateGradient, bool /*allowReuse*/, Mat& workspace) override
    {
#ifdef USE_MKL2017DNN
        if (BackwardKernelMKL(srcGrad, in, kernelGrad, accumulateGradient, workspace)) return;
#else
        UNUSED(accumulateGradient);
#endif

        size_t batchSize = srcGrad.GetNumCols();
        size_t subBatchSize = m_maxTempMemSizeInSamples == 0 ? batchSize : min(batchSize, m_maxTempMemSizeInSamples);

        const auto& inT = m_geometry->InputShape();
        const auto& kernT = m_geometry->KernelShape();
        const auto& outT = m_geometry->OutputShape();

        size_t dimCount = inT.GetRank();
        size_t mapOutCount = m_geometry->GetMapCount(dimCount - 1);
        size_t mapOutSize = outT.GetNumElements() / mapOutCount;

        assert(kernT[dimCount - 1] == inT[dimCount - 1]);
        if (kernT[dimCount - 1] != inT[dimCount - 1])
        {
            RuntimeError("GEMM convolution engine does not support this convolution configuration. "
                         "It is possible to make GEMM engine work with this configuration by defining "
                         "input/output/kernel using tensors of higher(+1) dimension. Geometry: %s", ((string)*m_geometry).c_str());
        }

        size_t unrollRows = kernT.GetNumElements();
        size_t unrollCols = mapOutSize * subBatchSize;

        // Reserve space for:
        // 1. Unrolled inputs.
        // 2. Transposed source gradients (optional).
        workspace.Resize(unrollCols, unrollRows + (subBatchSize > 1 ? mapOutCount : 0));

        for (size_t start = 0; start < batchSize; start += subBatchSize)
        {
            size_t curBatchSize = min(subBatchSize, batchSize - start);
            // 1. Transpose and reshape srcGrad.
            auto srcGradSlice = srcGrad.ColumnSlice(start, curBatchSize);
            if (curBatchSize > 1)
            {
                auto srcGradTranSlice = workspace.ColumnSlice(unrollRows, mapOutCount);
                if (curBatchSize != subBatchSize)
                {
                    srcGradTranSlice.Reshape(mapOutCount * mapOutSize, subBatchSize);
                    srcGradTranSlice = srcGradTranSlice.ColumnSlice(0, curBatchSize);
                }
                // Reshape to transposed shape - required by AssignTransposeOf.
                srcGradTranSlice.Reshape(srcGradSlice.GetNumCols(), srcGradSlice.GetNumRows());
                srcGradTranSlice.AssignTransposeOf(srcGradSlice);
                srcGradSlice = srcGradTranSlice.ColumnSlice(0, srcGradTranSlice.GetNumCols());
            }
            srcGradSlice.Reshape(mapOutSize * curBatchSize, mapOutCount);

            // 2. Unroll inputs.
            auto inputSlice = in.ColumnSlice(start, curBatchSize);
            auto unrolledInputSlice = workspace.ColumnSlice(0, unrollRows);
            if (curBatchSize != subBatchSize)
            {
                unrolledInputSlice.Reshape(mapOutSize * unrollRows, subBatchSize);
                unrolledInputSlice = unrolledInputSlice.ColumnSlice(0, curBatchSize);
            }
            unrolledInputSlice.Reshape(mapOutSize * curBatchSize, unrollRows);
            unrolledInputSlice.SetValue(0);
            inputSlice.UnrollConvolutionInputForKernelBackprop(mapOutSize, m_mpRowCol, *m_mpRowRun, *m_runs, unrolledInputSlice);

            // cudnn layout uses row-major kernel weight matrix.
            auto kernGrad = kernelGrad.ColumnSlice(0, kernelGrad.GetNumCols());
            kernGrad.Reshape(unrollRows, kernGrad.GetNumElements() / unrollRows);
            // 3. Multiply.
            Mat::MultiplyAndAdd(unrolledInputSlice, true, srcGradSlice, false, kernGrad);
        }
    }

#ifdef USE_MKL2017DNN
    class MKLConvolutionContext
    {
    public:
        enum ContextIndex
        {
            ContextIndex_Forward = 0,
            ContextIndex_BackwardData,
            ContextIndex_BackwardFilter,
            ContextIndex_Total
        };

    private:
        const ConvolveGeometry* m_prevGeometry = nullptr;
        size_t m_prevBatchSize = 0;
        int m_contextFlags = 0;

        // fixed dimension for MKL for now
        static const int m_dimension = 4;

        static const int NumInputs = 2;

        struct PrimitiveContext
        {
            MKLDnnResourceAdapter<ElemType> inputs[NumInputs];
            MKLDnnResourceAdapter<ElemType> output;

            dnnPrimitive_t primitive = nullptr;
            dnnPrimitiveAttributes_t attributes = nullptr;

            void Clear()
            {
                if (primitive) { dnnDelete<ElemType>(primitive); primitive = nullptr; }
                for (auto& i : inputs) i.Clear();
                output.Clear();
                if (attributes) { dnnPrimitiveAttributesDestroy<ElemType>(attributes); attributes = nullptr; }
            }

            ~PrimitiveContext()
            {
                Clear();
            }
        } m_context[ContextIndex_Total];

        static void GetSizesAndStrides(const TensorShape& shape, size_t lastDim, SmallVector<size_t>& sizes, SmallVector<size_t>& strides, size_t mapCount = 0)
        {
            GetSizesAndStridesWithGroups(shape, lastDim, sizes, strides, 1, mapCount);
        }

        static void GetSizesAndStridesWithGroups(const TensorShape& shape, size_t lastDim, SmallVector<size_t>& sizes, SmallVector<size_t>& strides, size_t groups=1, size_t mapCount = 0)
        {
            UNUSED(groups);
            sizes = shape.GetDims();
            if (mapCount)
            {
                if (mapCount != shape.GetDim(shape.GetRank() - 1))
                    RuntimeError("Mismatching outputShape and mapCount");

                // for outputShape, pad 1 before mapCount (the last dim in shape)
                sizes.pop_back();
                while (sizes.size() < m_dimension - 2) sizes.push_back(1);
                sizes.push_back(mapCount);
            }
            else
            {
                while (sizes.size() < m_dimension - 1) sizes.push_back(1);
            }
            assert(lastDim % groups == 0);
            sizes.push_back(lastDim / groups);
            strides.clear();
            strides.push_back(1);
            for (int i = 1; i <= sizes.size(); i++)
            {
                strides.push_back(sizes[i - 1] * strides[i - 1]);
            }
            if (groups > 1) sizes.push_back(groups);
        }

        static void GetInputOffsets(const ConvolveGeometry* geometry, SmallVector<int>& inputOffset)
        {
            size_t dim_size = geometry->InputShape().GetRank();
            for (size_t i = 0; i < dim_size; i++)
            {
                inputOffset.push_back(-geometry->GetLowerPad(i));
            }
        }

    public:
        MKLConvolutionContext() :
            m_prevBatchSize(0),
            m_prevGeometry(nullptr)
        {}

        bool Supported(const ConvolveGeometry* geometry, bool forward)
        {
#ifdef USE_MKLDNN
            //TODO: test code for linking with mkldnn.dll, will extend to support dilated convolution with MKL-DNN later
            mkldnn_primitive_attr_t attr;
            mkldnn_primitive_attr_create(&attr);
            mkldnn_primitive_attr_destroy(attr);
#endif

            //MKL2017 does not support asymmetric padding yet
            if (geometry->IsAsymmetricPadding(/*useMKL=*/true)) {
                fprintf(stderr, "WARNING: Detected asymmetric padding issue with lowerPad != higherPad, not supported by MKL. Switching to GEMM convolution engine. \n");
                return false;
            }

            //MKL-DNN calls does not support 4th dimention now, we will update the code once MKL release the update.
            return forward ? (geometry->InputShape().GetRank() < m_dimension) : (geometry->OutputShape().GetRank() < m_dimension);
        }

        void Prepare(size_t batchSize, const ConvolveGeometry* geometry, ContextIndex contextIndex)
        {
            int flag = (1 << contextIndex);
            bool sameGeometryAndBatchSize = (geometry == m_prevGeometry && batchSize == m_prevBatchSize);
            if (sameGeometryAndBatchSize && !!(m_contextFlags & flag)) return;

            if (!sameGeometryAndBatchSize)
                m_contextFlags = 0;

            if (m_contextFlags)
            {
                if (m_prevGeometry != geometry || m_prevBatchSize != batchSize)
                    RuntimeError("Inconsistent convolution geometry or batch size between forward and backward");
            }
            else
            {
                m_prevGeometry = geometry;
                m_prevBatchSize = batchSize;
            }
            m_contextFlags |= flag;

            size_t mapCount = geometry->GetMapCount(geometry->KernelShape().GetRank() - 1);

            SmallVector<size_t> outputSize, outputStrides, filterSize, filterStrides, inputSize,  inputStrides;
            SmallVector<int>    inputOffset;

            GetSizesAndStrides(geometry->OutputShape(), batchSize, outputSize, outputStrides, mapCount);
            GetSizesAndStridesWithGroups(geometry->KernelShape(), mapCount, filterSize, filterStrides, geometry->Groups());
            GetSizesAndStrides(geometry->InputShape(), batchSize, inputSize, inputStrides);
            GetInputOffsets(geometry, inputOffset);

            const auto& convolutionStride = geometry->Stride().GetDims();
            const int filter_dimension = m_dimension + (geometry->Groups() > 1 ? 1 : 0); // Or we could do filterSize.size() also.

            auto& ctx = m_context[contextIndex];
            ctx.Clear();

            dnnLayout_t ltUserInputs[NumInputs], ltPrimInputs[NumInputs];
            dnnLayout_t ltUserOutput, ltPrimOutput;
            dnnResourceType_t inputTypes[NumInputs];
            dnnResourceType_t outputType;
            switch (contextIndex)
            {
            case ContextIndex_Forward:
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserInputs[0], m_dimension, inputSize.begin(), inputStrides.begin()));
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserInputs[1], filter_dimension, filterSize.begin(), filterStrides.begin()));
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserOutput, m_dimension, outputSize.begin(), outputStrides.begin()));
                CHECK_MKL(dnnPrimitiveAttributesCreate<ElemType>(&ctx.attributes));
                if(geometry->Groups() > 1)
                    CHECK_MKL(dnnGroupsConvolutionCreateForward<ElemType>(&ctx.primitive, ctx.attributes, dnnAlgorithmConvolutionDirect, geometry->Groups(), m_dimension, inputSize.begin(), outputSize.begin(), filterSize.begin(), convolutionStride.begin(), inputOffset.begin(), dnnBorderZeros));
                else
                    CHECK_MKL(dnnConvolutionCreateForward<ElemType>(&ctx.primitive, ctx.attributes, dnnAlgorithmConvolutionDirect, m_dimension, inputSize.begin(), outputSize.begin(), filterSize.begin(), convolutionStride.begin(), inputOffset.begin(), dnnBorderZeros));
                inputTypes[0] = dnnResourceSrc;
                inputTypes[1] = dnnResourceFilter;
                outputType = dnnResourceDst;
                break;
            case ContextIndex_BackwardData:
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserInputs[0], m_dimension, outputSize.begin(), outputStrides.begin()));
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserInputs[1], filter_dimension, filterSize.begin(), filterStrides.begin()));
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserOutput, m_dimension, inputSize.begin(), inputStrides.begin()));
                CHECK_MKL(dnnPrimitiveAttributesCreate<ElemType>(&ctx.attributes));
                if (geometry->Groups() > 1)
                    CHECK_MKL(dnnGroupsConvolutionCreateBackwardData<ElemType>(&ctx.primitive, ctx.attributes, dnnAlgorithmConvolutionDirect, geometry->Groups(), m_dimension, inputSize.begin(), outputSize.begin(), filterSize.begin(), convolutionStride.begin(), inputOffset.begin(), dnnBorderZeros));
                else
                    CHECK_MKL(dnnConvolutionCreateBackwardData<ElemType>(&ctx.primitive, ctx.attributes, dnnAlgorithmConvolutionDirect, m_dimension, inputSize.begin(), outputSize.begin(), filterSize.begin(), convolutionStride.begin(), inputOffset.begin(), dnnBorderZeros));
                inputTypes[0] = dnnResourceDiffDst;
                inputTypes[1] = dnnResourceFilter;
                outputType = dnnResourceDiffSrc;
                break;
            case ContextIndex_BackwardFilter:
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserInputs[0], m_dimension, outputSize.begin(), outputStrides.begin()));
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserInputs[1], m_dimension, inputSize.begin(), inputStrides.begin()));
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserOutput, filter_dimension, filterSize.begin(), filterStrides.begin()));
                CHECK_MKL(dnnPrimitiveAttributesCreate<ElemType>(&ctx.attributes));
                if (geometry->Groups() > 1)
                    CHECK_MKL(dnnGroupsConvolutionCreateBackwardFilter<ElemType>(&ctx.primitive, ctx.attributes, dnnAlgorithmConvolutionDirect, geometry->Groups(), m_dimension, inputSize.begin(), outputSize.begin(), filterSize.begin(), convolutionStride.begin(), inputOffset.begin(), dnnBorderZeros));
                else
                    CHECK_MKL(dnnConvolutionCreateBackwardFilter<ElemType>(&ctx.primitive, ctx.attributes, dnnAlgorithmConvolutionDirect, m_dimension, inputSize.begin(), outputSize.begin(), filterSize.begin(), convolutionStride.begin(), inputOffset.begin(), dnnBorderZeros));
                inputTypes[0] = dnnResourceDiffDst;
                inputTypes[1] = dnnResourceSrc;
                outputType = dnnResourceDiffFilter;
                break;
            default:
                RuntimeError("Unexpected context type %d", (int)contextIndex);
            }

            for (int i = 0; i < NumInputs; i++)
            {
                CHECK_MKL(dnnLayoutCreateFromPrimitive<ElemType>(&ltPrimInputs[i], ctx.primitive, inputTypes[i]));
                ctx.inputs[i].Create(ltUserInputs[i], ltPrimInputs[i], inputTypes[i], true);
            }

            CHECK_MKL(dnnLayoutCreateFromPrimitive<ElemType>(&ltPrimOutput, ctx.primitive, outputType));
            ctx.output.Create(ltUserOutput, ltPrimOutput, outputType, false);
        }

        void Execute(void* userInput0, void* userInput1, void* userOutput, ContextIndex contextIndex)
        {
            auto& ctx = m_context[contextIndex];
            void* userInputs[] = { userInput0, userInput1 };
            void* resources[dnnResourceNumber] = { 0 };

            for(int i = 0; i < NumInputs; i++)
                ctx.inputs[i].PrepareForExecution(userInputs[i], resources);

            ctx.output.PrepareForExecution(userOutput, resources);

            CHECK_MKL(dnnExecute<ElemType>(ctx.primitive, resources));

            ctx.output.ConvertOutput(userOutput);
        }
    };

    MKLConvolutionContext m_mklContext;

    // convolution implementation with MKL 2017 DNN functions
    bool ForwardCoreMKL(const Mat& in, const Mat& kernel, Mat& out)
    {
        if (!m_mklContext.Supported(m_geometry.get(), true)) return false;
        
        m_mklContext.Prepare(in.GetNumCols(), m_geometry.get(), MKLConvolutionContext::ContextIndex_Forward);
        m_mklContext.Execute(in.Data(), kernel.Data(), out.Data(), MKLConvolutionContext::ContextIndex_Forward);

        return true;
    }

    bool BackwardDataMKL(const Mat& srcGrad, const Mat& kernel, Mat& grad, bool accumulateGradient, Mat& workspace)
    {
        if (!m_mklContext.Supported(m_geometry.get(), false)) return false;

        m_mklContext.Prepare(srcGrad.GetNumCols(), m_geometry.get(), MKLConvolutionContext::ContextIndex_BackwardData);

        if (accumulateGradient)
            workspace.AssignValuesOf(grad);

        m_mklContext.Execute(srcGrad.Data(), kernel.Data(), grad.Data(), MKLConvolutionContext::ContextIndex_BackwardData);

        if (accumulateGradient)
            grad.AssignSumOf(grad, workspace);

        return true;
    }

    bool BackwardKernelMKL(const Mat& srcGrad, const Mat& in, Mat& kernelGrad, bool accumulateGradient, Mat& workspace)
    {
        if (!m_mklContext.Supported(m_geometry.get(), false)) return false;

        m_mklContext.Prepare(srcGrad.GetNumCols(), m_geometry.get(), MKLConvolutionContext::ContextIndex_BackwardFilter);

        if (accumulateGradient)
            workspace.AssignValuesOf(kernelGrad);

        m_mklContext.Execute(srcGrad.Data(), in.Data(), kernelGrad.Data(), MKLConvolutionContext::ContextIndex_BackwardFilter);

        if (accumulateGradient)
            kernelGrad.AssignSumOf(kernelGrad, workspace);

        return true;
    }

#endif

public:
    static bool IsSupported(DEVICEID_TYPE deviceId, ConvolveGeometryPtr geometry)
    {
        return deviceId < 0 &&
               find(begin(geometry->Sharing()), end(geometry->Sharing()), false) == end(geometry->Sharing());
    }

    static bool IsMklEnabled()
    {
#ifdef USE_MKL2017DNN
        return true;
#else
        return false;
#endif
    }
};

template <class ElemType>
std::unique_ptr<ConvolutionEngine<ElemType>> ConvolutionEngine<ElemType>::Create(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId,
                                                                                 ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind,
                                                                                 ConvolutionEngineKind enabledEngines, std::wstring logPrefix,
                                                                                 bool forceDeterministicAlgorithms, bool poolIncludePad,
                                                                                 bool inputHasFreeDimension)
{
    if (!logPrefix.empty())
        logPrefix += L": ";

    auto isEnabled = [=](ConvolutionEngineKind eng) { return ((int)enabledEngines & (int)eng) != 0; };
    // Note: in some cases do not throw exception even if parameters do not match as Create
    // can be called from places like MEL with default parameters and never be used.
    // The check will be done later in engine's EnsureCompatible call if the egnine is actually used.
    auto engStr = (std::string)(*geometry);
    // Only legacy engine supports HWC layout.
    if (imageLayout == ImageLayoutKind::HWC)
    {
        if (!isEnabled(ConvolutionEngineKind::Legacy))
            RuntimeError("Trying to use Legacy convolution engine when it's disabled.");

        if (GetMathLibTraceLevel() > 0)
            fprintf(stderr, "%lsusing legacy convolution engine for geometry: %s.\n", logPrefix.c_str(), engStr.c_str());

        return std::make_unique<LegacyConvolutionEngine<ElemType>>(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, poolIncludePad);
    }

    // Check if we can use cuDNN engine. Do not need to validate tensors as ConvolveGeometry has already done that.
    if (isEnabled(ConvolutionEngineKind::CuDnn) &&
        CuDnnConvolutionEngineFactory<ElemType>::IsSupported(deviceId, geometry, poolKind))
    {
        if (GetMathLibTraceLevel() > 0)
            fprintf(stderr, "%lsusing cuDNN convolution engine for geometry: %s.\n", logPrefix.c_str(), engStr.c_str());

        return CuDnnConvolutionEngineFactory<ElemType>::Create(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind,
                                                               forceDeterministicAlgorithms, poolIncludePad, inputHasFreeDimension);
    }

    if (geometry->Groups() == 1)
    {
        if (isEnabled(ConvolutionEngineKind::Gemm) && GemmConvolutionEngine<ElemType>::IsSupported(deviceId, geometry))
        {
            if (GetMathLibTraceLevel() > 0)
                fprintf(stderr, "%lsusing GEMM convolution engine for geometry: %s.\n", logPrefix.c_str(), engStr.c_str());

            return std::make_unique<GemmConvolutionEngine<ElemType>>(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, poolIncludePad);
        }

        if (!isEnabled(ConvolutionEngineKind::Reference))
            RuntimeError("Reference convolution is disabled and no other engine supports such configuration (or disabled).");

        if (GetMathLibTraceLevel() > 0)
            fprintf(stderr, "%lsusing reference convolution engine for geometry, could be VERY SLOW: %s.\n", logPrefix.c_str(), engStr.c_str());

        return std::make_unique<ReferenceConvolutionEngine<ElemType>>(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, poolIncludePad);
    }
    else if (geometry->Groups() > 1)
    {
        if (!(geometry->InputShape().GetRank() < 4))
        {
            RuntimeError("Group convolution, i.e. groups > 1, for 3-dimensional convolution or higher is not supported on the CPU. Please use GPU, if possible.");
        }
        // For group convolution, MKL 2017 is required. If it is not enabled, we throw an error.
        if (GemmConvolutionEngine<ElemType>::IsMklEnabled())
        {
            if (isEnabled(ConvolutionEngineKind::Gemm) && GemmConvolutionEngine<ElemType>::IsSupported(deviceId, geometry))
            {
                if (GetMathLibTraceLevel() > 0)
                    fprintf(stderr, "%lsusing GEMM convolution engine for geometry: %s.\n", logPrefix.c_str(), engStr.c_str());

                return std::make_unique<GemmConvolutionEngine<ElemType>>(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, poolIncludePad);
            }
            RuntimeError("Gemm convolution is not supported/enabled. Cannot execute group convolution (groups > 1) on CPU.");
        }
        RuntimeError("MKL 2017 not enabled. For group convolution (groups > 1) on CPU, MKL 2017 is required. Please install/enable MKL 2017.");
    }
    else
        LogicError("Invalid value for 'groups' parameter for convolution: groups must be greater than or equal to 1.");
}

// only GPU supports fp16 convolution
template <>
std::unique_ptr<ConvolutionEngine<half>> ConvolutionEngine<half>::Create(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId,
    ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind,
    ConvolutionEngineKind enabledEngines, std::wstring logPrefix,
    bool forceDeterministicAlgorithms, bool poolIncludePad,
    bool inputHasFreeDimension)
{
    if (!logPrefix.empty())
        logPrefix += L": ";

    auto isEnabled = [=](ConvolutionEngineKind eng) { return ((int)enabledEngines & (int)eng) != 0; };
    // Note: in some cases do not throw exception even if parameters do not match as Create
    // can be called from places like MEL with default parameters and never be used.
    // The check will be done later in engine's EnsureCompatible call if the egnine is actually used.
    auto engStr = (std::string)(*geometry);

    // Check if we can use cuDNN engine. Do not need to validate tensors as ConvolveGeometry has already done that.
    if (isEnabled(ConvolutionEngineKind::CuDnn) &&
        CuDnnConvolutionEngineFactory<half>::IsSupported(deviceId, geometry, poolKind))
    {
        if (GetMathLibTraceLevel() > 0)
            fprintf(stderr, "%lsusing cuDNN convolution engine for geometry: %s.\n", logPrefix.c_str(), engStr.c_str());

        return CuDnnConvolutionEngineFactory<half>::Create(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind,
            forceDeterministicAlgorithms, poolIncludePad, inputHasFreeDimension);
    }

    RuntimeError("FP16 convolution is only supported via cuDNN.");

    return nullptr;
}

template class ConvolutionEngine<float>;
template class ConvolutionEngine<double>;
template class ConvolutionEngine<half>;

}}}
