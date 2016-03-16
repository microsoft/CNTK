//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "ConvolutionEngine.h"
#include "CuDnnFactories.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void ConvolutionEngine<ElemType>::Forward(const Mat& in, const Mat& filter, Mat& out, Mat& workspace)
{
    const auto& g = *m_geometry;
    assert(g.InputShape().GetNumElements() == in.GetNumRows());
    assert(g.OutputShape().GetNumElements() == out.GetNumRows());
    size_t batchSize = in.GetNumCols();
    assert(batchSize == out.GetNumCols());
    // REVIEW alexeyk: add shape-aware asserts?
    assert(g.KernelShape().GetNumElements() * g.MapCount().GetNumElements() == filter.GetNumElements());
#ifdef NDEBUG
    UNUSED(g);
    UNUSED(batchSize);
#endif

    EnsureCompatible();
    EnsureConvolutionInitialized();
    ForwardCore(in, filter, out, workspace);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::BackwardData(const Mat& srcGrad, const Mat& filter, Mat& grad, Mat& workspace)
{
    const auto& g = *m_geometry;
    assert(g.InputShape().GetNumElements() == grad.GetNumRows());
    assert(g.OutputShape().GetNumElements() == srcGrad.GetNumRows());
    size_t batchSize = srcGrad.GetNumCols();
    assert(batchSize == grad.GetNumCols());
    assert(g.KernelShape().GetNumElements() * g.MapCount().GetNumElements() == filter.GetNumElements());
#ifdef NDEBUG
    UNUSED(g);
    UNUSED(batchSize);
#endif

    EnsureCompatible();
    EnsureConvolutionInitialized();
    BackwardDataCore(srcGrad, filter, grad, workspace);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::BackwardFilter(const Mat& srcGrad, const Mat& in, Mat& filter, bool allowReuse, Mat& workspace)
{
    const auto& g = *m_geometry;
    assert(g.InputShape().GetNumElements() == in.GetNumRows());
    assert(g.OutputShape().GetNumElements() == srcGrad.GetNumRows());
    size_t batchSize = in.GetNumCols();
    assert(batchSize == srcGrad.GetNumCols());
    assert(g.KernelShape().GetNumElements() * g.MapCount().GetNumElements() == filter.GetNumElements());
#ifdef NDEBUG
    UNUSED(g);
    UNUSED(batchSize);
#endif

    EnsureCompatible();
    EnsureConvolutionInitialized();
    BackwardFilterCore(srcGrad, in, filter, allowReuse, workspace);
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
void ConvolutionEngine<ElemType>::BackwardPooling(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad)
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
    BackwardPoolingCore(out, srcGrad, in, grad);
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
    ReferenceConvolutionEngine(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind)
        : Base(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind), 
        m_mpRowCol(geometry->MpRowCol().size(), 1, const_cast<int*>(geometry->MpRowCol().data()), deviceId, IsGpu(deviceId) ? matrixFlagNormal : matrixFlagDontOwnBuffer)
    {
    }

protected:
    using Base::m_geometry;
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_maxTempMemSizeInSamples;
    using Base::m_poolKind;

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

    void ForwardCore(const Mat& in, const Mat& filter, Mat& out, Mat& /*workspace*/) override
    {
        in.NDConvolutionForward(filter, m_mpRowCol, *m_mpRowIwht, *m_mpRowRun, *m_runs, out);
    }

    void BackwardDataCore(const Mat& srcGrad, const Mat& filter, Mat& grad, Mat& /*workspace*/) override
    {
        srcGrad.NDConvolutionBackwardData(filter, m_mpRowCol, *m_mpRowIwht, *m_mpRowRun, *m_runs, grad);
    }

    void BackwardFilterCore(const Mat& srcGrad, const Mat& in, Mat& filterGrad, bool /*allowReuse*/, Mat& /*workspace*/) override
    {
        srcGrad.NDConvolutionBackwardFilter(in, m_mpRowCol, *m_mpRowIwht, *m_mpRowRun, *m_runs, filterGrad);
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
            in.NDMaxPoolingForward(m_mpRowCol, *m_mpRowIndices, *m_indices, out);
        }
        else if (m_poolKind == PoolKind::Average)
        {
            in.NDAveragePoolingForward(m_mpRowCol, *m_mpRowIndices, *m_indices, out);
        }
        else
            InvalidArgument("Pooling type %d is not supported.", (int)m_poolKind);

    }

    void BackwardPoolingCore(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad) override
    {
        if (m_poolKind == PoolKind::Max)
        {
            srcGrad.NDMaxPoolingBackward(out, in, m_mpRowCol, *m_mpRowIndices, *m_indices, grad);
        }
        else if (m_poolKind == PoolKind::Average)
        {
            srcGrad.NDAveragePoolingBackward(m_mpRowCol, *m_mpRowIndices, *m_indices, grad);
        }
        else
            InvalidArgument("Pooling type %d is not supported.", (int)m_poolKind);
    }

private:
    static bool IsGpu(DEVICEID_TYPE deviceId)
    {
        return deviceId >= 0;
    }

private:
    using IntMatPtr = std::unique_ptr<Matrix<int>>;

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
    LegacyConvolutionEngine(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind)
        : Base(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind), 
        m_inT(m_geometry->InputShape(), ImageLayoutKind::CHW), m_outT(m_geometry->OutputShape(), ImageLayoutKind::CHW),
        m_filterT(m_geometry->KernelShape(), ImageLayoutKind::CHW), m_strideT(m_geometry->Stride(), ImageLayoutKind::CHW)
    {
        // Legacy engine uses a non-standard formula to compute padding (it may "overpad")
        // so default auto-padding of ConvolveGeometry does not work here and instead
        // precise padding is used.
        const auto& lowerPad = m_geometry->LowerPad();
        const auto& upperPad = m_geometry->UpperPad();
        m_padding = (m_geometry->AutoPad().size() == 0 || !m_geometry->AutoPad()[0]) &&
                     (lowerPad.size() == 1 && upperPad.size() == 1 && lowerPad[0] > 0);
    }

protected:
    using Base::m_geometry;
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_maxTempMemSizeInSamples;
    using Base::m_poolKind;

    void EnsureCompatible() override
    {
        if (m_imageLayout != ImageLayoutKind::HWC)
            RuntimeError("Legacy convolution engine supports only HWC/legacy layout.");

        const auto& autoPad = m_geometry->AutoPad();
        if (autoPad.size() > 0 && autoPad[0])
            RuntimeError("Legacy convolution engine supports precise padding via LowerPad/UpperPad parameters only.");
    }

    void EnsureConvolutionInitialized() override
    {
    }

    void ForwardCore(const Mat& in, const Mat& filter, Mat& out, Mat& workspace) override
    {
        size_t batchSize = in.GetNumCols();
        size_t packedInputRows = m_filterT.w() * m_filterT.h() * m_filterT.c();
        size_t packedInputColsPerSample = m_outT.w() * m_outT.h();
        size_t outputSizePerChannel = packedInputColsPerSample;
        // size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
        // size_t inputDim = inT.w() * inT.h() * inT.c();  // size of each input sample

        size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

        assert(filter.GetNumCols() == packedInputRows && filter.GetNumRows() == m_outT.c());
        UNUSED(packedInputRows);

        // GPU and 1-dimensional image
        m_gpuSparseOpt = (m_filterT.h() == 1 &&
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
                inputSubBatch.SetValue(in.ColumnSlice(startSampleId, smallBatchSize), in.GetFormat());

            if (m_gpuSparseOpt)
            {
                if (m_filterT.w() * m_inT.c() != filter.GetNumCols())
                    LogicError("Kernel width and weight matrix dimensions don't match.");

                inputSubBatch.Reshape(m_inT.c() * m_inT.w(), m_inT.h() * smallBatchSize);
                Mat outputSubBatch = out.ColumnSlice(startSampleId, m_outT.h() * smallBatchSize);
                Mat::ConvolveAndWeightedAdd(1, filter, false, inputSubBatch, false, 0, outputSubBatch,
                                            static_cast<int>(m_inT.c()), m_strideT.w(), m_padding, true);
            }
            else
            {
                inputSubBatch.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, true);
                workspace.AssignPackedConvolutionInput(inputSubBatch,
                                                       m_inT.w(), m_inT.h(), m_inT.c(),
                                                       m_outT.w(), m_outT.h(), m_outT.c(),
                                                       m_filterT.w(), m_filterT.h(), m_strideT.w(), m_strideT.h(),
                                                       m_padding);

                Mat outputSubBatch = out.ColumnSlice(outputSizePerChannel * startSampleId, outputSizePerChannel * smallBatchSize);

                // workspace.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                // BUGBUG: This ^^ destroys the content of the matrix. Also it seems not to change the size. Does it? Should this be a Reshape()?
                Mat::Multiply(filter, false, workspace, false, outputSubBatch);
            }
        }

        out.Reshape(m_outT.c() * outputSizePerChannel, batchSize); // each sample becomes a column

        assert(m_outT.w() * m_outT.h() * m_outT.c() == out.GetNumRows());
        assert(batchSize == out.GetNumCols());
    }

    void BackwardDataCore(const Mat& srcGrad, const Mat& filter, Mat& grad, Mat& workspace) override
    {
        size_t batchSize = srcGrad.GetNumCols();
        size_t packedInputRows = m_filterT.w() * m_filterT.h() * m_filterT.c();
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
            Matrix<ElemType>::Multiply(filter, true, outputGradientSubBatch, false, workspace);

            Matrix<ElemType> inputGradientSubBatch = grad.ColumnSlice(startSampleId, smallBatchSize);
            workspace.UnpackConvolutionInput(inputGradientSubBatch,
                                             m_inT.w(), m_inT.h(), m_inT.c(),
                                             m_outT.w(), m_outT.h(), m_outT.c(),
                                             m_filterT.w(), m_filterT.h(), m_strideT.w(), m_strideT.h(),
                                             m_padding);
        }

        assert(m_outT.w() * m_outT.h() * m_outT.c() == srcGrad.GetNumRows());
        assert(batchSize == srcGrad.GetNumCols());
    }

    void BackwardFilterCore(const Mat& srcGrad, const Mat& in, Mat& filterGrad, bool allowReuse, Mat& workspace) override
    {
        size_t batchSize = in.GetNumCols();
        size_t packedInputRows = m_filterT.w() * m_filterT.h() * m_filterT.c();
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
            Matrix<ElemType>::MultiplyAndAdd(srcGradTmp, false, workspace, true, filterGrad);
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

                    filterGrad.Reshape(m_outT.c() * m_filterT.w(), m_inT.c());
                    Matrix<ElemType>::ConvolveAndWeightedAdd(1, outputGradientSubBatchReordered, true, inputSubBatchSparseReordered, false, 1, filterGrad, smallBatchSize * m_inT.h(), m_strideT.w(), m_padding, false);
                    filterGrad.Reshape(m_outT.c(), m_inT.c() * m_filterT.w());
                }
                else
                {
                    workspace.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                    Matrix<ElemType> inputSubBatch = in.ColumnSlice(startSampleID, smallBatchSize);
                    inputSubBatch.SwitchToMatrixType(MatrixType::DENSE, inputSubBatch.GetFormat(), true);
                    workspace.AssignPackedConvolutionInput(inputSubBatch,
                                                           m_inT.w(), m_inT.h(), m_inT.c(),
                                                           m_outT.w(), m_outT.h(), m_outT.c(),
                                                           m_filterT.w(), m_filterT.h(), m_strideT.w(), m_strideT.h(),
                                                           m_padding);

                    Matrix<ElemType>::MultiplyAndAdd(outputGradientSubBatch, false, workspace, true, filterGrad);
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
                                       m_filterT.w(), m_filterT.h(), m_strideT.w(), m_strideT.h());
        }
        else if (m_poolKind == PoolKind::Average)
        {
            out.AssignAveragePoolingResult(in, m_inT.c(), m_inT.w(), m_inT.h(), m_inT.w() * m_inT.h() * m_inT.c(),
                                           m_outT.w(), m_outT.h(), m_outT.w() * m_outT.h() * m_outT.c(),
                                           m_filterT.w(), m_filterT.h(), m_strideT.w(), m_strideT.h());
        }
        else
            InvalidArgument("Pooling type %d is not supported.", (int)m_poolKind);
    }

    void BackwardPoolingCore(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad) override
    {
        if (m_poolKind == PoolKind::Max)
        {
            grad.AddMaxPoolingGradient(srcGrad, in, out,
                                       m_inT.c(), m_inT.w(), m_inT.h(), m_inT.w() * m_inT.h() * m_inT.c(),
                                       m_outT.w(), m_outT.h(), m_outT.w() * m_outT.h() * m_outT.c(),
                                       m_filterT.w(), m_filterT.h(), m_strideT.w(), m_strideT.h());
        }
        else if (m_poolKind == PoolKind::Average)
        {
            grad.AddAveragePoolingGradient(srcGrad, m_inT.c(), m_inT.w(), m_inT.h(), m_inT.w() * m_inT.h() * m_inT.c(),
                                           m_outT.w(), m_outT.h(), m_outT.w() * m_outT.h() * m_outT.c(),
                                           m_filterT.w(), m_filterT.h(), m_strideT.w(), m_strideT.h());
        }
        else
            InvalidArgument("Pooling type %d is not supported.", (int)m_poolKind);
    }

private:
    ImageDimensions m_inT;
    ImageDimensions m_outT;
    ImageDimensions m_filterT;
    ImageDimensions m_strideT;
    bool m_padding;

    bool m_gpuSparseOpt;
    bool m_gpuSparse1D;
};

template <class ElemType>
std::unique_ptr<ConvolutionEngine<ElemType>> ConvolutionEngine<ElemType>::Create(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId,
                                                                                 ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind,
                                                                                 ConvolutionEngineKind enabledEngines)
{
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
        // REVIEW alexeyk: should honor m_traceLevel here.
        fprintf(stderr, "\nUsing legacy convolution engine for geometry %s.\n", engStr.c_str());
        return std::make_unique<LegacyConvolutionEngine<ElemType>>(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind);
    }

    // Check if we can use cuDNN engine. Do not need to validate tensors as ConvolveGeometry has already done that.
    if (isEnabled(ConvolutionEngineKind::CuDnn) &&
        CuDnnConvolutionEngineFactory<ElemType>::IsSupported(geometry, poolKind))
    {
        fprintf(stderr, "\nUsing cuDNN convolution engine for geometry %s.\n", engStr.c_str());
        return CuDnnConvolutionEngineFactory<ElemType>::Create(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind);
    }

    if (!isEnabled(ConvolutionEngineKind::Reference))
        RuntimeError("Reference convolution is disabled and no other engine supports such configuratin (or disabled).");
    fprintf(stderr, "\nUsing reference convolution engine for geometry %s.\n", engStr.c_str());
    return std::make_unique<ReferenceConvolutionEngine<ElemType>>(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind);
}


template class ConvolutionEngine<float>;
template class ConvolutionEngine<double>;

}}}
