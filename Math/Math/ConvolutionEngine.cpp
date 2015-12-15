//
// <copyright file="ConvolutionEngine.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "ConvolutionEngine.h"
#include "CuDnnConvolutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class DefaultConvolutionEngine : public ConvolutionEngine<ElemType>
    {
    public:
    	using Base = ConvolutionEngine<ElemType>;
        using typename Base::Mat;
        using typename Base::Tensor4D;
        using typename Base::Filter;
        using typename Base::ConvDesc;

    public:
        DefaultConvolutionEngine(DEVICEID_TYPE deviceId, size_t maxTempMemSizeInSamples)
            : m_ones(deviceId), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples)
        {
        }

    public:
        void Forward(const Tensor4D& inT, const Mat& in, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
            const Tensor4D& outT, Mat& out, Mat& workspace) override
        {
            assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
            assert(inT.n() == in.GetNumCols());
            assert(filterT.k() == filter.GetNumRows());
            assert(filterT.w() * filterT.h() * filterT.c() == filter.GetNumCols());
            assert(inT.c() == filterT.c());
            assert(outT.c() == filterT.k());

            size_t packedInputRows = filterT.w() * filterT.h() * filterT.c();
            size_t packedInputColsPerSample = outT.w() * outT.h();
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = inT.w() * inT.h() * inT.c();  //size of each input sample

            size_t batchSize = inT.n();
            size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

            assert(filter.GetNumCols() == packedInputRows && filter.GetNumRows() == outT.c());

            // GPU and 1-dimensional image
            bool gpuSparse1D = (inT.h() == 1 &&
                in.GetCurrentMatrixLocation() == CurrentDataLocation::GPU &&
                in.GetMatrixType() == MatrixType::SPARSE);

            out.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, false);

            // Reshaping is only necessary if we are going to use the unpacking trick
            if (!gpuSparse1D)
                out.Reshape(outT.c(), outputSizePerChannel * batchSize);

            size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples);
            size_t numSubBatches = (batchSize + subBatchSize - 1) / subBatchSize;

            for (size_t i = 0; i < numSubBatches; i++)
            {
                size_t startSampleId = i * subBatchSize;
                size_t endSampleId = min(batchSize, startSampleId + subBatchSize);
                size_t smallBatchSize = endSampleId - startSampleId;

                workspace.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                Mat inputSubBatch;

                // We optimize for three different scenarios here by handling them slightly differently.
                // [Scenario 1] Dense: Unroll using AssignPackedConvolutionInput and multiply.
                // [Scenario 2] Sparse 1-D convolution on GPU: for text scenarios we have a specific kernel.
                // [Scenario 3] Sparse all others: convert to dense. Temporary work-around - allocating/de-allocating memory is costly!
                if (in.GetMatrixType() == MatrixType::DENSE)
                    inputSubBatch = in.ColumnSlice(startSampleId, smallBatchSize);
                else
                {
                    inputSubBatch.SetValue(in.ColumnSlice(startSampleId, smallBatchSize), in.GetFormat());
                    inputSubBatch.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, true);
                }

                if (gpuSparse1D)
                {
                    if (filterT.w() * inT.c() != filter.GetNumCols())
                        LogicError("Kernel width and weight matrix dimensions don't match.");

                    Mat outputSubBatch = out.ColumnSlice(startSampleId, smallBatchSize);
                    Mat::ConvolveAndWeightedAdd(1, filter, false, inputSubBatch, false, 0, outputSubBatch,
                        static_cast<int>(inT.c()), convDesc.wStride(), convDesc.padding(), true);
                }
                else
                {
                    workspace.AssignPackedConvolutionInput(inputSubBatch,
                        inT.w(), inT.h(), inT.c(),
                        outT.w(), outT.h(), outT.c(),
                        filterT.w(), filterT.h(), convDesc.wStride(), convDesc.hStride(),
                        convDesc.padding());

                    Mat outputSubBatch = out.ColumnSlice(outputSizePerChannel * startSampleId, outputSizePerChannel * smallBatchSize);
                    Mat::Multiply(filter, false, workspace, false, outputSubBatch);
                }
            }

            out.Reshape(outT.c() * outputSizePerChannel, batchSize);  //each sample becomes a column

            assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
            assert(outT.n() == out.GetNumCols());
        }

        void BackwardData(const Tensor4D& srcGradT, const Mat& srcGrad, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
            const Tensor4D& gradT, Mat& grad, Mat& workspace) override
        {
            assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
            assert(srcGradT.n() == srcGrad.GetNumCols());
            assert(filterT.k() == filter.GetNumRows());
            assert(filterT.w() * filterT.h() * filterT.c() == filter.GetNumCols());
            assert(srcGradT.c() == filterT.k());
            assert(gradT.c() == filterT.c());
            assert(gradT.w() * gradT.h() * gradT.c() == grad.GetNumRows());
            assert(gradT.n() == grad.GetNumCols());

            size_t packedInputRows = filterT.w() * filterT.h() * filterT.c();
            size_t packedInputColsPerSample = srcGradT.w() * srcGradT.h();
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = gradT.w() * gradT.h() * gradT.c();  //size of each input sample

            size_t batchSize = srcGradT.n();

            size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

            // Create slice which is the same as full matrix so we can reshape it.
            Matrix<ElemType> srcGradTmp = srcGrad.ColumnSlice(0, srcGrad.GetNumCols());
            srcGradTmp.Reshape(srcGradT.c(), outputSizePerChannel * batchSize);  //reshape to match the longernal operation

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
                    gradT.w(), gradT.h(), gradT.c(),
                    srcGradT.w(), srcGradT.h(), srcGradT.c(),
                    filterT.w(), filterT.h(), convDesc.wStride(), convDesc.hStride(),
                    convDesc.padding());
            }

            assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
            assert(srcGradT.n() == srcGrad.GetNumCols());
        }

        void BackwardFilter(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& inT, const Mat& in, const ConvDesc& convDesc, 
            const Filter& filterT, Mat& filter, bool allowReuse, Mat& workspace) override
        {
            assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
            assert(srcGradT.n() == srcGrad.GetNumCols());
            assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
            assert(inT.n() == in.GetNumCols());
            assert(srcGradT.c() == filterT.k());
            assert(inT.c() == filterT.c());
            assert(filterT.k() == filter.GetNumRows());
            assert(filterT.w() * filterT.h() * filterT.c() == filter.GetNumCols());

            size_t packedInputRows = filterT.w() * filterT.h() * filterT.c();
            size_t packedInputColsPerSample = srcGradT.w() * srcGradT.h();
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = m_inputImageLayout.width * m_inputImageLayout.height * m_inputImageLayout.channels;  //size of each input sample

            size_t batchSize = inT.n();

            size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

            //const Matrix<ElemType> & weightMatrix = input0;
            //inputGradientValues.Resize(weightMatrix.GetNumRows(), weightMatrix.GetNumCols()); //should have been resized when preparing gradient computation

            // Create slice which is the same as full matrix so we can reshape it.
            Matrix<ElemType> srcGradTmp = srcGrad.ColumnSlice(0, srcGrad.GetNumCols());
            srcGradTmp.Reshape(srcGradT.c(), outputSizePerChannel * batchSize);  //reshape to match the longernal operation

            size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples);
            size_t numSubBatches = (batchSize + subBatchSize - 1) / subBatchSize;

            // GPU and 1-dimensional image
            bool gpuSparse1D = (inT.h() == 1 &&
                in.GetCurrentMatrixLocation() == CurrentDataLocation::GPU &&
                in.GetMatrixType() == MatrixType::SPARSE);

            if (numSubBatches == 1 && allowReuse && !gpuSparse1D)  //reuse packed input from evaluation step if it's not changed by either subbatch or recurrent steps.
                // REVIEW alexeyk: the following makes an assumption that data in workspace was filled by Forward call and remained unchanged. Find way to enforce/verify that.
                Matrix<ElemType>::MultiplyAndAdd(srcGradTmp, false, workspace, true, filter);
            else
            {
                for (size_t i = 0; i < numSubBatches; i++)
                {
                    size_t startSampleID = i * subBatchSize;
                    size_t endSampleID = min(batchSize, startSampleID + subBatchSize);
                    size_t smallBatchSize = endSampleID - startSampleID;

                    workspace.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                    Matrix<ElemType> inputSubBatch = in.ColumnSlice(startSampleID, smallBatchSize);
                    inputSubBatch.SwitchToMatrixType(MatrixType::DENSE, inputSubBatch.GetFormat(), true);
                    workspace.AssignPackedConvolutionInput(inputSubBatch,
                        inT.w(), inT.h(), inT.c(),
                        srcGradT.w(), srcGradT.h(), srcGradT.c(),
                        filterT.w(), filterT.h(), convDesc.wStride(), convDesc.hStride(),
                        convDesc.padding());

                    Matrix<ElemType> outputGradientSubBatch = srcGradTmp.ColumnSlice(startSampleID * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
                    Matrix<ElemType>::MultiplyAndAdd(outputGradientSubBatch, false, workspace, true, filter);
                }
            }

            assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
            assert(srcGradT.n() == srcGrad.GetNumCols());
        }

        void AddBias(const Tensor4D& outT, const Mat& out, const Tensor4D& biasT, const Mat& bias, Mat& dst) override
        {
            assert(biasT.c() == outT.c());
            assert(biasT.w() == 1);
            assert(biasT.h() == 1);
            assert(biasT.n() == 1);
            assert(bias.GetNumRows() == biasT.c());
            assert(bias.GetNumCols() == 1);
            assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
            assert(outT.n() == out.GetNumCols());

            Mat o = out.ColumnSlice(0, out.GetNumCols());
            Mat d = dst.Reshaped(biasT.c(), outT.w() * outT.h() * outT.n());
            d.AssignSumOf(o.Reshaped(biasT.c(), outT.w() * outT.h() * outT.n()), bias);
        }

        void BackwardBias(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& biasT, Mat& biasGrad) override
        {
            assert(biasT.c() == srcGradT.c());
            assert(biasT.w() == 1);
            assert(biasT.h() == 1);
            assert(biasT.n() == 1);
            assert(biasGrad.GetNumRows() == biasT.c());
            assert(biasGrad.GetNumCols() == 1);

            Mat sg = srcGrad.ColumnSlice(0, srcGrad.GetNumCols());
            size_t ccol = srcGradT.w() * srcGradT.h() * srcGradT.n();
            // REVIEW alexeyk: should be replaced by ConstOnes eventually.
            m_ones.Resize(ccol, 1);
            m_ones.SetValue(1);
            Mat::MultiplyAndAdd(sg.Reshaped(biasT.c(), ccol), false, m_ones, false, biasGrad);
        }

        void NormalizeBatch(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias, 
            bool spatial, double expAvgFactor, Mat& runMean, Mat& runInvStdDev, Mat& out, Mat& saveMean, Mat& saveInvStdDev) override
        {
            UNUSED(inT); UNUSED(in); UNUSED(scaleBiasT); UNUSED(scale); UNUSED(bias); UNUSED(out); UNUSED(spatial); UNUSED(expAvgFactor);
            UNUSED(runMean); UNUSED(runInvStdDev); UNUSED(saveMean); UNUSED(saveInvStdDev);
            RuntimeError("Not yet implemented.");
        }

        void NormalizeBatchInference(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
            bool spatial, const Mat& runMean, const Mat& runInvStdDev, Mat& out) override
        {
            UNUSED(inT); UNUSED(in); UNUSED(scaleBiasT); UNUSED(scale); UNUSED(bias); UNUSED(out); UNUSED(spatial);
            UNUSED(runMean); UNUSED(runInvStdDev);
            RuntimeError("Not yet implemented.");
        }

        void BackwardNormalizeBatch(const Tensor4D& inT, const Mat& in, const Mat& srcGrad, Mat& grad, 
            const Tensor4D& scaleBiasT, const Mat& scale, bool spatial, const Mat& saveMean, const Mat& saveInvStdDev,
            Mat& scaleGrad, Mat& biasGrad) override
        {
            UNUSED(inT); UNUSED(in); UNUSED(srcGrad); UNUSED(grad); UNUSED(scaleBiasT); UNUSED(scale); UNUSED(scaleGrad); UNUSED(biasGrad); UNUSED(spatial); 
            UNUSED(saveMean); UNUSED(saveInvStdDev);
            RuntimeError("Not yet implemented.");
        }

    private:
        size_t m_maxTempMemSizeInSamples;
        Mat m_ones;
    };

    template class ConvolutionEngine<float>;
    template class ConvolutionEngine<double>;

    template<class ElemType>
    class DefaultPoolingEngine : public PoolingEngine<ElemType>
    {
    public:
    	using Base = PoolingEngine<ElemType>;
        using typename Base::Tensor4D;
        using typename Base::PoolDesc;
        using typename Base::Mat;

    public:
        void Forward(const Tensor4D& inT, const Mat& in, const PoolDesc& poolDesc, const Tensor4D& outT, Mat& out) override
        {
            assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
            assert(inT.n() == in.GetNumCols());
            assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
            assert(outT.n() == out.GetNumCols());

            if (poolDesc.kind() == PoolDesc::PoolKind::Max)
            {
                out.AssignMaxPoolingResult(in, inT.c(), inT.w(), inT.h(), inT.w() * inT.h() * inT.c(),
                    outT.w(), outT.h(), outT.w() * outT.h() * outT.c(),
                    poolDesc.w(), poolDesc.h(), poolDesc.wStride(), poolDesc.hStride());
            }
            else if (poolDesc.kind() == PoolDesc::PoolKind::Average)
            {
                out.AssignAveragePoolingResult(in, inT.c(), inT.w(), inT.h(), inT.w() * inT.h() * inT.c(),
                    outT.w(), outT.h(), outT.w() * outT.h() * outT.c(),
                    poolDesc.w(), poolDesc.h(), poolDesc.wStride(), poolDesc.hStride());
            }
            else
                assert(false);
        }

        void Backward(const Tensor4D& outT, const Mat& out, const Mat& srcGrad, const PoolDesc& poolDesc, const Tensor4D& inT, const Mat& in, Mat& grad) override
        {
            assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
            assert(outT.n() == out.GetNumCols());
            assert(out.GetNumRows() == srcGrad.GetNumRows());
            assert(out.GetNumCols() == srcGrad.GetNumCols());
            assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
            assert(inT.n() == in.GetNumCols());
            assert(in.GetNumRows() == grad.GetNumRows());
            assert(in.GetNumCols() == grad.GetNumCols());

            if (poolDesc.kind() == PoolDesc::PoolKind::Max)
            {
                grad.AddMaxPoolingGradient(srcGrad, in, out,
                    inT.c(), inT.w(), inT.h(), inT.w() * inT.h() * inT.c(),
                    outT.w(), outT.h(), outT.w() * outT.h() * outT.c(),
                    poolDesc.w(), poolDesc.h(), poolDesc.wStride(), poolDesc.hStride());
            }
            else if (poolDesc.kind() == PoolDesc::PoolKind::Average)
            {
                grad.AddAveragePoolingGradient(srcGrad, inT.c(), inT.w(), inT.h(), inT.w() * inT.h() * inT.c(),
                    outT.w(), outT.h(), outT.w() * outT.h() * outT.c(),
                    poolDesc.w(), poolDesc.h(), poolDesc.wStride(), poolDesc.hStride());
            }
            else
                assert(false);
        }
    };

    template class PoolingEngine<float>;
    template class PoolingEngine<double>;

    template<class ElemType>
    class DefaultConvolutionEngineFactory : public ConvolutionEngineFactory<ElemType>
    {
    public:
    	using Base = ConvolutionEngineFactory<ElemType>;
        using typename Base::Tensor4D;
        using typename Base::Tensor4DPtr;
        using typename Base::Filter;
        using typename Base::FilterPtr;
        using typename Base::ConvDesc;
        using typename Base::ConvDescPtr;
        using typename Base::PoolDesc;
        using typename Base::PoolDescPtr;

        using typename Base::ConvEnginePtr;
        using typename Base::PoolEnginePtr;

    public:
        Tensor4DPtr CreateTensor(size_t w, size_t h, size_t c, size_t n) override
        {
            return std::make_unique<ConvolutionTensor4D>(w, h, c, n);
        }

        FilterPtr CreateFilter(size_t w, size_t h, size_t c, size_t k) override
        {
            return std::make_unique<Filter>(w, h, c, k);
        }

        ConvDescPtr CreateConvDescriptor(const Tensor4D& /*inT*/, const Filter& /*filterT*/,
            size_t wStride, size_t hStride, bool padding) override
        {
            return std::make_unique<ConvDesc>(wStride, hStride, padding);
        }

        PoolDescPtr CreatePoolDescriptor(typename PoolDesc::PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad) override
        {
            return std::make_unique<PoolDesc>(kind, w, h, wStride, hStride, wPad, hPad);
        }

        ConvEnginePtr CreateConvEngine(DEVICEID_TYPE deviceId, size_t maxTempMemSizeInSamples) override
        {
            return std::make_unique<DefaultConvolutionEngine<ElemType>>(deviceId, maxTempMemSizeInSamples);
        }

        PoolEnginePtr CreatePoolEngine(DEVICEID_TYPE /*deviceId*/) override
        {
            return std::make_unique<DefaultPoolingEngine<ElemType>>();
        }
    };

    template<class ElemType>
    std::unique_ptr<ConvolutionEngineFactory<ElemType>> ConvolutionEngineFactory<ElemType>::Create(DEVICEID_TYPE deviceId)
    {
        // REVIEW alexeyk: make cuDNN default when running on GPU and compiled with cuDNN, add config parameter to enable runtime switch between implementations.
        if (deviceId >= 0 && CuDnnConvolutionEngineFactory<ElemType>::IsSupported(deviceId))
            return std::make_unique<CuDnnConvolutionEngineFactory<ElemType>>();
        return std::make_unique<DefaultConvolutionEngineFactory<ElemType>>();
    }

    template class ConvolutionEngineFactory<float>;
    template class ConvolutionEngineFactory<double>;

}}}
