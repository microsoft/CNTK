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
        DefaultConvolutionEngine(DEVICEID_TYPE deviceId, size_t maxTempMemSizeInSamples)
            : m_tempMatrix(deviceId), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples)
        {
        }

    public:
        void Forward(const Tensor4D& inT, const Mat& in, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
            const Tensor4D& outT, Mat& out) override
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
            out.Resize(outT.c(), outputSizePerChannel * batchSize);

            size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples);
            size_t numSubBatches = (batchSize + subBatchSize - 1) / subBatchSize;

            for (size_t i = 0; i < numSubBatches; i++)
            {
                size_t startSampleId = i * subBatchSize;
                size_t endSampleId = min(batchSize, startSampleId + subBatchSize);
                size_t smallBatchSize = endSampleId - startSampleId;

                m_tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                Mat inputSubBatch = in.ColumnSlice(startSampleId, smallBatchSize);
                m_tempMatrix.AssignPackedConvolutionInput(inputSubBatch,
                    inT.w(), inT.h(), inT.c(),
                    outT.w(), outT.h(), outT.c(),
                    filterT.w(), filterT.h(), convDesc.wStride(), convDesc.hStride(),
                    convDesc.padding());

                Mat outputSubBatch = out.ColumnSlice(outputSizePerChannel * startSampleId, outputSizePerChannel * smallBatchSize);
                Mat::Multiply(filter, false, m_tempMatrix, false, outputSubBatch);
            }

            out.Reshape(outT.c() * outputSizePerChannel, batchSize);  //each sample becomes a column

            assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
            assert(outT.n() == out.GetNumCols());
        }

        void BackwardData(const Tensor4D& srcGradT, const Mat& srcGrad, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
            const Tensor4D& gradT, Mat& grad) override
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

                m_tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                Matrix<ElemType> outputGradientSubBatch = srcGradTmp.ColumnSlice(startSampleId * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
                Matrix<ElemType>::Multiply(filter, true, outputGradientSubBatch, false, m_tempMatrix);

                Matrix<ElemType> inputGradientSubBatch = grad.ColumnSlice(startSampleId, smallBatchSize);
                m_tempMatrix.UnpackConvolutionInput(inputGradientSubBatch,
                    gradT.w(), gradT.h(), gradT.c(),
                    srcGradT.w(), srcGradT.h(), srcGradT.c(),
                    filterT.w(), filterT.h(), convDesc.wStride(), convDesc.hStride(),
                    convDesc.padding());
            }

            assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
            assert(srcGradT.n() == srcGrad.GetNumCols());
        }

        void BackwardFilter(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& inT, const Mat& in, const ConvDesc& convDesc, 
            const Filter& filterT, Mat& filter, bool allowReuse) override
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

            if (numSubBatches == 1 && allowReuse)  //reuse packed input from evaluation step if it's not changed by either subbatch or recurrent steps.
                // REVIEW alexeyk: the following makes an assumption that data in m_tempMatrix was filled by Forward call and remained unchanged. Find way to enforce/verify that.
                Matrix<ElemType>::MultiplyAndAdd(srcGradTmp, false, m_tempMatrix, true, filter);
            else
            {
                for (size_t i = 0; i < numSubBatches; i++)
                {
                    size_t startSampleID = i * subBatchSize;
                    size_t endSampleID = min(batchSize, startSampleID + subBatchSize);
                    size_t smallBatchSize = endSampleID - startSampleID;

                    m_tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                    Matrix<ElemType> inputSubBatch = in.ColumnSlice(startSampleID, smallBatchSize);
                    m_tempMatrix.AssignPackedConvolutionInput(inputSubBatch,
                        inT.w(), inT.h(), inT.c(),
                        srcGradT.w(), srcGradT.h(), srcGradT.c(),
                        filterT.w(), filterT.h(), convDesc.wStride(), convDesc.hStride(),
                        convDesc.padding());

                    Matrix<ElemType> outputGradientSubBatch = srcGradTmp.ColumnSlice(startSampleID * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
                    Matrix<ElemType>::MultiplyAndAdd(outputGradientSubBatch, false, m_tempMatrix, true, filter);
                }
            }

            assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
            assert(srcGradT.n() == srcGrad.GetNumCols());
        }

        Tensor4DPtr CreateTensor(size_t w, size_t h, size_t c, size_t n) override
        {
            return std::make_unique<ConvolutionTensor4D>(w, h, c, n);
        }

        FilterPtr CreateFilter(size_t w, size_t h, size_t c, size_t k) override
        {
            return std::make_unique<Filter>(w, h, c, k);
        }

        ConvDescPtr CreateConvDescriptor(const Tensor4D& inT, const Filter& filterT, 
            size_t wStride, size_t hStride, bool padding) override
        {
            return std::make_unique<ConvDesc>(inT, filterT, wStride, hStride, padding);
        }

    private:
        size_t m_maxTempMemSizeInSamples;
        Mat m_tempMatrix;
    };

    template<class ElemType>
    std::unique_ptr<ConvolutionEngine<ElemType>> ConvolutionEngine<ElemType>::Create(DEVICEID_TYPE deviceId, size_t maxTempMemSizeInSamples)
    {
        // REVIEW alexeyk: make cuDNN default when running on GPU and compiled with cuDNN, add config parameter to enable runtime switch between implementations.
        if (deviceId >= 0 && CuDnnConvolutionEngine<ElemType>::IsSupported())
            return std::make_unique<CuDnnConvolutionEngine<ElemType>>(deviceId, maxTempMemSizeInSamples);
        return std::make_unique<DefaultConvolutionEngine<ElemType>>(deviceId, maxTempMemSizeInSamples);
    }

    template class ConvolutionEngine<float>;
    template class ConvolutionEngine<double>;
}}}
