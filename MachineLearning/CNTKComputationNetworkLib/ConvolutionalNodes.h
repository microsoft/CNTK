//
// <copyright file="ConvolutionalNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "ComputationNode.h"
#include "InputAndParamNodes.h"

#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <assert.h>
#include <atomic>
#include <sstream>
#include <iostream>

namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // ConvolutionNode (convolutionWeights, inputFeature)
    // -----------------------------------------------------------------------

    // convolutional network 
    // This follows "high performance convolutional neural networks for document processing" by Kumar Chellapilla, Sidde Puri, and Patrice Simard.
    // Each sample is stored as a column-major matrix (height, width) of float[numChannels] (r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11).
    template<class ElemType>
    class ConvolutionNode : public ComputationNode<ElemType>, public NumInputs<2>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"Convolution"; }
    public:
        ConvolutionNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            m_kernelWidth(SIZE_MAX), m_kernelHeight(SIZE_MAX),
            // initialize to dummy values so we catch missing initialization
            m_horizontalSubsample(SIZE_MAX), m_verticalSubsample(SIZE_MAX),
            m_zeroPadding(false), m_maxTempMemSizeInSamples(SIZE_MAX)            
        {
            m_outputImageLayout.channels = 0;
        }
        ConvolutionNode(DEVICEID_TYPE deviceId, const wstring & name, const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels, const size_t horizontalSubsample, const size_t verticalSubsample, const bool zeroPadding = false, const size_t maxTempMemSizeInSamples = 0) :
            Base(deviceId, name),
            m_kernelWidth(kernelWidth), m_kernelHeight(kernelHeight),
            m_horizontalSubsample(horizontalSubsample), m_verticalSubsample(verticalSubsample),
            m_zeroPadding(zeroPadding), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples)
        {
            m_outputImageLayout.channels = outputChannels;
        }

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            fstream <<  m_kernelWidth << m_kernelHeight << m_horizontalSubsample << m_verticalSubsample; 
            fstream << m_outputImageLayout.channels << m_zeroPadding << m_maxTempMemSizeInSamples; 
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_kernelWidth >> m_kernelHeight >> m_horizontalSubsample >> m_verticalSubsample; 
            fstream >> m_outputImageLayout.channels >> m_zeroPadding >> m_maxTempMemSizeInSamples; 
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<ConvolutionNode<ElemType>>(nodeP);
                node->m_kernelWidth = m_kernelWidth;
                node->m_kernelHeight = m_kernelHeight;

                node->m_horizontalSubsample = m_horizontalSubsample;
                node->m_verticalSubsample = m_verticalSubsample;

                node->m_zeroPadding = m_zeroPadding;

                node->m_maxTempMemSizeInSamples = m_maxTempMemSizeInSamples;

                *node->m_tempMatrix = *m_tempMatrix;
            }
        }

        //void ComputeInputPartialMap(const size_t inputIndex) 
        //{
        //    if (inputIndex > 1)
        //        InvalidArgument("Convolution operation only takes two inputs.");
        //
        //    if (inputIndex == 0)  //derivative with regard to the weight matrix
        //        ComputeInputPartialOverWeight(GradientValues(), Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_tempMatrix, true);
        //    else  // derivative with regard to the input feature
        //        ComputeInputPartialOverInputFeature(GradientValues(), Inputs(1)->GradientValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_tempMatrix);
        //}

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            //if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(Inputs(1)->GetNumParallelSequences(), m_pMBLayout));

            if (inputIndex == 0)  //derivative with regard to the weight matrix
                ComputeInputPartialOverWeight(sliceOutputGrad, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), sliceInput1Value, *m_tempMatrix, !frameRange.IsAllFrames());
            else if (inputIndex == 1)  // derivative with regard to the input feature
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
                ComputeInputPartialOverInputFeature(sliceOutputGrad, sliceInput1Grad, Inputs(0)->FunctionValues(), sliceInput1Value, *m_tempMatrix);
            }
        }

    private:
        void ComputeInputPartialOverWeight(Matrix<ElemType> &gradientValues,
            Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &/*input0*/, const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix, const bool inLoop)
        {
            size_t packedInputRows = m_kernelWidth * m_kernelHeight * m_inputImageLayout.channels;
            size_t packedInputColsPerSample = m_outputImageLayout.width * m_outputImageLayout.height;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = m_inputImageLayout.width * m_inputImageLayout.height * m_inputImageLayout.channels;  //size of each input sample

            size_t batchSize = input1.GetNumCols(); //right child is the input sample

            size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

            //const Matrix<ElemType> & weightMatrix = input0;
            //inputGradientValues.Resize(weightMatrix.GetNumRows(), weightMatrix.GetNumCols()); //should have been resized when preparing gradient computation

            gradientValues.Reshape(m_outputImageLayout.channels, outputSizePerChannel * batchSize);  //reshape to match the longernal operation

            size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples);
            size_t numSubBatches = (batchSize + subBatchSize - 1) / subBatchSize;

            if (numSubBatches == 1 && !inLoop)  //reuse packed input from evaluation step if it's not changed by either subbatch or recurrent steps.
                Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, tempMatrix, true, inputGradientValues);
            else
            {
                for (size_t i = 0; i<numSubBatches; i++)
                {
                    size_t startSampleID = i*subBatchSize;
                    size_t endSampleID = min(batchSize, startSampleID + subBatchSize);
                    size_t smallBatchSize = endSampleID - startSampleID;

                    tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                    Matrix<ElemType> inputSubBatch = input1.ColumnSlice(startSampleID, smallBatchSize);
                    inputSubBatch.SwitchToMatrixType(MatrixType::DENSE, inputSubBatch.GetFormat(), true);
                    tempMatrix.AssignPackedConvolutionInput(inputSubBatch,
                                                            m_inputImageLayout.width, m_inputImageLayout.height, m_inputImageLayout.channels,
                                                            m_outputImageLayout.width, m_outputImageLayout.height, m_outputImageLayout.channels,
                                                            m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample,
                                                            m_zeroPadding);

                    Matrix<ElemType> outputGradientSubBatch = gradientValues.ColumnSlice(startSampleID * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
                    Matrix<ElemType>::MultiplyAndAdd(outputGradientSubBatch, false, tempMatrix, true, inputGradientValues);
                }
            }

            gradientValues.Reshape(m_outputImageLayout.channels * outputSizePerChannel, batchSize);  //change back
        }

        //compute gradient over the packed input and then convert the result to the original input
        void ComputeInputPartialOverInputFeature(Matrix<ElemType> &gradientValues, const Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix)
        {
            size_t packedInputRows = m_kernelWidth * m_kernelHeight * m_inputImageLayout.channels;
            size_t packedInputColsPerSample = m_outputImageLayout.width * m_outputImageLayout.height;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = m_inputImageLayout.width * m_inputImageLayout.height * m_inputImageLayout.channels;  //size of each input sample

            size_t batchSize = input1.GetNumCols(); //right child is the input sample

            size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

            const Matrix<ElemType> & weightMatrix = input0;

            gradientValues.Reshape(m_outputImageLayout.channels, outputSizePerChannel * batchSize);  //reshape to match the longernal operation

            size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples);
            size_t numSubBatches = (batchSize + subBatchSize - 1) / subBatchSize;

            for (size_t i = 0; i<numSubBatches; i++)
            {
                size_t startSampleID = i*subBatchSize;
                size_t endSampleID = min(batchSize, startSampleID + subBatchSize);
                size_t smallBatchSize = endSampleID - startSampleID;

                tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                Matrix<ElemType> outputGradientSubBatch = gradientValues.ColumnSlice(startSampleID * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
                Matrix<ElemType>::Multiply(weightMatrix, true, outputGradientSubBatch, false, tempMatrix);

                Matrix<ElemType> inputGradientSubBatch = inputGradientValues.ColumnSlice(startSampleID, smallBatchSize);
                tempMatrix.UnpackConvolutionInput(inputGradientSubBatch,
                                                  m_inputImageLayout.width, m_inputImageLayout.height, m_inputImageLayout.channels,
                                                  m_outputImageLayout.width, m_outputImageLayout.height, m_outputImageLayout.channels,
                                                  m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample,
                                                  m_zeroPadding);
            }

            gradientValues.Reshape(m_outputImageLayout.channels * outputSizePerChannel, batchSize);  //change back
        }
    public:

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            //if (frameRange.IsAllFrames()) { EvaluateThisNodeMap(); return; }
            Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value, *m_tempMatrix);
        }

    private:
        void EvaluateThisNodeS(Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0, 
                               const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix)
        {
#if NANCHECK
            input0.HasNan("Convolution-input0");
            input1.HasNan("Convolution-input1");
#endif
            size_t packedInputRows = m_kernelWidth * m_kernelHeight * m_inputImageLayout.channels;
            size_t packedInputColsPerSample = m_outputImageLayout.width * m_outputImageLayout.height;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = m_inputImageLayout.width * m_inputImageLayout.height * m_inputImageLayout.channels;  //size of each input sample

            size_t batchSize = input1.GetNumCols();  //right child is the input sample

            size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0? batchSize : m_maxTempMemSizeInSamples);

            const Matrix<ElemType> & weightMatrix = input0;
            assert(weightMatrix.GetNumCols() == packedInputRows && weightMatrix.GetNumRows() == m_outputImageLayout.channels);
            functionValues.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, false);
            functionValues.Resize(m_outputImageLayout.channels, outputSizePerChannel * batchSize);

            size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples); 
            size_t numSubBatches = (batchSize+subBatchSize-1)/subBatchSize; 

            for (size_t i=0; i<numSubBatches; i++) 
            {
                size_t startSampleID = i*subBatchSize; 
                size_t endSampleID = min(batchSize, startSampleID + subBatchSize); 
                size_t smallBatchSize = endSampleID-startSampleID;

                // GPU and 1-dimensional image
                bool is1DConvolutionOnGPU = (input1.GetCurrentMatrixLocation() == CurrentDataLocation::GPU && m_inputImageLayout.height == 1);

                tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                Matrix<ElemType>  inputSubBatch;

                // We optimize for three different scenarios here by handling them slightly differently.
                // Scenario 1 (dense input): Unroll using AssignPackedConvolutionInput and multiply.
                // Scenario 2 (sparse 1-dimensional image): for text scenarios we have a specific kernel.
                // Scenario 3 (sparse all others): convert to dense. Temporary work-around - allocating/de-allocating memory is costly!
                if (input1.GetMatrixType() == MatrixType::DENSE || is1DConvolutionOnGPU)
                {
                    inputSubBatch = input1.ColumnSlice(startSampleID, smallBatchSize);
                }
                else
                {
                    inputSubBatch.SetValue(input1.ColumnSlice(startSampleID, smallBatchSize), input1.GetFormat());
                    inputSubBatch.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, true);
                }

                Matrix<ElemType>  outputSubBatch = functionValues.ColumnSlice(outputSizePerChannel * startSampleID, outputSizePerChannel * smallBatchSize);

                if (is1DConvolutionOnGPU)
                {
                    Matrix<ElemType>::ConvolveAndWeightedAdd(1, weightMatrix, false, inputSubBatch, false, 1, outputSubBatch,
                        m_horizontalSubsample * m_inputImageLayout.channels, m_zeroPadding);
                }
                else
                {
                    tempMatrix.AssignPackedConvolutionInput(inputSubBatch,
                        m_inputImageLayout.width, m_inputImageLayout.height, m_inputImageLayout.channels,
                        m_outputImageLayout.width, m_outputImageLayout.height, m_outputImageLayout.channels,
                        m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample,
                        m_zeroPadding);

                    Matrix<ElemType>::Multiply(weightMatrix, false, tempMatrix, false, outputSubBatch);
                }
            }

            functionValues.Reshape(m_outputImageLayout.channels * outputSizePerChannel, batchSize);  //each sample becomes a column

#if NANCHECK
            functionValues.HasNan("Convolution");
#endif
        }
    public:

        // note: this also infers dimensions from chilren
        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            //we may want to remove this check in the future if we want to support the case that the weight itself is result of some computation 
            //if (Inputs(0)->OperationName() != OperationNameOf(LearnableParameter))
            //    LogicError("ConvolutionNode requires the first input to be LearnableParameter type.");

            if (m_horizontalSubsample > m_kernelWidth || m_verticalSubsample > m_kernelHeight)
                InvalidArgument("In ConvolutionNode horizontalSubsample must <= kernelWidth and verticalSubsample must <= kernelHeight.");

            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();

            size_t weightCols = m_kernelWidth * m_kernelHeight * m_inputImageLayout.channels;

            if (Inputs(0)->FunctionValues().HasNoElements())
                ValidateInferChildDims(0, m_outputImageLayout.channels, weightCols);

            if (isFinalValidationPass && (Inputs(0)->GetNumCols() != weightCols || Inputs(0)->GetNumRows() != m_outputImageLayout.channels))
                LogicError("convolutionWeight matrix %ls should have dimension [%d, %d] which is [outputChannels, kernelWidth * kernelHeight * inputChannels]", m_children[0]->NodeName().c_str(), m_outputImageLayout.channels, weightCols);

            size_t inputDim = m_inputImageLayout.width * m_inputImageLayout.height * m_inputImageLayout.channels;
            if (Inputs(1)->GetNumRows() == 0)
                ValidateInferChildDims(1, inputDim, Inputs(1)->GetNumCols());

            if (isFinalValidationPass && Inputs(1)->GetNumRows() != inputDim)
                LogicError("each column of input to the convolution node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", NodeName().c_str(), inputDim);

            //if (Inputs(0)->GetNumRows() == 0 || Inputs(1)->GetNumRows() == 0 )
            //    LogicError("Convolution operation: one of the operands has 0 elements.");
            
            size_t outputDim = m_outputImageLayout.width * m_outputImageLayout.height * m_outputImageLayout.channels;
            Resize(outputDim, Inputs(1)->GetNumCols());
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(1, false);

            if (m_inputImageLayout.width < m_kernelWidth || m_inputImageLayout.height < m_kernelHeight)
                InvalidArgument("inputWidth must >= kernelWidth and inputHeight must >= kernelHeight.");

            if (m_zeroPadding)
            {
                const int kernelWidthCenter = m_kernelWidth % 2;
                const int kernelHeightCenter = m_kernelHeight % 2;
                m_outputImageLayout.width = (m_inputImageLayout.width-kernelWidthCenter)/m_horizontalSubsample + 1;
                m_outputImageLayout.height = (m_inputImageLayout.height-kernelHeightCenter)/m_verticalSubsample + 1;
            }
            else
            {
                m_outputImageLayout.width = (m_inputImageLayout.width-m_kernelWidth)/m_horizontalSubsample + 1;
                m_outputImageLayout.height = (m_inputImageLayout.height-m_kernelHeight)/m_verticalSubsample + 1;
            }    
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const override
        {
            Base::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", m_inputImageLayout.width, m_inputImageLayout.height, m_inputImageLayout.channels);
            fstream << string(str);
            sprintf(str, "Kernel[Width:%lu, Height:%lu]  SubSample[Horizontal:%lu, Vertical:%lu]\n", m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_outputImageLayout.width, m_outputImageLayout.height, m_outputImageLayout.channels);
            fstream << string(str);
            sprintf(str, "ZeroPadding=%ls  maxTempMemSizeInSamples=%lu\n", m_zeroPadding? L"true" : L"false", m_maxTempMemSizeInSamples);
            fstream << string(str);
        }

        void SetmMaxTempMemSizeInSamples(const size_t maxTempMemSizeInSamples)
        {
            m_maxTempMemSizeInSamples = maxTempMemSizeInSamples;
        }

        //request matrices needed to do node function value evaluation
        virtual void RequestMatricesBeforeEval(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeEval(matrixPool);
            RequestMatrixFromPool(m_tempMatrix, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        virtual void ReleaseMatricesAfterGradientComp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterGradientComp(matrixPool);
            ReleaseMatrixToPool(m_tempMatrix, matrixPool);
        }

    private:
        size_t m_kernelWidth, m_kernelHeight;
        size_t m_horizontalSubsample, m_verticalSubsample;
        bool m_zeroPadding;

        shared_ptr<Matrix<ElemType>> m_tempMatrix;
        size_t m_maxTempMemSizeInSamples; // can change during runtime
    };

    template class ConvolutionNode<float>; 
    template class ConvolutionNode<double>;

    // -----------------------------------------------------------------------
    // PoolingNodeBase (input)
    // -----------------------------------------------------------------------

    // Max/Average Pooling: support multi channel
    // Each sample is stored as a column-major matrix (height, width) of float[numChannels] (r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11).
    template<class ElemType>
    class PoolingNodeBase : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        PoolingNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            m_windowWidth(SIZE_MAX), m_windowHeight(SIZE_MAX),
            m_horizontalSubsample(SIZE_MAX), m_verticalSubsample(SIZE_MAX)
        { }
        PoolingNodeBase(DEVICEID_TYPE deviceId, const wstring & name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample) :
            Base(deviceId, name),
            m_windowWidth(windowWidth), m_windowHeight(windowHeight),
            m_horizontalSubsample(horizontalSubsample), m_verticalSubsample(verticalSubsample)
        { }

        virtual void SaveToFile(File& fstream) const override
        {
            Base::SaveToFile(fstream);
            fstream << m_windowWidth << m_windowHeight << m_horizontalSubsample << m_verticalSubsample;
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion) override
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_windowWidth >> m_windowHeight >> m_horizontalSubsample >> m_verticalSubsample;
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<PoolingNodeBase<ElemType>>(nodeP);

                node->m_windowWidth = m_windowWidth;
                node->m_windowHeight = m_windowHeight;

                node->m_horizontalSubsample = m_horizontalSubsample;
                node->m_verticalSubsample = m_verticalSubsample;

                node->m_inputSizePerSample = m_inputSizePerSample;
                node->m_outputSizePerSample = m_outputSizePerSample;
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t /*inputIndex*/, const FrameRange & frameRange) override
        {
            //if (frameRange.IsAllFrames()) { ComputeInputPartialMap(inputIndex); return; } // TODO: remove these one by one
            Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));

            ComputeInputPartialV(sliceOutputGrad, sliceInput0Grad, sliceInput0Value, sliceOutputValue);
        }

        // this function must be overriden by Max or AveragePoolingNode
        virtual void ComputeInputPartialV(const Matrix<ElemType> &gradientValues, Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &functionValues) = 0;

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            //if (frameRange.IsAllFrames()) { EvaluateThisNodeMap(); return; }
            Matrix<ElemType> sliceInput0Value = Inputs(0)->ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange/*TODO: delete this:*/.Check_t(GetNumParallelSequences(), m_pMBLayout));
            EvaluateThisNodeV(sliceOutputValue, sliceInput0Value);
        }

        // this function must be overriden by Max or AveragePoolingNode
        virtual void EvaluateThisNodeV(Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0) = 0;

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (m_horizontalSubsample > m_windowWidth || m_verticalSubsample > m_windowHeight)
                InvalidArgument("PoolingNodeBase: horizontalSubsample must <= windowWidth and verticalSubsample must <= windowHeight.");

            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();

            m_inputSizePerSample = m_inputImageLayout.width * m_inputImageLayout.height * m_inputImageLayout.channels;
            m_outputSizePerSample = m_outputImageLayout.width * m_outputImageLayout.height * m_outputImageLayout.channels;

            if (Inputs(0)->GetNumRows() == 0)
                ValidateInferChildDims(0, m_inputSizePerSample, Inputs(0)->GetNumCols());

            if (isFinalValidationPass && Inputs(0)->GetNumRows() != m_inputSizePerSample)
                LogicError("each column of input to the MaxPooling node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", NodeName().c_str(), m_inputSizePerSample);

            m_functionValues->Resize(m_outputSizePerSample, Inputs(0)->GetNumCols());
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            if (m_inputImageLayout.width < m_windowWidth || m_inputImageLayout.height < m_windowHeight)
                InvalidArgument("PoolingNodeBase: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

            m_outputImageLayout.width = (m_inputImageLayout.width - m_windowWidth) / m_horizontalSubsample + 1;
            m_outputImageLayout.height = (m_inputImageLayout.height - m_windowHeight) / m_verticalSubsample + 1;
            m_outputImageLayout.channels = m_inputImageLayout.channels;
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const override
        {
            Base::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", m_inputImageLayout.width, m_inputImageLayout.height, m_inputImageLayout.channels);
            fstream << string(str);
            sprintf(str, "PoolingWindow[Width:%lu, Height:%lu]  SubSampling[Horizontal:%lu, Vertical:%lu]\n", m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_outputImageLayout.width, m_outputImageLayout.height, m_outputImageLayout.channels);
            fstream << string(str);
            sprintf(str, "TotalSizePerSample[Input:%lu, Output:%lu]  \n", m_inputSizePerSample, m_outputSizePerSample);
            fstream << string(str);
        }

    protected:
        size_t m_windowWidth, m_windowHeight;
        size_t m_horizontalSubsample, m_verticalSubsample;
        size_t m_inputSizePerSample, m_outputSizePerSample;
    };

    // add this at the start of each derived class, to get access to the members of ComputationNode
    // See #define of 'UsingComputationNodeMembersBoilerplate' for more explanation.
#define UsingPoolingNodeBaseMembers UsingComputationNodeMembersBoilerplate; \
    protected:  \
        using Base::m_windowWidth; using Base::m_windowHeight; using Base::m_horizontalSubsample; using Base::m_verticalSubsample; using Base::m_inputSizePerSample; using Base::m_outputSizePerSample; \
    public:

    // -----------------------------------------------------------------------
    // MaxPoolingNode
    // -----------------------------------------------------------------------

    template<class ElemType>
    class MaxPoolingNode : public PoolingNodeBase<ElemType>
    {
        typedef PoolingNodeBase<ElemType> Base; UsingPoolingNodeBaseMembers;
        static const std::wstring TypeName() { return L"MaxPooling"; }
    public:
        MaxPoolingNode(DEVICEID_TYPE deviceId, const wstring & name) : Base(deviceId, name) { }
        MaxPoolingNode(DEVICEID_TYPE deviceId, const wstring & name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample) :
            Base(deviceId, name, windowWidth, windowHeight, horizontalSubsample, verticalSubsample)
        { }

        /*implement*/ void ComputeInputPartialV(const Matrix<ElemType> &gradientValues, Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &functionValues)
        {
            inputGradientValues.AddMaxPoolingGradient(gradientValues, input0, functionValues, m_inputImageLayout.channels,
                                                      m_inputImageLayout.width, m_inputImageLayout.height, m_inputSizePerSample, 
                                                      m_outputImageLayout.width, m_outputImageLayout.height, m_outputSizePerSample, 
                                                      m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
        }

        /*implement*/ void EvaluateThisNodeV(Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0)
        {
            functionValues.AssignMaxPoolingResult(input0, m_inputImageLayout.channels,
                                                  m_inputImageLayout.width, m_inputImageLayout.height, m_inputSizePerSample, 
                                                  m_outputImageLayout.width, m_outputImageLayout.height, m_outputSizePerSample, 
                                                  m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
        }
    };

    template class MaxPoolingNode<float>; 
    template class MaxPoolingNode<double>;    

    // -----------------------------------------------------------------------
    // AveragePoolingNode
    // -----------------------------------------------------------------------

    template<class ElemType>
    class AveragePoolingNode : public PoolingNodeBase<ElemType>
    {
        typedef PoolingNodeBase<ElemType> Base; UsingPoolingNodeBaseMembers;
        static const std::wstring TypeName() { return L"AveragePooling"; }
    public:
        AveragePoolingNode(DEVICEID_TYPE deviceId, const wstring & name) : Base(deviceId, name) { }
        AveragePoolingNode(DEVICEID_TYPE deviceId, const wstring & name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample) :
            Base(deviceId, name, windowWidth, windowHeight, horizontalSubsample, verticalSubsample)
        { }

        /*implement*/ void ComputeInputPartialV(const Matrix<ElemType> &gradientValues, Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &/*input0*/, const Matrix<ElemType> &/*functionValues*/)
        {
            inputGradientValues.AddAveragePoolingGradient(gradientValues, m_inputImageLayout.channels,
                                                          m_inputImageLayout.width, m_inputImageLayout.height, m_inputSizePerSample, 
                                                          m_outputImageLayout.width, m_outputImageLayout.height, m_outputSizePerSample, 
                                                          m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
        }

        /*implement*/ void EvaluateThisNodeV(Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0)
        {
            functionValues.AssignAveragePoolingResult(input0, m_inputImageLayout.channels,
                                                      m_inputImageLayout.width, m_inputImageLayout.height, m_inputSizePerSample, 
                                                      m_outputImageLayout.width, m_outputImageLayout.height, m_outputSizePerSample, 
                                                      m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
        }
    };

    template class AveragePoolingNode<float>; 
    template class AveragePoolingNode<double>;    

}}}
