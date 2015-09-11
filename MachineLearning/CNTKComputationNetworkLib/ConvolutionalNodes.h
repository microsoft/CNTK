//
// <copyright file="ConvolutionalNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

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

#include "Basics.h"
#include "Matrix.h"
#include "ComputationNode.h"
#include "InputAndParamNodes.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // ConvolutionNode
    // -----------------------------------------------------------------------

    //convolutional network 
    //follow "high performance convolutional neural networks for document processing" by Kumar chellapilla, Sidde Puri, and Patrice Simard
    //assume each column is an input sample. Each sample is stored in [channel, row, col]  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
    template<class ElemType>
    class ConvolutionNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        ConvolutionNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_tempMatrix(deviceId),
            m_kernelWidth(SIZE_MAX), m_kernelHeight(SIZE_MAX),
            // initialize to dummy values so we catch missing initialization
            m_horizontalSubsample(SIZE_MAX), m_verticalSubsample(SIZE_MAX),
            m_zeroPadding(false), m_maxTempMemSizeInSamples(SIZE_MAX)            
        {
            m_outputChannels = 0;
        }
        ConvolutionNode(DEVICEID_TYPE deviceId, const wstring & name, const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels, const size_t horizontalSubsample, const size_t verticalSubsample, const bool zeroPadding = false, const size_t maxTempMemSizeInSamples = 0) :
            ComputationNode<ElemType>(deviceId, name),
            m_tempMatrix(deviceId),
            m_kernelWidth(kernelWidth), m_kernelHeight(kernelHeight),
            m_horizontalSubsample(horizontalSubsample), m_verticalSubsample(verticalSubsample),
            m_zeroPadding(zeroPadding), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples)
        {
            m_outputChannels = outputChannels;
        }

        virtual void SaveToFile(File& fstream) const
        {
            Base::SaveToFile(fstream);
            fstream <<  m_kernelWidth << m_kernelHeight << m_horizontalSubsample << m_verticalSubsample; 
            fstream << m_outputChannels << m_zeroPadding << m_maxTempMemSizeInSamples; 
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion)
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_kernelWidth >> m_kernelHeight >> m_horizontalSubsample >> m_verticalSubsample; 
            fstream >> m_outputChannels >> m_zeroPadding >> m_maxTempMemSizeInSamples; 
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
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

                node->m_tempMatrix = m_tempMatrix;
            }
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"Convolution";} 

        //virtual void ComputeInputPartial(const size_t inputIndex) 
        //{
        //    if (inputIndex > 1)
        //        throw std::invalid_argument("Convolution operation only takes two inputs.");
        //
        //    if (inputIndex == 0)  //derivative with regard to the weight matrix
        //        ComputeInputPartialOverWeight(GradientValues(), Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_tempMatrix, true);
        //    else  // derivative with regard to the input feature
        //        ComputeInputPartialOverInputFeature(GradientValues(), Inputs(1)->GradientValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_tempMatrix);
        //}

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) 
        {
            if (inputIndex > 1)
                InvalidArgument("Convolution operation only takes two inputs.");

            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)  //derivative with regard to the weight matrix
                ComputeInputPartialOverWeight(sliceOutputGrad, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), sliceInput1Value, m_tempMatrix, !frameRange.IsAllFrames());
            else  // derivative with regard to the input feature
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                ComputeInputPartialOverInputFeature(sliceOutputGrad, sliceInput1Grad, Inputs(0)->FunctionValues(), sliceInput1Value, m_tempMatrix);
            }
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) 
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            EvaluateThisNodeS(sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value, m_tempMatrix);
        }

        void EvaluateThisNodeS(Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0, 
                               const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix)
        {
#if NANCHECK
            input0.HasNan("Convolution-input0");
            input1.HasNan("Convolution-input1");
#endif
            size_t packedInputRows = m_kernelWidth * m_kernelHeight * m_inputChannels;
            size_t packedInputColsPerSample = m_outputWidth * m_outputHeight;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = m_inputWidth * m_inputHeight * m_inputChannels;  //size of each input sample

            long batchSize = (long)input1.GetNumCols();  //right child is the input sample

            long maxTempMemSizeInSamples = (long)(m_maxTempMemSizeInSamples == 0? batchSize : m_maxTempMemSizeInSamples);

            const Matrix<ElemType> & weightMatrix = input0;
            assert(weightMatrix.GetNumCols() == packedInputRows && weightMatrix.GetNumRows() == m_outputChannels);
            functionValues.Resize(m_outputChannels, outputSizePerChannel * batchSize);

            long subBatchSize = (long)min(batchSize, maxTempMemSizeInSamples); 
            long numSubBatches = (batchSize+subBatchSize-1)/subBatchSize; 

            for (long i=0; i<numSubBatches; i++) 
            {
                long startSampleID = i*subBatchSize; 
                long endSampleID = min(batchSize, startSampleID + subBatchSize); 
                long smallBatchSize = endSampleID-startSampleID; 

                tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                Matrix<ElemType>  inputSubBatch = input1.ColumnSlice(startSampleID, smallBatchSize);
                tempMatrix.AssignPackedConvolutionInput(inputSubBatch, 
                                                        m_inputWidth, m_inputHeight, m_inputChannels,
                                                        m_outputWidth, m_outputHeight, m_outputChannels,
                                                        m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample, 
                                                        m_zeroPadding); 

                Matrix<ElemType>  outputSubBatch = functionValues.ColumnSlice(outputSizePerChannel * startSampleID, outputSizePerChannel * smallBatchSize);
                Matrix<ElemType>::Multiply(weightMatrix, false, tempMatrix, false, outputSubBatch);
            }

            functionValues.Reshape(m_outputChannels * outputSizePerChannel, batchSize);  //each sample becomes a column

#if NANCHECK
            functionValues.HasNan("Convolution");
#endif
        }

        // note: this also infers dimensions from chilren
        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                LogicError("ConvolutionNode requires two inputs.");

            //we may want to remove this check in the future if we want to support the case that the weight itself is result of some computation 
            //if (Inputs(0)->OperationName() != OperationNameOf(LearnableParameter))
            //    throw std::logic_error("ConvolutionNode requires the first input to be LearnableParameter type.");

            if (m_horizontalSubsample > m_kernelWidth || m_verticalSubsample > m_kernelHeight)
                InvalidArgument("In ConvolutionNode horizontalSubsample must <= kernelWidth and verticalSubsample must <= kernelHeight.");

            InferImageDimsFromInputs();

            size_t weightCols = m_kernelWidth * m_kernelHeight * m_inputChannels;

            if (Inputs(0)->OperationName() == OperationNameOf(LearnableParameter) && Inputs(0)->FunctionValues().HasNoElements())
                Inputs(0)->FunctionValues().Resize(m_outputChannels, weightCols);

            if (Inputs(0)->FunctionValues().GetNumCols() != weightCols || Inputs(0)->FunctionValues().GetNumRows() != m_outputChannels)
            {
                // TODO: move into LogicError call
                msra::strfun::strprintf msg("convolutionWeight matrix %ls should have dimension [%d, %d] which is [outputChannels, kernelWidth * kernelHeight * inputChannels]", 
                                            m_children[0]->NodeName().c_str(), m_outputChannels, weightCols);
                LogicError(msg.c_str());
            }

            size_t inputDim = m_inputWidth * m_inputHeight * m_inputChannels;
            if (Inputs(1)->OperationName() == OperationNameOf(LearnableParameter) && Inputs(1)->FunctionValues().GetNumRows() == 0)
                Inputs(1)->FunctionValues().Resize(inputDim, Inputs(1)->FunctionValues().GetNumCols());

            if (Inputs(1)->FunctionValues().GetNumRows() != inputDim)
            {
                msra::strfun::strprintf msg("each column of input to the convolution node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", 
                                            NodeName().c_str(), inputDim);
                LogicError(msg.c_str());
            }

            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements() )
                LogicError("Convolution operation: one of the operants has 0 element.");
            
            size_t outputDim = m_outputWidth * m_outputHeight * m_outputChannels;
            FunctionValues().Resize(outputDim, Inputs(1)->FunctionValues().GetNumCols());
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(1, false);

            if (m_inputWidth < m_kernelWidth || m_inputHeight < m_kernelHeight)
                throw std::invalid_argument("inputWidth must >= kernelWidth and inputHeight must >= kernelHeight.");

            if (m_zeroPadding)
            {
                const int kernelWidthCenter = m_kernelWidth % 2;
                const int kernelHeightCenter = m_kernelHeight % 2;
                m_outputWidth = (m_inputWidth-kernelWidthCenter)/m_horizontalSubsample + 1;
                m_outputHeight = (m_inputHeight-kernelHeightCenter)/m_verticalSubsample + 1;
            }
            else
            {
                m_outputWidth = (m_inputWidth-m_kernelWidth)/m_horizontalSubsample + 1;
                m_outputHeight = (m_inputHeight-m_kernelHeight)/m_verticalSubsample + 1;
            }    
        }

        virtual void AttachInputs(const ComputationNodePtr convolutionWeight, const ComputationNodePtr inputFeature) 
        {
            m_children.resize(2);
            m_children[0] = convolutionWeight;
            m_children[1] = inputFeature;
        }

        virtual void MoveMatricesToDevice(const DEVICEID_TYPE deviceId)
        {
            Base::MoveMatricesToDevice(deviceId);
            m_tempMatrix.TransferToDeviceIfNotThereAndNotAutoPlace(deviceId);
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            Base::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", m_inputWidth, m_inputHeight, m_inputChannels);
            fstream << string(str);
            sprintf(str, "Kernel[Width:%lu, Height:%lu]  SubSample[Horizontal:%lu, Vertical:%lu]\n", m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_outputWidth, m_outputHeight, m_outputChannels);
            fstream << string(str);
            sprintf(str, "ZeroPadding=%ls  maxTempMemSizeInSamples=%lu\n", m_zeroPadding? L"true" : L"false", m_maxTempMemSizeInSamples);
            fstream << string(str);
        }

        void SetmMaxTempMemSizeInSamples(const size_t maxTempMemSizeInSamples)
        {
            m_maxTempMemSizeInSamples = maxTempMemSizeInSamples;
        }

    private:
        void ComputeInputPartialOverWeight(Matrix<ElemType> &gradientValues, 
                                           Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &/*input0*/, const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix, const bool inLoop)
        {
            size_t packedInputRows = m_kernelWidth * m_kernelHeight * m_inputChannels;
            size_t packedInputColsPerSample = m_outputWidth * m_outputHeight;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = m_inputWidth * m_inputHeight * m_inputChannels;  //size of each input sample

            long batchSize = (long) input1.GetNumCols(); //right child is the input sample

            long maxTempMemSizeInSamples = (long) (m_maxTempMemSizeInSamples == 0? batchSize : m_maxTempMemSizeInSamples);

            //const Matrix<ElemType> & weightMatrix = input0;
            //inputGradientValues.Resize(weightMatrix.GetNumRows(), weightMatrix.GetNumCols()); //should have been resized when preparing gradient computation

            gradientValues.Reshape(m_outputChannels,  outputSizePerChannel * batchSize);  //reshape to match the longernal operation

            long subBatchSize = min(batchSize, maxTempMemSizeInSamples); 
            long numSubBatches = (batchSize+subBatchSize-1)/subBatchSize; 

            if (numSubBatches == 1 && !inLoop)  //reuse packed input from evaluation step if it's not changed by either subbatch or recurrent steps.
                Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, tempMatrix, true, inputGradientValues);
            else
            {
                for (long i=0; i<numSubBatches; i++) 
                {
                    long startSampleID = i*subBatchSize; 
                    long endSampleID = min(batchSize, startSampleID + subBatchSize); 
                    long smallBatchSize = endSampleID-startSampleID; 

                    tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                    Matrix<ElemType> inputSubBatch = input1.ColumnSlice(startSampleID, smallBatchSize);
                    tempMatrix.AssignPackedConvolutionInput(inputSubBatch, 
                                                                     m_inputWidth, m_inputHeight, m_inputChannels,
                                                                     m_outputWidth, m_outputHeight, m_outputChannels,
                                                                     m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample, 
                                                                     m_zeroPadding); 

                    Matrix<ElemType> outputGradientSubBatch = gradientValues.ColumnSlice(startSampleID * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
                    Matrix<ElemType>::MultiplyAndAdd(outputGradientSubBatch, false, tempMatrix, true, inputGradientValues);
                }
            }

            gradientValues.Reshape(m_outputChannels * outputSizePerChannel, batchSize);  //change back
        }

        //compute gradient over the packed input and then convert the result to the original input
        void ComputeInputPartialOverInputFeature(Matrix<ElemType> &gradientValues, const Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix)
        {
            size_t packedInputRows = m_kernelWidth * m_kernelHeight * m_inputChannels;
            size_t packedInputColsPerSample = m_outputWidth * m_outputHeight;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = m_inputWidth * m_inputHeight * m_inputChannels;  //size of each input sample

            long batchSize = (long) input1.GetNumCols(); //right child is the input sample

            long maxTempMemSizeInSamples = (long) (m_maxTempMemSizeInSamples == 0? batchSize : m_maxTempMemSizeInSamples);

            const Matrix<ElemType> & weightMatrix = input0;

            gradientValues.Reshape(m_outputChannels,  outputSizePerChannel * batchSize);  //reshape to match the longernal operation

            long subBatchSize = min(batchSize, maxTempMemSizeInSamples); 
            long numSubBatches = (batchSize+subBatchSize-1)/subBatchSize; 

            for (long i=0; i<numSubBatches; i++) 
            {
                long startSampleID = i*subBatchSize; 
                long endSampleID = min(batchSize, startSampleID + subBatchSize); 
                long smallBatchSize = endSampleID-startSampleID; 

                tempMatrix.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                Matrix<ElemType> outputGradientSubBatch = gradientValues.ColumnSlice(startSampleID * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
                Matrix<ElemType>::Multiply(weightMatrix, true, outputGradientSubBatch, false,  tempMatrix);

                Matrix<ElemType> inputGradientSubBatch = inputGradientValues.ColumnSlice(startSampleID, smallBatchSize);
                tempMatrix.UnpackConvolutionInput(inputGradientSubBatch, 
                                                  m_inputWidth, m_inputHeight, m_inputChannels,
                                                  m_outputWidth, m_outputHeight, m_outputChannels,
                                                  m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample, 
                                                  m_zeroPadding); 
            }

            gradientValues.Reshape(m_outputChannels * outputSizePerChannel, batchSize);  //change back
        }
        

    private:
        size_t m_kernelWidth, m_kernelHeight;
        size_t m_horizontalSubsample, m_verticalSubsample;
        bool m_zeroPadding;

        Matrix<ElemType> m_tempMatrix; 
        size_t m_maxTempMemSizeInSamples; // can change during runtime
    };

    template class ConvolutionNode<float>; 
    template class ConvolutionNode<double>;

    // -----------------------------------------------------------------------
    // PoolingNodeBase
    // -----------------------------------------------------------------------

    //Max/Average Pooling: support multi channel
    //assume each column is an input sample. Each sample is stored in  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
    template<class ElemType>
    class PoolingNodeBase : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        PoolingNodeBase(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_windowWidth(SIZE_MAX), m_windowHeight(SIZE_MAX),
            m_horizontalSubsample(SIZE_MAX), m_verticalSubsample(SIZE_MAX)
        { }
        PoolingNodeBase(DEVICEID_TYPE deviceId, const wstring & name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample) :
            ComputationNode<ElemType>(deviceId, name),
            m_windowWidth(windowWidth), m_windowHeight(windowHeight),
            m_horizontalSubsample(horizontalSubsample), m_verticalSubsample(verticalSubsample)
        { }

        virtual void SaveToFile(File& fstream) const
        {
            Base::SaveToFile(fstream);
            fstream << m_windowWidth << m_windowHeight << m_horizontalSubsample << m_verticalSubsample;
        }

        virtual void LoadFromFile(File& fstream, size_t modelVersion)
        {
            Base::LoadFromFile(fstream, modelVersion);
            fstream >> m_windowWidth >> m_windowHeight >> m_horizontalSubsample >> m_verticalSubsample;
        }

        virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
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

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("MaxPooling operation only takes one inputs.");

            Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialV(sliceOutputGrad, sliceInput0Grad, sliceInput0Value, sliceOutputValue);
        }

        // this function must be overriden by Max or AveragePoolingNode
        virtual void ComputeInputPartialV(const Matrix<ElemType> &gradientValues, Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &functionValues) = 0;

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange)
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.FrameSlice(frameRange/*TODO: delete the next two parameters*/, frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            EvaluateThisNodeV(sliceOutputValue, sliceInput0Value);
        }

        // this function must be overriden by Max or AveragePoolingNode
        virtual void EvaluateThisNodeV(Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0) = 0;

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1)
                LogicError("PoolingNodes require one input.");

            if (m_horizontalSubsample > m_windowWidth || m_verticalSubsample > m_windowHeight)
                InvalidArgument("PoolingNodeBase: horizontalSubsample must <= windowWidth and verticalSubsample must <= windowHeight.");

            InferImageDimsFromInputs();

            m_inputSizePerSample = m_inputWidth * m_inputHeight * m_inputChannels;
            m_outputSizePerSample = m_outputWidth * m_outputHeight * m_outputChannels;

            if (Inputs(0)->OperationName() == OperationNameOf(LearnableParameter) && Inputs(0)->FunctionValues().GetNumRows() == 0)
                Inputs(0)->FunctionValues().Resize(m_inputSizePerSample, Inputs(0)->FunctionValues().GetNumCols());

            if (Inputs(0)->FunctionValues().GetNumRows() != m_inputSizePerSample)
            {
                msra::strfun::strprintf msg("each column of input to the MaxPooling node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", NodeName().c_str(), m_inputSizePerSample);
                LogicError(msg.c_str());
            }

            if (Inputs(0)->FunctionValues().HasNoElements())
                LogicError("PoolingNodeBase operation: the input node has 0 element.");

            m_functionValues.Resize(m_outputSizePerSample, Inputs(0)->FunctionValues().GetNumCols());
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            if (m_inputWidth < m_windowWidth || m_inputHeight < m_windowHeight)
                throw std::invalid_argument("PoolingNodeBase: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

            m_outputWidth = (m_inputWidth - m_windowWidth) / m_horizontalSubsample + 1;
            m_outputHeight = (m_inputHeight - m_windowHeight) / m_verticalSubsample + 1;
            m_outputChannels = m_inputChannels;
        }

        virtual void AttachInputs(const ComputationNodePtr inputFeature)
        {
            m_children.resize(1);
            m_children[0] = inputFeature;
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const
        {
            Base::DumpNodeInfo(printValues, fstream);

            char str[4096];
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", m_inputWidth, m_inputHeight, m_inputChannels);
            fstream << string(str);
            sprintf(str, "PoolingWindow[Width:%lu, Height:%lu]  SubSampling[Horizontal:%lu, Vertical:%lu]\n", m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_outputWidth, m_outputHeight, m_outputChannels);
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
    // See #define of 'UsingComputationNodeMembers' for more explanation.
#define UsingPoolingNodeBaseMembers UsingComputationNodeMembers; \
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
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        MaxPoolingNode(DEVICEID_TYPE deviceId, const wstring & name) : Base(deviceId, name) { }
        MaxPoolingNode(DEVICEID_TYPE deviceId, const wstring & name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample) :
            Base(deviceId, name, windowWidth, windowHeight, horizontalSubsample, verticalSubsample)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"MaxPooling";}

        /*implement*/ void ComputeInputPartialV(const Matrix<ElemType> &gradientValues, Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &functionValues)
        {
            inputGradientValues.AddMaxPoolingGradient(gradientValues, input0, functionValues, m_inputChannels,
                                                      m_inputWidth, m_inputHeight, m_inputSizePerSample, 
                                                      m_outputWidth, m_outputHeight, m_outputSizePerSample, 
                                                      m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
        }

        /*implement*/ void EvaluateThisNodeV(Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0)
        {
            functionValues.AssignMaxPoolingResult(input0, m_inputChannels,
                                                  m_inputWidth, m_inputHeight, m_inputSizePerSample, 
                                                  m_outputWidth, m_outputHeight, m_outputSizePerSample, 
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
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new typename std::remove_reference<decltype(*this)>::type(deviceId, name); }
        AveragePoolingNode(DEVICEID_TYPE deviceId, const wstring & name) : Base(deviceId, name) { }
        AveragePoolingNode(DEVICEID_TYPE deviceId, const wstring & name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample) :
            Base(deviceId, name, windowWidth, windowHeight, horizontalSubsample, verticalSubsample)
        { }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"AveragePooling";}

        /*implement*/ void ComputeInputPartialV(const Matrix<ElemType> &gradientValues, Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &/*input0*/, const Matrix<ElemType> &/*functionValues*/)
        {
            inputGradientValues.AddAveragePoolingGradient(gradientValues, m_inputChannels,
                                                          m_inputWidth, m_inputHeight, m_inputSizePerSample, 
                                                          m_outputWidth, m_outputHeight, m_outputSizePerSample, 
                                                          m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
        }

        /*implement*/ void EvaluateThisNodeV(Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0)
        {
            functionValues.AssignAveragePoolingResult(input0, m_inputChannels,
                                                      m_inputWidth, m_inputHeight, m_inputSizePerSample, 
                                                      m_outputWidth, m_outputHeight, m_outputSizePerSample, 
                                                      m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
        }
    };

    template class AveragePoolingNode<float>; 
    template class AveragePoolingNode<double>;    

}}}
