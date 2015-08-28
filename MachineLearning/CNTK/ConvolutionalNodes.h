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

namespace Microsoft { namespace MSR { namespace CNTK {

    // convolution parameters structure, to make it easier to pass these around all these parameters
    struct ConvolutionParams
    {
        size_t inputWidth, inputHeight, inputChannels;
        size_t kernelWidth, kernelHeight;
        size_t horizontalSubsample, verticalSubsample;
        size_t outputWidth, outputHeight, outputChannels;
        size_t maxTempMemSizeInSamples;
        bool zeroPadding;
    };

    //convolutional network 
    //follow "high performance convolutional neural networks for document processing" by Kumar chellapilla, Sidde Puri, and Patrice Simard
    //assume each column is an input sample. Each sample is stored in [channel, row, col]  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
    template<class ElemType>
    class ConvolutionNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new std::remove_reference<decltype(*this)>::type(deviceId, name); }
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

        ConvolutionParams GetConvolutionParams() const
        {
            ConvolutionParams convParam;
            convParam.inputWidth = m_inputWidth;
            convParam.inputHeight = m_inputHeight;
            convParam.inputChannels = m_inputChannels;

            convParam.kernelWidth = m_kernelWidth;
            convParam.kernelHeight = m_kernelHeight;

            convParam.horizontalSubsample = m_horizontalSubsample;
            convParam.verticalSubsample = m_verticalSubsample;

            convParam.outputWidth = m_outputWidth;
            convParam.outputHeight = m_outputHeight;
            convParam.outputChannels = m_outputChannels;

            convParam.zeroPadding = m_zeroPadding;

            convParam.maxTempMemSizeInSamples = m_maxTempMemSizeInSamples;
            return convParam;
        }

        virtual void ComputeInputPartial(const size_t inputIndex) 
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Convolution operation only takes two inputs.");

            if (inputIndex == 0)  //derivative with regard to the weight matrix
            {
                ComputeInputPartialOverWeight(this, GradientValues(), Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_tempMatrix, true);
            }
            else  // derivative with regard to the input feature
            {
                ComputeInputPartialOverInputFeature(this, GradientValues(), Inputs(1)->GradientValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_tempMatrix);
            }
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) 
        {
            if (inputIndex > 1)
                throw std::invalid_argument("Convolution operation only takes two inputs.");

            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            if (inputIndex == 0)  //derivative with regard to the weight matrix
            {
                ComputeInputPartialOverWeight(this, sliceOutputGrad, Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), sliceInput1Value, m_tempMatrix);
            }
            else  // derivative with regard to the input feature
            {
                Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientValues().ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
                ComputeInputPartialOverInputFeature(this, sliceOutputGrad, sliceInput1Grad, Inputs(0)->FunctionValues(), sliceInput1Value, m_tempMatrix);
            }
        }

        virtual void EvaluateThisNode()  
        {
            EvaluateThisNodeS(this, FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_tempMatrix);
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) 
        {
            Matrix<ElemType> sliceInput1Value = Inputs(1)->FunctionValues().ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(this, sliceOutputValue, Inputs(0)->FunctionValues(), sliceInput1Value, m_tempMatrix);
        }

        static void WINAPI EvaluateThisNodeS(const ConvolutionNode<ElemType>* pConv, Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0, 
            const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix)
        {
#if NANCHECK
            input0.HasNan("Convolution-input0");
            input1.HasNan("Convolution-input1");
#endif
            ConvolutionParams convParam = pConv->GetConvolutionParams();

            size_t packedInputRows = convParam.kernelWidth * convParam.kernelHeight * convParam.inputChannels;
            size_t packedInputColsPerSample = convParam.outputWidth * convParam.outputHeight;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = convParam.inputWidth * convParam.inputHeight * convParam.inputChannels;  //size of each input sample

            long batchSize = (long)input1.GetNumCols();  //right child is the input sample

            long maxTempMemSizeInSamples = (long)(convParam.maxTempMemSizeInSamples == 0? batchSize : convParam.maxTempMemSizeInSamples);

            const Matrix<ElemType> & weightMatrix = input0;
            assert(weightMatrix.GetNumCols() == packedInputRows && weightMatrix.GetNumRows() == convParam.outputChannels);
            functionValues.Resize(convParam.outputChannels, outputSizePerChannel * batchSize);

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
                                                                 convParam.inputWidth, convParam.inputHeight, convParam.inputChannels,
                                                                 convParam.outputWidth, convParam.outputHeight, convParam.outputChannels,
                                                                 convParam.kernelWidth, convParam.kernelHeight, convParam.horizontalSubsample, convParam.verticalSubsample, 
                                                                 convParam.zeroPadding); 

                Matrix<ElemType>  outputSubBatch = functionValues.ColumnSlice(outputSizePerChannel * startSampleID, outputSizePerChannel * smallBatchSize);
                Matrix<ElemType>::Multiply(weightMatrix, false, tempMatrix, false, outputSubBatch);
            }

            functionValues.Reshape(convParam.outputChannels * outputSizePerChannel, batchSize);  //each sample becomes a column

#if NANCHECK
            functionValues.HasNan("Convolution");
#endif
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 2) 
                throw std::logic_error("ConvolutionNode requires two inputs.");

            //we may want to remove this check in the future if we want to support the case that the weight itself is result of some computation 
            //if (Inputs(0)->OperationName() != LearnableParameter<ElemType>::TypeName())
            //    throw std::logic_error("ConvolutionNode requires the first input to be LearnableParameter type.");

            if (m_horizontalSubsample > m_kernelWidth || m_verticalSubsample > m_kernelHeight)
                throw std::invalid_argument("In ConvolutionNode horizontalSubsample must <= kernelWidth and verticalSubsample must <= kernelHeight.");

            InferImageDimsFromInputs();

            size_t weightCols = m_kernelWidth * m_kernelHeight * m_inputChannels;

            if (Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(0)->FunctionValues().HasNoElements())
            {
                Inputs(0)->FunctionValues().Resize(m_outputChannels, weightCols);
            }

            if (m_children[0]->FunctionValues().GetNumCols() != weightCols || m_children[0]->FunctionValues().GetNumRows() != m_outputChannels)
            {
                msra::strfun::strprintf msg("convolutionWeight matrix %ls should have dimension [%d, %d] which is [outputChannels, kernelWidth * kernelHeight * inputChannels]", 
                    m_children[0]->NodeName().c_str(), m_outputChannels, weightCols);
                throw std::logic_error(msg.c_str());            
            }

            size_t inputDim = m_inputWidth * m_inputHeight * m_inputChannels;
            if (Inputs(1)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(1)->FunctionValues().GetNumRows() == 0)
            {
                Inputs(1)->FunctionValues().Resize(inputDim, Inputs(1)->FunctionValues().GetNumCols());
            }

            if (m_children[1]->FunctionValues().GetNumRows() != inputDim)
            {
                msra::strfun::strprintf msg("each column of input to the convolution node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", 
                    NodeName().c_str(), inputDim);
                throw std::logic_error(msg.c_str());            
            }

            if (Inputs(0)->FunctionValues().HasNoElements() || Inputs(1)->FunctionValues().HasNoElements() )
                throw std::logic_error("Convolution operation: one of the operants has 0 element.");
            
            size_t outputDim = m_outputWidth * m_outputHeight * m_outputChannels;
            FunctionValues().Resize(outputDim, m_children[1]->FunctionValues().GetNumCols());
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
        static void WINAPI ComputeInputPartialOverWeight(const ConvolutionNode<ElemType>* pConv, Matrix<ElemType> &gradientValues, 
            Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &/*input0*/, const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix, const bool inLoop=false)
        {
            ConvolutionParams convParam = pConv->GetConvolutionParams();

            size_t packedInputRows = convParam.kernelWidth * convParam.kernelHeight * convParam.inputChannels;
            size_t packedInputColsPerSample = convParam.outputWidth * convParam.outputHeight;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = convParam.inputWidth * convParam.inputHeight * convParam.inputChannels;  //size of each input sample

            long batchSize = (long) input1.GetNumCols(); //right child is the input sample

            long maxTempMemSizeInSamples = (long) (convParam.maxTempMemSizeInSamples == 0? batchSize : convParam.maxTempMemSizeInSamples);

            //const Matrix<ElemType> & weightMatrix = input0;
            //inputGradientValues.Resize(weightMatrix.GetNumRows(), weightMatrix.GetNumCols()); //should have been resized when preparing gradient computation

            gradientValues.Reshape(convParam.outputChannels,  outputSizePerChannel * batchSize);  //reshape to match the longernal operation

            long subBatchSize = min(batchSize, maxTempMemSizeInSamples); 
            long numSubBatches = (batchSize+subBatchSize-1)/subBatchSize; 

            if (numSubBatches == 1 && !inLoop)  //reuse packed input from evaluation step if it's not changed by either subbatch or recurrent steps.
            {
                Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, tempMatrix, true, inputGradientValues);
            }
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
                                                                     convParam.inputWidth, convParam.inputHeight, convParam.inputChannels,
                                                                     convParam.outputWidth, convParam.outputHeight, convParam.outputChannels,
                                                                     convParam.kernelWidth, convParam.kernelHeight, convParam.horizontalSubsample, convParam.verticalSubsample, 
                                                                     convParam.zeroPadding); 

                    Matrix<ElemType> outputGradientSubBatch = gradientValues.ColumnSlice(startSampleID * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
                    Matrix<ElemType>::MultiplyAndAdd(outputGradientSubBatch, false, tempMatrix, true, inputGradientValues);
                }
            }

            gradientValues.Reshape(convParam.outputChannels * outputSizePerChannel, batchSize);  //change back
        }

        //compute gradient over the packed input and then convert the result to the original input
        static void WINAPI ComputeInputPartialOverInputFeature(const ConvolutionNode<ElemType>* pConv, Matrix<ElemType> &gradientValues, const Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &input1, Matrix<ElemType> &tempMatrix)
        {
            
            ConvolutionParams convParam = pConv->GetConvolutionParams();
            size_t packedInputRows = convParam.kernelWidth * convParam.kernelHeight * convParam.inputChannels;
            size_t packedInputColsPerSample = convParam.outputWidth * convParam.outputHeight;
            size_t outputSizePerChannel = packedInputColsPerSample;
            //size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
            //size_t inputDim = convParam.inputWidth * convParam.inputHeight * convParam.inputChannels;  //size of each input sample

            long batchSize = (long) input1.GetNumCols(); //right child is the input sample

            long maxTempMemSizeInSamples = (long) (convParam.maxTempMemSizeInSamples == 0? batchSize : convParam.maxTempMemSizeInSamples);

            const Matrix<ElemType> & weightMatrix = input0;

            gradientValues.Reshape(convParam.outputChannels,  outputSizePerChannel * batchSize);  //reshape to match the longernal operation

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
                                                                 convParam.inputWidth, convParam.inputHeight, convParam.inputChannels,
                                                                 convParam.outputWidth, convParam.outputHeight, convParam.outputChannels,
                                                                 convParam.kernelWidth, convParam.kernelHeight, convParam.horizontalSubsample, convParam.verticalSubsample, 
                                                                 convParam.zeroPadding); 
            }

            gradientValues.Reshape(convParam.outputChannels * outputSizePerChannel, batchSize);  //change back
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

    struct PoolParams
    {
        size_t inputWidth, inputHeight, inputChannels;
        size_t windowWidth, windowHeight;
        size_t horizontalSubsample, verticalSubsample;
        size_t outputWidth, outputHeight, outputChannels;
        size_t inputSizePerSample, outputSizePerSample;
    };

    //Max Pooling: support multi channel
    //assume each column is an input sample. Each sample is stored in  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
    template<class ElemType>
    class MaxPoolingNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new std::remove_reference<decltype(*this)>::type(deviceId, name); }
        MaxPoolingNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_windowWidth(SIZE_MAX), m_windowHeight(SIZE_MAX),
            m_horizontalSubsample(SIZE_MAX), m_verticalSubsample(SIZE_MAX)
        { }
        MaxPoolingNode(DEVICEID_TYPE deviceId, const wstring & name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample) :
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
                auto node = dynamic_pointer_cast<MaxPoolingNode<ElemType>>(nodeP);
                node->m_inputWidth = m_inputWidth;
                node->m_inputHeight = m_inputHeight;
                node->m_inputChannels = m_inputChannels;

                node->m_windowWidth = m_windowWidth;
                node->m_windowHeight = m_windowHeight;

                node->m_horizontalSubsample = m_horizontalSubsample;
                node->m_verticalSubsample = m_verticalSubsample;

                node->m_outputWidth = m_outputWidth;
                node->m_outputHeight = m_outputHeight;
                node->m_outputChannels = m_outputChannels;

                node->m_inputSizePerSample = m_inputSizePerSample;
                node->m_outputSizePerSample = m_outputSizePerSample;
            }
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"MaxPooling";}

        PoolParams GetPoolParams() const
        {
            PoolParams poolParams;
            poolParams.inputWidth = m_inputWidth;
            poolParams.inputHeight = m_inputHeight;
            poolParams.inputChannels = m_inputChannels;

            poolParams.windowWidth = m_windowWidth;
            poolParams.windowHeight = m_windowHeight;

            poolParams.horizontalSubsample = m_horizontalSubsample;
            poolParams.verticalSubsample = m_verticalSubsample;

            poolParams.outputWidth = m_outputWidth;
            poolParams.outputHeight = m_outputHeight;
            poolParams.outputChannels = m_outputChannels;

            poolParams.inputSizePerSample = m_inputSizePerSample;
            poolParams.outputSizePerSample = m_outputSizePerSample;
            return poolParams;
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("MaxPooling operation only takes one inputs.");

            ComputeInputPartialS(this, GradientValues(), Inputs(0)->GradientValues(), Inputs(0)->FunctionValues(), FunctionValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) 
        {
            if (inputIndex > 0)
                throw std::invalid_argument("MaxPooling operation only takes one inputs.");

            Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            ComputeInputPartialS(this, sliceOutputGrad, sliceInput0Grad, sliceInput0Value, sliceOutputValue);
        }

        static void WINAPI ComputeInputPartialS(const MaxPoolingNode<ElemType>* ppool, const Matrix<ElemType> &gradientValues, Matrix<ElemType> &inputGradientValues, const Matrix<ElemType> &input0, const Matrix<ElemType> &functionValues)
        {
            PoolParams poolParams = ppool->GetPoolParams();

            inputGradientValues.AddMaxPoolingGradient(gradientValues, input0, functionValues, poolParams.inputChannels,
                                                    poolParams.inputWidth, poolParams.inputHeight, poolParams.inputSizePerSample, 
                                                    poolParams.outputWidth, poolParams.outputHeight, poolParams.outputSizePerSample, 
                                                    poolParams.windowWidth, poolParams.windowHeight, poolParams.horizontalSubsample, poolParams.verticalSubsample);
        }

        virtual void EvaluateThisNode()  
        {
#if NANCHECK
            Inputs(0)->FunctionValues().HasNan("MaxPooling-input0");
#endif
            EvaluateThisNodeS(this, FunctionValues(), Inputs(0)->FunctionValues());
#if NANCHECK
            m_functionValues.HasNan("MaxPooling");
#endif
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) 
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(this, sliceOutputValue, sliceInput0Value);
        }

        static void WINAPI EvaluateThisNodeS(const MaxPoolingNode<ElemType>* ppool, Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0)
        {
            PoolParams poolParams = ppool->GetPoolParams();
            functionValues.AssignMaxPoolingResult(input0, poolParams.inputChannels,
                                                 poolParams.inputWidth, poolParams.inputHeight, poolParams.inputSizePerSample, 
                                                 poolParams.outputWidth, poolParams.outputHeight, poolParams.outputSizePerSample, 
                                                 poolParams.windowWidth, poolParams.windowHeight, poolParams.horizontalSubsample, poolParams.verticalSubsample);
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("MaxPoolingNode requires one input.");

            if (m_horizontalSubsample > m_windowWidth || m_verticalSubsample > m_windowHeight)
                throw std::invalid_argument("MaxPoolingNode: horizontalSubsample must <= windowWidth and verticalSubsample must <= windowHeight.");

            InferImageDimsFromInputs();

            m_inputSizePerSample = m_inputWidth * m_inputHeight * m_inputChannels;
            m_outputSizePerSample = m_outputWidth * m_outputHeight * m_outputChannels;

            if (Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(0)->FunctionValues().GetNumRows() == 0)
            {
                Inputs(0)->FunctionValues().Resize(m_inputSizePerSample, Inputs(0)->FunctionValues().GetNumCols());
            }

            if (m_children[0]->FunctionValues().GetNumRows() != m_inputSizePerSample)
            {
                msra::strfun::strprintf msg("each column of input to the MaxPooling node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", 
                    NodeName().c_str(), m_inputSizePerSample);
                throw std::logic_error(msg.c_str());            
            }
            
            if (Inputs(0)->FunctionValues().HasNoElements())
                throw std::logic_error("MaxPoolingNode operation: the input node has 0 element.");

            m_functionValues.Resize(m_outputSizePerSample, m_children[0]->FunctionValues().GetNumCols());
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            if (m_inputWidth < m_windowWidth || m_inputHeight < m_windowHeight)
                throw std::invalid_argument("MaxPoolingNode: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

            m_outputWidth = (m_inputWidth-m_windowWidth)/m_horizontalSubsample + 1;
            m_outputHeight = (m_inputHeight-m_windowHeight)/m_verticalSubsample + 1;
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

    private:
        size_t m_windowWidth, m_windowHeight;
        size_t m_horizontalSubsample, m_verticalSubsample;
        size_t m_inputSizePerSample, m_outputSizePerSample;
    };

    template class MaxPoolingNode<float>; 
    template class MaxPoolingNode<double>;    

    //Average Pooling: support multi channel
    //assume each column is an input sample. Each sample is stored in  (r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11)
    template<class ElemType>
    class AveragePoolingNode : public ComputationNode<ElemType>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;
    public:
        virtual ComputationNode<ElemType> * NewThis(DEVICEID_TYPE deviceId, const wstring & name) { return new std::remove_reference<decltype(*this)>::type(deviceId, name); }
        AveragePoolingNode(DEVICEID_TYPE deviceId, const wstring & name) :
            ComputationNode<ElemType>(deviceId, name),
            m_windowWidth(SIZE_MAX), m_windowHeight(SIZE_MAX),
            m_horizontalSubsample(SIZE_MAX), m_verticalSubsample(SIZE_MAX)
        { }
        AveragePoolingNode(DEVICEID_TYPE deviceId, const wstring & name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample) :
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
                auto node = dynamic_pointer_cast<AveragePoolingNode<ElemType>>(nodeP);
                node->m_inputWidth = m_inputWidth;
                node->m_inputHeight = m_inputHeight;
                node->m_inputChannels = m_inputChannels;

                node->m_windowWidth = m_windowWidth;
                node->m_windowHeight = m_windowHeight;

                node->m_horizontalSubsample = m_horizontalSubsample;
                node->m_verticalSubsample = m_verticalSubsample;

                node->m_outputWidth = m_outputWidth;
                node->m_outputHeight = m_outputHeight;
                node->m_outputChannels = m_outputChannels;

                node->m_inputSizePerSample = m_inputSizePerSample;
                node->m_outputSizePerSample = m_outputSizePerSample;
            }
        }

        virtual const std::wstring OperationName() const {return TypeName();}
        static const std::wstring TypeName() {return L"AveragePooling";}
        PoolParams GetPoolParams() const
        {
            PoolParams poolParams;
            poolParams.inputWidth = m_inputWidth;
            poolParams.inputHeight = m_inputHeight;
            poolParams.inputChannels = m_inputChannels;

            poolParams.windowWidth = m_windowWidth;
            poolParams.windowHeight = m_windowHeight;

            poolParams.horizontalSubsample = m_horizontalSubsample;
            poolParams.verticalSubsample = m_verticalSubsample;

            poolParams.outputWidth = m_outputWidth;
            poolParams.outputHeight = m_outputHeight;
            poolParams.outputChannels = m_outputChannels;

            poolParams.inputSizePerSample = m_inputSizePerSample;
            poolParams.outputSizePerSample = m_outputSizePerSample;
            return poolParams;
        }

        virtual void ComputeInputPartial(const size_t inputIndex)
        {
            if (inputIndex > 0)
                throw std::invalid_argument("AveragePooling operation only takes one inputs.");

            ComputeInputPartialS(this, GradientValues(), Inputs(0)->GradientValues());
        }

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) 
        {
            if (inputIndex > 0)
                throw std::invalid_argument("AveragePooling operation only takes one inputs.");

            Matrix<ElemType> sliceInput0Grad = Inputs(0)->GradientValues().ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputGrad = GradientValues().ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            ComputeInputPartialS(this, sliceOutputGrad, sliceInput0Grad);
        }

        static void WINAPI ComputeInputPartialS(const AveragePoolingNode<ElemType>* ppool, const Matrix<ElemType> &gradientValues, Matrix<ElemType> &inputGradientValues)
        {
            PoolParams poolParams = ppool->GetPoolParams();

            inputGradientValues.AddAveragePoolingGradient(gradientValues, poolParams.inputChannels,
                                                    poolParams.inputWidth, poolParams.inputHeight, poolParams.inputSizePerSample, 
                                                    poolParams.outputWidth, poolParams.outputHeight, poolParams.outputSizePerSample, 
                                                    poolParams.windowWidth, poolParams.windowHeight, poolParams.horizontalSubsample, poolParams.verticalSubsample);
        }

        virtual void EvaluateThisNode()  
        {
#if NANCHECK
            Inputs(0)->FunctionValues().HasNan("AveragePooling-input0");
#endif
            EvaluateThisNodeS(this, FunctionValues(), Inputs(0)->FunctionValues());
#if NANCHECK
            m_functionValues.HasNan("AveragePooling");
#endif
        }

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) 
        {
            Matrix<ElemType> sliceInput0Value = Inputs(0)->FunctionValues().ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);
            Matrix<ElemType> sliceOutputValue = m_functionValues.ColumnSlice(frameRange.t() * m_samplesInRecurrentStep, m_samplesInRecurrentStep);

            EvaluateThisNodeS(this, sliceOutputValue, sliceInput0Value);
        }

        static void WINAPI EvaluateThisNodeS(const AveragePoolingNode<ElemType>* ppool, Matrix<ElemType> &functionValues, const Matrix<ElemType> &input0)
        {
            PoolParams poolParams = ppool->GetPoolParams();
            
            functionValues.AssignAveragePoolingResult(input0, poolParams.inputChannels,
                                                 poolParams.inputWidth, poolParams.inputHeight, poolParams.inputSizePerSample, 
                                                 poolParams.outputWidth, poolParams.outputHeight, poolParams.outputSizePerSample, 
                                                 poolParams.windowWidth, poolParams.windowHeight, poolParams.horizontalSubsample, poolParams.verticalSubsample);
        }

        virtual void Validate()
        {
            PrintSelfBeforeValidation();

            if (m_children.size() != 1) 
                throw std::logic_error("AveragePoolingNode requires one input.");

            if (m_horizontalSubsample > m_windowWidth || m_verticalSubsample > m_windowHeight)
                throw std::invalid_argument("AveragePoolingNode: horizontalSubsample must <= windowWidth and verticalSubsample must <= windowHeight.");

            InferImageDimsFromInputs();

            m_inputSizePerSample = m_inputWidth * m_inputHeight * m_inputChannels;
            m_outputSizePerSample = m_outputWidth * m_outputHeight * m_outputChannels;

            if (Inputs(0)->OperationName() == LearnableParameter<ElemType>::TypeName() && Inputs(0)->FunctionValues().GetNumRows() == 0)
            {
                Inputs(0)->FunctionValues().Resize(m_inputSizePerSample, Inputs(0)->FunctionValues().GetNumCols());
            }

            if (m_children[0]->FunctionValues().GetNumRows() != m_inputSizePerSample)
            {
                msra::strfun::strprintf msg("each column of input to the AveragePooling node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", 
                    NodeName().c_str(), m_inputSizePerSample);
                throw std::logic_error(msg.c_str());            
            }
                        
            if (Inputs(0)->FunctionValues().HasNoElements())
                throw std::logic_error("AveragePoolingNode operation: the input node has 0 element.");

            FunctionValues().Resize(m_outputSizePerSample, m_children[0]->FunctionValues().GetNumCols());
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            if (m_inputWidth < m_windowWidth || m_inputHeight < m_windowHeight)
                throw std::invalid_argument("AveragePoolingNode: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

            m_outputWidth = (m_inputWidth-m_windowWidth)/m_horizontalSubsample + 1;
            m_outputHeight = (m_inputHeight-m_windowHeight)/m_verticalSubsample + 1;
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
            sprintf(str, "PoolingWindow[Width:%lu, Height:%lu]  SubSample[Horizontal:%lu, Vertical:%lu]\n", m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_outputWidth, m_outputHeight, m_outputChannels);
            fstream << string(str);
            sprintf(str, "TotalSizePerSample[Input:%lu, Output:%lu]\n", m_inputSizePerSample, m_outputSizePerSample);
            fstream << string(str);
        }

    private:
        size_t m_windowWidth, m_windowHeight;
        size_t m_horizontalSubsample, m_verticalSubsample;
        size_t m_inputSizePerSample, m_outputSizePerSample;
    };

    template class AveragePoolingNode<float>; 
    template class AveragePoolingNode<double>;    

}}}
