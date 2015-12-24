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
#include "ConvolutionEngine.h"

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
            m_sampleLayout = ImageLayoutWHC(1, 1, 0);           // TODO: what is this magic #channels == 0? Can this even be initialized at this time, or only inferred?
        }
        ConvolutionNode(DEVICEID_TYPE deviceId, const wstring & name, const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels, const size_t horizontalSubsample, const size_t verticalSubsample, const bool zeroPadding = false, const size_t maxTempMemSizeInSamples = 0) :
            Base(deviceId, name),
            m_kernelWidth(kernelWidth), m_kernelHeight(kernelHeight),
            m_horizontalSubsample(horizontalSubsample), m_verticalSubsample(verticalSubsample),
            m_zeroPadding(zeroPadding), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples)
        {
            m_sampleLayout = ImageLayoutWHC(1, 1, outputChannels);
            m_factory = ConvolutionEngineFactory<ElemType>::Create(deviceId);
        }
        ConvolutionNode(const ScriptableObjects::IConfigRecordPtr configp) :
            ConvolutionNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"kernelWidth"), configp->Get(L"kernelHeight"), configp->Get(L"outputChannels"),
                            configp->Get(L"horizontalSubsample"), configp->Get(L"verticalSubsample"),
                            configp->Get(L"zeroPadding"), configp->Get(L"maxTempMemSizeInSamples"))
        {
            // weightNodeName, inputValueNodeName, kernelWidth, kernelHeight, outputChannels, horizontalSubsample, verticalSubsample, zeroPadding = false, maxTempMemSizeInSamples = 0
            AttachInputs(configp, this->GetExpectedNumInputs());
        }

        void Save(File& fstream) const override
        {
            Base::Save(fstream);
            fstream <<  m_kernelWidth << m_kernelHeight << m_horizontalSubsample << m_verticalSubsample;
            fstream << m_sampleLayout.GetNumChannels();
            fstream << m_zeroPadding << m_maxTempMemSizeInSamples;
        }

        void Load(File& fstream, size_t modelVersion) override
        {
            Base::Load(fstream, modelVersion);
            fstream >> m_kernelWidth >> m_kernelHeight >> m_horizontalSubsample >> m_verticalSubsample; 
            size_t outputChannels;
            fstream >> outputChannels;
            m_sampleLayout = ImageLayoutWHC(1, 1, outputChannels);
            fstream >> m_zeroPadding >> m_maxTempMemSizeInSamples;
        }

        void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
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

        void BackpropTo(const size_t inputIndex, const FrameRange & fr) override
        {
            Matrix<ElemType> sliceOutputGrad = GradientFor(fr);
            Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);

            size_t batchSize = sliceInput1Value.GetNumCols();
            m_inT->setN(batchSize);
            m_outT->setN(batchSize);
            assert(m_convEng != nullptr);
            if (inputIndex == 0)  //derivative with respect to the weight matrix
            {
                Matrix<ElemType>& grad = Input(0)->Gradient();
                m_convEng->BackwardFilter(*m_outT, sliceOutputGrad, *m_inT, sliceInput1Value, *m_convDesc, *m_filterT, grad, fr.IsAllFrames(), *m_tempMatrix);
            }
            else if (inputIndex == 1)  // derivative with respect to the input feature
            {
                const Matrix<ElemType>& input0 = Input(0)->Value();
                Matrix<ElemType> sliceInput1Grad = Input(1)->GradientFor(fr);
                m_convEng->BackwardData(*m_outT, sliceOutputGrad, *m_filterT, input0, *m_convDesc, *m_inT, sliceInput1Grad, *m_tempMatrix);
            }
        }

        virtual bool OutputUsedInComputingInputNodesGradients() const override
        {
            // The ConvolutionNode does not require its output value for computing
            // the gradients of its input nodes
            return false;
        }

        void ForwardProp(const FrameRange & fr) override
        {
            const Matrix<ElemType>& input0 = Input(0)->Value();
            Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
            Matrix<ElemType> sliceOutputValue = ValueFor(fr);

            // REVIEW alexeyk: setting batch size, can it be done elsewhere in a single place?
            size_t batchSize = sliceInput1Value.GetNumCols();
            m_inT->setN(batchSize);
            m_outT->setN(batchSize);
            assert(m_convEng != nullptr);
#if NANCHECK
            input0.HasNan("Convolution-input0");
            sliceInput1Value.HasNan("Convolution-input1");
#endif
            m_convEng->Forward(*m_inT, sliceInput1Value, *m_filterT, input0, *m_convDesc, *m_outT, sliceOutputValue, *m_tempMatrix);
#if NANCHECK
            sliceOutputValue.HasNan("Convolution");
#endif
        }

        void AddBias(const Matrix<ElemType>& output, const Matrix<ElemType>& bias, Matrix<ElemType>& dst)
        {
            assert(m_convEng != nullptr);
            m_convEng->AddBias(*m_outT, output, *m_biasT, bias, dst);
        }

        void BackwardBias(const Matrix<ElemType>& srcGrad, Matrix<ElemType>& biasGrad)
        {
            assert(m_convEng != nullptr);
            m_convEng->BackwardBias(*m_outT, srcGrad, *m_biasT, biasGrad);
        }

        // note: this also infers dimensions from chilren
        void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);
            InferMBLayoutFromInputsForStandardCase();

            InferImageDimsFromInputs();

            auto inputSampleLayout = GetInputSampleLayout(1);

            // TODO: SetDims(layout)
            size_t weightCols = m_kernelWidth * m_kernelHeight * inputSampleLayout.GetNumChannels();

            if (Input(0)->Value().HasNoElements())
                ValidateInferInputDims(0, m_sampleLayout.GetNumChannels(), weightCols);

            if (isFinalValidationPass && (Input(0)->GetNumCols() != weightCols || Input(0)->GetNumRows() != m_sampleLayout.GetNumChannels()))
                LogicError("convolutionWeight matrix %ls should have dimension [%d, %d] which is [outputChannels, kernelWidth * kernelHeight * inputChannels]", m_inputs[0]->NodeName().c_str(), (int)m_sampleLayout.GetNumChannels(), (int)weightCols);

            size_t inputDim = inputSampleLayout.GetWidth() * inputSampleLayout.GetHeight() * inputSampleLayout.GetNumChannels();
            if (Input(1)->GetNumRows() == 0)
                ValidateInferInputDims(1, inputDim, Input(1)->GetNumCols());

            if (isFinalValidationPass && Input(1)->GetNumRows() != inputDim)
                LogicError("each column of input to the convolution node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", NodeName().c_str(), (int)inputDim);

            size_t outputDim = m_sampleLayout.GetWidth() * m_sampleLayout.GetHeight() * m_sampleLayout.GetNumChannels();
            SetDims(outputDim, Input(1)->GetNumCols());
        }

        void InferImageDimsFromInputs() override
        {
            auto inputSampleLayout = GetInputSampleLayout(1);

            if (inputSampleLayout.GetWidth() < m_kernelWidth || inputSampleLayout.GetHeight() < m_kernelHeight)
                InvalidArgument("inputWidth must >= kernelWidth and inputHeight must >= kernelHeight.");

            if (m_zeroPadding)
            {
                const int kernelWidthCenter = m_kernelWidth % 2;
                const int kernelHeightCenter = m_kernelHeight % 2;
                m_sampleLayout = ImageLayoutWHC(
                    (inputSampleLayout.GetWidth()  - kernelWidthCenter)  / m_horizontalSubsample + 1,
                    (inputSampleLayout.GetHeight() - kernelHeightCenter) / m_verticalSubsample   + 1,
                    m_sampleLayout.GetNumChannels());
            }
            else
            {
                m_sampleLayout = ImageLayoutWHC(
                    (inputSampleLayout.GetWidth()  - m_kernelWidth)  / m_horizontalSubsample + 1,
                    (inputSampleLayout.GetHeight() - m_kernelHeight) / m_verticalSubsample   + 1,
                    m_sampleLayout.GetNumChannels());
            }    

            // REVIEW alexeyk: is there a better place to create engines?
            if (m_factory == nullptr)
                m_factory = ConvolutionEngineFactory<ElemType>::Create(m_deviceId);
            if (m_convEng == nullptr)
                m_convEng = m_factory->CreateConvEngine(m_deviceId, m_maxTempMemSizeInSamples);
            if (m_inT == nullptr)
                m_inT = m_factory->CreateTensor(inputSampleLayout.GetWidth(), inputSampleLayout.GetHeight(), inputSampleLayout.GetNumChannels(), 1);
            if (m_filterT == nullptr)
                m_filterT = m_factory->CreateFilter(m_kernelWidth, m_kernelHeight, inputSampleLayout.GetNumChannels(), m_sampleLayout.GetNumChannels());
            if (m_outT == nullptr)
                m_outT = m_factory->CreateTensor(m_sampleLayout.GetWidth(), m_sampleLayout.GetHeight(), m_sampleLayout.GetNumChannels(), 1);
            if (m_convDesc == nullptr)
                m_convDesc = m_factory->CreateConvDescriptor(*m_inT, *m_filterT, m_horizontalSubsample, m_verticalSubsample, m_zeroPadding);
            // REVIEW alexeyk: create per-channel (shared) bias. Consider adding other types of biases.
            if (m_biasT == nullptr)
                m_biasT = m_factory->CreateTensor(1, 1, m_sampleLayout.GetNumChannels(), 1);
        }

        void DumpNodeInfo(const bool printValues, File& fstream) const override
        {
            Base::DumpNodeInfo(printValues, fstream);

            auto inputSampleLayout = GetInputSampleLayout(1);

            char str[4096];
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", inputSampleLayout.GetWidth(), inputSampleLayout.GetHeight(), inputSampleLayout.GetNumChannels());
            fstream << string(str);
            sprintf(str, "Kernel[Width:%lu, Height:%lu]  SubSample[Horizontal:%lu, Vertical:%lu]\n", m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_sampleLayout.GetWidth(), m_sampleLayout.GetHeight(), m_sampleLayout.GetNumChannels());
            fstream << string(str);
            sprintf(str, "ZeroPadding=%ls  maxTempMemSizeInSamples=%lu\n", m_zeroPadding? L"true" : L"false", m_maxTempMemSizeInSamples);
            fstream << string(str);
        }

        void SetmMaxTempMemSizeInSamples(const size_t maxTempMemSizeInSamples)
        {
            m_maxTempMemSizeInSamples = maxTempMemSizeInSamples;
        }

        //request matrices needed to do node function value evaluation
        void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
        {
            Base::RequestMatricesBeforeForwardProp(matrixPool);
            RequestMatrixFromPool(m_tempMatrix, matrixPool);
        }

        //release gradient and temp matrices that no longer needed after all the children's gradients are computed.
        void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
        {
            Base::ReleaseMatricesAfterBackprop(matrixPool);
            ReleaseMatrixToPool(m_tempMatrix, matrixPool);
        }

    private:
        std::unique_ptr<ConvolutionEngineFactory<ElemType>> m_factory;
        std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;

        std::unique_ptr<ConvolutionTensor4D> m_inT;
        std::unique_ptr<ConvolutionFilter> m_filterT;
        std::unique_ptr<ConvolutionTensor4D> m_outT;
        std::unique_ptr<ConvolutionDescriptor> m_convDesc;
        std::unique_ptr<ConvolutionTensor4D> m_biasT;

        size_t m_kernelWidth, m_kernelHeight;
        size_t m_horizontalSubsample, m_verticalSubsample;
        bool m_zeroPadding;
        bool m_1DConvolutionOnGPUSparse;

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
        {
            m_factory = ConvolutionEngineFactory<ElemType>::Create(deviceId);
        }
        PoolingNodeBase(const ScriptableObjects::IConfigRecordPtr configp) :
            PoolingNodeBase(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"windowWidth"), configp->Get(L"windowHeight"), configp->Get(L"horizontalSubsample"), configp->Get(L"verticalSubsample"))
        {
            // input, windowWidth, windowHeight, horizontalSubsample, verticalSubsample
            AttachInputs(configp, this->GetExpectedNumInputs());
        }

        void Save(File& fstream) const override
        {
            Base::Save(fstream);
            fstream << m_windowWidth << m_windowHeight << m_horizontalSubsample << m_verticalSubsample;
        }

        void Load(File& fstream, size_t modelVersion) override
        {
            Base::Load(fstream, modelVersion);
            fstream >> m_windowWidth >> m_windowHeight >> m_horizontalSubsample >> m_verticalSubsample;
        }

        void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
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

        void BackpropTo(const size_t /*inputIndex*/, const FrameRange & fr) override
        {
            Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);
            Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

            Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
            Matrix<ElemType> sliceOutputValue = ValueFor(fr);

            size_t batchSize = sliceInput0Value.GetNumCols();
            m_inT->setN(batchSize);
            m_outT->setN(batchSize);
            assert(m_poolEng != nullptr);
            assert(m_poolDesc != nullptr);
            m_poolEng->Backward(*m_outT, sliceOutputValue, sliceOutputGrad, *m_poolDesc, *m_inT, sliceInput0Value, sliceInput0Grad);
        }

        void ForwardProp(const FrameRange & fr) override
        {
            Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
            Matrix<ElemType> sliceOutputValue = ValueFor(fr);

            size_t batchSize = sliceInput0Value.GetNumCols();
            m_inT->setN(batchSize);
            m_outT->setN(batchSize);
            assert(m_poolEng != nullptr);
            assert(m_poolDesc != nullptr);
            m_poolEng->Forward(*m_inT, sliceInput0Value, *m_poolDesc, *m_outT, sliceOutputValue);
        }

        void Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);
            InferMBLayoutFromInputsForStandardCase();

            InferImageDimsFromInputs();

            auto inputSampleLayout = GetInputSampleLayout(0);

            m_inputSizePerSample = inputSampleLayout.GetWidth() * inputSampleLayout.GetHeight() * inputSampleLayout.GetNumChannels();
            m_outputSizePerSample = m_sampleLayout.GetWidth() * m_sampleLayout.GetHeight() * m_sampleLayout.GetNumChannels();

            if (Input(0)->GetNumRows() == 0)
                ValidateInferInputDims(0, m_inputSizePerSample, Input(0)->GetNumCols());

            if (isFinalValidationPass && Input(0)->GetNumRows() != m_inputSizePerSample)
                LogicError("each column of input to the MaxPooling node %ls is a sample and should have dimension %d, which is inputWidth * inputHeight * inputChannels", NodeName().c_str(), (int)m_inputSizePerSample);

            SetDims(m_outputSizePerSample, Input(0)->GetNumCols());
        }

        void InferImageDimsFromInputs() override
        {
            auto inputSampleLayout = GetInputSampleLayout(0);

            if (inputSampleLayout.GetWidth() < m_windowWidth || inputSampleLayout.GetHeight() < m_windowHeight)
                InvalidArgument("PoolingNodeBase: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

            m_sampleLayout = ImageLayoutWHC(
                (inputSampleLayout.GetWidth()  - m_windowWidth)  / m_horizontalSubsample + 1,
                (inputSampleLayout.GetHeight() - m_windowHeight) / m_verticalSubsample + 1,
                inputSampleLayout.GetNumChannels());

            // REVIEW alexeyk: is there a better place to create engines?
            if (m_factory == nullptr)
                m_factory = ConvolutionEngineFactory<ElemType>::Create(m_deviceId);
            if (m_poolEng == nullptr)
                m_poolEng = m_factory->CreatePoolEngine(m_deviceId);
            if (m_inT == nullptr)
                m_inT = m_factory->CreateTensor(inputSampleLayout.GetWidth(), inputSampleLayout.GetHeight(), inputSampleLayout.GetNumChannels(), 1);
            if (m_outT == nullptr)
                m_outT = m_factory->CreateTensor(m_sampleLayout.GetWidth(), m_sampleLayout.GetHeight(), m_sampleLayout.GetNumChannels(), 1);
        }

        void DumpNodeInfo(const bool printValues, File& fstream) const override
        {
            Base::DumpNodeInfo(printValues, fstream);

            auto inputSampleLayout = GetInputSampleLayout(0);

            char str[4096];
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", inputSampleLayout.GetWidth(), inputSampleLayout.GetHeight(), inputSampleLayout.GetNumChannels());
            fstream << string(str);
            sprintf(str, "PoolingWindow[Width:%lu, Height:%lu]  SubSampling[Horizontal:%lu, Vertical:%lu]\n", m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_sampleLayout.GetWidth(), m_sampleLayout.GetHeight(), m_sampleLayout.GetNumChannels());
            fstream << string(str);
            sprintf(str, "TotalSizePerSample[Input:%lu, Output:%lu]  \n", m_inputSizePerSample, m_outputSizePerSample);
            fstream << string(str);
        }

    protected:
        std::unique_ptr<ConvolutionEngineFactory<ElemType>> m_factory;
        std::unique_ptr<PoolingEngine<ElemType>> m_poolEng;

        std::unique_ptr<ConvolutionTensor4D> m_inT;
        std::unique_ptr<ConvolutionTensor4D> m_outT;
        std::unique_ptr<PoolingDescriptor> m_poolDesc;

        size_t m_windowWidth, m_windowHeight;
        size_t m_horizontalSubsample, m_verticalSubsample;
        size_t m_inputSizePerSample, m_outputSizePerSample;
    };

    // add this at the start of each derived class, to get access to the members of ComputationNode
    // See #define of 'UsingComputationNodeMembersBoilerplate' for more explanation.
#define UsingPoolingNodeBaseMembers UsingComputationNodeMembersBoilerplate; \
    protected:  \
        using Base::m_factory; using Base::m_poolDesc; using Base::m_windowWidth; using Base::m_windowHeight; using Base::m_horizontalSubsample; using Base::m_verticalSubsample; using Base::m_inputSizePerSample; using Base::m_outputSizePerSample; \
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
        MaxPoolingNode(const ScriptableObjects::IConfigRecordPtr configp) :
            Base(configp)
        { }

        void InferImageDimsFromInputs() override
        {
            Base::InferImageDimsFromInputs();
            if (m_poolDesc == nullptr)
                m_poolDesc = m_factory->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Max, m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample, 0, 0);
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
        AveragePoolingNode(const ScriptableObjects::IConfigRecordPtr configp) :
            Base(configp)
        { }

        virtual bool OutputUsedInComputingInputNodesGradients() const override
        {
            // The AveragePoolingNode does not require its output value for computing
            // the gradients of its input nodes
            return false;
        }

        virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override
        {
            // The AveragePoolingNode does not require any of it's input's values for computing
            // the gradients of its input nodes
            UNREFERENCED_PARAMETER(childIndex);
            return false;
        }

        void InferImageDimsFromInputs() override
        {
            Base::InferImageDimsFromInputs();
            if (m_poolDesc == nullptr)
                m_poolDesc = m_factory->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Average, m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample, 0, 0);
        }
    };

    template class AveragePoolingNode<float>; 
    template class AveragePoolingNode<double>;    

    // Implements batch normalization technique as described in:
    // Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [S. Ioffe, C. Szegedy]
    // http://arxiv.org/abs/1502.03167
    // REVIEW alexeyk: is this a right header for this node? It's not really limited to only convolutional nodes.
    template<class ElemType>
    class BatchNormalizationNode : public ComputationNode<ElemType>, public NumInputs<5>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"BatchNormalization"; }
    public:
        BatchNormalizationNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name), m_eval(false), m_spatial(false), m_expAvgFactor(0)
        {
        }
        BatchNormalizationNode(DEVICEID_TYPE deviceId, const wstring & name, bool eval, bool spatial, double expAvgFactor) :
            Base(deviceId, name), m_eval(eval), m_spatial(spatial), m_expAvgFactor(expAvgFactor)
        {
        }
        BatchNormalizationNode(const ScriptableObjects::IConfigRecordPtr configp) :
            BatchNormalizationNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"eval"), configp->Get(L"spatial"), configp->Get(L"expAvgFactor"))
        {
            AttachInputs(configp, this->GetExpectedNumInputs());
        }

        void Save(File& fstream) const override
        {
            Base::Save(fstream);
            fstream << m_version.VerWrittenCur() << m_version.VerReadableCur();

            fstream << m_eval;
            fstream << m_spatial;
            fstream << m_expAvgFactor;
        }

        void Load(File& fstream, size_t modelVersion) override
        {
            Base::Load(fstream, modelVersion);

            // Read and check version.
            // REVIEW alexeyk: extract version checking so it can be re-used in other places.
            int32_t verWritten;
            int32_t verReadable;
            fstream >> verWritten >> verReadable;

            if (verReadable > verWritten)
                RuntimeError("Corrupt model file.");
            if (verWritten < m_version.VerWeCanReadBack())
                RuntimeError("Model is too old.");
            if (verReadable > m_version.VerWrittenCur())
                RuntimeError("Model is too new.");

            fstream >> m_eval;
            fstream >> m_spatial;
            fstream >> m_expAvgFactor;
        }

        void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<BatchNormalizationNode<ElemType>>(nodeP);
                assert(node != nullptr);

                node->m_eval = m_eval;
                node->m_spatial = m_spatial;
                node->m_expAvgFactor = m_expAvgFactor;
            }
        }

        void BackpropTo(const size_t inputIndex, const FrameRange & fr) override
        {
            if (m_eval)
                LogicError("BatchNormalization does not compute derivatives in inference mode.");

            if (inputIndex == 0) // derivative with respect to the input.
            {
                Matrix<ElemType> sliceOutputGrad = GradientFor(fr);
                Matrix<ElemType> sliceInputValue = Input(0)->ValueFor(fr);
                const Matrix<ElemType>& scale = Input(1)->Value();
                const Matrix<ElemType>& bias = Input(2)->Value();

                size_t batchSize = sliceInputValue.GetNumCols();
                m_inT->setN(batchSize);
                assert(m_convEng != nullptr);

                Matrix<ElemType> sliceInputGrad = Input(0)->GradientFor(fr);
                m_dScale->Resize(scale.GetNumRows(), scale.GetNumCols());
                m_dBias->Resize(bias.GetNumRows(), bias.GetNumCols());
                // Compute all derivatives in one step. Save derivatives with respect to scale and bias in temp matrices.
                m_convEng->BackwardNormalizeBatch(*m_inT, sliceInputValue, sliceOutputGrad, sliceInputGrad, *m_scaleBiasT, scale, m_spatial,
                    *m_saveMean, *m_saveInvStdDev, *m_dScale, *m_dBias);
            }
            else if (inputIndex == 1) // derivative with respect to the scale
            {
                // Derivative with respect to the scale was precomputed during input derivative computation.
                Matrix<ElemType>& grad = Input(1)->Gradient();
                grad.SetValue(grad.GetNumRows(), grad.GetNumCols(), grad.GetDeviceId(), m_dScale->BufferPointer());
            }
            else if (inputIndex == 2) // derivative with respect to the bias
            {
                // Derivative with respect to the bias was precomputed during input derivative computation.
                Matrix<ElemType>& grad = Input(2)->Gradient();
                grad.SetValue(grad.GetNumRows(), grad.GetNumCols(), grad.GetDeviceId(), m_dBias->BufferPointer());
            }
            // No derivatives with respect to running mean and InvStdDev.
        }

        void ForwardProp(const FrameRange & fr) override
        {
            Matrix<ElemType> sliceInputValue = Input(0)->ValueFor(fr);

            const Matrix<ElemType>& scale = Input(1)->Value();
            const Matrix<ElemType>& bias = Input(2)->Value();
            Matrix<ElemType>& runMean = Input(3)->Value();
            Matrix<ElemType>& runInvStdDev = Input(4)->Value();
            assert(scale.GetNumRows() == bias.GetNumRows());
            assert(scale.GetNumCols() == bias.GetNumCols());
            assert(runMean.GetNumRows() == scale.GetNumRows());
            assert(runMean.GetNumCols() == scale.GetNumCols());
            assert(runMean.GetNumRows() == runInvStdDev.GetNumRows());
            assert(runMean.GetNumCols() == runInvStdDev.GetNumCols());

            Matrix<ElemType> sliceOutputValue = ValueFor(fr);

            size_t batchSize = sliceInputValue.GetNumCols();
            m_inT->setN(batchSize);
            assert(m_convEng != nullptr);
#if NANCHECK
            sliceInputValue.HasNan("BatchNormalization-input");
#endif
            if (m_eval)
                m_convEng->NormalizeBatchInference(*m_inT, sliceInputValue, *m_scaleBiasT, scale, bias, m_spatial, runMean, runInvStdDev, sliceOutputValue);
            else
            {
                m_convEng->NormalizeBatch(*m_inT, sliceInputValue, *m_scaleBiasT, scale, bias, m_spatial, m_expAvgFactor, runMean, runInvStdDev,
                    sliceOutputValue, *m_saveMean, *m_saveInvStdDev);
            }
#if NANCHECK
            sliceOutputValue.HasNan("BatchNormalization");
#endif
        }

        void Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);
            InferMBLayoutFromInputsForStandardCase();

            InferImageDimsFromInputs();
            SetDims(m_sampleLayout.GetWidth() * m_sampleLayout.GetHeight() * m_sampleLayout.GetNumChannels(), Input(0)->GetNumCols());
        }

        void InferImageDimsFromInputs() override
        {
            m_sampleLayout = GetInputSampleLayout(0);

            if (m_factory == nullptr)
                m_factory = ConvolutionEngineFactory<ElemType>::Create(m_deviceId);
            if (m_convEng == nullptr)
                m_convEng = m_factory->CreateConvEngine(m_deviceId, 0);
            if (m_inT == nullptr)
                m_inT = m_factory->CreateTensor(m_sampleLayout.GetWidth(), m_sampleLayout.GetHeight(), m_sampleLayout.GetNumChannels(), 1);
            if (m_scaleBiasT == nullptr)
            {
                if (m_spatial)
                    m_scaleBiasT = m_factory->CreateTensor(1, 1, m_sampleLayout.GetNumChannels(), 1);
                else
                    m_scaleBiasT = m_factory->CreateTensor(m_sampleLayout.GetWidth(), m_sampleLayout.GetHeight(), m_sampleLayout.GetNumChannels(), 1);
            }
        }

        void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
        {
            Base::RequestMatricesBeforeForwardProp(matrixPool);
            if (!m_eval)
            {
                RequestMatrixFromPool(m_saveMean, matrixPool);
                RequestMatrixFromPool(m_saveInvStdDev, matrixPool);
            }
        }

        void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
        {
            Base::RequestMatricesBeforeBackprop(matrixPool);
            if (!m_eval)
            {
                RequestMatrixFromPool(m_dScale, matrixPool);
                RequestMatrixFromPool(m_dBias, matrixPool);
            }
        }

        void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
        {
            Base::ReleaseMatricesAfterBackprop(matrixPool);
            if (!m_eval)
            {
                ReleaseMatrixToPool(m_saveMean, matrixPool);
                ReleaseMatrixToPool(m_saveInvStdDev, matrixPool);
                ReleaseMatrixToPool(m_dScale, matrixPool);
                ReleaseMatrixToPool(m_dBias, matrixPool);
            }
        }

    private:
        struct VersionInfo
        {
            int32_t VerWrittenCur() const     { return 0x00010001; } // Initial
            int32_t VerReadableCur() const    { return 0x00010001; }
            int32_t VerWeCanReadBack() const  { return 0x00010001; }
        };
        VersionInfo m_version;

    private:
        std::unique_ptr<ConvolutionEngineFactory<ElemType>> m_factory;
        std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;
        std::unique_ptr<ConvolutionTensor4D> m_inT;
        std::unique_ptr<ConvolutionTensor4D> m_scaleBiasT;

        // Determines whether to use training or inference(evaluation) mode.
        bool m_eval;
        // Determines whether to use per-activation (used after non-convolutional layers like fully connected)
        // or spatial (used after convolutional layers).
        bool m_spatial;
        // Smoothing factor.
        double m_expAvgFactor;
        // Stores pre-computed on forward pass mean values that are used in gradient computation.
        shared_ptr<Matrix<ElemType>> m_saveMean;
        // Stores pre-computed on forward pass InvStdDev values that are used in gradient computation.
        shared_ptr<Matrix<ElemType>> m_saveInvStdDev;
        // Stores scale derivatives
        shared_ptr<Matrix<ElemType>> m_dScale;
        // Stores bias derivatives.
        shared_ptr<Matrix<ElemType>> m_dBias;
    };

    template class BatchNormalizationNode<float>; 
    template class BatchNormalizationNode<double>;    

}}}
