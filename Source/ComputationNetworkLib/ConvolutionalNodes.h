//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
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

// Convolutions (incl. pooling) support two different storage formats:
// BUGBUG: These are currently hard-selected depending on circumstances, without being reflected in TensoShape.
//
// * legacy mode (CPU and GPU without cudnn): Channels are tuples of scalars
//
//    This follows "high performance convolutional neural networks for document processing" by Kumar Chellapilla, Sidde Puri, and Patrice Simard.
//    Each sample is stored as a column-major matrix (height, width) of float[numChannels] (r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11).
//
//     - input :  [C  x W  x H      x T]  or  ARRAY[1..T] OF                ARRAY[1..H]  OF ARRAY[1..W]  OF ARRAY[1..C]
//     - output : [C' x W' x H'     x T]  or  ARRAY[1..T] OF                ARRAY[1..H'] OF ARRAY[1..W'] OF ARRAY[1..C']
//     - filter : [C' x W" x H" x C    ]  or                 ARRAY[1..C] OF ARRAY[1..H"] OF ARRAY[1..W"] OF ARRAY[1..C']
//
// * GPU with cudnn: Channels are planes
//
//     - input :   [W  x H  x C       x T]   or  ARRAY[1..T] OF                 ARRAY[1..C]  OF ARRAY[1..H]  OF ARRAY[1..W]
//     - output :  [W' x H' x      C' x T]   or  ARRAY[1..T] OF ARRAY[1..C'] OF                 ARRAY[1..H'] OF ARRAY[1..W']
//     - filter :  [W" x H" x C  x C'    ]   or                 ARRAY[1..C'] OF ARRAY[1..C]  OF ARRAY[1..H]  OF ARRAY[1..W]
//
// where:
//  - using ' for output and " for filter
//  - T = samples (NVidia calls this N)
//  - W, H = width, height (W', H' for output, W", H" for kernel)
//  - C = input channels
//     - 3 for color images, 1 for B&W images
//     - for hidden layer: dimension of activation vector for each pixel
//  - C' = output channels = dimension of activation vector for each pixel (also called N by NVidia, inconsistently)
template <class ElemType>
class ConvolutionNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Convolution";
    }

public:
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          m_kernelWidth(SIZE_MAX),
          m_kernelHeight(SIZE_MAX),
          // initialize to dummy values so we catch missing initialization
          m_horizontalSubsample(SIZE_MAX),
          m_verticalSubsample(SIZE_MAX),
          m_zeroPadding(false),
          m_maxTempMemSizeInSamples(SIZE_MAX),
          m_imageLayoutKind(ImageLayoutKind::HWC)
    {
        SetDims(ImageDimensions::AsTensorShape(1, 1, 0, m_imageLayoutKind), 0);
    }
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind,
                    const bool zeroPadding = false, const size_t maxTempMemSizeInSamples = 0)
        : Base(deviceId, name),
          m_outputChannels(outputChannels),
          m_kernelWidth(kernelWidth),
          m_kernelHeight(kernelHeight),
          m_horizontalSubsample(horizontalSubsample),
          m_verticalSubsample(verticalSubsample),
          m_zeroPadding(zeroPadding),
          m_maxTempMemSizeInSamples(maxTempMemSizeInSamples),
          m_imageLayoutKind(imageLayoutKind)
    {
        SetDims(ImageDimensions::AsTensorShape(1, 1, m_outputChannels, m_imageLayoutKind), 0); // TODO: necessary?
        m_factory = ConvolutionEngineFactory<ElemType>::Create(deviceId, ConvolutionEngineFactory<ElemType>::EngineType::Auto, m_imageLayoutKind);
    }
    ConvolutionNode(const ScriptableObjects::IConfigRecordPtr configp)
        : ConvolutionNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"kernelWidth"), configp->Get(L"kernelHeight"), configp->Get(L"outputChannels"),
                          configp->Get(L"horizontalSubsample"), configp->Get(L"verticalSubsample"), ImageLayoutKindFrom(configp->Get(L"imageLayout")),
                          configp->Get(L"zeroPadding"), configp->Get(L"maxTempMemSizeInSamples"))
    {
        // weightNodeName, inputValueNodeName, kernelWidth, kernelHeight, outputChannels, horizontalSubsample, verticalSubsample, zeroPadding = false, maxTempMemSizeInSamples = 0
        AttachInputs(configp, this->GetExpectedNumInputs());
    }

    void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_kernelWidth << m_kernelHeight << m_horizontalSubsample << m_verticalSubsample;
        uint32_t imageLayoutKind = (uint32_t) m_imageLayoutKind;
        uint32_t outputChannels = (uint32_t) m_outputChannels;
        fstream << outputChannels << imageLayoutKind;
        fstream << m_zeroPadding << m_maxTempMemSizeInSamples;
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_kernelWidth >> m_kernelHeight >> m_horizontalSubsample >> m_verticalSubsample;
        uint32_t imageLayoutKind, outputChannels;
        fstream >> outputChannels >> imageLayoutKind;
        m_imageLayoutKind = (ImageLayoutKind) imageLayoutKind;
        m_outputChannels = outputChannels;
        SetDims(ImageDimensions::AsTensorShape(1, 1, m_outputChannels, m_imageLayoutKind), 0); // TODO: needed?
        fstream >> m_zeroPadding >> m_maxTempMemSizeInSamples;
        m_factory = ConvolutionEngineFactory<ElemType>::Create(GetDeviceId(), ConvolutionEngineFactory<ElemType>::EngineType::Auto, m_imageLayoutKind);
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

            node->m_imageLayoutKind = m_imageLayoutKind;

            *node->m_tempMatrix = *m_tempMatrix;
        }
    }

    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        auto sliceOutputGrad = GradientFor(fr);
        auto sliceInput1Value = Input(1)->ValueFor(fr);

        size_t batchSize = sliceInput1Value.GetNumCols();
        m_inT->setN(batchSize);
        m_outT->setN(batchSize);
        assert(m_convEng != nullptr);
        if (inputIndex == 0) // derivative with respect to the weight matrix
        {
            auto& grad = Input(0)->GradientAsMatrix();
            m_convEng->BackwardFilter(*m_outT, sliceOutputGrad, *m_inT, sliceInput1Value, *m_convDesc, *m_filterT, grad, fr.IsAllFrames(), *m_tempMatrix);
        }
        else if (inputIndex == 1) // derivative with respect to the input feature
        {
            auto& input0 = Input(0)->ValueAsMatrix();
            auto sliceInput1Grad = Input(1)->GradientFor(fr);
            m_convEng->BackwardData(*m_outT, sliceOutputGrad, *m_filterT, input0, *m_convDesc, *m_inT, sliceInput1Grad, *m_tempMatrix);
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The ConvolutionNode does not require its output value for computing
        // the gradients of its input nodes
        return false;
    }

    void ForwardProp(const FrameRange& fr) override
    {
        const Matrix<ElemType>& input0 = Input(0)->ValueAsMatrix();
        Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        // update the tensor dimension w.r.t. number of samples
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

    // BUGBUG: Should not be here. Use PlusNode and m_sampleLayout.  TODO: Bad naming:'output' is actually an 'input'
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
    void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        // get input and output tensor shape and interpret as image dimensions
        auto inDims = ImageDimensions(GetInputSampleLayout(1), m_imageLayoutKind);

        if (isFinalValidationPass && (inDims.m_width < m_kernelWidth || inDims.m_height < m_kernelHeight))
            InvalidArgument("%ls %ls operation requires that input width be >= kernelWidth and input height >= kernelHeight.", NodeName().c_str(), OperationName().c_str());

        // determine output tensor shape
        const int kernelWidthCenter = m_zeroPadding ? m_kernelWidth % 2 : m_kernelWidth;
        const int kernelHeightCenter = m_zeroPadding ? m_kernelHeight % 2 : m_kernelHeight;
        auto outDims = ImageDimensions(
            (inDims.m_width - kernelWidthCenter) / m_horizontalSubsample + 1,
            (inDims.m_height - kernelHeightCenter) / m_verticalSubsample + 1,
            m_outputChannels);

        size_t weightCols = m_kernelWidth * m_kernelHeight * inDims.m_numChannels;

        // check/infer input [0] (weights)
        // BUGBUG: For now, we treat the weights as a 2D matrix. They should be a tensor proper.
        Input(0)->ValidateInferInputDimsFrom(TensorShape(m_outputChannels, weightCols));

        if (isFinalValidationPass && (Input(0)->GetAsMatrixNumCols() != weightCols || Input(0)->GetAsMatrixNumRows() != m_outputChannels))
            LogicError("convolutionWeight matrix %ls should have dimension [%d, %d] which is [outputChannels, kernelWidth * kernelHeight * inputChannels]", Input(0)->NodeName().c_str(), (int) m_outputChannels, (int) weightCols);

        // that's our dimension
        SetDims(outDims.AsTensorShape(m_imageLayoutKind), true);

        if (isFinalValidationPass)
        {
            // set up the various engines and descriptor objects
            // REVIEW alexeyk: is there a better place to create engines?
            assert(m_factory);
            // if (m_factory == nullptr)
            //    m_factory = ConvolutionEngineFactory<ElemType>::Create(m_deviceId, ConvolutionEngineFactory<ElemType>::EngineType::Auto, m_imageLayoutKind);
            // TODO: This seems to expose too much internal knowlegde of the engine to the ConvolutionNode().
            //       Why not just pass everything to the engine creator, and get one object that holds everything.
            if (m_convEng == nullptr)
                m_convEng = m_factory->CreateConvEngine(m_deviceId, m_maxTempMemSizeInSamples);
            if (m_inT == nullptr)
                m_inT = m_factory->CreateTensor(inDims.m_width, inDims.m_height, inDims.m_numChannels, 1);
            if (m_filterT == nullptr)
                m_filterT = m_factory->CreateFilter(m_kernelWidth, m_kernelHeight, inDims.m_numChannels, m_outputChannels);
            if (m_outT == nullptr)
                m_outT = m_factory->CreateTensor(outDims.m_width, outDims.m_height, outDims.m_numChannels, 1);
            if (m_convDesc == nullptr)
                m_convDesc = m_factory->CreateConvDescriptor(*m_inT, *m_filterT, m_horizontalSubsample, m_verticalSubsample, m_zeroPadding);
            // REVIEW alexeyk: create per-channel bias (shared across all pixels). Consider adding other types of biases.
            if (m_biasT == nullptr)
                m_biasT = m_factory->CreateTensor(1, 1, outDims.m_numChannels, 1);
        }
    }

    void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override
    {
        Base::DumpNodeInfo(printValues, printMetadata, fstream);

        auto inDims = ImageDimensions(GetInputSampleLayout(1), m_imageLayoutKind);
        auto outDims = ImageDimensions(m_sampleLayout, m_imageLayoutKind);

        char str[4096];
        sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", inDims.m_width, inDims.m_height, inDims.m_numChannels);
        fstream << string(str);
        sprintf(str, "Kernel[Width:%lu, Height:%lu]  SubSample[Horizontal:%lu, Vertical:%lu]\n", m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample);
        fstream << string(str);
        sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", outDims.m_width, outDims.m_height, outDims.m_numChannels);
        fstream << string(str);
        sprintf(str, "zeroPadding=%ls  maxTempMemSizeInSamples=%lu\n", m_zeroPadding ? L"true" : L"false", m_maxTempMemSizeInSamples);
        fstream << string(str);
    }

    void SetmMaxTempMemSizeInSamples(const size_t maxTempMemSizeInSamples)
    {
        m_maxTempMemSizeInSamples = maxTempMemSizeInSamples;
    }

    // request matrices needed to do node function value evaluation
    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_tempMatrix, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_tempMatrix, matrixPool);
    }

private:
    size_t m_outputChannels;
    size_t m_kernelWidth, m_kernelHeight;
    size_t m_horizontalSubsample, m_verticalSubsample;
    bool m_zeroPadding;
    bool m_1DConvolutionOnGPUSparse;

    shared_ptr<Matrix<ElemType>> m_tempMatrix;
    size_t m_maxTempMemSizeInSamples; // can change during runtime

    ImageLayoutKind m_imageLayoutKind; // how to interpret the tensor (which dimensions are X/Y and C)

    std::unique_ptr<ConvolutionEngineFactory<ElemType>> m_factory;
    std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;

    std::unique_ptr<ConvolutionTensor4D> m_inT;
    std::unique_ptr<ConvolutionFilter> m_filterT;
    std::unique_ptr<ConvolutionTensor4D> m_outT;
    std::unique_ptr<ConvolutionDescriptor> m_convDesc;
    std::unique_ptr<ConvolutionTensor4D> m_biasT;
};

template class ConvolutionNode<float>;
template class ConvolutionNode<double>;

// -----------------------------------------------------------------------
// PoolingNodeBase (input)
// -----------------------------------------------------------------------

template <class ElemType>
class PoolingNodeBase : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembers;

public:
    PoolingNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          m_windowWidth(SIZE_MAX),
          m_windowHeight(SIZE_MAX),
          m_horizontalSubsample(SIZE_MAX),
          m_verticalSubsample(SIZE_MAX),
          m_imageLayoutKind(ImageLayoutKind::HWC)
    {
    }
    PoolingNodeBase(DEVICEID_TYPE deviceId, const wstring& name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind)
        : Base(deviceId, name),
          m_windowWidth(windowWidth),
          m_windowHeight(windowHeight),
          m_horizontalSubsample(horizontalSubsample),
          m_verticalSubsample(verticalSubsample),
          m_imageLayoutKind(imageLayoutKind)
    {
        m_factory = ConvolutionEngineFactory<ElemType>::Create(deviceId, ConvolutionEngineFactory<ElemType>::EngineType::Auto, m_imageLayoutKind);
    }
    PoolingNodeBase(const ScriptableObjects::IConfigRecordPtr configp)
        : PoolingNodeBase(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"windowWidth"), configp->Get(L"windowHeight"), configp->Get(L"horizontalSubsample"), configp->Get(L"verticalSubsample"), ImageLayoutKindFrom(configp->Get(L"imageLayout")))
    {
        // input, windowWidth, windowHeight, horizontalSubsample, verticalSubsample
        AttachInputs(configp, this->GetExpectedNumInputs());
    }

    void Save(File& fstream) const override
    {
        Base::Save(fstream);
        uint32_t imageLayoutKind = (uint32_t) m_imageLayoutKind;
        uint32_t windowWidth = (uint32_t) m_windowWidth;
        fstream << windowWidth << imageLayoutKind << m_windowHeight << m_horizontalSubsample << m_verticalSubsample;
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        uint32_t imageLayoutKind, windowWidth;
        fstream >> windowWidth >> imageLayoutKind >> m_windowHeight >> m_horizontalSubsample >> m_verticalSubsample;
        m_windowWidth = windowWidth;
        m_imageLayoutKind = (ImageLayoutKind) imageLayoutKind;
        m_factory = ConvolutionEngineFactory<ElemType>::Create(GetDeviceId(), ConvolutionEngineFactory<ElemType>::EngineType::Auto, m_imageLayoutKind);
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

            node->m_imageLayoutKind = m_imageLayoutKind;
        }
    }

    void BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
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

    void ForwardProp(const FrameRange& fr) override
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

        // get input tensor shape and interpret as image dimensions
        auto inDims = ImageDimensions(GetInputSampleLayout(0), m_imageLayoutKind);

        if (isFinalValidationPass && (inDims.m_width < m_windowWidth || inDims.m_height < m_windowHeight))
            InvalidArgument("PoolingNodeBase: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

        // determine output tensor shape
        auto outDims = ImageDimensions(
            (inDims.m_width - m_windowWidth) / m_horizontalSubsample + 1,
            (inDims.m_height - m_windowHeight) / m_verticalSubsample + 1,
            inDims.m_numChannels);

        m_inputSizePerSample = inDims.m_width * inDims.m_height * inDims.m_numChannels;

        SetDims(outDims.AsTensorShape(m_imageLayoutKind), true);

        if (isFinalValidationPass)
        {
            // set up various engines and descriptor objects
            // REVIEW alexeyk: is there a better place to create engines?
            assert(m_factory);
            // if (m_factory == nullptr)
            //    m_factory = ConvolutionEngineFactory<ElemType>::Create(m_deviceId, ConvolutionEngineFactory<ElemType>::EngineType::Auto, m_imageLayoutKind);
            if (m_poolEng == nullptr)
                m_poolEng = m_factory->CreatePoolEngine(m_deviceId);
            if (m_inT == nullptr)
                m_inT = m_factory->CreateTensor(inDims.m_width, inDims.m_height, inDims.m_numChannels, 1);
            if (m_outT == nullptr)
                m_outT = m_factory->CreateTensor(outDims.m_width, outDims.m_height, outDims.m_numChannels, 1);
        }
    }

    void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override
    {
        Base::DumpNodeInfo(printValues, printMetadata, fstream);

        if (printMetadata)
        {
            auto inputSampleLayout = GetInputSampleLayout(0);

            char str[4096];
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", inputSampleLayout[1], inputSampleLayout[2], inputSampleLayout[0]);
            fstream << string(str);
            sprintf(str, "PoolingWindow[Width:%lu, Height:%lu]  SubSampling[Horizontal:%lu, Vertical:%lu]\n", m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_sampleLayout[1], m_sampleLayout[2], m_sampleLayout[0]);
            fstream << string(str);
            sprintf(str, "TotalSizePerSample[Input:%lu, Output:%lu]  \n", m_inputSizePerSample, m_outputSizePerSample);
            fstream << string(str);
        }
    }

protected:
    size_t m_windowWidth, m_windowHeight;
    size_t m_horizontalSubsample, m_verticalSubsample;
    size_t m_inputSizePerSample, m_outputSizePerSample;

    ImageLayoutKind m_imageLayoutKind; // how to interpret the tensor (which dimensions are X/Y and C)

    std::unique_ptr<ConvolutionEngineFactory<ElemType>> m_factory;
    std::unique_ptr<PoolingEngine<ElemType>> m_poolEng;

    std::unique_ptr<ConvolutionTensor4D> m_inT;
    std::unique_ptr<ConvolutionTensor4D> m_outT;
    std::unique_ptr<PoolingDescriptor> m_poolDesc;
};

// add this at the start of each derived class, to get access to the members of ComputationNode
// See #define of 'UsingComputationNodeMembersBoilerplate' for more explanation.
#define UsingPoolingNodeBaseMembers         \
    UsingComputationNodeMembersBoilerplate; \
    \
protected:                                  \
    using Base::m_factory;                  \
    using Base::m_poolDesc;                 \
    using Base::m_windowWidth;              \
    using Base::m_windowHeight;             \
    using Base::m_horizontalSubsample;      \
    using Base::m_verticalSubsample;        \
    using Base::m_inputSizePerSample;       \
    using Base::m_outputSizePerSample;      \
    \
public:

// -----------------------------------------------------------------------
// MaxPoolingNode
// -----------------------------------------------------------------------

template <class ElemType>
class MaxPoolingNode : public PoolingNodeBase<ElemType>
{
    typedef PoolingNodeBase<ElemType> Base;
    UsingPoolingNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"MaxPooling";
    }

public:
    MaxPoolingNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    MaxPoolingNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind)
        : Base(deviceId, name, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayoutKind)
    {
    }
    MaxPoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp)
    {
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (isFinalValidationPass && m_poolDesc == nullptr)
            m_poolDesc = m_factory->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Max, m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample, 0, 0);
    }
};

template class MaxPoolingNode<float>;
template class MaxPoolingNode<double>;

// -----------------------------------------------------------------------
// AveragePoolingNode
// -----------------------------------------------------------------------

template <class ElemType>
class AveragePoolingNode : public PoolingNodeBase<ElemType>
{
    typedef PoolingNodeBase<ElemType> Base;
    UsingPoolingNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"AveragePooling";
    }

public:
    AveragePoolingNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    AveragePoolingNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind)
        : Base(deviceId, name, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayoutKind)
    {
    }
    AveragePoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp)
    {
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (isFinalValidationPass && m_poolDesc == nullptr)
            m_poolDesc = m_factory->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Average, m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample, 0, 0);
    }
};

template class AveragePoolingNode<float>;
template class AveragePoolingNode<double>;

} } }
