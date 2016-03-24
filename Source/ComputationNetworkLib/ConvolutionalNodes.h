//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "ComputationNode.h"
#include "ConvolutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class ConvolutionNode : public ComputationNode<ElemType>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Convolution";
    }

public:
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& strideShape,
                    const std::vector<bool>& sharing, const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                    ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, PoolKind poolKind)
                    : Base(deviceId, name), m_legacy(false), m_kernelShape(kernelShape), m_mapCount(mapCount), m_stride(strideShape), m_sharing(sharing),
                    m_autoPad(autoPadding), m_lowerPad(lowerPad), m_upperPad(upperPad),
                    m_imageLayout(imageLayout), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples), m_poolKind(poolKind)
    {
    }
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels,
                    const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind,
                    const bool zeroPadding, const size_t maxTempMemSizeInSamples)
                    : Base(deviceId, name), m_legacy(true), m_kernelShape(kernelWidth, kernelHeight, 1), m_mapCount(1, 1, outputChannels),
                    m_stride(horizontalSubsample, verticalSubsample, 1), m_autoPad(zeroPadding), m_lowerPad(0), m_upperPad(0)
    {
    }
    ConvolutionNode(const ScriptableObjects::IConfigRecordPtr configp)
        : ConvolutionNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"kernelShape"), configp->Get(L"mapCount"), configp->Get(L"strideShape"),
        configp->Get(L"dimSharing"), configp->Get(L"dimPadding"), configp->Get(L"dimPadLower"), configp->Get(L"dimPadUpper"),
        ImageLayoutKindFrom(configp->Get(L"imageLayout")), configp->Get(L"maxTempMemSizeInSamples"), PoolKindFrom(configp->Get(L"pool")))
    {
        AttachInputs(configp, GetExpectedNumInputs());
    }

public:
    void Save(File& fstream) const override
    {
        Base::Save(fstream);

        m_kernelShape.Save(fstream);
        m_mapCount.Save(fstream);
        m_stride.Save(fstream);
        fstream << m_sharing;
        fstream << m_autoPad;
        m_lowerPad.Save(fstream);
        m_upperPad.Save(fstream);
        fstream << (int32_t)m_imageLayout;
        fstream << m_maxTempMemSizeInSamples;
        fstream << (int32_t)m_poolKind;
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);

        m_kernelShape.Load(fstream);
        m_mapCount.Load(fstream);
        m_stride.Load(fstream);
        fstream >> m_sharing;
        fstream >> m_autoPad;
        m_lowerPad.Load(fstream);
        m_upperPad.Load(fstream);
        int32_t layout;
        fstream >> layout;
        m_imageLayout = (ImageLayoutKind)layout;
        fstream >> m_maxTempMemSizeInSamples;
        int32_t k;
        fstream >> k;
        m_poolKind = (PoolKind)k;
    }

    void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<ConvolutionNode<ElemType>>(nodeP);
            node->m_kernelShape = m_kernelShape;
            node->m_mapCount = m_mapCount;
            node->m_stride = m_stride;
            node->m_sharing = m_sharing;
            node->m_autoPad = m_autoPad;
            node->m_lowerPad = m_lowerPad;
            node->m_upperPad = m_upperPad;
            node->m_imageLayout = m_imageLayout;
            node->m_maxTempMemSizeInSamples = m_maxTempMemSizeInSamples;
            node->m_poolKind = m_poolKind;
        }
    }

    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        auto sliceOutputGrad = GradientFor(fr);

        if (m_poolKind == PoolKind::None)
        {
            if (inputIndex == 0) // derivative with respect to the weight matrix
            {
                auto& grad = Input(0)->GradientAsMatrix();
                auto sliceInput1Value = Input(1)->ValueFor(fr);
                m_convEng->BackwardKernel(sliceOutputGrad, sliceInput1Value, grad, fr.IsAllFrames(), *m_tempMatrix);
            }
            else if (inputIndex == 1) // derivative with respect to the input feature
            {
                auto& input0 = Input(0)->ValueAsMatrix();
                auto sliceInput1Grad = Input(1)->GradientFor(fr);
                m_convEng->BackwardData(sliceOutputGrad, input0, sliceInput1Grad, *m_tempMatrix);
            }
        }
        else
        {
            Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);

            Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
            Matrix<ElemType> sliceOutputValue = ValueFor(fr);

            m_convEng->BackwardPooling(sliceOutputValue, sliceOutputGrad, sliceInput0Value, sliceInput0Grad);
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The ConvolutionNode requires output values only for max pooling.
        return m_poolKind == PoolKind::Max;
    }

    void ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        if (m_poolKind == PoolKind::None)
        {
            const Matrix<ElemType>& input0 = Input(0)->ValueAsMatrix();
            Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
            m_convEng->Forward(sliceInput1Value, input0, sliceOutputValue, *m_tempMatrix);
        }
        else
        {
            const Matrix<ElemType>& input0 = Input(0)->ValueFor(fr);
            m_convEng->ForwardPooling(input0, sliceOutputValue);
        }
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        if (m_imageLayout != ImageLayoutKind::CHW)
        {
            InvalidArgument(
                "%ls %ls supports only cuDNN (CHW) data layout. "
                "Please specify imageLayout=\"cudnn\" in Convolution node in your script "
                "and make sure input data layout is CHW", NodeName().c_str(), OperationName().c_str());
        }

        auto inputShape = GetInputSampleLayout(GetExpectedNumInputs() - 1);
        auto dimsOut = ConvolveGeometry::ComputeOutputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                            m_sharing, m_autoPad, m_lowerPad, m_upperPad);
        SetDims(dimsOut, HasMBLayout());

        if (isFinalValidationPass)
        {
            if (m_convEng == nullptr)
            {
                auto geometry = std::make_shared<ConvolveGeometry>(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                   m_sharing, m_autoPad, m_lowerPad, m_upperPad);
                m_convEng = ConvolutionEngine<ElemType>::Create(geometry, m_deviceId, m_imageLayout,
                                                                m_maxTempMemSizeInSamples, m_poolKind);
            }
        }
    }

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_tempMatrix, matrixPool);
    }

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_tempMatrix, matrixPool);
    }

    void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override
    {
        Base::DumpNodeInfo(printValues, printMetadata, fstream);

        if (m_convEng != nullptr)
            fstream << "Geometry: " << string(*m_convEng->Geometry()) << "\n";
        fstream << "PoolKind: " << (int)m_poolKind << "\n";
    }

    void SetmMaxTempMemSizeInSamples(const size_t maxTempMemSizeInSamples)
    {
        m_maxTempMemSizeInSamples = maxTempMemSizeInSamples;
    }

private:

    size_t GetExpectedNumInputs() const
    {
        return m_poolKind == PoolKind::None ? 2 : 1;
    }

private:
    ImageLayoutKind m_imageLayout;

    TensorShape m_kernelShape;
    TensorShape m_mapCount;
    TensorShape m_stride;
    std::vector<bool> m_sharing;
    std::vector<bool> m_autoPad;
    TensorShape m_lowerPad;
    TensorShape m_upperPad;

    size_t m_maxTempMemSizeInSamples;
    shared_ptr<Matrix<ElemType>> m_tempMatrix;

    PoolKind m_poolKind;

    std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;

    bool m_legacy;
};

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
//template <class ElemType>
//class Convolution2DNode : public ComputationNode<ElemType>, public NumInputs<2>
//{
//    typedef ComputationNode<ElemType> Base;
//    UsingComputationNodeMembersBoilerplate;
//    static const std::wstring TypeName()
//    {
//        return L"Convolution2D";
//    }
//
//public:
//    Convolution2DNode(DEVICEID_TYPE deviceId, const wstring& name)
//        : Base(deviceId, name),
//          m_kernelWidth(SIZE_MAX),
//          m_kernelHeight(SIZE_MAX),
//          // initialize to dummy values so we catch missing initialization
//          m_horizontalSubsample(SIZE_MAX),
//          m_verticalSubsample(SIZE_MAX),
//          m_zeroPadding(false),
//          m_maxTempMemSizeInSamples(SIZE_MAX),
//          m_imageLayoutKind(ImageLayoutKind::HWC)
//    {
//        SetDims(ImageDimensions::AsTensorShape(1, 1, 0, m_imageLayoutKind), 0);
//    }
//    Convolution2DNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind,
//                    const bool zeroPadding = false, const size_t maxTempMemSizeInSamples = 0)
//        : Base(deviceId, name),
//          m_outputChannels(outputChannels),
//          m_kernelWidth(kernelWidth),
//          m_kernelHeight(kernelHeight),
//          m_horizontalSubsample(horizontalSubsample),
//          m_verticalSubsample(verticalSubsample),
//          m_zeroPadding(zeroPadding),
//          m_maxTempMemSizeInSamples(maxTempMemSizeInSamples),
//          m_imageLayoutKind(imageLayoutKind)
//    {
//        SetDims(ImageDimensions::AsTensorShape(1, 1, m_outputChannels, m_imageLayoutKind), 0); // TODO: necessary?
//    }
//    Convolution2DNode(const ScriptableObjects::IConfigRecordPtr configp)
//        : Convolution2DNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"kernelWidth"), configp->Get(L"kernelHeight"), configp->Get(L"outputChannels"),
//                          configp->Get(L"horizontalSubsample"), configp->Get(L"verticalSubsample"), ImageLayoutKindFrom(configp->Get(L"imageLayout")),
//                          configp->Get(L"zeroPadding"), configp->Get(L"maxTempMemSizeInSamples"))
//    {
//        // weightNodeName, inputValueNodeName, kernelWidth, kernelHeight, outputChannels, horizontalSubsample, verticalSubsample, zeroPadding = false, maxTempMemSizeInSamples = 0
//        AttachInputs(configp, this->GetExpectedNumInputs());
//    }
//
//    void Save(File& fstream) const override
//    {
//        Base::Save(fstream);
//        fstream << m_kernelWidth << m_kernelHeight << m_horizontalSubsample << m_verticalSubsample;
//        uint32_t imageLayoutKind = (uint32_t) m_imageLayoutKind;
//        uint32_t outputChannels = (uint32_t) m_outputChannels;
//        fstream << outputChannels << imageLayoutKind;
//        fstream << m_zeroPadding << m_maxTempMemSizeInSamples;
//    }
//
//    void Load(File& fstream, size_t modelVersion) override
//    {
//        Base::Load(fstream, modelVersion);
//        fstream >> m_kernelWidth >> m_kernelHeight >> m_horizontalSubsample >> m_verticalSubsample;
//        uint32_t imageLayoutKind, outputChannels;
//        fstream >> outputChannels >> imageLayoutKind;
//        m_imageLayoutKind = (ImageLayoutKind) imageLayoutKind;
//        m_outputChannels = outputChannels;
//        SetDims(ImageDimensions::AsTensorShape(1, 1, m_outputChannels, m_imageLayoutKind), HasMBLayout()); // TODO: needed?
//        fstream >> m_zeroPadding >> m_maxTempMemSizeInSamples;
//    }
//
//    void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
//    {
//        Base::CopyTo(nodeP, newName, flags);
//        if (flags & CopyNodeFlags::copyNodeValue)
//        {
//            auto node = dynamic_pointer_cast<Convolution2DNode<ElemType>>(nodeP);
//            node->m_kernelWidth = m_kernelWidth;
//            node->m_kernelHeight = m_kernelHeight;
//
//            node->m_horizontalSubsample = m_horizontalSubsample;
//            node->m_verticalSubsample = m_verticalSubsample;
//
//            node->m_zeroPadding = m_zeroPadding;
//
//            node->m_maxTempMemSizeInSamples = m_maxTempMemSizeInSamples;
//
//            node->m_imageLayoutKind = m_imageLayoutKind;
//
//            node->m_tempMatrix->SetValue(*m_tempMatrix);
//        }
//    }
//
//    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override
//    {
//        auto sliceOutputGrad = GradientFor(fr);
//        auto sliceInput1Value = Input(1)->ValueFor(fr);
//
//        if (inputIndex == 0) // derivative with respect to the weight matrix
//        {
//            auto& grad = Input(0)->GradientAsMatrix();
//            m_convEng->BackwardKernel(sliceOutputGrad, sliceInput1Value, grad, fr.IsAllFrames(), *m_tempMatrix);
//        }
//        else if (inputIndex == 1) // derivative with respect to the input feature
//        {
//            auto& input0 = Input(0)->ValueAsMatrix();
//            auto sliceInput1Grad = Input(1)->GradientFor(fr);
//            m_convEng->BackwardData(sliceOutputGrad, input0, sliceInput1Grad, *m_tempMatrix);
//        }
//    }
//
//    virtual bool OutputUsedInComputingInputNodesGradients() const override
//    {
//        // The Convolution2DNode does not require its output value for computing
//        // the gradients of its input nodes
//        return false;
//    }
//
//    void ForwardProp(const FrameRange& fr) override
//    {
//        const Matrix<ElemType>& input0 = Input(0)->ValueAsMatrix();
//        Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
//        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
//
//        // update the tensor dimension w.r.t. number of samples
//#if NANCHECK
//        input0.HasNan("Convolution-input0");
//        sliceInput1Value.HasNan("Convolution-input1");
//#endif
//        m_convEng->Forward(sliceInput1Value, input0, sliceOutputValue, *m_tempMatrix);
//#if NANCHECK
//        sliceOutputValue.HasNan("Convolution");
//#endif
//    }
//
//    void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
//    {
//        Base::Validate(isFinalValidationPass);
//        InferMBLayoutFromInputsForStandardCase();
//
//        // get input and output tensor shape and interpret as image dimensions
//        auto inDims = ImageDimensions(GetInputSampleLayout(1), m_imageLayoutKind);
//
//        if (isFinalValidationPass && (inDims.m_width < m_kernelWidth || inDims.m_height < m_kernelHeight))
//            InvalidArgument("%ls %ls operation requires that input width be >= kernelWidth and input height >= kernelHeight.", NodeName().c_str(), OperationName().c_str());
//
//        // determine output tensor shape
//        const int kernelWidthCenter  = m_zeroPadding ? m_kernelWidth  % 2 : m_kernelWidth;
//        const int kernelHeightCenter = m_zeroPadding ? m_kernelHeight % 2 : m_kernelHeight;
//        auto outDims = ImageDimensions(
//            (inDims.m_width  - kernelWidthCenter)  / m_horizontalSubsample + 1,
//            (inDims.m_height - kernelHeightCenter) / m_verticalSubsample   + 1,
//            m_outputChannels);
//
//        size_t weightCols = m_kernelWidth * m_kernelHeight * inDims.m_numChannels;
//
//        // check/infer input [0] (weights)
//        // BUGBUG: For now, we treat the weights as a 2D matrix. They should be a tensor proper.
//        Input(0)->ValidateInferInputDimsFrom(TensorShape(m_outputChannels, weightCols));
//
//        if (isFinalValidationPass && (Input(0)->GetAsMatrixNumCols() != weightCols || Input(0)->GetAsMatrixNumRows() != m_outputChannels))
//            LogicError("convolutionWeight matrix %ls should have dimension [%d, %d] which is [outputChannels, kernelWidth * kernelHeight * inputChannels]", Input(0)->NodeName().c_str(), (int) m_outputChannels, (int) weightCols);
//
//        // that's our dimension
//        SetDims(outDims.AsTensorShape(m_imageLayoutKind), true);
//
//        if (isFinalValidationPass)
//        {
//            // set up the various engines and descriptor objects
//            if (m_convEng == nullptr)
//            {
//                // Note that ConvolveGeometry always uses CHW layout.
//                auto pad = TensorShape(m_zeroPadding ? m_kernelWidth / 2 : 0,
//                                       m_zeroPadding ? m_kernelHeight / 2 : 0,
//                                       0);
//                auto geometry = std::make_shared<ConvolveGeometry>(inDims.AsTensorShape(ImageLayoutKind::CHW),
//                                                                   TensorShape(m_kernelWidth, m_kernelHeight, inDims.m_numChannels),
//                                                                   TensorShape(m_outputChannels),
//                                                                   TensorShape(m_horizontalSubsample, m_verticalSubsample, inDims.m_numChannels),
//                                                                   ConvolveGeometry::BoolVec{true},
//                                                                   // Note: this will have pad=true in channel dimension so must use inDims.m_numChannels stride in c dimension of the stride tensor.
//                                                                   ConvolveGeometry::BoolVec{m_zeroPadding && (m_imageLayoutKind == ImageLayoutKind::CHW)},
//                                                                   pad, pad);
//                m_convEng = ConvolutionEngine<ElemType>::Create(geometry, m_deviceId, m_imageLayoutKind, m_maxTempMemSizeInSamples);
//            }
//        }
//    }
//
//    void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override
//    {
//        Base::DumpNodeInfo(printValues, printMetadata, fstream);
//
//        auto inDims = ImageDimensions(GetInputSampleLayout(1), m_imageLayoutKind);
//        auto outDims = ImageDimensions(m_sampleLayout, m_imageLayoutKind);
//
//        char str[4096];
//        sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", inDims.m_width, inDims.m_height, inDims.m_numChannels);
//        fstream << string(str);
//        sprintf(str, "Kernel[Width:%lu, Height:%lu]  SubSample[Horizontal:%lu, Vertical:%lu]\n", m_kernelWidth, m_kernelHeight, m_horizontalSubsample, m_verticalSubsample);
//        fstream << string(str);
//        sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", outDims.m_width, outDims.m_height, outDims.m_numChannels);
//        fstream << string(str);
//        sprintf(str, "zeroPadding=%ls  maxTempMemSizeInSamples=%lu\n", m_zeroPadding ? L"true" : L"false", m_maxTempMemSizeInSamples);
//        fstream << string(str);
//    }
//
//    void SetmMaxTempMemSizeInSamples(const size_t maxTempMemSizeInSamples)
//    {
//        m_maxTempMemSizeInSamples = maxTempMemSizeInSamples;
//    }
//
//    // request matrices needed to do node function value evaluation
//    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
//    {
//        Base::RequestMatricesBeforeForwardProp(matrixPool);
//        RequestMatrixFromPool(m_tempMatrix, matrixPool);
//    }
//
//    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
//    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
//    {
//        Base::ReleaseMatricesAfterBackprop(matrixPool);
//        ReleaseMatrixToPool(m_tempMatrix, matrixPool);
//    }
//
//private:
//    size_t m_outputChannels;
//    size_t m_kernelWidth, m_kernelHeight;
//    size_t m_horizontalSubsample, m_verticalSubsample;
//    bool m_zeroPadding;
//    bool m_1DConvolutionOnGPUSparse;
//
//    shared_ptr<Matrix<ElemType>> m_tempMatrix;
//    size_t m_maxTempMemSizeInSamples; // can change during runtime
//
//    ImageLayoutKind m_imageLayoutKind; // how to interpret the tensor (which dimensions are X/Y and C)
//
//    std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;
//};

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
        m_imageLayoutKind = (ImageLayoutKind)imageLayoutKind;
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

        m_convEng->BackwardPooling(sliceOutputValue, sliceOutputGrad, sliceInput0Value, sliceInput0Grad);
    }

    void ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        m_convEng->ForwardPooling(sliceInput0Value, sliceOutputValue);
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
            m_geometry = std::make_shared<ConvolveGeometry>(inDims.AsTensorShape(m_imageLayoutKind),
                                                            ImageDimensions(m_windowWidth, m_windowHeight, 1).AsTensorShape(m_imageLayoutKind),
                                                            TensorShape(1),
                                                            ImageDimensions(m_horizontalSubsample, m_verticalSubsample, 1).AsTensorShape(m_imageLayoutKind),
                                                            ConvolveGeometry::BoolVec{true},
                                                            ConvolveGeometry::BoolVec{false},
                                                            TensorShape(0),
                                                            TensorShape(0));
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

    ConvolveGeometryPtr m_geometry;
    std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;
};

// add this at the start of each derived class, to get access to the members of ComputationNode
// See #define of 'UsingComputationNodeMembersBoilerplate' for more explanation.
#define UsingPoolingNodeBaseMembers         \
    UsingComputationNodeMembersBoilerplate; \
    \
protected:                                  \
    using Base::m_geometry;                 \
    using Base::m_convEng;                  \
    using Base::m_windowWidth;              \
    using Base::m_windowHeight;             \
    using Base::m_horizontalSubsample;      \
    using Base::m_verticalSubsample;        \
    using Base::m_inputSizePerSample;       \
    using Base::m_outputSizePerSample;      \
    using Base::m_imageLayoutKind;          \
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
        if (isFinalValidationPass && m_convEng == nullptr)
            m_convEng = ConvolutionEngine<ElemType>::Create(m_geometry, m_deviceId, m_imageLayoutKind, 0, PoolKind::Max);
    }
};

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
        if (isFinalValidationPass && m_convEng == nullptr)
            m_convEng = ConvolutionEngine<ElemType>::Create(m_geometry, m_deviceId, m_imageLayoutKind, 0, PoolKind::Average);
    }
};

} } }
