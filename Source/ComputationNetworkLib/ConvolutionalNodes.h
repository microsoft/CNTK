//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "Globals.h"
#include "Matrix.h"
#include "ComputationNode.h"
#include "ConvolutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// ConvolutionNodeBase
// -----------------------------------------------------------------------

// ConvolutionNodeBase is a base class for ND-convolution(ConvolutionNode) and ND-pooling(PoolingNode).
// 
// 2D convolutions (incl. pooling) support two different storage formats:
//
// * legacy ("HWC") mode: Channels are tuples of scalars
//
//    This follows "high performance convolutional neural networks for document processing" by Kumar Chellapilla, Sidde Puri, and Patrice Simard.
//    Each sample is stored as a column-major matrix (height, width) of float[numChannels] (r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11).
//
//     - input :  [C x W  x H      x T]  or  ARRAY[1..T] OF                ARRAY[1..H]  OF ARRAY[1..W]  OF ARRAY[1..C]
//     - output : [K x W' x H'     x T]  or  ARRAY[1..T] OF                ARRAY[1..H'] OF ARRAY[1..W'] OF ARRAY[1..K]
//     - filter : [K x W" x H" x C    ]  or                 ARRAY[1..C] OF ARRAY[1..H"] OF ARRAY[1..W"] OF ARRAY[1..K]
//
// * cudnn ("CHW") mode (works both GPU and CPU): Channels are planes
//
//     - input :   [W  x H  x C      x T]   or  ARRAY[1..T] OF                ARRAY[1..C]  OF ARRAY[1..H]  OF ARRAY[1..W]
//     - output :  [W' x H' x      K x T]   or  ARRAY[1..T] OF ARRAY[1..K] OF                 ARRAY[1..H'] OF ARRAY[1..W']
//     - filter :  [W" x H" x C  x K    ]   or                 ARRAY[1..K] OF ARRAY[1..C]  OF ARRAY[1..H]  OF ARRAY[1..W]
//
// where:
//  - using ' for output and " for filter
//  - T = samples (NVidia calls this N)
//  - W, H = width, height (W', H' for output, W", H" for kernel)
//  - C = input channels
//     - 3 for color images, 1 for B&W images
//     - for hidden layer: dimension of activation vector for each pixel
//  - K = output channels = dimension of activation vector for each pixel (also called N by NVidia, inconsistently)
//
// For ND-convolution/pooling only second format ('cudnn') is supported.
// 
template <class ElemType>
class ConvolutionNodeBase : public ComputationNode<ElemType>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;

public:
    ConvolutionNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name), m_poolKind(PoolKind::None), m_transpose(false), m_maxTempMemSizeInSamples(0)
    {
    }
    ConvolutionNodeBase(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& strideShape,
                        const std::vector<bool>& sharing, const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                        PoolKind poolKind, bool transpose, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples)
                        : Base(deviceId, name), m_kernelShape(kernelShape), m_mapCount(mapCount), m_stride(strideShape), m_sharing(sharing),
                        m_autoPad(autoPadding), m_lowerPad(lowerPad), m_upperPad(upperPad), m_poolKind(poolKind), m_transpose(transpose),
                        m_imageLayout(imageLayout), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples)
    {
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
        fstream << (int32_t)m_poolKind;
        fstream << (int32_t)m_imageLayout;
        fstream << m_maxTempMemSizeInSamples;
        fstream << m_transpose;
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);

        // Let ConvolutionNode handle older models.
        if (modelVersion >= CNTK_MODEL_VERSION_5)
        {
            m_kernelShape.Load(fstream);
            m_mapCount.Load(fstream);
            m_stride.Load(fstream);
            fstream >> m_sharing;
            fstream >> m_autoPad;
            m_lowerPad.Load(fstream);
            m_upperPad.Load(fstream);
            int32_t k;
            fstream >> k;
            m_poolKind = (PoolKind)k;
            int32_t layout;
            fstream >> layout;
            m_imageLayout = (ImageLayoutKind)layout;
            fstream >> m_maxTempMemSizeInSamples;
        }
        if (modelVersion >= CNTK_MODEL_VERSION_9)
        {
            fstream >> m_transpose;
        }
    }

    void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<ConvolutionNodeBase<ElemType>>(nodeP);
            node->m_kernelShape = m_kernelShape;
            node->m_mapCount = m_mapCount;
            node->m_stride = m_stride;
            node->m_sharing = m_sharing;
            node->m_autoPad = m_autoPad;
            node->m_lowerPad = m_lowerPad;
            node->m_upperPad = m_upperPad;
            node->m_poolKind = m_poolKind;
            node->m_transpose = m_transpose;
            node->m_imageLayout = m_imageLayout;
            node->m_maxTempMemSizeInSamples = m_maxTempMemSizeInSamples;
        }
    }

    void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override
    {
        Base::DumpNodeInfo(printValues, printMetadata, fstream);

        if (m_convEng != nullptr)
            fstream << "Geometry: " << string(*m_convEng->Geometry()) << "\n";
        fstream << "PoolKind: " << (int)m_poolKind << "\n";
    }

    TensorShape KernelShape() const { return m_kernelShape; }
    TensorShape MapCount() const { return m_mapCount; }
    TensorShape Strides() const { return m_stride; }
    std::vector<bool> Sharing() const { return m_sharing; }
    std::vector<bool> AutoPad() const { return m_autoPad; }
    TensorShape LowerPad() const { return m_lowerPad; }
    TensorShape UpperPad() const { return m_upperPad; }
    bool Transpose() const { return m_transpose; }
    size_t MaxTempMemSizeInSamples() const { return m_maxTempMemSizeInSamples; }
    PoolKind PoolingKind() const { return m_poolKind; }

    // bottomlessly expand shape to filterRank, then expand to inputRank using defaults or given 'from' values
    template<class V, typename T>
    static void FixVectorShape(size_t filterRank, size_t inputRank, V& shape, T deflt, const V& from = V())
    {
        if (shape.size() == 0)
            return; // let ComputeOutputShape() deal with this special case
        // repeat the last value until we have the same rank as the filter
        while (shape.size() < filterRank)
            shape.push_back(shape.back());
        // increase to input rank
        // If 'from' is given then clone the value from there. This is meant to be the input dimensions for convolution.
        while (shape.size() < inputRank)
            shape.push_back(shape.size() < from.size() ? from[shape.size()] : deflt);
    }

private:
    static void FixTensorShape(size_t filterRank, size_t inputRank, TensorShape& shape, size_t deflt, const TensorShape& from = TensorShape())
    {
        auto dims = shape.GetDims();
        FixVectorShape(filterRank, inputRank, dims, deflt, from.GetDims());
        shape = TensorShape(dims);
    }
protected:
    // infer reduction dimensions if not given
    void InferReductionDims(const TensorShape& inputShape, const TensorShape& fromShape)
    {
        // If kernel has a lower rank than the input then the remaining dimensions are to be reduced over.
        size_t filterRank = m_kernelShape.size();
        FixTensorShape(filterRank, inputShape.size(), m_kernelShape, 1,     fromShape); // convolve over red dim; pool over 1
        FixTensorShape(filterRank, inputShape.size(), m_stride,      1,     fromShape); // stride for reduction dims is red dim or 1
        FixVectorShape(filterRank, inputShape.size(), m_autoPad,     false);            // no padding for reduction dims
        FixTensorShape(filterRank, inputShape.size(), m_lowerPad,    0);
        FixTensorShape(filterRank, inputShape.size(), m_upperPad,    0);
        FixVectorShape(filterRank, inputShape.size(), m_sharing,     true);
    }

    // Derived classes implement transforms calculation. Since all derived classes are filter based we consolidate common
    // filter transform calculation here to be reused by derived classes. For example convolution and de-convolution
    // have same transform but inversed, hence both of them may reuse this method and one will call inverse in addition
    // (similar holds for pooling nodes).
    SpaceTransform ComputeFilterTransform()
    {
        std::shared_ptr<const ConvolveGeometry> geometry = m_convEng->Geometry();

        SpaceTransform result;
        result.m_axisTransforms.resize(2);

        result.m_axisTransforms[0].scale = (float)(geometry->GetStride(0));
        result.m_axisTransforms[0].translate = (float)((geometry->KernelShape()[0] - 1) / 2 - geometry->GetLowerPad(0));

        result.m_axisTransforms[1].scale = (float)(geometry->GetStride(1));
        result.m_axisTransforms[1].translate = (float)((geometry->KernelShape()[1] - 1) / 2 - geometry->GetLowerPad(1));

        return result;
    }

protected:
    TensorShape m_kernelShape;
    TensorShape m_mapCount;
    TensorShape m_stride;
    std::vector<bool> m_sharing;
    std::vector<bool> m_autoPad;
    TensorShape m_lowerPad;
    TensorShape m_upperPad;
    PoolKind m_poolKind;
    bool m_transpose; // means de-convolution ...I think
    ImageLayoutKind m_imageLayout;
    
    size_t m_maxTempMemSizeInSamples;
    shared_ptr<Matrix<ElemType>> m_tempMatrix;

    std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;
};

#define UsingConvolutionNodeBaseMembers     \
    UsingComputationNodeMembersBoilerplate; \
protected:                                  \
    using Base::m_kernelShape;              \
    using Base::m_mapCount;                 \
    using Base::m_stride;                   \
    using Base::m_sharing;                  \
    using Base::m_autoPad;                  \
    using Base::m_lowerPad;                 \
    using Base::m_upperPad;                 \
    using Base::m_poolKind;                 \
    using Base::m_transpose;                \
    using Base::m_imageLayout;              \
    using Base::m_maxTempMemSizeInSamples;  \
    using Base::m_tempMatrix;               \
    using Base::m_convEng;                  \
    using Base::InferReductionDims;         \
public:

// -----------------------------------------------------------------------
// ConvolutionNode (convolutionWeights, inputFeature)
// -----------------------------------------------------------------------

template <class ElemType>
class ConvolutionNode : public ConvolutionNodeBase<ElemType>, public NumInputs<2>, public TransformerNode
{
    typedef ConvolutionNodeBase<ElemType> Base; UsingConvolutionNodeBaseMembers;
    static const std::wstring TypeName() { return L"Convolution"; }
public:
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& strideShape,
                    const std::vector<bool>& sharing, const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                    bool transpose, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples)
                    : Base(deviceId, name, kernelShape, mapCount, strideShape, sharing, autoPadding, lowerPad, upperPad, PoolKind::None, transpose, imageLayout, maxTempMemSizeInSamples),
                    m_convolution2D(false)
    {
    }
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels,
                    const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayout,
                    bool zeroPadding, size_t maxTempMemSizeInSamples)
                    : ConvolutionNode(deviceId, name, TensorShape(kernelWidth, kernelHeight, 1), TensorShape(1, 1, outputChannels),
                                      TensorShape(horizontalSubsample, verticalSubsample, 1), vector<bool>{true},
                                      vector<bool>{zeroPadding}, TensorShape(0), TensorShape(0),
                                      false, imageLayout, maxTempMemSizeInSamples)
    {
        m_convolution2D = true;
    }
    ConvolutionNode(const ScriptableObjects::IConfigRecordPtr configp)
        : ConvolutionNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"kernelShape"), configp->Get(L"mapCount"), configp->Get(L"strideShape"),
                          configp->Get(L"dimSharing"), configp->Get(L"dimPadding"), configp->Get(L"dimPadLower"), configp->Get(L"dimPadUpper"),
                          configp->Get(L"transpose"), ImageLayoutKindFrom(configp->Get(L"imageLayout")), configp->Get(L"maxTempMemSizeInSamples"))
    {
        AttachInputsFromConfig(configp, GetExpectedNumInputs());
    }

    virtual bool ImplementsGradientOverwriteOptimization() const override { return m_convEng->ImplementsGradientOverwriteOptimization(); }

public:
    void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_convolution2D;
        TensorShape(1).Save(fstream); // Write out a dummy tensor, so that model created can be used later after implementing reading this tensor in this model version
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);

        // Back compat: load pre-ND convolution models.
        if (modelVersion < CNTK_MODEL_VERSION_5)
        {
            size_t kW, kH, sW, sH;
            fstream >> kW;
            fstream >> kH;
            fstream >> sW;
            fstream >> sH;
            uint32_t imageLayout, mapCount;
            fstream >> mapCount;
            fstream >> imageLayout;
            m_imageLayout = (ImageLayoutKind)imageLayout;
            bool pad;
            fstream >> pad;
            fstream >> m_maxTempMemSizeInSamples;
            m_poolKind = PoolKind::None;
            m_convolution2D = true;

            m_kernelShape = TensorShape(kW, kH, 1);
            m_mapCount = TensorShape(mapCount);
            m_stride = TensorShape(sW, sH, 1);
            m_sharing = vector<bool>{true};
            m_autoPad = vector<bool>{pad};
            m_lowerPad = TensorShape(0);
            m_upperPad = TensorShape(0);
        }
        else
        {
            fstream >> m_convolution2D;
            if (modelVersion >= CNTK_MODEL_VERSION_18)
            {
                TensorShape dummyTensorHolder;
                dummyTensorHolder.Load(fstream);
                if(dummyTensorHolder!=TensorShape(1)) LogicError("Loading tensor that is currently not supported.");
            }
        }
    }

    void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<ConvolutionNode<ElemType>>(nodeP);
            node->m_convolution2D = m_convolution2D;
        }
    }

    void ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
        const Matrix<ElemType>& input0 = InputRef(0).ValueAsMatrix();
        Matrix<ElemType> sliceInput1Value = InputRef(1).ValueFor(fr);
        if (!m_transpose)
            m_convEng->Forward(sliceInput1Value, input0, sliceOutputValue, *m_tempMatrix);
        else
        {
            // BackwardData adds results to the output so need to zero them out first.
            // REVIEW alexeyk: should be rolled into BackwardData itself.
            sliceOutputValue.SetValue(0);
            m_convEng->BackwardData(sliceInput1Value, input0, sliceOutputValue, /*accumulateGradient =*/ true, *m_tempMatrix);
        }
    }

    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        auto sliceOutputGrad = GradientFor(fr);
        if (inputIndex == 0) // derivative with respect to the weight matrix
        {
            auto& grad = InputRef(0).GradientAsMatrix();
            auto sliceInput1Value = InputRef(1).ValueFor(fr);
            if (!m_transpose)
                m_convEng->BackwardKernel(sliceOutputGrad, sliceInput1Value, grad, !Input(inputIndex)->ParentOverwritesGradient(), fr.IsAllFrames(), *m_tempMatrix);
            else
                m_convEng->BackwardKernel(sliceInput1Value, sliceOutputGrad, grad, !Input(inputIndex)->ParentOverwritesGradient(), fr.IsAllFrames(), *m_tempMatrix);
        }
        else if (inputIndex == 1) // derivative with respect to the input feature
        {
            auto& input0 = InputRef(0).ValueAsMatrix();
            auto sliceInput1Grad = InputRef(1).GradientFor(fr);
            if (!m_transpose)
                m_convEng->BackwardData(sliceOutputGrad, input0, sliceInput1Grad, !Input(inputIndex)->ParentOverwritesGradient(), *m_tempMatrix);
            else
            {
                // REVIEW alexeyk: Forward overwrites values in sliceInput1Grad. Should handle correctly instead.
                m_convEng->Forward(sliceOutputGrad, input0, sliceInput1Grad, *m_tempMatrix);
            }
        }
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        size_t inputIdx = GetExpectedNumInputs() - 1;
        TensorShape inputShape;
        TensorShape outputShape;
        // If 2D convolution syntax is used then some of the tensor dimensions need to be inferred.
        if (m_convolution2D)
        // NOTE: when m_convolution2D is true, it's a legacy branch. Code should not enter here any more. 
        {
            // Need to update some tensors with correct input dims.
            auto inDims = ImageDimensions(GetInputSampleLayout(inputIdx), m_imageLayout);
            // inputShape is used in ConvolveGeometry which supports only CHW layout.
            inputShape = inDims.AsTensorShape(ImageLayoutKind::CHW);
            size_t kW = m_kernelShape[0];
            size_t kH = m_kernelShape[1];
            size_t sW = m_stride[0];
            size_t sH = m_stride[1];
            m_kernelShape = TensorShape(kW, kH, inDims.m_numChannels);
            m_stride = TensorShape(sW, sH, inDims.m_numChannels);

            size_t mapCount = m_mapCount.GetNumElements();
            size_t weightCols = kW * kH * inDims.m_numChannels;

            // if mapCount is 0 then take it from the input matrix
            if (mapCount == 0)
                Input(0)->GetAsMatrixNumRows();

            // check/infer input [0] (weights)
            // BUGBUG: For now, we treat the weights as a 2D matrix. They should be a tensor proper.
            Input(0)->ValidateInferInputDimsFrom(TensorShape(mapCount, weightCols));

            if (isFinalValidationPass && (Input(0)->GetAsMatrixNumCols() != weightCols || Input(0)->GetAsMatrixNumRows() != mapCount))
            {
                LogicError("Convolution weight matrix %ls should have dimension [%d, %d] which is [outputChannels, kernelWidth * kernelHeight * inputChannels]",
                           Input(0)->NodeName().c_str(), (int)mapCount, (int)weightCols);
            }

            outputShape = ConvolveGeometry::ComputeOutputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                m_sharing, m_autoPad, m_lowerPad, m_upperPad);
            // ConvolveGeometry always uses CHW.
            SetDims(ImageDimensions(outputShape, ImageLayoutKind::CHW).AsTensorShape(m_imageLayout), HasMBLayout());
        }
        else
        {
            inputShape = GetInputSampleLayout(inputIdx);
            // infer reduction dimensions if not given
            InferReductionDims(inputShape, inputShape);
            if (!m_transpose)
            {
                outputShape = ConvolveGeometry::ComputeOutputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                    m_sharing, m_autoPad, m_lowerPad, m_upperPad);
            }
            else
            {
                // In case of transpose (deconvolution), node input (inputShape) is really the output of the convolution
                // and node output (outDims) is convolution input. ConvolveGeometry does not care about deconvolutions (it does not have to).
                outputShape = ConvolveGeometry::ComputeInputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                   m_sharing, m_autoPad, m_lowerPad, m_upperPad);
            }

            if (m_imageLayout == ImageLayoutKind::CHW) 
                SetDims(outputShape, HasMBLayout());
            else    // legacy format 
                SetDims(ImageDimensions(outputShape, ImageLayoutKind::CHW).AsTensorShape(m_imageLayout), HasMBLayout());
        }

        // update LearnableParameter if it has 0 dimensions (to be inferred)
        // Typically this would be the #inputChannels (C).
        if (Input(0)->GetSampleLayout().GetNumElements() == 0)
        {
            // BUGBUG: Inference does not support sharing. Problem is that we have the information too late.
            //         In this case, users will have to specify the correct dimensions. Good luck.
#if 1       // old style for back compat with previous results. Randomization will differ.
            if (Input(0)->GetSampleLayout().GetRank() == 2)
                Input(0)->ValidateInferInputDimsFrom(TensorShape(m_mapCount.GetNumElements(), m_kernelShape.GetNumElements()));
            else
#endif
            {
                auto weightShape = m_kernelShape.GetDims();
                for (auto outDim : m_mapCount.GetDims())
                    weightShape.push_back(outDim);
                Input(0)->ValidateInferInputDimsFrom(TensorShape(weightShape));
            }
        }

        if (isFinalValidationPass)
        {
            if (m_convEng == nullptr)
            {
                auto geometry = std::make_shared<ConvolveGeometry>(!m_transpose ? inputShape : outputShape,
                                                                   m_kernelShape, m_mapCount, m_stride, 
                                                                   m_sharing, m_autoPad, m_lowerPad, m_upperPad);
                m_convEng = ConvolutionEngine<ElemType>::Create(geometry, m_deviceId, m_imageLayout,
                                                                m_maxTempMemSizeInSamples, m_poolKind,
                                                                ConvolutionEngineKind::All, NodeName(), Globals::ShouldForceDeterministicAlgorithms());
            }

            if (Input(0)->GetSampleLayout().GetNumElements() != m_kernelShape.GetNumElements() * m_convEng->Geometry()->KernelCount())
            {
                //LogicError("Convolution weight matrix %ls should have dimension [%d, %d] which is [kernelCount, kernelWidth * kernelHeight * inputChannels]",
                //           Input(0)->NodeName().c_str(), (int)m_convEng->Geometry()->KernelCount(), (int)m_kernelShape.GetNumElements());
                LogicError("Convolution weight matrix %ls should have dimension [(filter shape) x (input channels) x (output channels)]",
                           Input(0)->NodeName().c_str());
            }
        }
    }

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_tempMatrix, matrixPool);
    }

    //void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) override
    //{
    //    Base::ReleaseMatricesAfterForwardProp(matrixPool);
    //    ReleaseMatrixToPool(m_tempMatrix, matrixPool);
    //}

    //void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
    //{
    //    Base::RequestMatricesBeforeBackprop(matrixPool);
    //    RequestMatrixFromPool(m_tempMatrix, matrixPool);
    //}

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_tempMatrix, matrixPool);
    }

    void SetmMaxTempMemSizeInSamples(const size_t maxTempMemSizeInSamples)
    {
        m_maxTempMemSizeInSamples = maxTempMemSizeInSamples;
        if (m_convEng != nullptr)
            m_convEng->SetmMaxTempMemSizeInSamples(maxTempMemSizeInSamples);
    }

    bool IsConvolution2D() const { return m_convolution2D; }

private:
    using TransformerNode::m_transforms;
    using ConvolutionNodeBase<ElemType>::ComputeFilterTransform;

    virtual void /*TransformerNode::*/ComputeTransforms() override
    {
        if (m_transforms[1].m_axisTransforms.empty())
        {
            m_transforms[1] = ComputeFilterTransform();
            if (!m_transpose)
            {
                // Convolution, need to inverse transform.
                m_transforms[1] = m_transforms[1].Inverse();
            }
            // else: Deconvolution, nothing to do.
        }
        // else: transform already computed, no need to do computation again.
    }

    virtual bool /*TransformerNode::*/SupportsTransformOnInput(size_t inputIndex) override
    {
        // We support transforms just on convolution input.
        return (inputIndex == 1);
    }

protected:
    // Flag that indicates whether the node is created using 2D-syntax.
    bool m_convolution2D;
};

// -----------------------------------------------------------------------
// ROIPoolingNode (inputFeatures, inputROIs)--pooling for object detection.
//
// Each input image has a fixed number of regions of interest (ROIs),
// specified as bounding boxes (x, y, w, h) that are relative to the
// image size [W x H]. This node is meant as a replacement for the
// final pooling layer of an image classification network.  The first
// fully-connected layer expects a fixed size input, but for object
// detection we want each ROI to look like an image to the network so
// we can get a label for it. The ROIs have different spatial sizes,
// so this node does Max Pooling, but with an adaptive pooling window,
// so that each ROI output has the spatial size expected by the first
// fully-connected layer. Images are Input(0). ROIs are Input(1). 
//
// Input0: Images       [W x H x C x N]
// Input1: ROIs         [4 x roisPerImage x N], 
// output: Pooled ROIs  [PW x PH x C x roisPerImage x N]
// where PW = Pooled Width, PH = Pooled Height, C = Channels, N = Batch Size
//
// See http://arxiv.org/abs/1504.08083
// -----------------------------------------------------------------------
template <class ElemType>
class ROIPoolingNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"ROIPooling"; }
public:
    ROIPoolingNode(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& roiOutputShape = TensorShape())
        : Base(deviceId, name), m_roiOutputShape(roiOutputShape), m_argmaxData(Matrix<ElemType>::Zeros(1, 1, deviceId))
    {
    }
    ROIPoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
        : ROIPoolingNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"roiOutputShape"))
    {
        AttachInputsFromConfig(configp, GetExpectedNumInputs());
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

    // Input0: Images       [W x H x C x N]
    // Input1: ROIs         [4 x roisPerImage x N], 
    // output: Pooled ROIs  [PW x PH x C x roisPerImage x N]
    // where PW = Pooled Width, PH = Pooled Height, C = Channels, N = Batch Size
    //
    // Explanation: this node has a target output shape of 
    // [Pooled Width x Pooled Height x Channels], as does any pooling
    // layer. However, we want each /ROI/ to have that output size,
    // not each image. After this node, operations in the network
    // should be on ROIs, not on the full images. The forward pass
    // loops over images and the ROIs associated with each image; for
    // every ROI, it treats the subset of the image specified by that
    // ROI as a full image and does max pooling over that subset,
    // using whatever window size will correspond to an output of
    // [Pooled Width x Pooled Height x Channels]. Hence, 
    // the output tensor is [PW x PH x C x roisPerImage x N]
    // An example validation output looks like this:
    // Validating --> z.roiOut = ROIPooling (z.conv5Out.conv5.y, rois) : [61 x 61 x 256 x *], [4 x 64 x *] -> [6 x 6 x 256 x 64 x *]
    void ForwardProp(const FrameRange& fr) override
    {
        // [4 x roisPerImage x N] -- first dimension is roiSize (4), second is rois-per-image, third is mb size
        size_t roisPerImage = (size_t)GetInputSampleLayout(1)[1];

        auto inputShape = GetInputSampleLayout(0);
        Matrix<ElemType> inputSlice = Input(0)->ValueFor(fr);
        Matrix<ElemType> ROIs = Input(1)->ValueFor(fr);

        // our output slice for this minibatch.
        Matrix<ElemType> outputSlice = ValueFor(fr);

        // input slice is [W x H x C x N]; cols are images.
        // ROIs is [4 x roisPerImage x N]; cols are ROIs for different images.
        // each ROI is (x, y, w, h) relative to original image size.
        size_t inputW = (size_t)inputShape[0];
        size_t inputH = (size_t)inputShape[1];
        size_t numChannels = (size_t)inputShape[2];
        size_t outW = m_roiOutputShape[0];
        size_t outH = m_roiOutputShape[1];

        m_tempMatrix->Resize(outW * outH * numChannels * roisPerImage, inputSlice.GetNumCols());
        inputSlice.ROIPoolingForward(roisPerImage, inputSlice.GetNumCols(), 
            numChannels, inputW, inputH, outW, outH, ROIs, outputSlice, *m_tempMatrix);
    }

    void Save(File& fstream) const override
    {
        Base::Save(fstream);
        m_roiOutputShape.Save(fstream);
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        m_roiOutputShape.Load(fstream);
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        auto inShape = GetInputSampleLayout(0);   // layout of input shape is width x height x numChannels
        auto roiShape = GetInputSampleLayout(1); // layout of ROI shape is 4 x roisPerImage

        if (isFinalValidationPass && (m_roiOutputShape.size() != 2))
            InvalidArgument("ROIPoolingNode: roi output shape must have two dimensions ([W x H]).");

        if (isFinalValidationPass && (inShape[0] < m_roiOutputShape[0] || inShape[1] < m_roiOutputShape[1]))
            InvalidArgument("ROIPoolingNode: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

        if (isFinalValidationPass && (inShape[2] < 1))
            InvalidArgument("ROIPoolingNode: input must have at least one channel ([W x H x C]).");

        if (isFinalValidationPass && (roiShape[0] != 4))
            InvalidArgument("ROIPoolingNode: ROI input must have the following shape: [4 x roisPerImage].");

        if (isFinalValidationPass && (roiShape[1] < 1))
            InvalidArgument("ROIPoolingNode: ROI input must contain at least one ROI ([4 x roisPerImage]).");

        // set output dimensions to [W x H x C x roisPerImage]
        SetDims(TensorShape(m_roiOutputShape[0], m_roiOutputShape[1], inShape[2], roiShape[1]), HasMBLayout());
    }

    // similar to usual MaxPooling backpropagation. Send gradients
    // back through to the locations that were used as the "max." Only
    // difference: needs to sum gradients over all the ROIs that may
    // have used that location. One image location could be in
    // multiple ROIs--in that case each ROI may contribute a gradient term.
    void BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
    {
        auto inputShape = GetInputSampleLayout(0);
        Matrix<ElemType> inputSlice = Input(0)->ValueFor(fr);

        int inputW = inputShape[0];
        int inputH = inputShape[1];
        int numChannels = inputShape[2];

        auto inputGrad = Input(0)->GradientFor(fr);
        auto pooledGrad = GradientFor(fr);

        int roisPerImage = GetInputSampleLayout(1)[1];
        auto roiData = Input(1)->ValueFor(fr);

        pooledGrad.ROIPoolingBackward(roisPerImage, inputSlice.GetNumCols(), numChannels, 
            inputW, inputH, m_roiOutputShape[0], m_roiOutputShape[1], roiData, inputGrad, *m_tempMatrix);
    }

    void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<ROIPoolingNode<ElemType>>(nodeP);
            node->m_roiOutputShape = m_roiOutputShape;
        }
    }

    TensorShape ROIOutputShape() const { return m_roiOutputShape; }

protected:
    TensorShape m_roiOutputShape;
    shared_ptr<Matrix<ElemType>> m_tempMatrix;
    Matrix<ElemType> m_argmaxData;
};

// -----------------------------------------------------------------------
// PoolingNode (inputFeature)
// Performs max or average ND pooling.
// -----------------------------------------------------------------------

template <class ElemType>
class PoolingNode : public ConvolutionNodeBase<ElemType>, public NumInputs<1>, public TransformerNode
{
    typedef ConvolutionNodeBase<ElemType> Base; UsingConvolutionNodeBaseMembers;
    static const std::wstring TypeName() { return L"Pooling"; }
public:
    PoolingNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    PoolingNode(DEVICEID_TYPE deviceId, const wstring& name, PoolKind pool, const TensorShape& kernelShape, const TensorShape& strideShape,
                const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                ImageLayoutKind imageLayout)
                : Base(deviceId, name, kernelShape, TensorShape(1), strideShape, vector<bool>{true}, autoPadding, lowerPad, upperPad, pool, false, imageLayout, 0)
    {
    }
    PoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
        : PoolingNode(configp->Get(L"deviceId"), L"<placeholder>", PoolKindFrom(configp->Get(L"pool")), configp->Get(L"kernelShape"),
                      configp->Get(L"strideShape"),
                      configp->Get(L"dimPadding"), configp->Get(L"dimPadLower"), configp->Get(L"dimPadUpper"),
                      ImageLayoutKindFrom(configp->Get(L"imageLayout")))
    {
        AttachInputsFromConfig(configp, GetExpectedNumInputs());
    }

public:
    void ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
        const Matrix<ElemType>& input0 = InputRef(0).ValueFor(fr);
        m_convEng->ForwardPooling(input0, sliceOutputValue);
    }

    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        auto sliceOutputGrad = GradientFor(fr);
        Matrix<ElemType> sliceInput0Grad = InputRef(0).GradientFor(fr);
        Matrix<ElemType> sliceInput0Value = InputRef(0).ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        m_convEng->BackwardPooling(sliceOutputValue, sliceOutputGrad, sliceInput0Value, sliceInput0Grad);
    }

    bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The PoolingNode requires output values only for max pooling.
        return m_poolKind == PoolKind::Max;
    }

public:
    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        if (m_imageLayout != ImageLayoutKind::CHW)
        {
            InvalidArgument(
                "%ls %ls supports only cuDNN (CHW) data layout. "
                "Please specify imageLayout=\"cudnn\" in %ls node in your script "
                "and make sure input data layout is CHW", NodeName().c_str(), OperationName().c_str(), NodeName().c_str());
        }

        const auto& inputShape = GetInputSampleLayout(0);

        // infer reduction dimensions if not given
        InferReductionDims(inputShape, TensorShape());

        auto outDims = ConvolveGeometry::ComputeOutputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                            m_sharing, m_autoPad, m_lowerPad, m_upperPad);
        SetDims(outDims, HasMBLayout());
        if (isFinalValidationPass)
        {
            if (m_convEng == nullptr)
            {
                auto geometry = std::make_shared<ConvolveGeometry>(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                   m_sharing, m_autoPad, m_lowerPad, m_upperPad);
                m_convEng = ConvolutionEngine<ElemType>::Create(geometry, m_deviceId, m_imageLayout,
                                                                m_maxTempMemSizeInSamples, m_poolKind,
                                                                ConvolutionEngineKind::All, NodeName());
            }
        }
    }

private:
    using TransformerNode::m_transforms;
    using ConvolutionNodeBase<ElemType>::ComputeFilterTransform;

    virtual void /*TransformerNode::*/ComputeTransforms() override
    {
        if (m_transforms[0].m_axisTransforms.empty())
        {
            m_transforms[0] = ComputeFilterTransform();
            m_transforms[0] = m_transforms[0].Inverse();
        }
        // else: transform already computed, no need to do it again.
    }

    virtual bool /*TransformerNode::*/SupportsTransformOnInput(size_t /*inputIndex*/) override
    {
        // We support transforms on all inputs (one here).
        return true;
    }
};

// -----------------------------------------------------------------------
// MaxUnpoolingNode (unpoolInputValues, poolInputValues)
// Performs "max unpooling" operation. Max unpooling mirrors the operation
// performed by max pooling node and depends on the values provided to
// the max pooling node (so unlike deconvolution operation, it is not
// completely independent). Unpooling takes 2 inputs: features to be unpooled,
// which tensor has the same shape as corresponding max pooling node output
// and inputs for the original pooling node. Unpooling node
// produces an output which has the same dimensions as input to the
// corresponding max pooling node (i.e. poolInputValues).
// TODO: need to add support for other pooling types, for example,
// average unpooling. Note that in this case, generic unpooling operation
// will take different number of inputs depending on pooling type.
// -----------------------------------------------------------------------

template <class ElemType>
class MaxUnpoolingNode : public ConvolutionNodeBase<ElemType>, public NumInputs<2>, public TransformerNode
{
    typedef ConvolutionNodeBase<ElemType> Base;
    UsingConvolutionNodeBaseMembers;
    static const std::wstring TypeName() { return L"MaxUnpooling"; }

public:
    MaxUnpoolingNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    MaxUnpoolingNode(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& kernelShape, const TensorShape& strideShape,
                       const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                       ImageLayoutKind imageLayout)
                       : Base(deviceId, name, kernelShape, TensorShape(1), strideShape, vector<bool>{true}, autoPadding, lowerPad, upperPad, PoolKind::Max, true, imageLayout, 0)
    {
    }
    MaxUnpoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
        : MaxUnpoolingNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"kernelShape"),
                           configp->Get(L"strideShape"), configp->Get(L"dimPadding"), configp->Get(L"dimPadLower"), configp->Get(L"dimPadUpper"),
                           ImageLayoutKindFrom(configp->Get(L"imageLayout")))
    {
        AttachInputsFromConfig(configp, GetExpectedNumInputs());
    }

public:
    void ForwardProp(const FrameRange& fr) override
    {
        const Matrix<ElemType>& unpoolInput = InputRef(0).ValueFor(fr);
        const Matrix<ElemType>& poolInput = InputRef(1).ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
        m_convEng->MaxUnpooling(unpoolInput, poolInput, sliceOutputValue);
    }

    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        if (inputIndex != 0)
            return;

        auto sliceOutputGrad = GradientFor(fr);
        Matrix<ElemType> sliceInput0Grad = InputRef(0).GradientFor(fr);
        // BUGBUG: ForwardPooling overwrites values in sliceInput1Grad. Should handle correctly instead.
        m_convEng->ForwardPooling(sliceOutputGrad, sliceInput0Grad);
    }

    bool OutputUsedInComputingInputNodesGradients() const override { return false; }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        if (m_imageLayout != ImageLayoutKind::CHW)
        {
            InvalidArgument(
                "%ls %ls supports only cuDNN (CHW) data layout. "
                "Please specify imageLayout=\"cudnn\" in %ls node in your script "
                "and make sure input data layout is CHW", NodeName().c_str(), OperationName().c_str(), NodeName().c_str());
        }

        auto inputShape = GetInputSampleLayout(0);

        // infer reduction dimensions if not given
        InferReductionDims(inputShape, TensorShape());

        // Same as in case of deconvolution, node input (inputShape) is really the output of the max pooling
        // and node output (outDims) is pooling input.
        auto outputShape = GetInputSampleLayout(1);
        auto inferredShape = ConvolveGeometry::ComputeOutputShape(outputShape, m_kernelShape, m_mapCount, m_stride,
                                                               m_sharing, m_autoPad, m_lowerPad, m_upperPad);
        if (inputShape != inferredShape)
            InvalidArgument("%ls %ls the shape of the unpooling operand %ls is different from "
                            "the result of pooling the poolingInput argument using"
                            "the provided options %ls", NodeName().c_str(), OperationName().c_str(), 
                            static_cast<std::wstring>(inputShape).c_str(), 
                            static_cast<std::wstring>(inferredShape).c_str());

        SetDims(outputShape, HasMBLayout());
        if (isFinalValidationPass)
        {
            if (m_convEng == nullptr)
            {
                auto geometry = std::make_shared<ConvolveGeometry>(outputShape, m_kernelShape, m_mapCount, m_stride,
                                                                   m_sharing, m_autoPad, m_lowerPad, m_upperPad);
                // Create reference engine as it's the only engine that implements unpooling.
                m_convEng = ConvolutionEngine<ElemType>::Create(geometry, m_deviceId, m_imageLayout,
                                                                m_maxTempMemSizeInSamples, m_poolKind,
                                                                ConvolutionEngineKind::Reference,
                                                                NodeName());
            }
        }
    }

private:
    using TransformerNode::m_transforms;
    using ConvolutionNodeBase<ElemType>::ComputeFilterTransform;

    virtual void /*TransformerNode::*/ComputeTransforms() override
    {
        if (m_transforms.empty())
        {
            m_transforms[0] = ComputeFilterTransform();
        }
        // else: transform already computed, no need to do it again.
    }

    virtual bool /*TransformerNode::*/SupportsTransformOnInput(size_t inputIndex) override
    {
        // We support transform for just unpool input.
        return (inputIndex == 0);
    }
};

// -----------------------------------------------------------------------
// Legacy PoolingNodeBase (input)
// -----------------------------------------------------------------------

template <class ElemType>
class PoolingNodeBase : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembers;

public:
    PoolingNodeBase(DEVICEID_TYPE deviceId, const wstring& name, PoolKind poolKind)
        : Base(deviceId, name),
        m_windowWidth(SIZE_MAX),
        m_windowHeight(SIZE_MAX),
        m_horizontalSubsample(SIZE_MAX),
        m_verticalSubsample(SIZE_MAX),
        m_imageLayoutKind(ImageLayoutKind::HWC),
        m_poolKind(poolKind)
    {
    }
    PoolingNodeBase(DEVICEID_TYPE deviceId, const wstring& name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind, PoolKind poolKind)
        : Base(deviceId, name),
        m_windowWidth(windowWidth),
        m_windowHeight(windowHeight),
        m_horizontalSubsample(horizontalSubsample),
        m_verticalSubsample(verticalSubsample),
        m_imageLayoutKind(imageLayoutKind),
        m_poolKind(poolKind)
    {
        ConvertToTensorShape();
    }
    PoolingNodeBase(const ScriptableObjects::IConfigRecordPtr configp, PoolKind poolKind)
        : PoolingNodeBase(configp->Get(L"deviceId"), 
            L"<placeholder>", 
            configp->Get(L"windowWidth"), 
            configp->Get(L"windowHeight"), 
            configp->Get(L"horizontalSubsample"), 
            configp->Get(L"verticalSubsample"),
            ImageLayoutKindFrom(configp->Get(L"imageLayout")), 
            poolKind)
    {
        // input, windowWidth, windowHeight, horizontalSubsample, verticalSubsample
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    void Save(File& fstream) const override
    {
        Base::Save(fstream);
        uint32_t imageLayoutKind = (uint32_t)m_imageLayoutKind;
        uint32_t windowWidth = (uint32_t)m_windowWidth;
        fstream << windowWidth << imageLayoutKind << m_windowHeight << m_horizontalSubsample << m_verticalSubsample;
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        uint32_t imageLayoutKind, windowWidth;
        fstream >> windowWidth >> imageLayoutKind >> m_windowHeight >> m_horizontalSubsample >> m_verticalSubsample;
        m_windowWidth = windowWidth;
        m_imageLayoutKind = (ImageLayoutKind)imageLayoutKind;

        ConvertToTensorShape();
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

            node->ConvertToTensorShape();
        }
    }

    void ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Value = InputRef(0).ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        m_convEng->ForwardPooling(sliceInput0Value, sliceOutputValue);
    }

    void BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Grad = InputRef(0).GradientFor(fr);
        Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

        Matrix<ElemType> sliceInput0Value = InputRef(0).ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        m_convEng->BackwardPooling(sliceOutputValue, sliceOutputGrad, sliceInput0Value, sliceInput0Grad);
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

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

        SetDims(outDims.AsTensorShape(m_imageLayoutKind), HasMBLayout());

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
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", (unsigned long)inputSampleLayout[1], (unsigned long)inputSampleLayout[2], (unsigned long)inputSampleLayout[0]);
            fstream << string(str);
            sprintf(str, "PoolingWindow[Width:%lu, Height:%lu]  SubSampling[Horizontal:%lu, Vertical:%lu]\n", (unsigned long)m_windowWidth, (unsigned long)m_windowHeight, (unsigned long)m_horizontalSubsample, (unsigned long)m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", (unsigned long)m_sampleLayout[1], (unsigned long)m_sampleLayout[2], (unsigned long)m_sampleLayout[0]);
            fstream << string(str);
            sprintf(str, "TotalSizePerSample[Input:%lu, Output:%lu]  \n", (unsigned long)m_inputSizePerSample, (unsigned long)m_outputSizePerSample);
            fstream << string(str);
        }
    }

    bool IsImageLayoutCHW() const { return m_imageLayoutKind == ImageLayoutKind::CHW; }
    TensorShape KernelShape() const { return m_kernelShape; }
    TensorShape Strides() const { return m_stride; }
    std::vector<bool> Sharing() const { return m_sharing; }
    std::vector<bool> AutoPad() const { return m_autoPad; }
    TensorShape LowerPad() const { return m_lowerPad; }
    TensorShape UpperPad() const { return m_upperPad; }
    PoolKind PoolingKind() const { return m_poolKind; }

protected:
    void ConvertToTensorShape()
    {
        m_kernelShape = ImageDimensions(m_windowWidth, m_windowHeight, 1).AsTensorShape(m_imageLayoutKind);
        m_stride      = ImageDimensions(m_horizontalSubsample, m_verticalSubsample, 1).AsTensorShape(m_imageLayoutKind);
        m_sharing     = { true };
        m_autoPad     = { false };
        m_lowerPad    = TensorShape(0);
        m_upperPad    = TensorShape(0);
    }

protected:
    size_t m_windowWidth, m_windowHeight;
    size_t m_horizontalSubsample, m_verticalSubsample;
    size_t m_inputSizePerSample, m_outputSizePerSample;

    ImageLayoutKind m_imageLayoutKind; // how to interpret the tensor (which dimensions are X/Y and C)

    // Mapping to V2 PoolingNode description..
    PoolKind m_poolKind;
    TensorShape m_kernelShape;
    TensorShape m_stride;
    std::vector<bool> m_sharing;
    std::vector<bool> m_autoPad;
    TensorShape m_lowerPad;
    TensorShape m_upperPad;

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
// Legacy MaxPoolingNode
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
        : Base(deviceId, name, PoolKind::Max)
    {
    }
    MaxPoolingNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind)
        : Base(deviceId, name, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayoutKind, PoolKind::Max)
    {
    }
    MaxPoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp, PoolKind::Max)
    {
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (isFinalValidationPass && m_convEng == nullptr)
        {
            m_convEng = ConvolutionEngine<ElemType>::Create(m_geometry, m_deviceId, m_imageLayoutKind,
                                                            0, PoolKind::Max,
                                                            ConvolutionEngineKind::All, NodeName());
        }
    }
};

// -----------------------------------------------------------------------
// Legacy AveragePoolingNode
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
        : Base(deviceId, name, PoolKind::Average)
    {
    }
    AveragePoolingNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind)
        : Base(deviceId, name, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayoutKind, PoolKind::Average)
    {
    }
    AveragePoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp, PoolKind::Average)
    {
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (isFinalValidationPass && m_convEng == nullptr)
        {
            m_convEng = ConvolutionEngine<ElemType>::Create(m_geometry, m_deviceId, m_imageLayoutKind,
                                                            0, PoolKind::Average, 
                                                            ConvolutionEngineKind::All, NodeName());
        }
    }
};

} } }
