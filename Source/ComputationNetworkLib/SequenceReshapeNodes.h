//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "Matrix.h"
#include "CNTKLibrary.h"
#include "Utils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// ToSequenceNodeBase(sourceData)
// Abstract base class for ToSequenceNode and ToSequenceLikeNode.
// Converts the sourceData to a sequence using the most significant static axis [-1] as the sequence axis.
// -----------------------------------------------------------------------

template <class ElemType>
class ToSequenceNodeBase : public ComputationNodeNonLooping<ElemType>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembers; using Base::OperationName;

public:
    ToSequenceNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {}

    virtual std::vector<size_t> GetSequenceLengths() { NOT_IMPLEMENTED; }
    virtual MBLayoutPtr DetermineMBLayout() { NOT_IMPLEMENTED; }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        let& inMBLayout = InputRef(0).GetMBLayout();
        if (inMBLayout->GetNumTimeSteps() != 1)
            InvalidArgument("%ls %ls operation can only operate on non-sequence input data.", NodeName().c_str(), OperationName().c_str());

        auto numSequences = inMBLayout->GetNumSequences();
        auto inputSampleLayout = InputRef(0).GetSampleLayout();
        auto inputDataTensorShape = inputSampleLayout;
        inputDataTensorShape.AppendInPlace(inputDataTensorShape.GetRank(), numSequences);
        let& inputDataMatrix = InputRef(0).Value();
        auto inputDataNDArrayView = ::CNTK::MakeSharedObject<::CNTK::NDArrayView>(::CNTK::AsDataType<ElemType>(),
            ::CNTK::AsDeviceDescriptor(inputDataMatrix.GetDeviceId()),
            ::CNTK::AsStorageFormat(inputDataMatrix.GetFormat()),
            ::CNTK::AsNDShape(inputDataTensorShape),
            /*readOnly =*/ true,
            new TensorView<ElemType>(InputRef(0).ValuePtr(), inputDataTensorShape));

        std::vector<size_t> sequenceLengths = GetSequenceLengths();
        auto inputDataValue = ::CNTK::MakeSharedObject<::CNTK::Value>(inputDataNDArrayView, ::CNTK::CreateMask(sequenceLengths));
        auto dummyVar = ::CNTK::InputVariable(::CNTK::AsNDShape(GetSampleLayout()), this->IsValueSparse(), ::CNTK::AsDataType<ElemType>());
#ifdef _MSC_VER
        auto& outputValuePtrRef = ValuePtrRef();
#else
        auto& outputValuePtrRef = this->template ValuePtrRef();
#endif
        auto packedMatrixAndLayout = ::CNTK::Utils::GetCNTKImplMatrixAndMBLayoutFromValueObject(dummyVar, inputDataValue, nullptr, outputValuePtrRef, m_tempGatherIndices);

        let& outMBLayout = GetMBLayout();
        outMBLayout->CopyFrom(packedMatrixAndLayout.second, /*keepName=*/true);
        if (packedMatrixAndLayout.first != outputValuePtrRef)
            Value().AssignValuesOf(*packedMatrixAndLayout.first);
    }

    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t inputIndex) override
    {
        assert(inputIndex == 0);

        ElemType gapPadValue = 0;
        auto gradient = ComputationNode<ElemType>::Unpack(GetSampleLayout(), Gradient(), m_pMBLayout, m_tempUnpackedData, m_tempScatterIndices, std::shared_ptr<Matrix<char>>(nullptr), /*batchMajor=*/ false, &gapPadValue);
        auto inputGradient = InputRef(inputIndex).GradientTensorFor(InputRef(inputIndex).GetSampleLayout().GetRank(), FrameRange(InputRef(inputIndex).GetMBLayout()));

        if (InputRef(inputIndex).ParentOverwritesGradient())
            inputGradient.AssignCopyOf(gradient);
        else
            inputGradient.AddCopyOf(gradient);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override { return false; }

    virtual void Validate(bool isFinalValidationPass) override
    {
        ComputationNodeBase::Validate(isFinalValidationPass);
        ComputationNodeBase::m_isValueSparse = Input(0)->IsValueSparse();

        if (!m_pMBLayout)
            m_pMBLayout = DetermineMBLayout();

        TensorShape inputShape = Input(0)->GetSampleLayout();
        if (isFinalValidationPass)
        {
            // we generate its own MBLayout
            if (!Input(0)->HasMBLayout())
                InvalidArgument("%ls %ls operation can only operate on minibatch data (which have a layout).", NodeName().c_str(), OperationName().c_str());

            if (inputShape.GetRank() <= 1)
                InvalidArgument("%ls %ls operation can only operate on input data of rank >= 2.", NodeName().c_str(), OperationName().c_str());
        }

        SmallVector<size_t> outDims = inputShape.GetDims();
        outDims.resize(inputShape.GetRank() - 1);
        SetDims(TensorShape(outDims), true);
    }

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_tempGatherIndices, matrixPool, 1, HasMBLayout());
    }

    void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterForwardProp(matrixPool);
        ReleaseMatrixToPool(m_tempGatherIndices, matrixPool);
    }

    void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_tempScatterIndices, matrixPool, 1, HasMBLayout());
        RequestMatrixFromPool(m_tempUnpackedData, matrixPool, InputRef(0).GetSampleLayout().GetNumElements(), InputRef(0).HasMBLayout());
    }

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_tempScatterIndices, matrixPool);
        ReleaseMatrixToPool(m_tempUnpackedData, matrixPool);
    }

private:
    shared_ptr<Matrix<ElemType>> m_tempGatherIndices;
    shared_ptr<Matrix<ElemType>> m_tempScatterIndices;
    shared_ptr<Matrix<ElemType>> m_tempUnpackedData;
};

// -----------------------------------------------------------------------
// ToSequenceNode(sourceData, sequenceLengths)
// Converts the sourceData to a sequence using the most significant static axis [-1] as the sequence axis.
// The sequenceLengths input is optional; if unspecified, all sequences are assumed to be of the same length
// i.e. the dimensionality of the most significant static axis
// -----------------------------------------------------------------------

template <class ElemType>
class ToSequenceNode : public ToSequenceNodeBase<ElemType>
{
    typedef ToSequenceNodeBase<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"ToSequence"; }

    static const std::wstring DefaultToSequenceNodeDynamicAxisName() { return L"ToSequenceNodeAxis"; }
public:
    DeclareConstructorFromConfig(ToSequenceNode);
    ToSequenceNode(DEVICEID_TYPE deviceId, const wstring& name, const wstring& dynamicAxisName = DefaultToSequenceNodeDynamicAxisName())
        : Base(deviceId, name), m_dynamicAxisName(dynamicAxisName)
    {}

    virtual std::vector<size_t> GetSequenceLengths() override
    {
        let& inMBLayout = InputRef(0).GetMBLayout();
        auto numSequences = inMBLayout->GetNumSequences();
        auto inputSampleLayout = InputRef(0).GetSampleLayout();
        std::vector<size_t> sequenceLengths(numSequences, inputSampleLayout[inputSampleLayout.GetRank() - 1]);
        if (GetNumInputs() > 1)
        {
            if (*Input(0)->GetMBLayout() != *Input(1)->GetMBLayout())
                LogicError("%ls %ls operation requires both its inputs to have identical minibatch layouts.", NodeName().c_str(), OperationName().c_str());

            let& sequenceLengthsInputValue = InputRef(1).Value();
            for (size_t i = 0; i < numSequences; ++i)
                sequenceLengths[i] = (size_t)sequenceLengthsInputValue(0, i);
        }

        return sequenceLengths;
    }

    virtual MBLayoutPtr DetermineMBLayout() override
    {
        assert(!m_pMBLayout);

        // this generates a new layout
        auto newLayout = make_shared<MBLayout>();
        newLayout->SetUniqueAxisName(m_dynamicAxisName);
        return newLayout;
    }

    virtual void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);

        if ((GetNumInputs() > 1) && (Input(1)->NeedsGradient()))
            InvalidArgument("%ls %ls operation does not support gradient propgation to its Input(1) denoting sequence lengths.", NodeName().c_str(), OperationName().c_str());

        if (isFinalValidationPass)
        {
            if ((GetNumInputs() > 1) && (Input(1)->GetSampleLayout().GetNumElements() != 1))
                InvalidArgument("%ls %ls operation's Input(1) denotes sequence lengths and must be a scalar.", NodeName().c_str(), OperationName().c_str());
        }
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_dynamicAxisName;
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_dynamicAxisName;
    }

    std::wstring DynamicAxisName() const { return m_dynamicAxisName; }

private:
    std::wstring m_dynamicAxisName;
};

template class ToSequenceNode<float>;
template class ToSequenceNode<double>;

// -----------------------------------------------------------------------
// ToSequenceLikeNode(sourceData, dynamicAxesLike)
// Converts the sourceData to a sequence using the most significant static axis [-1] as the sequence axis.
// The 'dynamicAxesLike' operand is used to obtain the lengths of the generated sequences. 
// The dynamic axes of the generated sequence are required to match the dynamic axes of the 'dynamicAxesLike' operand.
// -----------------------------------------------------------------------

template <class ElemType>
class ToSequenceLikeNode : public ToSequenceNodeBase<ElemType>
{
    typedef ToSequenceNodeBase<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"ToSequenceLike"; }

public:
    DeclareConstructorFromConfig(ToSequenceLikeNode);
    ToSequenceLikeNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {}

    virtual std::vector<size_t> GetSequenceLengths() override
    {
        let& inMBLayout = InputRef(0).GetMBLayout();
        auto sequencesMBLayout = InputRef(1).GetMBLayout();
        auto inputSampleLayout = InputRef(0).GetSampleLayout();
        auto sequenceMBLayoutNumTimeSteps = sequencesMBLayout->GetNumTimeSteps();
        auto inputSampleLayoutTrailingDimension = inputSampleLayout[inputSampleLayout.GetRank() - 1];
        if (sequenceMBLayoutNumTimeSteps != inputSampleLayoutTrailingDimension)
            InvalidArgument("%ls %ls operation's Input(0) trailing dimension (%zu) must match the number of timesteps (%zu) in Input(1)'s MBLayout.",
                            NodeName().c_str(), OperationName().c_str(), inputSampleLayoutTrailingDimension, sequenceMBLayoutNumTimeSteps);

        if (inMBLayout->GetNumSequences() != sequencesMBLayout->GetNumSequences())
            InvalidArgument("%ls %ls operation's Input(0) MBLayout must have the same number of sequences as Input(1)'s MBLayout.", NodeName().c_str(), OperationName().c_str());

        auto numSequences = inMBLayout->GetNumSequences();
        let& sequences = sequencesMBLayout->GetAllSequences();
        std::vector<size_t> sequenceLengths(numSequences, inputSampleLayoutTrailingDimension);
        for (size_t i = 0; i < sequences.size(); i++)
        {
            let& seq = sequences[i];
            if (seq.seqId == GAP_SEQUENCE_ID)
                continue;

            sequenceLengths[seq.seqId] = seq.GetNumTimeSteps();
        }

        return sequenceLengths;
    }

    virtual MBLayoutPtr DetermineMBLayout() override
    {
        assert(!m_pMBLayout);
        return InputRef(1).GetMBLayout();
    }

    virtual void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);

        if (isFinalValidationPass && !InputRef(1).HasMBLayout())
            RuntimeError("%ls %ls operation requires Input(1) must have a dynamic axis.", NodeName().c_str(), OperationName().c_str());
    }
};

template class ToSequenceLikeNode<float>;
template class ToSequenceLikeNode<double>;

// ------------------------------------------------------------------------------------------------
// UnpackSequenceNode(sequenceData)
// Converts the sequenceData to non-sequence data by unpacking the sequence along the 
// the most significant static axis [-1] and padding any gaps with the specified padding value.
// The node has 2 outputs; viz. the unpacked non-sequence data and a mask denoting the gaps in the 
// unpacked output due to differences across lengths of the sequences in 'sequenceData'
// ------------------------------------------------------------------------------------------------

template <class ElemType>
class UnpackSequenceNode : public ComputationNodeNonLooping<ElemType>, public MultiOutputNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"UnpackSequence"; }

public:
    DeclareConstructorFromConfig(UnpackSequenceNode);
    UnpackSequenceNode(DEVICEID_TYPE deviceId, const wstring& name, ElemType paddingValue = 0, bool suppressMaskOutput = false)
        : Base(deviceId, name), MultiOutputNode<ElemType>(suppressMaskOutput ? 1 : 2), m_paddingValue(paddingValue), m_suppressMaskOutput(suppressMaskOutput)
    {}

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        auto inputMBLayout = InputRef(0).GetMBLayout();
        if (inputMBLayout->HasSequenceBeyondBegin() || inputMBLayout->HasSequenceBeyondEnd())
            LogicError("%ls: %s node cannot perform sequence axis reduction for truncated sequence.", Base::NodeDescription().c_str(), typeid(*this).name());

        GetMBLayout()->InitAsFrameMode(inputMBLayout->GetNumSequences());
        UpdateFunctionValuesSize();

#ifdef _MSC_VER
        auto& outputValuePtrRef = ValuePtrRef();
#else
        auto& outputValuePtrRef = this->template ValuePtrRef();
#endif

        // Directly unpack into Value() matrix
        auto unpackedInput = ComputationNode<ElemType>::Unpack(InputRef(0).GetSampleLayout(), InputRef(0).Value(), InputRef(0).GetMBLayout(), outputValuePtrRef, m_tempScatterIndices, m_tempMask, /*batchMajor=*/ false, &m_paddingValue);
        if (unpackedInput.GetSOBPtr() != outputValuePtrRef)
            Value().AssignValuesOf(*unpackedInput.GetSOBPtr());

        if (!m_suppressMaskOutput)
        {
            auto numSequences = GetMBLayout()->GetNumSequences();
            size_t maxSequenceLength = inputMBLayout->GetNumTimeSteps();
            std::vector<ElemType> maskValues(maxSequenceLength * numSequences, 0);
            let& inputSequences = inputMBLayout->GetAllSequences();
            size_t j = 0;
            for (size_t i = 0; i < inputSequences.size(); i++)
            {
                let& seq = inputSequences[i];
                if (seq.seqId == GAP_SEQUENCE_ID)
                    continue;

                auto currentSequenceLength = seq.GetNumTimeSteps();
                auto rangeStart = maskValues.begin() + (j * maxSequenceLength);
                std::fill(rangeStart, rangeStart + currentSequenceLength, (ElemType)1);
                j++;
            }
            assert(j == numSequences);

            this->m_outputsValue[1]->SetValue(maxSequenceLength, numSequences, outputValuePtrRef->GetDeviceId(), maskValues.data());
        }
    }

    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t inputIndex) override
    {

        auto numSequences = GetMBLayout()->GetNumSequences();
        auto gradientSampleLayout = GetSampleLayout();
        auto gradientDataTensorShape = gradientSampleLayout;
        gradientDataTensorShape.AppendInPlace(gradientDataTensorShape.GetRank(), numSequences);
        let& gradientDataMatrix = Gradient();
        auto gradientDataNDArrayView = ::CNTK::MakeSharedObject<::CNTK::NDArrayView>(::CNTK::AsDataType<ElemType>(),
            ::CNTK::AsDeviceDescriptor(gradientDataMatrix.GetDeviceId()),
            ::CNTK::AsStorageFormat(gradientDataMatrix.GetFormat()),
            ::CNTK::AsNDShape(gradientDataTensorShape),
            /*readOnly =*/ true,
            new TensorView<ElemType>(GradientPtr(), gradientDataTensorShape));

        std::vector<size_t> sequenceLengths(numSequences);
        let& inMBLayout = InputRef(0).GetMBLayout();
        let& inputSequences = inMBLayout->GetAllSequences();
        size_t j = 0;
        for (size_t i = 0; i < inputSequences.size(); i++)
        {
            let& seq = inputSequences[i];
            if (seq.seqId == GAP_SEQUENCE_ID)
                continue;

            sequenceLengths[j] = seq.GetNumTimeSteps();
            j++;
        }
        assert(j == numSequences);

        auto gradientDataValue = ::CNTK::MakeSharedObject<::CNTK::Value>(gradientDataNDArrayView, ::CNTK::CreateMask(sequenceLengths));
        auto dummyVar = ::CNTK::InputVariable(::CNTK::AsNDShape(InputRef(0).GetSampleLayout()), gradientDataNDArrayView->IsSparse(), ::CNTK::AsDataType<ElemType>());
        auto packedGradientMatrixAndLayout = ::CNTK::Utils::GetCNTKImplMatrixAndMBLayoutFromValueObject(dummyVar, gradientDataValue, nullptr, m_tempPackedGradientData, m_tempGatherIndices);

        if (*packedGradientMatrixAndLayout.second != *inMBLayout)
            LogicError("%ls: %s node unpacked gradient MBLayout does not match input MBLayout.", Base::NodeDescription().c_str(), typeid(*this).name());

        InputRef(0).Gradient() += (*packedGradientMatrixAndLayout.first);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t childIndex) const override { return false; }

    virtual bool NeedsDynamicValidation() const override { return true; }

    virtual void Validate(bool isFinalValidationPass) override
    {
        ComputationNodeBase::Validate(isFinalValidationPass);
        ComputationNodeBase::m_isValueSparse = Input(0)->IsValueSparse();

        if (!m_pMBLayout)
        {
            m_pMBLayout = make_shared<MBLayout>(); // this generates a new layout
            m_pMBLayout->SetUniqueAxisName(ComputationNodeBase::DefaultNoSequenceAxisName);

            this->m_outputsMBLayout[0] = m_pMBLayout;
            if (!m_suppressMaskOutput)
                this->m_outputsMBLayout[1] = m_pMBLayout;
        }

        TensorShape inputShape = Input(0)->GetSampleLayout();
        SmallVector<size_t> outDims = inputShape.GetDims();
        outDims.resize(inputShape.GetRank() + 1, 1);
        SmallVector<size_t> maskDims = {1};
        if (isFinalValidationPass)
        {
            // we generate its own MBLayout
            auto inputMBLayout = InputRef(0).GetMBLayout();
            if (!inputMBLayout)
                InvalidArgument("%ls %ls operation can only operate on minibatch data (which have a layout).", NodeName().c_str(), OperationName().c_str());

            if (inputMBLayout->GetNumTimeSteps() == 0)
                LogicError("%ls %ls operation's final validation pass must not be invoked before the input MBLayout has been initialized and populated.", NodeName().c_str(), OperationName().c_str());

            outDims[inputShape.GetRank()] = inputMBLayout->GetNumTimeSteps();
            maskDims[0] = inputMBLayout->GetNumTimeSteps();
        }

        this->m_outputsShape[0] = TensorShape(outDims);
        if (!m_suppressMaskOutput)
            this->m_outputsShape[1] = TensorShape(maskDims);

        SetDims(TensorShape(outDims), true);
    }

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        m_tempMask = std::make_shared<Matrix<char>>(Base::m_deviceId);
        RequestMatrixFromPool(m_tempScatterIndices, matrixPool, 1, HasMBLayout());
    }

    void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterForwardProp(matrixPool);
        ReleaseMatrixToPool(m_tempScatterIndices, matrixPool);
    }

    void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_tempGatherIndices, matrixPool, 1, HasMBLayout());
        RequestMatrixFromPool(m_tempPackedGradientData, matrixPool, InputRef(0).GetSampleLayout().GetNumElements(), InputRef(0).HasMBLayout());
    }

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_tempGatherIndices, matrixPool);
        ReleaseMatrixToPool(m_tempPackedGradientData, matrixPool);
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_paddingValue;
        fstream >> m_suppressMaskOutput;
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_paddingValue;
        fstream << m_suppressMaskOutput;
    }

    ElemType PaddingValue() const { return m_paddingValue; }
    bool SuppressOutputMask() const { return m_suppressMaskOutput; }

private:
    ElemType m_paddingValue;
    bool m_suppressMaskOutput;

    shared_ptr<Matrix<char>> m_tempMask;
    shared_ptr<Matrix<ElemType>> m_tempScatterIndices;
    shared_ptr<Matrix<ElemType>> m_tempGatherIndices;
    shared_ptr<Matrix<ElemType>> m_tempPackedGradientData;
};

template class UnpackSequenceNode<float>;
template class UnpackSequenceNode<double>;

}}}
