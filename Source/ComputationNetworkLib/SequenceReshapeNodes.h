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
// ToSequenceNode(sourceData, sequenceLengths)
// Converts the sourceData to a sequence using the most significant static axis [-1] as the sequence axis.
// The sequenceLengths input is optional; if unspecified, all sequences are assumed to be of the same length
// i.e. the dimensionality of the most significant static axis
// -----------------------------------------------------------------------

template <class ElemType>
class ToSequenceNode : public ComputationNodeNonLooping<ElemType>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"ToSequence"; }

    static const std::wstring DefaultToSequenceNodeDynamicAxisName() { return L"ToSequenceNodeAxis"; }
public:
    DeclareConstructorFromConfig(ToSequenceNode);
    ToSequenceNode(DEVICEID_TYPE deviceId, const wstring& name, const wstring& dynamicAxisName = DefaultToSequenceNodeDynamicAxisName())
        : Base(deviceId, name), m_dynamicAxisName(dynamicAxisName)
    {}

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        let& inMBLayout = InputRef(0).GetMBLayout();
        if (inMBLayout->GetNumTimeSteps() != 1)
            InvalidArgument("%ls %ls operation can only operate on non-sequence input data.", NodeName().c_str(), OperationName().c_str());

        if ((GetNumInputs() > 1) && (*Input(0)->GetMBLayout() != *Input(1)->GetMBLayout()))
            LogicError("%ls %ls operation requires both its inputs to have identical minibatch layouts.", NodeName().c_str(), OperationName().c_str());

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

        std::vector<size_t> sequenceLengths(numSequences, inputSampleLayout[inputSampleLayout.GetRank() - 1]);
        if (GetNumInputs() > 1)
        {
            let& sequenceLengthsInputValue = InputRef(1).Value();
            for (size_t i = 0; i < numSequences; ++i)
                sequenceLengths[i] = (size_t)sequenceLengthsInputValue(0, i);
        }

        auto inputDataValue = ::CNTK::MakeSharedObject<::CNTK::Value>(inputDataNDArrayView, ::CNTK::CreateMask(sequenceLengths));
        auto dummyVar = ::CNTK::InputVariable(::CNTK::AsNDShape(GetSampleLayout()), this->IsValueSparse(), ::CNTK::DataType::Float);
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

        auto gradient = ComputationNode<ElemType>::Unpack(GetSampleLayout(), Gradient(), m_pMBLayout, m_tempUnpackedData, m_tempScatterIndices, /*batchMajor=*/ false, /*maskGaps=*/ true);
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

        if ((GetNumInputs() > 1) && (Input(1)->NeedsGradient()))
            InvalidArgument("%ls %ls operation does not support gradient propgation to its Input(1) denoting sequence lengths.", NodeName().c_str(), OperationName().c_str());

        if (!m_pMBLayout)
        {
            m_pMBLayout = make_shared<MBLayout>(); // this generates a new layout
            m_pMBLayout->SetUniqueAxisName(m_dynamicAxisName);
        }

        TensorShape inputShape = Input(0)->GetSampleLayout();
        if (isFinalValidationPass)
        {
            // we generate its own MBLayout
            if (!Input(0)->HasMBLayout())
                InvalidArgument("%ls %ls operation can only operate on minibatch data (which have a layout).", NodeName().c_str(), OperationName().c_str());

            if (inputShape.GetRank() <= 1)
                InvalidArgument("%ls %ls operation can only operate on input data of rank >= 2.", NodeName().c_str(), OperationName().c_str());

            if ((GetNumInputs() > 1) && (Input(1)->GetSampleLayout().GetNumElements() != 1))
                InvalidArgument("%ls %ls operation's Input(1) denotes sequence lengths and must be a scalar.", NodeName().c_str(), OperationName().c_str());
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

    shared_ptr<Matrix<ElemType>> m_tempGatherIndices;
    shared_ptr<Matrix<ElemType>> m_tempScatterIndices;
    shared_ptr<Matrix<ElemType>> m_tempUnpackedData;
};

template class ToSequenceNode<float>;
template class ToSequenceNode<double>;

}}}
