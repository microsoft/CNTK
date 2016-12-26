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
// UserDefinedV2Function
// Proxy ComputationNode type for a V2 user-defined custom Function, instances
// of which can be part of a CNTK computation network.
// The actual implementation of the operation itself is external to the CNTK engine.
// -----------------------------------------------------------------------

// TODO: We currently only support external nodes that cannot be part of CNTK recurrent loops
template <class ElemType>
class UserDefinedV2FunctionNode final : public ComputationNodeNonLooping<ElemType>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"UserDefinedV2Function"; }

public:
    UserDefinedV2FunctionNode(DEVICEID_TYPE deviceId, const wstring& name, const ::CNTK::FunctionPtr& externalFunction = nullptr)
        : Base(deviceId, name), m_externalFunction(externalFunction)
    {
        if (!m_externalFunction)
            LogicError("UserDefinedV2FunctionNode ctor should never be called with externalFunction == nullptr");
    }

    virtual void ForwardPropNonLooping() override
    {
        // Get the arguments of the external function
        auto arguments = m_externalFunction->Arguments();
        std::unordered_map<::CNTK::Variable, ::CNTK::ValuePtr> argumentValues;
        auto numInputs = GetNumInputs();
        size_t j = 0;
        for (size_t i = 0; i < numInputs; ++i)
        {
            auto& input = InputRef(i);
            if (input.template Is<LearnableParameter<ElemType>>())
                continue;

            auto argumentVar = arguments[j++];
            auto argumentValue = ::CNTK::Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout(argumentVar, input.Value(), input.GetMBLayout());
            argumentValues.insert(std::make_pair(argumentVar, argumentValue));
        }
        assert(j == arguments.size());

        // TODO: Instead of passing null for output values, we should have the forward call directly produce the outputs in the output Value() of this node
        std::unordered_map<::CNTK::Variable, ::CNTK::ValuePtr> outputValue = { { m_externalFunction->Output(), nullptr } };
        std::unordered_set<::CNTK::Variable> outputsToRetainBackwardStateFor;
        if (Environment().IsTraining())
            outputsToRetainBackwardStateFor.insert(m_externalFunction->Output());

        auto computeDevice = ::CNTK::AsDeviceDescriptor(InputRef(0).Value().GetDeviceId());
        m_currentBackpropStatePtr = m_externalFunction->Forward(argumentValues, outputValue, computeDevice, outputsToRetainBackwardStateFor);

        // Copy the computed output to Value() of this node
        // TODO: We currently assume that the external Function does not generate a new MBLayout
        auto outputMatrixAndLayout = ::CNTK::Utils::GetCNTKImplMatrixAndMBLayoutFromValueObject<ElemType>(outputValue.begin()->first, outputValue.begin()->second);
        Value().AssignValuesOf(*outputMatrixAndLayout.first);

        if (*GetMBLayout() != *outputMatrixAndLayout.second)
            LogicError("The MBLayout of the output computed by the external function (%S) does not match the expected MBLayout", this->GetName().c_str());
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        auto input = m_externalFunction->Inputs()[inputIndex];

        auto gradientValue = ::CNTK::Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout(m_externalFunction->Output(), Gradient(), GetMBLayout());
        std::unordered_map<::CNTK::Variable, ::CNTK::ValuePtr> outputGradientValue = { { m_externalFunction->Output(), gradientValue } };
        std::unordered_map<::CNTK::Variable, ::CNTK::ValuePtr> inputGradientValue = { { input, nullptr } };
        m_externalFunction->Backward(m_currentBackpropStatePtr, outputGradientValue, inputGradientValue);

        // Accumulate the computed input gradient value into the existing input gradient value
        // TODO: We should directly pass the actual input gradient tensor to the Backward method 
        // instead of allocating a new value and accumulating it ourselves
        auto newInputGradientMatrixAndLayout = ::CNTK::Utils::GetCNTKImplMatrixAndMBLayoutFromValueObject<ElemType>(inputGradientValue.begin()->first, inputGradientValue.begin()->second);
        InputRef(inputIndex).Gradient() += *newInputGradientMatrixAndLayout.first;

        if (*InputRef(inputIndex).GetMBLayout() != *newInputGradientMatrixAndLayout.second)
            LogicError("The MBLayout of the input (%lu) gradient computed by the external function (%S) does not match the expected MBLayout", (unsigned long)inputIndex, this->GetName().c_str());
    }

    virtual void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);

        auto output = m_externalFunction->Output();

        // TODO: Add proper MBLayout inference/validation
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        // As we currently do not support external user defined Functions that generate new MBLayout,
        // let's verify that the dynamix axes of the external Function's output are consistent with the
        // input whose MBLayout we have linked to the output
        auto numInputs = GetNumInputs();
        auto outputMBLayout = GetMBLayout();
        for (size_t i = 0; i < numInputs; ++i)
        {
            auto& input = InputRef(i);
            if (input.GetMBLayout() == outputMBLayout)
            {
                if (m_externalFunction->Inputs()[i].DynamicAxes() != output.DynamicAxes())
                    LogicError("The dynamic axes of the external user defined Function's output do not match the dynamic axes of the input whose MBLayout has been selected for the Computation node's output");
            }
        }

        // The external Function can only have a single output
        auto numOutputs = m_externalFunction->Outputs().size();
        if (numOutputs != 1)
            InvalidArgument("Found user defined function (%S) with %lu outputs. User defined functions must have exactly one output", this->GetName().c_str(), (unsigned long)numOutputs);

        if (output.GetDataType() != ::CNTK::AsDataType<ElemType>())
        {
            LogicError("The DataType (%s) of the external user defined Function's output does not match the internal ComputationNode's ElemType (%s)",
                       DataTypeName(output.GetDataType()),
                       DataTypeName(::CNTK::AsDataType<ElemType>()));
        }

        auto outputNDShape = output.Shape();
        if (outputNDShape.IsUnknown() || outputNDShape.HasInferredDimension())
            LogicError("The output shape of an external user defined Function should be fully determined by the time CNTK engine validation executes");

        auto outputTensorShape = ::CNTK::AsTensorShape(outputNDShape);
        SetDims(outputTensorShape, HasMBLayout());
    }

private:
    ::CNTK::FunctionPtr m_externalFunction;
    ::CNTK::BackPropStatePtr m_currentBackpropStatePtr;
};

template class UserDefinedV2FunctionNode<float>;
template class UserDefinedV2FunctionNode<double>;

}}}
