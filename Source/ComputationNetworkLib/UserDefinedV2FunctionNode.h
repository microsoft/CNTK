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

template <typename ElemType>
class OutputMultiplexerNode;

// -----------------------------------------------------------------------
// UserDefinedV2Function
// Proxy ComputationNode type for a V2 user-defined custom Function, instances
// of which can be part of a CNTK computation network.
// The actual implementation of the operation itself is external to the CNTK engine.
// -----------------------------------------------------------------------

// TODO: We currently only support external nodes that cannot be part of CNTK recurrent loops
template <class ElemType>
class UserDefinedV2FunctionNode final : public ComputationNodeNonLooping<ElemType>, public MultiOutputNode<ElemType>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"UserDefinedV2Function"; }
    
    friend class OutputMultiplexerNode<ElemType>;

public:
    UserDefinedV2FunctionNode(DEVICEID_TYPE deviceId, const wstring& name, const ::CNTK::FunctionPtr& externalFunction = nullptr)
        : Base(deviceId, name), m_externalFunction(externalFunction), MultiOutputNode<ElemType>(externalFunction ? externalFunction->Outputs().size() : 0)
    {
        if (!m_externalFunction)
            LogicError("UserDefinedV2FunctionNode ctor should never be called with externalFunction == nullptr");
    }

    virtual bool ForceDynamicValidation() const override 
    {
        auto outputs = m_externalFunction->Outputs();
        return std::any_of(outputs.begin(), outputs.end(), [](const ::CNTK::Variable& output) { return output.Shape().HasFreeDimension(); });
    }

    virtual void ForwardPropNonLooping() override
    {
        this->m_outputsValue[0] = m_value;

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
            auto argumentShape = ::CNTK::AsNDShape(input.GetSampleLayout());
            auto argumentValue = ::CNTK::Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout(argumentShape, argumentVar.DynamicAxes(), input.Value(), input.GetMBLayout());
            argumentValues.insert(std::make_pair(argumentVar, argumentValue));
        }
        assert(j == arguments.size());

        auto outputs = m_externalFunction->Outputs();

        // TODO: Instead of passing null for output values, we should have the forward call directly produce the outputs in the output Value() of this node
        std::unordered_map<::CNTK::Variable, ::CNTK::ValuePtr> outputValues;
        for (auto output : outputs)
            outputValues.insert({output, nullptr});

        std::unordered_set<::CNTK::Variable> outputsToRetainBackwardStateFor;
        if (Environment().IsTraining())
            outputsToRetainBackwardStateFor.insert(outputs.begin(), outputs.end());

        auto computeDevice = ::CNTK::AsDeviceDescriptor(InputRef(0).Value().GetDeviceId());
        m_currentBackpropStatePtr = m_externalFunction->Forward(argumentValues, outputValues, computeDevice, outputsToRetainBackwardStateFor);

        // Copy the computed output
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto output = outputs[i];
            ::CNTK::NDShape inferredVarShape;
            auto outputMatrixAndLayout = ::CNTK::Utils::GetCNTKImplMatrixAndMBLayoutFromValueObject<ElemType>(output, outputValues[output], &inferredVarShape);

            if (inferredVarShape.IsUnknown() || inferredVarShape.HasUnboundDimension())
                LogicError("The output shape '%S' of an external user defined Function '%S' must be fully defined.", inferredVarShape.AsString().c_str(), m_externalFunction->AsString().c_str());

            if (output.Shape().HasFreeDimension())
            {
                this->m_outputsShape[i] = ::CNTK::AsTensorShape(inferredVarShape);
                if (i == 0)
                    SetDims(this->m_outputsShape[i], HasMBLayout());
            }

            this->m_outputsValue[i]->SetValue(*outputMatrixAndLayout.first);

            if ((this->m_outputsMBLayout[i] != nullptr) && (outputMatrixAndLayout.second == nullptr))
                LogicError("The UserDefinedFunction node has a non-null output MBLayout but none found from the '%S' user Function::Forward output Value", m_externalFunction->Name().c_str());
            else if ((this->m_outputsMBLayout[i] == nullptr) && (outputMatrixAndLayout.second != nullptr))
                LogicError("The UserDefinedFunction node does not have an output MBLayout but the '%S' user Function::Forward output Value has a non-null layout", m_externalFunction->Name().c_str());
            else if ((this->m_outputsMBLayout[i] == nullptr) && (outputMatrixAndLayout.second == nullptr))
                ;
            else
            {
                if (this->m_outputsHasNewMBLayout[i])
                    this->m_outputsMBLayout[i]->CopyFrom(outputMatrixAndLayout.second);
                else
                {
                    if (*this->m_outputsMBLayout[i] != *outputMatrixAndLayout.second)
                        LogicError("The MBLayout 'NumSequences=%zu, NumTimeSteps=%zu' of the output computed by the external function '%S' does not match the expected MBLayout 'NumSequences=%zu, NumTimeSteps=%zu'.",
                            outputMatrixAndLayout.second->GetNumSequences(), outputMatrixAndLayout.second->GetNumTimeSteps(),
                            m_externalFunction->Name().c_str(),
                            this->m_outputsMBLayout[i]->GetNumSequences(), this->m_outputsMBLayout[i]->GetNumTimeSteps());
                }
            }
        }
    }

    virtual void BackpropToNonLooping(size_t /*inputIndex*/) override
    {
        if (m_currentBackpropStatePtr == nullptr)
            return;

        this->m_outputsGradient[0] = m_gradient;

        std::unordered_map<::CNTK::Variable, ::CNTK::ValuePtr> outputGradientValues;
        auto outputs = m_externalFunction->Outputs();
        bool noOutputNeedsGradient = std::all_of(outputs.begin(), outputs.end(), [](const ::CNTK::Variable& outVar) { return !outVar.NeedsGradient(); });
        if (noOutputNeedsGradient)
            return;

        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto output = outputs[i];

            // TODO: We unpack the same output gradients each time this method is called for a different input.
            // We should be able to cache the unpacked values during backpropagation of gradients to the first
            // input, and reuse them for subsequence inputs.
            ::CNTK::ValuePtr gradientValue;
            if (output.NeedsGradient())
                gradientValue = ::CNTK::Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout(::CNTK::AsNDShape(this->m_outputsShape[i]), output.DynamicAxes(), *this->m_outputsGradient[i], this->m_outputsMBLayout[i]);

            outputGradientValues.insert({ output, gradientValue });
        }

        std::vector<::CNTK::Variable> externalFunctionUniqueInputs;
        auto externalFunctionInputs = m_externalFunction->Inputs();
        for (auto input : externalFunctionInputs)
        {
            if (std::find(externalFunctionUniqueInputs.begin(), externalFunctionUniqueInputs.end(), input) == externalFunctionUniqueInputs.end())
                externalFunctionUniqueInputs.push_back(input);
        }

        std::unordered_map<::CNTK::Variable, ::CNTK::ValuePtr> inputGradientValues;
        for (size_t i = 0; i < externalFunctionUniqueInputs.size(); ++i)
        {
            if (InputRef(i).NeedsGradient())
                inputGradientValues.insert({ externalFunctionUniqueInputs[i], nullptr });
        }

        m_externalFunction->Backward(m_currentBackpropStatePtr, outputGradientValues, inputGradientValues);

        // Accumulate the computed input gradient value into the existing input gradient value
        // TODO: We should directly pass the actual input gradient tensor to the Backward method 
        // instead of allocating a new value and accumulating it ourselves
        for (size_t i = 0; i < externalFunctionUniqueInputs.size(); ++i)
        {
            if (!InputRef(i).NeedsGradient())
                continue;

            InputRef(i).LazyZeroGradient(); // set gradient to 0 if this is the first time

            auto input = externalFunctionUniqueInputs[i];
            auto inputGradientValue = inputGradientValues[input];
            if (!inputGradientValue)
                continue;

            auto newInputGradientMatrixAndLayout = ::CNTK::Utils::GetCNTKImplMatrixAndMBLayoutFromValueObject<ElemType>(input, inputGradientValue);
            InputRef(i).Gradient() += *newInputGradientMatrixAndLayout.first;

            if (*InputRef(i).GetMBLayout() != *newInputGradientMatrixAndLayout.second)
                LogicError("The MBLayout 'NumSequences=%zu, NumTimeSteps=%zu' of the Input(%zu) gradient computed by the external function '%S' does not match the expected MBLayout 'NumSequences=%zu, NumTimeSteps=%zu'.",
                    newInputGradientMatrixAndLayout.second->GetNumSequences(), newInputGradientMatrixAndLayout.second->GetNumTimeSteps(),
                    i, this->GetName().c_str(),
                    InputRef(i).GetMBLayout()->GetNumSequences(), InputRef(i).GetMBLayout()->GetNumTimeSteps());
        }

        m_currentBackpropStatePtr = nullptr;
    }

    virtual void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);

        auto outputs = m_externalFunction->Outputs();
        bool layoutNotInitialized = (m_pMBLayout == nullptr);
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto output = outputs[i];

            if (output.GetDataType() != ::CNTK::AsDataType<ElemType>())
            {
                LogicError("The DataType '%s' of the external user defined Function's output does not match the internal ComputationNode's ElemType '%s'.",
                    DataTypeName(output.GetDataType()),
                    DataTypeName(::CNTK::AsDataType<ElemType>()));
            }

            auto outputNDShape = output.Shape();
            if (layoutNotInitialized)
            {
                auto outputDynamicAxes = output.DynamicAxes();
                if (outputDynamicAxes.empty())
                {
                    this->m_outputsHasNewMBLayout[i] = true;
                    this->m_outputsMBLayout[i] = nullptr;
                }
                else
                {
                    this->m_outputsMBLayout[i] = make_shared<MBLayout>(); // this generates a new layout
                    this->m_outputsMBLayout[i]->SetUniqueAxisName(InternalDynamicAxisNameFromDynamicAxes(output.DynamicAxes()));
                    this->m_outputsHasNewMBLayout[i] = true;
                }
            }

            for (size_t k = 0; k < outputNDShape.Rank(); ++k)
            {
                if ((outputNDShape[k] == ::CNTK::NDShape::FreeDimension) || (outputNDShape[k] == ::CNTK::NDShape::InferredDimension))
                    outputNDShape[k] = 1;
            }

            this->m_outputsShape[i] = ::CNTK::AsTensorShape(outputNDShape);

            if (i == 0)
            {
                if (layoutNotInitialized)
                    m_pMBLayout = this->m_outputsMBLayout[i];

                SetDims(this->m_outputsShape[i], HasMBLayout());
            }
        }
    }

private:
    ::CNTK::FunctionPtr m_externalFunction;
    ::CNTK::BackPropStatePtr m_currentBackpropStatePtr;
};

template class UserDefinedV2FunctionNode<float>;
template class UserDefinedV2FunctionNode<double>;

}}}
