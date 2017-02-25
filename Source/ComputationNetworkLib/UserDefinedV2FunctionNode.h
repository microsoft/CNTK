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
class SelectUserDefinedV2FunctionOutputNode;

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
    
    friend class SelectUserDefinedV2FunctionOutputNode<ElemType>;

public:
    UserDefinedV2FunctionNode(DEVICEID_TYPE deviceId, const wstring& name, const ::CNTK::FunctionPtr& externalFunction = nullptr)
        : Base(deviceId, name), m_externalFunction(externalFunction)
    {
        if (!m_externalFunction)
            LogicError("UserDefinedV2FunctionNode ctor should never be called with externalFunction == nullptr");

        m_numOutputs = m_externalFunction->Outputs().size();
        m_values.resize(m_numOutputs);
        m_gradients.resize(m_numOutputs);
        m_MBLayouts.resize(m_numOutputs);
        m_outputHasNewMBLayout.resize(m_numOutputs);
    }

    virtual void ForwardPropNonLooping() override
    {
        m_values[0] = m_value;

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
            auto outputMatrixAndLayout = ::CNTK::Utils::GetCNTKImplMatrixAndMBLayoutFromValueObject<ElemType>(output, outputValues[output]);
            m_values[i]->SetValue(*outputMatrixAndLayout.first);

            if ((m_MBLayouts[i] != nullptr) && (outputMatrixAndLayout.second == nullptr))
                LogicError("The UserDefinedFunction node has a non-null output MBLayout but none found from the (%S) user Function::Forward output Value", m_externalFunction->Name().c_str());
            else if ((m_MBLayouts[i] == nullptr) && (outputMatrixAndLayout.second != nullptr))
                LogicError("The UserDefinedFunction node does not have an output MBLayout but the (%S) user Function::Forward output Value have a non-null layout", m_externalFunction->Name().c_str());
            else if ((m_MBLayouts[i] == nullptr) && (outputMatrixAndLayout.second == nullptr))
                ;
            else
            {
                if (m_outputHasNewMBLayout[i])
                    m_MBLayouts[i]->CopyFrom(outputMatrixAndLayout.second);
                else
                {
                    if (*m_MBLayouts[i] != *outputMatrixAndLayout.second)
                        LogicError("The MBLayout of the output computed by the external function (%S) does not match the expected MBLayout", m_externalFunction->Name().c_str());
                }
            }
        }
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        m_gradients[0] = m_gradient;

        std::vector<::CNTK::Variable> externalFunctionUniqueInputs;
        auto externalFunctionInputs = m_externalFunction->Inputs();
        for (auto input : externalFunctionInputs)
        {
            if (std::find(externalFunctionUniqueInputs.begin(), externalFunctionUniqueInputs.end(), input) == externalFunctionUniqueInputs.end())
                externalFunctionUniqueInputs.push_back(input);
        }

        auto input = externalFunctionUniqueInputs[inputIndex];

        std::unordered_map<::CNTK::Variable, ::CNTK::ValuePtr> outputGradientValues;
        auto outputs = m_externalFunction->Outputs();
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto output = outputs[i];

            // TODO: We unpack the same output gradients each time this method is called for a different input.
            // We should be able to cache the unpacked values during backpropagation of gradients to the first
            // input, and reuse them for subsequence inputs.
            auto gradientValue = ::CNTK::Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout(output, *m_gradients[i], m_MBLayouts[i]);
            outputGradientValues.insert({ output, gradientValue });
        }

        std::unordered_map<::CNTK::Variable, ::CNTK::ValuePtr> inputGradientValue = { { input, nullptr } };
        m_externalFunction->Backward(m_currentBackpropStatePtr, outputGradientValues, inputGradientValue);

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

        auto outputs = m_externalFunction->Outputs();
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto output = outputs[i];

            if (output.GetDataType() != ::CNTK::AsDataType<ElemType>())
            {
                LogicError("The DataType (%s) of the external user defined Function's output does not match the internal ComputationNode's ElemType (%s)",
                    DataTypeName(output.GetDataType()),
                    DataTypeName(::CNTK::AsDataType<ElemType>()));
            }

            auto outputNDShape = output.Shape();
            if (outputNDShape.IsUnknown() || outputNDShape.HasInferredDimension())
                LogicError("The output shape of an external user defined Function should be fully determined by the time CNTK engine validation executes");

            auto outputDynamicAxes = output.DynamicAxes();
            if (outputDynamicAxes.empty())
            {
                m_outputHasNewMBLayout[i] = true;
                m_MBLayouts[i] = nullptr;
            }
            else
            {
                auto argumentVariables = m_externalFunction->Arguments();
                size_t j = 0;
                auto numInputs = GetNumInputs();
                for (size_t k = 0; k < numInputs; ++k)
                {
                    auto& input = InputRef(k);
                    if (input.template Is<LearnableParameter<ElemType>>())
                        continue;

                    auto argumentVar = argumentVariables[j];
                    if (argumentVar.DynamicAxes() == outputDynamicAxes)
                    {
                        m_MBLayouts[i] = input.GetMBLayout();
                        break;
                    }

                    j++;
                }

                if (!m_MBLayouts[i])
                {
                    m_MBLayouts[i] = make_shared<MBLayout>(); // this generates a new layout
                    m_MBLayouts[i]->SetUniqueAxisName(InternalDynamicAxisNameFromDynamicAxes(output.DynamicAxes()));
                    m_outputHasNewMBLayout[i] = true;
                }
                else
                    m_outputHasNewMBLayout[i] = false;
            }

            if (i == 0)
            {
                m_pMBLayout = m_MBLayouts[i];
                SetDims(::CNTK::AsTensorShape(outputNDShape), HasMBLayout());
            }
        }
    }

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        for (size_t i = 1 ; i < m_numOutputs; ++i)
            RequestMatrixFromPool(m_values[i], matrixPool);
    }

    void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        for (size_t i = 1; i < m_numOutputs; ++i)
            RequestMatrixFromPool(m_gradients[i], matrixPool);
    }

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        for (size_t i = 1; i < m_numOutputs; ++i)
            ReleaseMatrixToPool(m_values[i], matrixPool);
        for (size_t i = 1; i < m_numOutputs; ++i)
            ReleaseMatrixToPool(m_gradients[i], matrixPool);
    }

private:
    ::CNTK::FunctionPtr m_externalFunction;
    ::CNTK::BackPropStatePtr m_currentBackpropStatePtr;

    size_t m_numOutputs;
    std::vector<std::shared_ptr<Matrix<ElemType>>> m_values;
    std::vector<std::shared_ptr<Matrix<ElemType>>> m_gradients;
    std::vector<std::shared_ptr<MBLayout>> m_MBLayouts;
    std::vector<bool> m_outputHasNewMBLayout;
};

template class UserDefinedV2FunctionNode<float>;
template class UserDefinedV2FunctionNode<double>;

// -----------------------------------------------------------------------
// SelectUserDefinedV2FunctionOutputNode(userDefinedV2FunctionNode, outputIndex)
// ComputationNode for selecting one of the multiple outputs of UserDefinedV2FunctionNode
// This is needed since the CNTK computation engin natively does not support
// nodes with multiple outputs and hence, we need a separate node to multiplex 
// the additional outputs.
// -----------------------------------------------------------------------

// TODO: We currently only support external nodes that cannot be part of CNTK recurrent loops
template <class ElemType>
class SelectUserDefinedV2FunctionOutputNode final : public ComputationNodeNonLooping<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"SelectUserDefinedV2FunctionOutput"; }

public:
    SelectUserDefinedV2FunctionOutputNode(DEVICEID_TYPE deviceId, const wstring& name, size_t outputIndex = 0)
        : Base(deviceId, name), m_outputIndex(outputIndex)
    {}

    virtual void ForwardPropNonLooping() override
    {
        // TODO: We should avoid this copy but that requires carefully managing the 
        // lifetimes of the Value objects since to be able to directly use the 
        // input Value as its output, we have to make sure that the input's Value
        // is not reused until all dependents of this node are finished.
        auto inputNode = Input(0)->template As<UserDefinedV2FunctionNode<ElemType>>();
        Value().AssignValuesOf(*inputNode->m_values[m_outputIndex]);
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        // TODO: We should avoid this copy but that requires carefully managing the 
        // lifetimes of the Gradient objects since to be able to directly use the 
        // Gradient as input's gradient, we have to make sure that the Gradient
        // is not reused until all the inputs are finished backpropagating to their inputs.
        auto inputNode = Input(0)->template As<UserDefinedV2FunctionNode<ElemType>>();
        inputNode->m_gradients[m_outputIndex]->SetValue(Gradient());
    }

    virtual void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);

        auto inputNode = Input(0)->template As<UserDefinedV2FunctionNode<ElemType>>();
        m_pMBLayout = inputNode->m_MBLayouts[m_outputIndex];

        auto outputNDShape = inputNode->m_externalFunction->Outputs()[m_outputIndex].Shape();
        SetDims(::CNTK::AsTensorShape(outputNDShape), HasMBLayout());
    }

private:
    size_t m_outputIndex;
};

template class SelectUserDefinedV2FunctionOutputNode<float>;
template class SelectUserDefinedV2FunctionOutputNode<double>;

}}}
