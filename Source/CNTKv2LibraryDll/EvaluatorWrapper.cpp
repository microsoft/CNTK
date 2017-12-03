//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _SCL_SECURE_NO_WARNINGS

#include "stdafx.h"
#include "EvaluatorWrapper.h"

namespace CNTK
{
    using namespace std;
    using namespace std::placeholders;

    // Main interface
    CNTKEvaluatorWrapper::CNTKEvaluatorWrapper(FunctionPtr model, DeviceDescriptor device)
        : m_func(model), m_device(device)
    {
        for (const auto arg : m_func->Arguments())
            m_arguments.insert(make_pair(arg.Name(), arg));

        for (const auto arg : m_func->Outputs())
            m_outputs.insert(make_pair(arg.Name(), arg));
    }

    CNTKEvaluatorWrapper::CNTKEvaluatorWrapper(const wchar_t* modelFilePath, DeviceDescriptor device) :
        CNTKEvaluatorWrapper(Function::Load(modelFilePath, device), device)
    {}

    CNTKEvaluatorWrapper::CNTKEvaluatorWrapper(const wchar_t* modelFilePath, const wchar_t* device) :
        CNTKEvaluatorWrapper(modelFilePath, GetDeviceDescriptor(device))
    {}

    void CNTKEvaluatorWrapper::GetModelArgumentsInfo(CNTK_Variable** inputs, uint32_t* numInputs)
    {
        assert(inputs != nullptr);
        assert(numInputs != nullptr);
        return GetVariableInfo(m_func->Arguments(), inputs, numInputs);
    }

    void CNTKEvaluatorWrapper::GetModelOutputsInfo(CNTK_Variable** outputs, uint32_t* numOutputs)
    {
        assert(outputs != nullptr);
        assert(numOutputs != nullptr);
        return GetVariableInfo(m_func->Outputs(), outputs, numOutputs);
    }

    void CNTKEvaluatorWrapper::EvaluateSequence(
        const CNTK_Variable* inputs,
        const CNTK_Value* inputValues,
        const bool* inputResetFlags,
        uint32_t numInputs,
        const CNTK_Variable* outputs,
        uint32_t numOutputs,
        CNTK_Value** outputValues)
    {
        // Prepare inputs.
        unordered_map<Variable, ValuePtr> preparedInputs;
        for (uint32_t i = 0; i < numInputs; ++i)
        {
            auto var = m_arguments.find(inputs[i].name);
            if (var == m_arguments.end())
                InvalidArgument("Unexpected argument.");

            auto inputValue = inputValues[i];
            auto inputShape = ToNDShape(inputValue.shape);
            auto inputSampleShape = inputShape.SubShape(0, inputShape.Rank() - 1);

            // Prepare the mask.
            NDMaskPtr mask = nullptr;
            if (!inputResetFlags[i])
            {
                mask = make_shared<NDMask>(inputShape);
                mask->MarkSequenceBegin({ 0 });
            }

            ValuePtr value = nullptr;
            NDArrayViewPtr data = nullptr;
            if (inputShape == var->second.Shape())
            {
                data = make_shared<NDArrayView>(DataType::Float, inputShape, inputValue.data, inputValue.dataSize * sizeof(float), m_device);
            }
            else if (inputSampleShape == var->second.Shape())
            {
                data = make_shared<NDArrayView>(DataType::Float, inputShape.AppendShape(NDShape{ 1 }), inputValue.data, inputValue.dataSize * sizeof(float), m_device);
            }
            else
                InvalidArgument("Unexpected dimensionality of the input '%ls'.", inputs[i].name);

            value = make_shared<Value>(data, mask);

            preparedInputs[var->second] = value;
        }

        // Prepare outputs.
        unordered_map<Variable, ValuePtr> preparedOutputs;
        for (uint32_t i = 0; i < numOutputs; ++i)
        {
            auto var = m_outputs.find(outputs[i].name);
            if (var == m_outputs.end())
                InvalidArgument("Unexpected output.");

            ValuePtr value = nullptr;
            if (*outputValues != nullptr) // Buffer has been preallocated, TODO: make sure the user did not mess up.
            {
                auto buffer = *outputValues[i];
                auto data = make_shared<NDArrayView>(DataType::Float, ToNDShape(buffer.shape), buffer.data, buffer.dataSize * sizeof(float), m_device);
                value = make_shared<Value>(data);
            }

            preparedOutputs[var->second] = value;
        }

        m_func->Evaluate(preparedInputs, preparedOutputs, m_device);

        if (preparedOutputs.size() != numOutputs)
            RuntimeError("Number of evaluated outputs '%d' does not match passed value '%d'.",
                (int)preparedOutputs.size(), (int)numOutputs);

        if (*outputValues != nullptr)
            return;

        // Copy to outputs if non were provided.
        auto arrayValueCleaner = std::bind(CleanAndDestroyValues, _1, preparedOutputs.size());
        unique_ptr<CNTK_Value, decltype(arrayValueCleaner)> result(new CNTK_Value[preparedOutputs.size()], arrayValueCleaner);
        memset(result.get(), 0, sizeof(CNTK_Value) * preparedOutputs.size());
        for (uint32_t i = 0; i < numOutputs; ++i)
        {
            auto var = m_outputs.find(outputs[i].name);
            assert(var != m_outputs.end());

            auto varToValue = preparedOutputs.find(var->second);
            if (varToValue == preparedOutputs.end())
                RuntimeError("Could not retrieve ouput for variable '%ls'", outputs[i].name);

            auto value = varToValue->second;

            {
                // Making sure with cleaners we do not leak anything on exception.
                CNTK_Value v{ {0, 0}, 0, 0 };
                unique_ptr<CNTK_Value, decltype(&CNTK_CleanValue)> valCleaner(&v, CNTK_CleanValue);

                auto size = value->Data()->Shape().TotalSize();
                v.dataSize = (uint32_t)size;
                v.data = new float[v.dataSize];
                copy(value->Data()->DataBuffer<float>(), value->Data()->DataBuffer<float>() + size, v.data);
                v.shape = FromNDShape(value->Shape());
                result.get()[i] = v;

                valCleaner.release();
            }
        }

        *outputValues = result.release();
    }

    unique_ptr<EvaluatorWrapper> CNTKEvaluatorWrapper::Clone(CNTK_ParameterCloningMethod method, bool flatten)
    {
        FunctionPtr cloned;
        if (flatten)
            cloned = m_func->CloneFlattened(ToNative(method));
        else
            cloned = m_func->Clone(ToNative(method));
        return unique_ptr<EvaluatorWrapper>(new CNTKEvaluatorWrapper(cloned, m_device));
    }
}
