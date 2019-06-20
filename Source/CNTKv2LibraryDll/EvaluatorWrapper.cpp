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
            m_arguments.insert(make_pair(WStringToString(arg.Name()), arg));

        for (const auto arg : m_func->Outputs())
            m_outputs.insert(make_pair(WStringToString(arg.Name()), arg));
    }

    CNTKEvaluatorWrapper::CNTKEvaluatorWrapper(const char* modelFilePath, DeviceDescriptor device) :
        CNTKEvaluatorWrapper(Function::Load(StringToWString(modelFilePath), device), device)
    {}

    CNTKEvaluatorWrapper::CNTKEvaluatorWrapper(const char* modelFilePath, const CNTK_DeviceDescriptor* device) :
        CNTKEvaluatorWrapper(modelFilePath, GetDeviceDescriptor(device))
    {}

    CNTKEvaluatorWrapper::CNTKEvaluatorWrapper(const void* modelData, int modelDataLen, DeviceDescriptor device) :
        CNTKEvaluatorWrapper(Function::Load(static_cast<const char*>(modelData), modelDataLen, device), device)
    {}

    CNTKEvaluatorWrapper::CNTKEvaluatorWrapper(const void* modelData, int modelDataLen, const CNTK_DeviceDescriptor* device) :
        CNTKEvaluatorWrapper(modelData, modelDataLen, GetDeviceDescriptor(device))
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

            // TODO: Avoid copying.
            preparedInputs[var->second] =
                Value::CreateSequence(inputShape.SubShape(0, var->second.Shape().Rank()), std::vector<float>(inputValue.data, inputValue.data + inputShape.TotalSize()), inputResetFlags[i], m_device);
        }

        // Prepare outputs.
        unordered_map<Variable, ValuePtr> preparedOutputs;
        for (uint32_t i = 0; i < numOutputs; ++i)
        {
            auto var = m_outputs.find(outputs[i].name);
            if (var == m_outputs.end())
                InvalidArgument("Unexpected output.");

            ValuePtr value = nullptr;
            if (*outputValues != nullptr) // Buffer has been preallocated.
            {
                auto buffer = *outputValues[i];
                auto shape = ToNDShape(buffer.shape);
                NDShape maskShape = shape.SubShape(var->second.Shape().Rank(), shape.Rank());
                auto data = make_shared<NDArrayView>(DataType::Float, shape, buffer.data, shape.TotalSize() * sizeof(float), DeviceDescriptor::CPUDevice());
                value = make_shared<Value>(data, make_shared<NDMask>(maskShape));
            }
            preparedOutputs[var->second] = value;
        }

        m_func->Evaluate(preparedInputs, preparedOutputs, m_device);

        if (preparedOutputs.size() != numOutputs)
            RuntimeError("Number of evaluated outputs '%d' does not match passed value '%d'.",
                (int)preparedOutputs.size(), (int)numOutputs);

        if (*outputValues != nullptr)
            return;

        // Copy to outputs if none was provided.
        auto arrayValueCleaner = std::bind(CleanAndDestroyValues, _1, preparedOutputs.size());
        unique_ptr<CNTK_Value, decltype(arrayValueCleaner)> result(new CNTK_Value[preparedOutputs.size()], arrayValueCleaner);
        memset(result.get(), 0, sizeof(CNTK_Value) * preparedOutputs.size());
        for (uint32_t i = 0; i < numOutputs; ++i)
        {
            auto var = m_outputs.find(outputs[i].name);
            assert(var != m_outputs.end());

            auto varToValue = preparedOutputs.find(var->second);
            if (varToValue == preparedOutputs.end())
                RuntimeError("Could not retrieve ouput for variable '%s'", outputs[i].name);

            auto value = varToValue->second;

            {
                // Making sure with cleaners we do not leak anything on exception.
                CNTK_Value v{ {0, 0}, 0 };
                unique_ptr<CNTK_Value, decltype(&CNTK_CleanValue)> valCleaner(&v, CNTK_CleanValue);
                v.shape = FromNDShape(value->Shape());
                auto size = value->Shape().TotalSize();
                v.data = new float[size];
                auto data = value->Data();
                if (value->Device().Type() == DeviceKind::GPU)
                {
                    data = std::make_shared<NDArrayView>(DataType::Float, data->Shape(), DeviceDescriptor::CPUDevice());
                    data->CopyFrom(*(value->Data()));
                }
                std::copy(data->DataBuffer<float>(), data->DataBuffer<float>() + size, v.data);
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
