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

    // Auxiliary functions
    static ParameterCloningMethod ToNative(CNTK_ParameterCloningMethod method)
    {
        switch (method)
        {
        case CNTK_ModelParameterShare:
            return ParameterCloningMethod::Share;
        case CNTK_ModelParameterClone:
            return ParameterCloningMethod::Clone;
        case CNTK_ModelParameterFreeze:
            return ParameterCloningMethod::Freeze;
        default:
            InvalidArgument("Cloning method is invalid");
            return ParameterCloningMethod::Invalid;
        }
    }

    static inline NDShape ToNDShape(const CNTK_Shape& shape)
    {
        vector<size_t> dimensions;
        dimensions.reserve(shape.size);
        for (size_t i = 0; i < shape.size; ++i)
            dimensions.push_back(shape.value[i]);
        return NDShape(dimensions);
    }

    static inline CNTK_Shape FromNDShape(const NDShape& shape) noexcept
    {
        CNTK_Shape result;
        result.size = (uint32_t)shape.Rank();
        result.value = new uint32_t[result.size];
        for (size_t i = 0; i < shape.Dimensions().size(); i++)
            result.value[i] = (uint32_t)shape.Dimensions()[i];
        return result;
    }

    DeviceDescriptor GetDeviceDescriptor(const wchar_t* device)
    {
        if (!device)
            InvalidArgument("Device is not allowed to be null.");
        if (wstring(L"cpu") != device)
            RuntimeError("Device '%ls' is not supported. Currently only CPU device is supported.", device);
        return DeviceDescriptor::CPUDevice();
    }

    // Main interface
    EvaluatorWrapper::EvaluatorWrapper(FunctionPtr model, DeviceDescriptor device)
        : m_func(model), m_device(device)
    {
        for (const auto arg : m_func->Arguments())
            m_arguments.insert(make_pair(arg.Name(), arg));

        for (const auto arg : m_func->Outputs())
            m_outputs.insert(make_pair(arg.Name(), arg));
    }

    EvaluatorWrapper::EvaluatorWrapper(const wchar_t* modelFilePath, DeviceDescriptor device) :
        EvaluatorWrapper::EvaluatorWrapper(Function::Load(modelFilePath, m_device), device)
    {}

    EvaluatorWrapper::EvaluatorWrapper(const wchar_t* modelFilePath, const wchar_t* device) :
        EvaluatorWrapper(modelFilePath, GetDeviceDescriptor(device))
    {}

    void EvaluatorWrapper::GetModelArgumentsInfo(CNTK_Variable** inputs, uint32_t* numInputs)
    {
        assert(inputs != nullptr);
        assert(numInputs != nullptr);
        return GetVariableInfo(m_func->Arguments(), inputs, numInputs);
    }

    void EvaluatorWrapper::GetModelOutputsInfo(CNTK_Variable** outputs, uint32_t* numOutputs)
    {
        assert(outputs != nullptr);
        assert(numOutputs != nullptr);
        return GetVariableInfo(m_func->Outputs(), outputs, numOutputs);
    }

    static void CleanAndDestroyVariables(CNTK_Variable* array, size_t length)
    {
        for (size_t i = 0; i < length; i++)
            CNTK_CleanVariable(&array[i]);
        delete[] array;
    }

    static void CleanAndDestroyValues(CNTK_Value* array, size_t length)
    {
        for (size_t i = 0; i < length; i++)
            CNTK_CleanValue(&array[i]);
        delete[] array;
    }

    void EvaluatorWrapper::GetVariableInfo(const vector<Variable>& vars, CNTK_Variable** resultVars, uint32_t* numResultVars)
    {
        assert(numResultVars != nullptr);
        auto arrayVarCleaner = std::bind(CleanAndDestroyVariables, _1, vars.size());
        unique_ptr<CNTK_Variable, decltype(arrayVarCleaner)> result(new CNTK_Variable[vars.size()], arrayVarCleaner);
        memset(result.get(), 0, sizeof(CNTK_Variable) * vars.size());

        for (size_t i = 0; i < vars.size(); i++)
        {
            // Making sure with cleaners we do not leak anything on exception.
            CNTK_Variable resultVar{ 0 ,{ 0, 0 } };
            unique_ptr<CNTK_Variable, decltype(&CNTK_CleanVariable)> varCleaner(&resultVar, CNTK_CleanVariable);

            const auto& var = vars[i];
            resultVar.name = new wchar_t[var.Name().size() + 1];
            copy(var.Name().c_str(), var.Name().c_str() + var.Name().size(), resultVar.name);
            resultVar.name[var.Name().size()] = 0;
            resultVar.shape = FromNDShape(var.Shape());
            result.get()[i] = resultVar;

            varCleaner.release();
        }

        *numResultVars = (uint32_t)vars.size();
        *resultVars = result.release();
    }

    void EvaluatorWrapper::EvaluateSequence(
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

    unique_ptr<EvaluatorWrapper> EvaluatorWrapper::Clone(CNTK_ParameterCloningMethod method, bool flatten)
    {
        FunctionPtr cloned;
        if (flatten)
            cloned = m_func->CloneFlattened(ToNative(method));
        else
            cloned = m_func->Clone(ToNative(method));
        return unique_ptr<EvaluatorWrapper>(new EvaluatorWrapper(cloned, m_device));
    }
}
