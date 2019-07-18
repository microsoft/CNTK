//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <algorithm>
#include <boost/noncopyable.hpp>
#include <memory>
#include <vector>
#include <functional>
#include <codecvt>
#include <locale>

#include "CNTKLibrary.h"
#include "CNTKLibraryC.h"

namespace CNTK
{
    // Helper functions.
    inline ParameterCloningMethod ToNative(CNTK_ParameterCloningMethod method)
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

    inline DeviceDescriptor GetDeviceDescriptor(const CNTK_DeviceDescriptor* device)
    {
        if (!device || device->kind == CNTK_DeviceKind_CPU)
            return DeviceDescriptor::CPUDevice();
        if (device->kind == CNTK_DeviceKind_GPU)
            return DeviceDescriptor::GPUDevice(device->id);
        else
            RuntimeError("Invalid device kind. Currently only GPU and CPU devices are supported.");
    }

    inline NDShape ToNDShape(const CNTK_Shape& shape)
    {
        std::vector<size_t> dimensions;
        dimensions.reserve(shape.size);
        for (size_t i = 0; i < shape.size; ++i)
            dimensions.push_back(shape.value[i]);
        return NDShape(dimensions);
    }

    inline CNTK_Shape FromNDShape(const NDShape& shape) noexcept
    {
        CNTK_Shape result;
        result.size = (uint32_t)shape.Rank();
        result.value = new uint32_t[result.size];
        for (size_t i = 0; i < shape.Dimensions().size(); i++)
            result.value[i] = (uint32_t)shape.Dimensions()[i];
        return result;
    }

    inline void CleanAndDestroyVariables(CNTK_Variable* array, size_t length)
    {
        for (size_t i = 0; i < length; i++)
            CNTK_CleanVariable(&array[i]);
        delete[] array;
    }

    inline void CleanAndDestroyValues(CNTK_Value* array, size_t length)
    {
        for (size_t i = 0; i < length; i++)
            CNTK_CleanValue(&array[i]);
        delete[] array;
    }

    template<class T1, class T2, class T3>
    struct local_codecvt : std::codecvt<T1, T2, T3> {
        ~local_codecvt() { }
    };
    typedef local_codecvt<wchar_t, char, std::mbstate_t> cntk_codecvt;

    inline std::wstring StringToWString(const std::string &s)
    {
       return std::wstring_convert<cntk_codecvt>().from_bytes(s);
    }

    inline std::string WStringToString(const std::wstring &ws)
    {
       return std::wstring_convert<cntk_codecvt>().to_bytes(ws);
    }

    // Evaluator interface
    class EvaluatorWrapper : boost::noncopyable
    {
    public:
        virtual void GetModelArgumentsInfo(CNTK_Variable** inputs, uint32_t* numInputs) = 0;
        virtual void GetModelOutputsInfo(CNTK_Variable** outputs, uint32_t* numOutputs) = 0;

        virtual std::unique_ptr<EvaluatorWrapper> Clone(CNTK_ParameterCloningMethod method, bool flatten) = 0;
        virtual void EvaluateSequence(
            const CNTK_Variable* inputs,
            const CNTK_Value* inputValues,
            const bool* inputResetFlags,
            uint32_t numInputs,
            const CNTK_Variable* outputs,
            uint32_t numOutputs,
            CNTK_Value** outputValues) = 0;
        virtual ~EvaluatorWrapper() {}

    protected:
        void GetVariableInfo(const std::vector<Variable>& vars, CNTK_Variable** resultVars, uint32_t* numResultVars)
        {
            if (numResultVars == nullptr)
                InvalidArgument("numResultVars is not allowed to be null");

            auto arrayVarCleaner = std::bind(CleanAndDestroyVariables, std::placeholders::_1, vars.size());
            std::unique_ptr<CNTK_Variable, decltype(arrayVarCleaner)> result(new CNTK_Variable[vars.size()], arrayVarCleaner);
            memset(result.get(), 0, sizeof(CNTK_Variable) * vars.size());

            for (size_t i = 0; i < vars.size(); i++)
            {
                // Making sure with cleaners we do not leak anything on exception.
                CNTK_Variable resultVar{ 0 ,{ 0, 0 } };
                std::unique_ptr<CNTK_Variable, decltype(&CNTK_CleanVariable)> varCleaner(&resultVar, CNTK_CleanVariable);

                const auto& var = vars[i];
                resultVar.name = new char[var.Name().size() + 1];
                std::string name = WStringToString(var.Name());
                std::copy(name.c_str(), name.c_str() + name.size(), resultVar.name);
                resultVar.name[name.size()] = 0;
                resultVar.shape = FromNDShape(var.Shape());
                result.get()[i] = resultVar;

                varCleaner.release();
            }

            *numResultVars = (uint32_t)vars.size();
            *resultVars = result.release();
        }
    };

    //
    // A wrapper for evaluation functionality of the library exposed in C interface.
    //
    class CNTKEvaluatorWrapper : public EvaluatorWrapper
    {
    public:
        CNTKEvaluatorWrapper(const char* modelFilePath, const CNTK_DeviceDescriptor* device);
        CNTKEvaluatorWrapper(const char* modelFilePath, DeviceDescriptor device);
        CNTKEvaluatorWrapper(const void* modelData, int modelDataLen, const CNTK_DeviceDescriptor* device);
        CNTKEvaluatorWrapper(const void* modelData, int modelDataLen, DeviceDescriptor device);
        CNTKEvaluatorWrapper(FunctionPtr model, DeviceDescriptor device);

        void GetModelArgumentsInfo(CNTK_Variable** inputs, uint32_t* numInputs) override;
        void GetModelOutputsInfo(CNTK_Variable** outputs, uint32_t* numOutputs) override;

        std::unique_ptr<EvaluatorWrapper> Clone(CNTK_ParameterCloningMethod method, bool flatten) override;

        void EvaluateSequence(
            const CNTK_Variable* inputs,
            const CNTK_Value* inputValues,
            const bool* inputResetFlags,
            uint32_t numInputs,
            const CNTK_Variable* outputs,
            uint32_t numOutputs,
            CNTK_Value** outputValues) override;

    private:
        FunctionPtr m_func;
        DeviceDescriptor m_device;
        std::unordered_map<std::string, Variable> m_arguments;
        std::unordered_map<std::string, Variable> m_outputs;
    };
}

//#pragma warning(pop)
