//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <boost/noncopyable.hpp>
#include <memory>

#include "CNTKLibrary.h"
#include "CNTKLibraryC.h"

namespace CNTK
{
    //
    // A wrapper for evaluation functionality of the library exposed in C interface.
    //
    class EvaluatorWrapper : boost::noncopyable
    {
    public:
        EvaluatorWrapper(const wchar_t* modelFilePath, const wchar_t* device);
        EvaluatorWrapper(const wchar_t* modelFilePath, DeviceDescriptor device);
        EvaluatorWrapper(FunctionPtr model, DeviceDescriptor device);

        void GetModelArgumentsInfo(CNTK_Variable** inputs, uint32_t* numInputs);
        void GetModelOutputsInfo(CNTK_Variable** outputs, uint32_t* numOutputs);

        std::unique_ptr<EvaluatorWrapper> Clone(CNTK_ParameterCloningMethod method, bool flatten);

        void EvaluateSequence(
            const CNTK_Variable* inputs,
            const CNTK_Value* inputValues,
            const bool* inputResetFlags,
            uint32_t numInputs,
            const CNTK_Variable* outputs,
            uint32_t numOutputs,
            CNTK_Value** outputValues);

    private:
        // Auxiliary functinos.
        void GetVariableInfo(const std::vector<Variable>& vars, CNTK_Variable** outputs, uint32_t* numOutputs);

        FunctionPtr m_func;
        DeviceDescriptor m_device;
        std::unordered_map<std::wstring, Variable> m_arguments;
        std::unordered_map<std::wstring, Variable> m_outputs;
    };
}
