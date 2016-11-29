//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalMultithreads.cpp : Sample application shows how to evaluate a model in multiple threading environment. 
//
#include "CNTKLibrary.h"

using namespace CNTK;
using namespace std;

void RetrieveInputBuffers(
    FunctionPtr evalFunc,
    unordered_map<wstring, vector<float>>& inputs)
{
    for (auto& input : evalFunc->Arguments())
    {
        // TODO: HERE is our INPUT VECTOR
        vector<float> inputData(input.Shape().TotalSize());
        inputs[input.Name()] = inputData;
    }
}

void ExecuteModel(
    FunctionPtr evalFunc,
    unordered_map<wstring, vector<float>>& inputs,
    unordered_map<wstring, vector<float>>& outputs)
{
    auto device = DeviceDescriptor::CPUDevice();

    // Prepare inputs
    unordered_map<Variable, ValuePtr> inputsVars;
    for (auto& input : evalFunc->Arguments())
    {
        // Todo: add convenience APIs to simplify data preparation here.
        ValuePtr value = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(input.Shape(), inputs[input.Name()], true /* isReadOnly */));

        fprintf(stderr, "  input: %S %S (#%lu elements)\n", input.Name().c_str(), value->Shape().AsString().c_str(), (unsigned long)inputs[input.Name()].size());
        inputsVars[input] = value;
    }

    // Prepare outputs.
    unordered_map<Variable, ValuePtr> outputsVars;
    for (auto output : evalFunc->Outputs())
    {
        outputsVars[output] = nullptr; // actual value will be filled by evaluating the model.
    }

    // Compute outputs by evaluating the model
    evalFunc->Forward(inputsVars, outputsVars, device);

    // Show results by printing the outputs
    for (auto& output : outputsVars)
    {
        ValuePtr value = output.second;

        // TODO: HERE is our OUTPUT VECTOR
        // TODO: add convenience APIs to simplify data retrieval here.
        vector<float> outputData(value->Data()->DataBuffer<float>(), value->Data()->DataBuffer<float>() + value->Data()->Shape().TotalSize());

        fprintf(stderr, "  output: %S %S (#%lu elements)\n", output.first.Name().c_str(), value->Shape().AsString().c_str(), (unsigned long)outputData.size());
        outputs[output.first.Name()] = outputData;
    }
}
