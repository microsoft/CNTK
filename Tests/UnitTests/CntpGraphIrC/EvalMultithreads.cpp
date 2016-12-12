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

void ExecuteModelOnGivenData(
    FunctionPtr evalFunc,
    unordered_map<wstring, vector<float>>& inputs,
    unordered_map<wstring, vector<float>>& outputs,
    const DeviceDescriptor& device)
{
    // Prepare inputs
    unordered_map<Variable, ValuePtr> inputsVars;
    for (auto& input : evalFunc->Arguments())
    {
        // Todo: add convenience APIs to simplify data preparation here.
        ValuePtr value = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(input.Shape(), inputs[input.Name()], true /* isReadOnly */));
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
        outputs[output.first.Name()] = outputData;
    }
}

void ExecuteModelOnRandomData(
    std::string filename,
    std::unordered_map<std::wstring, std::vector<float>>& inputs,
    std::unordered_map<std::wstring, std::vector<float>>& outputs,
    const CNTK::DeviceDescriptor& device)
{
    auto filenameW = std::wstring(filename.begin(), filename.end());
    auto modelFuncPtr = CNTK::Function::LoadModel(filenameW, device);

    if (inputs.size() == 0)
    {
        fprintf(stderr, "No input data given. Filling with random data.\n");

        RetrieveInputBuffers(modelFuncPtr, inputs);

        for (auto& inputTuple : inputs)
        {
            auto& inputData = inputTuple.second;

            // add some random data to the input vector
            for (size_t i = 0; i < inputData.size(); ++i)
            {
                inputData[i] = ((float)rand()) / RAND_MAX;
            }

            fprintf(stderr, "Input  %S #%lu elements.\n", inputTuple.first.c_str(), (unsigned long)inputTuple.second.size());
        }
    }

    //
    // Execute the original function
    //

    ExecuteModelOnGivenData(modelFuncPtr, inputs, outputs, device);
}
