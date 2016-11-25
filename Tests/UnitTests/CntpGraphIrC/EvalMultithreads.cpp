//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalMultithreads.cpp : Sample application shows how to evaluate a model in multiple threading environment. 
//
#include "CNTKLibrary.h"

using namespace CNTK;


void EvaluateGraph(FunctionPtr evalFunc, const DeviceDescriptor& device)
{
    fprintf(stderr, "input  count #%lu\n", (unsigned long)evalFunc->Arguments().size());
    fprintf(stderr, "output count #%lu\n", (unsigned long)evalFunc->Outputs().size());

    // TODO remove once we get rid of random input data
    srand(2);

    // Prepare inputs
    std::unordered_map<Variable, ValuePtr> inputs;
    for (auto& input : evalFunc->Arguments())
    {
        // TODO: HERE is out INPUT VECTOR
        std::vector<float> inputData(input.Shape().TotalSize());

        // add some random data to the input vector
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            inputData[i] = ((float)rand()) / RAND_MAX;
        }

        // Todo: add convenience APIs to simplify data preparation here.
        ValuePtr value = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(input.Shape(), inputData, true /* isReadOnly */));

        fprintf(stderr, "  input: %S %S (#%lu elements)\n", input.Name().c_str(), value->Shape().AsString().c_str(), inputData.size());
        inputs[input] = value;
    }

    // Prepare outputs.
    std::unordered_map<Variable, ValuePtr> outputs;
    for (auto output : evalFunc->Outputs())
    {
        outputs[output] = nullptr; // actual value will be filled by evaluating the model.
    }

    // Compute outputs by evaluating the model
    evalFunc->Forward(inputs, outputs, device);
    fprintf(stderr, "\n");

    // Show results by printing the outputs
    for (auto& output : outputs)
    {
        ValuePtr value = output.second;

        // TODO: HERE is out OUTPUT VECTOR
        // TODO: add convenience APIs to simplify data retrieval here.
        std::vector<float> outputData(value->Data()->DataBuffer<float>(), value->Data()->DataBuffer<float>() + value->Data()->Shape().TotalSize());

        fprintf(stderr, "  output: %S %S (#%lu elements)\n", output.first.Name().c_str(), value->Shape().AsString().c_str(), outputData.size());
    }
}
