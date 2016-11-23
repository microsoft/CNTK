//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalMultithreads.cpp : Sample application shows how to evaluate a model in multiple threading environment. 
//
#include <functional>
#include <thread>
#include <iostream>
#include "CNTKLibrary.h"
#include "LSTM/LstmGraphNode.h"

using namespace CNTK;


bool GetVariableByName(std::vector<Variable> variableLists, std::wstring varName, Variable& var)
{
    for (auto it = variableLists.begin(); it != variableLists.end(); ++it)
    {
        if (it->Name().compare(varName) == 0)
        {
            var = *it;
            return true;
        }
    }
    return false;
}

inline bool GetInputVariableByName(FunctionPtr evalFunc, std::wstring varName, Variable& var)
{
    return GetVariableByName(evalFunc->Arguments(), varName, var);
}

inline bool GetOutputVaraiableByName(FunctionPtr evalFunc, std::wstring varName, Variable& var)
{
    return GetVariableByName(evalFunc->Outputs(), varName, var);
}

void EvaluateGraph(FunctionPtr evalFunc, const DeviceDescriptor& device)
{
    std::vector<std::wstring> inputNodeNames = { L"rawAnswer", L"rawContext", L"rawQuery"/*, L"contextSeqAxis", L"sourceSeqAxis"*/ };

    std::vector<Variable> inputVars;
    for (auto inputNodeName : inputNodeNames)
    {
        Variable inputVar;

        if (!GetInputVariableByName(evalFunc, inputNodeName, inputVar))
        {
            fprintf(stderr, "Input variable %S is not available.\n", inputNodeName.c_str());
            throw("Input variable not found error.");
        }

        inputVars.push_back(inputVar);
    }

    // Evaluate the network in several runs 
    size_t numSamples = 3;
    size_t iterationCount = 4;
    unsigned int randSeed = 2;
    srand(randSeed);
    for (size_t t = 0; t < iterationCount; ++t)
    {
        printf("\n\n\n");

        std::unordered_map<Variable, ValuePtr> arguments;

        for (auto inputVar : inputVars)
        {
            std::vector<float> inputData(inputVar.Shape().TotalSize() * numSamples);

            for (size_t i = 0; i < inputData.size(); ++i)
            {
                inputData[i] = ((float)rand()) / RAND_MAX;
            }

            // Create input data shape. Adding sequence length and numSamples as axes.
            // Todo: remove sequence length when only numSamples is supported.
            // Todo: add convenience APIs to simplify data preparation here.
            NDShape inputShape = inputVar.Shape().AppendShape({ 1, numSamples });
            ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputData, true));

            arguments[inputVar] = inputValue;
        }

        // Define output.
        std::unordered_map<Variable, ValuePtr> outputs;

        for (auto ov : evalFunc->Outputs())
        {
            ValuePtr outputValue;
            Variable outputVar;

            outputVar = ov;
            outputs[ov] = outputValue;
        }

        // Evaluate the model
        evalFunc->Forward(arguments, outputs, device);

        ////for (auto outputTuple : outputs)
        ////{
        ////    // Get output value
        ////    auto outputVar = outputTuple.first;
        ////    auto outputValue = outputTuple.second;

        ////    // Todo: remove sequence length when only numSamples is supported.
        ////    // Todo: add convenience APIs to simplify retrieval of output results.
        ////    NDShape outputShape = outputVar.Shape().AppendShape({ 1, numSamples });
        ////    std::vector<float> outputData(outputShape.TotalSize());
        ////    NDArrayViewPtr cpuArrayOutput = MakeSharedObject<NDArrayView>(outputShape, outputData, false);
        ////    cpuArrayOutput->CopyFrom(*outputValue->Data());

        ////    assert(outputData.size() == outputVar.Shape()[0] * numSamples);
        ////    fprintf(stderr, "Evaluation result:\n");
        ////    size_t dataIndex = 0;
        ////    auto outputDim = outputVar.Shape()[0];
        ////    for (size_t i = 0; i < numSamples; i++)
        ////    {
        ////        fprintf(stderr, "Iteration:%lu, Sample %lu:\n", t, i);
        ////        fprintf(stderr, "    ");
        ////        dataIndex = i * outputDim;
        ////        for (size_t j = 0; j < std::min((size_t)10, outputDim); j++)
        ////        {
        ////            fprintf(stderr, "%f ", outputData[dataIndex++]);
        ////        }
        ////        if (outputDim > 10)
        ////        {
        ////            fprintf(stderr, "...");
        ////        }
        ////        fprintf(stderr, "\n");
        ////    }
        ////}
    }
}


