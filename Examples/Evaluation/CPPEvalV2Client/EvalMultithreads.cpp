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

using namespace CNTK;

void OutputFunctionInfo(FunctionPtr);
FunctionPtr FullyConnectedDNNLayerWithSharedParameters(Variable, const Parameter&, const Parameter&, const std::function<FunctionPtr(const FunctionPtr&)>&);
void CreateFunctionAndEvaluateWithSharedParameters(size_t, size_t, size_t, const Parameter&, const Parameter&, const Parameter[], const Parameter[], const Parameter&, const DeviceDescriptor&);
FunctionPtr SetupFullyConnectedLinearLayer(Variable, size_t, const DeviceDescriptor&, const std::wstring&);
FunctionPtr SetupFullyConnectedDNNLayer(Variable, size_t, const DeviceDescriptor& device, const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity);
void RunEvaluationClassifier(FunctionPtr, const DeviceDescriptor&);
void RunEvaluationOneHidden(FunctionPtr, const DeviceDescriptor&);

/// <summary>
/// Shows how to create Function whose parameters can be shared by multi evaluation threads.
/// </summary>
/// <description>
/// It first creates all parameters needed for the Function, and then spawns multi threads. 
/// Althought each thread creates a new instance of function, all threads share the same parameters.
/// After that, each thread runs evaluation independently.
/// </description>
void MultiThreadsEvaluationWithNewFunction(const DeviceDescriptor& device, const int threadCount)
{
    const size_t inputDim = 937;
    const size_t numOutputClasses = 9304;
    const size_t numHiddenLayers = 6;
    const size_t hiddenLayersDim = 2048;

    // Define model parameters that should be shared among evaluation requests against the same model
    auto inputTimesParam = Parameter(NDArrayView::RandomUniform<float>({hiddenLayersDim, inputDim}, -0.5, 0.5, 1, device));
    auto inputPlusParam = Parameter({hiddenLayersDim}, 0.0f, device);
    Parameter hiddenLayerTimesParam[numHiddenLayers - 1] = {
        Parameter(NDArrayView::RandomUniform<float>({hiddenLayersDim, hiddenLayersDim}, -0.5, 0.5, 1, device)),
        Parameter(NDArrayView::RandomUniform<float>({hiddenLayersDim, hiddenLayersDim}, -0.5, 0.5, 1, device)),
        Parameter(NDArrayView::RandomUniform<float>({hiddenLayersDim, hiddenLayersDim}, -0.5, 0.5, 1, device)),
        Parameter(NDArrayView::RandomUniform<float>({hiddenLayersDim, hiddenLayersDim}, -0.5, 0.5, 1, device)),
        Parameter(NDArrayView::RandomUniform<float>({hiddenLayersDim, hiddenLayersDim}, -0.5, 0.5, 1, device))
    };
    Parameter hiddenLayerPlusParam[numHiddenLayers - 1] = {
        Parameter({hiddenLayersDim}, 0.0f, device),
        Parameter({hiddenLayersDim}, 0.0f, device),
        Parameter({hiddenLayersDim}, 0.0f, device),
        Parameter({hiddenLayersDim}, 0.0f, device),
        Parameter({hiddenLayersDim}, 0.0f, device),
    };
    auto outputTimesParam = Parameter(NDArrayView::RandomUniform<float>({numOutputClasses, hiddenLayersDim}, -0.5, 0.5, 1, device));

    // Run evaluation in parallel    
    std::vector<std::thread> threadList(threadCount);
    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th] = std::thread(CreateFunctionAndEvaluateWithSharedParameters, inputDim, numOutputClasses, numHiddenLayers, inputTimesParam, inputPlusParam, hiddenLayerTimesParam, hiddenLayerPlusParam, outputTimesParam, device);
    }

    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th].join();
        fprintf(stderr, "thread %d joined.\n", th);
        fflush(stderr);
    }
}

/// <summary>
/// Shows how to use Clone() to share function parameters among multi evaluation threads.
/// </summary>
/// <description>
/// It first creates a new function with parameters, then spawns multi threads. Each thread uses Clone() to create a new
/// instance of function and then use this instance to do evaluation.
/// All cloned functions share the same parameters.
/// </description>
void MultiThreadsEvaluationWithClone(const DeviceDescriptor& device, const int threadCount)
{
    using namespace std::placeholders;

    const size_t inputDim = 937;
    const size_t numOutputClasses = 9304;
    const size_t numHiddenLayers = 6;
    const size_t hiddenLayersDim = 2048;

    auto inputVar = InputVariable({inputDim}, DataType::Float, L"features");

    assert(numHiddenLayers >= 1);
    auto classifierRoot = SetupFullyConnectedDNNLayer(inputVar, hiddenLayersDim, device, std::bind(Sigmoid, _1, L""));
    for (size_t i = 1; i < numHiddenLayers; ++i)
    {
        classifierRoot = SetupFullyConnectedDNNLayer(classifierRoot, hiddenLayersDim, device, std::bind(Sigmoid, _1, L""));
    }

    auto outputTimesParam = Parameter(NDArrayView::RandomUniform<float>({numOutputClasses, hiddenLayersDim}, -0.5, 0.5, 1, device));
    auto classifierFunc = Times(outputTimesParam, classifierRoot, 1, L"classifierOutput");

    // Now test the structure
    if (classifierFunc->Parameters().size() != ((numHiddenLayers * 2) + 1))
    {
        throw std::runtime_error("MultiThreadsEvaluationWithClone: Function does not have expected Parameter count");
    }

    OutputFunctionInfo(classifierFunc);
    fprintf(stderr, "MultiThreadsEvaluationWithClone on device=%d\n", device.Id());

    // Run evaluation in parallel
    std::vector<std::thread> threadList(threadCount);
    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th] = std::thread(RunEvaluationClassifier, classifierFunc->Clone(), device);
    }

    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th].join();
        fprintf(stderr, "thread %d joined.\n", th);
        fflush(stderr);
    }
}

/// <summary>
/// Shows how to use LoadLegacyModel() and Clone() to share function parameters among multi evaluation threads.
/// </summary>
/// <description>
/// It first loads a model, then spawns multi threads. Each thread uses Clone() to create a new
/// instance of function and then use this instance to do evaluation.
/// All cloned functions share the same parameters.
/// </description>
void MultiThreadsEvaluationWithLoadModel(const DeviceDescriptor& device, const int threadCount)
{
    // The model file will be trained and copied to the current runtime directory first.
    auto modelFuncPtr = CNTK::LoadLegacyModel(DataType::Float, L"01_OneHidden", device);


    OutputFunctionInfo(modelFuncPtr);
    fprintf(stderr, "MultiThreadsEvaluationWithLoadModel on device=%d\n", device.Id());

    // Run evaluation in parallel.
    std::vector<std::thread> threadList(threadCount);
    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th] = std::thread(RunEvaluationOneHidden, modelFuncPtr->Clone(), device);
    }

    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th].join();
        fprintf(stderr, "thread %d joined.\n", th);
        fflush(stderr);
    }
}

inline FunctionPtr FullyConnectedDNNLayerWithSharedParameters(Variable input,
                                                              const Parameter& timesParam,
                                                              const Parameter& plusParam,
                                                              const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity)
{
    assert(input.Shape().Rank() == 1);

    // Todo: assume that timesParam has matched outputDim and inputDim 
    auto timesFunction = Times(timesParam, input);

    // Todo: assume that timesParam has matched outputDim 
    auto plusFunction = Plus(plusParam, timesFunction);

    return nonLinearity(plusFunction);
}

inline FunctionPtr FullyConnectedFeedForwardClassifierNetWithSharedParameters(Variable input,
                                                                              size_t numHiddenLayers,
                                                                              const Parameter& inputTimesParam,
                                                                              const Parameter& inputPlusParam,
                                                                              const Parameter hiddenLayerTimesParam[],
                                                                              const Parameter hiddenLayerPlusParam[],
                                                                              const Parameter& outputTimesParam,
                                                                              const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity)
{
    assert(numHiddenLayers >= 1);
    auto classifierRoot = FullyConnectedDNNLayerWithSharedParameters(input, inputTimesParam, inputPlusParam, nonLinearity);

    for (size_t i = 1; i < numHiddenLayers; ++i)
    {
        classifierRoot = FullyConnectedDNNLayerWithSharedParameters(classifierRoot, hiddenLayerTimesParam[i - 1], hiddenLayerPlusParam[i - 1], nonLinearity);
    }

    // Todo: assume that outputTimesParam has matched output dim and hiddenLayerDim
    classifierRoot = Times(outputTimesParam, classifierRoot);
    return classifierRoot;
}

void CreateFunctionAndEvaluateWithSharedParameters(size_t inputDim,
                                                   size_t numOutputClasses,
                                                   size_t numHiddenLayers,
                                                   const Parameter& inputTimesParam,
                                                   const Parameter& inputPlusParam,
                                                   const Parameter hiddenLayerTimesParam[],
                                                   const Parameter hiddenLayerPlusParam[],
                                                   const Parameter& outputTimesParam,
                                                   const DeviceDescriptor& computeDevice)
{
    using namespace std::placeholders;

    // Create network using shared parameters
    auto inputVar = InputVariable({inputDim}, DataType::Float, L"Features");
    auto classifierOutputFunction = FullyConnectedFeedForwardClassifierNetWithSharedParameters(inputVar,
                                                                                               numHiddenLayers,
                                                                                               inputTimesParam,
                                                                                               inputPlusParam,
                                                                                               hiddenLayerTimesParam,
                                                                                               hiddenLayerPlusParam,
                                                                                               outputTimesParam,
                                                                                               std::bind(Sigmoid, _1, L""));

    auto labelsVar = InputVariable({numOutputClasses}, DataType::Float, L"Labels");
    auto trainingLossFunction = CNTK::CrossEntropyWithSoftmax(classifierOutputFunction, labelsVar, L"LossFunction");
    auto predictionFunction = CNTK::ClassificationError(classifierOutputFunction, labelsVar, L"ClassificationError");

    auto ffNet = CNTK::Combine({trainingLossFunction, predictionFunction, classifierOutputFunction}, L"ClassifierModel");

    if (ffNet->Parameters().size() != ((numHiddenLayers * 2) + 1))
    {
        throw std::runtime_error("CreateFunctionAndEvaluateWithSharedParameters: Function does not have expected Parameter count");
    }

    if (ffNet->Arguments().size() != 2)
    {
        throw std::runtime_error("CreateFunctionAndEvaluateWithSharedParameters: Function does not have expected Argument count");
    }

    if (ffNet->Outputs().size() != 3)
    {
        throw std::runtime_error("CreateFunctionAndEvaluateWithSharedParameters: Function does not have expected Output count");
    }

    // Evaluate the network in several runs 
    size_t iterationCount = 4;
    unsigned int randSeed = 2;
    srand(randSeed);
    size_t numSamples = 3;
    for (size_t t = 0; t < iterationCount; ++t)
    {
        std::vector<float> inputData(inputDim * numSamples);
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            inputData[i] = ((float)rand()) / RAND_MAX;
        }

        NDShape inputShape = {inputDim, 1, numSamples};
        ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true));

        std::vector<float> labelData(numOutputClasses * numSamples, 0);
        for (size_t i = 0; i < numSamples; ++i)
        {
            labelData[(i*numOutputClasses) + (rand() % numOutputClasses)] = 1;
        }

        NDShape labelShape = {numOutputClasses, 1, numSamples};
        ValuePtr labelValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(labelShape, labelData.data(), labelData.size(), DeviceDescriptor::CPUDevice(), true));

        ValuePtr outputValue, predictionErrorValue;
        std::unordered_map<Variable, ValuePtr> outputs = {{classifierOutputFunction->Output(), outputValue}, {predictionFunction->Output(), predictionErrorValue}};
        ffNet->Forward({{inputVar, inputValue}, {labelsVar, labelValue}}, outputs, computeDevice);
    }
}


inline FunctionPtr SetupFullyConnectedLinearLayer(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::wstring& outputName = L"")
{
    assert(input.Shape().Rank() == 1);
    size_t inputDim = input.Shape()[0];

    auto timesParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({outputDim, inputDim}, -0.05, 0.05, 1, device));
    auto timesFunction = CNTK::Times(timesParam, input);

    auto plusParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({outputDim}, -0.05, 0.05, 1, device));
    return CNTK::Plus(plusParam, timesFunction, outputName);
}

inline FunctionPtr SetupFullyConnectedDNNLayer(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity)
{
    return nonLinearity(SetupFullyConnectedLinearLayer(input, outputDim, device));
}

void OutputFunctionInfo(FunctionPtr func)
{
    auto inputVariables = func->Arguments();
    fprintf(stderr, "Function %S: Input Variables (count=%lu)\n", func->Name().c_str(), inputVariables.size());
    for_each(inputVariables.begin(), inputVariables.end(), [](const Variable v) {
        fprintf(stderr, "    name=%S, kind=%d\n", v.Name().c_str(), static_cast<int>(v.Kind()));
    });

    auto outputVariables = func->Outputs();
    fprintf(stderr, "Function %S: Output Variables (count=%lu)\n", func->Name().c_str(), outputVariables.size());
    for_each(outputVariables.begin(), outputVariables.end(), [](const Variable v) {
        fprintf(stderr, "    name=%S, kind=%d\n", v.Name().c_str(), static_cast<int>(v.Kind()));
    });
}

bool GetVariableByName(std::vector<Variable> variableLists, std::wstring varName, Variable& var)
{
    for (std::vector<Variable>::iterator it = variableLists.begin(); it != variableLists.end(); ++it)
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

void RunEvaluationClassifier(FunctionPtr evalFunc, const DeviceDescriptor& device)
{
    const std::wstring inputNodeName = L"features";

    Variable inputVar;
    if (!GetInputVariableByName(evalFunc, inputNodeName, inputVar))
    {
        fprintf(stderr, "Input variable %S is not available.\n", inputNodeName.c_str());
        throw("Input variable not found error.");
    }

    // Evaluate the network in several runs 
    size_t iterationCount = 4;
    unsigned int randSeed = 2;
    srand(randSeed);
    size_t numSamples = 3;
    std::vector<float> inputData(inputVar.Shape().TotalSize() * numSamples);
    for (size_t t = 0; t < iterationCount; ++t)
    {
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            inputData[i] = ((float)rand()) / RAND_MAX;
        }

        // Create input data shape. Adding sequence length and numSamples as axes.
        // Todo: remove sequence length when only numSamples is supported.
        // Todo: add convenience APIs to simplify data preparation here.
        NDShape inputShape = inputVar.Shape().AppendShape({1, numSamples});
        ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputData, true));

        // Define output.
        ValuePtr outputValue;
        auto outputVar = evalFunc->Output();
        std::unordered_map<Variable, ValuePtr> outputs = {{outputVar, outputValue}};

        // Evaluate the model
        evalFunc->Forward({{inputVar, inputValue}}, outputs, device);

        // Get output value
        outputValue = outputs[outputVar];

        // Todo: remove sequence length when only numSamples is supported.
        // Todo: add convenience APIs to simplify retrieval of output results.
        NDShape outputShape = outputVar.Shape().AppendShape({1, numSamples});
        std::vector<float> outputData(outputShape.TotalSize());
        NDArrayViewPtr cpuArrayOutput = MakeSharedObject<NDArrayView>(outputShape, outputData, false);
        cpuArrayOutput->CopyFrom(*outputValue->Data());

        assert(outputData.size() == outputVar.Shape()[0] * numSamples);
        fprintf(stderr, "Evaluation result:\n");
        size_t dataIndex = 0;
        auto outputDim = outputVar.Shape()[0];
        for (size_t i = 0; i < numSamples; i++)
        {
            fprintf(stderr, "Iteration:%lu, Sample %lu:\n", t, i);
            fprintf(stderr, "    ");
            dataIndex = i * outputDim;
            for (size_t j = 0; j < std::min((size_t)10, outputDim); j++)
            {
                fprintf(stderr, "%f ", outputData[dataIndex++]);
            }
            if (outputDim > 10)
            {
                fprintf(stderr, "...");
            }
            fprintf(stderr, "\n");
        }
    }
}

void RunEvaluationOneHidden(FunctionPtr evalFunc, const DeviceDescriptor& device)
{
    const std::wstring inputNodeName = L"features";
    const std::wstring outputNodeName = L"out.z_output";

    Variable inputVar;
    if (!GetInputVariableByName(evalFunc, inputNodeName, inputVar))
    {
        fprintf(stderr, "Input variable %S is not available.\n", inputNodeName.c_str());
        throw("Input variable not found error.");
    }

    Variable outputVar;
    if (!GetOutputVaraiableByName(evalFunc, outputNodeName, outputVar))
    {
        fprintf(stderr, "Output variable %S is not available.\n", outputNodeName.c_str());
        throw("Output variable not found error.");
    }

    // Evaluate the network in several runs 
    size_t iterationCount = 4;   
    size_t numSamples = 3;
    for (size_t t = 0; t < iterationCount; ++t)
    {
        std::vector<float> inputData(inputVar.Shape().TotalSize() * numSamples);
        for (size_t i = 0; i < inputData.size(); ++i)
        {
            inputData[i] = static_cast<float>(i % 255);
        }

        NDShape inputShape = inputVar.Shape().AppendShape({1, numSamples});
        ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputData, true));

        ValuePtr outputValue;
        std::unordered_map<Variable, ValuePtr> outputs = {{outputVar, outputValue}};
        evalFunc->Forward({{inputVar, inputValue}}, outputs, device);

        outputValue = outputs[outputVar];        
        NDShape outputShape = outputVar.Shape().AppendShape({1, numSamples});
        std::vector<float> outputData(outputShape.TotalSize());
        NDArrayViewPtr cpuArrayOutput = MakeSharedObject<NDArrayView>(outputShape, outputData, false);
        cpuArrayOutput->CopyFrom(*outputValue->Data());

        assert(outputData.size() == outputVar.Shape()[0] * numSamples);
        fprintf(stderr, "Evaluation result:\n");
        size_t dataIndex = 0;
        auto outputDim = outputVar.Shape()[0];
        for (size_t i = 0; i < numSamples; i++)
        {
            fprintf(stderr, "Iteration:%lu, Sample %lu:\n", t, i);
            fprintf(stderr, "Ouput:");
            for (size_t j = 0; j < outputDim; j++)
            {
                fprintf(stderr, "%f ", outputData[dataIndex++]);
            }
            fprintf(stderr, "\n");
        }
    }
}

void MultiThreadsEvaluation(bool isGPUAvailable)
{
#ifndef CPUONLY
    if (isGPUAvailable)
    {
        fprintf(stderr, "Run evaluation on GPU device using GPU build.\n");
    }
    else
    {
        fprintf(stderr, "Run evaluation on CPU device using GPU build.\n");
    }
#else
    fprintf(stderr, "Run evaluation using CPU-only build.\n");
#endif

    // Test multi-threads evaluation with new function
    fprintf(stderr, "Test multi-threaded evaluation with new function on CPU.\n");
    MultiThreadsEvaluationWithNewFunction(DeviceDescriptor::CPUDevice(), 2);
    if (isGPUAvailable)
    {
        fprintf(stderr, "Test multi-threaded evaluation with new function on GPU\n");
        MultiThreadsEvaluationWithNewFunction(DeviceDescriptor::GPUDevice(0), 2);
    }

    // Test multi-threads evaluation using clone.
    fprintf(stderr, "Test multi-threaded evaluation using clone on CPU.\n");
    MultiThreadsEvaluationWithClone(DeviceDescriptor::CPUDevice(), 2);
    if (isGPUAvailable)
    {
        fprintf(stderr, "Test multi-threaded evaluation using clone on GPU.\n");
        MultiThreadsEvaluationWithClone(DeviceDescriptor::GPUDevice(0), 2);
    }

    // test multi-threads evaluation with loading existing models
    fprintf(stderr, "Test multi-threaded evaluation with loading existing models on CPU.\n");
    MultiThreadsEvaluationWithLoadModel(DeviceDescriptor::CPUDevice(), 2);
    if (isGPUAvailable)
    {
        fprintf(stderr, "Test multi-threaded evaluation with loading existing models on GPU.\n");
        MultiThreadsEvaluationWithLoadModel(DeviceDescriptor::GPUDevice(0), 2);
    }

    fflush(stderr);

}
