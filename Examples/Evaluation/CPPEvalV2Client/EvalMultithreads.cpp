#include <functional>
#include <thread>
#include <iostream>
#include "CNTKLibrary.h"

using namespace CNTK;

FunctionPtr FullyConnectedDNNLayerWithSharedParameters(Variable input,
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

FunctionPtr FullyConnectedFeedForwardClassifierNetWithSharedParameters(Variable input,
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
        classifierRoot = FullyConnectedDNNLayerWithSharedParameters(classifierRoot, hiddenLayerTimesParam[i - 1], hiddenLayerPlusParam[i - 1], nonLinearity);

    // Todo: assume that outputTimesParam has matched output dim and hiddenLayerDim
    classifierRoot = Times(outputTimesParam, classifierRoot);
    return classifierRoot;
}

void EvaluationNewNetworkWithSharedParameters(size_t inputDim,
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
        throw std::runtime_error("EvaluationNewNetworkWithSharedParameters: Function does not have expected Parameter count");

    if (ffNet->Arguments().size() != 2)
        throw std::runtime_error("EvaluationNewNetworkWithSharedParameters: Function does not have expected Argument count");

    if (ffNet->Outputs().size() != 3)
        throw std::runtime_error("EvaluationNewNetworkWithSharedParameters: Function does not have expected Output count");

    // Evaluate the network in several runs 
    size_t iterationCount = 4;
    unsigned int randSeed = 2;
    srand(randSeed);
    size_t numSamples = 3;
    for (size_t t = 0; t < iterationCount; ++t)
    {
        std::vector<float> inputData(inputDim * numSamples);
        for (size_t i = 0; i < inputData.size(); ++i)
            inputData[i] = ((float)rand()) / RAND_MAX;

        NDShape inputShape = {inputDim, 1, numSamples};
        ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true));

        std::vector<float> labelData(numOutputClasses * numSamples, 0);
        for (size_t i = 0; i < numSamples; ++i)
            labelData[(i*numOutputClasses) + (rand() % numOutputClasses)] = 1;

        NDShape labelShape = {numOutputClasses, 1, numSamples};
        ValuePtr labelValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(labelShape, labelData.data(), labelData.size(), DeviceDescriptor::CPUDevice(), true));

        ValuePtr outputValue, predictionErrorValue;
        std::unordered_map<Variable, ValuePtr> outputs = {{classifierOutputFunction->Output(), outputValue}, {predictionFunction->Output(), predictionErrorValue}};
        ffNet->Forward({{inputVar, inputValue}, {labelsVar, labelValue}}, outputs, computeDevice);
    }
}

void EvalMultiThreadsWithNewNetwork(const DeviceDescriptor& device, const int threadCount)
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
    for (int  th = 0; th < threadCount; ++th)
    {
        threadList[th] = std::thread(EvaluationNewNetworkWithSharedParameters, inputDim, numOutputClasses, numHiddenLayers, inputTimesParam, inputPlusParam, hiddenLayerTimesParam, hiddenLayerPlusParam, outputTimesParam, device);
    }

    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th].join();
        fprintf(stderr, "thread %d joined.\n", th);
        fflush(stderr);
    }
}
