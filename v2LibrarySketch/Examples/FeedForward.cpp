#include "Model.h"
#include "SGDOptimizer.h"
#include <assert.h>

using namespace CNTK;

Function ReLULayer(Variable input, size_t outputDim)
{
    size_t inputDim = input.Shape().back();

    Variable timesParam = Variable({ outputDim, inputDim }, L"TimesParam");
    auto timesNode = CNTK::Times(timesParam, input);

    Variable plusParam = Variable({ outputDim }, L"BiasParam");
    auto plusNode = CNTK::Plus(plusParam, timesNode.Output());

    return CNTK::ReLU(plusNode.Output());
}

Function FullyConnectedFeedForwardNet(size_t inputDim, size_t numOutputClasses, size_t hiddenLayerDim, size_t numHiddenLayers)
{
    Variable nextLayerInput({ inputDim }, L"features");

    assert(numHiddenLayers >= 1);
    Function prevReLUNode = ReLULayer(nextLayerInput, hiddenLayerDim);
    for (size_t i = 1; i < numHiddenLayers; ++i)
        prevReLUNode = ReLULayer(prevReLUNode.Output(), hiddenLayerDim);

    return prevReLUNode;
}

void TrainFeedForwardClassifier(Reader trainingDataReader)
{
    const size_t inputDim = 937;
    const size_t numOutputClasses = 9404;
    const size_t numHiddenLayers = 6;
    const size_t hiddenLayersDim = 2048;
    Function netOutputNode = FullyConnectedFeedForwardNet(inputDim, numOutputClasses, hiddenLayersDim, numHiddenLayers);

    Variable labels = Variable({ numOutputClasses }, L"labels");
    Function trainingLossNode = CNTK::CrossEntropyWithSoftmax(netOutputNode.Output(), labels, L"lossFunction");
    Function predictionNode = CNTK::PredictionError(netOutputNode.Output(), labels, L"predictionError");

    // Initialize parameters
    auto timesParameters = predictionNode.Inputs(L"Times");
    auto biasParameters = predictionNode.Inputs(L"Bias");
    std::unordered_map<Variable, Value> initialParameters;
    for each (auto param in timesParameters)
        initialParameters.insert({ param, CNTK::RandomNormal(param.Shape(), 0, 1) });

    for each (auto param in biasParameters)
        initialParameters.insert({ param, CNTK::Const(param.Shape(), 0) });

    Model feedForwardClassifier({ trainingLossNode, predictionNode }, initialParameters);

    size_t momentumTimeConstant = 1024;
    // Train with 100000 samples; checkpoint every 10000 samples
    TrainingControl driver = CNTK::BasicTrainingControl(100000, 10000, L"feedForward.net");
    SGDOptimizer feedForwardClassifierOptimizer(feedForwardClassifier, { CNTK::MomentumLearner(predictionNode.Inputs(L"Param"), momentumTimeConstant) });

    Reader textReader = CNTK::TextReader(/*reader config*/);
    feedForwardClassifierOptimizer.TrainCorpus(textReader, trainingLossNode, driver);
}
