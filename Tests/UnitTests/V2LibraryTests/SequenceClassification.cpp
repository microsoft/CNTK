#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

using namespace std::placeholders;

FunctionPtr Embedding(const Variable& input, size_t embeddingDim, const DeviceDescriptor& device)
{
    assert(input.Shape().Rank() == 1);
    size_t inputDim = input.Shape()[0];

    auto embeddingParameters = Parameter(CNTK::NDArrayView::RandomUniform<float>({ embeddingDim, inputDim }, -0.05, 0.05, 1, device));
    return Times(embeddingParameters, input);
}

FunctionPtr LSTMSequenceClassiferNet(const Variable& input, size_t numOutputClasses, size_t embeddingDim, size_t LSTMDim, size_t cellDim, const DeviceDescriptor& device, const std::wstring& outputName)
{
    auto embeddingFunction = Embedding(input, embeddingDim, device);
    auto pastValueRecurrenceHook = [](const Variable& x) { return PastValue(x); };
    auto LSTMFunction = LSTMPComponentWithSelfStabilization<float>(embeddingFunction, { LSTMDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;
    auto thoughtVectorFunction = Sequence::Last(LSTMFunction);

    return FullyConnectedLinearLayer(thoughtVectorFunction, numOutputClasses, device, outputName);
}

void TrainLSTMSequenceClassifer(const DeviceDescriptor& device, bool testSaveAndReLoad)
{
    const size_t inputDim = 2000;
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    const size_t numOutputClasses = 5;

    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, L"features");
    auto classifierOutput = LSTMSequenceClassiferNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels", { Axis::DefaultBatchAxis() });
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

    if (testSaveAndReLoad)
    {
        Variable classifierOutputVar = classifierOutput;
        Variable trainingLossVar = trainingLoss;
        Variable predictionVar = prediction;
        auto oneHiddenLayerClassifier = CNTK::Combine({ trainingLoss, prediction, classifierOutput }, L"classifierModel");
        SaveAndReloadModel<float>(oneHiddenLayerClassifier, { &features, &labels, &trainingLossVar, &predictionVar, &classifierOutputVar }, device);

        classifierOutput = classifierOutputVar;
        trainingLoss = trainingLossVar;
        prediction = predictionVar;
    }

    auto minibatchSource = TextFormatMinibatchSource(L"Train.ctf", { { L"features", inputDim, true, L"x" }, { L"labels", numOutputClasses, false, L"y" } }, 0);
    const size_t minibatchSize = 200;
    
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    double learningRatePerSample = 0.0005;
    size_t momentumTimeConstant = 256;
    double momentumPerSample = std::exp(-1.0 / momentumTimeConstant);
    Trainer trainer(classifierOutput, trainingLoss, prediction, { MomentumSGDLearner(classifierOutput->Parameters(), learningRatePerSample, momentumPerSample) });

    size_t outputFrequencyInMinibatches = 1;
    for (size_t i = 0; true; i++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;

        trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
    }
}

void TrainLSTMSequenceClassifer()
{
#ifndef CPUONLY
    TrainLSTMSequenceClassifer(DeviceDescriptor::GPUDevice(0), true);
#endif
    TrainLSTMSequenceClassifer(DeviceDescriptor::CPUDevice(), false);
}
