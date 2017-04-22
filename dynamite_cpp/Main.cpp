//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"
#include "Layers.h"

#include <iostream>
#include <cstdio>

using namespace CNTK;

using namespace std;

using namespace std::placeholders;

inline FunctionPtr LSTMSequenceClassifierNet(const Variable& input, size_t numOutputClasses, size_t embeddingDim, size_t LSTMDim, size_t cellDim, const DeviceDescriptor& device, const std::wstring& outputName)
{
    auto embeddingFunction = Embedding(input, embeddingDim, device);
    auto pastValueRecurrenceHook = [](const Variable& x) { return PastValue(x); };
    auto LSTMFunction = LSTMPComponentWithSelfStabilization<float>(embeddingFunction, { LSTMDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;
    auto thoughtVectorFunction = Sequence::Last(LSTMFunction);

    return FullyConnectedLinearLayer(thoughtVectorFunction, numOutputClasses, device, outputName);
}

void TrainLSTMSequenceClassifier(const DeviceDescriptor& device, bool useSparseLabels, bool testSaveAndReLoad)
{
    const size_t inputDim = 2000;
    const size_t cellDim = 25;
    const size_t hiddenDim = 25;
    const size_t embeddingDim = 50;
    const size_t numOutputClasses = 5;

    auto featuresName = L"features";
    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, featuresName);
    auto classifierOutput = LSTMSequenceClassifierNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, L"classifierOutput");

    auto labelsName = L"labels";
    auto labels = InputVariable({ numOutputClasses }, useSparseLabels, DataType::Float, labelsName, { Axis::DefaultBatchAxis() });
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

    wstring path = L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/";
    auto minibatchSource = TextFormatMinibatchSource(path + L"Train.ctf",
    {
        { featuresName, inputDim, true, L"x" },
        { labelsName, numOutputClasses, false, L"y" }
    }, MinibatchSource::FullDataSweep);
    const size_t minibatchSize = 200;

    auto featureStreamInfo = minibatchSource->StreamInfo(featuresName);
    auto labelStreamInfo = minibatchSource->StreamInfo(labelsName);

    LearningRatePerSampleSchedule learningRatePerSample = 0.0005;
    MomentumAsTimeConstantSchedule momentumTimeConstant = 256;
    auto trainer = CreateTrainer(classifierOutput, trainingLoss, prediction,
    { MomentumSGDLearner(classifierOutput->Parameters(), learningRatePerSample,
        momentumTimeConstant, /*unitGainMomentum = */true) });

    size_t outputFrequencyInMinibatches = 1;
    for (size_t i = 0; true; i++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;

        trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
    }
}

int main(int argc, char *argv[])
{
    TrainLSTMSequenceClassifier(DeviceDescriptor::GPUDevice(0), true, false);
}
