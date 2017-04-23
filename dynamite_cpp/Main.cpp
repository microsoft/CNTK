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

// Embedding(50) >> Recurrence(RNNStep(25)) >> Last >> Dense(5)
inline FunctionPtr CreateModel(const Variable& input, size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim, const DeviceDescriptor& device, const std::wstring& outputName)
{
    auto r = Embedding(input, embeddingDim, device);

    auto dh = PlaceholderVariable({ hiddenDim }, ((Variable)r).DynamicAxes());
    r = RNNStep<float>(r, dh, device);
    r->ReplacePlaceholders({ { dh, PastValue(r) } });

    r = Sequence::Last(r);
    r = FullyConnectedLinearLayer(r, numOutputClasses, device, outputName);
    return r;
}

void TrainSequenceClassifier(const DeviceDescriptor& device, bool useSparseLabels)
{
    const size_t inputDim         = 2000;
    const size_t embeddingDim     = 50;
    const size_t hiddenDim        = 25;
    const size_t numOutputClasses = 5;

    const wstring featuresName = L"features";
    const wstring labelsName   = L"labels";

    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, featuresName);
    auto classifierOutput = CreateModel(features, numOutputClasses, embeddingDim, hiddenDim, device, L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, useSparseLabels, DataType::Float, labelsName, { Axis::DefaultBatchAxis() });
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

    const wstring path = L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/";
    auto minibatchSource = TextFormatMinibatchSource(path + L"Train.ctf",
    {
        { featuresName, inputDim,         true,  L"x" },
        { labelsName,   numOutputClasses, false, L"y" }
    }, MinibatchSource::FullDataSweep);

    auto featureStreamInfo = minibatchSource->StreamInfo(featuresName);
    auto labelStreamInfo   = minibatchSource->StreamInfo(labelsName);

    auto learner = SGDLearner(classifierOutput->Parameters(), LearningRatePerSampleSchedule(0.05));
    auto trainer = CreateTrainer(classifierOutput, trainingLoss, prediction, { learner });

    const size_t minibatchSize = 200;
    for (size_t i = 0; true; i++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;

        trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
        PrintTrainingProgress(trainer, i, /*outputFrequencyInMinibatches=*/ 1);
    }
}

int main(int argc, char *argv[])
{
    TrainSequenceClassifier(DeviceDescriptor::GPUDevice(0), true);
}
