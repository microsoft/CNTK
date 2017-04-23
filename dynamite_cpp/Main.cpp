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

FunctionPtr Embedding(const Variable& input, size_t embeddingDim, const DeviceDescriptor& device)
{
    assert(input.Shape().Rank() == 1);
    size_t inputDim = input.Shape()[0];
    auto embeddingParameters = Parameter({ embeddingDim, inputDim }, DataType::Float, GlorotUniformInitializer(), device);
    return Times(embeddingParameters, input);
}

template <typename ElementType>
FunctionPtr RNNStep(Variable input, Variable prevOutput, const DeviceDescriptor& device)
{
    size_t outputDim = prevOutput.Shape()[0];

    auto createBiasParam = [device](size_t dim) {
        return Parameter({ dim }, (ElementType)0.0, device);
    };

    unsigned long seed2 = 1;
    auto createProjectionParam = [device, &seed2](size_t outputDim, size_t inputDim) {
        return Parameter({ outputDim, inputDim },
            AsDataType<ElementType>(),
            GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
    };

    auto W = createProjectionParam(outputDim, NDShape::InferredDimension);
    auto R = createProjectionParam(outputDim, outputDim);
    auto b = createBiasParam(outputDim);

    auto h = ReLU(b + Times(W, input) + Times(R, prevOutput));

    return h;
}

FunctionPtr Dense(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::wstring& outputName = L"")
{
    assert(input.Shape().Rank() == 1);
    size_t inputDim = input.Shape()[0];

    auto timesParam = Parameter({ outputDim, inputDim }, DataType::Float, GlorotUniformInitializer(DefaultParamInitScale, SentinelValueForInferParamInitRank, SentinelValueForInferParamInitRank, 1), device, L"timesParam");
    auto timesFunction = Times(timesParam, input, L"times");

    auto plusParam = Parameter({ outputDim }, 0.0f, device, L"plusParam");
    return Plus(plusParam, timesFunction, outputName);
}

// Embedding(50) >> Recurrence(RNNStep(25)) >> Last >> Dense(5)
inline FunctionPtr CreateModel(const Variable& input, size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim, const DeviceDescriptor& device)
{
    auto r = Embedding(input, embeddingDim, device);

    auto dh = PlaceholderVariable({ hiddenDim }, ((Variable)r).DynamicAxes());
    r = RNNStep<float>(r, dh, device);
    r->ReplacePlaceholders({ { dh, PastValue(r) } });

    r = Sequence::Last(r);
    r = Dense(r, numOutputClasses, device);
    return r;
}

void TrainSequenceClassifier(const DeviceDescriptor& device, bool useSparseLabels)
{
    const size_t inputDim         = 2000;
    const size_t embeddingDim     = 50;
    const size_t hiddenDim        = 25;
    const size_t numOutputClasses = 5;

    const wstring trainingCTFPath = L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf";

    const wstring featuresName = L"features";
    const wstring labelsName   = L"labels";

    auto features = InputVariable({ inputDim }, true /*isSparse*/, DataType::Float, featuresName);
    auto classifierOutput = CreateModel(features, numOutputClasses, embeddingDim, hiddenDim, device);

    auto labels = InputVariable({ numOutputClasses }, useSparseLabels, DataType::Float, labelsName, { Axis::DefaultBatchAxis() });
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels);
    auto prediction   = CNTK::ClassificationError    (classifierOutput, labels);

    auto minibatchSource = TextFormatMinibatchSource(trainingCTFPath,
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
