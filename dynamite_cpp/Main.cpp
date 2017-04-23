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

typedef function<FunctionPtr(Variable)> UnaryFunction;

UnaryFunction Embedding(size_t embeddingDim, const DeviceDescriptor& device)
{
    auto E = Parameter({ embeddingDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);

    return [=](Variable x) { return Times(E, x); };
}

template <typename ElementType>
FunctionPtr RNNStep(Variable prevOutput, Variable input, const DeviceDescriptor& device)
{
    size_t outputDim = prevOutput.Shape()[0];

    auto W = Parameter({ outputDim, NDShape::InferredDimension }, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto R = Parameter({ outputDim, outputDim                  }, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto b = Parameter({ outputDim }, (ElementType)0.0, device);

    return ReLU(Times(W, input) + Times(R, prevOutput) + b);
}

UnaryFunction Linear(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    auto b = Parameter({ outputDim }, 0.0f, device);

    return [=](Variable x) { return Times(W, x) + b; };
}

FunctionPtr Sequential(const vector<FunctionPtr>& fns)
{
    for (size_t i = 1; i < fns.size(); i++)
        fns[i]->ReplacePlaceholder(fns[i - 1]->Output());
    return fns.back();
}

UnaryFunction Recurrence(const function<FunctionPtr(Variable,Variable)>& stepFunction)
{
    return [=](Variable x)
    {
        auto dh = PlaceholderVariable({ 25 }, Axis::UnknownDynamicAxes());
        auto rec = stepFunction(PastValue(dh), x);
        rec->ReplacePlaceholders({ { dh, rec } });
        return rec;
    };
}

// Embedding(50) >> Recurrence(RNNStep(25)) >> Last >> Linear(5)
FunctionPtr CreateModel(size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim, const DeviceDescriptor& device)
{
    return Sequential({
        Embedding(embeddingDim, device)(PlaceholderVariable()),
        Recurrence([device](Variable dh, Variable x) -> FunctionPtr { return RNNStep<float>(dh, x, device); })(PlaceholderVariable()),
        Sequence::Last(PlaceholderVariable()),
        Linear(numOutputClasses, device)(PlaceholderVariable())
    });
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
    auto classifierOutput = CreateModel(numOutputClasses, embeddingDim, hiddenDim, device);
    classifierOutput->ReplacePlaceholder(features);

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
    auto trainer = CreateTrainer(nullptr, trainingLoss, prediction, { learner });

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
    try
    {
        //TrainSequenceClassifier(DeviceDescriptor::GPUDevice(0), true);
        TrainSequenceClassifier(DeviceDescriptor::CPUDevice(), true);
    }
    catch (exception& e)
    {
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what());
    }
}
