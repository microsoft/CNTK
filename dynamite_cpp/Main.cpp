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

typedef function<Variable(Variable)> UnaryModel;
typedef function<Variable(Variable, Variable)> BinaryModel;

UnaryModel Embedding(size_t embeddingDim, const DeviceDescriptor& device)
{
    auto E = Parameter({ embeddingDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    return [=](Variable x)
    {
        return Times(E, x);
    };
}

BinaryModel RNNStep(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    auto R = Parameter({ outputDim, outputDim                  }, DataType::Float, GlorotUniformInitializer(), device);
    auto b = Parameter({ outputDim }, 0.0f, device);
    return [=](Variable prevOutput, Variable input)
    {
        return ReLU(Times(W, input) + Times(R, prevOutput) + b);
    };
}

UnaryModel Linear(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    auto b = Parameter({ outputDim }, 0.0f, device);
    return [=](Variable x) { return Times(W, x) + b; };
}

UnaryModel Sequential(const vector<UnaryModel>& fns)
{
    return [=](Variable x)
    {
        auto arg = Combine({ x });
        for (const auto& f : fns)
            arg = f(arg);
        return arg;
    };
}

UnaryModel Recurrence(const BinaryModel& stepFunction)
{
    return [=](Variable x)
    {
        auto dh = PlaceholderVariable();
        auto rec = stepFunction(PastValue(dh), x);
        FunctionPtr(rec)->ReplacePlaceholders({ { dh, rec } });
        return rec;
    };
}

UnaryModel Fold(const BinaryModel& stepFunction)
{
    auto recurrence = Recurrence(stepFunction);
    return [=](Variable x)
    {
        return Sequence::Last(recurrence(x));
    };
}

UnaryModel CreateModelFunction(size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim, const DeviceDescriptor& device)
{
    return Sequential({
        Embedding(embeddingDim, device),
        Fold(RNNStep(hiddenDim, device)),
        Linear(numOutputClasses, device)
    });
}

function<pair<Variable,Variable>(Variable, Variable)> CreateCriterionFunction(UnaryModel model)
{
    return [=](Variable features, Variable labels)
    {
        auto z = model(features);

        auto loss   = CNTK::CrossEntropyWithSoftmax(z, labels);
        auto metric = CNTK::ClassificationError    (z, labels);

        return make_pair(loss, metric);
    };
}

void TrainSequenceClassifier(const DeviceDescriptor& device, bool useSparseLabels)
{
    const size_t inputDim         = 2000;
    const size_t embeddingDim     = 50;
    const size_t hiddenDim        = 25;
    const size_t numOutputClasses = 5;

    const wstring trainingCTFPath = L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf";

    // model and criterion function
    auto model_fn = CreateModelFunction(numOutputClasses, embeddingDim, hiddenDim, device);
    auto criterion_fn = CreateCriterionFunction(model_fn);

    // data
    const wstring featuresName = L"features";
    const wstring labelsName   = L"labels";

    auto minibatchSource = TextFormatMinibatchSource(trainingCTFPath,
    {
        { featuresName, inputDim,         true,  L"x" },
        { labelsName,   numOutputClasses, false, L"y" }
    }, MinibatchSource::FullDataSweep);

    auto featureStreamInfo = minibatchSource->StreamInfo(featuresName);
    auto labelStreamInfo   = minibatchSource->StreamInfo(labelsName);

    // build the graph
    auto features = InputVariable({ inputDim },         true /*isSparse*/, DataType::Float, featuresName);
    auto labels   = InputVariable({ numOutputClasses }, useSparseLabels,   DataType::Float, labelsName, { Axis::DefaultBatchAxis() });

    auto criterion = criterion_fn(features, labels);
    auto loss   = criterion.first;
    auto metric = criterion.second;

    // train
    auto learner = SGDLearner(FunctionPtr(loss)->Parameters(), LearningRatePerSampleSchedule(0.05));
    auto trainer = CreateTrainer(nullptr, loss, metric, { learner });

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
