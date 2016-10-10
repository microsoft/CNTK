//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

using namespace std::placeholders;

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

void TestLearningRateControl(const DeviceDescriptor& device)
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

    auto minibatchSource = TextFormatMinibatchSource(L"Train.ctf", { { L"features", inputDim, true, L"x" }, { L"labels", numOutputClasses, false, L"y" } }, 0);
    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    const size_t minibatchSize = 200;
    auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
    auto actualMBSize = minibatchData[labelStreamInfo].m_numSamples;

    LearningRatesPerSample learningRateSchedule({ { 2, 0.0005 }, { 2, 0.00025 } }, actualMBSize);
    auto learner = SGDLearner(classifierOutput->Parameters(), learningRateSchedule);
    Trainer trainer(classifierOutput, trainingLoss, prediction, { learner });
    FloatingPointCompare(learner->LearningRate(), 0.0005, "Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    FloatingPointCompare(learner->LearningRate(), 0.0005, "Learner::LearningRate does not match expectation");

    const wchar_t* modelFile = L"seq2seq.model";
    trainer.SaveCheckpoint(modelFile);

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto MB2Loss = trainer.PreviousMinibatchLossAverage();
    FloatingPointCompare(learner->LearningRate(), 0.00025, "Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto MB3Loss = trainer.PreviousMinibatchLossAverage();
    FloatingPointCompare(learner->LearningRate(), 0.00025, "Learner::LearningRate does not match expectation");

    trainer.RestoreFromCheckpoint(modelFile);
    FloatingPointCompare(learner->LearningRate(), 0.0005, "Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto postRestoreMB2Loss = trainer.PreviousMinibatchLossAverage();
    FloatingPointCompare(postRestoreMB2Loss, MB2Loss, "Post checkpoint restoration training loss does not match expectation");

    FloatingPointCompare(learner->LearningRate(), 0.00025, "Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    auto postRestoreMB3Loss = trainer.PreviousMinibatchLossAverage();
    FloatingPointCompare(postRestoreMB3Loss, MB3Loss, "Post checkpoint restoration training loss does not match expectation");

    trainer.RestoreFromCheckpoint(modelFile);
    FloatingPointCompare(learner->LearningRate(), 0.0005, "Learner::LearningRate does not match expectation");

    learner->ResetLearningRate(0.0004);
    FloatingPointCompare(learner->LearningRate(), 0.0004, "Learner::LearningRate does not match expectation");

    trainer.SaveCheckpoint(modelFile);
    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    postRestoreMB2Loss = trainer.PreviousMinibatchLossAverage();
    FloatingPointCompare(postRestoreMB2Loss, MB2Loss, "Post checkpoint restoration training loss does not match expectation");

    FloatingPointCompare(learner->LearningRate(), 0.0004, "Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    postRestoreMB3Loss = trainer.PreviousMinibatchLossAverage();
    FloatingPointCompare(postRestoreMB3Loss, MB3Loss, "Post checkpoint restoration training loss does not match expectation");

    FloatingPointCompare(learner->LearningRate(), 0.0004, "Learner::LearningRate does not match expectation");

    trainer.RestoreFromCheckpoint(modelFile);
    FloatingPointCompare(learner->LearningRate(), 0.0004, "Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    postRestoreMB2Loss = trainer.PreviousMinibatchLossAverage();
    FloatingPointCompare(postRestoreMB2Loss, MB2Loss, "Post checkpoint restoration training loss does not match expectation");

    FloatingPointCompare(learner->LearningRate(), 0.0004, "Learner::LearningRate does not match expectation");

    trainer.TrainMinibatch({ { features, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
    postRestoreMB3Loss = trainer.PreviousMinibatchLossAverage();
    FloatingPointCompare(postRestoreMB3Loss, MB3Loss, "Post checkpoint restoration training loss does not match expectation");

    FloatingPointCompare(learner->LearningRate(), 0.0004, "Learner::LearningRate does not match expectation");
}

void TrainLSTMSequenceClassifer()
{
    if (IsGPUAvailable())
    {
        TestLearningRateControl(DeviceDescriptor::GPUDevice(0));
    }
    else
    {
        fprintf(stderr, "Cannot run TestLearningRateControl test on CPU device.\n");
    }

    if (IsGPUAvailable())
    {
        TrainLSTMSequenceClassifer(DeviceDescriptor::GPUDevice(0), true);
    }
    TrainLSTMSequenceClassifer(DeviceDescriptor::CPUDevice(), false);
}
