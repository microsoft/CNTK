//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;
using namespace std::placeholders;

extern bool Is1bitSGDAvailable();

void TrainSimpleDistributedFeedForwardClassifer(const DeviceDescriptor& device, DistributedTrainerPtr distributedTrainer, size_t rank)
{
    UNUSED(rank);
    const size_t inputDim = 2;
    const size_t numOutputClasses = 2;
    const size_t hiddenLayerDim = 50;
    const size_t numHiddenLayers = 2;

    const size_t minibatchSize = 25;
    const size_t numSamplesPerSweep = 10000;
    const size_t numSweepsToTrainWith = 2;
    const size_t numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;

    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";
    auto minibatchSource = TextFormatMinibatchSource(L"SimpleDataTrain_cntk_text.txt", { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } }, MinibatchSource::FullDataSweep, false);
    auto featureStreamInfo = minibatchSource->StreamInfo(featureStreamName);
    auto labelStreamInfo = minibatchSource->StreamInfo(labelsStreamName);

    std::unordered_map<StreamInformation, std::pair<NDArrayViewPtr, NDArrayViewPtr>> inputMeansAndInvStdDevs = { { featureStreamInfo, { nullptr, nullptr } } };
    ComputeInputPerDimMeansAndInvStdDevs(minibatchSource, inputMeansAndInvStdDevs);

    auto nonLinearity = std::bind(Sigmoid, _1, L"Sigmoid");
    auto input = InputVariable({ inputDim }, DataType::Float, L"features");
    auto normalizedinput = PerDimMeanVarianceNormalize(input, inputMeansAndInvStdDevs[featureStreamInfo].first, inputMeansAndInvStdDevs[featureStreamInfo].second);
    auto classifierOutput = FullyConnectedDNNLayer(normalizedinput, hiddenLayerDim, device, nonLinearity, std::wstring(L"FullyConnectedInput"));
    for (size_t i = 1; i < numHiddenLayers; ++i)
        classifierOutput = FullyConnectedDNNLayer(classifierOutput, hiddenLayerDim, device, nonLinearity, std::wstring(L"FullyConnectedHidden"));

    auto outputTimesParam = Parameter(NDArrayView::RandomUniform<float>({ numOutputClasses, hiddenLayerDim }, -0.05, 0.05, 1, device), L"outputTimesParam");
    auto outputBiasParam = Parameter(NDArrayView::RandomUniform<float>({ numOutputClasses }, -0.05, 0.05, 1, device), L"outputBiasParam");
    classifierOutput = Plus(outputBiasParam, Times(outputTimesParam, classifierOutput), L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels");
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");;
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

    auto learningRatePerSample = LearningRatePerSampleSchedule(0.02);
    minibatchSource = TextFormatMinibatchSource(L"SimpleDataTrain_cntk_text.txt", { { L"features", inputDim }, { L"labels", numOutputClasses } });
    Trainer trainer(classifierOutput, trainingLoss, prediction, { SGDLearner(classifierOutput->Parameters(), learningRatePerSample) }, distributedTrainer);
    size_t outputFrequencyInMinibatches = 20;
    for (size_t i = 0; i < numMinibatchesToTrain; ++i)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        trainer.TrainMinibatch({ { input, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
    }
}

void TestFrameMode()
{
    {
        auto communicator = MPICommunicator();
        auto distributedTrainer = CreateDataParallelDistributedTrainer(communicator, false);
        TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::CPUDevice(), distributedTrainer, communicator->CurrentWorker().m_globalRank);

        if (IsGPUAvailable())
            TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::GPUDevice(0), distributedTrainer, communicator->CurrentWorker().m_globalRank);
    }

    if (Is1bitSGDAvailable())
    {
        {
            auto communicator = QuantizedMPICommunicator(true, true, 32);
            auto distributedTrainer = CreateQuantizedDataParallelDistributedTrainer(communicator, false);
            TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::CPUDevice(), distributedTrainer, communicator->CurrentWorker().m_globalRank);

            if (IsGPUAvailable())
                TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::GPUDevice(0), distributedTrainer, communicator->CurrentWorker().m_globalRank);
        }

        {
            auto communicator = MPICommunicator();
            auto distributedTrainer = CreateBlockMomentumDistributedTrainer(communicator, 1024);
            TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::CPUDevice(), distributedTrainer, communicator->CurrentWorker().m_globalRank);

            if (IsGPUAvailable())
                TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::GPUDevice(0), distributedTrainer, communicator->CurrentWorker().m_globalRank);
        }
    }
}