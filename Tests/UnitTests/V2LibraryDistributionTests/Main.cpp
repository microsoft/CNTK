//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;
using namespace std::placeholders;

bool Is1bitSGDAvailable()
{
    static bool is1bitSGDAvailable;
    static bool isInitialized = false;

    if (!isInitialized)
    {
        const char* p = getenv("TEST_1BIT_SGD");

        // Check the environment variable TEST_1BIT_SGD to decide whether to run on a CPU-only device.
        if (p != nullptr && 0 == strcmp(p, "0"))
        {
            is1bitSGDAvailable = false;
        }
        else
        {
            is1bitSGDAvailable = true;
        }
        isInitialized = true;
    }

    return is1bitSGDAvailable;
}

// TODO: Move to other file.
void TrainSimpleDistributedFeedForwardClassifer(const DeviceDescriptor& device, DistributedTrainerPtr distributedTrainer, size_t rank)
{
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
    auto classifierOutput = FullyConnectedDNNLayer(normalizedinput, hiddenLayerDim, device, nonLinearity, std::wstring(L"FullyConnectedInput") );
    for (size_t i = 1; i < numHiddenLayers; ++i)
        classifierOutput = FullyConnectedDNNLayer(classifierOutput, hiddenLayerDim, device, nonLinearity, std::wstring(L"FullyConnectedHidden"));

    auto outputTimesParam = Parameter(NDArrayView::RandomUniform<float>({ numOutputClasses, hiddenLayerDim }, -0.05, 0.05, 1, device), L"outputTimesParam");
    auto outputBiasParam = Parameter(NDArrayView::RandomUniform<float>({ numOutputClasses }, -0.05, 0.05, 1, device), L"outputBiasParam");
    classifierOutput = Plus(outputBiasParam, Times(outputTimesParam, classifierOutput), L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels");
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");;
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

    // Test save and reload of model
    {
        Variable classifierOutputVar = classifierOutput;
        Variable trainingLossVar = trainingLoss;
        Variable predictionVar = prediction;
        auto combinedNet = Combine({ trainingLoss, prediction, classifierOutput }, L"feedForwardClassifier");
        SaveAndReloadModel<float>(combinedNet, { &input, &labels, &trainingLossVar, &predictionVar, &classifierOutputVar }, device, rank);

        classifierOutput = classifierOutputVar;
        trainingLoss = trainingLossVar;
        prediction = predictionVar;
    }

    double learningRatePerSample = 0.02;
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

int main(int /*argc*/, char* /*argv*/[])
{

#if defined(_MSC_VER)
    // in case of asserts in debug mode, print the message into stderr and throw exception
    if (_CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, HandleDebugAssert) == -1) {
        fprintf(stderr, "_CrtSetReportHook2 failed.\n");
        return -1;
    }
#endif

    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking 
    // which will have a silent performance degradation otherwise
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

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

    fprintf(stderr, "\nCNTKv2LibraryDistribution tests: Passed\n");
    fflush(stderr);

#if defined(_MSC_VER)
    _CrtSetReportHook2(_CRT_RPTHOOK_REMOVE, HandleDebugAssert);
#endif

    DistributedCommunicator::Finalize();
    return 0;
}
