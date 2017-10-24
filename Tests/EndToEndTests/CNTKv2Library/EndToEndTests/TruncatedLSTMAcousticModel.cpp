//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

using namespace std::placeholders;

static FunctionPtr LSTMAcousticSequenceClassifierNet(const Variable& input, size_t numOutputClasses, size_t LSTMDim, size_t cellDim, size_t numLSTMs, const DeviceDescriptor& device, const std::wstring& outputName)
{
    auto pastValueRecurrenceHook = [](const Variable& x) { return PastValue(x); };
    FunctionPtr r = input;
    for (size_t i = 0; i < numLSTMs; ++i)
        r = LSTMPComponentWithSelfStabilization<float>(r, { LSTMDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;

    return FullyConnectedLinearLayer(r, numOutputClasses, device, outputName);
}

void TrainTruncatedLSTMAcousticModelClassifier(const DeviceDescriptor& device, bool testSaveAndReLoad)
{
    const size_t baseFeaturesDim = 33;
    const size_t cellDim = 1024;
    const size_t hiddenDim = 256;
    const size_t numOutputClasses = 132;
    const size_t numLSTMLayers = 3;

    auto features = InputVariable({ baseFeaturesDim }, DataType::Float, L"features");
    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels");

    const size_t numSamplesForFeatureStatistics = MinibatchSource::FullDataSweep;
    
    auto config = GetHTKMinibatchSourceConfig(baseFeaturesDim, numOutputClasses, numSamplesForFeatureStatistics, false);
    config.isFrameModeEnabled = true;
    auto minibatchSource = CreateCompositeMinibatchSource(config);

    auto featureStreamInfo = minibatchSource->StreamInfo(features);
    std::unordered_map<StreamInformation, std::pair<NDArrayViewPtr, NDArrayViewPtr>> featureMeansAndInvStdDevs = { { featureStreamInfo, { nullptr, nullptr } } };
    ComputeInputPerDimMeansAndInvStdDevs(minibatchSource, featureMeansAndInvStdDevs);

    auto normalizedFeatures = PerDimMeanVarianceNormalize(features, featureMeansAndInvStdDevs[featureStreamInfo].first, featureMeansAndInvStdDevs[featureStreamInfo].second);
    auto classifierOutput = LSTMAcousticSequenceClassifierNet(normalizedFeatures, numOutputClasses, hiddenDim, cellDim, numLSTMLayers, device, L"classifierOutput");

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

    const size_t numTrainingSamples = 81920;
    const size_t truncationLength = 20;

    config = GetHTKMinibatchSourceConfig(baseFeaturesDim, numOutputClasses, numTrainingSamples);
    config.truncationLength = truncationLength;
    minibatchSource = CreateCompositeMinibatchSource(config);

    const size_t numberParallelSequencesPerMB1 = 16;
    const size_t numberParallelSequencesPerMB2 = 32;
    const size_t numMinibatchesToChangeMBSizeAfter = 5;

    featureStreamInfo = minibatchSource->StreamInfo(features);
    auto labelStreamInfo = minibatchSource->StreamInfo(labels);

    LearningRateSchedule learningRatePerSample = TrainingParameterPerSampleSchedule(0.000781);
    MomentumSchedule momentumTimeConstant = MomentumAsTimeConstantSchedule(6074);
    auto learner = MomentumSGDLearner(classifierOutput->Parameters(), learningRatePerSample, momentumTimeConstant, /*unitGainMomentum = */true);
    auto trainer = CreateTrainer(classifierOutput, trainingLoss, prediction, {learner});

    size_t outputFrequencyInMinibatches = 1;
    for (size_t i = 0; true; i++)
    {
        const size_t numberParallelSequencesPerMB = (i >= numMinibatchesToChangeMBSizeAfter) ? numberParallelSequencesPerMB2 : numberParallelSequencesPerMB1;
        const size_t minibatchSize = truncationLength * numberParallelSequencesPerMB;

        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;

        // Make sure our truncation length setting was honored
        auto actualMaxSequenceLength = minibatchData[featureStreamInfo].data->Shape()[featureStreamInfo.m_sampleLayout.Rank()];
        if (actualMaxSequenceLength != truncationLength)
            ReportFailure("Actual max sequence length (%d) in minibatch data does not equal specified truncation length (%d)", (int)actualMaxSequenceLength, (int)truncationLength);

        trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
    }
}

void TrainTruncatedLSTMAcousticModelClassifier()
{
    fprintf(stderr, "\nTrainTruncatedLSTMAcousticModelClassifier..\n");

    if (ShouldRunOnGpu())
        TrainTruncatedLSTMAcousticModelClassifier(DeviceDescriptor::GPUDevice(0), true);

    if (ShouldRunOnCpu())
        TrainTruncatedLSTMAcousticModelClassifier(DeviceDescriptor::CPUDevice(), false);
}
