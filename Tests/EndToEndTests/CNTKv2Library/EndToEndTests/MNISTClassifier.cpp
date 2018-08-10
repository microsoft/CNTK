//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

using namespace std::placeholders;

void TrainSimpleFeedForwardClassifier(const DeviceDescriptor& device)
{
    const size_t inputDim = 2;
    const size_t numOutputClasses = 2;
    const size_t hiddenLayerDim = 50;
    const size_t numHiddenLayers = 2;

    const size_t minibatchSize = 50;
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

    auto nonLinearity = std::bind(Sigmoid, _1, L"");
    auto input = InputVariable({ inputDim }, DataType::Float, L"features");
    auto normalizedinput = PerDimMeanVarianceNormalize(input, inputMeansAndInvStdDevs[featureStreamInfo].first, inputMeansAndInvStdDevs[featureStreamInfo].second);
    auto classifierOutput = FullyConnectedDNNLayer(normalizedinput, hiddenLayerDim, device, nonLinearity);
    for (size_t i = 1; i < numHiddenLayers; ++i)
        classifierOutput = FullyConnectedDNNLayer(classifierOutput, hiddenLayerDim, device, nonLinearity);

    auto outputTimesParam = Parameter(NDArrayView::RandomUniform<float>({ numOutputClasses, hiddenLayerDim }, -0.05, 0.05, 1, device));
    auto outputBiasParam = Parameter(NDArrayView::RandomUniform<float>({ numOutputClasses }, -0.05, 0.05, 1, device));
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
        SaveAndReloadModel<float>(combinedNet, { &input, &labels, &trainingLossVar, &predictionVar, &classifierOutputVar }, device);

        classifierOutput = classifierOutputVar;
        trainingLoss = trainingLossVar;
        prediction = predictionVar;
    }

    LearningRateSchedule learningRatePerSample = TrainingParameterPerSampleSchedule(0.02);
    minibatchSource = TextFormatMinibatchSource(L"SimpleDataTrain_cntk_text.txt", { { L"features", inputDim }, { L"labels", numOutputClasses } });
    auto trainer = CreateTrainer(classifierOutput, trainingLoss, prediction, { SGDLearner(classifierOutput->Parameters(), learningRatePerSample) });
    size_t outputFrequencyInMinibatches = 20;
    size_t trainingCheckpointFrequency = 100;
    for (size_t i = 0; i < numMinibatchesToTrain; ++i)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        trainer->TrainMinibatch({ { input, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);

        if ((i % trainingCheckpointFrequency) == (trainingCheckpointFrequency - 1))
        {
            const wchar_t* ckpName = L"feedForward.net";
            trainer->SaveCheckpoint(ckpName);
            trainer->RestoreFromCheckpoint(ckpName);
        }
    }
}

void TrainMNISTClassifier(const DeviceDescriptor& device)
{
    const size_t inputDim = 784;
    const size_t numOutputClasses = 10;
    const size_t hiddenLayerDim = 200;

    auto input = InputVariable({ inputDim }, DataType::Float, L"features");
    auto scaledInput = ElementTimes(Constant::Scalar(0.00390625f, device), input);
    auto classifierOutput = FullyConnectedDNNLayer(scaledInput, hiddenLayerDim, device, std::bind(Sigmoid, _1, L""));
    auto outputTimesParam = Parameter(NDArrayView::RandomUniform<float>({ numOutputClasses, hiddenLayerDim }, -0.05, 0.05, 1, device));
    auto outputBiasParam = Parameter(NDArrayView::RandomUniform<float>({ numOutputClasses }, -0.05, 0.05, 1, device));
    classifierOutput = Plus(outputBiasParam, Times(outputTimesParam, classifierOutput), L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels");
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");;
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

    // Test save and reload of model
    {
        Variable classifierOutputVar = classifierOutput;
        Variable trainingLossVar = trainingLoss;
        Variable predictionVar = prediction;
        auto combinedNet = Combine({ trainingLoss, prediction, classifierOutput }, L"MNISTClassifier");
        SaveAndReloadModel<float>(combinedNet, { &input, &labels, &trainingLossVar, &predictionVar, &classifierOutputVar }, device);

        classifierOutput = classifierOutputVar;
        trainingLoss = trainingLossVar;
        prediction = predictionVar;
    }

    const size_t minibatchSize = 64;
    const size_t numSamplesPerSweep = 60000;
    const size_t numSweepsToTrainWith = 2;
    const size_t numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;

    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";
    auto minibatchSource = TextFormatMinibatchSource(L"Train-28x28_cntk_text.txt", { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } });

    auto featureStreamInfo = minibatchSource->StreamInfo(featureStreamName);
    auto labelStreamInfo = minibatchSource->StreamInfo(labelsStreamName);

    LearningRateSchedule learningRatePerSample = TrainingParameterPerSampleSchedule(0.003125);
    auto trainer = CreateTrainer(classifierOutput, trainingLoss, prediction, { SGDLearner(classifierOutput->Parameters(), learningRatePerSample) });

    size_t outputFrequencyInMinibatches = 20;
    for (size_t i = 0; i < numMinibatchesToTrain; ++i)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        trainer->TrainMinibatch({ { input, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
    }
}

void TrainMNISTSeqClassifier(const DeviceDescriptor& device)
{
    const size_t inputDim = 28;
    const size_t numOutputClasses = 10;
    const size_t filterDim = 5;
    const size_t numInputChannels = 1;
    const size_t filterCount = 8;
    const size_t filterCount2 = 16;
    const size_t convStrides = 2;
    const size_t convOutDim = inputDim / convStrides / convStrides;

    auto input = InputVariable({inputDim * inputDim}, AsDataType<float>(), L"features");
    auto scaledInput = ElementTimes(Constant::Scalar((float) 0.00390625f, device), input);
    auto reshapedInput = Reshape(scaledInput, {inputDim, inputDim, numInputChannels});

    auto unpackDefaultSeqInput = TransposeAxes(Sequence::First(reshapedInput), Axis(-1), Axis(-2));
    auto packedInput = ToSequence(unpackDefaultSeqInput, Sequence::BroadcastAs(Constant::Scalar((float) inputDim), unpackDefaultSeqInput), L"MNIST ConvSeq Axis", L"ToSequence MNIST ConvSeq Axis");

    auto labelsVar = InputVariable({numOutputClasses}, AsDataType<float>(), L"labels");

    auto convParam = Parameter({filterDim, filterDim, numInputChannels, filterCount}, AsDataType<float>(), GlorotUniformInitializer(), device);
    auto convFunc = Convolution(convParam, packedInput, {convStrides, convStrides, numInputChannels}, {true}, {true, true, false}, {1}, 1, 1, 0, true);

    auto convb = Parameter({1, filterCount}, AsDataType<float>(), GlorotUniformInitializer(), device);
    auto relu = LeakyReLU(Plus(convFunc, convb), 0.01);

    auto convParam2 = Parameter({filterDim, filterDim, filterCount, filterCount2}, AsDataType<float>(), GlorotUniformInitializer(), device);
    auto convFunc2 = Convolution(convParam2, relu, {convStrides, convStrides, filterCount}, {true}, {true, true, false}, { 1 }, 1, 1, 0, true);

    auto convb2 = Parameter({1, filterCount2}, AsDataType<float>(), GlorotUniformInitializer(), device);
    auto relu2 = LeakyReLU(Plus(convFunc2, convb2), 0.01);

    auto unpackRelu2 = TransposeAxes(Sequence::Unpack(relu2, 0.0f, true), Axis(-1), Axis(-2));
    unpackRelu2 = ToSequence(Reshape(unpackRelu2, {convOutDim, convOutDim, filterCount2, 1}), L"MNIST Output Original Seq Axis");

    auto outTimesParams = Parameter({numOutputClasses, convOutDim, convOutDim, filterCount2}, AsDataType<float>(), GlorotUniformInitializer(), device);
    auto outBiasParams = Parameter({numOutputClasses}, AsDataType<float>(), GlorotUniformInitializer(), device);

    auto output = Plus(outBiasParams, Times(outTimesParams, unpackRelu2), L"output");

    auto labelsVarCompat = Sequence::BroadcastAs(labelsVar, output);

    auto trainingLoss = CrossEntropyWithSoftmax(output, labelsVarCompat, L"lossFunction");
    auto prediction = ClassificationError(output, labelsVarCompat, L"predictionError");

    // train

    const size_t minibatchSize = 64;
    const size_t numSamplesPerSweep = 60000;
    const size_t numSweepsToTrainWith = 2;
    const size_t numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;

    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";
    auto minibatchSource = TextFormatMinibatchSource(L"Train-28x28_cntk_text.txt", {{featureStreamName, inputDim * inputDim}, {labelsStreamName, numOutputClasses}});

    auto featureStreamInfo = minibatchSource->StreamInfo(featureStreamName);
    auto labelStreamInfo = minibatchSource->StreamInfo(labelsStreamName);

    LearningRateSchedule learningRatePerSample = TrainingParameterPerSampleSchedule(0.003125);
    auto trainer = CreateTrainer(output, trainingLoss, prediction, {SGDLearner(output->Parameters(), learningRatePerSample)});

    size_t outputFrequencyInMinibatches = 20;
    for (size_t i = 0; i < numMinibatchesToTrain; ++i)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        trainer->TrainMinibatch({{input, minibatchData[featureStreamInfo]}, {labelsVar, minibatchData[labelStreamInfo]}}, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
    }
}

void MNISTClassifierTests()
{
    fprintf(stderr, "\nMNISTClassifierTests..\n");

    if (ShouldRunOnCpu())
        TrainSimpleFeedForwardClassifier(DeviceDescriptor::CPUDevice());
    if (ShouldRunOnGpu())
    {
        TrainMNISTClassifier(DeviceDescriptor::GPUDevice(0));
        TrainMNISTSeqClassifier(DeviceDescriptor::GPUDevice(0));
    }
}
