//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <boost/test/unit_test.hpp>

#include <exception>
#include <algorithm>
#include "CNTKLibrary.h"
#include <functional>
#include <fstream>
#include <random>
#include "stdafx.h"
#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"
#include <numeric>
#include "CNTKLibraryC.h"

using namespace CNTK;

namespace CNTK
{
namespace Test
{

static unsigned long seed = 1;

inline void PrintTrainingProgressTest(const TrainerPtr trainer, size_t minibatchIdx, size_t outputFrequencyInMinibatches)
{
    if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer->PreviousMinibatchSampleCount() != 0)
    {
        double trainLossValue = trainer->PreviousMinibatchLossAverage();
        double evaluationValue = trainer->PreviousMinibatchEvaluationAverage();
        BOOST_TEST_MESSAGE("Minibatch " << (int) minibatchIdx << ": CrossEntropy loss = " << trainLossValue << ", Evaluation criterion = " << evaluationValue << "\n");
    }
}

void TrainMNISTClassifier(const DeviceDescriptor& device)
{
    const size_t inputDim = 784;
    const size_t numOutputClasses = 10;
    const size_t hiddenLayerDim = 200;

    auto input = InputVariable({inputDim}, DataType::Float, L"features");
    auto scaledInput = ElementTimes(Constant::Scalar(0.00390625f, device), input);
    auto classifierOutput = FullyConnectedDNNLayer(scaledInput, hiddenLayerDim, device, std::bind(Sigmoid, std::placeholders::_1, L""));
    auto outputTimesParam = Parameter(NDArrayView::RandomUniform<float>({numOutputClasses, hiddenLayerDim}, -0.05, 0.05, 1, device));
    auto outputBiasParam = Parameter(NDArrayView::RandomUniform<float>({numOutputClasses}, -0.05, 0.05, 1, device));
    classifierOutput = Plus(outputBiasParam, Times(outputTimesParam, classifierOutput), L"classifierOutput");

    auto labels = InputVariable({numOutputClasses}, DataType::Float, L"labels");
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    ;
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

    // Test save and reload of model
    {
        Variable classifierOutputVar = classifierOutput;
        Variable trainingLossVar = trainingLoss;
        Variable predictionVar = prediction;
        auto combinedNet = Combine({trainingLoss, prediction, classifierOutput}, L"MNISTClassifier");
        SaveAndReloadModel<float>(combinedNet, {&input, &labels, &trainingLossVar, &predictionVar, &classifierOutputVar}, device);

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
    auto minibatchSource = TextFormatMinibatchSource(L"data/Train-28x28_cntk_text.txt", {{featureStreamName, inputDim}, {labelsStreamName, numOutputClasses}}, MinibatchSource::FullDataSweep, false);

    auto featureStreamInfo = minibatchSource->StreamInfo(featureStreamName);
    auto labelStreamInfo = minibatchSource->StreamInfo(labelsStreamName);

    LearningRateSchedule learningRatePerSample = TrainingParameterPerSampleSchedule(0.003125);
    auto trainer = CreateTrainer(classifierOutput, trainingLoss, prediction, {SGDLearner(classifierOutput->Parameters(), learningRatePerSample)});

    size_t outputFrequencyInMinibatches = 20;
    for (size_t i = 0; i < numMinibatchesToTrain; ++i)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        trainer->TrainMinibatch({{input, minibatchData[featureStreamInfo]}, {labels, minibatchData[labelStreamInfo]}}, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
    }
}

template <typename ElementType>
void RunMNISTConvNet(const DeviceDescriptor& device)
{
    const size_t inputDim = 28;
    const size_t numOutputClasses = 10;
    const size_t filterDim = 5;
    const size_t numInputChannels = 1;
    const size_t filterCount = 8;
    const size_t filterCount2 = 16;
    const size_t convStrides = 2;
    const size_t convOutDim = inputDim / convStrides / convStrides;

    auto input = InputVariable({inputDim * inputDim}, AsDataType<ElementType>(), L"features");
    auto scaledInput = ElementTimes(Constant::Scalar((ElementType) 0.00390625f, device), input);
    auto reshapedInput = Reshape(scaledInput, {inputDim, inputDim, numInputChannels});
    auto labelsVar = InputVariable({numOutputClasses}, AsDataType<ElementType>(), L"labels");

    auto convParam = Parameter({filterDim, filterDim, numInputChannels, filterCount}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto convFunc = Convolution(convParam, reshapedInput, {convStrides, convStrides, numInputChannels});
    auto convb = Parameter({1, 1, filterCount}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto relu = LeakyReLU(Plus(convFunc, convb), 0.01);
    auto convParam2 = Parameter({filterDim, filterDim, filterCount, filterCount2}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto convFunc2 = Convolution(convParam2, relu, {convStrides, convStrides, filterCount});
    auto convb2 = Parameter({1, 1, filterCount2}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto relu2 = LeakyReLU(Plus(convFunc2, convb2), 0.01);

    auto outTimesParams = Parameter({numOutputClasses, convOutDim, convOutDim, filterCount2}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto outBiasParams = Parameter({numOutputClasses}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);

    auto output = Plus(outBiasParams, Times(outTimesParams, relu2), L"output");

    auto trainingLoss = CrossEntropyWithSoftmax(output, labelsVar, L"lossFunction");
    auto prediction = ClassificationError(output, labelsVar, L"predictionError");

    // train

    const size_t minibatchSize = 64;
    const size_t numSamplesPerSweep = 60000;
    const size_t numSweepsToTrainWith = 2;
    const size_t numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;

    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";
    auto minibatchSource = TextFormatMinibatchSource(L"data/Train-28x28_cntk_text.txt", {{featureStreamName, inputDim * inputDim}, {labelsStreamName, numOutputClasses}});

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

template <typename ElementType>
void RunSeqMNISTConvNet(const DeviceDescriptor& device)
{
    const size_t inputDim = 28;
    const size_t numOutputClasses = 10;
    const size_t filterDim = 5;
    const size_t numInputChannels = 1;
    const size_t filterCount = 8;
    const size_t filterCount2 = 16;
    const size_t convStrides = 2;
    const size_t convOutDim = inputDim / convStrides / convStrides;

    auto input = InputVariable({inputDim * inputDim}, AsDataType<ElementType>(), L"features");
    auto scaledInput = ElementTimes(Constant::Scalar((ElementType) 0.00390625f, device), input);
    auto reshapedInput = Reshape(scaledInput, {inputDim, inputDim, numInputChannels});

    // pack input into sequences.
    //auto unpackDefaultSeqInput = Sequence::First(reshapedInput);
	// Sequence::First is to tell the code that we have only one sample in each sequence, which is the case anyway so there is no modification or loss here. 
    auto unpackDefaultSeqInput = TransposeAxes(Sequence::First(reshapedInput), Axis(-1), Axis(-2));
    auto packedInput = ToSequence(unpackDefaultSeqInput, Sequence::BroadcastAs(Constant::Scalar((ElementType) inputDim), unpackDefaultSeqInput), L"MNIST ConvSeq Axis", L"ToSequence MNIST ConvSeq Axis");

    wprintf(L"packed Input shape: %s\n", packedInput->Output().Shape().AsString().c_str());

    auto labelsVar = InputVariable({numOutputClasses}, AsDataType<ElementType>(), L"labels");

    auto convParam = Parameter({filterDim, filterDim, numInputChannels, filterCount}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto convFunc = Convolution(convParam, packedInput, {convStrides, convStrides, numInputChannels}, {true}, {true}, true);

    wprintf(L"First Conv output shape: %s\n", convFunc->Output().Shape().AsString().c_str());

	//convFunc = TransposeAxes(Sequence::Unpack(convFunc, 0.0f, true), Axis(-1), Axis(-2));

    wprintf(L"First Conv input for bias shape: %s\n", convFunc->Output().Shape().AsString().c_str());

    auto convb = Parameter({1, filterCount}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto relu = LeakyReLU(Plus(convFunc, convb), 0.01);

	//relu = ToSequence(TransposeAxes(relu, Axis(-1), Axis(-2)), L"Trans for relu", L"relu to seq");

    wprintf(L"First relu output shape: %s\n", relu->Output().Shape().AsString().c_str());

    auto convParam2 = Parameter({filterDim, filterDim, filterCount, filterCount2}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto convFunc2 = Convolution(convParam2, relu, {convStrides, convStrides, filterCount}, {true}, {true}, true);

    wprintf(L"Second Conv output shape: %s\n", convFunc2->Output().Shape().AsString().c_str());

	//convFunc2 = TransposeAxes(Sequence::Unpack(convFunc2, 0.0f, true), Axis(-1), Axis(-2));

    wprintf(L"Second Conv input for bias shape: %s\n", convFunc2->Output().Shape().AsString().c_str());

	auto convb2 = Parameter({1, filterCount2}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto relu2 = LeakyReLU(Plus(convFunc2, convb2), 0.01);

	//relu2 = ToSequence(TransposeAxes(relu2, Axis(-1), Axis(-2)), L"Trans for relu", L"relu2 to seq");

    wprintf(L"Second relu output shape: %s\n", relu2->Output().Shape().AsString().c_str());

    // unpack output and pad with 0
    auto unpackRelu2 = TransposeAxes(Sequence::Unpack(relu2, 0.0f, true), Axis(-1), Axis(-2));
    //unpackRelu2 = Slice(unpackRelu2, {Axis(-2)}, {0}, {7});
    unpackRelu2 = ToSequence(Reshape(unpackRelu2, {convOutDim, convOutDim, filterCount2, 1}), L"MNIST Output Original Seq Axis");

    wprintf(L"Unpacked second relu output shape: %s\n", unpackRelu2->Output().Shape().AsString().c_str());

    auto outTimesParams = Parameter({numOutputClasses, convOutDim, convOutDim, filterCount2}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto outBiasParams = Parameter({numOutputClasses}, AsDataType<ElementType>(), GlorotUniformInitializer(), device);

    auto output = Plus(outBiasParams, Times(outTimesParams, unpackRelu2), L"output");

    wprintf(L"Final output shape: %s\n", output->Output().Shape().AsString().c_str());

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
    auto minibatchSource = TextFormatMinibatchSource(L"data/Train-28x28_cntk_text.txt", {{featureStreamName, inputDim * inputDim}, {labelsStreamName, numOutputClasses}});

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

/// <summary>
/// Print out the evalaution results.
/// </summary>
template <typename ElementType>
void PrintOutput(size_t sampleSize, std::vector<std::vector<ElementType>> outputBuffer)
{
    printf("The batch contains %d sequences.\n", (int) outputBuffer.size());
    for (size_t seqNo = 0; seqNo < outputBuffer.size(); seqNo++)
    {
        auto seq = outputBuffer[seqNo];
        if (seq.size() % sampleSize != 0)
            throw("The number of elements in the sequence is not a multiple of sample size");

        printf("Sequence %d contains %d samples.\n", (int) seqNo, (int) (seq.size() / sampleSize));
        size_t sampleNo = 0;
        for (size_t i = 0; i < seq.size();)
        {
            if (i % sampleSize == 0)
                printf("    sample %d: ", (int) sampleNo);
            printf("%f", seq[i++]);
            if (i % sampleSize == 0)
            {
                printf(".\n");
                sampleNo++;
            }
            else
                printf(", ");
        }
    }
}

template <typename ElementType>
ValuePtr CreateBatchWithVariableSequence(const std::vector<size_t>& sampleSizes, size_t batchSize, const std::vector<size_t>& sequenceSize, const std::vector<ElementType>& batchData, const DeviceDescriptor& device, bool readOnly = false, bool sequential = false)
{
    //auto shapeSize = sampleShape.TotalSize();
    //if (batchData.size() % shapeSize != 0)
    //    InvalidArgument("The number of elements (%zu) in the vector containing batch data must be a multiple of the size (%zu) of the sample shape '%S'.",
    //                    batchData.size(), shapeSize, sampleShape.AsString().c_str());

    if (sequenceSize.size() != batchSize)
        InvalidArgument("The number of sequences (%zu) in the vector containing sequence size must match batch size (%zu)", sequenceSize.size(), batchSize);

    std::vector<NDArrayViewPtr> sequencesView(batchSize);
    size_t curBatchDataIdx = 0;
    for (size_t i = 0; i < batchSize; i++)
    {
        auto sampleShape = NDShape({sampleSizes[i]});
        if (sequential)
            sampleShape = sampleShape.AppendShape({1});
        auto sequenceDataShape = sampleShape.AppendShape({sequenceSize[i]});
        sequencesView[i] = MakeSharedObject<NDArrayView>(sequenceDataShape, batchData.data() + curBatchDataIdx, sampleSizes[i] * sequenceSize[i], DeviceDescriptor::CPUDevice());
        curBatchDataIdx += sampleSizes[i] * sequenceSize[i];
    }

    auto sampleShape = NDShape({sampleSizes[0]});
    if (sequential)
        sampleShape = sampleShape.AppendShape({1});

    // TODO : for the moment try to find a way to append sequences to batch (concatenate batches).
    return Value::Create(sampleShape, sequencesView, {}, device, readOnly, true);
}

template <typename ElementType>
ValuePtr CreateBatchWithVariableSequence(const NDShape& sampleShape, size_t batchSize, const std::vector<size_t>& sequenceSize, const std::vector<ElementType>& batchData, const DeviceDescriptor& device, bool readOnly = false)
{
    auto shapeSize = sampleShape.TotalSize();
    if (batchData.size() % shapeSize != 0)
        InvalidArgument("The number of elements (%zu) in the vector containing batch data must be a multiple of the size (%zu) of the sample shape '%S'.",
                        batchData.size(), shapeSize, sampleShape.AsString().c_str());

    if (sequenceSize.size() != batchSize)
        InvalidArgument("The number of sequences (%zu) in the vector containing sequence size must match batch size (%zu)", sequenceSize.size(), batchSize);

    std::vector<NDArrayViewPtr> sequencesView(batchSize);
    size_t curBatchDataIdx = 0;
    for (size_t i = 0; i < batchSize; i++)
    {
        auto sequenceDataShape = sampleShape.AppendShape({sequenceSize[i]});
        sequencesView[i] = MakeSharedObject<NDArrayView>(sequenceDataShape, batchData.data() + curBatchDataIdx, shapeSize * sequenceSize[i], DeviceDescriptor::CPUDevice());
        curBatchDataIdx += shapeSize * sequenceSize[i];
    }

    return Value::Create(sampleShape, sequencesView, {}, device, readOnly, true);
}

template <typename ElementType>
void Run1DFreeDimConvLayer(const DeviceDescriptor& device)
{
    // TODO   : Test different length of input to try free dimension.
    //		    After that, can be sure what to ask Spandan about free dimension.

    // Result : Sample sizes seem to be required to be same in one sequence.
    //          Hard to find how to create batch of sequences of different sample size. Paramter it receives has one universal shape.

    // TODO 2 : Using sparse?
    //          Normally should work to have different sample size as we use sparse.

    auto input = InputVariable({NDShape::FreeDimension}, AsDataType<ElementType>(), L"features");
    auto convParam = Parameter({3, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto conv = Convolution(convParam, input, {2});
    auto convb = Parameter({1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto relu = LeakyReLU(Plus(conv, convb), 0.01);

    const size_t inputDataSize = 10;
    const size_t channelSize = 1;
    const std::vector<size_t> sequenceSize = {3, 2, 1, 1};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10, 10, 10, 10};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = relu->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    relu->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(inputDataSize / 2, outputData);
}

template <typename ElementType>
void Run1DFreeDimSimpConvLayer(const DeviceDescriptor& device)
{
    auto input = InputVariable({NDShape::FreeDimension}, AsDataType<ElementType>(), L"features");
    auto convParam = Parameter({3, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto conv = Convolution(convParam, input, {2});
    auto convParam2 = Parameter({2, 1}, AsDataType<ElementType>(), (ElementType) 0.5f, device);
    auto conv2 = Convolution(convParam2, conv, {2});

    const size_t inputDataSize = 10;
    const size_t channelSize = 1;
    const std::vector<size_t> sequenceSize = {2, 3, 1, 1};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10, 10, 10, 10};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = conv2->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv2->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(inputDataSize / 2 / 2 + 1, outputData);
}

template <typename ElementType>
void Run1DSeqConvLayer(const DeviceDescriptor& device, bool auto_padding = true)
{
    auto input = InputVariable({1}, AsDataType<ElementType>(), L"features");
    auto convParam = Parameter({3, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    //auto conv = Convolution(convParam, input, {2, 1}, {true}, {true}, true);
	// test auto fixing filter shape
    auto conv = Convolution(convParam, input, {2,}, {true}, {auto_padding}, true);
    auto convb = Parameter({1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto relu = LeakyReLU(Plus(conv, convb), 0.01);

    const std::vector<size_t> sequenceSize = {5, 10, 8, 4};
    const size_t batchSize = sequenceSize.size();
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto sampleShape = NDShape({1});

    auto inputVal = CreateBatchWithVariableSequence(sampleShape, batchSize, sequenceSize, inputData, device);
    auto outputVar = relu->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    relu->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(1, outputData);
}

template <typename ElementType>
void RunConvSeqByUnpack_byhand(const DeviceDescriptor& device)
{
    const size_t numFilters = 2;

    auto input = InputVariable({10, 1}, AsDataType<ElementType>(), L"features");
    auto convParam = Parameter({3, 2, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto unpackInputOutputs = Sequence::Unpack(input, (ElementType) 0.0f, false, L"unpack input");
    auto unpackInput = unpackInputOutputs->Outputs()[0];
    auto unpackInputMask = unpackInputOutputs->Outputs()[1]; // TODO : can we compute output mask by input mask, and convert output back to sequence using mask?
    auto transposeInput = TransposeAxes(unpackInput, Axis(-1), Axis(-2), L"transpose axis input");
    auto conv = Convolution(convParam, transposeInput, {2, 2}); //auto conv = Convolution(convParam, input, {2, 2}, /*sharing = */ {true}, /*autoPadding = */ {true}, /*sequential = */ true);

    auto unpackInputMaskReduceSum = ReduceSum(unpackInputMask, Axis(-1));
    auto seqKernelSize = convParam.Shape()[convParam.Shape().Rank() - 2];
    auto convOutputSeqSize = Ceil(ElementDivide(unpackInputMaskReduceSum, Constant::Scalar((ElementType) seqKernelSize)));

    auto transOut = TransposeAxes(conv, Axis(-1), Axis(-2), L"transpose axis output");

    auto resPack = ToSequence(transOut, convOutputSeqSize, L"pack output axis", L"pack output"); // provide sequence length as parameter.

    const size_t channelSize = 1;
    const std::vector<size_t> sequenceSize = {4, 5, 2, 2};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10, 10, 10, 10};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device, false, true);
    auto outputVar = resPack->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    resPack->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(5, outputData);
}

template <typename ElementType>
void RunConvSeqByUnpack(const DeviceDescriptor& device)
{
    const size_t numFilters = 2;

    auto input = InputVariable({10, 1}, AsDataType<ElementType>(), L"features");
    auto convParam = Parameter({3, 2, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto conv = Convolution(convParam, input, {2, 2}, /*sharing = */ {true}, /*autoPadding = */ {true}, /*sequential = */ true);

    const size_t channelSize = 1;
    const std::vector<size_t> sequenceSize = {4, 5, 2, 2};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10, 10, 10, 10};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device, false, true);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(5, outputData);
}

template <typename ElementType>
void RunConvSeqByUnpackTestMaskReduce(const DeviceDescriptor& device)
{
    const size_t numFilters = 2;

    auto input = InputVariable({10, 1}, AsDataType<ElementType>(), L"features");
    auto convParam = Parameter({3, 2, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto unpackInputOutputs = Sequence::Unpack(input, (ElementType) 0.0f, false, L"unpack input");
    auto unpackInputMask = unpackInputOutputs->Outputs()[1]; // TODO : can we compute output mask by input mask, and convert output back to sequence using mask?

    auto unpackInputMaskReduceSum = ReduceSum(unpackInputMask, Axis(-1));
    auto seqKernelSize = convParam.Shape()[convParam.Shape().Rank() - 2];

    auto convOutputSeqSize = Ceil(ElementDivide(unpackInputMaskReduceSum, Constant::Scalar((ElementType) seqKernelSize)));

    const size_t channelSize = 1;
    const std::vector<size_t> sequenceSize = {4, 5, 2, 2};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10, 10, 10, 10};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device, false, true);
    auto outputVar = convOutputSeqSize->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    convOutputSeqSize->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(1, outputData);
}

template <typename ElementType>
void RunConvMatchResSeqByUnpack(const DeviceDescriptor& device)
{
    const size_t numFilters = 2;

    auto input = InputVariable({5, 10, 1}, AsDataType<ElementType>(), L"features");
    auto convParam = Parameter({3, 2, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    auto conv = Convolution(convParam, input, {2, 2});

    const size_t channelSize = 1;
    const std::vector<size_t> sequenceSize = {4, 5, 2, 2};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {10, 10, 10, 10};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * 5;
    }

    std::vector<ElementType> inputData(dataSize);
    size_t k = 0;
    size_t l = 0;
    for (size_t i = 0; i < sequenceSize.size(); ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t z = 0; z < 10; ++z)
            {
                if (j >= sequenceSize[i])
                    inputData[k] = static_cast<ElementType>(0);
                else
                {
                    inputData[k] = static_cast<ElementType>(l % 255);
                    l++;
                }
                k++;
            }
        }
    }

    //const std::vector<size_t> sampleSizes_ = {50, 50, 50, 50};

    auto inputVal = CreateBatchWithVariableSequence(NDShape({5, 10, 1}), batchSize, {1, 1, 1, 1}, inputData, device);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(15, outputData);
}


template <typename ElementType>
void TestRNNDataReader(const DeviceDescriptor& device)
{
    using namespace std::placeholders;


}


template <typename ElementType>
void RunConvRankTests1(const DeviceDescriptor& device)
{
    auto input_ = InputVariable({12}, AsDataType<ElementType>());
    auto input = Reshape(input_, {4, 3, 1});
    auto params = Parameter({3, 2, 1}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
	// requires kernel dim <= input dim. 

	auto conv = Convolution(params, input, {2, 2});

	const size_t inputDataSize = 12;
    const std::vector<size_t> sequenceSize = {2, 3};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {inputDataSize, inputDataSize};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input_, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(4, outputData);
}


template <typename ElementType>
void RunConvRankTests2(const DeviceDescriptor& device)
{
    auto input_ = InputVariable({36}, AsDataType<ElementType>());
    auto input = Reshape(input_, {4, 3, 3});
    auto params = Parameter({3, 3, 2, 1, 4}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    // requires kernel dim >= input dim .... 

    auto conv = Convolution(params, input, {2, 2});

    const size_t inputDataSize = 36;
    const std::vector<size_t> sequenceSize = {2, 3};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {inputDataSize, inputDataSize};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input_, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(32, outputData);
}


template <typename ElementType>
void RunConvRankTests3(const DeviceDescriptor& device)
{
    auto input_ = InputVariable({6}, AsDataType<ElementType>());
    auto input = Reshape(input_, {6});
    auto params = Parameter({2,4}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    // requires kernel dim >= input dim ....

    auto conv = Convolution(params, input, {5});

    const size_t inputDataSize = 6;
    const std::vector<size_t> sequenceSize = {2, 3};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {inputDataSize, inputDataSize};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input_, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(8, outputData);
}

template <typename ElementType>
void RunConvRankTests4(const DeviceDescriptor& device)
{
    auto input_ = InputVariable({9}, AsDataType<ElementType>());
    auto input = Reshape(input_, {3,3});
    auto params = Parameter({2,3, 4}, AsDataType<ElementType>(), (ElementType) 1.0f, device);
    // requires kernel dim >= input dim ....

    auto conv = Convolution(params, input, {2});

    const size_t inputDataSize = 9;
    const std::vector<size_t> sequenceSize = {2, 3};
    const size_t batchSize = sequenceSize.size();
    const std::vector<size_t> sampleSizes = {inputDataSize, inputDataSize};
    size_t dataSize = 0;
    for (size_t i = 0; i < sequenceSize.size(); i++)
    {
        dataSize += sampleSizes[i] * sequenceSize[i];
    }

    std::vector<ElementType> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i)
    {
        inputData[i] = static_cast<ElementType>(i % 255);
    }

    auto inputVal = CreateBatchWithVariableSequence(sampleSizes, batchSize, sequenceSize, inputData, device);
    auto outputVar = conv->Output();
    std::unordered_map<Variable, ValuePtr> inputDataMap = {{input_, inputVal}};

    std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

    conv->Evaluate(inputDataMap, outputDataMap, device);

    auto outputVal = outputDataMap[outputVar];
    std::vector<std::vector<ElementType>> outputData;

    Internal::SetAutomaticUnpackingOfPackedValues(false);
    outputVal->CopyVariableValueTo(outputVar, outputData);
    Internal::SetAutomaticUnpackingOfPackedValues(true);

    PrintOutput<ElementType>(16, outputData);
}



template <typename ElementType>
void TestConvolutionNetworkSequentialAxisCreation(const DeviceDescriptor& device, bool testSaveAndReload)
{
    if (testSaveAndReload)
    {
        ;
    }

	// testing now
    //RunMNISTConvNet<ElementType>(device);
    //RunSeqMNISTConvNet<ElementType>(device);

    //Run1DFreeDimConvLayer<ElementType>(device);

    //Run1DFreeDimSimpConvLayer<ElementType>(device);

	// testing now
    Run1DSeqConvLayer<ElementType>(device);
    Run1DSeqConvLayer<ElementType>(device, false);

    //RunConvSeqByUnpack<ElementType>(device);
    //RunConvSeqByUnpack_byhand<ElementType>(device);
    //RunConvSeqByUnpackTestMaskReduce<ElementType>(device);

    //RunConvMatchResSeqByUnpack<ElementType>(device);

	// Testing different input, kernel, stride rank settings
    //RunConvRankTests1<ElementType>(device);
    //RunConvRankTests2<ElementType>(device);
    //RunConvRankTests3<ElementType>(device);
    //RunConvRankTests4<ElementType>(device);

    BOOST_TEST(1 == 1, "Placeholder test.");
}

BOOST_AUTO_TEST_SUITE(ConvolutionFunctionSuite)

BOOST_AUTO_TEST_CASE(ConvolutionNetworkSequentialAxisCreationInCPU)
{
    if (ShouldRunOnCpu())
        TestConvolutionNetworkSequentialAxisCreation<float>(DeviceDescriptor::CPUDevice(), false);
}

BOOST_AUTO_TEST_CASE(ConvolutionNetworkSequentialAxisCreationInGPU)
{
    if (ShouldRunOnGpu())
        TestConvolutionNetworkSequentialAxisCreation<float>(DeviceDescriptor::GPUDevice(0), false);
}

BOOST_AUTO_TEST_SUITE_END()
}
}