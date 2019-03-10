//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"
#include <numeric>
#include "CNTKLibraryC.h"

using namespace CNTK;

namespace CNTK { namespace Test {

static unsigned long seed = 1;

template <typename ElementType>
FunctionPtr LSTMNet(Variable features, size_t cellDim, size_t hiddenDim, size_t numOutputClasses, size_t numLSTMLayers, const DeviceDescriptor& device, const std::wstring& outputName)
{
    using namespace std::placeholders;

    assert(numLSTMLayers >= 1);
    FunctionPtr classifierRoot = features;
    auto pastValueRecurrenceHook = [](const Variable& x) { return PastValue(x); };
    for (size_t i = 0; i < numLSTMLayers; ++i) {
        classifierRoot = LSTMPComponentWithSelfStabilization<ElementType>(classifierRoot, { hiddenDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;
    }

    auto W = Parameter(NDArrayView::RandomUniform<ElementType>({ numOutputClasses, hiddenDim }, -0.5, 0.5, seed++, device));
    auto b = Parameter({ numOutputClasses }, (ElementType)0.0, device);

    auto sW = Parameter({}, (ElementType)0.0, device);
    auto expsW = Exp(sW);

    return Plus(Times(W, ElementTimes(expsW, classifierRoot)), b, outputName);
}

template <typename ElementType>
void TestRecurrentNetworkCreation(const DeviceDescriptor& device, bool testSaveAndReLoad)
{
    const size_t inputDim = 937;
    const size_t numLSTMLayers = 3;
    const size_t cellDim = 1024;
    const size_t hiddenDim = 512;
    const size_t numOutputClasses = 9304;

    auto features = InputVariable({ inputDim }, AsDataType<ElementType>(), L"features");
    auto classifierOutput = LSTMNet<ElementType>(features, cellDim, hiddenDim, numOutputClasses, numLSTMLayers, device, L"classifierOutput");

    auto labelsVar = InputVariable({ numOutputClasses }, AsDataType<ElementType>(), L"labels");
    auto trainingLoss = ReduceSum(CrossEntropyWithSoftmax(classifierOutput, labelsVar), Axis::AllAxes(), L"lossFunction");
    auto prediction = ReduceSum(ClassificationError(classifierOutput, labelsVar), Axis::AllAxes(), L"classificationError");

    auto LSTMClassifier = Combine({ trainingLoss, prediction, classifierOutput }, L"LSTMClassifier");

    BOOST_TEST(LSTMClassifier->Arguments().size() == 2, "Function does not have expected Argument count");

    BOOST_TEST(LSTMClassifier->Outputs().size() == 3, "Function does not have expected Output count");

    const size_t numParameterVariablesPerLSTMLayer = 20;
    BOOST_TEST(LSTMClassifier->Parameters().size() == ((numLSTMLayers * numParameterVariablesPerLSTMLayer) + 3), "Function does not have expected Parameter count");

    if (testSaveAndReLoad)
    {
        Variable classifierOutputVar = classifierOutput;
        Variable trainingLossVar = trainingLoss;
        Variable predictionVar = prediction;

        SaveAndReloadModel<ElementType>(LSTMClassifier, { &features, &labelsVar, &trainingLossVar, &predictionVar, &classifierOutputVar }, device);

        classifierOutput = classifierOutputVar;
        trainingLoss = trainingLossVar;
        prediction = predictionVar;
    }

    // Run Forward and backward a few times
    size_t iterationCount = 3;
    unsigned int randSeed = 2;
    srand(randSeed);
    size_t numSequences = 7;
    size_t maxAllowedSequenceLength = 11;
    for (size_t i = 0; i < iterationCount; ++i)
    {
        std::vector<size_t> sequenceLengths = GenerateSequenceLengths(numSequences, maxAllowedSequenceLength);
        
        ValuePtr inputValue = GenerateSequences<ElementType>(sequenceLengths, { inputDim }, device, false);

        std::vector<std::vector<ElementType>> labelsData;
        for (size_t i2 = 0; i2 < numSequences; ++i2)
        {
            std::vector<ElementType> currentSequence(numOutputClasses * sequenceLengths[i2]);
            for (size_t j = 0; j < sequenceLengths[i2]; ++j)
                currentSequence[(j * numOutputClasses) + (rand() % numOutputClasses)] = 1;

            labelsData.push_back(std::move(currentSequence));
        }

        ValuePtr labelValue = Value::Create({ numOutputClasses }, labelsData, device, true);

        ValuePtr outputValue, predictionErrorValue;
        std::unordered_map<Variable, ValuePtr> outputs = { { classifierOutput, outputValue }, { prediction, predictionErrorValue } };
        auto backpropState = LSTMClassifier->Forward({ { features, inputValue }, { labelsVar, labelValue } }, outputs, device, { trainingLoss });

        // Perform backprop
        NDShape outputShape = trainingLoss->Output().Shape();
        std::vector<ElementType> rootGradientsData(outputShape.TotalSize(), 1);
        ValuePtr rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, rootGradientsData.data(), rootGradientsData.size(), DeviceDescriptor::CPUDevice(), true));
        std::unordered_map<Variable, ValuePtr> paramGradients;
        auto allParams = LSTMClassifier->Parameters();
        for (auto iter = allParams.begin(); iter != allParams.end(); ++iter)
            paramGradients[*iter] = nullptr;

        LSTMClassifier->Backward(backpropState, { { trainingLoss, rootGradientValue } }, paramGradients);
    }
}

template <typename ElementType>
void TestSimpleRecurrence(size_t inputDim,
                          size_t outputDim,
                          size_t maxAllowedSequenceLength,
                          size_t numSequences,
                          const DeviceDescriptor& device,
                          bool testSaveAndReLoad,
                          size_t numIterations,
                          bool useFutureValue,
                          bool useSparseInputs,
                          bool useOneHotSparseInputs = false,
                          unsigned int seed = 1)
{
    BOOST_TEST((!useOneHotSparseInputs || useSparseInputs), "useOneHotSparseInputs option can only be true when useSparseInputs is true");

    Parameter timesParam(MakeSharedObject<NDArrayView>((ElementType)0.5, NDShape({ outputDim, inputDim }), device), L"timesParameters");
    Parameter plusParam(MakeSharedObject<NDArrayView>((ElementType)0.1, std::initializer_list<size_t>({ outputDim }), device), L"plusParameters");

    auto inputVar = InputVariable({ inputDim }, useSparseInputs, AsDataType<ElementType>(), true, L"input");

    auto placeholder = PlaceholderVariable(std::initializer_list<size_t>({ outputDim }));
    auto plusOutput = Plus(plusParam, Plus(placeholder, Times(timesParam, inputVar)), L"plusOutput");
    FunctionPtr placeholderReplacement;
    if (useFutureValue)
        placeholderReplacement = FutureValue(plusOutput);
    else
        placeholderReplacement = PastValue(plusOutput);

    plusOutput = plusOutput->ReplacePlaceholders({ { placeholder, placeholderReplacement } });

    auto reducedOutput = ReduceSum(plusOutput, Axis::AllAxes(), L"sum");

    if (testSaveAndReLoad)
    {
        Variable plusOutputVar = plusOutput;
        Variable reducedOutputVar = reducedOutput;

        auto rootFunc = Combine({ reducedOutput, plusOutput });
        SaveAndReloadModel<ElementType>(rootFunc, { &inputVar, &timesParam, &plusParam, &plusOutputVar, &reducedOutputVar }, device);

        plusOutput = plusOutputVar;
        reducedOutput = reducedOutputVar;
    }

    srand(seed);
    for (size_t iterIdx = 0; iterIdx < numIterations; ++iterIdx)
    {
        std::vector<size_t> sequenceLengths(numSequences);
        size_t maxActualSequenceLength = 0;
        for (size_t i = 0; i < numSequences; ++i)
        {
            sequenceLengths[i] = (rand() % maxAllowedSequenceLength) + 1;
            if (sequenceLengths[i] > maxActualSequenceLength)
                maxActualSequenceLength = sequenceLengths[i];
        }

        NDShape inputShape = inputVar.Shape().AppendShape({ maxActualSequenceLength, numSequences });
        ValuePtr inputValue;
        size_t totalNumInputSamples = maxActualSequenceLength * numSequences;
        std::vector<ElementType> inputData(inputDim * totalNumInputSamples, useSparseInputs ? 0 : std::numeric_limits<ElementType>::quiet_NaN());
        if (useOneHotSparseInputs)
        {
            std::vector<std::vector<size_t>> oneHotSequences;
            for (size_t i = 0; i < numSequences; ++i)
            {
                std::vector<size_t> currentSequence(sequenceLengths[i]);
                for (size_t j = 0; j < sequenceLengths[i]; ++j)
                {
                    size_t hotRowIndex = rand() % inputDim;
                    currentSequence[j] = hotRowIndex;
                    size_t sampleIdx = (i * maxActualSequenceLength) + j;
                    inputData[(sampleIdx * inputDim) + hotRowIndex] = 1;
                }

                oneHotSequences.push_back(std::move(currentSequence));
            }

            inputValue = Value::Create<ElementType>({ inputDim }, oneHotSequences, DeviceDescriptor::CPUDevice(), true);
        }
        else
        {
            for (size_t i = 0; i < numSequences; ++i)
            {
                for (size_t j = 0; j < maxActualSequenceLength; ++j)
                {
                    size_t sampleIdx = (i * maxActualSequenceLength) + j;
                    size_t maxNumberOfNonZeroValuesPerSparseInputSample = std::max<size_t>(inputDim / 200, 1);
                    size_t numActualValuesWritten = 0;
                    for (size_t k = 0; k < inputDim; ++k)
                    {
                        if ((j < sequenceLengths[i]) && (!useSparseInputs || ((numActualValuesWritten < maxNumberOfNonZeroValuesPerSparseInputSample) && ((rand() % inputDim) < maxNumberOfNonZeroValuesPerSparseInputSample))))
                        {
                            numActualValuesWritten++;
                            inputData[(sampleIdx * inputDim) + k] = ((ElementType)rand()) / RAND_MAX;
                        }
                    }
                }
            }

            NDArrayViewPtr inputValueData = MakeSharedObject<NDArrayView>(inputShape, inputData.data(), inputData.size(), DeviceDescriptor::CPUDevice(), true);
            if (useSparseInputs)
            {
                NDArrayViewPtr sparseInputValueData = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), StorageFormat::SparseCSC, inputShape, DeviceDescriptor::CPUDevice());
                sparseInputValueData->CopyFrom(*inputValueData);
                inputValueData = sparseInputValueData->Alias(true);
            }

            NDMaskPtr inputMask = MakeSharedObject<NDMask>(NDShape({ maxActualSequenceLength, numSequences }));
            for (size_t i = 0; i < numSequences; ++i)
            {
                inputMask->MarkSequenceBegin({0, i});
                inputMask->InvalidateSection({ sequenceLengths[i], i }, { NDShape::InferredDimension, 1 });
            }

            inputValue = MakeSharedObject<Value>(inputValueData, inputMask);
        }

        NDShape reducedOutputShape = {};
        std::vector<ElementType> reducedOutputData(reducedOutputShape.TotalSize());
        ValuePtr reducedOutputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(reducedOutputShape, reducedOutputData.data(), reducedOutputData.size(), DeviceDescriptor::CPUDevice(), false));

        NDShape plusOutputShape = plusOutput->Output().Shape().AppendShape({ maxActualSequenceLength, numSequences });
        std::vector<ElementType> plusOutputData(plusOutputShape.TotalSize(), 0);
        ValuePtr plusOutputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(plusOutputShape, plusOutputData.data(), plusOutputData.size(), DeviceDescriptor::CPUDevice(), false), MakeSharedObject<NDMask>(inputValue->Mask()->Shape()));

        std::unordered_map<Variable, ValuePtr> outputs = { { reducedOutput, reducedOutputValue }, { plusOutput, plusOutputValue } };
        auto backpropState = reducedOutput->Forward({ { inputVar, inputValue } }, outputs, device, { plusOutput });

        // Perform backprop
        std::vector<ElementType> rootGradientsData(plusOutputShape.TotalSize(), std::numeric_limits<ElementType>::quiet_NaN());
        for (size_t i = 0; i < numSequences; ++i)
        {
            for (size_t j = 0; j < maxActualSequenceLength; ++j)
            {
                size_t sampleIdx = (i * maxActualSequenceLength) + j;
                for (size_t k = 0; k < outputDim; ++k)
                {
                    if (j < sequenceLengths[i])
                        rootGradientsData[(sampleIdx * outputDim) + k] = 1;
                }
            }
        }

        ValuePtr rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(plusOutputShape, rootGradientsData.data(), rootGradientsData.size(), DeviceDescriptor::CPUDevice(), true), inputValue->Mask()->DeepClone());

        std::vector<ElementType> plusParameterGradientData(plusParam.Shape().TotalSize());
        std::vector<ElementType> timesParameterGradientData(timesParam.Shape().TotalSize());
        std::vector<ElementType> inputGradientData(inputShape.TotalSize());
        ValuePtr plusParameterGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(plusParam.Shape(), plusParameterGradientData.data(), plusParameterGradientData.size(), DeviceDescriptor::CPUDevice(), false));
        ValuePtr timesParameterGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(timesParam.Shape(), timesParameterGradientData.data(), timesParameterGradientData.size(), DeviceDescriptor::CPUDevice(), false));
        ValuePtr inputGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputShape, inputGradientData.data(), inputGradientData.size(), DeviceDescriptor::CPUDevice(), false), inputValue->Mask()->DeepClone());

        std::unordered_map<Variable, ValuePtr> outGradients = { { inputVar, inputGradientValue }, { plusParam, plusParameterGradientValue }, { timesParam, timesParameterGradientValue } };
        reducedOutput->Backward(backpropState, { { plusOutput, rootGradientValue } }, outGradients);

        // Verify forward prop results
        std::vector<ElementType> expectedPlusOutputData(plusOutputShape.TotalSize(), 0);
        ElementType expectedReducedValue = 0;
        for (size_t i = 0; i < numSequences; ++i)
        {
            size_t currentSequenceLength = sequenceLengths[i];
            if (useFutureValue)
            {
                for (int j = (int)(currentSequenceLength - 1); j >= 0; j--)
                {
                    ElementType value = 0;
                    for (size_t k = 0; k < inputDim; ++k)
                        value += (ElementType)(0.5 * inputData[(((i * maxActualSequenceLength) + j) * inputDim) + k]);

                    for (size_t k = 0; k < outputDim; ++k)
                    {
                        expectedPlusOutputData[(((i * maxActualSequenceLength) + j) * outputDim) + k] = (ElementType)(value + 0.1);

                        if (j != (currentSequenceLength - 1))
                            expectedPlusOutputData[(((i * maxActualSequenceLength) + j) * outputDim) + k] += expectedPlusOutputData[(((i * maxActualSequenceLength) + (j + 1)) * outputDim) + k];
                    }

                    expectedReducedValue += (outputDim * (ElementType)((value + 0.1) * (j + 1)));
                }
            }
            else
            {
                for (size_t j = 0; j < currentSequenceLength; j++)
                {
                    ElementType value = 0;
                    for (size_t k = 0; k < inputDim; ++k)
                        value += (ElementType)(0.5 * inputData[(((i * maxActualSequenceLength) + j) * inputDim) + k]);

                    for (size_t k = 0; k < outputDim; ++k)
                    {
                        expectedPlusOutputData[(((i * maxActualSequenceLength) + j) * outputDim) + k] = (ElementType)(value + 0.1);

                        if (j != 0)
                            expectedPlusOutputData[(((i * maxActualSequenceLength) + j) * outputDim) + k] += expectedPlusOutputData[(((i * maxActualSequenceLength) + (j - 1)) * outputDim) + k];
                    }

                    expectedReducedValue += (outputDim * (ElementType)((value + 0.1) * (currentSequenceLength - j)));
                }
            }
        }

        FloatingPointVectorCompare(reducedOutputData, std::vector<ElementType>({ expectedReducedValue }), "Forward prop results do not match expected results");
        FloatingPointVectorCompare(plusOutputData, expectedPlusOutputData, "Forward prop results do not match expected results");

        // Verify backward prop results
        ElementType expectedPlusParameterGradientValue = 0;
        for (size_t i = 0; i < numSequences; ++i)
        {
            size_t currentSequenceLength = sequenceLengths[i];
            expectedPlusParameterGradientValue += (currentSequenceLength * (currentSequenceLength + 1)) / 2;
        }

        for (size_t k = 0; k < plusParam.Shape().TotalSize(); ++k)
            if (plusParameterGradientData[k] != expectedPlusParameterGradientValue)
                BOOST_ERROR("Backprop prop results do not match expected results for Plus params gradients");

        std::vector<ElementType> expectedTimesParamsGradientValues(timesParam.Shape().TotalSize(), 0);
        for (size_t i = 0; i < numSequences; ++i)
        {
            size_t currentSequenceLength = sequenceLengths[i];
            for (size_t k = 0; k < inputDim; ++k)
            {
                ElementType gradVal = 0;
                for (size_t j = 0; j < currentSequenceLength; j++)
                {
                    if (useFutureValue)
                        gradVal += (j + 1) * inputData[(((i * maxActualSequenceLength) + j) * inputDim) + k];
                    else
                        gradVal += (currentSequenceLength - j) * inputData[(((i * maxActualSequenceLength) + j) * inputDim) + k];
                }

                for (size_t j = 0; j < outputDim; ++j)
                    expectedTimesParamsGradientValues[(k * outputDim) + j] += gradVal;
            }
        }

        FloatingPointVectorCompare(timesParameterGradientData, expectedTimesParamsGradientValues, "Backprop prop results do not match expected results for Times params gradients");

        std::vector<ElementType> expectedInputGradientValues(inputShape.TotalSize(), 0);
        for (size_t i = 0; i < numSequences; ++i)
        {
            size_t currentSequenceLength = sequenceLengths[i];
            for (size_t j = 0; j < currentSequenceLength; j++)
            {
                ElementType gradVal = 0;
                for (size_t k = 0; k < outputDim; ++k)
                {
                    if (useFutureValue)
                        gradVal += (ElementType)((j + 1) * 0.5);
                    else
                        gradVal += (ElementType)((currentSequenceLength - j) * 0.5);
                }

                for (size_t k = 0; k < inputDim; ++k)
                    expectedInputGradientValues[(((i * maxActualSequenceLength) + j) * inputDim) + k] = gradVal;
            }
        }

        FloatingPointVectorCompare(inputGradientData, expectedInputGradientValues, "Backprop prop results do not match expected results for Times params gradients");
    }
}

BOOST_AUTO_TEST_SUITE(RecurrentFunctionSuite)

BOOST_AUTO_TEST_CASE(SimpleRecurrenceInCPU)
{
    TestSimpleRecurrence<float>(2, 1, 4, 1, DeviceDescriptor::CPUDevice(), true, 3, false, false);
}

BOOST_AUTO_TEST_CASE(SimpleRecurrenceInGPU)
{
    if (ShouldRunOnGpu())
        TestSimpleRecurrence<double>(11, 9, 16, 7, DeviceDescriptor::GPUDevice(0), true, 5, true, false);
}

BOOST_AUTO_TEST_CASE(SimpleLargeRecurrenceInCPU)
{
    if (ShouldRunOnCpu())
    {
        TestSimpleRecurrence<float>(5000, 200, 19, 6, DeviceDescriptor::CPUDevice(), true, 2, false, true, true);
        TestSimpleRecurrence<double>(1000, 9, 16, 3, DeviceDescriptor::CPUDevice(), false, 2, true, true);
    }
}

BOOST_AUTO_TEST_CASE(SimpleLargeRecurrenceInGPU)
{
    if (ShouldRunOnGpu())
    {
        TestSimpleRecurrence<float>(5000, 200, 19, 6, DeviceDescriptor::GPUDevice(0), false, 3, false, true);
        TestSimpleRecurrence<double>(1000, 9, 16, 3, DeviceDescriptor::GPUDevice(0), true, 3, true, true, true);
    }
}

BOOST_AUTO_TEST_CASE(RecurrentNetworkCreationInCPU)
{
    if (ShouldRunOnCpu())
        TestRecurrentNetworkCreation<double>(DeviceDescriptor::CPUDevice(), false);
}

BOOST_AUTO_TEST_CASE(RecurrentNetworkCreationInGPU)
{
    if (ShouldRunOnGpu())
        TestRecurrentNetworkCreation<float>(DeviceDescriptor::GPUDevice(0), true);
}

void ParityCandCppLSTMModel(DeviceDescriptor device, CNTK_DeviceDescriptor cdevice)
{
    const size_t inputDim = 937;
    const size_t numLSTMLayers = 3;
    const size_t cellDim = 1024;
    const size_t hiddenDim = 512;
    const size_t numOutputClasses = 9304;

    auto features = InputVariable({ inputDim }, AsDataType<float>(), L"features");
    auto classifier = LSTMNet<float>(features, cellDim, hiddenDim, numOutputClasses, numLSTMLayers, device, L"classifierOutput");

    auto output = classifier->Output();

    // Save to use in C later.
    const std::wstring tempModelPath = L"test.model";
    if ((_wunlink(tempModelPath.c_str()) != 0) && (errno != ENOENT))
        BOOST_ERROR("Error deleting temp model file 'test.model'");
    classifier->Save(tempModelPath);

    // Prepare input
    std::mt19937_64 generator(13);
    const size_t numberOfFrames = 3;
    std::vector<float> inputData;
    for (int i = 0; i < inputDim * numberOfFrames; ++i)
        inputData.push_back((float)generator());

    // Run C++ forward.
    auto value = Value::CreateSequence(NDShape{ inputDim }, inputData, DeviceDescriptor::CPUDevice());
    std::unordered_map<Variable, ValuePtr> outputs{ { output, nullptr } };
    classifier->Evaluate({ { features, value } }, outputs, device);

    auto outputAsVector = [&output](std::unordered_map<Variable, ValuePtr>& outputs)
    {
        NDArrayViewPtr o = std::make_shared<NDArrayView>(DataType::Float, outputs[output]->Shape(), DeviceDescriptor::CPUDevice());
        o->CopyFrom(*(outputs[output]->Data()));
        const float* buf = o->DataBuffer<float>();
        return std::vector<float>(buf, buf + outputs[output]->Shape().TotalSize());
    };

    auto result = outputAsVector(outputs);

    // Run 3 frame segment with and without resetting.

    auto threeFramesData = MakeSharedObject<NDArrayView>(DataType::Float, NDShape{ inputDim, numberOfFrames, 1 }, inputData.data(), inputData.size() * sizeof(float), DeviceDescriptor::CPUDevice());
    auto mask = MakeSharedObject<NDMask>(NDShape{ numberOfFrames, 1 });
    mask->Clear();
    auto threeFramesValue = MakeSharedObject<Value>(threeFramesData, mask);

    // Make sure we variate the sequence size without reset.
    auto twoFramesShape = NDShape{ inputDim, numberOfFrames - 1, 1 };
    auto twoFramesData = MakeSharedObject<NDArrayView>(DataType::Float, twoFramesShape, inputData.data(), twoFramesShape.TotalSize() * sizeof(float), DeviceDescriptor::CPUDevice());
    auto twoFramesValue = MakeSharedObject<Value>(twoFramesData, MakeSharedObject<NDMask>(NDShape{ numberOfFrames - 1, 1 }));

    mask = MakeSharedObject<NDMask>(NDShape{ numberOfFrames, 1 });
    mask->MarkSequenceBegin({ 0, 0 });
    auto threeFramesValueWithReset = MakeSharedObject<Value>(threeFramesData, mask);

    // With reset.
    outputs[output] = nullptr;
    classifier->Evaluate({ { features, threeFramesValueWithReset } }, outputs, device);
    auto result1 = outputAsVector(outputs);

    // Without reset.
    outputs[output] = nullptr;
    classifier->Evaluate({ { features, threeFramesValue } }, outputs, device);
    auto result2 = outputAsVector(outputs);

    // With reset.
    outputs[output] = nullptr;
    classifier->Evaluate({ { features, threeFramesValueWithReset } }, outputs, device);
    auto result3 = outputAsVector(outputs);

    // With reset.
    outputs[output] = nullptr;
    classifier->Evaluate({ { features, threeFramesValueWithReset } }, outputs, device);
    auto result4 = outputAsVector(outputs);

    // Without reset.
    outputs[output] = nullptr;
    classifier->Evaluate({ { features, twoFramesValue } }, outputs, device);
    auto result5 = outputAsVector(outputs);

    RequireClose(result1, result3, 0.00001f, 0.01f);
    RequireClose(result1, result4, 0.00001f, 0.01f);

    auto norm1 = GetL1Norm(result1, result4);
    auto norm2 = GetL1Norm(result1, result3);
    auto norm3 = GetL1Norm(result1, result2);

    BOOST_REQUIRE_CLOSE(norm1, 0.0, 0.1);
    BOOST_REQUIRE_LT(std::abs(norm1 - norm2), 0.0001);
    BOOST_REQUIRE_GT(std::abs(norm1 - norm3), 0.0001);

    // Create the C model from the saved file.
    CNTK_ModelHandle model;
    auto rc = CNTK_LoadModel("test.model", &cdevice, &model);
    BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
    if (_wunlink(tempModelPath.c_str()) != 0)
        BOOST_ERROR("Error deleting temp model file 'tempModelPath'");

    // Create a copy
    CNTK_ModelHandle cloned;
    rc = CNTK_CloneModel(model, CNTK_ModelParameterShare, false, &cloned);
    BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);

    // Run C original forward.
    CNTK_Variable* outputInfos;
    uint32_t numOutputs = 0;
    rc = CNTK_GetModelOutputsInfo(model, &outputInfos, &numOutputs);
    BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
    BOOST_REQUIRE_EQUAL(numOutputs, classifier->Outputs().size());
    BOOST_REQUIRE_EQUAL(numOutputs, 1u);

    CNTK_Variable* argumentInfos;
    uint32_t numArguments = 0;
    rc = CNTK_GetModelArgumentsInfo(model, &argumentInfos, &numArguments);
    BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);

    // Sequence mode.
    {
        auto three = std::vector<uint32_t>{ (uint32_t)inputDim, (uint32_t)numberOfFrames };
        CNTK_Shape threeFramesShape;
        threeFramesShape.size = 2;
        threeFramesShape.value = three.data();

        CNTK_Value threeFrames;
        threeFrames.data = inputData.data();
        threeFrames.shape = threeFramesShape;

        auto two = std::vector<uint32_t>{ (uint32_t)inputDim, (uint32_t)numberOfFrames - 1 };
        CNTK_Shape twoFramesShapeC;
        twoFramesShapeC.size = 2;
        twoFramesShapeC.value = two.data();

        CNTK_Value twoFrames;
        twoFrames.data = inputData.data();
        twoFrames.shape = twoFramesShapeC;

        bool sequenceFlags[]{ true };

        // With reset.
        NDShape outputShape;
        CNTK_Value* outputValues = nullptr;
        rc = CNTK_EvaluateSequence(model, argumentInfos, &threeFrames, sequenceFlags, numArguments,
            outputInfos, numOutputs, &outputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
        outputShape = NDShape(std::vector<size_t>(outputValues[0].shape.value, outputValues[0].shape.value + outputValues[0].shape.size));
        std::vector<float> cresult1(outputValues[0].data, outputValues[0].data + outputShape.TotalSize());

        // Without reset.
        sequenceFlags[0] = false;
        rc = CNTK_EvaluateSequence(model, argumentInfos, &threeFrames, sequenceFlags, numArguments,
            outputInfos, numOutputs, &outputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
        outputShape = NDShape(std::vector<size_t>(outputValues[0].shape.value, outputValues[0].shape.value + outputValues[0].shape.size));
        std::vector<float> cresult2(outputValues[0].data, outputValues[0].data + outputShape.TotalSize());

        // With reset.
        sequenceFlags[0] = true;
        rc = CNTK_EvaluateSequence(model, argumentInfos, &threeFrames, sequenceFlags, numArguments,
            outputInfos, numOutputs, &outputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
        outputShape = NDShape(std::vector<size_t>(outputValues[0].shape.value, outputValues[0].shape.value + outputValues[0].shape.size));
        std::vector<float> cresult3(outputValues[0].data, outputValues[0].data + outputShape.TotalSize());

        // With reset.
        sequenceFlags[0] = true;
        rc = CNTK_EvaluateSequence(model, argumentInfos, &threeFrames, sequenceFlags, numArguments,
            outputInfos, numOutputs, &outputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
        outputShape = NDShape(std::vector<size_t>(outputValues[0].shape.value, outputValues[0].shape.value + outputValues[0].shape.size));
        std::vector<float> cresult4(outputValues[0].data, outputValues[0].data + outputShape.TotalSize());

        sequenceFlags[0] = false;
        outputValues[0].shape.value[1] = numberOfFrames - 1;
        rc = CNTK_EvaluateSequence(model, argumentInfos, &twoFrames, sequenceFlags, numArguments,
            outputInfos, numOutputs, &outputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
        outputShape = NDShape(std::vector<size_t>(outputValues[0].shape.value, outputValues[0].shape.value + outputValues[0].shape.size));
        std::vector<float> cresult5(outputValues[0].data, outputValues[0].data + outputShape.TotalSize());

        for (uint32_t i = 0; i < numOutputs; i++)
            CNTK_CleanValue(&outputValues[i]);
        CNTK_ReleaseArray(outputValues);

        // Run C cloned forward.
        CNTK_Value* clonedOutputValues = nullptr;
        sequenceFlags[0] = true;
        rc = CNTK_EvaluateSequence(cloned, argumentInfos, &threeFrames, sequenceFlags, numArguments,
            outputInfos, numOutputs, &clonedOutputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);

        // Check that C and C++ results match.
        outputShape = NDShape(std::vector<size_t>(clonedOutputValues[0].shape.value, clonedOutputValues[0].shape.value + clonedOutputValues[0].shape.size));
        std::vector<float> c2(clonedOutputValues[0].data, clonedOutputValues[0].data + outputShape.TotalSize());

        for (uint32_t i = 0; i < numOutputs; i++)
            CNTK_CleanValue(&clonedOutputValues[i]);
        CNTK_ReleaseArray(clonedOutputValues);

        RequireClose(result, cresult1, 0.00001f, 0.01f);
        RequireClose(result1, cresult1, 0.00001f, 0.01f);
        RequireClose(result2, cresult2, 0.00001f, 0.01f);
        RequireClose(result3, cresult3, 0.00001f, 0.01f);
        RequireClose(result4, cresult4, 0.00001f, 0.01f);
        RequireClose(result5, cresult5, 0.00001f, 0.01f);

        BOOST_REQUIRE_EQUAL_COLLECTIONS(c2.begin(), c2.end(),
            cresult1.begin(), cresult1.end());
    }

    // Frame mode.
    {
        uint32_t s = (uint32_t)inputDim;
        CNTK_Shape shape;
        shape.size = 1;
        shape.value = &s;

        CNTK_Value frame1;
        frame1.data = inputData.data();
        frame1.shape = shape;

        bool sequenceFlags[]{ true };
        NDShape outputShape;

        // First frame with reset.
        CNTK_Value* outputValues = nullptr;
        rc = CNTK_EvaluateSequence(model, argumentInfos, &frame1, sequenceFlags, numArguments,
            outputInfos, numOutputs, &outputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
        outputShape = NDShape(std::vector<size_t>(outputValues[0].shape.value, outputValues[0].shape.value + outputValues[0].shape.size));
        std::vector<float> resultFrame1(outputValues[0].data, outputValues[0].data + outputShape.TotalSize());

        // Second frame without reset.
        auto frame2 = frame1;
        frame2.data += inputDim;
        sequenceFlags[0] = false;
        rc = CNTK_EvaluateSequence(model, argumentInfos, &frame2, sequenceFlags, numArguments,
            outputInfos, numOutputs, &outputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
        outputShape = NDShape(std::vector<size_t>(outputValues[0].shape.value, outputValues[0].shape.value + outputValues[0].shape.size));
        std::vector<float> resultFrame2(outputValues[0].data, outputValues[0].data + outputShape.TotalSize());

        // Third frame without reset.
        auto frame3 = frame2;
        frame3.data += inputDim;
        rc = CNTK_EvaluateSequence(model, argumentInfos, &frame3, sequenceFlags, numArguments,
            outputInfos, numOutputs, &outputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
        outputShape = NDShape(std::vector<size_t>(outputValues[0].shape.value, outputValues[0].shape.value + outputValues[0].shape.size));
        std::vector<float> resultFrame3(outputValues[0].data, outputValues[0].data + outputShape.TotalSize());

        auto cresult1 = CombineVectors({ resultFrame1, resultFrame2, resultFrame3 });

        // Without reset.
        rc = CNTK_EvaluateSequence(model, argumentInfos, &frame1, sequenceFlags, numArguments,
            outputInfos, numOutputs, &outputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
        outputShape = NDShape(std::vector<size_t>(outputValues[0].shape.value, outputValues[0].shape.value + outputValues[0].shape.size));
        resultFrame1.assign(outputValues[0].data, outputValues[0].data + outputShape.TotalSize());

        rc = CNTK_EvaluateSequence(model, argumentInfos, &frame2, sequenceFlags, numArguments,
            outputInfos, numOutputs, &outputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
        outputShape = NDShape(std::vector<size_t>(outputValues[0].shape.value, outputValues[0].shape.value + outputValues[0].shape.size));
        resultFrame2.assign(outputValues[0].data, outputValues[0].data + outputShape.TotalSize());

        rc = CNTK_EvaluateSequence(model, argumentInfos, &frame3, sequenceFlags, numArguments,
            outputInfos, numOutputs, &outputValues);
        BOOST_REQUIRE_EQUAL(rc.value, CNTK_SUCCESS);
        outputShape = NDShape(std::vector<size_t>(outputValues[0].shape.value, outputValues[0].shape.value + outputValues[0].shape.size));
        resultFrame3.assign(outputValues[0].data, outputValues[0].data + outputShape.TotalSize());

        for (uint32_t i = 0; i < numOutputs; i++)
            CNTK_CleanValue(&outputValues[i]);
        CNTK_ReleaseArray(outputValues);

        auto cresult2 = CombineVectors({ resultFrame1, resultFrame2, resultFrame3 });

        RequireClose(result1, cresult1, 0.00001f, 0.01f);
        RequireClose(result2, cresult2, 0.00001f, 0.01f);
    }

    // Cleanup C code.
    CNTK_ReleaseModel(model);
    CNTK_ReleaseModel(cloned);

    for (uint32_t i = 0; i < numOutputs; i++)
        CNTK_CleanVariable(&outputInfos[i]);
    CNTK_ReleaseArray(outputInfos);

    for (uint32_t i = 0; i < numArguments; i++)
        CNTK_CleanVariable(&argumentInfos[i]);
    CNTK_ReleaseArray(argumentInfos);
}

BOOST_AUTO_TEST_CASE(TestParityCandCppLSTMModel)
{
    CNTK_DeviceDescriptor* devices = nullptr;
    uint32_t size = 0;
    CNTK_AllDevices(&devices, &size);
    BOOST_REQUIRE(size == DeviceDescriptor::AllDevices().size());

    if (ShouldRunOnCpu())
    {
        CNTK_DeviceDescriptor cpu { CNTK_DeviceKind_CPU, 0 };
        size_t count = 0;
        for (size_t i = 0; i < size; ++i)
            if (devices[i].kind == CNTK_DeviceKind_CPU)
            {
                cpu = devices[i];
                count++;
            }

        BOOST_REQUIRE(count == 1);
        ParityCandCppLSTMModel(DeviceDescriptor::CPUDevice(), cpu);
    }

    if (ShouldRunOnGpu())
    {
        size_t count = 0;
        CNTK_DeviceDescriptor gpu{ CNTK_DeviceKind_GPU, 0 };
        for (size_t i = 0; i < size; ++i)
            if (devices[i].kind == CNTK_DeviceKind_GPU && devices[i].id == 0)
            {
                gpu = devices[i];
                count++;
            }
        BOOST_REQUIRE(count == 1);
        CNTK_DeviceDescriptor deflt{};
        CNTK_DefaultDevice(&deflt);
        BOOST_REQUIRE(deflt.kind == CNTK_DeviceKind_GPU);
        ParityCandCppLSTMModel(DeviceDescriptor::GPUDevice(0), gpu);
    }

    CNTK_ReleaseArray(devices);
}

BOOST_AUTO_TEST_SUITE_END()

}}
