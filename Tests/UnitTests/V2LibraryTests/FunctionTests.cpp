//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

void TestReduceSum(size_t sampleRank, const DeviceDescriptor& device)
{
    size_t numSequences = 7;
    size_t maxAllowedSequenceLength = 11;
    size_t maxDimSize = 23;
    NDShape inputShape(sampleRank);
    for (size_t i = 0; i < sampleRank; ++i)
        inputShape[i] = (rand() % maxDimSize) + 1;

    auto sequenceLengths = GenerateSequenceLengths(numSequences, maxAllowedSequenceLength);
    auto sequences = GenerateSequences<float>(sequenceLengths, inputShape);
    ValuePtr sequencesValue = Value::Create(inputShape, sequences, device, true);

    // Test ReduceSum along a static axis
    {
        auto testReduceSum = [&sequences, &sequenceLengths, inputShape, sequencesValue, device, sampleRank](int reductionAxis, bool useNegativeAxisIndex)
        {
            size_t maxActualSequenceLength = sequencesValue->Shape()[inputShape.Rank()];
            size_t numSequences = sequencesValue->Shape()[inputShape.Rank() + 1];

            auto inputVar = InputVariable(inputShape, DataType::Float, L"input");
            FunctionPtr reduceSumFunc;

            bool reduceAll = (reductionAxis < 0);
            if (reduceAll)
                reduceSumFunc = ReduceSum(inputVar);
            else
                reduceSumFunc = ReduceSum(inputVar, Axis(useNegativeAxisIndex ? (reductionAxis - (int)sampleRank) : reductionAxis));

            NDShape outputShape = reduceSumFunc->Output().Shape();
            NDShape outputDataShape = outputShape;
            if (!reduceAll)
                outputDataShape = outputDataShape.AppendShape({ maxActualSequenceLength, numSequences });

            std::vector<float> outputData(outputDataShape.TotalSize());
            ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputDataShape, outputData, false), reduceAll ? nullptr : sequencesValue->Mask()->DeepClone());

            std::unordered_map<Variable, ValuePtr> outputs = { { reduceSumFunc->Output(), outputValue } };
            reduceSumFunc->Forward({ { inputVar, sequencesValue } }, outputs, device);

            std::vector<size_t> inputShapeStrides = GetStrides(inputShape);
            std::vector<size_t> outputShapeStrides = GetStrides(outputShape);

            std::vector<float> expectedPerFrameTotals(outputShape.TotalSize() * maxActualSequenceLength * numSequences, 0.0f);
            float expectedTotal = 0.0f;
            for (size_t i = 0; i < numSequences; ++i)
            {
                size_t currentSequenceLength = sequenceLengths[i];
                for (size_t j = 0; j < currentSequenceLength; ++j)
                {
                    for (size_t k = 0; k < inputShape.TotalSize(); ++k)
                    {
                        auto inputIdx = UnflattenedShape(k, inputShapeStrides);
                        auto outputIdx = inputIdx;
                        if (!reduceAll)
                            outputIdx[reductionAxis] = 0;
                        else
                            outputIdx = {};

                        auto flatOutputIdx = FlattenedIndex(outputIdx, outputShapeStrides);
                        float value = sequences[i][(j * inputShape.TotalSize()) + k];
                        expectedPerFrameTotals[(((i * maxActualSequenceLength) + j) * outputShape.TotalSize()) + flatOutputIdx] += value;
                        expectedTotal += value;
                    }
                }
            }

            if (reduceAll)
                FloatingPointVectorCompare(outputData, std::vector<float>({ expectedTotal }), "testReduceSum: Forward prop results do not match expected results");
            else
                FloatingPointVectorCompare(outputData, expectedPerFrameTotals, "testReduceSum: Forward prop results do not match expected results");
        };

        // Reduce over all axes
        testReduceSum(-1, false);

        int reductionAxis = 0;
        testReduceSum(reductionAxis, true);

        if (reductionAxis < (inputShape.Rank() - 1))
            reductionAxis++;

        testReduceSum(reductionAxis, false);

        if (reductionAxis < (inputShape.Rank() - 1))
            reductionAxis++;

        testReduceSum(reductionAxis, true);
    }

    // Test ReduceSum along a dynamic axis
    {
        auto testReduceSum = [&sequences, &sequenceLengths, inputShape, sequencesValue, device]()
        {
            size_t numSequences = sequencesValue->Shape()[inputShape.Rank() + 1];

            auto inputVar = InputVariable({ inputShape }, DataType::Float, L"input");
            FunctionPtr reduceSumFunc = Sequence::ReduceSum(inputVar);

            NDShape maskShape = { 1, numSequences };
            NDShape outputShape = reduceSumFunc->Output().Shape();
            auto outputDataShape = outputShape.AppendShape(maskShape);

            std::vector<float> outputData(outputDataShape.TotalSize());
            auto maskPtr = MakeSharedObject<NDMask>(maskShape, device);
            ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputDataShape, outputData, false), maskPtr);

            std::unordered_map<Variable, ValuePtr> outputs = { { reduceSumFunc->Output(), outputValue } };
            reduceSumFunc->Forward({ { inputVar, sequencesValue } }, outputs, device);

            std::vector<float> expectedTotals(outputDataShape.TotalSize(), 0.0f);
            for (size_t i = 0; i < numSequences; ++i)
            {
                size_t currentSequenceLength = sequenceLengths[i];
                for (size_t j = 0; j < currentSequenceLength; ++j)
                {
                    for (size_t k = 0; k < inputShape.TotalSize(); ++k)
                    {
                        float value = sequences[i][(j * inputShape.TotalSize()) + k];
                        expectedTotals[(i * inputShape.TotalSize()) + k] += value;
                    }
                }
            }

            FloatingPointVectorCompare(outputData, expectedTotals, "testReduceSum: Forward prop results do not match expected results");
        };

        testReduceSum();
    }
}

void TestSlice(size_t sampleRank, const DeviceDescriptor& device)
{
    size_t numSequences = 7;
    size_t maxAllowedSequenceLength = 11;
    size_t maxDimSize = 23;
    size_t minDimSize = 5;
    NDShape inputShape(sampleRank);
    for (size_t i = 0; i < sampleRank; ++i)
        inputShape[i] = (rand() % maxDimSize) + minDimSize;

    auto sequenceLengths = GenerateSequenceLengths(numSequences, maxAllowedSequenceLength);
    auto sequences = GenerateSequences<float>(sequenceLengths, inputShape);
    ValuePtr sequencesValue = Value::Create(inputShape, sequences, device, true);

    // Test slice along a static axis
    {
        auto testStaticAxisSlice = [&sequences, &sequenceLengths, inputShape, sequencesValue, device, sampleRank](int sliceAxis, int beginOffset, int endOffset, bool useNegativeAxisIndex)
        {
            size_t maxActualSequenceLength = sequencesValue->Shape()[inputShape.Rank()];
            size_t numSequences = sequencesValue->Shape()[inputShape.Rank() + 1];

            auto inputVar = InputVariable(inputShape, DataType::Float, L"input");
            auto sliceFunc = Slice(inputVar, Axis(useNegativeAxisIndex ? (sliceAxis - (int)sampleRank) : sliceAxis), beginOffset, endOffset);

            NDShape outputShape = sliceFunc->Output().Shape();
            auto outputDataShape = outputShape.AppendShape({ maxActualSequenceLength, numSequences });
            std::vector<float> outputData(outputDataShape.TotalSize());
            ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputDataShape, outputData, false), sequencesValue->Mask()->DeepClone());

            std::unordered_map<Variable, ValuePtr> outputs = { { sliceFunc->Output(), outputValue } };
            sliceFunc->Forward({ { inputVar, sequencesValue } }, outputs, device);

            std::vector<size_t> inputShapeStrides = GetStrides(inputShape);
            std::vector<size_t> outputShapeStrides = GetStrides(outputShape);

            size_t sliceStartOffset = (beginOffset >= 0) ? beginOffset : (inputShape[sliceAxis] + beginOffset);
            std::vector<float> expectedOutputValues(outputShape.TotalSize() * maxActualSequenceLength * numSequences);
            for (size_t i = 0; i < numSequences; ++i)
            {
                size_t currentSequenceLength = sequenceLengths[i];
                for (size_t j = 0; j < currentSequenceLength; ++j)
                {
                    for (size_t k = 0; k < outputShape.TotalSize(); ++k)
                    {
                        auto outputIdx = UnflattenedShape(k, outputShapeStrides);
                        auto inputIdx = outputIdx;
                        inputIdx[sliceAxis] += sliceStartOffset;
                        auto flatInputIdx = FlattenedIndex(inputIdx, inputShapeStrides);
                        expectedOutputValues[(((i * maxActualSequenceLength) + j) * outputShape.TotalSize()) + k] = sequences[i][(j * inputShape.TotalSize()) + flatInputIdx];
                    }
                }
            }

            FloatingPointVectorCompare(outputData, expectedOutputValues, "testStaticAxisSlice: Forward prop results do not match expected results");
        };

        int sliceAxis = 0;
        testStaticAxisSlice(sliceAxis, 3, 5, true);

        if (sliceAxis < (inputShape.Rank() - 1))
            sliceAxis++;

        testStaticAxisSlice(sliceAxis, -1, 0, false);

        if (sliceAxis < (inputShape.Rank() - 1))
            sliceAxis++;

        testStaticAxisSlice(sliceAxis, -3, -1, true);
    }

    // Test slice along a dynamic axis
    {
        auto testDynamicAxisSlice = [&sequences, &sequenceLengths, inputShape, sequencesValue, device](int beginOffset, int endOffset)
        {
            size_t maxActualSequenceLength = sequencesValue->Shape()[inputShape.Rank()];
            size_t numSequences = sequencesValue->Shape()[inputShape.Rank() + 1];

            int endAndBeginOffsetDiff = endOffset - beginOffset;
            size_t maxSliceLength = (endAndBeginOffsetDiff > 0) ? endAndBeginOffsetDiff : maxActualSequenceLength + endAndBeginOffsetDiff;

            auto inputVar = InputVariable(inputShape, DataType::Float, L"input");
            auto sliceFunc = Sequence::Slice(inputVar, beginOffset, endOffset);
            sliceFunc = sliceFunc + sliceFunc;

            size_t outputSequenceAxisLength = maxSliceLength;
            size_t outputBatchAxisLength = numSequences;
            NDShape outputShape = sliceFunc->Output().Shape().AppendShape({ outputSequenceAxisLength, outputBatchAxisLength });
            std::vector<float> outputData(outputShape.TotalSize(), 0);
            NDMaskPtr mask;
            if (endAndBeginOffsetDiff < 0)
            {
                ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData, false));
                mask = MakeSharedObject<NDMask>(std::initializer_list<size_t>({ outputSequenceAxisLength, outputBatchAxisLength }), device);
            }
            ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData, false), mask);

            std::unordered_map<Variable, ValuePtr> outputs = { { sliceFunc->Output(), outputValue } };
            sliceFunc->Forward({ { inputVar, sequencesValue } }, outputs, device);

            size_t startSequenceIdx = 0;
            size_t endSequenceIdx = numSequences;

            std::vector<float> expectedOutputValues(inputShape.TotalSize() * outputSequenceAxisLength * outputBatchAxisLength);
            for (size_t i = startSequenceIdx; i < endSequenceIdx; ++i)
            {
                size_t currentSequenceLength = sequenceLengths[i];
                size_t startFrameIdx = ((beginOffset >= 0) ? beginOffset : (currentSequenceLength + beginOffset));
                size_t endFrameIdx = ((endOffset > 0) ? endOffset : (currentSequenceLength + endOffset));
                size_t j = startFrameIdx;
                for (; j < endFrameIdx; ++j)
                {
                    for (size_t k = 0; k < inputShape.TotalSize(); ++k)
                        expectedOutputValues[((((i - startSequenceIdx) * outputSequenceAxisLength) + (j - startFrameIdx)) * inputShape.TotalSize()) + k] = 2 * sequences[i][(j * inputShape.TotalSize()) + k];
                }

                // Zero out the invalid portions of the actual output
                for (; j < (outputSequenceAxisLength + startFrameIdx); ++j)
                    for (size_t k = 0; k < inputShape.TotalSize(); ++k)
                        outputData[((((i - startSequenceIdx) * outputSequenceAxisLength) + (j - startFrameIdx)) * inputShape.TotalSize()) + k] = 0;
            }

            FloatingPointVectorCompare(outputData, expectedOutputValues, "testDynamicAxisSlice: Forward prop results do not match expected results");
        };

        testDynamicAxisSlice(0, 1);
        testDynamicAxisSlice(0, 2);
        testDynamicAxisSlice(-1, 0);
        testDynamicAxisSlice(-2, 0);
        testDynamicAxisSlice(0, -1);
        testDynamicAxisSlice(1, 0);
    }
}

void TestRecurrentFunctionCloning()
{
    size_t inputDim = 2;
    size_t outputDim = 3;
    auto device = DeviceDescriptor::CPUDevice();
    Parameter timesParam(MakeSharedObject<NDArrayView>(0.5f, NDShape({ outputDim, inputDim }), device), L"timesParameters");
    Parameter plusParam(MakeSharedObject<NDArrayView>(0.1f, std::initializer_list<size_t>({ outputDim }), device), L"plusParameters");

    auto inputVar = InputVariable({ inputDim }, false, DataType::Float, true, L"input");

    auto placeholder = PlaceholderVariable(std::initializer_list<size_t>({ outputDim }));
    auto plusOutput = Plus(plusParam, Plus(placeholder, Times(timesParam, inputVar)), L"plusOutput");
    auto placeholderReplacement = PastValue(plusOutput);
    plusOutput = plusOutput->ReplacePlaceholders({ { placeholder, placeholderReplacement } });

    auto reducedOutput = ReduceSum(plusOutput, L"sum");
    auto rootFuncOriginal = Combine({ reducedOutput, plusOutput });

    std::unordered_set<FunctionPtr> visitedFunctions;

    auto clonedFunctionWithParametersCloned = rootFuncOriginal->Clone();
    CompareFunctions(rootFuncOriginal, clonedFunctionWithParametersCloned, ParameterCloningMethod::Clone, {}, visitedFunctions);

    visitedFunctions.clear();
    auto clonedFunctionWithParametersShared = clonedFunctionWithParametersCloned->Clone(ParameterCloningMethod::Share);
    CompareFunctions(clonedFunctionWithParametersCloned, clonedFunctionWithParametersShared, ParameterCloningMethod::Share, {}, visitedFunctions);

    visitedFunctions.clear();
    auto replacementInputVar = InputVariable({ inputDim }, true, DataType::Float, false, L"input2");
    std::unordered_map<Variable, Variable> cloningReplacements = { { *(clonedFunctionWithParametersShared->Arguments().begin()), replacementInputVar } };
    auto clonedFunctionWithParametersFrozen = clonedFunctionWithParametersShared->Clone(ParameterCloningMethod::Freeze, cloningReplacements);
    CompareFunctions(clonedFunctionWithParametersShared, clonedFunctionWithParametersFrozen, ParameterCloningMethod::Freeze, cloningReplacements, visitedFunctions);
}

void TestTranspose(size_t numAxes, int axis1, int axis2, const DeviceDescriptor& device)
{
    srand(1);

    size_t maxDimSize = 15;
    NDShape inputShape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
        inputShape[i] = (rand() % maxDimSize) + 1;

    auto inputVar = InputVariable(inputShape, DataType::Float, false, L"leftInput");
    auto transposeFunc = TransposeAxes(inputVar, Axis(axis1), Axis(axis2));

    std::vector<float> inputData(inputShape.TotalSize());
    for (size_t i = 0; i < inputData.size(); ++i)
        inputData[i] = ((float)rand()) / RAND_MAX;

    auto inputValueShape = inputShape.AppendShape({ 1, 1 });
    ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(inputValueShape, inputData, true));

    NDShape outputShape = transposeFunc->Output().Shape();
    NDShape outputValueShape = outputShape.AppendShape({ 1, 1 });
    std::vector<float> outputData(outputValueShape.TotalSize());
    ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputValueShape, outputData, false));

    std::unordered_map<Variable, ValuePtr> outputs = { { transposeFunc->Output(), outputValue } };
    transposeFunc->Forward({ { inputVar, inputValue } }, outputs, device);

    std::vector<size_t> inputShapeStrides = GetStrides(inputShape);
    std::vector<size_t> outputShapeStrides = GetStrides(outputShape);

    // Verify forward prop results
    std::vector<float> expectedOutputValues(outputShape.TotalSize());
    for (size_t i = 0; i < expectedOutputValues.size(); ++i)
    {
        auto unflattenedShape = UnflattenedShape(i, outputShapeStrides);
        std::swap(unflattenedShape[axis1], unflattenedShape[axis2]);
        size_t flattenedIndex = FlattenedIndex(unflattenedShape, inputShapeStrides);
        expectedOutputValues[i] = inputData[flattenedIndex];
    }

    FloatingPointVectorCompare(outputData, expectedOutputValues, "TestTimesAndPlus: Forward prop results do not match expected results");
}

void TestTimesNodeShapeInference()
{
    auto timesNodeShapeInferenceTest = [](size_t inputRank, size_t outputRank, int inputRankToMap) {
        
        auto device = DeviceDescriptor::CPUDevice();

        size_t maxDimSize = 15;
        NDShape outputShape(outputRank);
        for (size_t i = 0; i < outputRank; ++i)
            outputShape[i] = (rand() % maxDimSize) + 1;

        NDShape paramShape = outputShape;
        if (inputRankToMap > 0)
            paramShape = paramShape.AppendShape({ NDShape::InferredDimension });
        else
            paramShape = paramShape.AppendShape(NDShape(inputRank));

        auto timesParam = Parameter(paramShape, DataType::Float, ConstantInitializer(), device);

        auto placeholderInput = PlaceholderVariable();
        auto timesFunction = Times(timesParam, placeholderInput, outputRank, inputRankToMap);

        NDShape inputShape(inputRank);
        for (size_t i = 0; i < inputRank; ++i)
            inputShape[i] = (rand() % maxDimSize) + 1;

        auto actualInput = InputVariable(inputShape, DataType::Float);
        timesFunction->ReplacePlaceholders({ { placeholderInput, actualInput } });

        // Verify that the inferred shape of the param, input and output matches expectation
        auto expectedInputShape = inputShape;
        auto expectedParamShape = outputShape;
        if (inputRankToMap > 0)
            expectedParamShape = expectedParamShape.AppendShape(inputShape.SubShape(0, inputRank - inputRankToMap));
        else
            expectedParamShape = expectedParamShape.AppendShape(inputShape);

        auto expectedOutputShape = outputShape;
        if (inputRankToMap > 0)
            expectedOutputShape = expectedOutputShape.AppendShape(inputShape.SubShape(inputRank - inputRankToMap));

        auto actualInputShape = timesFunction->Arguments()[0].Shape();
        auto actualParamShape = timesFunction->Parameters()[0].Shape();
        auto actualOutputShape = timesFunction->Output().Shape();

        if (actualInputShape != expectedInputShape)
            ReportFailure("Times nodes actual input shape (%S) does not match expectation (%S)", actualInputShape.AsString().c_str(), expectedInputShape.AsString().c_str());

        if (actualParamShape != expectedParamShape)
            ReportFailure("Times nodes actual parameter shape (%S) does not match expectation (%S)", actualParamShape.AsString().c_str(), expectedParamShape.AsString().c_str());

        if (actualOutputShape != expectedOutputShape)
            ReportFailure("Times nodes actual output shape (%S) does not match expectation (%S)", actualOutputShape.AsString().c_str(), expectedOutputShape.AsString().c_str());
    };

    timesNodeShapeInferenceTest(2, 2, -1);
    timesNodeShapeInferenceTest(2, 1, 1);
    timesNodeShapeInferenceTest(1, 2, 0);
    timesNodeShapeInferenceTest(3, 2, 2);
}

template <typename ElementType>
void TestChangingParameterValues(size_t rank, const DeviceDescriptor& device)
{
    size_t maxDimSize = 15;
    NDShape shape(rank);
    for (size_t i = 0; i < rank; ++i)
        shape[i] = (rand() % maxDimSize) + 1;

    auto numElements = shape.TotalSize();

    auto param = Parameter(shape, AsDataType<ElementType>(), GlorotUniformInitializer(), device);
    auto plus = Plus(param, param);
    

    std::vector<ElementType> outputData(numElements);
    ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(shape, outputData, false));

    std::unordered_map<Variable, ValuePtr> outputs = { { plus->Output(), outputValue } };
    plus->Forward({}, outputs, device);

    NDArrayViewPtr cpuView;
    auto getParameterData = [&cpuView](const Parameter& p) -> const ElementType*
    {
          cpuView = (p.Value()->Device() != DeviceDescriptor::CPUDevice()) ?
                     p.Value()->DeepClone(DeviceDescriptor::CPUDevice()) : p.Value();
          return cpuView->DataBuffer<ElementType>();
    };

    auto parameterData = getParameterData(param);

    for (int i = 0; i < numElements; i++)
    {
        FloatingPointCompare<ElementType>(outputData[i], 2 * parameterData[i],
                                          "Function output does not match the expected value.");
    }

    // Change parameter values element-wise, through a pointer to the writable data buffer.
    // This only works in CPU mode. In GPU mode the buffer needs to be copied over to a cpuView,
    // modified there, and then copied back again by calling CopyFrom. The latter is essentially 
    // what SetValue below does. 
    if (device == DeviceDescriptor::CPUDevice())
    {
        auto data = param.Value()->WritableDataBuffer<ElementType>();

        for (int i = 0; i < numElements; i++)
        {
            data[i] *= i;
        }

        param.RecordValueUpdate();
        plus->Forward({}, outputs, device);
        parameterData = getParameterData(param);
        for (int i = 0; i < numElements; i++)
        {

            FloatingPointCompare<ElementType>(outputData[i], 2 * parameterData[i],
                                              "Function output does not match the expected value.");
        }
    }

    // Change parameter values directly by calling Parameter::SetValue.
    std::vector<ElementType> newValues(numElements);
    for (int i = 0; i < numElements; i++)
    {
        newValues[i] = ElementType(1.0) / (i + ElementType(1.0));
    }
    auto newValuesNDarray = MakeSharedObject<NDArrayView>(shape, newValues, false);

    param.SetValue(newValuesNDarray);
    plus->Forward({}, outputs, device);
    parameterData = getParameterData(param);
    for (int i = 0; i < numElements; i++)
    {
        auto denom = (i + ElementType(1.0));
        FloatingPointCompare<ElementType>(parameterData[i], ElementType(1.0) / denom, 
                                          "Parameter valued does not match the expected value.");
        FloatingPointCompare<ElementType>(outputData[i], ElementType(2.0) / denom, 
                                          "Function output does not match the expected value.");
    }
}

void TestRecurrenceShapeInference()
{
    auto testShapeInferenceInRecurrence = [](size_t inputRank, size_t outputRank) {
        auto placeholderInput = PlaceholderVariable(NDShape(inputRank));

        srand(1);

        size_t maxDimSize = 15;
        NDShape inputShape(inputRank);
        for (size_t i = 0; i < inputRank; ++i)
            inputShape[i] = (rand() % maxDimSize) + 1;

        NDShape outputShape(outputRank);
        for (size_t i = 0; i < outputRank; ++i)
            outputShape[i] = (rand() % maxDimSize) + 1;

        auto device = DeviceDescriptor::CPUDevice();
        Parameter timesParam(outputShape.AppendShape(placeholderInput.Shape()), DataType::Float, GlorotUniformInitializer((int)outputRank), device, L"timesParameters");
        Parameter plusParam(NDShape(outputRank, NDShape::InferredDimension), DataType::Float, ConstantInitializer(), device, L"plusParameters");

        auto recurrenceForwardReference = PlaceholderVariable(NDShape(outputRank, NDShape::InferredDimension));
        auto projectionOutput = Times(timesParam, placeholderInput, outputRank);
        auto firstPlusOutput = Plus(recurrenceForwardReference, projectionOutput);
        auto plusOutput = Plus(plusParam, firstPlusOutput, L"plusOutput");
        auto placeholderReplacement = PastValue(plusOutput);
        plusOutput = plusOutput->ReplacePlaceholders({ { recurrenceForwardReference, placeholderReplacement } });

        auto reducedOutput = ReduceSum(plusOutput, L"sum");
        auto rootFuncOriginal = Combine({ reducedOutput, plusOutput });

        auto inputVar = InputVariable(inputShape, false, DataType::Float, true, L"input", { Axis::NewUniqueDynamicAxis(L"inputSequence"), Axis::DefaultBatchAxis() });
        rootFuncOriginal->ReplacePlaceholders({ { placeholderInput, inputVar } });

        if (timesParam.Shape() != outputShape.AppendShape(inputShape))
            ReportFailure("timesParams shape does not match expectation; expected = %S, actual = %S", outputShape.AppendShape(inputShape).AsString().c_str(), timesParam.Shape().AsString().c_str());

        if (plusParam.Shape() != outputShape)
            ReportFailure("plusParam shape does not match expectation; expected = %S, actual = %S", outputShape.AsString().c_str(), plusParam.Shape().AsString().c_str());

        if (plusOutput->Output().DynamicAxes() != inputVar.DynamicAxes())
            ReportFailure("plusOutput dynamic axes do not match expectation!");
    };

    testShapeInferenceInRecurrence(1, 1);
    testShapeInferenceInRecurrence(2, 1);
    testShapeInferenceInRecurrence(1, 2);
    testShapeInferenceInRecurrence(2, 2);
}

void FunctionTests()
{
    fprintf(stderr, "\nFunctionTests..\n");

    TestChangingParameterValues<float>(2, DeviceDescriptor::CPUDevice());
    if (IsGPUAvailable())
    {
        TestChangingParameterValues<double>(3, DeviceDescriptor::GPUDevice(0));
    } 
    else
    {
        TestChangingParameterValues<double>(3, DeviceDescriptor::CPUDevice());
    }

    TestTimesNodeShapeInference();
    TestRecurrenceShapeInference();

    TestSlice(2, DeviceDescriptor::CPUDevice());
    if (IsGPUAvailable())
    {
        TestSlice(1, DeviceDescriptor::GPUDevice(0));
    }

    TestReduceSum(1, DeviceDescriptor::CPUDevice());
    if (IsGPUAvailable())
    {
        TestReduceSum(2, DeviceDescriptor::GPUDevice(0));
    }

    TestRecurrentFunctionCloning();

    TestTranspose(2, 0, 1, DeviceDescriptor::CPUDevice());
    if (IsGPUAvailable())
    {
        TestTranspose(3, 1, 2, DeviceDescriptor::GPUDevice(0));
    }
}

