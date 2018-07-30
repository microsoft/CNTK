//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Common.h"
#include <numeric>

using namespace CNTK;

namespace CNTK { namespace Test {

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
                reduceSumFunc = ReduceSum(inputVar, Axis::AllAxes());
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

            NDShape maskShape = { numSequences };
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
            std::vector<Axis> axis; 
            std::vector<int> beginOffsetVec, endOffsetVec; 
            axis.push_back(Axis(useNegativeAxisIndex ? (sliceAxis - (int)sampleRank) : sliceAxis)); 
            beginOffsetVec.push_back(beginOffset); 
            endOffsetVec.push_back(endOffset); 
            auto sliceFunc = Slice(inputVar, axis, beginOffsetVec, endOffsetVec);

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
            NDShape outputShape = sliceFunc->Output().Shape();
            if (endAndBeginOffsetDiff != 1)
                outputShape = outputShape.AppendShape({ outputSequenceAxisLength });
            outputShape = outputShape.AppendShape({ outputBatchAxisLength });
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

    auto reducedOutput = ReduceSum(plusOutput, Axis::AllAxes(), L"sum");
    auto rootFuncOriginal = Combine({ reducedOutput, plusOutput });

    std::unordered_set<FunctionPtr> visitedFunctions;

    auto clonedFunctionWithParametersCloned = rootFuncOriginal->Clone();
    CompareFunctions(rootFuncOriginal, clonedFunctionWithParametersCloned, ParameterCloningMethod::Clone, {}, visitedFunctions);

    visitedFunctions.clear();
    auto clonedFunctionWithParametersShared = clonedFunctionWithParametersCloned->Clone(ParameterCloningMethod::Share);
    CompareFunctions(clonedFunctionWithParametersCloned, clonedFunctionWithParametersShared, ParameterCloningMethod::Share, {}, visitedFunctions);

    visitedFunctions.clear();
    auto replacementInputVar = InputVariable({ inputDim }, true, DataType::Float, true, L"input2");
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

// We test splice by comparing against a reference implementation that transposes 
// the axis to splice with axis 0 for each of the inputs and then splices along axis0
void TestSplice(size_t numInputs, size_t maxNumInputAxes, size_t spliceAxis, const DeviceDescriptor& device)
{
    size_t maxDimSize = 15;
    size_t minDimSize = 3;
    NDShape maxRankInputShape(maxNumInputAxes);
    for (size_t i = 0; i < maxNumInputAxes; ++i)
        maxRankInputShape[i] = (rand() % maxDimSize) + minDimSize; // We have each axis dimensionality be at least 2

    size_t numSequences = 5;
    size_t maxAllowedSequenceLength = 9;
    auto sequenceLengths = GenerateSequenceLengths(numSequences, maxAllowedSequenceLength);

    size_t maxRankInputIndex = rand() % numInputs;
    std::vector<NDShape> inputShapes(numInputs);
    std::vector<Variable> inputVars(numInputs);
    std::vector<std::vector<std::vector<float>>> inputsData(numInputs);
    std::unordered_map<Variable, ValuePtr> argumentValues;
    //std::vector<ValuePtr> inputsValue(numInputs);
    for (size_t i = 0; i < numInputs; ++i)
    {
        if (i == maxRankInputIndex)
            inputShapes[i] = maxRankInputShape;
        else
        {
            size_t rank = (rand() % maxNumInputAxes) + 1;
            inputShapes[i] = maxRankInputShape.SubShape(0, rank);
            if (rank > spliceAxis)
                inputShapes[i][spliceAxis] = (rand() % maxDimSize) + 2;
        }

        inputVars[i] = InputVariable(inputShapes[i], DataType::Float, true);
        inputsData[i] = GenerateSequences<float>(sequenceLengths, inputShapes[i]);
        argumentValues.insert({ inputVars[i], Value::Create(inputShapes[i], inputsData[i], device, true) });
    }

    auto spliceFunc = Splice(inputVars, Axis((int)spliceAxis), L"splice");
    std::unordered_map<Variable, ValuePtr> spliceOutputs = { { spliceFunc->Output(), nullptr } };
    auto spliceFuncBackpropState = spliceFunc->Forward(argumentValues, spliceOutputs, device, { spliceFunc->Output() });

    FunctionPtr spliceUsingTransposeFunc;
    std::vector<FunctionPtr> transposeInputFuncs(numInputs);
    std::vector<Variable> transposedInputs(numInputs);
    for (size_t i = 0; i < numInputs; ++i)
    {
        transposeInputFuncs[i] = TransposeAxes(inputVars[i], Axis(0), Axis((int)spliceAxis));
        transposedInputs[i] = transposeInputFuncs[i];
    }

    auto spliceTransposedFunc = Splice(transposedInputs, Axis(0));
    spliceUsingTransposeFunc = TransposeAxes(spliceTransposedFunc, Axis(0), Axis((int)spliceAxis));
    std::unordered_map<Variable, ValuePtr> spliceUsingTransposeOutputs = { { spliceUsingTransposeFunc->Output(), nullptr } };
    auto spliceUsingTransposeFuncBackpropState = spliceUsingTransposeFunc->Forward(argumentValues, spliceUsingTransposeOutputs, device, { spliceUsingTransposeFunc->Output() });

    // Verify the results
    // Temporarily enable the unpacking of packed value objects for result verification
    auto automaticUnpackingOfPackedValuesDisabled = Internal::IsAutomaticUnpackingOfPackedValuesDisabled();
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ false);

    if (!Internal::AreEqual(*spliceOutputs.begin()->second, *spliceUsingTransposeOutputs.begin()->second, relativeTolerance, absoluteTolerance))
        ReportFailure("Splice actual output does not match expectation");

    // Test backprop
    std::unordered_map<Variable, ValuePtr> sliceInputGradients;
    std::unordered_map<Variable, ValuePtr> sliceUsingTransposeInputGradients;
    for (size_t i = 0; i < numInputs; ++i)
    {
        sliceInputGradients.insert({ inputVars[i], nullptr });
        sliceUsingTransposeInputGradients.insert({ inputVars[i], nullptr });
    }

    std::unordered_map<Variable, ValuePtr> sliceRootGradients = { { spliceFunc->Output(), MakeSharedObject<Value>(spliceOutputs.begin()->second->Data(), spliceOutputs.begin()->second->Mask()) } };
    spliceFunc->Backward(spliceFuncBackpropState, sliceRootGradients, sliceInputGradients);

    std::unordered_map<Variable, ValuePtr> sliceUsingTransposeRootGradients = { { spliceUsingTransposeFunc->Output(), MakeSharedObject<Value>(spliceOutputs.begin()->second->Data(), spliceOutputs.begin()->second->Mask()) } };
    spliceUsingTransposeFunc->Backward(spliceUsingTransposeFuncBackpropState, sliceUsingTransposeRootGradients, sliceUsingTransposeInputGradients);
    for (size_t i = 0; i < numInputs; ++i)
    {
        auto actualInputGradientValue = sliceInputGradients[inputVars[i]];
        auto expectedInputGradientValue = sliceUsingTransposeInputGradients[inputVars[i]];
        if (!Internal::AreEqual(*actualInputGradientValue, *expectedInputGradientValue, relativeTolerance, absoluteTolerance))
            ReportFailure("Splice actual gradient does not match expectation");
    }

    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ automaticUnpackingOfPackedValuesDisabled);
}

void TestSplice()
{
    srand(1);

    if (ShouldRunOnCpu())
    {
        TestSplice(4, 2, 0, DeviceDescriptor::CPUDevice());
        TestSplice(3, 3, 2, DeviceDescriptor::CPUDevice());
        TestSplice(2, 3, 3, DeviceDescriptor::CPUDevice());
    }

    if (ShouldRunOnGpu())
    {
        TestSplice(4, 3, 1, DeviceDescriptor::GPUDevice(0));
        TestSplice(3, 4, 2, DeviceDescriptor::GPUDevice(0));
        TestSplice(3, 5, 6, DeviceDescriptor::GPUDevice(0));
    }
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

void TestTimesIndirectSparseInputGradientSparse(const DeviceDescriptor& device)
{
    size_t dim = 5;
    size_t numSequences = 1;

    auto timesParam = Parameter(NDShape({ 1, dim }), DataType::Float, 0.0, device);

    auto input = InputVariable(NDShape({ dim }), /* isSparse*/ true, DataType::Float);
    auto timesFunction = Times(timesParam, Sequence::First(input));

    auto inputValue = Value::CreateSequence<float>(dim, { 2 }, device, true);
    std::unordered_map<Variable, ValuePtr> inputMap;
    inputMap.insert(std::make_pair(input, inputValue));

    std::vector<float> outputData(numSequences);
    ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(NDShape({ 1, numSequences }), outputData, false));
    std::unordered_map<Variable, ValuePtr> outputMap;
    outputMap.insert(std::make_pair(timesFunction->Output(), outputValue));

    auto backState = timesFunction->Forward(inputMap, outputMap, device, { timesFunction->Output() });

    std::unordered_map<Variable, ValuePtr> rootGradients;
    std::vector<float> rootGradient(numSequences, 1.0f);
    rootGradients.insert(std::make_pair(timesFunction->Output(), MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(NDShape({ numSequences }), rootGradient, false))));

    std::unordered_map<Variable, ValuePtr> inputGradients;
    inputGradients.insert(std::make_pair(timesParam, nullptr));

    timesFunction->Backward(backState, rootGradients, inputGradients);

    ValuePtr paramGradient = inputGradients[timesParam];

    if (!paramGradient->IsSparse())
        ReportFailure("Gradient is expected to be sparse.");
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
    plus->Forward(std::unordered_map<Variable, ValuePtr>({}), outputs, device);

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
        plus->Forward(std::unordered_map<Variable, ValuePtr>({}), outputs, device);
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
    plus->Forward(std::unordered_map<Variable, ValuePtr>({}), outputs, device);
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

        auto reducedOutput = ReduceSum(plusOutput, Axis::AllAxes(), L"sum");
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

void TestFunctionOutputs(const DeviceDescriptor& device)
{
    // Simple
    {
        Variable o1, o2;
        {
            size_t inputDim = 1;
            size_t outputDim = 2;

            auto inputVar = InputVariable({ inputDim }, false, DataType::Float, true, L"features", { Axis::NewUniqueDynamicAxis(L"inputSequence"), Axis::DefaultBatchAxis() });
            auto plusParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ inputDim }, -0.05, 0.05, 1, device));
            auto plusFunc = CNTK::Plus(plusParam, inputVar);

            auto timesParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.05, 0.05, 1, device));
            auto timesFunc = CNTK::Times(timesParam, plusFunc);

            o1 = plusFunc->Output();
            o2 = timesFunc->Output();

            // Here all function smart pointers going out of scope.
        }

        FunctionPtr combine = CNTK::Combine({ o1, o2 });
        auto args = combine->Arguments();
    }

    // Recurrent
    {
        Variable o;
        {
            auto stateFwd = PlaceholderVariable(L"p1");

            auto placeHolder = PlaceholderVariable(L"p2");
            auto abs = Abs(placeHolder, L"abs");

            auto z = abs->Clone(ParameterCloningMethod::Share, { { placeHolder, PastValue(stateFwd, 1, L"pastValue") } });
            auto newState = z->ReplacePlaceholders({ {stateFwd, z->Output() } });
            o = newState->Output();
        }

        auto args = o.Owner()->Arguments();
        for (auto a : args)
        {
            if (a.Name() == L"" || a.Owner()->Name() == L"")
                RuntimeError("Unexpected name");
        }
    }
}

void TestOutputVariableName(const DeviceDescriptor& device)
{
    size_t inputDim = 10;
    size_t outputDim = 20;
    const std::wstring timesFuncName = L"TimesFunc";
    const std::wstring plusFuncName = L"PlusFunc";
    const std::wstring combineFuncName = L"CombineFunc";
    const std::wstring outputName = L"ModelOutput";

    auto inputVar = InputVariable({ inputDim }, DataType::Float, L"features");

    auto plusParam = Parameter(NDArrayView::RandomUniform<float>({ inputDim }, -0.05, 0.05, 1, device));
    auto plusFunc = Plus(plusParam, inputVar, plusFuncName);

    auto timesParam = Parameter(NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.05, 0.05, 1, device));
    auto timesFunc = Times(timesParam, plusFunc, timesFuncName);

    auto combineFunc = Combine({ timesFunc, plusFunc }, combineFuncName);

    FunctionPtr output = Alias(combineFunc->Outputs()[0], outputName);

    // Check function name and output variable name
    if (timesFunc->Name() != timesFuncName)
        ReportFailure("The function name does not match. expected = %S, actual = %S\n", timesFuncName.c_str(), timesFunc->Name().c_str());

    if (timesFunc->Output().Name() != timesFuncName)
        ReportFailure("The output variable name does not match. expected = %S, actual = %S\n", timesFuncName.c_str(), timesFunc->Output().Name().c_str());

    if (plusFunc->Name() != plusFuncName)
        ReportFailure("The function name does not match. expected = %S, actual = %S\n", plusFuncName.c_str(), plusFunc->Name().c_str());

    if (plusFunc->Output().Name() != plusFuncName)
        ReportFailure("The output variable name does not match. expected = %S, actual = %S\n", plusFuncName.c_str(), plusFunc->Output().Name().c_str());

    // Check combined function with multiple outputs
    if (combineFunc->Name() != combineFuncName)
        ReportFailure("The function name does not match. expected = %S, actual = %S\n", combineFuncName.c_str(), combineFunc->Name().c_str());

    if (combineFunc->Outputs()[0].Name() != timesFuncName)
        ReportFailure("The output variable name of combine function does not match. expected = %S, actual = %S\n", timesFuncName.c_str(), combineFunc->Outputs()[0].Name().c_str());

    if (combineFunc->Outputs()[1].Name() != plusFuncName)
        ReportFailure("The output variable name of combine function does not match. expected = %S, actual = %S\n", plusFuncName.c_str(), combineFunc->Outputs()[0].Name().c_str());

    // Check output variable using alias.
    if (output->Output().Name() != outputName)
        ReportFailure("THe output variable created by alias does not match. expected = %S, actual = %S\n", outputName.c_str(), output->Output().Name().c_str());

    // Check the output variable has correct shape size.
    if (output->Output().Shape().TotalSize() != outputDim)
        ReportFailure("The output variable does not have expected shape size. exptected = %ld, actual = %ld\n",
                      static_cast<unsigned long>(outputDim),
                      static_cast<unsigned long>(output->Output().Shape().TotalSize()));

    // Change the output order of combine function.
    // Todo: it is allowed to have duplicated function name?
    combineFunc = Combine({ plusFunc, timesFunc }, combineFuncName);

    // Make sure that the alias maps to the correct output variable when the output order changes
    output = Alias(combineFunc->Outputs()[1], outputName);

    // Check combined function with multiple outputs
    if (combineFunc->Name() != combineFuncName)
        ReportFailure("The function name does not match. expected = %S, actual = %S\n", combineFuncName.c_str(), combineFunc->Name().c_str());

    if (combineFunc->Outputs()[0].Name() != plusFuncName)
        ReportFailure("The output variable name of combine function does not match. expected = %S, actual = %S\n", plusFuncName.c_str(), combineFunc->Outputs()[0].Name().c_str());

    if (combineFunc->Outputs()[1].Name() != timesFuncName)
        ReportFailure("The output variable name of combine function does not match. expected = %S, actual = %S\n", timesFuncName.c_str(), combineFunc->Outputs()[0].Name().c_str());

    // Check that the output variable using alias is not affected
    if (output->Output().Name() != outputName)
        ReportFailure("THe output variable created by alias does not match. expected = %S, actual = %S\n", outputName.c_str(), output->Output().Name().c_str());

    // Check the output variable has correct shape size.
    if (output->Output().Shape().TotalSize() != outputDim)
        ReportFailure("The output variable does not have expected shape size. exptected = %ld, actual = %ld\n",
            static_cast<unsigned long>(outputDim),
            static_cast<unsigned long>(output->Output().Shape().TotalSize()));
}

void CheckFindByNameResult(FunctionPtr actual, FunctionPtr expected)
{
    if (actual == nullptr)
    {
        if (expected != nullptr)
            ReportFailure("The expected function '%S' has not been found.", expected->Name().c_str());
    }
    else 
    {
        if (expected == nullptr)
            ReportFailure("Found a function '%S', but null is expected.", actual->Name().c_str());
        else if (expected->Name().compare(actual->Name()) != 0)
            ReportFailure("The found function '%S' does have the same name as the exepected one '%S'", actual->Name().c_str(), expected->Name().c_str());
    }
}

void CheckFindAllWithNameResult(std::vector<FunctionPtr> actual, std::wstring expectedName, size_t expectedSize)
{
    if (actual.size() != expectedSize)
        ReportFailure("The number of found functions does not match the expected number.");
    else
    {
        for (size_t i = 0; i < actual.size(); i++)
        {
            if (actual[i]->Name().compare(expectedName) != 0)
                ReportFailure("The found function '%S' does have the same name as the exepected one '%S'", actual[i]->Name().c_str(), expectedName.c_str());
        }
    }
}

void TestFindName(const DeviceDescriptor& device)
{
    size_t inputDim = 10;
    size_t outputDim = 20;
    const std::wstring timesFuncName = L"TimesFunc";
    const std::wstring plusFuncName = L"PlusFunc";
    const std::wstring anotherPlusFuncName = L"AnotherPlusFunc";
    const std::wstring minusFuncName = L"MinusFunc";
    const std::wstring anotherMinusFuncName = L"AnotherMinusFunc";
    const std::wstring blockFuncName = L"BlockFunc";
    const std::wstring nonExistingFuncName = L"NonExistingFunc";
    const std::wstring nestedBlockFuncName = L"NestedBlockFunc";
    const std::wstring emptyFuncName = L"";
    const std::wstring placeholderName = L"inputPlaceholder";
    const std::wstring variableName = L"features";
    const std::wstring aliasFuncName = L"aliasFunc";

    auto inputVar1 = InputVariable({ inputDim }, DataType::Float, L"features");

    auto inputPlaceholder1 = PlaceholderVariable(L"inputPlaceholder");
    auto timesParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.05, 0.05, 1, device));
    auto timesFunc1 = CNTK::Times(timesParam, inputPlaceholder1, timesFuncName);
    auto plusFunc1 = CNTK::Plus(Constant::Scalar(2.0f), timesFunc1, plusFuncName);
    auto plusFunc2 = CNTK::Plus(Constant::Scalar(2.0f), plusFunc1, plusFuncName);
    auto emptyNameFunc1 = CNTK::Plus(plusFunc1, plusFunc2);
    auto minusFunc1 = CNTK::Minus(plusFunc2, emptyNameFunc1, minusFuncName);

    // Test FindByName for the case without any block function
    CheckFindByNameResult(minusFunc1->FindByName(timesFuncName), timesFunc1);
    CheckFindByNameResult(minusFunc1->FindByName(minusFuncName), minusFunc1);
    CheckFindByNameResult(minusFunc1->FindByName(emptyFuncName), emptyNameFunc1);
    CheckFindByNameResult(minusFunc1->FindByName(nonExistingFuncName), nullptr);
    CheckFindByNameResult(minusFunc1->FindByName(placeholderName), nullptr);
    VerifyException([&minusFunc1, &plusFuncName]() {
        minusFunc1->FindByName(plusFuncName);
    }, "The expected exception has not been caugth: multiple functions with the same name.");

    // Test FindAllWithName for the case without any block function
    CheckFindAllWithNameResult(minusFunc1->FindAllWithName(timesFuncName), timesFuncName, 1);
    CheckFindAllWithNameResult(minusFunc1->FindAllWithName(minusFuncName), minusFuncName, 1);
    CheckFindAllWithNameResult(minusFunc1->FindAllWithName(emptyFuncName), emptyFuncName, 1);
    CheckFindAllWithNameResult(minusFunc1->FindAllWithName(nonExistingFuncName), nonExistingFuncName, 0);
    CheckFindAllWithNameResult(minusFunc1->FindAllWithName(placeholderName), placeholderName, 0);
    CheckFindAllWithNameResult(minusFunc1->FindAllWithName(plusFuncName), plusFuncName, 2);

    // Build a block function
    auto blockFunc = CNTK::AsBlock(std::move(minusFunc1), { { inputPlaceholder1, inputVar1 } }, L"TimesPlusMinus", blockFuncName);

    // Build a nested block function
    auto inputPlaceholder2 = PlaceholderVariable(L"inputPlaceholder");
    auto inputPlaceholder3 = PlaceholderVariable(L"inputPlaceholder");
    auto inputVar2 = InputVariable({ outputDim }, DataType::Float, L"features");
    auto anotherMinusFunc1 = CNTK::Minus(inputPlaceholder2, Constant::Scalar(3.0f), anotherMinusFuncName);
    auto plusFunc3 = CNTK::Plus(Constant::Scalar(3.0f), anotherMinusFunc1, plusFuncName);
    auto cloneBlockFunc = blockFunc->Clone(ParameterCloningMethod::Clone, { { inputVar1, inputPlaceholder3 } });
    auto minusFunc2 = CNTK::Minus(cloneBlockFunc, plusFunc3, minusFuncName);
    auto plusFunc4 = CNTK::Plus(minusFunc2, Constant::Scalar(3.0f), plusFuncName);
    auto nestedBlockFunc = CNTK::AsBlock(std::move(plusFunc4), { { inputPlaceholder2, inputVar2 },{ inputPlaceholder3, inputVar1 } }, L"NestedBlock", nestedBlockFuncName);

    // Build a function having both block and nested block functions
    auto inputVar3 = InputVariable({ outputDim }, DataType::Float, variableName);
    auto plusFunc5 = CNTK::Plus(inputVar3, blockFunc, plusFuncName);
    auto anotherPlusFunc1 = CNTK::Plus(plusFunc5, nestedBlockFunc, anotherPlusFuncName);
    auto minusFunc3 = CNTK::Minus(anotherPlusFunc1, Constant::Scalar(3.0f), minusFuncName);

    // Test FindByName with block functions, nestedSearchInsideBlockFunction is false.
    CheckFindByNameResult(minusFunc3->FindByName(anotherPlusFuncName), anotherPlusFunc1);
    CheckFindByNameResult(minusFunc3->FindByName(anotherMinusFuncName), nullptr);
    CheckFindByNameResult(minusFunc3->FindByName(nonExistingFuncName), nullptr);
    CheckFindByNameResult(minusFunc3->FindByName(variableName), nullptr);
    CheckFindByNameResult(minusFunc3->FindByName(nestedBlockFuncName), nestedBlockFunc);
    CheckFindByNameResult(minusFunc3->FindByName(plusFuncName), plusFunc5);
    CheckFindByNameResult(minusFunc3->FindByName(timesFuncName), nullptr);
    CheckFindByNameResult(minusFunc3->FindByName(emptyFuncName), nullptr);
    CheckFindByNameResult(minusFunc3->FindByName(minusFuncName), minusFunc3);
    CheckFindByNameResult(minusFunc3->FindByName(blockFuncName), blockFunc);

    // Test FindByName with block functions, nestedSearchInsideBlockFunction is true
    CheckFindByNameResult(minusFunc3->FindByName(anotherPlusFuncName, true), anotherPlusFunc1);
    CheckFindByNameResult(minusFunc3->FindByName(anotherMinusFuncName, true), anotherMinusFunc1);
    CheckFindByNameResult(minusFunc3->FindByName(nonExistingFuncName, true), nullptr);
    CheckFindByNameResult(minusFunc3->FindByName(variableName, true), nullptr);
    CheckFindByNameResult(minusFunc3->FindByName(nestedBlockFuncName, true), nestedBlockFunc);
    VerifyException([&minusFunc3, &plusFuncName]() {
        minusFunc3->FindByName(plusFuncName, true);
    }, "The expected exception has not been caugth: multiple functions with the same name.");
    VerifyException([&minusFunc3, &timesFuncName]() {
        minusFunc3->FindByName(timesFuncName, true);
    }, "The expected exception has not been caugth: multiple functions with the same name.");
    VerifyException([&minusFunc3, &emptyFuncName]() {
        minusFunc3->FindByName(emptyFuncName, true);
    }, "The expected exception has not been caugth: multiple functions with the same name.");
    VerifyException([&minusFunc3, &minusFuncName]() {
        minusFunc3->FindByName(minusFuncName, true);
    }, "The expected exception has not been caugth: multiple functions with the same name.");
    VerifyException([&minusFunc3, &blockFuncName]() {
        minusFunc3->FindByName(blockFuncName, true);
    }, "The expected exception has not been caugth: multiple functions with the same name.");

    // Test FindByName with multiple block functions, nestedSearchInsideBlockFunction is true
    auto placeholder1 = PlaceholderVariable(L"inputPlaceholder1");
    auto firstBlockPlusFuncName = L"FirstBlockPlus";
    auto blockRoot1 = Plus(placeholder1, Constant::Scalar(1.0f), firstBlockPlusFuncName);
    auto placeholder2 = PlaceholderVariable(L"inputPlaceholder2");
    auto secondBlockPlusFuncName = L"SecondBlockPlus";
    auto blockRoot2 = Plus(placeholder2, Constant::Scalar(1.0f), secondBlockPlusFuncName);
    auto block2 = AsBlock(std::move(blockRoot2), { { placeholder2, InputVariable({}, DataType::Float)} }, L"Plus");
    auto block1 = AsBlock(std::move(blockRoot1), { { placeholder1, block2->Output() } }, L"Plus");
    CheckFindByNameResult(block1->FindByName(firstBlockPlusFuncName, true), blockRoot1);
    CheckFindByNameResult(block1->FindByName(secondBlockPlusFuncName, true), blockRoot2);

    // Test FindAllWithName with block functions, nestedSearchInsideBlockFunction is false.
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(anotherPlusFuncName), anotherPlusFuncName, 1);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(anotherMinusFuncName), anotherMinusFuncName, 0);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(nonExistingFuncName), nonExistingFuncName, 0);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(variableName), variableName, 0);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(nestedBlockFuncName), nestedBlockFuncName, 1);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(plusFuncName), plusFuncName, 1);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(timesFuncName), timesFuncName, 0);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(emptyFuncName), emptyFuncName, 0);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(minusFuncName), minusFuncName, 1);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(blockFuncName), blockFuncName, 1);

    // Test FindAllWithName with block functions, nestedSearchInsideBlockFunction is true
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(anotherPlusFuncName, true), anotherPlusFuncName, 1);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(anotherMinusFuncName, true), anotherMinusFuncName, 1);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(nonExistingFuncName, true), nonExistingFuncName, 0);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(variableName, true), variableName, 0);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(nestedBlockFuncName, true), nestedBlockFuncName, 1);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(plusFuncName, true), plusFuncName, 7);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(timesFuncName, true), timesFuncName, 2);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(emptyFuncName,true), emptyFuncName, 2);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(minusFuncName, true), minusFuncName, 4); 
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(blockFuncName, true), blockFuncName, 2);

    // Test alias
    auto aliasFunc1 = Alias(anotherPlusFunc1, aliasFuncName);
    // The Alias does not really create an alias for the function, but indeed create a new function having alias as name.
    // The new function is not a part of existing graph, except it is explicitly referenced in the graph.
    // TODO: change the tests when Alias is a real alias of a function.
    CheckFindByNameResult(minusFunc3->FindByName(aliasFuncName), nullptr);
    CheckFindAllWithNameResult(minusFunc3->FindAllWithName(aliasFuncName, true), aliasFuncName, 0);
    auto minusFunc4 = CNTK::Minus(aliasFunc1, minusFunc3, minusFuncName);
    CheckFindByNameResult(minusFunc4->FindByName(aliasFuncName), aliasFunc1);
    CheckFindAllWithNameResult(minusFunc4->FindAllWithName(aliasFuncName, true), aliasFuncName, 1);
}


std::function<std::vector<float>(FunctionPtr)> CreateForwardFunctor(const DeviceDescriptor& device, const Variable& inputVar)
{
    auto shape = inputVar.Shape();
    auto size = shape.TotalSize();
    auto inputData = std::make_shared<std::vector<float>>(size, 1.0f);
    ValuePtr inputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(shape, *inputData, false));
     
    auto outputData = std::make_shared<std::vector<float>>(size, 0.0f);
    ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(shape.AppendShape({ 1, 1 }), *outputData, false));

    return [device, inputVar, inputValue, outputValue, inputData, outputData](FunctionPtr f) -> std::vector<float> {
        std::unordered_map<Variable, ValuePtr> inputMap { { inputVar, inputValue } };
        std::unordered_map<Variable, ValuePtr> outputMap { { f->Output(), outputValue } };
        f->Forward(inputMap, outputMap, device, { f->Output() });
        return *outputData;
    };
}

void SetDropoutRate(const DeviceDescriptor& device) 
{
    auto zeroCount = [](const std::vector<float>& v) 
    {
        size_t count = 0;
        for (auto e : v) if (e == 0.0) count++;
        return count;
    };

    auto shape = NDShape({ 10, 10, 10, 10 });
    auto input = InputVariable(shape, DataType::Float);
    auto dropout = Dropout(input, 0.0, 5336, L"Dropout");
    auto forwardFunc = CreateForwardFunctor(device, input);

    BOOST_TEST(zeroCount(forwardFunc(dropout)) == 0); // initially dropout is disabled;

    for (auto dropoutRate : { 0.9, 0.4, 0.0, 0.1 }) 
    {
        dropout->SetAttribute(L"dropoutRate", dropoutRate);
        BOOST_TEST(abs(zeroCount(forwardFunc(dropout)) - dropoutRate*shape.TotalSize()) < 100);
    }

    auto plusParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>(shape, -0.5, 0.5, 1, device));
    auto combine = Combine({ Plus(plusParam, Plus(dropout, ElementTimes(plusParam, Constant::Scalar(-1.0f)))) });

    auto dropout2 = combine->FindByName(L"Dropout");

    for (auto dropoutRate : { 0.3, 0.7, 0.2 })
    {
        dropout2->SetAttribute(L"dropoutRate", dropoutRate);
        BOOST_TEST(abs(zeroCount(forwardFunc(combine)) - dropoutRate*shape.TotalSize()) < 100);
    }
}

void SetRandomSeed(const DeviceDescriptor& device)
{
    auto diff = [](const std::vector<float>& a, const std::vector<float>& b)
    {
        bool foundDifference = false;
        for (int i = 0; !foundDifference && i < a.size() && i < b.size(); ++i)
        {
            foundDifference = (a[i] != b[i]);
        }
        return foundDifference;
    };

    auto shape = NDShape({ 10, 10, 10, 10 });
    auto input = InputVariable(shape, DataType::Float);
    auto dropout = Dropout(input, 0.5, 5336, L"Dropout");
    auto forwardFunc = CreateForwardFunctor(device, input);

    auto result1 = forwardFunc(dropout);

    dropout->SetAttribute(L"rngSeed", 5337);

    auto result2 = forwardFunc(dropout);

    BOOST_TEST(diff(result1, result2));

    dropout->SetAttribute(L"rngSeed", 5336);
    auto result3 = forwardFunc(dropout);
    BOOST_TEST(!diff(result1, result3));

    auto plusParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>(shape, -0.5, 0.5, 1, device));
    auto combine = Combine({ Plus(plusParam, Plus(dropout, ElementTimes(plusParam, Constant::Scalar(-1.0f)))) });
 
    forwardFunc(combine); // this will force the composite function to construct a new computation network.

    auto dropout2 = combine->FindByName(L"Dropout");
    dropout2->SetAttribute(L"rngSeed", 5337);
    auto result4 = forwardFunc(combine);
    // there could be small differences between result2 and result4 due to rounding errors.
    FloatingPointVectorCompare(result2, result4, "SetRandomSeed: output does match the expected after resetting the dropout seed.");
}

void TestMatMul(const DeviceDescriptor& device)
{
    srand(1);
    auto diff_size = [](const std::vector<size_t>& a, const std::vector<size_t>& b)
    {
        bool foundDifference = false;
        if (a.size() != b.size()) return true;
        for (int i = 0; !foundDifference && i < a.size() && i < b.size(); ++i)
        {
            foundDifference = (a[i] != b[i]);
        }
        return foundDifference;
    };

    std::vector<std::vector<size_t>> inputShapeVec0{{3, 4}, {3, 4, 2, 2}, {64, 4, 1}, {64, NDShape::InferredDimension, 1}};
    std::vector<std::vector<size_t>> inputShapeVec1{{5, 3}, {5, 3, 2, 2}, {2, 64}, {2, 64}};
    std::vector<std::vector<size_t>> outShapeVec{{5, 4}, {5, 4, 2, 2}, {2, 4, 1}, {2, NDShape::InferredDimension, 1}};
    std::vector<std::vector<size_t>> inputValueShapeVec0{ { 3, 4 },{ 3, 4, 2, 2 },{ 64, 4, 1 },{ 64, 4, 1 } };
    std::vector<std::vector<size_t>> inputValueShapeVec1{ { 5, 3 },{ 5, 3, 2, 2 },{ 2, 64 },{ 2, 64 } };
    std::vector<std::vector<size_t>> outValueShapeVec{ { 5, 4 },{ 5, 4, 2, 2 },{ 2, 4, 1 },{ 2, 4, 1 } };
    std::vector<size_t> inputTotalSizeVec0 = { 12, 48, 256, 256 };
    std::vector<size_t> inputTotalSizeVec1 = { 15, 60, 128, 128 };
    std::vector<size_t> outputTotalSizeVec = { 20, 80, 8, 8 };
    std::vector<size_t> inputSubSizeVec0 = { 12, 12, 256, 256 };
    std::vector<size_t> inputSubSizeVec1 = { 15, 15, 128, 128 };
    std::vector<size_t> outputSubSizeVec = { 20, 20, 8, 8 };


    size_t testCases = inputShapeVec0.size();
    for (size_t test_i = 0; test_i < testCases; ++test_i)
    {
        auto shape0 = NDShape(inputShapeVec0[test_i]);
        auto shape1 = NDShape(inputShapeVec1[test_i]);
        auto valueShape0 = NDShape(inputValueShapeVec0[test_i]);
        auto valueShape1 = NDShape(inputValueShapeVec1[test_i]);
        auto outShape = NDShape(outShapeVec[test_i]);
        auto outValueShape = NDShape(outValueShapeVec[test_i]);

        size_t inputTotalSize0 = inputTotalSizeVec0[test_i];
        size_t inputTotalSize1 = inputTotalSizeVec1[test_i];
        size_t outputTotalSize = outputTotalSizeVec[test_i];
        size_t inputSubSize0 = inputSubSizeVec0[test_i];
        size_t inputSubSize1 = inputSubSizeVec1[test_i];
        size_t outputSubSize = outputSubSizeVec[test_i];

        auto input0 = InputVariable(shape0, DataType::Float);
        auto input1 = InputVariable(shape1, DataType::Float);
        auto result = ::CNTK::Internal::MatMul(input0, input1);

        std::vector<float> inputData0(inputTotalSize0);
        std::vector<float> inputData1(inputTotalSize1);
        for (size_t i = 0; i < inputData0.size(); ++i)
            inputData0[i] = ((float)rand()) / RAND_MAX;
        for (size_t i = 0; i < inputData1.size(); ++i)
            inputData1[i] = ((float)rand()) / RAND_MAX;

        ValuePtr inputValue0 = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(valueShape0.AppendShape({1,1}), inputData0, true));
        ValuePtr inputValue1 = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(valueShape1.AppendShape({ 1,1 }), inputData1, true));

        NDShape outputShape = result->Output().Shape();
        BOOST_TEST(!diff_size(outShape.Dimensions(), outputShape.Dimensions()));
        std::vector<float> outputData(outputTotalSize);
        ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outValueShape.AppendShape({ 1,1 }), outputData, false));

        std::unordered_map<Variable, ValuePtr> outputs = {{result->Output(), outputValue}};
        result->Forward({{input0, inputValue0}, {input1, inputValue1}}, outputs, device);

        std::vector<float> expectedOutputValues(outputTotalSize);
        {
            for (size_t i = 0; i < outputTotalSize / outputSubSize; i++)
            {
                std::vector<float> inputTimesData0(inputSubSize0);
                std::vector<float> inputTimesData1(inputSubSize1);
                std::vector<float> outTimesData(outputSubSize);
                auto inputTimes0 = InputVariable(shape0.SubShape(0, 2), DataType::Float);
                auto inputTimes1 = InputVariable(shape1.SubShape(0, 2), DataType::Float);
                auto timesResult = Times(inputTimes1, inputTimes0);

                for (size_t j = 0; j < inputSubSize0; ++j)
                {
                    inputTimesData0[j] = inputData0[i * inputSubSize0 + j];
                }
                for (size_t j = 0; j < inputSubSize1; ++j)
                {
                    inputTimesData1[j] = inputData1[i * inputSubSize1 + j];
                }

                ValuePtr inputTimesValue0 = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(valueShape0.SubShape(0, 2).AppendShape({ 1,1 }), inputTimesData0, true));
                ValuePtr inputTimesValue1 = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(valueShape1.SubShape(0, 2).AppendShape({ 1,1 }), inputTimesData1, true));
                ValuePtr outputTimesValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outValueShape.SubShape(0, 2).AppendShape({ 1,1 }), outTimesData, false));
                std::unordered_map<Variable, ValuePtr> timesOutputs = { {timesResult->Output(), outputTimesValue}};
                timesResult->Forward({{inputTimes0, inputTimesValue0 }, {inputTimes1, inputTimesValue1 }}, timesOutputs, device);

                for (size_t j = 0; j < outputSubSize; ++j)
                {
                    expectedOutputValues[i * outputSubSize + j] = outTimesData[j];
                }
            }
        }

        FloatingPointVectorCompare(outputData, expectedOutputValues, "TestMatMul: Forward prop results do not match expected results.");
    }
}

BOOST_AUTO_TEST_SUITE(FunctionSuite)

BOOST_AUTO_TEST_CASE(FindNameInCPU)
{
    if (ShouldRunOnCpu())
        TestFindName(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(FindNameInGPU)
{
    if (ShouldRunOnGpu())
        TestFindName(DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_CASE(Splice)
{
    // Handles device internally.
    TestSplice();
}

BOOST_AUTO_TEST_CASE(ChangingParameterValuesInCPU)
{
    if (ShouldRunOnCpu())
    {
        TestChangingParameterValues<float>(2, DeviceDescriptor::CPUDevice());
        TestChangingParameterValues<double>(3, DeviceDescriptor::CPUDevice());
    }
}

BOOST_AUTO_TEST_CASE(ChangingParameterValuesInGPU)
{
    if (ShouldRunOnGpu())
        TestChangingParameterValues<double>(3, DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_CASE(TimesNodeShapeInference)
{
    if (ShouldRunOnCpu())
        TestTimesNodeShapeInference();
}

BOOST_AUTO_TEST_CASE(RecurrenceShapeInference)
{
    if (ShouldRunOnCpu())
        TestRecurrenceShapeInference();
}

BOOST_AUTO_TEST_CASE(SliceInCPU)
{
    if (ShouldRunOnCpu())
        TestSlice(2, DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(SliceInGPU)
{
    if (ShouldRunOnGpu())
        TestSlice(1, DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_CASE(ReduceSumInCPU)
{
    if (ShouldRunOnCpu())
        TestReduceSum(1, DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(ReduceSumInGPU)
{
    if (ShouldRunOnGpu())
        TestReduceSum(2, DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_CASE(RecurrentFunctionCloning)
{
    if (ShouldRunOnCpu())
        TestRecurrentFunctionCloning();
}

BOOST_AUTO_TEST_CASE(TransposeInCPU)
{
    if (ShouldRunOnCpu())
        TestTranspose(2, 0, 1, DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(TransposeInGPU)
{
    if (ShouldRunOnGpu())
        TestTranspose(3, 1, 2, DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_CASE(OutputVariableNameInCPU)
{
    if (ShouldRunOnCpu())
        TestOutputVariableName(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(FunctionOutputs)
{
    if (ShouldRunOnCpu())
        TestFunctionOutputs(DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(TimesIndirectSparseGradType)
{
    if (ShouldRunOnCpu())
        TestTimesIndirectSparseInputGradientSparse(DeviceDescriptor::CPUDevice());
}


BOOST_AUTO_TEST_CASE(TestSettingDropoutRate)
{
    if (ShouldRunOnCpu())
        SetDropoutRate(DeviceDescriptor::CPUDevice());
    
    if (ShouldRunOnGpu())
        SetDropoutRate(DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_CASE(TestSettingRandomSeed)
{
    if (ShouldRunOnCpu())
        SetRandomSeed(DeviceDescriptor::CPUDevice());

    if (ShouldRunOnGpu())
        SetRandomSeed(DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_CASE(MatMul)
{
    if (ShouldRunOnCpu())
        TestMatMul(DeviceDescriptor::CPUDevice());
    if (ShouldRunOnGpu())
        TestMatMul(DeviceDescriptor::GPUDevice(0));
}

BOOST_AUTO_TEST_SUITE_END()

}}
