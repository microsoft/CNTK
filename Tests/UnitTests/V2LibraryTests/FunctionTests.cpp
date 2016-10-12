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
        auto testReduceSum = [&sequences, &sequenceLengths, inputShape, sequencesValue, device](int reductionAxis)
        {
            size_t maxActualSequenceLength = sequencesValue->Shape()[inputShape.Rank()];
            size_t numSequences = sequencesValue->Shape()[inputShape.Rank() + 1];

            auto inputVar = InputVariable(inputShape, DataType::Float, L"input");
            FunctionPtr reduceSumFunc;

            bool reduceAll = (reductionAxis < 0);
            if (reduceAll)
                reduceSumFunc = ReduceSum(inputVar);
            else
                reduceSumFunc = ReduceSum(inputVar, Axis(reductionAxis));

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
        testReduceSum(-1);

        int reductionAxis = 0;
        testReduceSum(reductionAxis);

        if (reductionAxis < (inputShape.Rank() - 1))
            reductionAxis++;

        testReduceSum(reductionAxis);

        if (reductionAxis < (inputShape.Rank() - 1))
            reductionAxis++;

        testReduceSum(reductionAxis);
    }

    // Test ReduceSum along a dynamic axis
    {
        auto testReduceSum = [&sequences, &sequenceLengths, inputShape, sequencesValue, device](const Axis& axis)
        {
            if (axis.IsStaticAxis())
                RuntimeError("Called the dynamic axis ReduceSum test with a static axis");

            size_t maxActualSequenceLength = sequencesValue->Shape()[inputShape.Rank()];
            size_t numSequences = sequencesValue->Shape()[inputShape.Rank() + 1];

            auto inputVar = InputVariable({ inputShape }, DataType::Float, L"input");
            FunctionPtr reduceSumFunc = ReduceSum(inputVar, axis);

            NDShape maskShape = { ((axis == Axis::DefaultBatchAxis()) ? maxActualSequenceLength : 1), ((axis == Axis::DefaultBatchAxis()) ? 1 : numSequences) };
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
                        if (axis == Axis::DefaultBatchAxis())
                            expectedTotals[(j * inputShape.TotalSize()) + k] += value;
                        else
                            expectedTotals[(i * inputShape.TotalSize()) + k] += value;
                    }
                }
            }

            FloatingPointVectorCompare(outputData, expectedTotals, "testReduceSum: Forward prop results do not match expected results");
        };

        testReduceSum(Axis::DefaultDynamicAxis());
    }
}

void TestSlice(size_t sampleRank, const DeviceDescriptor& device)
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

    // Test slice along a static axis
    {
        auto testStaticAxisSlice = [&sequences, &sequenceLengths, inputShape, sequencesValue, device](size_t sliceAxis, int beginOffset, int endOffset)
        {
            size_t maxActualSequenceLength = sequencesValue->Shape()[inputShape.Rank()];
            size_t numSequences = sequencesValue->Shape()[inputShape.Rank() + 1];

            auto inputVar = InputVariable(inputShape, DataType::Float, L"input");
            auto sliceFunc = Slice(inputVar, Axis(sliceAxis), beginOffset, endOffset);

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

        size_t sliceAxis = 0;
        testStaticAxisSlice(sliceAxis, 3, 5);

        if (sliceAxis < (inputShape.Rank() - 1))
            sliceAxis++;

        testStaticAxisSlice(sliceAxis, -1, 0);

        if (sliceAxis < (inputShape.Rank() - 1))
            sliceAxis++;

        testStaticAxisSlice(sliceAxis, - 3, -1);
    }

    // Test slice along a dynamic axis
    {
        auto testDynamicAxisSlice = [&sequences, &sequenceLengths, inputShape, sequencesValue, device](const Axis& axis, int beginOffset, int endOffset)
        {
            if (axis.IsStaticAxis())
                RuntimeError("Called the dynamic axis slice test with a static axis");

            size_t maxActualSequenceLength = sequencesValue->Shape()[inputShape.Rank()];
            size_t numSequences = sequencesValue->Shape()[inputShape.Rank() + 1];

            size_t sliceLength = endOffset - beginOffset;

            auto inputVar = InputVariable(inputShape, DataType::Float, L"input");
            auto sliceFunc = Slice(inputVar, axis, beginOffset, endOffset);

            size_t outputSequenceAxisLength = (axis == Axis::DefaultDynamicAxis()) ? sliceLength : maxActualSequenceLength;
            size_t outputBatchAxisLength = (axis == Axis::DefaultBatchAxis()) ? sliceLength : numSequences;
            NDShape outputShape = sliceFunc->Output().Shape().AppendShape({ outputSequenceAxisLength, outputBatchAxisLength });
            std::vector<float> outputData(outputShape.TotalSize());
            ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData, false));

            std::unordered_map<Variable, ValuePtr> outputs = { { sliceFunc->Output(), outputValue } };
            sliceFunc->Forward({ { inputVar, sequencesValue } }, outputs, device);

            size_t startSequenceIdx = (axis == Axis::DefaultBatchAxis()) ? ((beginOffset >= 0) ? beginOffset : (numSequences + beginOffset)) : 0;
            size_t endSequenceIdx = (axis == Axis::DefaultBatchAxis()) ? ((endOffset > 0) ? endOffset : (numSequences + endOffset)) : numSequences;

            std::vector<float> expectedOutputValues(inputShape.TotalSize() * outputSequenceAxisLength * outputBatchAxisLength);
            for (size_t i = startSequenceIdx; i < endSequenceIdx; ++i)
            {
                size_t currentSequenceLength = sequenceLengths[i];
                size_t startFrameIdx = (axis == Axis::DefaultDynamicAxis()) ? ((beginOffset >= 0) ? beginOffset : (currentSequenceLength + beginOffset)) : 0;
                size_t endFrameIdx = (axis == Axis::DefaultDynamicAxis()) ? ((endOffset > 0) ? endOffset : (currentSequenceLength + endOffset)) : currentSequenceLength;
                for (size_t j = startFrameIdx; j < endFrameIdx; ++j)
                {
                    for (size_t k = 0; k < inputShape.TotalSize(); ++k)
                        expectedOutputValues[((((i - startSequenceIdx) * outputSequenceAxisLength) + (j - startFrameIdx)) * inputShape.TotalSize()) + k] = sequences[i][(j * inputShape.TotalSize()) + k];
                }
            }

            FloatingPointVectorCompare(outputData, expectedOutputValues, "testDynamicAxisSlice: Forward prop results do not match expected results");
        };

        testDynamicAxisSlice(Axis::DefaultDynamicAxis(), 0, 1);
        testDynamicAxisSlice(Axis::DefaultDynamicAxis(), -1, 0);
    }
}

void CompareFunctions(const FunctionPtr& first, const FunctionPtr& second, ParameterCloningMethod parameterCloningMethod, const std::unordered_map<Variable, Variable>& replacements, std::unordered_set<FunctionPtr>& visitedFunctions)
{
    if ((first->RootFunction() == nullptr) != (second->RootFunction() == nullptr))
        throw std::runtime_error("CompareFunctions: Both functions should be primitives or both should be composites");

    if (first->Name() != second->Name())
        throw std::runtime_error("CompareFunctions: Both functions' names should match");

    if (first->Attributes() != second->Attributes())
        throw std::runtime_error("CompareFunctions: Both functions' attributes should match");

    auto firstPrimitive = (first->RootFunction() == nullptr) ? first : first->RootFunction();
    auto secondPrimitive = (second->RootFunction() == nullptr) ? second : second->RootFunction();

    // All the outputs must be equivalent
    if (firstPrimitive->Outputs().size() != secondPrimitive->Outputs().size())
        throw std::runtime_error("CompareFunctions: Both functions' should have same number of outputs");

    visitedFunctions.insert(firstPrimitive);

    for (size_t i = 0; i < firstPrimitive->Outputs().size(); ++i)
    {
        auto firstFunctionOutput = firstPrimitive->Outputs()[i];
        auto secondFunctionOutput = secondPrimitive->Outputs()[i];

        if ((firstFunctionOutput.Name() != secondFunctionOutput.Name()) ||
            (firstFunctionOutput.DynamicAxes() != secondFunctionOutput.DynamicAxes()) ||
            (firstFunctionOutput.GetDataType() != secondFunctionOutput.GetDataType()) ||
            (firstFunctionOutput.IsSparse() != secondFunctionOutput.IsSparse()) ||
            (firstFunctionOutput.Kind() != secondFunctionOutput.Kind()) ||
            (firstFunctionOutput.Shape() != secondFunctionOutput.Shape()))
        {
            throw std::runtime_error("CompareFunctions: Both functions' outputs should match");
        }
    }

    // All of the inputs must be identical
    if (firstPrimitive->Inputs().size() != secondPrimitive->Inputs().size())
        throw std::runtime_error("CompareFunctions: Both functions' should have same number of inputs");

    for (size_t i = 0; i < firstPrimitive->Inputs().size(); ++i)
    {
        auto firstFunctionInput = firstPrimitive->Inputs()[i];
        auto secondFunctionInput = secondPrimitive->Inputs()[i];

        if (replacements.find(firstFunctionInput) != replacements.end())
        {
            if (replacements.at(firstFunctionInput) != secondFunctionInput)
                throw std::runtime_error("CompareFunctions: The 2nd function does not have the expected replacement");
        }
        else
        {
            if (firstFunctionInput.IsOutput())
            {
                if (visitedFunctions.find(firstFunctionInput.Owner()) == visitedFunctions.end())
                    CompareFunctions(firstFunctionInput.Owner(), secondFunctionInput.Owner(), parameterCloningMethod, replacements, visitedFunctions);
            }
            else
            {
                if ((firstFunctionInput.Name() != secondFunctionInput.Name()) ||
                    (firstFunctionInput.DynamicAxes() != secondFunctionInput.DynamicAxes()) ||
                    (firstFunctionInput.IsSparse() != secondFunctionInput.IsSparse()) ||
                    (firstFunctionInput.Shape() != secondFunctionInput.Shape()) ||
                    (firstFunctionInput.GetDataType() != secondFunctionInput.GetDataType()))
                {
                    throw std::runtime_error("CompareFunctions: The leaves of the functions are not equivalent");
                }

                if ((firstFunctionInput.Kind() != VariableKind::Parameter) && ((firstFunctionInput.Kind() != secondFunctionInput.Kind()) || (firstFunctionInput.NeedsGradient() != secondFunctionInput.NeedsGradient())))
                    throw std::runtime_error("CompareFunctions: The leaves of the functions are not equivalent");

                switch (firstFunctionInput.Kind())
                {
                case VariableKind::Parameter:
                    if ((parameterCloningMethod == ParameterCloningMethod::Share) && (firstFunctionInput != secondFunctionInput))
                        throw std::runtime_error("CompareFunctions: The parameters of the functions are not equivalent per the specified cloning method");

                    if ((parameterCloningMethod == ParameterCloningMethod::Clone) &&
                        ((firstFunctionInput == secondFunctionInput) || (DictionaryValue(*(Parameter(firstFunctionInput).Value())) != DictionaryValue(*(Parameter(secondFunctionInput).Value())))))
                    {
                        throw std::runtime_error("CompareFunctions: The parameters of the functions are not equivalent per the specified cloning method");
                    }

                    if ((parameterCloningMethod == ParameterCloningMethod::Freeze) &&
                        ((firstFunctionInput == secondFunctionInput) || !secondFunctionInput.IsConstant() || (DictionaryValue(*(Parameter(firstFunctionInput).Value())) != DictionaryValue(*(Constant(secondFunctionInput).Value())))))
                    {
                        throw std::runtime_error("CompareFunctions: The parameters of the functions are not equivalent per the specified cloning method");
                    }

                    break;
                case VariableKind::Constant:
                    if (DictionaryValue(*(Constant(firstFunctionInput).Value())) != DictionaryValue(*(Constant(secondFunctionInput).Value())))
                        throw std::runtime_error("CompareFunctions: The constants of the functions are not equivalent");

                    break;
                }
            }
        }
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

    auto placeholder = PlaceholderVariable({ outputDim });
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

void TestTranspose(size_t numAxes, size_t axis1, size_t axis2, const DeviceDescriptor& device)
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

void FunctionTests()
{
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
